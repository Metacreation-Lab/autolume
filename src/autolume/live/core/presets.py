"""Save and restore a performance look as one versioned JSON file.

Which parameters persist is driven by `ParamSpec.preset`, so the format cannot
drift from the parameters that exist. Reading is forgiving in one direction
only: unknown or malformed entries are logged and skipped, missing parameters
keep their current value, but a file that is not a preset raises. Bindings are
the exception to missing meaning unchanged, since a preset recalls a whole look:
no bindings section clears every mapping. Values are applied
through `apply_value`, so a hand edited file cannot push a parameter out of
range, and expressions are never evaluated here.

Arrays (a latent vector, a vec keyframe) do not fit a JSON scalar, so every one
in the payload goes through one descriptor, `{"dtype", "shape", "b64"}`, and one
strict encoder/decoder pair. Bytes are always little-endian `float32`, written
and read explicitly rather than in the machine's native order, so a preset
written on one machine reads correctly on another. A rejected array falls back
to its default and costs the preset nothing else, exactly like a rejected
scalar.

The model is not a plain parameter: it is saved as its own `model` key, a bare
filename when it lives under the local models folder and an absolute path
otherwise, so a shared preset can resolve on the machine that opens it. The
second model used for network mixing (`pkl2`) is saved the same way, under a
sibling `model2` key, through the same resolution helpers, so a mixing look
stays portable across machines too.

The bending chain, per-layer noise and ratios, adjuster directions and
mixing layer origins are sparse or small structured sections of their own
(`transforms`, `layer_noise`, `layer_ratios`, `directions`,
`combined_layers`). A preset is never validated against a model: an unknown
layer or op name loads as-is, the render path is what logs and skips it.
`directions` reuses the same array descriptor as `latent_vec`, reinterpreted
as a 2D block of up to eight equal-length vectors rather than inventing a
second encoding.
"""

import base64
import binascii
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autolume.live.core.params import REGISTRY, Binding, ControlState, Keyframe, Transform

logger = logging.getLogger(__name__)

FORMAT = "autolume-live-preset"
VERSION = 1

# Structured control address. It carries the whole payload so the control loop
# applies a preset in a single state replacement.
PRESET_APPLY = "/preset/apply"

_BINDING_FIELDS = ("target", "source", "expression")
_ARRAY_DTYPE = "float32"

# The loop's stops and the vector, at rest: what a payload missing either
# section resolves to, same as a freshly opened `ControlState`.
_DEFAULT_KEYFRAMES: tuple[Keyframe, ...] = ControlState().keyframes
_DEFAULT_LATENT_VEC: tuple[float, ...] = ControlState().latent_vec

# Tells a key that is absent from the payload apart from one that is present and
# holds something unusable. The two mean different things and are reported
# differently, and `None` cannot stand in because it is a value a file can hold.
_ABSENT = object()


@dataclass(frozen=True)
class PresetData:
    """A payload, split into what the control loop needs to apply it.

    Growing this from the old `(params, bindings)` tuple, because a tuple that
    keeps widening is how a call site gets silently miswired the day a new
    section is added.
    """

    params: dict
    bindings: tuple[Binding, ...]
    latent_vec: tuple[float, ...]
    keyframes: tuple[Keyframe, ...]
    missing_model: str | None
    missing_model2: str | None
    transforms: tuple[Transform, ...]
    layer_noise: tuple[tuple[str, float], ...]
    layer_ratios: tuple[tuple[str, float, float], ...]
    directions: tuple[tuple[float, ...], ...]
    combined_layers: tuple[str, ...]


def _jsonable(value: object) -> object:
    """Return `value` as something `json.dump` accepts.

    Paths reach the state from file dialogs and are not serializable, so they
    are written as text. This is the recurring `WindowsPath` bug class.
    """
    return os.fspath(value) if isinstance(value, os.PathLike) else value


def _encode_array(values: tuple[float, ...]) -> dict | None:
    """One array, as `{"dtype", "shape", "b64"}`, or None for an empty one."""
    if not values:
        return None
    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return {
        "dtype": _ARRAY_DTYPE,
        "shape": list(array.shape),
        "b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _decode_array(raw: object, what: str) -> tuple[float, ...] | None:
    """Strict, total array decode: an array, or None with one log line.

    Every rule below rejects only this one array, never raises, and never lets
    a NaN or an infinity reach the caller.
    """
    if not isinstance(raw, dict):
        logger.warning("Skipping %s, not an array object", what)
        return None
    if raw.get("dtype") != _ARRAY_DTYPE:
        logger.warning("Skipping %s, unsupported dtype %r", what, raw.get("dtype"))
        return None
    shape = raw.get("shape")
    shape_ok = isinstance(shape, list) and bool(shape) and all(
        isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape
    )
    if not shape_ok:
        logger.warning("Skipping %s, invalid shape %r", what, shape)
        return None
    b64 = raw.get("b64")
    if not isinstance(b64, str):
        logger.warning("Skipping %s, missing base64 data", what)
        return None
    try:
        raw_bytes = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        logger.warning("Skipping %s, undecodable base64 data", what)
        return None
    expected = math.prod(shape) * 4
    if len(raw_bytes) != expected:
        logger.warning("Skipping %s, byte length disagrees with shape", what)
        return None
    # Explicitly little-endian on the way in, matching the encoder: a preset
    # written on a big-endian machine must decode the same values here.
    array = np.frombuffer(raw_bytes, dtype="<f4")
    if not bool(np.all(np.isfinite(array))):
        logger.warning("Skipping %s, contains a non finite value", what)
        return None
    return tuple(float(v) for v in array)


def _models_dir() -> Path:
    # Imported here, matching `preset_dir`: the user data root is a legacy
    # flat root module, and only model resolution needs it.
    from utils.user_data import data_path

    return data_path("models")


def _model_reference(pkl_path: object) -> dict | None:
    """What to write for the `model` key: a bare name, a path, or None."""
    if not pkl_path:
        return None
    resolved = Path(os.fspath(pkl_path))
    try:
        relative = resolved.relative_to(_models_dir())
    except ValueError:
        return {"path": str(resolved)}
    return {"name": relative.as_posix()}


def _read_model(raw: object) -> tuple[str | None, str | None]:
    """Resolve the `model` key to `(path to apply, name reported missing)`.

    Both are None when the key is absent or null, which leaves the currently
    loaded model alone rather than unloading it.
    """
    if raw is _ABSENT or raw is None:
        return None, None
    if not isinstance(raw, dict):
        logger.warning("Ignoring preset model reference of type %s", type(raw).__name__)
        return None, None
    name = raw.get("name")
    if isinstance(name, str):
        candidate = _models_dir() / name
        if candidate.exists():
            return str(candidate), None
        logger.warning("Preset model %s was not found in the models folder", name)
        return None, name
    path = raw.get("path")
    if isinstance(path, str):
        if Path(path).exists():
            return path, None
        logger.warning("Preset model %s was not found", path)
        return None, Path(path).name
    logger.warning("Ignoring malformed preset model reference %r", raw)
    return None, None


def _finite_or_none(value: object) -> float | None:
    """`value` as a finite float, or None if it cannot be one.

    Unlike `_finite_number`, there is no default to fall back to: the callers
    below drop the whole entry it belongs to rather than substitute a value,
    matching how `mapping.py` drops a whole layer noise/ratio event on an
    uncoercible number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _encode_transform(transform: Transform) -> dict | None:
    """One transform as JSON, or None if its params are not all finite.

    `json.dump` defaults to writing a bare NaN/Infinity token for a non-finite
    float, which is not valid JSON for a strict parser reading this
    user-facing file back. Dropping the whole transform (not just the bad
    param) is the only sound choice: a `Transform`'s params are positional and
    fixed by its op's arity, so there is no way to omit just one and keep the
    rest meaningful.
    """
    params = [float(p) for p in transform.params]
    if not all(math.isfinite(p) for p in params):
        logger.warning(
            "Not writing transform %s on %s, non finite params %r",
            transform.op,
            transform.layer,
            params,
        )
        return None
    return {
        "op": transform.op,
        "layer": transform.layer,
        "params": params,
        "indices": [int(i) for i in transform.indices],
    }


def _read_number_list(raw: object, what: str) -> tuple[float, ...] | None:
    if not isinstance(raw, list):
        logger.warning("Skipping %s, not a list", what)
        return None
    values = []
    for value in raw:
        number = _finite_or_none(value)
        if number is None:
            logger.warning("Skipping %s, %r is not a finite number", what, value)
            return None
        values.append(number)
    return tuple(values)


def _read_index_list(raw: object, what: str) -> tuple[int, ...] | None:
    if not isinstance(raw, list):
        logger.warning("Skipping %s, not a list", what)
        return None
    values = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            logger.warning("Skipping %s, %r is not a valid index", what, value)
            return None
        values.append(value)
    return tuple(values)


def _read_transform(entry: object, index: int) -> Transform | None:
    """Structural validation, plus the one thing that is knowable without a
    model: `op` and its arity. The eleven-operator set is a fixed design
    decision (`mapping.py`'s `_OPERATOR_ARITY`), not a property of any
    particular model, so unlike `layer` it does not fall under "a preset is
    not validated against a model" and is rejected here rather than handed
    off to the render path.

    Imported locally rather than at module level: `mapping.py` imports from
    this module, so a top-level import here would be circular. By the time
    this function runs, both modules have already finished loading, the same
    reasoning `_models_dir` already relies on for its own local import.
    """
    if not isinstance(entry, dict):
        logger.warning("Skipping transform %d, not an object", index)
        return None
    from autolume.live.core.mapping import _OPERATOR_ARITY

    op = entry.get("op")
    layer = entry.get("layer")
    if not isinstance(op, str) or op not in _OPERATOR_ARITY:
        logger.warning("Skipping transform %d, invalid op %r", index, op)
        return None
    if not isinstance(layer, str) or not layer:
        logger.warning("Skipping transform %d, invalid layer %r", index, layer)
        return None
    params = _read_number_list(entry.get("params"), f"transform {index} params")
    if params is None:
        return None
    if len(params) != _OPERATOR_ARITY[op]:
        logger.warning(
            "Skipping transform %d, op %s expects %d params, got %d",
            index,
            op,
            _OPERATOR_ARITY[op],
            len(params),
        )
        return None
    indices = _read_index_list(entry.get("indices"), f"transform {index} indices")
    if indices is None:
        return None
    return Transform(op=op, layer=layer, params=params, indices=indices)


def _read_transforms(raw: object) -> tuple[Transform, ...]:
    if raw is _ABSENT or raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("Ignoring preset transforms of type %s", type(raw).__name__)
        return ()
    return tuple(
        transform
        for index, entry in enumerate(raw)
        if (transform := _read_transform(entry, index)) is not None
    )


def _encode_layer_noise(entries: tuple[tuple[str, float], ...]) -> list[dict]:
    # Defends the sparse invariant even against a directly built ControlState
    # that skipped mapping.py's own neutral-drop, not only against what a
    # legitimate edit path would ever produce. Also drops a non-finite
    # strength rather than writing it: `json.dump` would otherwise emit a
    # literal NaN/Infinity token, which is not valid JSON for a strict parser
    # reading this user-facing file back.
    result = []
    for layer, strength in entries:
        if strength == 0.0:
            continue
        if not math.isfinite(strength):
            logger.warning(
                "Not writing layer_noise for %s, non finite strength %r",
                layer,
                strength,
            )
            continue
        result.append({"layer": layer, "strength": strength})
    return result


def _read_layer_noise(raw: object) -> tuple[tuple[str, float], ...]:
    if raw is _ABSENT or raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("Ignoring preset layer_noise of type %s", type(raw).__name__)
        return ()
    # Last-wins by layer: `mapping.py` maintains at most one entry per layer,
    # so a hand edited file with two rows for the same layer must not load
    # both and leave a stale duplicate `/bend/noise` can no longer see.
    seen: dict[str, float] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            logger.warning("Skipping layer_noise %d, not an object", index)
            continue
        layer = entry.get("layer")
        if not isinstance(layer, str) or not layer:
            logger.warning("Skipping layer_noise %d, invalid layer %r", index, layer)
            continue
        strength = _finite_or_none(entry.get("strength"))
        if strength is None:
            logger.warning(
                "Skipping layer_noise %d, invalid strength %r",
                index,
                entry.get("strength"),
            )
            continue
        if layer in seen:
            logger.warning("Overriding duplicate layer_noise entry for %s", layer)
        # Neutral is stored as absence, matching the sparse invariant
        # mapping.py keeps at apply time: a hand edited file that writes one
        # anyway must not materialize it back, and a later neutral entry for
        # a layer already seen removes it rather than merely being skipped.
        if strength == 0.0:
            seen.pop(layer, None)
            continue
        seen[layer] = strength
    return tuple(seen.items())


def _encode_layer_ratios(entries: tuple[tuple[str, float, float], ...]) -> list[dict]:
    result = []
    for layer, rx, ry in entries:
        if rx == 1.0 and ry == 1.0:
            continue
        if not (math.isfinite(rx) and math.isfinite(ry)):
            logger.warning(
                "Not writing layer_ratios for %s, non finite ratio (%r, %r)",
                layer,
                rx,
                ry,
            )
            continue
        result.append({"layer": layer, "rx": rx, "ry": ry})
    return result


def _read_layer_ratios(raw: object) -> tuple[tuple[str, float, float], ...]:
    if raw is _ABSENT or raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("Ignoring preset layer_ratios of type %s", type(raw).__name__)
        return ()
    # Last-wins by layer, same reasoning as `_read_layer_noise` above.
    seen: dict[str, tuple[float, float]] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            logger.warning("Skipping layer_ratios %d, not an object", index)
            continue
        layer = entry.get("layer")
        if not isinstance(layer, str) or not layer:
            logger.warning("Skipping layer_ratios %d, invalid layer %r", index, layer)
            continue
        rx = _finite_or_none(entry.get("rx"))
        ry = _finite_or_none(entry.get("ry"))
        if rx is None or ry is None:
            logger.warning("Skipping layer_ratios %d, invalid ratio %r", index, entry)
            continue
        if layer in seen:
            logger.warning("Overriding duplicate layer_ratios entry for %s", layer)
        if rx == 1.0 and ry == 1.0:
            seen.pop(layer, None)
            continue
        seen[layer] = (rx, ry)
    return tuple((layer, rx, ry) for layer, (rx, ry) in seen.items())


def _encode_directions(directions: tuple[tuple[float, ...], ...]) -> dict | None:
    """Directions as a 2D array descriptor: up to eight equal-length rows.

    Reuses `_encode_array`'s dtype/base64 machinery on the flattened values,
    then overrides the shape it would have computed (1D) with the real
    `[rows, cols]` shape, rather than inventing a second encoding.

    Guards ragged rows and an over-eight count itself, the same standard
    already applied to `layer_noise`/`layer_ratios`: a directly built
    `ControlState` that skips `mapping.py`'s own invariants must not silently
    write corrupted or truncated data, flattening ragged rows into a
    plausible-looking but wrong rectangle, or losing rows past eight with no
    warning.
    """
    if not directions:
        return None
    if len(directions) > 8:
        logger.warning(
            "Not writing directions, more than eight vectors (%d)", len(directions)
        )
        return None
    lengths = {len(vector) for vector in directions}
    if len(lengths) > 1:
        logger.warning(
            "Not writing directions, vectors of differing lengths %r", sorted(lengths)
        )
        return None
    flat = tuple(value for vector in directions for value in vector)
    encoded = _encode_array(flat)
    if encoded is None:
        return None
    encoded["shape"] = [len(directions), len(directions[0])]
    return encoded


def _read_directions(raw: object) -> tuple[tuple[float, ...], ...]:
    if raw is _ABSENT or raw is None:
        return ()
    flat = _decode_array(raw, "directions")
    if flat is None:
        return ()
    shape = raw.get("shape") if isinstance(raw, dict) else None
    if not (isinstance(shape, list) and len(shape) == 2):
        logger.warning("Skipping directions, expected a 2D shape, got %r", shape)
        return ()
    rows, cols = shape
    if rows > 8:
        logger.warning("Skipping directions, more than eight vectors (%d)", rows)
        return ()
    return tuple(tuple(flat[i * cols : (i + 1) * cols]) for i in range(rows))


def _encode_combined_layers(entries: tuple[str, ...]) -> list[str]:
    return list(entries)


def _read_combined_layers(raw: object) -> tuple[str, ...]:
    if raw is _ABSENT or raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("Ignoring preset combined_layers of type %s", type(raw).__name__)
        return ()
    entries = []
    for index, entry in enumerate(raw):
        if entry not in ("A", "B", "X"):
            logger.warning("Skipping combined_layers %d, invalid entry %r", index, entry)
            continue
        entries.append(entry)
    return tuple(entries)


def to_payload(state: ControlState) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "model": _model_reference(state.pkl_path),
        "model2": _model_reference(state.pkl2),
        "params": {
            name: _jsonable(getattr(state, name))
            for name, spec in REGISTRY.items()
            if spec.preset
        },
        "bindings": [
            {
                "target": binding.target,
                "source": binding.source,
                "expression": binding.expression,
                "enabled": binding.enabled,
            }
            for binding in state.bindings
        ],
        "latent_vec": _encode_array(state.latent_vec),
        "keyframes": [
            {
                "kind": keyframe.kind,
                "seed_x": keyframe.seed_x,
                "seed_y": keyframe.seed_y,
                "project": keyframe.project,
                "vec": _encode_array(keyframe.vec),
            }
            for keyframe in state.keyframes
        ],
        "transforms": [
            encoded
            for t in state.transforms
            if (encoded := _encode_transform(t)) is not None
        ],
        "layer_noise": _encode_layer_noise(state.layer_noise),
        "layer_ratios": _encode_layer_ratios(state.layer_ratios),
        "directions": _encode_directions(state.directions),
        "combined_layers": _encode_combined_layers(state.combined_layers),
    }


def _check_envelope(payload: object) -> None:
    """Reject anything that is not a preset, and warn about a newer format.

    A wrong `format` is a wrong file, not an old file, so it raises rather than
    loading a surprising subset of it.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"preset payload is {type(payload).__name__}, not an object")
    if payload.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} file: format is {payload.get('format')!r}")
    version = payload.get("version")
    if not isinstance(version, int):
        logger.warning(
            "Preset version %r is not a number, loading best effort", version
        )
    elif version > VERSION:
        logger.warning(
            "Preset version %d is newer than %d, loading what this build knows",
            version,
            VERSION,
        )


def _read_params(raw: object) -> dict:
    if raw is _ABSENT:
        logger.info("Preset holds no parameters, current values are kept")
        return {}
    if not isinstance(raw, dict):
        logger.warning("Ignoring preset params of type %s", type(raw).__name__)
        return {}
    values = {}
    for name, value in raw.items():
        spec = REGISTRY.get(name)
        if spec is None or not spec.preset:
            logger.warning(
                "Skipping preset parameter %s, this build does not persist it", name
            )
            continue
        # `json.load` accepts bare NaN, Infinity and -Infinity, so a hand edited
        # file carries them in. They cannot be clamped into range, so they are
        # rejected here where the rest of the malformed entry handling lives.
        if isinstance(value, float) and not math.isfinite(value):
            logger.warning(
                "Skipping preset parameter %s, %r is not a finite number", name, value
            )
            continue
        # A null is how an unset parameter is written. Keeping the current
        # value matches how a missing key behaves.
        if value is not None:
            values[name] = value
    return values


def _read_bindings(raw: object) -> tuple[Binding, ...]:
    if raw is _ABSENT:
        # A preset is a whole look, so no bindings section means no bindings.
        # Said plainly, because it discards mappings the performer set up.
        logger.warning("Preset holds no bindings, clearing every mapping")
        return ()
    if not isinstance(raw, list):
        logger.warning("Ignoring preset bindings of type %s", type(raw).__name__)
        return ()
    bindings: list[Binding] = []
    seen: set[str] = set()
    for entry in raw:
        binding = _read_binding(entry)
        if binding is None:
            continue
        # At most one binding per target, enforced here because this is where a
        # hand edited file could introduce a second one. Downstream code that
        # updates a binding only replaces the first match.
        if binding.target in seen:
            logger.warning("Skipping extra preset binding for %s", binding.target)
            continue
        seen.add(binding.target)
        bindings.append(binding)
    return tuple(bindings)


def _read_binding(entry: object) -> Binding | None:
    if not isinstance(entry, dict):
        logger.warning("Skipping preset binding of type %s", type(entry).__name__)
        return None
    fields = {name: entry.get(name) for name in _BINDING_FIELDS}
    fields["expression"] = entry.get("expression", "x")
    if not all(isinstance(fields[name], str) for name in _BINDING_FIELDS):
        logger.warning("Skipping malformed preset binding %r", entry)
        return None
    if fields["target"] not in REGISTRY:
        logger.warning(
            "Skipping preset binding for %s, this build has no such parameter",
            fields["target"],
        )
        return None
    # A bad expression is kept so the performer can see and fix the typo. It is
    # compiled by the control loop, which records the error on the binding.
    return Binding(enabled=bool(entry.get("enabled", True)), error=None, **fields)


def _read_latent_vec(raw: object) -> tuple[float, ...]:
    if raw is _ABSENT or raw is None:
        return _DEFAULT_LATENT_VEC
    decoded = _decode_array(raw, "latent_vec")
    return decoded if decoded is not None else _DEFAULT_LATENT_VEC


def _finite_number(value: object, default: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        logger.warning("Skipping %s, not a number", what)
        return default
    if not math.isfinite(value):
        logger.warning("Skipping %s, not a finite number", what)
        return default
    return float(value)


def _read_keyframe(entry: object, index: int) -> Keyframe | None:
    if not isinstance(entry, dict):
        logger.warning("Skipping keyframe %d, not an object", index)
        return None
    kind = entry.get("kind")
    if kind not in ("seed", "vec"):
        logger.warning("Skipping keyframe %d, invalid kind %r", index, kind)
        return None
    seed_x = _finite_number(entry.get("seed_x", 0.0), 0.0, f"keyframe {index} seed_x")
    seed_y = _finite_number(entry.get("seed_y", 0.0), 0.0, f"keyframe {index} seed_y")
    project = entry.get("project", True)
    if not isinstance(project, bool):
        logger.warning("Skipping keyframe %d, project is not a bool", index)
        project = True
    vec: tuple[float, ...] = ()
    if kind == "vec":
        vec = _decode_array(entry.get("vec"), f"keyframe {index} vec") or ()
    return Keyframe(kind=kind, seed_x=seed_x, seed_y=seed_y, vec=vec, project=project)


def _read_keyframes(raw: object) -> tuple[Keyframe, ...]:
    if raw is _ABSENT or raw is None:
        return _DEFAULT_KEYFRAMES
    if not isinstance(raw, list) or not raw:
        logger.warning("Ignoring preset keyframes of type %s", type(raw).__name__)
        return _DEFAULT_KEYFRAMES
    keyframes = [
        keyframe
        for index, entry in enumerate(raw)
        if (keyframe := _read_keyframe(entry, index)) is not None
    ]
    if not keyframes:
        logger.warning("Every preset keyframe was invalid, using the default loop")
        return _DEFAULT_KEYFRAMES
    return tuple(keyframes)


def from_payload(payload: dict) -> PresetData:
    """Split a payload into what the control loop needs, all validated.

    Raises `ValueError` if `payload` is not a preset. Values are returned
    uncoerced; clamping happens when they are applied to a state. The model,
    when it resolves, rides along in `params["pkl_path"]` so the caller applies
    it through the same path as every other parameter; the second mixing
    model does the same in `params["pkl2"]`. Unresolved names are reported
    separately as `missing_model`/`missing_model2` rather than raising, since
    a preset with a model this machine does not have is not a broken preset.
    """
    _check_envelope(payload)
    params = _read_params(payload.get("params", _ABSENT))
    model_path, missing_model = _read_model(payload.get("model", _ABSENT))
    if model_path is not None:
        params["pkl_path"] = model_path
    model2_path, missing_model2 = _read_model(payload.get("model2", _ABSENT))
    if model2_path is not None:
        params["pkl2"] = model2_path
    return PresetData(
        params=params,
        bindings=_read_bindings(payload.get("bindings", _ABSENT)),
        latent_vec=_read_latent_vec(payload.get("latent_vec", _ABSENT)),
        keyframes=_read_keyframes(payload.get("keyframes", _ABSENT)),
        missing_model=missing_model,
        missing_model2=missing_model2,
        transforms=_read_transforms(payload.get("transforms", _ABSENT)),
        layer_noise=_read_layer_noise(payload.get("layer_noise", _ABSENT)),
        layer_ratios=_read_layer_ratios(payload.get("layer_ratios", _ABSENT)),
        directions=_read_directions(payload.get("directions", _ABSENT)),
        combined_layers=_read_combined_layers(payload.get("combined_layers", _ABSENT)),
    )


def save(state: ControlState, path: str | Path) -> None:
    """Write a preset, replacing any existing one only once it is complete.

    A performer's saved look is not worth losing to an interrupted or full disk
    write, so the new file is built beside the old one and swapped in.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as fp:
            json.dump(to_payload(state), fp, indent=2)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load(path: str | Path) -> dict:
    """Read and validate a preset file, returning its payload."""
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    _check_envelope(payload)
    return payload


def preset_dir() -> Path:
    """Directory holding the user's presets, created on demand."""
    # Imported here because the user data root is a legacy flat root module,
    # and only this function needs it. The control path stays independent of it.
    from utils.user_data import data_path

    directory = data_path("live") / "presets"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_presets(directory: str | Path | None = None) -> list[str]:
    """Preset names, without the `.json` suffix, sorted."""
    root = Path(directory) if directory is not None else preset_dir()
    return sorted(path.stem for path in root.glob("*.json") if path.is_file())
