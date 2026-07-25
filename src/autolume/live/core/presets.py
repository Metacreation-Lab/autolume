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
otherwise, so a shared preset can resolve on the machine that opens it.
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

from autolume.live.core.params import REGISTRY, Binding, ControlState, Keyframe

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


def to_payload(state: ControlState) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "model": _model_reference(state.pkl_path),
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
    it through the same path as every other parameter.
    """
    _check_envelope(payload)
    params = _read_params(payload.get("params", _ABSENT))
    model_path, missing_model = _read_model(payload.get("model", _ABSENT))
    if model_path is not None:
        params["pkl_path"] = model_path
    return PresetData(
        params=params,
        bindings=_read_bindings(payload.get("bindings", _ABSENT)),
        latent_vec=_read_latent_vec(payload.get("latent_vec", _ABSENT)),
        keyframes=_read_keyframes(payload.get("keyframes", _ABSENT)),
        missing_model=missing_model,
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
