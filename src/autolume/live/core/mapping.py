"""Apply transport-agnostic control events onto the control state."""

import dataclasses
import logging
import math

from autolume.live.core.events import ControlEvent
from autolume.live.core.expr import ExpressionError, compile_expression
from autolume.live.core.params import (
    ADJUST_DIRECTIONS,
    BEND_NOISE,
    BEND_RATIO,
    BEND_REMOVE,
    BEND_SET,
    BINDING_CLEAR,
    BINDING_SET,
    BY_ADDRESS,
    KEYFRAME_REMOVE,
    KEYFRAME_SET,
    MIX_LAYERS,
    REGISTRY,
    VECTOR_RANDOMIZE,
    VECTOR_SET,
    Binding,
    ClearBinding,
    ControlState,
    Keyframe,
    RemoveKeyframe,
    RemoveTransform,
    SetCombinedLayers,
    SetDirections,
    SetKeyframe,
    SetLayerNoise,
    SetLayerRatio,
    SetTransform,
    SetVector,
    Transform,
    apply_value,
)
from autolume.live.core.presets import PRESET_APPLY, from_payload

logger = logging.getLogger(__name__)

# Arity of each of the eleven UI-exposed bending operators, derived from
# `bending/transform_layers.py`'s `ManipulationLayer.forward` call sites:
# `translate` builds a 2-vector (`torch.tensor([params])` fed straight to
# kornia's translate), every other operator reads only `params[0]`. `sobel`,
# `canny` and `resize` are deliberately excluded, they stay unexposed.
_OPERATOR_ARITY: dict[str, int] = {
    "translate": 2,
    "rotate": 1,
    "scale": 1,
    "erode": 1,
    "dilate": 1,
    "invert": 1,
    "flip-h": 1,
    "flip-v": 1,
    "binary-thresh": 1,
    "scalar-multiply": 1,
    "ablate": 1,
}

_VALID_LAYER_ORIGINS = ("A", "B", "X")

# `Scale` inverts params[0] into the affine matrix kornia's grid_sample uses
# to build sample coordinates (transform_layers.py:76-80): the closer the
# factor sits to zero, the larger that inverted coefficient gets, and past a
# certain magnitude grid_sample stops raising and instead takes the process
# down with a native SIGBUS that no try/except can contain. Measured
# empirically (see .superpowers/sdd/plan-4/scale-guard-report.md): on this
# machine the crash boundary sits at |factor| ~= 3.3e-9, symmetric in sign.
# This minimum sits roughly 300x above that measured boundary.
_MIN_SCALE_MAGNITUDE = 1e-6


def _set_binding(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, Binding):
        logger.warning("Ignoring non binding value %r on %s", value, BINDING_SET)
        return state
    fields = (value.target, value.source, value.expression)
    if not all(isinstance(field, str) for field in fields):
        logger.warning("Ignoring malformed binding %r on %s", value, BINDING_SET)
        return state
    if value.target not in REGISTRY:
        logger.warning("Ignoring binding for unknown parameter %s", value.target)
        return state
    try:
        compile_expression(value.expression)
    except ExpressionError as exc:
        binding = dataclasses.replace(value, error=str(exc))
    else:
        binding = dataclasses.replace(value, error=None)
    bindings = list(state.bindings)
    for index, existing in enumerate(bindings):
        if existing.target == binding.target:
            bindings[index] = binding
            break
    else:
        bindings.append(binding)
    return dataclasses.replace(state, bindings=tuple(bindings))


def _clear_binding(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, ClearBinding):
        logger.warning(
            "Ignoring non clear binding value %r on %s", value, BINDING_CLEAR
        )
        return state
    if not isinstance(value.target, str):
        logger.warning(
            "Ignoring malformed clear binding %r on %s", value, BINDING_CLEAR
        )
        return state
    remaining = tuple(b for b in state.bindings if b.target != value.target)
    if len(remaining) == len(state.bindings):
        return state
    return dataclasses.replace(state, bindings=remaining)


def _apply_preset(state: ControlState, value: object) -> ControlState:
    """Apply a whole preset payload as one state replacement.

    Every value goes through `apply_value`, so a hand edited file cannot push a
    parameter out of range, and the caller only ever sees the finished state.

    No `keyframe_count` special case any more: the registry carries no such
    parameter, so `from_payload`'s own unknown-parameter handling
    (`presets.py` `_read_params`) already drops a stray one from an older
    preset file, the same tolerance any other retired parameter gets.
    """
    if not isinstance(value, dict):
        logger.warning("Ignoring non preset value %r on %s", value, PRESET_APPLY)
        return state
    try:
        data = from_payload(value)
    except ValueError as exc:
        logger.warning("Ignoring malformed preset payload: %s", exc)
        return state
    applied = state
    for name, param_value in data.params.items():
        applied = apply_value(applied, name, param_value)
    return dataclasses.replace(
        applied,
        bindings=data.bindings,
        latent_vec=data.latent_vec,
        keyframes=data.keyframes,
        transforms=data.transforms,
        layer_noise=data.layer_noise,
        layer_ratios=data.layer_ratios,
        directions=data.directions,
        combined_layers=data.combined_layers,
    )


def _coerce_vector(raw: object) -> tuple[float, ...] | None:
    """Coerce a sequence to a tuple of finite floats, or None if it cannot be.

    Rejects the whole sequence on any non-finite or non-numeric entry rather
    than dropping the bad entries, so a malformed vector never lands half set.
    """
    try:
        values = [float(item) for item in raw]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return tuple(values)


def _set_vector(state: ControlState, value: object) -> ControlState:
    if isinstance(value, SetVector):
        raw = value.values
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        logger.warning("Ignoring non vector value %r on %s", value, VECTOR_SET)
        return state
    coerced = _coerce_vector(raw)
    if coerced is None:
        logger.warning(
            "Ignoring vector with a non finite or non numeric entry on %s", VECTOR_SET
        )
        return state
    return dataclasses.replace(state, latent_vec=coerced)


def _set_keyframe(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, SetKeyframe):
        logger.warning("Ignoring non keyframe value %r on %s", value, KEYFRAME_SET)
        return state
    if not isinstance(value.index, int) or isinstance(value.index, bool):
        logger.warning(
            "Ignoring malformed keyframe index %r on %s", value.index, KEYFRAME_SET
        )
        return state
    if not isinstance(value.keyframe, Keyframe):
        logger.warning(
            "Ignoring malformed keyframe %r on %s", value.keyframe, KEYFRAME_SET
        )
        return state
    keyframe = value.keyframe
    if keyframe.kind == "vec" and _coerce_vector(keyframe.vec) is None:
        logger.warning(
            "Ignoring vec keyframe with a non finite or non numeric entry on %s",
            KEYFRAME_SET,
        )
        return state
    keyframes = list(state.keyframes)
    index = value.index
    if index == len(keyframes):
        keyframes.append(keyframe)
    elif 0 <= index < len(keyframes):
        keyframes[index] = keyframe
    else:
        logger.warning(
            "Ignoring keyframe set at out of range index %d on %s", index, KEYFRAME_SET
        )
        return state
    return dataclasses.replace(state, keyframes=tuple(keyframes))


def _remove_keyframe(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, RemoveKeyframe):
        logger.warning("Ignoring non keyframe value %r on %s", value, KEYFRAME_REMOVE)
        return state
    if not isinstance(value.index, int) or isinstance(value.index, bool):
        logger.warning(
            "Ignoring malformed keyframe index %r on %s", value.index, KEYFRAME_REMOVE
        )
        return state
    keyframes = state.keyframes
    if len(keyframes) <= 1:
        logger.warning(
            "Ignoring keyframe remove on %s, a loop needs at least one keyframe",
            KEYFRAME_REMOVE,
        )
        return state
    if not (0 <= value.index < len(keyframes)):
        logger.warning(
            "Ignoring keyframe remove at out of range index %d on %s",
            value.index,
            KEYFRAME_REMOVE,
        )
        return state
    remaining = keyframes[: value.index] + keyframes[value.index + 1 :]
    return dataclasses.replace(state, keyframes=remaining)


def _wrap_loop_index(state: ControlState) -> ControlState:
    """Keep `loop_index` inside `[0, len(keyframes))`, wherever it drifted from.

    Plan 3 Task 2 specifies this: `loop_index` writes are wrapped modulo the
    keyframe count at application time. Called from every path that can move
    `loop_index` out of step with `keyframes`: a direct write to it, a
    keyframe removal, and a preset load, which can apply a stale `loop_index`
    against a keyframe list that decoded shorter than it was saved with.
    Guarded against a zero-length tuple even though a loop is supposed to
    always keep at least one keyframe: this runs on the control thread and
    must not raise.
    """
    count = len(state.keyframes)
    wrapped = state.loop_index % count if count else 0
    if wrapped == state.loop_index:
        return state
    return dataclasses.replace(state, loop_index=wrapped)


def _validate_transform(transform: Transform) -> Transform | None:
    """Normalise and validate `transform`, or None if it cannot be applied.

    Returns a `Transform` with `params`/`indices` normalised to tuples so
    downstream code never has to re-check their shape.
    """
    arity = _OPERATOR_ARITY.get(transform.op)
    if arity is None:
        return None
    if not isinstance(transform.layer, str) or not transform.layer:
        return None
    try:
        raw_params = tuple(transform.params)
    except TypeError:
        return None
    # bool is an int subclass, so float(True) == 1.0 would otherwise sail
    # through as a normal value, same trap the indices guard below already
    # defends against. Rejected uniformly across all eleven operators: no
    # legitimate producer sends True as a rotation angle or a kernel size.
    if any(isinstance(p, bool) for p in raw_params):
        return None
    try:
        params_values = tuple(float(p) for p in raw_params)
    except (TypeError, ValueError):
        return None
    if len(params_values) != arity or not all(math.isfinite(p) for p in params_values):
        return None
    # erode/dilate build a torch.ones((k, k)) kernel from params[0]
    # (transform_layers.py:39/50): a non-integral or non-positive k raises
    # there. "Finite floats of the op's arity" is a floor, not a ceiling, so
    # these two operators get an extra check the other nine do not need.
    if transform.op in ("erode", "dilate"):
        kernel = params_values[0]
        if not kernel.is_integer() or kernel < 1:
            return None
    # scale's factor is invertible to a coefficient that reaches kornia's
    # grid_sample, see `_MIN_SCALE_MAGNITUDE` above.
    if transform.op == "scale" and abs(params_values[0]) < _MIN_SCALE_MAGNITUDE:
        return None
    try:
        indices_values = tuple(transform.indices)
    except TypeError:
        return None
    if not all(
        isinstance(i, int) and not isinstance(i, bool) and i >= 0
        for i in indices_values
    ):
        return None
    return dataclasses.replace(transform, params=params_values, indices=indices_values)


def _set_transform(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, SetTransform):
        logger.warning("Ignoring non transform value %r on %s", value, BEND_SET)
        return state
    if not isinstance(value.index, int) or isinstance(value.index, bool):
        logger.warning(
            "Ignoring malformed transform index %r on %s", value.index, BEND_SET
        )
        return state
    if not isinstance(value.transform, Transform):
        logger.warning(
            "Ignoring malformed transform %r on %s", value.transform, BEND_SET
        )
        return state
    validated = _validate_transform(value.transform)
    if validated is None:
        logger.warning(
            "Ignoring invalid transform %r on %s", value.transform, BEND_SET
        )
        return state
    transforms = list(state.transforms)
    index = value.index
    if index == len(transforms):
        transforms.append(validated)
    elif 0 <= index < len(transforms):
        transforms[index] = validated
    else:
        logger.warning(
            "Ignoring transform set at out of range index %d on %s", index, BEND_SET
        )
        return state
    return dataclasses.replace(state, transforms=tuple(transforms))


def _remove_transform(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, RemoveTransform):
        logger.warning("Ignoring non transform value %r on %s", value, BEND_REMOVE)
        return state
    if not isinstance(value.index, int) or isinstance(value.index, bool):
        logger.warning(
            "Ignoring malformed transform index %r on %s", value.index, BEND_REMOVE
        )
        return state
    transforms = state.transforms
    if not (0 <= value.index < len(transforms)):
        logger.warning(
            "Ignoring transform remove at out of range index %d on %s",
            value.index,
            BEND_REMOVE,
        )
        return state
    remaining = transforms[: value.index] + transforms[value.index + 1 :]
    return dataclasses.replace(state, transforms=remaining)


def _set_layer_noise(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, SetLayerNoise):
        logger.warning("Ignoring non layer noise value %r on %s", value, BEND_NOISE)
        return state
    if not isinstance(value.layer, str) or not value.layer:
        logger.warning(
            "Ignoring malformed layer noise %r on %s", value, BEND_NOISE
        )
        return state
    try:
        strength = float(value.strength)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring uncoercible layer noise strength %r on %s",
            value.strength,
            BEND_NOISE,
        )
        return state
    if not math.isfinite(strength):
        logger.warning(
            "Ignoring non finite layer noise strength %r on %s",
            value.strength,
            BEND_NOISE,
        )
        return state
    entries = list(state.layer_noise)
    for index, (layer, _) in enumerate(entries):
        if layer == value.layer:
            if strength == 0.0:
                del entries[index]
            else:
                entries[index] = (value.layer, strength)
            break
    else:
        if strength != 0.0:
            entries.append((value.layer, strength))
    return dataclasses.replace(state, layer_noise=tuple(entries))


def _set_layer_ratio(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, SetLayerRatio):
        logger.warning("Ignoring non layer ratio value %r on %s", value, BEND_RATIO)
        return state
    if not isinstance(value.layer, str) or not value.layer:
        logger.warning(
            "Ignoring malformed layer ratio %r on %s", value, BEND_RATIO
        )
        return state
    try:
        rx = float(value.rx)
        ry = float(value.ry)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring uncoercible layer ratio %r on %s", value, BEND_RATIO
        )
        return state
    if not (math.isfinite(rx) and math.isfinite(ry)):
        logger.warning(
            "Ignoring non finite layer ratio %r on %s", value, BEND_RATIO
        )
        return state
    neutral = rx == 1.0 and ry == 1.0
    entries = list(state.layer_ratios)
    for index, (layer, _, _) in enumerate(entries):
        if layer == value.layer:
            if neutral:
                del entries[index]
            else:
                entries[index] = (value.layer, rx, ry)
            break
    else:
        if not neutral:
            entries.append((value.layer, rx, ry))
    return dataclasses.replace(state, layer_ratios=tuple(entries))


def _coerce_directions(raw: object) -> tuple[tuple[float, ...], ...] | None:
    """Coerce up to eight direction vectors, or None if they cannot be.

    Every vector goes through `_coerce_vector`, so a non-finite or non-numeric
    entry rejects that vector's whole event, same as `/vector/set`. More than
    eight vectors, vectors of differing lengths, or (once at least one vector
    is present) zero-length vectors are also rejected: a zero-length direction
    is a meaningless adjuster state, not just an unusual one.
    """
    try:
        vectors = list(raw)
    except TypeError:
        return None
    if len(vectors) > 8:
        return None
    coerced = []
    for vector in vectors:
        one = _coerce_vector(vector)
        if one is None:
            return None
        coerced.append(one)
    lengths = {len(v) for v in coerced}
    if len(lengths) > 1:
        return None
    if lengths and next(iter(lengths)) == 0:
        return None
    return tuple(coerced)


def _set_directions(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, SetDirections):
        logger.warning("Ignoring non directions value %r on %s", value, ADJUST_DIRECTIONS)
        return state
    coerced = _coerce_directions(value.vectors)
    if coerced is None:
        logger.warning(
            "Ignoring malformed directions on %s", ADJUST_DIRECTIONS
        )
        return state
    # Weights beyond the new count are zeroed, so a stale weight from a larger
    # loaded set cannot silently keep acting on a direction that no longer
    # exists.
    zeroed = {f"adjust_w{i}": 0.0 for i in range(len(coerced) + 1, 9)}
    return dataclasses.replace(state, directions=coerced, **zeroed)


def _set_combined_layers(state: ControlState, value: object) -> ControlState:
    if not isinstance(value, SetCombinedLayers):
        logger.warning("Ignoring non combined layers value %r on %s", value, MIX_LAYERS)
        return state
    try:
        entries = tuple(value.entries)
    except TypeError:
        logger.warning(
            "Ignoring malformed combined layers %r on %s", value.entries, MIX_LAYERS
        )
        return state
    if not all(entry in _VALID_LAYER_ORIGINS for entry in entries):
        logger.warning(
            "Ignoring combined layers with an invalid entry %r on %s",
            entries,
            MIX_LAYERS,
        )
        return state
    return dataclasses.replace(state, combined_layers=entries)


def apply_event(state: ControlState, event: ControlEvent) -> ControlState:
    if event.address == BINDING_SET:
        return _set_binding(state, event.value)
    if event.address == BINDING_CLEAR:
        return _clear_binding(state, event.value)
    if event.address == PRESET_APPLY:
        return _wrap_loop_index(_apply_preset(state, event.value))
    if event.address == VECTOR_SET:
        return _set_vector(state, event.value)
    if event.address == VECTOR_RANDOMIZE:
        # Recognized only. Materializing the vector needs the model's z_dim,
        # which mapping does not have, so the control loop applies this event
        # itself once it can (see design.md). Recognizing it here just keeps
        # it out of the unknown-address path below.
        return state
    if event.address == KEYFRAME_SET:
        return _set_keyframe(state, event.value)
    if event.address == KEYFRAME_REMOVE:
        return _wrap_loop_index(_remove_keyframe(state, event.value))
    if event.address == BEND_SET:
        return _set_transform(state, event.value)
    if event.address == BEND_REMOVE:
        return _remove_transform(state, event.value)
    if event.address == BEND_NOISE:
        return _set_layer_noise(state, event.value)
    if event.address == BEND_RATIO:
        return _set_layer_ratio(state, event.value)
    if event.address == ADJUST_DIRECTIONS:
        return _set_directions(state, event.value)
    if event.address == MIX_LAYERS:
        return _set_combined_layers(state, event.value)
    spec = BY_ADDRESS.get(event.address)
    if spec is None:
        logger.debug("Ignoring event for unknown address %s", event.address)
        return state
    if spec.name == "loop_index":
        return _wrap_loop_index(apply_value(state, spec.name, event.value, event.address))
    return apply_value(state, spec.name, event.value, event.address)
