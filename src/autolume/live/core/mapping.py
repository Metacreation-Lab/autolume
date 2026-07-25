"""Apply transport-agnostic control events onto the control state."""

import dataclasses
import logging
import math

from autolume.live.core.events import ControlEvent
from autolume.live.core.expr import ExpressionError, compile_expression
from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    BY_ADDRESS,
    KEYFRAME_REMOVE,
    KEYFRAME_SET,
    REGISTRY,
    VECTOR_RANDOMIZE,
    VECTOR_SET,
    Binding,
    ClearBinding,
    ControlState,
    Keyframe,
    RemoveKeyframe,
    SetKeyframe,
    SetVector,
    apply_value,
    default_keyframe,
)
from autolume.live.core.presets import PRESET_APPLY, from_payload

logger = logging.getLogger(__name__)


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

    `keyframe_count` is not applied from `params`: it is derived from the
    loaded `keyframes` tuple instead, here, so the two can never disagree the
    way they used to when a preset set one without the other.
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
        if name == "keyframe_count":
            continue
        applied = apply_value(applied, name, param_value)
    return dataclasses.replace(
        applied,
        bindings=data.bindings,
        latent_vec=data.latent_vec,
        keyframes=data.keyframes,
        keyframe_count=len(data.keyframes),
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
    return dataclasses.replace(
        state, keyframes=tuple(keyframes), keyframe_count=len(keyframes)
    )


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
    return dataclasses.replace(
        state, keyframes=remaining, keyframe_count=len(remaining)
    )


def _resize_keyframes(state: ControlState, value: object, address: str) -> ControlState:
    """Apply a `keyframe_count` write, resizing `keyframes` to match.

    Goes through `apply_value` for the coercion and clamping every other
    parameter gets, then grows or shrinks the tuple to the clamped count,
    preserving the prefix and filling new slots with seed keyframes.
    """
    resized = apply_value(state, "keyframe_count", value, address)
    # apply_value returns the same object, unchanged, when the value could
    # not be coerced. Nothing to resize in that case.
    if resized is state:
        return state
    count = resized.keyframe_count
    current = list(state.keyframes)
    if count > len(current):
        current.extend(default_keyframe(i) for i in range(len(current), count))
    elif count < len(current):
        current = current[:count]
    else:
        return resized
    return dataclasses.replace(resized, keyframes=tuple(current))


def apply_event(state: ControlState, event: ControlEvent) -> ControlState:
    if event.address == BINDING_SET:
        return _set_binding(state, event.value)
    if event.address == BINDING_CLEAR:
        return _clear_binding(state, event.value)
    if event.address == PRESET_APPLY:
        return _apply_preset(state, event.value)
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
        return _remove_keyframe(state, event.value)
    spec = BY_ADDRESS.get(event.address)
    if spec is None:
        logger.debug("Ignoring event for unknown address %s", event.address)
        return state
    if spec.name == "keyframe_count":
        return _resize_keyframes(state, event.value, event.address)
    return apply_value(state, spec.name, event.value, event.address)
