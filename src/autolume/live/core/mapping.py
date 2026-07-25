"""Apply transport-agnostic control events onto the control state."""

import dataclasses
import logging

from autolume.live.core.events import ControlEvent
from autolume.live.core.expr import ExpressionError, compile_expression
from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    BY_ADDRESS,
    REGISTRY,
    Binding,
    ClearBinding,
    ControlState,
    apply_value,
)

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


def apply_event(state: ControlState, event: ControlEvent) -> ControlState:
    if event.address == BINDING_SET:
        return _set_binding(state, event.value)
    if event.address == BINDING_CLEAR:
        return _clear_binding(state, event.value)
    spec = BY_ADDRESS.get(event.address)
    if spec is None:
        logger.debug("Ignoring event for unknown address %s", event.address)
        return state
    return apply_value(state, spec.name, event.value, event.address)
