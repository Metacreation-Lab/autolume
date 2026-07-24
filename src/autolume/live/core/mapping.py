"""Apply transport-agnostic control events onto the control state."""

import dataclasses
import logging

from autolume.live.core.events import ControlEvent
from autolume.live.core.params import BY_ADDRESS, ControlState, ParamKind, ParamSpec

logger = logging.getLogger(__name__)


def _coerce(spec: ParamSpec, value: object) -> object:
    if spec.kind is ParamKind.FLOAT:
        coerced = float(value)
    elif spec.kind is ParamKind.INT:
        coerced = int(round(float(value)))
    elif spec.kind is ParamKind.BOOL:
        return bool(float(value)) if not isinstance(value, bool) else value
    else:
        return str(value)
    if spec.minimum is not None:
        coerced = max(coerced, type(coerced)(spec.minimum))
    if spec.maximum is not None:
        coerced = min(coerced, type(coerced)(spec.maximum))
    return coerced


def apply_event(state: ControlState, event: ControlEvent) -> ControlState:
    spec = BY_ADDRESS.get(event.address)
    if spec is None:
        logger.debug("Ignoring event for unknown address %s", event.address)
        return state
    try:
        coerced = _coerce(spec, event.value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring uncoercible value %r for %s", event.value, event.address
        )
        return state
    return dataclasses.replace(state, **{spec.name: coerced})
