"""Declarative parameter registry and immutable state snapshots.

Every performable parameter is declared once here. UI bindings, transport
address lookup, preset serialization, and tests all derive from REGISTRY.
"""

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParamKind(Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STR = "str"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: ParamKind
    default: object
    address: str
    minimum: float | None = None
    maximum: float | None = None
    preset: bool = True


_SPECS = (
    ParamSpec("pkl_path", ParamKind.STR, None, "/model/path"),
    ParamSpec("latent_x", ParamKind.FLOAT, 0.0, "/latent/x"),
    ParamSpec("latent_y", ParamKind.FLOAT, 0.0, "/latent/y"),
    ParamSpec("anim_playing", ParamKind.BOOL, False, "/anim/playing"),
    ParamSpec("anim_speed_x", ParamKind.FLOAT, 0.25, "/anim/speed/x", -10.0, 10.0),
    ParamSpec("anim_speed_y", ParamKind.FLOAT, 0.0, "/anim/speed/y", -10.0, 10.0),
    ParamSpec("truncation_psi", ParamKind.FLOAT, 0.7, "/trunc/psi", -1.0, 2.0),
    ParamSpec("fps_cap", ParamKind.INT, 60, "/render/fps", 0, 240),
)

REGISTRY: dict[str, ParamSpec] = {spec.name: spec for spec in _SPECS}
BY_ADDRESS: dict[str, ParamSpec] = {spec.address: spec for spec in _SPECS}

# Structured control addresses. They carry Python objects rather than scalars,
# so they are reserved and never registry parameters.
BINDING_SET = "/binding/set"
BINDING_CLEAR = "/binding/clear"


@dataclass(frozen=True)
class Binding:
    """A source address driving one registry parameter through an expression.

    `error` holds the last compile or evaluation failure so the mapping panel
    can show it. It is runtime state and is never persisted.
    """

    target: str
    source: str
    expression: str = "x"
    enabled: bool = True
    error: str | None = None


@dataclass(frozen=True)
class ControlState:
    pkl_path: str | None = None
    latent_x: float = 0.0
    latent_y: float = 0.0
    anim_playing: bool = False
    anim_speed_x: float = 0.25
    anim_speed_y: float = 0.0
    truncation_psi: float = 0.7
    fps_cap: int = 60
    bindings: tuple[Binding, ...] = ()


@dataclass(frozen=True)
class RenderParams:
    pkl_path: str | None
    latent_x: float
    latent_y: float
    truncation_psi: float
    fps_cap: int


def to_render_params(state: ControlState) -> RenderParams:
    return RenderParams(
        pkl_path=state.pkl_path,
        latent_x=state.latent_x,
        latent_y=state.latent_y,
        truncation_psi=state.truncation_psi,
        fps_cap=state.fps_cap,
    )


def _coerce(spec: ParamSpec, value: object) -> object:
    """Coerce `value` to the kind of `spec` and clamp it to its bounds.

    Raises `TypeError` or `ValueError` if the value cannot be coerced.
    """
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


def apply_value(state: ControlState, name: str, value: object) -> ControlState:
    """Return `state` with parameter `name` set to a coerced, clamped `value`.

    An unknown name or an uncoercible value leaves the state unchanged.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        logger.debug("Ignoring value for unknown parameter %s", name)
        return state
    try:
        coerced = _coerce(spec, value)
    except (TypeError, ValueError):
        logger.warning("Ignoring uncoercible value %r for %s", value, name)
        return state
    return dataclasses.replace(state, **{name: coerced})
