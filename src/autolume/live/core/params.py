"""Declarative parameter registry and immutable state snapshots.

Every performable parameter is declared once here. UI bindings, transport
address lookup, preset serialization, and tests all derive from REGISTRY.
"""

import dataclasses
import logging
import math
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
    ParamSpec("global_noise", ParamKind.FLOAT, 1.0, "/noise/global", 0.0, 2.0),
    ParamSpec("noise_enabled", ParamKind.BOOL, True, "/noise/enabled"),
    ParamSpec("noise_seed", ParamKind.INT, 0, "/noise/seed", 0, 2**31 - 1),
    ParamSpec("noise_anim", ParamKind.BOOL, False, "/noise/anim"),
    # Not persisted: the frame limit is a property of the machine, not of the
    # look. A preset saved on a laptop capped at 30 must not cap stage hardware.
    ParamSpec("fps_cap", ParamKind.INT, 60, "/render/fps", 0, 240, preset=False),
)

REGISTRY: dict[str, ParamSpec] = {spec.name: spec for spec in _SPECS}
BY_ADDRESS: dict[str, ParamSpec] = {spec.address: spec for spec in _SPECS}

# Structured control addresses. They carry Python objects rather than scalars,
# so they are reserved and never registry parameters.
BINDING_SET = "/binding/set"
BINDING_CLEAR = "/binding/clear"


@dataclass(frozen=True)
class Binding:
    """One parameter's mapping row: what may write it, and through what.

    Remote input is off until a row says otherwise, so the row is what turns it
    on rather than what takes it away. A parameter with no row here accepts
    nothing from outside, whatever address arrives.

    `source` is the one address that reaches the parameter, and an empty source
    means the parameter's own canonical address, so a row that only says On
    opens the address the registry gives the parameter. `enabled` is the row's
    switch and governs every remote writer, which is why a row exists at all
    for a parameter with no source: switching remote input on is a state that
    has to be recorded somewhere, and recorded here it persists in a preset
    like any other mapping instead of needing a parallel set of names.

    `error` holds the last compile or evaluation failure so the mapping panel
    can show it. It is runtime state and is never persisted.
    """

    target: str
    source: str
    expression: str = "x"
    enabled: bool = True
    error: str | None = None


@dataclass(frozen=True)
class ClearBinding:
    """Request to remove the binding driving one registry parameter.

    A dedicated value object rather than a bare target string, so that no OSC
    message can express a clear and reconfigure a performance remotely.
    """

    target: str


def binding_for(bindings: tuple[Binding, ...], name: str) -> Binding | None:
    """The row governing `name`, or None when nothing has been recorded for it."""
    for binding in bindings:
        if binding.target == name:
            return binding
    return None


def listens_on(binding: Binding) -> str:
    """The one address a row lets through to its parameter.

    An empty source is not a broken row, it is the plainest one: the
    parameter's own canonical address. A row that only says On is how a
    performer opens `/anim/playing` without having to type it.
    """
    if binding.source:
        return binding.source
    spec = REGISTRY.get(binding.target)
    return spec.address if spec is not None else ""


@dataclass(frozen=True)
class ControlState:
    pkl_path: str | None = None
    latent_x: float = 0.0
    latent_y: float = 0.0
    anim_playing: bool = False
    anim_speed_x: float = 0.25
    anim_speed_y: float = 0.0
    truncation_psi: float = 0.7
    global_noise: float = 1.0
    noise_enabled: bool = True
    noise_seed: int = 0
    noise_anim: bool = False
    fps_cap: int = 60
    bindings: tuple[Binding, ...] = ()


@dataclass(frozen=True)
class RenderParams:
    pkl_path: str | None
    latent_x: float
    latent_y: float
    truncation_psi: float
    global_noise: float
    noise_enabled: bool
    noise_seed: int
    noise_anim: bool
    fps_cap: int


def to_render_params(state: ControlState) -> RenderParams:
    return RenderParams(
        pkl_path=state.pkl_path,
        latent_x=state.latent_x,
        latent_y=state.latent_y,
        truncation_psi=state.truncation_psi,
        global_noise=state.global_noise,
        noise_enabled=state.noise_enabled,
        noise_seed=state.noise_seed,
        noise_anim=state.noise_anim,
        fps_cap=state.fps_cap,
    )


def _finite(number: float) -> float:
    if not math.isfinite(number):
        raise ValueError(f"{number} is not a finite number")
    return number


def _coerce(spec: ParamSpec, value: object) -> object:
    """Coerce `value` to the kind of `spec` and clamp it to its bounds.

    Raises `TypeError`, `ValueError` or `OverflowError` if the value cannot be
    coerced. A NaN or an infinity is refused rather than clamped: `max` and
    `min` propagate a NaN instead of bounding it, so it would land in the state
    claiming to be within its declared range, and it is a broken input rather
    than an extreme one.
    """
    if spec.kind is ParamKind.FLOAT:
        coerced = _finite(float(value))
    elif spec.kind is ParamKind.INT:
        coerced = int(round(_finite(float(value))))
    elif spec.kind is ParamKind.BOOL:
        return bool(float(value)) if not isinstance(value, bool) else value
    else:
        return str(value)
    if spec.minimum is not None:
        coerced = max(coerced, type(coerced)(spec.minimum))
    if spec.maximum is not None:
        coerced = min(coerced, type(coerced)(spec.maximum))
    return coerced


def apply_value(
    state: ControlState, name: str, value: object, address: str | None = None
) -> ControlState:
    """Return `state` with parameter `name` set to a coerced, clamped `value`.

    An unknown name or an uncoercible value leaves the state unchanged. Pass the
    wire `address` the value arrived on so the warning names what the operator
    sent rather than our internal field name.

    `OverflowError` is caught alongside the type errors because it is what an
    oversized numeric input raises, and it is an `ArithmeticError` rather than a
    `ValueError`, so it would otherwise escape all the way to the control loop
    and cost the whole event.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        logger.debug("Ignoring value for unknown parameter %s", name)
        return state
    try:
        coerced = _coerce(spec, value)
    except (TypeError, ValueError, OverflowError):
        logger.warning("Ignoring uncoercible value %r on %s", value, address or name)
        return state
    return dataclasses.replace(state, **{name: coerced})
