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
    # Persisted separately as the preset's `model` key, not as a plain param:
    # it needs path resolution a scalar value cannot express. See presets.py.
    ParamSpec("pkl_path", ParamKind.STR, None, "/model/path", preset=False),
    ParamSpec("latent_x", ParamKind.FLOAT, 0.0, "/latent/x"),
    ParamSpec("latent_y", ParamKind.FLOAT, 0.0, "/latent/y"),
    ParamSpec("anim_playing", ParamKind.BOOL, False, "/anim/playing"),
    ParamSpec("anim_speed_x", ParamKind.FLOAT, 0.25, "/anim/speed/x", -10.0, 10.0),
    ParamSpec("anim_speed_y", ParamKind.FLOAT, 0.0, "/anim/speed/y", -10.0, 10.0),
    # 0.8 because that is what the old app opens at, and truncation is the one
    # parameter whose default is visible the moment a model loads: it decides
    # how far the first frame sits from the model's average, which is how
    # saturated and how distinctive it looks. A different default here reads as
    # the new app rendering the same model differently.
    ParamSpec("truncation_psi", ParamKind.FLOAT, 0.8, "/trunc/psi", -1.0, 2.0),
    ParamSpec("global_noise", ParamKind.FLOAT, 1.0, "/noise/global", 0.0, 2.0),
    ParamSpec("noise_enabled", ParamKind.BOOL, True, "/noise/enabled"),
    ParamSpec("noise_seed", ParamKind.INT, 0, "/noise/seed", 0, 2**31 - 1),
    ParamSpec("noise_anim", ParamKind.BOOL, False, "/noise/anim"),
    # Not persisted: the frame limit is a property of the machine, not of the
    # look. A preset saved on a laptop capped at 30 must not cap stage hardware.
    ParamSpec("fps_cap", ParamKind.INT, 60, "/render/fps", 0, 240, preset=False),
    # Latent navigation mode. Off is seed mode; vector_mode drifts a raw latent
    # vector instead of walking the seed grid.
    ParamSpec("vector_mode", ParamKind.BOOL, False, "/latent/vector"),
    ParamSpec("latent_project", ParamKind.BOOL, True, "/latent/project"),
    # Keyframe and noise loop playback. loop_active overrides latent
    # navigation entirely while it is set.
    ParamSpec("loop_active", ParamKind.BOOL, False, "/loop/anim"),
    ParamSpec("loop_uses_time", ParamKind.BOOL, True, "/loop/timemode"),
    ParamSpec("loop_time", ParamKind.FLOAT, 4.0, "/loop/time", 0.1, 600.0),
    ParamSpec("loop_speed", ParamKind.FLOAT, 0.0, "/loop/speed", -5.0, 5.0),
    ParamSpec("loop_alpha", ParamKind.FLOAT, 0.0, "/loop/alpha", 0.0, 1.0),
    ParamSpec("loop_index", ParamKind.INT, 0, "/loop/index", 0, 2**31 - 1),
    ParamSpec("keyframe_count", ParamKind.INT, 6, "/loop/keyframes", 1, 256),
    ParamSpec("perfect_loop", ParamKind.BOOL, False, "/loop/perfect"),
    ParamSpec("noise_loop", ParamKind.BOOL, False, "/loop/noise"),
    ParamSpec("noise_radius", ParamKind.FLOAT, 1.0, "/loop/radius", 0.01, 100.0),
    ParamSpec("noise_loop_seed", ParamKind.INT, 0, "/loop/seed", 0, 2**31 - 1),
    # Loop pulse: one OSC message per loop-start or loop-complete event, sent
    # to a user configured address, separate from the control input port.
    ParamSpec("pulse_address", ParamKind.STR, "", "/loop/pulse/address"),
    # The IP and port are the machine's, not the look's: a preset saved on one
    # LAN would silently misdirect pulses on another, the way an absolute
    # model path once did. The address they carry names the message and stays
    # part of the look, so it keeps preset=True.
    ParamSpec("pulse_ip", ParamKind.STR, "127.0.0.1", "/loop/pulse/ip", preset=False),
    ParamSpec(
        "pulse_port", ParamKind.INT, 5005, "/loop/pulse/port", 1, 65535, preset=False
    ),
)

REGISTRY: dict[str, ParamSpec] = {spec.name: spec for spec in _SPECS}
BY_ADDRESS: dict[str, ParamSpec] = {spec.address: spec for spec in _SPECS}

# Structured control addresses. They carry Python objects rather than scalars,
# so they are reserved and never registry parameters.
BINDING_SET = "/binding/set"
BINDING_CLEAR = "/binding/clear"
VECTOR_SET = "/vector/set"
# The seed is applied in the control loop, not here: materializing a vector
# needs the loaded model's z_dim, which the registry and mapping layer do not
# know. This address is reserved so mapping still recognizes it.
VECTOR_RANDOMIZE = "/vector/randomize"
KEYFRAME_SET = "/keyframe/set"
KEYFRAME_REMOVE = "/keyframe/remove"


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
class Keyframe:
    """One stop in a keyframe loop.

    `kind` is `"seed"` (blend from `seed_x`/`seed_y` like the latent grid) or
    `"vec"` (use `vec` directly). `project` mirrors `latent_project`: applied
    per keyframe rather than once for the whole loop, so a loop can mix
    projected and raw stops. Validation is not this dataclass's job: it
    happens where a `Keyframe` is applied to state, not where one is built.
    """

    kind: str
    seed_x: float = 0.0
    seed_y: float = 0.0
    vec: tuple[float, ...] = ()
    project: bool = True


def default_keyframe(index: int) -> Keyframe:
    """The keyframe a resized or freshly opened loop fills a slot with."""
    return Keyframe("seed", float(index), 0.0)


# Six seed keyframes at (i, 0), matching the old app's default seeds list.
_DEFAULT_KEYFRAMES: tuple[Keyframe, ...] = tuple(default_keyframe(i) for i in range(6))


@dataclass(frozen=True)
class SetVector:
    """Replace the performer's latent vector wholesale.

    A dedicated value object, matching `Binding`/`ClearBinding`, though a raw
    sequence of numbers is also accepted on `/vector/set` for OSC parity.
    """

    values: tuple[float, ...]


@dataclass(frozen=True)
class SetKeyframe:
    """Replace or append the keyframe at `index` in the loop."""

    index: int
    keyframe: Keyframe


@dataclass(frozen=True)
class RemoveKeyframe:
    """Remove the keyframe at `index` from the loop."""

    index: int


@dataclass(frozen=True)
class ControlState:
    """The whole control surface, as one value.

    The defaults restate the registry's rather than deriving from it, so that
    the fields stay statically typed and readable. `test_params` asserts the
    two agree for every parameter, because a state that opened on a different
    value than the registry advertises would be a silent divergence in the one
    place nothing else checks.
    """

    pkl_path: str | None = None
    latent_x: float = 0.0
    latent_y: float = 0.0
    anim_playing: bool = False
    anim_speed_x: float = 0.25
    anim_speed_y: float = 0.0
    truncation_psi: float = 0.8
    global_noise: float = 1.0
    noise_enabled: bool = True
    noise_seed: int = 0
    noise_anim: bool = False
    fps_cap: int = 60
    vector_mode: bool = False
    latent_project: bool = True
    loop_active: bool = False
    loop_uses_time: bool = True
    loop_time: float = 4.0
    loop_speed: float = 0.0
    loop_alpha: float = 0.0
    loop_index: int = 0
    keyframe_count: int = 6
    perfect_loop: bool = False
    noise_loop: bool = False
    noise_radius: float = 1.0
    noise_loop_seed: int = 0
    pulse_address: str = ""
    pulse_ip: str = "127.0.0.1"
    pulse_port: int = 5005
    # Structured state, not a registry parameter: empty means unset, and the
    # generator derives a deterministic fallback from it (see design.md).
    latent_vec: tuple[float, ...] = ()
    # Structured state, not a registry parameter: the loop's stops. Six seed
    # keyframes by default, matching the old app.
    keyframes: tuple[Keyframe, ...] = _DEFAULT_KEYFRAMES
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
    latent_vec: tuple[float, ...]
    latent_project: bool
    keyframes: tuple[Keyframe, ...]
    loop_alpha: float
    loop_index: int
    # "seed", "vec" or "loop": what the generator evaluates this frame, see
    # `_derive_mode`.
    mode: str


def _derive_mode(state: ControlState) -> str:
    """The generator mode for one frame, per the design's mode table.

    Loop playback overrides latent navigation entirely: while `loop_active`,
    a noise loop still evaluates as "vec" (a vector fed straight to the
    mapping network), everything else that loops does so via keyframes.
    Outside a loop, `vector_mode` picks between the seed grid and a raw
    vector.
    """
    if state.loop_active:
        return "vec" if state.noise_loop else "loop"
    return "vec" if state.vector_mode else "seed"


def to_render_params(state: ControlState) -> RenderParams:
    keyframe_count = len(state.keyframes)
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
        latent_vec=state.latent_vec,
        latent_project=state.latent_project,
        keyframes=state.keyframes,
        loop_alpha=state.loop_alpha,
        # Clamped here, not in the spec: the bound is the keyframe count,
        # which is dynamic.
        loop_index=min(state.loop_index, max(keyframe_count - 1, 0)),
        mode=_derive_mode(state),
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
