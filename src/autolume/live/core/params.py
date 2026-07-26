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
    # No keyframe_count: the list's length is derived from len(keyframes)
    # everywhere it is needed (to_render_params below, _wrap_loop_index in
    # mapping.py), rather than tracked as a second, separately writable
    # number that could disagree with the tuple's own length. It used to be
    # exactly that: a registry parameter a preset or a controller could set
    # to a value the keyframes tuple never matched. Removed rather than kept
    # read-only, since OSC has no legitimate reason to resize the list
    # (Add and per-row Remove are the only ways it changes, both already
    # structured edits through KEYFRAME_SET/KEYFRAME_REMOVE).
    ParamSpec("perfect_loop", ParamKind.BOOL, False, "/loop/perfect"),
    ParamSpec("noise_loop", ParamKind.BOOL, False, "/loop/noise"),
    # 10.0, not the table builder's own 100.0 ceiling: a table build is close
    # to linear in the step count (noiseloop.py), and at 100 it takes ~35s
    # against ~4s at 10, which read as the UI hanging. A preset carrying a
    # value above this clamps down to it, same as any other bound.
    ParamSpec("noise_radius", ParamKind.FLOAT, 1.0, "/loop/radius", 0.01, 10.0),
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
    # Image derivation. Applied render-side in the uint8 conversion path.
    ParamSpec("grayscale", ParamKind.BOOL, False, "/image/grayscale"),
    ParamSpec("img_scale_db", ParamKind.FLOAT, 0.0, "/image/contrast", -40.0, 40.0),
    ParamSpec("img_normalize", ParamKind.BOOL, False, "/image/normalize"),
    ParamSpec("base_channel", ParamKind.INT, 0, "/image/channel", 0, 8192),
    ParamSpec("capture_layer", ParamKind.STR, "", "/image/layer"),
    # Adjuster weights. Eight fixed slots, not dynamic: `direction = sum(w_i *
    # dir_i)`, computed in the generator from the directions in ControlState
    # and these weights, never here.
    ParamSpec("adjust_w1", ParamKind.FLOAT, 0.0, "/adjust/1", -5.0, 5.0),
    ParamSpec("adjust_w2", ParamKind.FLOAT, 0.0, "/adjust/2", -5.0, 5.0),
    ParamSpec("adjust_w3", ParamKind.FLOAT, 0.0, "/adjust/3", -5.0, 5.0),
    ParamSpec("adjust_w4", ParamKind.FLOAT, 0.0, "/adjust/4", -5.0, 5.0),
    ParamSpec("adjust_w5", ParamKind.FLOAT, 0.0, "/adjust/5", -5.0, 5.0),
    ParamSpec("adjust_w6", ParamKind.FLOAT, 0.0, "/adjust/6", -5.0, 5.0),
    ParamSpec("adjust_w7", ParamKind.FLOAT, 0.0, "/adjust/7", -5.0, 5.0),
    ParamSpec("adjust_w8", ParamKind.FLOAT, 0.0, "/adjust/8", -5.0, 5.0),
    # Persisted separately as the preset's `model2` key, not as a plain param,
    # the same as pkl_path above: it needs path resolution a scalar value
    # cannot express. See presets.py (Task 11).
    ParamSpec("pkl2", ParamKind.STR, None, "/mix/model", preset=False),
    ParamSpec("mixing_enabled", ParamKind.BOOL, False, "/mix/enabled"),
    # Machine settings below, not persisted: each describes the hardware or
    # network a performance runs on, not the look itself. A preset saved on
    # one machine must not silently reconfigure another's device, port, or
    # output surface.
    ParamSpec("use_superres", ParamKind.BOOL, False, "/render/superres", preset=False),
    ParamSpec("device", ParamKind.STR, "auto", "/render/device", preset=False),
    ParamSpec("force_fp32", ParamKind.BOOL, False, "/render/fp32", preset=False),
    ParamSpec("osc_port", ParamKind.INT, 1338, "/osc/port", 1, 65535, preset=False),
    ParamSpec("ndi_enabled", ParamKind.BOOL, False, "/ndi/enabled", preset=False),
    ParamSpec(
        "ndi_name", ParamKind.STR, "Autolume Live", "/ndi/name", preset=False
    ),
    ParamSpec("recording", ParamKind.BOOL, False, "/record", preset=False),
    ParamSpec("fullscreen", ParamKind.BOOL, False, "/output/fullscreen", preset=False),
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
BEND_SET = "/bend/set"
BEND_REMOVE = "/bend/remove"
BEND_NOISE = "/bend/noise"
BEND_RATIO = "/bend/ratio"
ADJUST_DIRECTIONS = "/adjust/directions"
MIX_LAYERS = "/mix/layers"


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
class Transform:
    """One bending operation in the chain, applied in order.

    Validation is not this dataclass's job, same as `Keyframe`: it happens
    where a `Transform` is applied to state (`mapping.py`), not where one is
    built.
    """

    op: str
    layer: str
    params: tuple[float, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class SetTransform:
    """Replace or append the transform at `index` in the bending chain."""

    index: int
    transform: Transform


@dataclass(frozen=True)
class RemoveTransform:
    """Remove the transform at `index` from the bending chain."""

    index: int


@dataclass(frozen=True)
class SetLayerNoise:
    """Set one layer's per-layer noise strength.

    A strength of 0 is neutral and is stored as absence, not as a zero entry.
    """

    layer: str
    strength: float


@dataclass(frozen=True)
class SetLayerRatio:
    """Set one layer's x/y noise ratio.

    `(1, 1)` is neutral and is stored as absence, not as a (1, 1) entry.
    """

    layer: str
    rx: float
    ry: float


@dataclass(frozen=True)
class SetDirections:
    """Replace the adjuster's direction vectors wholesale, up to eight.

    The eight weights that scale these directions are ordinary registry
    parameters (`adjust_w1`...`adjust_w8`), not part of this value object.
    """

    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class SetCombinedLayers:
    """Replace the mixing origin, `"A"`/`"B"`/`"X"`, for every mixed layer."""

    entries: tuple[str, ...]


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
    perfect_loop: bool = False
    noise_loop: bool = False
    noise_radius: float = 1.0
    noise_loop_seed: int = 0
    pulse_address: str = ""
    pulse_ip: str = "127.0.0.1"
    pulse_port: int = 5005
    grayscale: bool = False
    img_scale_db: float = 0.0
    img_normalize: bool = False
    base_channel: int = 0
    capture_layer: str = ""
    adjust_w1: float = 0.0
    adjust_w2: float = 0.0
    adjust_w3: float = 0.0
    adjust_w4: float = 0.0
    adjust_w5: float = 0.0
    adjust_w6: float = 0.0
    adjust_w7: float = 0.0
    adjust_w8: float = 0.0
    pkl2: str | None = None
    mixing_enabled: bool = False
    use_superres: bool = False
    device: str = "auto"
    force_fp32: bool = False
    osc_port: int = 1338
    ndi_enabled: bool = False
    ndi_name: str = "Autolume Live"
    recording: bool = False
    fullscreen: bool = False
    # Structured state, not a registry parameter: empty means unset, and the
    # generator derives a deterministic fallback from it (see design.md).
    latent_vec: tuple[float, ...] = ()
    # Structured state, not a registry parameter: the loop's stops. Six seed
    # keyframes by default, matching the old app.
    keyframes: tuple[Keyframe, ...] = _DEFAULT_KEYFRAMES
    bindings: tuple[Binding, ...] = ()
    # Structured state, not a registry parameter: the bending chain, applied
    # in order. Sparse, unlike keyframes: an empty tuple is the common case.
    transforms: tuple[Transform, ...] = ()
    # Structured state, not a registry parameter: sparse (layer, strength) and
    # (layer, rx, ry) rows. Only non-neutral entries are ever stored.
    layer_noise: tuple[tuple[str, float], ...] = ()
    layer_ratios: tuple[tuple[str, float, float], ...] = ()
    # Structured state, not a registry parameter: up to eight direction
    # vectors, scaled by the adjust_w1...adjust_w8 registry weights.
    directions: tuple[tuple[float, ...], ...] = ()
    # Structured state, not a registry parameter: one origin per mixed layer.
    combined_layers: tuple[str, ...] = ()


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
    # `derive_mode`.
    mode: str
    # Bending chain, applied in order.
    transforms: tuple[Transform, ...]
    # Per-layer noise/ratio overrides, as mappings for O(1) lookup by layer
    # name. Sparse: a layer absent from either mapping is neutral.
    layer_noise: dict[str, float]
    layer_ratios: dict[str, tuple[float, float]]
    # Adjuster: raw directions and the eight weights, not their product. The
    # generator computes `direction = sum(w_i * dir_i)` from these each frame,
    # so the control tick never has to.
    directions: tuple[tuple[float, ...], ...]
    adjust_w1: float
    adjust_w2: float
    adjust_w3: float
    adjust_w4: float
    adjust_w5: float
    adjust_w6: float
    adjust_w7: float
    adjust_w8: float
    # Image derivation, applied render-side in the uint8 conversion path.
    grayscale: bool
    img_scale_db: float
    img_normalize: bool
    base_channel: int
    capture_layer: str
    # Mixing.
    pkl2: str | None
    mixing_enabled: bool
    combined_layers: tuple[str, ...]
    # Render-side machine settings that the frame itself depends on: super-res
    # is applied to the float image before uint8 conversion (Task 6),
    # force_fp32 flows into the synthesis call (Task 9). Every other
    # machine-level registry row (device, ports, NDI, recording, fullscreen)
    # configures a sink or the window, never the frame, and stays off this
    # snapshot.
    use_superres: bool
    force_fp32: bool


def derive_mode(state: ControlState) -> str:
    """The generator mode for one frame, per the design's mode table.

    Loop playback overrides latent navigation entirely: while `loop_active`,
    a noise loop still evaluates as "vec" (a vector fed straight to the
    mapping network), everything else that loops does so via keyframes.
    Outside a loop, `vector_mode` picks between the seed grid and a raw
    vector.

    Public rather than private: `perform.py` calls this directly to grey
    `latent_project`, which `generator.py`'s `render_frame` reads only in
    the `"vec"` branch. The UI must ask the same question the generator
    dispatches on rather than re-deriving it, or the two can drift; this
    function is what makes that structurally impossible instead of merely
    intended.
    """
    if state.loop_active:
        return "vec" if state.noise_loop else "loop"
    return "vec" if state.vector_mode else "seed"


def _ratio_pairs(layer_ratios):
    """The `(rx, ry)` pairs out of either shape `layer_ratios` is held in.

    `ControlState` keeps sparse `(layer, rx, ry)` rows and `RenderParams` a
    mapping, and the predicate below has to answer both callers identically.
    """
    if isinstance(layer_ratios, dict):
        yield from layer_ratios.values()
        return
    for _layer, rx, ry in layer_ratios:
        yield rx, ry


def ratio_forces_const_noise(state) -> bool:
    """Whether a layer ratio is holding the noise mode on "const".

    A ratio away from neutral resizes the activation, and the synthesis
    layer's `random` branch draws its noise field at the layer's nominal
    resolution regardless, so the two do not match and every frame raises.
    Its `const` branch resizes the noise field along with the activation, so
    const is the only mode a ratio renders in at all, and `noise_mode`
    substitutes it whenever this holds.

    Takes a `ControlState` or a `RenderParams`, because both callers must get
    the same answer: `noise_mode` decides what synthesis is asked for, and the
    Bending panel decides whether to say so. A neutral pair stored explicitly
    does not count, since the sparse rows can carry one.
    """
    if not state.noise_enabled:
        return False
    if not (state.noise_anim or state.noise_seed != 0):
        return False
    return any(
        rx != 1.0 or ry != 1.0 for rx, ry in _ratio_pairs(state.layer_ratios)
    )


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
        # Wrapped here, not in the spec, and by the same modulo `loop.advance`
        # steps the index with: the bound is the keyframe count, which is
        # dynamic, and a `min` clamp used to disagree with `advance`'s `%` on
        # an out-of-range index (index 20 of 6 rendered keyframe 5 while the
        # integrator landed on keyframe 2). `mapping.py` normalises writes the
        # same way, so this rarely has anything to do; it stays because a
        # `ControlState` can still be built directly with a stale index.
        loop_index=state.loop_index % keyframe_count if keyframe_count else 0,
        mode=derive_mode(state),
        transforms=state.transforms,
        layer_noise=dict(state.layer_noise),
        layer_ratios={layer: (rx, ry) for layer, rx, ry in state.layer_ratios},
        directions=state.directions,
        adjust_w1=state.adjust_w1,
        adjust_w2=state.adjust_w2,
        adjust_w3=state.adjust_w3,
        adjust_w4=state.adjust_w4,
        adjust_w5=state.adjust_w5,
        adjust_w6=state.adjust_w6,
        adjust_w7=state.adjust_w7,
        adjust_w8=state.adjust_w8,
        grayscale=state.grayscale,
        img_scale_db=state.img_scale_db,
        img_normalize=state.img_normalize,
        base_channel=state.base_channel,
        capture_layer=state.capture_layer,
        pkl2=state.pkl2,
        mixing_enabled=state.mixing_enabled,
        combined_layers=state.combined_layers,
        use_superres=state.use_superres,
        force_fp32=state.force_fp32,
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
