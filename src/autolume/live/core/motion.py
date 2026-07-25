"""Wall-clock motion integration, run on the control tick.

One driver per parameter, and motion has the weakest claim on it. The hand
wins over everything, a binding wins over motion, and motion advances only
what nothing else is driving.

Touch grace exists so that a hand and an automated writer stop fighting over
the same value, and an integrator that ignores it defeats the feature for the
one parameter it was built to protect: with Animate on, dragging Latent x
would be futile, since the integrator keeps moving the store under the hold
and the widget snaps back on release. A binding plus Animate is the same
argument in slower motion, the binding setting an absolute value while the
integrator adds to it between source events, so the value drifts.

Every write goes through `apply_value`, so motion is coerced, clamped and
refused a non-finite value exactly like every other writer.

Plan 3 adds three more continuous writers: vector mode (below), keyframe
loops and simplex loops (Tasks 4 and 5). Each copies this function's
ownership template rather than inventing its own.
"""

import dataclasses

import numpy as np

from autolume.live.core.generator import ModelInfo
from autolume.live.core.params import ControlState, apply_value
from autolume.live.core.touch import TouchTracker

# The parameters motion drives, each with the speed that advances it.
_AXES = (("latent_x", "anim_speed_x"), ("latent_y", "anim_speed_y"))

# Not an axis: the vector walk drifts toward a target rather than advancing by
# a speed, so it is not in `_AXES`, but it is still a parameter motion can own.
_VECTOR_PARAM = "latent_vec"

MOTION_PARAMS = tuple(name for name, _ in _AXES) + (_VECTOR_PARAM,)

# Old-app calibration (widgets/latent_widget.py), kept for parity: how fast the
# walk closes on its target, and how close counts as "arrived".
_VECTOR_WALK_GAIN = 10.0
_VECTOR_RETARGET_THRESHOLD = 1.0


class WalkState:
    """The vector walk's target: the one bit of Plan 3 motion that is mutable.

    Everything else here is pure: `integrate` reads a `ControlState` and
    returns a new one. The walk's target has to survive between ticks without
    being part of the performance (design.md: runtime only, never persisted),
    so it lives outside `ControlState` the same way `TouchTracker` does,
    owned by the control loop and handed in rather than kept at module scope.
    """

    def __init__(self, rng: np.random.RandomState) -> None:
        self._rng = rng
        self.target: np.ndarray | None = None

    def retarget(self, dim: int) -> np.ndarray:
        self.target = self._rng.randn(dim)
        return self.target


def drives(
    state: ControlState,
    name: str,
    touch: TouchTracker | None = None,
    now: float = 0.0,
) -> bool:
    """Whether motion is what advances `name` right now.

    The integrator calls this on every axis it considers, so a UI asking the
    same question gets the same answer. A marker that claimed motion for a
    parameter the integrator leaves alone would be worse than no marker, and
    the only way to be sure of that is to share the rule rather than restate
    it.
    """
    if not state.anim_playing or name not in MOTION_PARAMS:
        return False
    # While a loop plays, it owns the latent, seed axes and vector alike; Task
    # 4 integrates the loop itself, but the ownership rule belongs here so the
    # marker and the integrator can never disagree about who is driving.
    if state.loop_active:
        return False
    # The vector walk is additionally gated on vector mode: off, it is not the
    # writer nothing else is driving, it is simply not in play this frame.
    if name == _VECTOR_PARAM and not state.vector_mode:
        return False
    # Only a row with a source of its own takes the parameter away from motion.
    # A row left sourceless is the parameter's own address, which writes when a
    # message happens to arrive rather than continuously, so it no more owns the
    # parameter than an unmapped one does.
    if any(
        binding.target == name and binding.enabled and binding.source
        for binding in state.bindings
    ):
        return False
    return touch is None or not touch.is_held(name, now)


def _walk_vector(
    state: ControlState, dt: float, walk: WalkState, model_info: ModelInfo | None
) -> ControlState:
    """Drift `latent_vec` toward a wandering target, retargeting up close.

    An empty vector is never invented here: the UI offers Randomize for that,
    and a vector whose length no longer matches the loaded model is treated
    the same way, since it is unset in every way that matters to this walk.
    """
    vec = state.latent_vec
    dim = len(vec)
    if dim == 0:
        return state
    if model_info is not None and dim != model_info.z_dim:
        return state
    current = np.asarray(vec, dtype=np.float64)
    target = walk.target
    if target is None or len(target) != dim:
        target = walk.retarget(dim)
    diff = target - current
    distance = float(np.linalg.norm(diff))
    if distance < _VECTOR_RETARGET_THRESHOLD:
        target = walk.retarget(dim)
        diff = target - current
        distance = float(np.linalg.norm(diff))
    if distance <= 0.0:
        return state
    # Speed is unsigned by definition: drift toward a target has no meaningful
    # reverse, unlike the signed seed walk above.
    step = diff / distance * abs(state.anim_speed_x) * _VECTOR_WALK_GAIN * dt
    moved = current + step
    if not np.all(np.isfinite(moved)):
        return state
    return dataclasses.replace(state, latent_vec=tuple(moved.tolist()))


def integrate(
    state: ControlState,
    dt: float,
    touch: TouchTracker | None = None,
    now: float = 0.0,
    model_info: ModelInfo | None = None,
    walk: WalkState | None = None,
) -> ControlState:
    if not state.anim_playing or dt <= 0.0:
        return state
    for name, speed_name in _AXES:
        if not drives(state, name, touch, now):
            continue
        step = getattr(state, name) + getattr(state, speed_name) * dt
        state = apply_value(state, name, step)
    if walk is not None and drives(state, _VECTOR_PARAM, touch, now):
        state = _walk_vector(state, dt, walk, model_info)
    return state
