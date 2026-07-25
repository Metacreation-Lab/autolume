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

Plan 3 adds three more continuous writers (vector mode, keyframe loops,
simplex loops) and each will copy this function, so the rule belongs in the
template while there is still only one writer to fix.
"""

from autolume.live.core.params import ControlState, apply_value
from autolume.live.core.touch import TouchTracker

# The parameters motion drives, each with the speed that advances it.
_AXES = (("latent_x", "anim_speed_x"), ("latent_y", "anim_speed_y"))

MOTION_PARAMS = tuple(name for name, _ in _AXES)


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
    if any(binding.target == name and binding.enabled for binding in state.bindings):
        return False
    return touch is None or not touch.is_held(name, now)


def integrate(
    state: ControlState,
    dt: float,
    touch: TouchTracker | None = None,
    now: float = 0.0,
) -> ControlState:
    if not state.anim_playing or dt <= 0.0:
        return state
    for name, speed_name in _AXES:
        if not drives(state, name, touch, now):
            continue
        step = getattr(state, name) + getattr(state, speed_name) * dt
        state = apply_value(state, name, step)
    return state
