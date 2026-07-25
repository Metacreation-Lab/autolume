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


def integrate(
    state: ControlState,
    dt: float,
    touch: TouchTracker | None = None,
    now: float = 0.0,
) -> ControlState:
    if not state.anim_playing or dt <= 0.0:
        return state
    bound = {binding.target for binding in state.bindings if binding.enabled}
    for name, speed_name in _AXES:
        if name in bound:
            continue
        if touch is not None and touch.is_held(name, now):
            continue
        step = getattr(state, name) + getattr(state, speed_name) * dt
        state = apply_value(state, name, step)
    return state
