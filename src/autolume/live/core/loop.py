"""Pure keyframe and noise loop phase math, run on the control tick.

`advance` mirrors `motion.integrate`'s contract: no clock read, no RNG, no
mutation, so `test_loop.py` drives it with synthetic `dt` alone. Unlike
`integrate`, it has no notion of the tick before this one, so it cannot tell
a fresh play from an ongoing one; `LoopStep.started` is always False out of
this function. Detecting that edge needs the previous tick's `loop_active`,
which only `ControlLoop` keeps (control.py), so that is where it gets set
before Task 7 reads it for the outbound pulse.

Time and speed mode share one model: a rate in alpha units per second, walked
across `alpha` with a `divmod`, not a per-segment loop. A tick's `dt` can be
arbitrarily large (a stalled thread, a resumed-from-sleep process), and stepping
one segment at a time for that gap would burn the tick budget it is meant to
protect; `divmod` lands on the same (alpha, index) in one shot.
"""

import math
from dataclasses import dataclass

from autolume.live.core.params import ControlState

# Old-app calibration (widgets/looping_widget.py): `step = 0.01 * speed` per UI
# frame at the old app's 60 fps default is 0.6 alpha per second, so speed 1
# here matches speed 1 there.
SPEED_ALPHA_PER_SECOND = 0.6


@dataclass(frozen=True)
class LoopStep:
    alpha: float
    index: int
    wrapped: bool  # a full cycle completed this step
    started: bool  # playback began this step (always False here, see above)


def _segment_count(state: ControlState) -> int:
    """How many stops make one cycle: one for the noise loop, one per keyframe.

    `len(state.keyframes)` rather than the `keyframe_count` registry field,
    matching `to_render_params`: the tuple is the data that actually drives
    playback, and the two are kept in sync by `mapping.py` in any case.
    """
    if state.noise_loop:
        return 1
    return max(len(state.keyframes), 1)


def _alpha_rate(state: ControlState, dt: float, segments: int) -> float | None:
    """Alpha units per second this tick, or None if nothing may drive it.

    A non-finite or non-positive `loop_time`, or a non-finite `loop_speed`,
    has no meaningful rate; refusing it here is what keeps a bad preset value
    from ever reaching the phase math below instead of producing a NaN or an
    infinite spin.
    """
    if state.loop_uses_time:
        loop_time = state.loop_time
        if not math.isfinite(loop_time) or loop_time <= 0.0:
            return None
        return segments * dt / loop_time
    speed = state.loop_speed
    if not math.isfinite(speed):
        return None
    return SPEED_ALPHA_PER_SECOND * speed * dt


def advance(state: ControlState, dt: float) -> LoopStep:
    """One control tick of keyframe or noise loop phase math.

    Signed: a negative rate (reverse speed) walks alpha below 0, decrementing
    the index and wrapping to the last segment, mirroring the forward case.
    The noise loop is a single segment, so every wrap of its one segment is a
    full cycle and `index` never leaves 0.
    """
    alpha = state.loop_alpha
    index = 0 if state.noise_loop else state.loop_index
    identity = LoopStep(alpha=alpha, index=index, wrapped=False, started=False)
    if not state.loop_active or not math.isfinite(dt) or dt <= 0.0:
        return identity
    segments = _segment_count(state)
    rate = _alpha_rate(state, dt, segments)
    if not rate:  # None (refused above) or exactly 0.0: nothing to step
        return identity
    total = alpha + rate
    if not math.isfinite(total):
        return identity
    crossings, frac = divmod(total, 1.0)
    raw_index = index + int(crossings)
    wrapped = raw_index < 0 or raw_index >= segments
    new_index = 0 if state.noise_loop else raw_index % segments
    return LoopStep(alpha=frac, index=new_index, wrapped=wrapped, started=False)
