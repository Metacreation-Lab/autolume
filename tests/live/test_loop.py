import dataclasses
import math

from autolume.live.core.loop import SPEED_ALPHA_PER_SECOND, LoopStep, advance
from autolume.live.core.params import ControlState, Keyframe

_SIX_KEYFRAMES = ControlState().keyframes
assert len(_SIX_KEYFRAMES) == 6


def test_inactive_loop_is_identity():
    state = ControlState(loop_active=False, loop_alpha=0.3, loop_index=2)
    step = advance(state, 0.5)
    assert step == LoopStep(alpha=0.3, index=2, wrapped=False, started=False)


def test_zero_or_negative_dt_is_identity():
    state = ControlState(loop_active=True, loop_speed=1.0, loop_uses_time=False)
    assert advance(state, 0.0).alpha == 0.0
    assert advance(state, -1.0).alpha == 0.0


def test_advance_never_reports_started():
    """`advance` has no visibility into the previous tick's `loop_active`.

    `ControlLoop` is the one that can compare this tick to the last, so it is
    the one that sets `started`; a pure phase step never does.
    """
    state = ControlState(loop_active=True, loop_speed=1.0, loop_uses_time=False)
    assert advance(state, 0.5).started is False


# --- time mode -------------------------------------------------------------


def test_time_mode_rate_is_segment_count_over_looptime():
    state = ControlState(loop_active=True, loop_uses_time=True, loop_time=4.0)
    step = advance(state, 0.1)
    expected = len(_SIX_KEYFRAMES) * 0.1 / 4.0
    assert abs(step.alpha - expected) < 1e-9
    assert step.index == 0
    assert step.wrapped is False


def test_time_mode_full_cycle_takes_loop_time_seconds():
    state = ControlState(loop_active=True, loop_uses_time=True, loop_time=4.0)
    total_wraps = 0
    for _ in range(8):  # 8 * 0.5 == loop_time, and 0.5 is exact in float
        step = advance(state, 0.5)
        total_wraps += step.wrapped
        state = dataclasses.replace(state, loop_alpha=step.alpha, loop_index=step.index)
    assert total_wraps == 1
    assert state.loop_index == 0
    assert state.loop_alpha == 0.0


def test_time_mode_respects_a_smaller_keyframe_count():
    state = ControlState(
        loop_active=True,
        loop_uses_time=True,
        loop_time=4.0,
        keyframes=(Keyframe("seed"), Keyframe("seed"), Keyframe("seed")),
    )
    step = advance(state, 0.1)
    assert abs(step.alpha - 3 * 0.1 / 4.0) < 1e-9


def test_zero_or_negative_looptime_is_identity():
    for bad in (0.0, -1.0, -600.0):
        state = ControlState(loop_active=True, loop_uses_time=True, loop_time=bad)
        assert advance(state, 0.1) == LoopStep(0.0, 0, False, False)


def test_non_finite_looptime_is_identity():
    for bad in (math.nan, math.inf, -math.inf):
        state = ControlState(loop_active=True, loop_uses_time=True, loop_time=bad)
        assert advance(state, 0.1) == LoopStep(0.0, 0, False, False)


# --- speed mode --------------------------------------------------------------


def test_speed_mode_rate_is_named_constant_times_speed_times_dt():
    state = ControlState(loop_active=True, loop_uses_time=False, loop_speed=2.0)
    step = advance(state, 0.1)
    assert abs(step.alpha - SPEED_ALPHA_PER_SECOND * 2.0 * 0.1) < 1e-9
    assert step.index == 0


def test_negative_speed_walks_the_index_backwards_and_wraps_to_the_last_segment():
    state = ControlState(
        loop_active=True,
        loop_uses_time=False,
        loop_speed=-1.0,
        loop_alpha=0.05,
        loop_index=0,
    )
    step = advance(state, 0.1)
    assert step.index == len(_SIX_KEYFRAMES) - 1
    assert step.wrapped is True


def test_non_finite_speed_is_identity():
    for bad in (math.nan, math.inf, -math.inf):
        state = ControlState(
            loop_active=True, loop_uses_time=False, loop_speed=bad, loop_alpha=0.5
        )
        assert advance(state, 0.1) == LoopStep(0.5, 0, False, False)


def test_zero_speed_is_a_no_op_even_at_the_upper_alpha_bound():
    # Regression guard: alpha sitting exactly at its clamp of 1.0 must not
    # register a spurious wrap just because a zero-rate step touched it.
    state = ControlState(
        loop_active=True, loop_uses_time=False, loop_speed=0.0, loop_alpha=1.0
    )
    step = advance(state, 0.1)
    assert step.alpha == 1.0
    assert step.wrapped is False


# --- multi-segment jump ------------------------------------------------------


def test_a_large_dt_lands_on_the_right_alpha_and_index_and_reports_wrapped():
    state = ControlState(loop_active=True, loop_uses_time=True, loop_time=4.0)
    step = advance(state, 6.0)  # 6 * 6.0 / 4.0 == 9.0 alpha units, 6 segments
    assert step.alpha == 0.0
    assert step.index == 3
    assert step.wrapped is True


def test_a_large_reverse_dt_lands_on_the_right_index_after_several_wraps():
    state = ControlState(
        loop_active=True, loop_uses_time=False, loop_speed=-1.0, loop_index=2
    )
    # rate = 0.6 * -1 * 17 == -10.2, matching the hand-traced example: 11
    # backward segment crossings from index 2 land on index 3, wrapped twice.
    step = advance(state, 17.0)
    assert step.index == 3
    assert step.wrapped is True


# --- noise loop ---------------------------------------------------------------


def test_noise_loop_pins_index_at_zero_and_wraps_alpha():
    state = ControlState(
        loop_active=True,
        noise_loop=True,
        loop_uses_time=True,
        loop_time=4.0,
        loop_index=3,  # stale, must not leak through
    )
    step = advance(state, 4.0)  # exactly one full cycle at rate 1 * dt / looptime
    assert step.index == 0
    assert step.alpha == 0.0
    assert step.wrapped is True


def test_noise_loop_does_not_wrap_mid_cycle():
    state = ControlState(
        loop_active=True, noise_loop=True, loop_uses_time=True, loop_time=4.0
    )
    step = advance(state, 1.0)
    assert step.index == 0
    assert abs(step.alpha - 0.25) < 1e-9
    assert step.wrapped is False
