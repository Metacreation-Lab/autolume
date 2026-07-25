from autolume.live.core.motion import integrate
from autolume.live.core.params import Binding, ControlState
from autolume.live.core.touch import TOUCH_GRACE, TouchTracker


def test_not_playing_is_identity():
    state = ControlState(latent_x=1.0)
    assert integrate(state, 0.5) == state


def test_playing_advances_by_speed_times_dt():
    state = ControlState(anim_playing=True, anim_speed_x=2.0, anim_speed_y=-1.0)
    out = integrate(state, 0.5)
    assert out.latent_x == 1.0
    assert out.latent_y == -0.5


def test_zero_or_negative_dt_is_identity():
    state = ControlState(anim_playing=True, anim_speed_x=2.0)
    assert integrate(state, 0.0) == state
    assert integrate(state, -1.0) == state


def test_deterministic_over_split_intervals():
    state = ControlState(anim_playing=True, anim_speed_x=1.0)
    whole = integrate(state, 1.0)
    halves = integrate(integrate(state, 0.5), 0.5)
    assert abs(whole.latent_x - halves.latent_x) < 1e-9


def test_a_held_axis_does_not_move_and_resumes_after_the_grace():
    state = ControlState(anim_playing=True, anim_speed_x=2.0, anim_speed_y=2.0)
    touch = TouchTracker()
    touch.begin("latent_x", 10.0)

    held = integrate(state, 0.5, touch, 10.0)
    assert held.latent_x == 0.0
    assert held.latent_y == 1.0

    touch.end("latent_x", 10.0)
    in_grace = integrate(state, 0.5, touch, 10.0 + TOUCH_GRACE / 2.0)
    assert in_grace.latent_x == 0.0

    resumed = integrate(state, 0.5, touch, 10.0 + TOUCH_GRACE)
    assert resumed.latent_x == 1.0


def test_an_enabled_binding_takes_the_axis_away_from_motion():
    state = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        anim_speed_y=2.0,
        bindings=(Binding("latent_x", "/audio/level"),),
    )
    out = integrate(state, 0.5)
    assert out.latent_x == 0.0
    assert out.latent_y == 1.0


def test_a_disabled_binding_leaves_the_axis_to_motion():
    state = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        bindings=(Binding("latent_x", "/audio/level", enabled=False),),
    )
    assert integrate(state, 0.5).latent_x == 1.0


def test_clearing_a_binding_hands_the_axis_back_to_motion():
    bound = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        bindings=(Binding("latent_x", "/audio/level"),),
    )
    assert integrate(bound, 0.5).latent_x == 0.0
    cleared = ControlState(anim_playing=True, anim_speed_x=2.0)
    assert integrate(cleared, 0.5).latent_x == 1.0


def test_a_binding_on_one_axis_does_not_stop_the_other():
    state = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        anim_speed_y=2.0,
        bindings=(Binding("latent_y", "/audio/level"),),
    )
    out = integrate(state, 0.5)
    assert out.latent_x == 1.0
    assert out.latent_y == 0.0


def test_a_non_finite_step_is_refused_like_any_other_write():
    state = ControlState(anim_playing=True, latent_x=1e308, anim_speed_x=1e308)
    assert integrate(state, 1.0).latent_x == 1e308
