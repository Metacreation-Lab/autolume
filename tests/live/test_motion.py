from autolume.live.core.motion import integrate
from autolume.live.core.params import ControlState


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
