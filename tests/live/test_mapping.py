from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.params import ControlState


def test_float_event_applies():
    state = apply_event(ControlState(), ControlEvent("/latent/x", 4.2))
    assert state.latent_x == 4.2


def test_int_coercion_and_clamp():
    state = apply_event(ControlState(), ControlEvent("/render/fps", 999.9))
    assert state.fps_cap == 240
    state = apply_event(state, ControlEvent("/render/fps", -5))
    assert state.fps_cap == 0


def test_bool_coercion_from_osc_float():
    state = apply_event(ControlState(), ControlEvent("/anim/playing", 1.0))
    assert state.anim_playing is True
    state = apply_event(state, ControlEvent("/anim/playing", 0.0))
    assert state.anim_playing is False


def test_str_param_applies():
    state = apply_event(ControlState(), ControlEvent("/model/path", "/tmp/m.pkl"))
    assert state.pkl_path == "/tmp/m.pkl"


def test_float_clamped_to_bounds():
    state = apply_event(ControlState(), ControlEvent("/trunc/psi", 5.0))
    assert state.truncation_psi == 2.0


def test_unknown_address_ignored():
    before = ControlState()
    after = apply_event(before, ControlEvent("/nope", 1.0))
    assert after == before


def test_uncoercible_value_ignored():
    before = ControlState()
    after = apply_event(before, ControlEvent("/latent/x", "not a number"))
    assert after == before
