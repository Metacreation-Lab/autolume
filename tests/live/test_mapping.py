import logging

import pytest

from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    KEYFRAME_REMOVE,
    KEYFRAME_SET,
    VECTOR_RANDOMIZE,
    VECTOR_SET,
    Binding,
    ClearBinding,
    ControlState,
    Keyframe,
    RemoveKeyframe,
    SetKeyframe,
    SetVector,
    default_keyframe,
)
from autolume.live.core.presets import FORMAT, PRESET_APPLY, VERSION

MAPPING_LOGGER = "autolume.live.core.mapping"


def warnings_from(caplog, logger_name):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelname == "WARNING"
    ]


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


@pytest.mark.parametrize(
    "address,value",
    [
        ("/noise/global", float("nan")),
        ("/noise/global", float("inf")),
        ("/trunc/psi", float("-inf")),
        ("/noise/seed", float("inf")),
        ("/render/fps", float("nan")),
    ],
)
def test_non_finite_wire_value_ignored(address, value):
    # A float NaN or infinity is legal on the OSC wire, so it reaches here.
    before = ControlState(global_noise=0.5, truncation_psi=0.5, noise_seed=7)
    assert apply_event(before, ControlEvent(address, value)) == before


def test_uncoercible_value_warning_names_the_wire_address(caplog):
    with caplog.at_level(logging.WARNING):
        apply_event(ControlState(), ControlEvent("/latent/x", "not a number"))
    messages = warnings_from(caplog, "autolume.live.core.params")
    assert messages
    assert any("/latent/x" in message for message in messages)


def set_binding(state, binding):
    return apply_event(state, ControlEvent(BINDING_SET, binding))


def clear_binding(state, target):
    return apply_event(state, ControlEvent(BINDING_CLEAR, ClearBinding(target)))


def test_binding_set_appends_then_replaces_in_place():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    state = set_binding(state, Binding("truncation_psi", "/audio/bass", "x*2"))
    assert [b.target for b in state.bindings] == ["latent_x", "truncation_psi"]

    state = set_binding(state, Binding("latent_x", "/ctl/1", "x+1"))
    assert [b.target for b in state.bindings] == ["latent_x", "truncation_psi"]
    assert state.bindings[0].source == "/ctl/1"
    assert state.bindings[0].expression == "x+1"


def test_binding_set_with_non_binding_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BINDING_SET, 1.0))
    assert after == before
    assert any("non binding value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_binding_set_with_unknown_target_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding("nope", "/audio/level"))
    assert after == before
    assert any("unknown parameter" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("expression", [None, 3.0, b"x", ["x"]])
def test_binding_set_with_non_str_expression_ignored(caplog, expression):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding("latent_x", "/audio/level", expression))
    assert after == before
    assert any("malformed binding" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("target", [["latent_x"], {"latent_x"}, None, 1.0])
def test_binding_set_with_non_str_target_ignored(caplog, target):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding(target, "/audio/level"))
    assert after == before
    assert any("malformed binding" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_binding_set_with_non_str_source_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding("latent_x", None))
    assert after == before
    assert any("malformed binding" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_binding_set_with_bad_expression_stores_error():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level", "x +* 2"))
    binding = state.bindings[0]
    assert binding.expression == "x +* 2"
    assert binding.error is not None


def test_binding_set_with_valid_expression_clears_error():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level", "nope(x)"))
    assert state.bindings[0].error is not None
    state = set_binding(state, Binding("latent_x", "/audio/level", "x*2", error="old"))
    assert state.bindings[0].error is None


def test_binding_clear_removes_only_that_target():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    state = set_binding(state, Binding("truncation_psi", "/audio/bass"))
    state = clear_binding(state, "latent_x")
    assert [b.target for b in state.bindings] == ["truncation_psi"]


def test_binding_clear_unbound_target_is_noop():
    before = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    assert clear_binding(before, "truncation_psi") == before


@pytest.mark.parametrize("value", [1.0, "latent_x", None, ["latent_x"]])
def test_binding_clear_with_non_clear_binding_value_ignored(caplog, value):
    before = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BINDING_CLEAR, value))
    assert after == before
    assert any(
        "non clear binding value" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


@pytest.mark.parametrize("target", [None, 1.0, ["latent_x"]])
def test_binding_clear_with_non_str_target_ignored(caplog, target):
    before = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BINDING_CLEAR, ClearBinding(target)))
    assert after == before
    assert any(
        "malformed clear binding" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def preset_payload(params=None, bindings=None):
    return {
        "format": FORMAT,
        "version": VERSION,
        "params": params if params is not None else {},
        "bindings": bindings if bindings is not None else [],
    }


def apply_preset(state, payload):
    return apply_event(state, ControlEvent(PRESET_APPLY, payload))


def test_preset_apply_sets_params_and_bindings_in_one_event():
    payload = preset_payload(
        {"truncation_psi": 1.4, "global_noise": 0.3},
        [{"target": "latent_x", "source": "/audio/level", "expression": "x*2"}],
    )
    state = apply_preset(ControlState(), payload)
    assert (state.truncation_psi, state.global_noise) == (1.4, 0.3)
    assert state.bindings == (Binding("latent_x", "/audio/level", "x*2"),)


def test_preset_apply_replaces_the_whole_binding_set():
    before = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    after = apply_preset(before, preset_payload({"latent_y": 2.0}))
    assert after.bindings == ()
    assert after.latent_y == 2.0


def test_preset_apply_clamps_out_of_range_values():
    state = apply_preset(ControlState(), preset_payload({"truncation_psi": 99.0}))
    assert state.truncation_psi == 2.0


@pytest.mark.parametrize(
    "value", [1.0, "look", None, ["look"], Binding("latent_x", "/x")]
)
def test_preset_apply_with_non_dict_value_ignored(caplog, value):
    before = ControlState(truncation_psi=1.1)
    with caplog.at_level(logging.WARNING):
        after = apply_preset(before, value)
    assert after == before
    assert any("non preset value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_preset_apply_with_wrong_format_ignored(caplog):
    before = ControlState(truncation_psi=1.1)
    payload = preset_payload({"truncation_psi": 0.2})
    payload["format"] = "some-other-app"
    with caplog.at_level(logging.WARNING):
        after = apply_preset(before, payload)
    assert after == before
    assert any("malformed preset" in m for m in warnings_from(caplog, MAPPING_LOGGER))


# --- /vector/set --------------------------------------------------------


def test_vector_set_from_set_vector_object():
    state = apply_event(ControlState(), ControlEvent(VECTOR_SET, SetVector((1.0, 2.0))))
    assert state.latent_vec == (1.0, 2.0)


def test_vector_set_from_raw_list_osc_parity():
    state = apply_event(ControlState(), ControlEvent(VECTOR_SET, [1.0, 2.0, 3.0]))
    assert state.latent_vec == (1.0, 2.0, 3.0)


def test_vector_set_coerces_ints_to_floats():
    state = apply_event(ControlState(), ControlEvent(VECTOR_SET, [1, 2, 3]))
    assert state.latent_vec == (1.0, 2.0, 3.0)
    assert all(isinstance(v, float) for v in state.latent_vec)


def test_vector_set_empty_clears_the_vector():
    before = apply_event(ControlState(), ControlEvent(VECTOR_SET, [1.0, 2.0]))
    after = apply_event(before, ControlEvent(VECTOR_SET, []))
    assert after.latent_vec == ()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_vector_set_rejects_non_finite_entry_wholesale(caplog, bad):
    before = ControlState(latent_vec=(9.0,))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(VECTOR_SET, [1.0, bad, 2.0]))
    assert after == before
    assert any(
        "non finite" in m or "non numeric" in m
        for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_vector_set_rejects_non_numeric_entry(caplog):
    before = ControlState(latent_vec=(9.0,))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(VECTOR_SET, [1.0, "nope"]))
    assert after == before
    assert warnings_from(caplog, MAPPING_LOGGER)


def test_vector_set_with_non_sequence_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(VECTOR_SET, "nope"))
    assert after == before
    assert any("non vector value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


# --- /vector/randomize ----------------------------------------------------


def test_vector_randomize_is_recognized_and_left_to_the_control_loop(caplog):
    before = ControlState()
    with caplog.at_level(logging.DEBUG):
        after = apply_event(before, ControlEvent(VECTOR_RANDOMIZE, 42))
    assert after == before
    assert not any(
        "unknown address" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )
    assert not [
        r for r in caplog.records if r.name == MAPPING_LOGGER and "unknown" in r.getMessage()
    ]


# --- /keyframe/set --------------------------------------------------------


def set_keyframe(state, index, keyframe):
    return apply_event(state, ControlEvent(KEYFRAME_SET, SetKeyframe(index, keyframe)))


def test_keyframe_set_replaces_at_index():
    state = set_keyframe(ControlState(), 0, Keyframe("seed", 9.0, 9.0))
    assert state.keyframes[0] == Keyframe("seed", 9.0, 9.0)
    assert len(state.keyframes) == 6
    assert state.keyframe_count == 6


def test_keyframe_set_at_len_appends():
    before = ControlState()
    keyframe = Keyframe("vec", vec=(1.0, 2.0))
    state = set_keyframe(before, len(before.keyframes), keyframe)
    assert len(state.keyframes) == 7
    assert state.keyframes[-1] == keyframe
    assert state.keyframe_count == 7


def test_keyframe_set_out_of_range_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_keyframe(before, 99, Keyframe("seed", 0.0, 0.0))
    assert after == before
    assert any("out of range" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_keyframe_set_rejects_non_finite_vec(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_keyframe(before, 0, Keyframe("vec", vec=(1.0, float("nan"))))
    assert after == before
    assert warnings_from(caplog, MAPPING_LOGGER)


def test_keyframe_set_with_non_set_keyframe_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(KEYFRAME_SET, (0, "seed", 1.0, 0.0)))
    assert after == before
    assert any("non keyframe value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_keyframe_set_with_osc_shaped_dict_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(
            before, ControlEvent(KEYFRAME_SET, {"index": 0, "kind": "seed"})
        )
    assert after == before
    assert any("non keyframe value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_keyframe_set_with_non_keyframe_payload_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(KEYFRAME_SET, SetKeyframe(0, "seed")))
    assert after == before
    assert any("malformed keyframe" in m for m in warnings_from(caplog, MAPPING_LOGGER))


# --- /keyframe/remove ------------------------------------------------------


def remove_keyframe(state, index):
    return apply_event(state, ControlEvent(KEYFRAME_REMOVE, RemoveKeyframe(index)))


def test_keyframe_remove_removes_at_index():
    before = ControlState()
    state = remove_keyframe(before, 1)
    assert len(state.keyframes) == 5
    assert state.keyframe_count == 5
    assert state.keyframes == before.keyframes[:1] + before.keyframes[2:]


def test_keyframe_remove_out_of_range_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = remove_keyframe(before, 99)
    assert after == before
    assert any("out of range" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_keyframe_remove_last_keyframe_guard(caplog):
    one = ControlState(keyframes=(default_keyframe(0),), keyframe_count=1)
    with caplog.at_level(logging.WARNING):
        after = remove_keyframe(one, 0)
    assert after == one
    assert any(
        "at least one keyframe" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_keyframe_remove_with_non_remove_keyframe_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(KEYFRAME_REMOVE, 1))
    assert after == before
    assert any("non keyframe value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


# --- keyframe_count resize --------------------------------------------------


def test_keyframe_count_grows_preserving_prefix_with_seed_fill():
    before = ControlState()
    state = apply_event(before, ControlEvent("/loop/keyframes", 9))
    assert state.keyframe_count == 9
    assert len(state.keyframes) == 9
    assert state.keyframes[:6] == before.keyframes
    assert state.keyframes[6] == default_keyframe(6)
    assert state.keyframes[8] == default_keyframe(8)


def test_keyframe_count_shrinks_preserving_prefix():
    before = ControlState()
    state = apply_event(before, ControlEvent("/loop/keyframes", 3))
    assert state.keyframe_count == 3
    assert state.keyframes == before.keyframes[:3]


def test_keyframe_count_clamped_to_registry_bounds():
    state = apply_event(ControlState(), ControlEvent("/loop/keyframes", 999))
    assert state.keyframe_count == 256
    assert len(state.keyframes) == 256


def test_keyframe_count_unchanged_is_a_noop_on_keyframes():
    before = ControlState()
    state = apply_event(before, ControlEvent("/loop/keyframes", 6))
    assert state.keyframes == before.keyframes


def test_keyframe_count_uncoercible_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent("/loop/keyframes", "nope"))
    assert after == before
