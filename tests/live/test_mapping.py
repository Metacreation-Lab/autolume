import logging

import pytest

from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.params import (
    ADJUST_DIRECTIONS,
    BEND_NOISE,
    BEND_RATIO,
    BEND_REMOVE,
    BEND_SET,
    BINDING_CLEAR,
    BINDING_SET,
    KEYFRAME_REMOVE,
    KEYFRAME_SET,
    MIX_LAYERS,
    VECTOR_RANDOMIZE,
    VECTOR_SET,
    Binding,
    ClearBinding,
    ControlState,
    Keyframe,
    RemoveKeyframe,
    RemoveTransform,
    SetCombinedLayers,
    SetDirections,
    SetKeyframe,
    SetLayerNoise,
    SetLayerRatio,
    SetTransform,
    SetVector,
    Transform,
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


def test_keyframe_set_at_len_appends():
    before = ControlState()
    keyframe = Keyframe("vec", vec=(1.0, 2.0))
    state = set_keyframe(before, len(before.keyframes), keyframe)
    assert len(state.keyframes) == 7
    assert state.keyframes[-1] == keyframe


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
    assert state.keyframes == before.keyframes[:1] + before.keyframes[2:]


def test_keyframe_remove_out_of_range_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = remove_keyframe(before, 99)
    assert after == before
    assert any("out of range" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_keyframe_remove_last_keyframe_guard(caplog):
    one = ControlState(keyframes=(default_keyframe(0),))
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


# --- keyframe_count removal (item 13) ---------------------------------------
#
# The registry carries no keyframe_count parameter, and /loop/keyframes
# resizes nothing, any more: Add and per-row Remove, through
# KEYFRAME_SET/KEYFRAME_REMOVE above, are the only ways the list changes.
# The resize semantics this section used to cover (grow fills with seed
# keyframes, shrink truncates the prefix, clamped to a bound) went with
# _resize_keyframes, the code that implemented them.


def test_a_write_to_the_old_keyframes_address_is_silently_ignored():
    before = ControlState()
    state = apply_event(before, ControlEvent("/loop/keyframes", 9))
    assert state == before


# --- loop_index normalisation -------------------------------------------
#
# Plan 3 Task 2: loop_index writes are wrapped modulo the keyframe count at
# application time. Pinned at every entry point that can move loop_index out
# of step with keyframes (task-9 review, finding "important 1").


def test_loop_index_write_wraps_to_the_keyframe_count():
    before = ControlState(keyframes=tuple(default_keyframe(i) for i in range(3)))
    state = apply_event(before, ControlEvent("/loop/index", 4))
    assert state.loop_index == 1


def test_loop_index_write_within_bounds_is_unchanged():
    before = ControlState(keyframes=tuple(default_keyframe(i) for i in range(3)))
    state = apply_event(before, ControlEvent("/loop/index", 2))
    assert state.loop_index == 2


def test_keyframe_remove_wraps_a_stale_loop_index():
    before = ControlState(loop_index=5)  # last of the default 6 keyframes
    state = remove_keyframe(before, 0)
    assert len(state.keyframes) == 5
    assert state.loop_index == 0  # 5 % 5


def test_keyframe_remove_leaves_an_in_range_loop_index_untouched():
    before = ControlState(loop_index=2)
    state = remove_keyframe(before, 5)
    assert state.loop_index == 2


def test_preset_apply_wraps_a_stale_loop_index_against_decoded_keyframes():
    payload = preset_payload({"loop_index": 99})
    payload["keyframes"] = [
        {"kind": "seed", "seed_x": 0.0, "seed_y": 0.0, "project": True},
        {"kind": "seed", "seed_x": 1.0, "seed_y": 0.0, "project": True},
        {"kind": "seed", "seed_x": 2.0, "seed_y": 0.0, "project": True},
    ]
    state = apply_preset(ControlState(), payload)
    assert len(state.keyframes) == 3
    assert state.loop_index == 0  # 99 % 3


# --- Plan 4 registry growth: image, adjuster and mixing registry rows ------


def test_image_derivation_addresses_apply():
    state = apply_event(ControlState(), ControlEvent("/image/grayscale", 1.0))
    state = apply_event(state, ControlEvent("/image/contrast", 12.0))
    state = apply_event(state, ControlEvent("/image/normalize", 1.0))
    state = apply_event(state, ControlEvent("/image/channel", 42))
    state = apply_event(state, ControlEvent("/image/layer", "L3"))
    assert state.grayscale is True
    assert state.img_scale_db == 12.0
    assert state.img_normalize is True
    assert state.base_channel == 42
    assert state.capture_layer == "L3"


def test_image_contrast_clamps_to_bounds():
    state = apply_event(ControlState(), ControlEvent("/image/contrast", 999.0))
    assert state.img_scale_db == 40.0
    state = apply_event(state, ControlEvent("/image/contrast", -999.0))
    assert state.img_scale_db == -40.0


@pytest.mark.parametrize("i", range(1, 9))
def test_adjuster_weight_addresses_apply_and_clamp(i):
    state = apply_event(ControlState(), ControlEvent(f"/adjust/{i}", 2.5))
    assert getattr(state, f"adjust_w{i}") == 2.5
    state = apply_event(state, ControlEvent(f"/adjust/{i}", 999.0))
    assert getattr(state, f"adjust_w{i}") == 5.0
    state = apply_event(state, ControlEvent(f"/adjust/{i}", -999.0))
    assert getattr(state, f"adjust_w{i}") == -5.0


def test_mixing_addresses_apply():
    state = apply_event(ControlState(), ControlEvent("/mix/model", "/tmp/other.pkl"))
    state = apply_event(state, ControlEvent("/mix/enabled", 1.0))
    assert state.pkl2 == "/tmp/other.pkl"
    assert state.mixing_enabled is True


def test_machine_level_addresses_apply():
    state = apply_event(ControlState(), ControlEvent("/render/superres", 1.0))
    state = apply_event(state, ControlEvent("/render/device", "cuda"))
    state = apply_event(state, ControlEvent("/render/fp32", 1.0))
    state = apply_event(state, ControlEvent("/osc/port", 9999))
    state = apply_event(state, ControlEvent("/ndi/enabled", 1.0))
    state = apply_event(state, ControlEvent("/ndi/name", "Stage NDI"))
    state = apply_event(state, ControlEvent("/record", 1.0))
    state = apply_event(state, ControlEvent("/output/fullscreen", 1.0))
    assert state.use_superres is True
    assert state.device == "cuda"
    assert state.force_fp32 is True
    assert state.osc_port == 9999
    assert state.ndi_enabled is True
    assert state.ndi_name == "Stage NDI"
    assert state.recording is True
    assert state.fullscreen is True


# --- /bend/set --------------------------------------------------------------


def set_transform(state, index, transform):
    return apply_event(state, ControlEvent(BEND_SET, SetTransform(index, transform)))


def test_bend_set_appends_then_replaces_in_place():
    t1 = Transform("translate", "L1", (1.0, 2.0), (0, 1))
    t2 = Transform("ablate", "L2", (1.0,), (0,))
    state = set_transform(ControlState(), 0, t1)
    state = set_transform(state, 1, t2)
    assert state.transforms == (t1, t2)

    t1b = Transform("rotate", "L1", (45.0,), (0,))
    state = set_transform(state, 0, t1b)
    assert state.transforms == (t1b, t2)


@pytest.mark.parametrize(
    "op,params_",
    [
        ("translate", (1.0, 2.0)),
        ("rotate", (45.0,)),
        ("scale", (1.5,)),
        ("erode", (3.0,)),
        ("dilate", (3.0,)),
        ("invert", (1.0,)),
        ("flip-h", (1.0,)),
        ("flip-v", (1.0,)),
        ("binary-thresh", (0.5,)),
        ("scalar-multiply", (2.0,)),
        ("ablate", (1.0,)),
    ],
)
def test_bend_set_accepts_every_operator_at_its_arity(op, params_):
    transform = Transform(op, "L1", params_, (0,))
    state = set_transform(ControlState(), 0, transform)
    assert state.transforms == (transform,)


def test_bend_set_coerces_int_params_to_float():
    transform = Transform("ablate", "L1", (1,), (0,))
    state = set_transform(ControlState(), 0, transform)
    assert state.transforms[0].params == (1.0,)
    assert all(isinstance(p, float) for p in state.transforms[0].params)


def test_bend_set_rejects_unknown_operator(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("sharpen", "L1", (1.0,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_set_rejects_unexposed_operator(caplog):
    # sobel/canny/resize exist in bending/transform_layers.py but are
    # deliberately not part of the eleven exposed operators.
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("sobel", "L1", (1.0,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_set_rejects_empty_layer(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("ablate", "", (1.0,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_set_rejects_wrong_arity(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("translate", "L1", (1.0,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_bend_set_rejects_non_finite_params(caplog, bad):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("scale", "L1", (bad,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("indices", [(-1,), (0, -2)])
def test_bend_set_rejects_negative_indices(caplog, indices):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("ablate", "L1", (1.0,), indices))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("op", ["erode", "dilate"])
def test_bend_set_rejects_a_float_kernel_size(caplog, op):
    # transform_layers.py builds torch.ones((k, k)) from params[0]; a
    # non-integral k raises there.
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform(op, "L1", (3.5,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("op", ["erode", "dilate"])
@pytest.mark.parametrize("kernel", [0.0, -1.0, -3.0])
def test_bend_set_rejects_a_non_positive_kernel_size(caplog, op, kernel):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform(op, "L1", (kernel,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("op", ["erode", "dilate"])
def test_bend_set_accepts_a_valid_integral_kernel_size(op):
    transform = Transform(op, "L1", (5.0,), (0,))
    state = set_transform(ControlState(), 0, transform)
    assert state.transforms == (transform,)


@pytest.mark.parametrize("scale", [0.0, 1e-9, -1e-9, 1e-7, -1e-7])
def test_bend_set_rejects_a_scale_factor_below_the_safety_minimum(caplog, scale):
    # Scale inverts params[0] into an affine coefficient that reaches
    # kornia's grid_sample; below _MIN_SCALE_MAGNITUDE that coefficient gets
    # large enough to crash the process with a native signal (measured in
    # scale-guard-report.md). Covers zero and both signs of the danger band.
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("scale", "L1", (scale,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("scale", [1e-6, -1e-6, 1e-3, -1e-3])
def test_bend_set_accepts_a_scale_factor_at_or_above_the_safety_minimum(scale):
    transform = Transform("scale", "L1", (scale,), (0,))
    state = set_transform(ControlState(), 0, transform)
    assert state.transforms == (transform,)


@pytest.mark.parametrize("scale", [0.5, 2.0, -0.5, -2.0])
def test_bend_set_accepts_a_normal_scale_factor(scale):
    transform = Transform("scale", "L1", (scale,), (0,))
    state = set_transform(ControlState(), 0, transform)
    assert state.transforms == (transform,)


def test_bend_set_rejects_a_bool_kernel_size(caplog):
    # bool is an int subclass: float(True) == 1.0, which is integral and >= 1,
    # so without an explicit guard a stray True would sail through as a
    # silent 1x1 no-op erode instead of being dropped.
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("erode", "L1", (True,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_set_rejects_a_bool_param_on_a_float_wanting_operator(caplog):
    # The bool guard is uniform across all eleven operators, not just
    # erode/dilate's kernel size.
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 0, Transform("scale", "L1", (True,), (0,)))
    assert after == before
    assert any("invalid transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_set_out_of_range_index_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_transform(before, 5, Transform("ablate", "L1", (1.0,), (0,)))
    assert after == before
    assert any("out of range" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("bad_index", [True, False, "0", 1.5, None])
def test_bend_set_rejects_non_int_index(caplog, bad_index):
    before = ControlState()
    transform = Transform("ablate", "L1", (1.0,), (0,))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BEND_SET, SetTransform(bad_index, transform)))
    assert after == before
    assert any(
        "malformed transform index" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_bend_set_with_non_transform_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BEND_SET, (0, "ablate", "L1")))
    assert after == before
    assert any("non transform value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_set_with_osc_shaped_dict_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(
            before, ControlEvent(BEND_SET, {"index": 0, "op": "ablate"})
        )
    assert after == before
    assert any("non transform value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_set_with_non_transform_payload_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BEND_SET, SetTransform(0, "ablate")))
    assert after == before
    assert any("malformed transform" in m for m in warnings_from(caplog, MAPPING_LOGGER))


# --- /bend/remove -------------------------------------------------------


def remove_transform(state, index):
    return apply_event(state, ControlEvent(BEND_REMOVE, RemoveTransform(index)))


def test_bend_remove_removes_at_index():
    t1 = Transform("translate", "L1", (1.0, 2.0), (0, 1))
    t2 = Transform("ablate", "L2", (1.0,), (0,))
    state = ControlState(transforms=(t1, t2))
    state = remove_transform(state, 0)
    assert state.transforms == (t2,)


def test_bend_remove_can_empty_the_chain():
    t1 = Transform("ablate", "L1", (1.0,), (0,))
    state = remove_transform(ControlState(transforms=(t1,)), 0)
    assert state.transforms == ()


def test_bend_remove_out_of_range_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = remove_transform(before, 0)
    assert after == before
    assert any("out of range" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("bad_index", [True, False, "0", 1.5, None])
def test_bend_remove_rejects_non_int_index(caplog, bad_index):
    before = ControlState(transforms=(Transform("ablate", "L1", (1.0,), (0,)),))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BEND_REMOVE, RemoveTransform(bad_index)))
    assert after == before
    assert any(
        "malformed transform index" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_bend_remove_with_non_transform_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BEND_REMOVE, 1))
    assert after == before
    assert any("non transform value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


# --- /bend/noise ----------------------------------------------------------


def set_layer_noise(state, layer, strength):
    return apply_event(state, ControlEvent(BEND_NOISE, SetLayerNoise(layer, strength)))


def test_bend_noise_stores_a_non_zero_strength():
    state = set_layer_noise(ControlState(), "L1", 0.5)
    assert state.layer_noise == (("L1", 0.5),)


def test_bend_noise_replaces_in_place():
    state = set_layer_noise(ControlState(), "L1", 0.5)
    state = set_layer_noise(state, "L2", 0.25)
    state = set_layer_noise(state, "L1", 0.75)
    assert state.layer_noise == (("L1", 0.75), ("L2", 0.25))


def test_bend_noise_zero_strength_removes_the_entry():
    state = set_layer_noise(ControlState(), "L1", 0.5)
    state = set_layer_noise(state, "L1", 0.0)
    assert state.layer_noise == ()


def test_bend_noise_zero_strength_on_absent_layer_is_a_noop():
    before = ControlState()
    state = set_layer_noise(before, "L1", 0.0)
    assert state.layer_noise == ()
    assert state == before


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_bend_noise_rejects_non_finite_strength(caplog, bad):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_layer_noise(before, "L1", bad)
    assert after == before
    assert any(
        "non finite layer noise strength" in m
        for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_bend_noise_rejects_uncoercible_strength(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_layer_noise(before, "L1", "nope")
    assert after == before
    assert any(
        "uncoercible layer noise strength" in m
        for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_bend_noise_rejects_empty_layer(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_layer_noise(before, "", 0.5)
    assert after == before
    assert any("malformed layer noise" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_bend_noise_with_non_layer_noise_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BEND_NOISE, ("L1", 0.5)))
    assert after == before
    assert any(
        "non layer noise value" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


# --- /bend/ratio ------------------------------------------------------------


def set_layer_ratio(state, layer, rx, ry):
    return apply_event(state, ControlEvent(BEND_RATIO, SetLayerRatio(layer, rx, ry)))


def test_bend_ratio_stores_a_non_neutral_ratio():
    state = set_layer_ratio(ControlState(), "L1", 2.0, 0.5)
    assert state.layer_ratios == (("L1", 2.0, 0.5),)


def test_bend_ratio_replaces_in_place():
    state = set_layer_ratio(ControlState(), "L1", 2.0, 0.5)
    state = set_layer_ratio(state, "L2", 0.5, 2.0)
    state = set_layer_ratio(state, "L1", 3.0, 3.0)
    assert state.layer_ratios == (("L1", 3.0, 3.0), ("L2", 0.5, 2.0))


def test_bend_ratio_neutral_removes_the_entry():
    state = set_layer_ratio(ControlState(), "L1", 2.0, 0.5)
    state = set_layer_ratio(state, "L1", 1.0, 1.0)
    assert state.layer_ratios == ()


def test_bend_ratio_neutral_on_absent_layer_is_a_noop():
    before = ControlState()
    state = set_layer_ratio(before, "L1", 1.0, 1.0)
    assert state.layer_ratios == ()
    assert state == before


@pytest.mark.parametrize(
    "rx,ry",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (float("-inf"), 1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (1.0, float("-inf")),
    ],
)
def test_bend_ratio_rejects_non_finite(caplog, rx, ry):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_layer_ratio(before, "L1", rx, ry)
    assert after == before
    assert any(
        "non finite layer ratio" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_bend_ratio_rejects_uncoercible_values(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_layer_ratio(before, "L1", "nope", 1.0)
    assert after == before
    assert any(
        "uncoercible layer ratio" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_bend_ratio_with_non_layer_ratio_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BEND_RATIO, ("L1", 2.0, 0.5)))
    assert after == before
    assert any(
        "non layer ratio value" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


# --- /adjust/directions -----------------------------------------------------


def set_directions(state, vectors):
    return apply_event(state, ControlEvent(ADJUST_DIRECTIONS, SetDirections(vectors)))


def test_adjust_directions_sets_the_vectors():
    vectors = ((1.0, 0.0), (0.0, 1.0))
    state = set_directions(ControlState(), vectors)
    assert state.directions == vectors


def test_adjust_directions_zeroes_weights_beyond_the_new_count():
    before = ControlState(
        adjust_w1=1.0, adjust_w2=2.0, adjust_w3=3.0, adjust_w4=4.0, adjust_w8=8.0
    )
    state = set_directions(before, ((1.0,), (2.0,), (3.0,)))
    assert (state.adjust_w1, state.adjust_w2, state.adjust_w3) == (1.0, 2.0, 3.0)
    assert state.adjust_w4 == 0.0
    assert state.adjust_w8 == 0.0


def test_adjust_directions_with_zero_vectors_zeroes_every_weight():
    before = ControlState(adjust_w1=1.0, adjust_w5=5.0)
    state = set_directions(before, ())
    assert state.directions == ()
    assert all(getattr(state, f"adjust_w{i}") == 0.0 for i in range(1, 9))


def test_adjust_directions_coerces_ints_to_floats():
    state = set_directions(ControlState(), ((1, 2), (3, 4)))
    assert state.directions == ((1.0, 2.0), (3.0, 4.0))


def test_adjust_directions_rejects_more_than_eight(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_directions(before, tuple((1.0,) for _ in range(9)))
    assert after == before
    assert any("malformed directions" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_adjust_directions_rejects_mismatched_lengths(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_directions(before, ((1.0, 2.0), (3.0,)))
    assert after == before
    assert any("malformed directions" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_adjust_directions_rejects_zero_length_vectors(caplog):
    # Distinct from loading zero directions (`()`, covered above): here two
    # vectors are present but each is empty, a meaningless adjuster state.
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_directions(before, ((), ()))
    assert after == before
    assert any("malformed directions" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_adjust_directions_rejects_non_finite_entry(caplog, bad):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_directions(before, ((1.0, bad),))
    assert after == before
    assert any("malformed directions" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_adjust_directions_with_non_directions_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(ADJUST_DIRECTIONS, [(1.0, 0.0)]))
    assert after == before
    assert any(
        "non directions value" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


# --- /mix/layers --------------------------------------------------------


def set_combined_layers(state, entries):
    return apply_event(state, ControlEvent(MIX_LAYERS, SetCombinedLayers(entries)))


def test_mix_layers_sets_the_entries():
    state = set_combined_layers(ControlState(), ("A", "B", "X"))
    assert state.combined_layers == ("A", "B", "X")


def test_mix_layers_rejects_an_invalid_entry(caplog):
    before = ControlState(combined_layers=("A",))
    with caplog.at_level(logging.WARNING):
        after = set_combined_layers(before, ("A", "C"))
    assert after == before
    assert any(
        "invalid entry" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


def test_mix_layers_with_non_combined_layers_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(MIX_LAYERS, ["A", "B"]))
    assert after == before
    assert any(
        "non combined layers value" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )
