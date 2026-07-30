import dataclasses

import pytest

from autolume.live.core import params

# Fields of ControlState that hold user intent rather than a registry parameter.
NON_PARAM_FIELDS = {
    "bindings",
    "latent_vec",
    "keyframes",
    "transforms",
    "layer_noise",
    "layer_ratios",
    "directions",
    "combined_layers",
}

# Address prefixes reserved for structured control events, which carry Python
# objects instead of scalars and are never registry parameters. "/bend/" is
# reserved wholesale because no registry row lives under it. "/adjust/" and
# "/mix/" cannot be reserved wholesale, since adjust_w1...adjust_w8 and
# pkl2/mixing_enabled are legitimate registry rows in those same namespaces;
# their structured siblings (/adjust/directions, /mix/layers) are checked by
# exact address instead, below.
RESERVED_PREFIXES = (
    "/binding/",
    "/touch/",
    "/preset/",
    "/vector/",
    "/keyframe/",
    "/bend/",
)

# Structured addresses whose namespace also hosts registry rows, so they
# cannot be covered by a reserved prefix. Checked by exact membership instead.
NAMESPACE_SHARED_STRUCTURED_ADDRESSES = (
    "/adjust/directions",
    "/mix/layers",
)


def test_registry_covers_control_state_fields():
    field_names = {f.name for f in dataclasses.fields(params.ControlState)}
    assert NON_PARAM_FIELDS <= field_names
    assert field_names - NON_PARAM_FIELDS == set(params.REGISTRY.keys())


def test_registry_defaults_match_control_state():
    state = params.ControlState()
    for name, spec in params.REGISTRY.items():
        assert getattr(state, name) == spec.default


def test_the_defaults_a_model_is_first_rendered_with_match_the_old_app():
    """Parity for the values that decide what the first frame looks like.

    These were checked against a real model rather than against the old app's
    source: a frame rendered with these defaults is byte-for-byte identical to
    what the old app draws for the same model. Truncation is the one that
    caught us. It had been 0.7 here and is 0.8 there, and it does not look like
    a bug, it looks like the new app rendering the model with duller colours,
    because that is exactly what pulling a latent closer to the average does.
    """
    old_app_defaults = {
        "latent_x": 0.0,
        "latent_y": 0.0,
        "anim_playing": False,
        "anim_speed_x": 0.25,
        "truncation_psi": 0.8,
        "global_noise": 1.0,
        "noise_enabled": True,
        "noise_seed": 0,
        "noise_anim": False,
    }
    state = params.ControlState()
    for name, value in old_app_defaults.items():
        assert getattr(state, name) == value, name


def test_addresses_are_unique_and_slash_prefixed():
    addresses = [spec.address for spec in params.REGISTRY.values()]
    assert len(addresses) == len(set(addresses))
    assert all(a.startswith("/") for a in addresses)
    assert set(params.BY_ADDRESS.keys()) == set(addresses)


def test_addresses_avoid_reserved_namespaces():
    for spec in params.REGISTRY.values():
        assert not spec.address.startswith(RESERVED_PREFIXES)
    structured = (
        params.BINDING_SET,
        params.BINDING_CLEAR,
        params.VECTOR_SET,
        params.VECTOR_RANDOMIZE,
        params.KEYFRAME_SET,
        params.KEYFRAME_REMOVE,
        params.BEND_SET,
        params.BEND_REMOVE,
        params.BEND_NOISE,
        params.BEND_RATIO,
    )
    for address in structured:
        assert address.startswith(RESERVED_PREFIXES)
        assert address not in params.BY_ADDRESS


def test_bend_adjust_and_mix_structured_addresses_are_pinned_literals():
    # Every test elsewhere compares against these same constants, so a typo
    # in one of them (e.g. BEND_SET = "/bend/st") would pass the whole suite
    # silently. Pin the literal wire strings here.
    assert params.BEND_SET == "/bend/set"
    assert params.BEND_REMOVE == "/bend/remove"
    assert params.BEND_NOISE == "/bend/noise"
    assert params.BEND_RATIO == "/bend/ratio"
    assert params.ADJUST_DIRECTIONS == "/adjust/directions"
    assert params.MIX_LAYERS == "/mix/layers"


def test_namespace_shared_structured_addresses_are_never_registered():
    # /adjust/directions and /mix/layers live in namespaces that also carry
    # registry rows (/adjust/1..8, /mix/model, /mix/enabled), so they cannot
    # be swept up by a reserved prefix. Checked individually instead.
    for address in NAMESPACE_SHARED_STRUCTURED_ADDRESSES:
        assert address not in params.BY_ADDRESS


def test_registry_addresses_do_not_collide_with_structured_addresses():
    structured = {
        params.VECTOR_SET,
        params.VECTOR_RANDOMIZE,
        params.KEYFRAME_SET,
        params.KEYFRAME_REMOVE,
        params.BINDING_SET,
        params.BINDING_CLEAR,
        params.BEND_SET,
        params.BEND_REMOVE,
        params.BEND_NOISE,
        params.BEND_RATIO,
        params.ADJUST_DIRECTIONS,
        params.MIX_LAYERS,
    }
    addresses = {spec.address for spec in params.REGISTRY.values()}
    assert not (addresses & structured)


def test_numeric_defaults_within_bounds():
    for spec in params.REGISTRY.values():
        if spec.minimum is not None:
            assert spec.default >= spec.minimum
        if spec.maximum is not None:
            assert spec.default <= spec.maximum


def test_noise_specs_declare_expected_addresses_and_kinds():
    expected = {
        "global_noise": ("/noise/global", params.ParamKind.FLOAT, 1.0, 0.0, 2.0),
        "noise_enabled": ("/noise/enabled", params.ParamKind.BOOL, True, None, None),
        "noise_seed": ("/noise/seed", params.ParamKind.INT, 0, 0, 2**31 - 1),
        "noise_anim": ("/noise/anim", params.ParamKind.BOOL, False, None, None),
    }
    for name, (address, kind, default, minimum, maximum) in expected.items():
        spec = params.REGISTRY[name]
        assert (spec.address, spec.kind, spec.default) == (address, kind, default)
        assert (spec.minimum, spec.maximum) == (minimum, maximum)


def test_to_render_params_projects_state():
    state = params.ControlState(latent_x=2.5, truncation_psi=1.1, fps_cap=30)
    rp = params.to_render_params(state)
    assert rp.latent_x == 2.5
    assert rp.truncation_psi == 1.1
    assert rp.fps_cap == 30
    assert rp.pkl_path is None


def test_to_render_params_projects_noise_state():
    state = params.ControlState(
        global_noise=0.25, noise_enabled=False, noise_seed=17, noise_anim=True
    )
    rp = params.to_render_params(state)
    assert rp.global_noise == 0.25
    assert rp.noise_enabled is False
    assert rp.noise_seed == 17
    assert rp.noise_anim is True


def test_apply_value_clamps_noise_params():
    state = params.apply_value(params.ControlState(), "global_noise", 9.0)
    assert state.global_noise == 2.0
    state = params.apply_value(state, "noise_seed", -4)
    assert state.noise_seed == 0


def test_apply_value_sets_float():
    state = params.apply_value(params.ControlState(), "latent_x", 4.2)
    assert state.latent_x == 4.2


def test_apply_value_clamps_to_bounds():
    state = params.apply_value(params.ControlState(), "truncation_psi", 5.0)
    assert state.truncation_psi == 2.0
    state = params.apply_value(state, "truncation_psi", -9.0)
    assert state.truncation_psi == -1.0


def test_apply_value_coerces_int_and_clamps():
    state = params.apply_value(params.ControlState(), "fps_cap", 30.6)
    assert state.fps_cap == 31
    state = params.apply_value(state, "fps_cap", 999.9)
    assert state.fps_cap == 240


def test_apply_value_coerces_bool_and_str():
    state = params.apply_value(params.ControlState(), "anim_playing", 1.0)
    assert state.anim_playing is True
    state = params.apply_value(state, "pkl_path", "/tmp/m.pkl")
    assert state.pkl_path == "/tmp/m.pkl"


def test_apply_value_unknown_name_ignored():
    before = params.ControlState()
    assert params.apply_value(before, "nope", 1.0) == before


def test_apply_value_uncoercible_ignored():
    before = params.ControlState()
    assert params.apply_value(before, "latent_x", "not a number") == before


# A non finite value cannot be clamped: max and min propagate a NaN instead of
# bounding it, so it would land in the state claiming to be within its declared
# range. It is a broken input rather than an extreme one, so it is refused.
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_apply_value_rejects_non_finite_float(value):
    before = params.ControlState(global_noise=0.5)
    assert params.apply_value(before, "global_noise", value) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_apply_value_rejects_non_finite_int(value):
    before = params.ControlState(noise_seed=7)
    assert params.apply_value(before, "noise_seed", value) == before


# D13: `bool(nan)` is True, an arbitrary answer to a meaningless question,
# and `str(nan)` is the text "nan" posing as a model reference. The same
# refusal as FLOAT and INT, on the grounds that a non-finite input is a
# broken input whatever parameter it lands on.
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_apply_value_rejects_non_finite_bool(value):
    before = params.ControlState(noise_enabled=False)
    assert params.apply_value(before, "noise_enabled", value) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_apply_value_rejects_non_finite_text(value):
    before = params.ControlState(pkl_path="/models/keep.pkl")
    assert params.apply_value(before, "pkl_path", value) == before


def test_apply_value_still_switches_a_bool_with_a_finite_number():
    before = params.ControlState(noise_enabled=False)
    assert params.apply_value(before, "noise_enabled", 1.0).noise_enabled is True


@pytest.mark.parametrize("name", ["latent_x", "noise_seed"])
def test_apply_value_rejects_a_number_too_large_for_a_float(name):
    before = params.ControlState()
    assert params.apply_value(before, name, 10**400) == before


def test_apply_value_keeps_unbounded_extremes_finite():
    state = params.apply_value(params.ControlState(), "latent_x", 1e30)
    assert state.latent_x == 1e30


# --- Plan 3 registry growth: motion parameters -----------------------------


def test_motion_specs_declare_expected_addresses_kinds_and_bounds():
    # Preset defaults to True; only the params that opt out list it explicitly.
    expected = {
        "vector_mode": ("/latent/vector", params.ParamKind.BOOL, False, None, None),
        "latent_project": ("/latent/project", params.ParamKind.BOOL, True, None, None),
        "loop_active": ("/loop/anim", params.ParamKind.BOOL, False, None, None),
        "loop_uses_time": ("/loop/timemode", params.ParamKind.BOOL, True, None, None),
        "loop_time": ("/loop/time", params.ParamKind.FLOAT, 4.0, 0.1, 600.0),
        "loop_speed": ("/loop/speed", params.ParamKind.FLOAT, 0.0, -5.0, 5.0),
        "loop_alpha": ("/loop/alpha", params.ParamKind.FLOAT, 0.0, 0.0, 1.0),
        "loop_index": ("/loop/index", params.ParamKind.INT, 0, 0, 2**31 - 1),
        "perfect_loop": ("/loop/perfect", params.ParamKind.BOOL, False, None, None),
        "noise_loop": ("/loop/noise", params.ParamKind.BOOL, False, None, None),
        "noise_radius": ("/loop/radius", params.ParamKind.FLOAT, 1.0, 0.01, 10.0),
        "noise_loop_seed": ("/loop/seed", params.ParamKind.INT, 0, 0, 2**31 - 1),
        "pulse_address": ("/loop/pulse/address", params.ParamKind.STR, "", None, None),
        "pulse_ip": ("/loop/pulse/ip", params.ParamKind.STR, "127.0.0.1", None, None),
        "pulse_port": ("/loop/pulse/port", params.ParamKind.INT, 5005, 1, 65535),
    }
    non_preset = {"pulse_ip", "pulse_port"}
    assert set(expected) <= set(params.REGISTRY)
    for name, (address, kind, default, minimum, maximum) in expected.items():
        spec = params.REGISTRY[name]
        assert (spec.address, spec.kind, spec.default) == (address, kind, default)
        assert (spec.minimum, spec.maximum) == (minimum, maximum)
        assert spec.preset is (name not in non_preset)


def test_registry_has_fifteen_motion_rows():
    # keyframe_count is not one of them any more (item 13): the list's
    # length is derived from len(keyframes) everywhere it is needed, not
    # tracked as a second, separately writable number.
    motion_names = {
        "vector_mode",
        "latent_project",
        "loop_active",
        "loop_uses_time",
        "loop_time",
        "loop_speed",
        "loop_alpha",
        "loop_index",
        "perfect_loop",
        "noise_loop",
        "noise_radius",
        "noise_loop_seed",
        "pulse_address",
        "pulse_ip",
        "pulse_port",
    }
    assert len(motion_names) == 15
    assert motion_names <= set(params.REGISTRY)


# --- Plan 3 registry growth: structured latent state ------------------------


def test_control_state_defaults_to_six_seed_keyframes():
    state = params.ControlState()
    assert len(state.keyframes) == 6
    for index, keyframe in enumerate(state.keyframes):
        assert keyframe == params.Keyframe("seed", float(index), 0.0)


def test_control_state_defaults_to_an_unset_vector():
    assert params.ControlState().latent_vec == ()


def test_default_keyframe_is_a_seed_keyframe_at_index_zero():
    keyframe = params.default_keyframe(3)
    assert keyframe == params.Keyframe("seed", 3.0, 0.0)
    assert keyframe.vec == ()
    assert keyframe.project is True


def test_keyframe_construction_does_not_validate():
    # Validation is the mapping layer's job, not the dataclass's.
    keyframe = params.Keyframe("vec", vec=(float("nan"),))
    assert keyframe.kind == "vec"


# --- Plan 3 registry growth: RenderParams derivation ------------------------


def test_to_render_params_carries_latent_vec_and_keyframes():
    vec = (0.1, 0.2, 0.3)
    keyframes = (params.Keyframe("vec", vec=vec),)
    state = params.ControlState(latent_vec=vec, keyframes=keyframes, latent_project=False)
    rp = params.to_render_params(state)
    assert rp.latent_vec == vec
    assert rp.keyframes == keyframes
    assert rp.latent_project is False


@pytest.mark.parametrize(
    "flags,expected_mode",
    [
        ({}, "seed"),
        ({"vector_mode": True}, "vec"),
        ({"loop_active": True}, "loop"),
        ({"loop_active": True, "vector_mode": True}, "loop"),
        ({"loop_active": True, "noise_loop": True}, "vec"),
        ({"noise_loop": True}, "seed"),
    ],
)
def test_to_render_params_derives_mode(flags, expected_mode):
    state = params.ControlState(**flags)
    assert params.to_render_params(state).mode == expected_mode


def test_to_render_params_wraps_loop_index_to_keyframe_count():
    keyframes = tuple(params.default_keyframe(i) for i in range(3))
    state = params.ControlState(keyframes=keyframes, loop_index=99)
    assert params.to_render_params(state).loop_index == 0  # 99 % 3, matching loop.advance


def test_to_render_params_loop_index_within_bounds_is_unchanged():
    keyframes = tuple(params.default_keyframe(i) for i in range(6))
    state = params.ControlState(keyframes=keyframes, loop_index=2)
    assert params.to_render_params(state).loop_index == 2


# --- Plan 4 registry growth: image, adjuster and mixing registry rows ------


def test_image_derivation_specs_declare_expected_addresses_kinds_and_bounds():
    expected = {
        "grayscale": ("/image/grayscale", params.ParamKind.BOOL, False, None, None),
        "img_scale_db": ("/image/contrast", params.ParamKind.FLOAT, 0.0, -40.0, 40.0),
        "img_normalize": ("/image/normalize", params.ParamKind.BOOL, False, None, None),
        "base_channel": ("/image/channel", params.ParamKind.INT, 0, 0, 8192),
        "capture_layer": ("/image/layer", params.ParamKind.STR, "", None, None),
    }
    for name, (address, kind, default, minimum, maximum) in expected.items():
        spec = params.REGISTRY[name]
        assert (spec.address, spec.kind, spec.default) == (address, kind, default)
        assert (spec.minimum, spec.maximum) == (minimum, maximum)
        assert spec.preset is True


def test_adjuster_weight_specs_declare_the_eight_fixed_slots():
    for i in range(1, 9):
        name = f"adjust_w{i}"
        spec = params.REGISTRY[name]
        assert spec.address == f"/adjust/{i}"
        assert spec.kind is params.ParamKind.FLOAT
        assert spec.default == 0.0
        assert (spec.minimum, spec.maximum) == (-5.0, 5.0)
        assert spec.preset is True
    assert "adjust_w9" not in params.REGISTRY


def test_mixing_specs_declare_expected_addresses_kinds_and_defaults():
    # pkl2 is preset=False like pkl_path: it needs path resolution a plain
    # param cannot express, and is persisted separately as the preset's
    # `model2` key (Task 11), never through the generic params dict.
    pkl2 = params.REGISTRY["pkl2"]
    assert (pkl2.address, pkl2.kind, pkl2.default) == ("/mix/model", params.ParamKind.STR, None)
    assert pkl2.preset is False

    mixing_enabled = params.REGISTRY["mixing_enabled"]
    assert (mixing_enabled.address, mixing_enabled.kind, mixing_enabled.default) == (
        "/mix/enabled",
        params.ParamKind.BOOL,
        False,
    )
    assert mixing_enabled.preset is True


def test_machine_level_specs_declare_expected_addresses_kinds_and_are_not_preset():
    expected = {
        "use_superres": ("/render/superres", params.ParamKind.BOOL, False, None, None),
        "device": ("/render/device", params.ParamKind.STR, "auto", None, None),
        "force_fp32": ("/render/fp32", params.ParamKind.BOOL, False, None, None),
        "osc_port": ("/osc/port", params.ParamKind.INT, 1338, 1, 65535),
        "ndi_enabled": ("/ndi/enabled", params.ParamKind.BOOL, False, None, None),
        "ndi_name": ("/ndi/name", params.ParamKind.STR, "Autolume Live", None, None),
        "recording": ("/record", params.ParamKind.BOOL, False, None, None),
        "fullscreen": ("/output/fullscreen", params.ParamKind.BOOL, False, None, None),
    }
    for name, (address, kind, default, minimum, maximum) in expected.items():
        spec = params.REGISTRY[name]
        assert (spec.address, spec.kind, spec.default) == (address, kind, default)
        assert (spec.minimum, spec.maximum) == (minimum, maximum)
        assert spec.preset is False


# --- Plan 4: Transform and the structured bending value objects ------------


def test_transform_construction_does_not_validate():
    # Validation is the mapping layer's job, not the dataclass's.
    transform = params.Transform("nope", "", params=(), indices=(-1,))
    assert transform.op == "nope"


def test_control_state_defaults_to_no_transforms_directions_or_mixing():
    state = params.ControlState()
    assert state.transforms == ()
    assert state.layer_noise == ()
    assert state.layer_ratios == ()
    assert state.directions == ()
    assert state.combined_layers == ()


# --- Plan 4: RenderParams derivation for bending, adjuster, image, mixing --


def test_to_render_params_carries_the_bending_chain_in_order():
    transforms = (
        params.Transform("translate", "L1", (1.0, 2.0), (0, 1)),
        params.Transform("ablate", "L2", (1.0,), (3,)),
    )
    state = params.ControlState(transforms=transforms)
    rp = params.to_render_params(state)
    assert rp.transforms == transforms


def test_to_render_params_projects_layer_noise_and_ratios_as_mappings():
    state = params.ControlState(
        layer_noise=(("L1", 0.5), ("L2", 1.5)),
        layer_ratios=(("L1", 2.0, 0.5),),
    )
    rp = params.to_render_params(state)
    assert rp.layer_noise == {"L1": 0.5, "L2": 1.5}
    assert rp.layer_ratios == {"L1": (2.0, 0.5)}


def test_to_render_params_projects_empty_layer_noise_and_ratios():
    rp = params.to_render_params(params.ControlState())
    assert rp.layer_noise == {}
    assert rp.layer_ratios == {}


def test_to_render_params_carries_directions_and_weights_without_computing_a_product():
    directions = ((1.0, 0.0), (0.0, 1.0))
    state = params.ControlState(directions=directions, adjust_w1=2.0, adjust_w2=3.0)
    rp = params.to_render_params(state)
    assert rp.directions == directions
    assert rp.adjust_w1 == 2.0
    assert rp.adjust_w2 == 3.0
    # The directions x weights product is the generator's job, not this
    # snapshot's: RenderParams must not carry a precomputed "direction" field.
    assert not hasattr(rp, "direction")


def test_to_render_params_carries_image_derivation_fields():
    state = params.ControlState(
        grayscale=True,
        img_scale_db=6.0,
        img_normalize=True,
        base_channel=12,
        capture_layer="L4",
    )
    rp = params.to_render_params(state)
    assert rp.grayscale is True
    assert rp.img_scale_db == 6.0
    assert rp.img_normalize is True
    assert rp.base_channel == 12
    assert rp.capture_layer == "L4"


def test_to_render_params_carries_mixing_fields():
    state = params.ControlState(
        pkl2="/tmp/second.pkl",
        mixing_enabled=True,
        combined_layers=("A", "B", "X"),
    )
    rp = params.to_render_params(state)
    assert rp.pkl2 == "/tmp/second.pkl"
    assert rp.mixing_enabled is True
    assert rp.combined_layers == ("A", "B", "X")


def test_to_render_params_default_mixing_and_image_fields():
    rp = params.to_render_params(params.ControlState())
    assert rp.pkl2 is None
    assert rp.mixing_enabled is False
    assert rp.combined_layers == ()
    assert rp.grayscale is False
    assert rp.img_scale_db == 0.0
    assert rp.img_normalize is False
    assert rp.base_channel == 0
    assert rp.capture_layer == ""
    assert rp.directions == ()
    assert all(getattr(rp, f"adjust_w{i}") == 0.0 for i in range(1, 9))


def test_to_render_params_carries_use_superres_and_force_fp32():
    state = params.ControlState(use_superres=True, force_fp32=True)
    rp = params.to_render_params(state)
    assert rp.use_superres is True
    assert rp.force_fp32 is True

    rp_default = params.to_render_params(params.ControlState())
    assert rp_default.use_superres is False
    assert rp_default.force_fp32 is False
