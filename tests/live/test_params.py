import dataclasses

import pytest

from autolume.live.core import params

# Fields of ControlState that hold user intent rather than a registry parameter.
NON_PARAM_FIELDS = {"bindings", "latent_vec", "keyframes"}

# Address prefixes reserved for structured control events, which carry Python
# objects instead of scalars and are never registry parameters.
RESERVED_PREFIXES = ("/binding/", "/touch/", "/preset/", "/vector/", "/keyframe/")


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
    )
    for address in structured:
        assert address.startswith(RESERVED_PREFIXES)
        assert address not in params.BY_ADDRESS


def test_registry_addresses_do_not_collide_with_structured_addresses():
    structured = {
        params.VECTOR_SET,
        params.VECTOR_RANDOMIZE,
        params.KEYFRAME_SET,
        params.KEYFRAME_REMOVE,
        params.BINDING_SET,
        params.BINDING_CLEAR,
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
        "keyframe_count": ("/loop/keyframes", params.ParamKind.INT, 6, 1, 256),
        "perfect_loop": ("/loop/perfect", params.ParamKind.BOOL, False, None, None),
        "noise_loop": ("/loop/noise", params.ParamKind.BOOL, False, None, None),
        "noise_radius": ("/loop/radius", params.ParamKind.FLOAT, 1.0, 0.01, 100.0),
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


def test_registry_has_sixteen_motion_rows():
    motion_names = {
        "vector_mode",
        "latent_project",
        "loop_active",
        "loop_uses_time",
        "loop_time",
        "loop_speed",
        "loop_alpha",
        "loop_index",
        "keyframe_count",
        "perfect_loop",
        "noise_loop",
        "noise_radius",
        "noise_loop_seed",
        "pulse_address",
        "pulse_ip",
        "pulse_port",
    }
    assert len(motion_names) == 16
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


def test_to_render_params_clamps_loop_index_to_keyframe_count():
    keyframes = tuple(params.default_keyframe(i) for i in range(3))
    state = params.ControlState(keyframes=keyframes, loop_index=99)
    assert params.to_render_params(state).loop_index == 2


def test_to_render_params_loop_index_within_bounds_is_unchanged():
    keyframes = tuple(params.default_keyframe(i) for i in range(6))
    state = params.ControlState(keyframes=keyframes, loop_index=2)
    assert params.to_render_params(state).loop_index == 2
