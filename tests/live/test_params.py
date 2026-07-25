import dataclasses

from autolume.live.core import params

# Fields of ControlState that hold user intent rather than a registry parameter.
NON_PARAM_FIELDS = {"bindings"}

# Address prefixes reserved for structured control events, which carry Python
# objects instead of scalars and are never registry parameters.
RESERVED_PREFIXES = ("/binding/", "/touch/", "/preset/")


def test_registry_covers_control_state_fields():
    field_names = {f.name for f in dataclasses.fields(params.ControlState)}
    assert NON_PARAM_FIELDS <= field_names
    assert field_names - NON_PARAM_FIELDS == set(params.REGISTRY.keys())


def test_registry_defaults_match_control_state():
    state = params.ControlState()
    for name, spec in params.REGISTRY.items():
        assert getattr(state, name) == spec.default


def test_addresses_are_unique_and_slash_prefixed():
    addresses = [spec.address for spec in params.REGISTRY.values()]
    assert len(addresses) == len(set(addresses))
    assert all(a.startswith("/") for a in addresses)
    assert set(params.BY_ADDRESS.keys()) == set(addresses)


def test_addresses_avoid_reserved_namespaces():
    for spec in params.REGISTRY.values():
        assert not spec.address.startswith(RESERVED_PREFIXES)
    assert params.BINDING_SET.startswith(RESERVED_PREFIXES)
    assert params.BINDING_CLEAR.startswith(RESERVED_PREFIXES)
    assert params.BINDING_SET not in params.BY_ADDRESS
    assert params.BINDING_CLEAR not in params.BY_ADDRESS


def test_numeric_defaults_within_bounds():
    for spec in params.REGISTRY.values():
        if spec.minimum is not None:
            assert spec.default >= spec.minimum
        if spec.maximum is not None:
            assert spec.default <= spec.maximum


def test_to_render_params_projects_state():
    state = params.ControlState(latent_x=2.5, truncation_psi=1.1, fps_cap=30)
    rp = params.to_render_params(state)
    assert rp.latent_x == 2.5
    assert rp.truncation_psi == 1.1
    assert rp.fps_cap == 30
    assert rp.pkl_path is None


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
