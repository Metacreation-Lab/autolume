import dataclasses

from autolume.live.core import params


def test_registry_covers_control_state_fields():
    field_names = {f.name for f in dataclasses.fields(params.ControlState)}
    assert field_names == set(params.REGISTRY.keys())


def test_registry_defaults_match_control_state():
    state = params.ControlState()
    for name, spec in params.REGISTRY.items():
        assert getattr(state, name) == spec.default


def test_addresses_are_unique_and_slash_prefixed():
    addresses = [spec.address for spec in params.REGISTRY.values()]
    assert len(addresses) == len(set(addresses))
    assert all(a.startswith("/") for a in addresses)
    assert set(params.BY_ADDRESS.keys()) == set(addresses)


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
