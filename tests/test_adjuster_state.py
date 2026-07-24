import pickle

import torch

from widgets.adjuster_state import STATE_VERSION, pack_state, unpack_state


def make_packed(n=3, dim=8):
    return pack_state(
        model_pkl="/models/a.pkl",
        dirs=torch.randn(n, dim),
        base_dirs=torch.randn(n, dim),
        weights=torch.zeros(n),
        use_osc=[False] * n,
        addresses=[""] * n,
        mappings=["x"] * n,
        base_is_feature=[True] * n,
    )


def test_roundtrip_through_pickle():
    packed = make_packed()
    state = unpack_state(pickle.loads(pickle.dumps(packed)))
    assert state is not None
    assert state["model_pkl"] == "/models/a.pkl"
    assert torch.equal(state["dirs"], packed["dirs"])
    assert torch.equal(state["base_dirs"], packed["base_dirs"])
    assert len(state["use_osc"]) == 3
    assert state["base_is_feature"] == [True, True, True]


def test_version_recorded():
    assert make_packed()["version"] == STATE_VERSION


def test_old_tuple_format_rejected():
    old = (torch.randn(6, 512), [""] * 6, "", torch.zeros(6),
           [False] * 6, [""] * 6, ["x"] * 6)
    assert unpack_state(old) is None


def test_wrong_version_rejected():
    packed = make_packed()
    packed["version"] = 99
    assert unpack_state(packed) is None


def test_length_mismatch_rejected():
    packed = make_packed()
    packed["use_osc"] = [False]
    assert unpack_state(packed) is None


def test_missing_key_rejected():
    packed = make_packed()
    del packed["base_dirs"]
    assert unpack_state(packed) is None


def test_zero_dim_tensor_rejected():
    packed = make_packed()
    packed["dirs"] = torch.tensor(1.0)
    assert unpack_state(packed) is None


def test_unsized_list_value_rejected():
    packed = make_packed()
    packed["use_osc"] = 7
    assert unpack_state(packed) is None
