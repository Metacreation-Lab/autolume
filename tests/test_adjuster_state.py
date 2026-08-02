import torch

from widgets.adjuster_state import (NUM_SLOTS, STATE_VERSION, make_slot,
                                    pack_state, unpack_state)


def slots(n=NUM_SLOTS, dim=16):
    return [make_slot(torch.randn(dim), component=i, sigma=float(i + 1),
                      zone="all", weight=0.5, name=f"s{i}",
                      use_osc=True, address=f"addr{i}", mapping="x")
            for i in range(n)]


def test_version_is_2():
    assert STATE_VERSION == 2


def test_roundtrip():
    state = pack_state("model.pkl", slots())
    out = unpack_state(state)
    assert out is not None
    assert out["model_pkl"] == "model.pkl"
    assert len(out["slots"]) == NUM_SLOTS
    for i, slot in enumerate(out["slots"]):
        assert slot["component"] == i
        assert slot["name"] == f"s{i}"
        assert slot["address"] == f"addr{i}"
        assert slot["zone"] == "all"
        assert slot["sigma"] == float(i + 1)
        assert torch.is_tensor(slot["direction"]) and slot["direction"].shape == (16,)


def test_none_model_pkl_roundtrips():
    assert unpack_state(pack_state(None, slots()))["model_pkl"] is None


def test_rejects_old_version_dict():
    assert unpack_state({"version": 1, "dirs": torch.randn(8, 16)}) is None


def test_rejects_legacy_tuple():
    assert unpack_state((torch.randn(6, 512), [""] * 6)) is None


def test_rejects_garbage_without_raising():
    for bad in (None, 42, {"version": 2}, {"version": 2, "model_pkl": "x", "slots": 3},
                {"version": 2, "model_pkl": "x", "slots": [{"weight": 1}] * 8}):
        assert unpack_state(bad) is None


def test_rejects_mismatched_direction_lengths():
    s = slots()
    s[3]["direction"] = torch.randn(8)
    assert unpack_state(pack_state("m.pkl", s)) is None


def test_rejects_invalid_zone():
    s = slots()
    s[0]["zone"] = "bass"
    assert unpack_state(pack_state("m.pkl", s)) is None


def test_trims_extra_slots():
    out = unpack_state(pack_state("m.pkl", slots(12)))
    assert len(out["slots"]) == NUM_SLOTS
    assert out["slots"][-1]["name"] == "s7"


def test_pads_missing_slots():
    out = unpack_state(pack_state("m.pkl", slots(5)))
    assert len(out["slots"]) == NUM_SLOTS
    for slot in out["slots"][5:]:
        assert slot["component"] is None
        assert slot["weight"] == 0.0
        assert slot["use_osc"] is False
        assert slot["direction"].shape == (16,)


def test_custom_zone_with_layers_roundtrips():
    s = slots()
    s[0]["zone"] = "custom"
    s[0]["layers"] = [True, False, True]
    out = unpack_state(pack_state("m.pkl", s))
    assert out["slots"][0]["zone"] == "custom"
    assert out["slots"][0]["layers"] == [True, False, True]


def test_custom_zone_without_layers_falls_back_to_all():
    s = slots()
    s[0]["zone"] = "custom"
    out = unpack_state(pack_state("m.pkl", s))
    assert out["slots"][0]["zone"] == "all"
    assert out["slots"][0]["layers"] is None


def test_slots_without_layers_key_still_load():
    state = pack_state("m.pkl", slots())
    for slot in state["slots"]:
        del slot["layers"]
    out = unpack_state(state)
    assert out is not None
    assert all(slot["layers"] is None for slot in out["slots"])


def test_duplicate_components_are_preserved():
    s = slots()
    s[1]["component"] = 0  # same component and zone as slot 0
    s[2]["component"] = 0
    s[2]["zone"] = "color"
    out = unpack_state(pack_state("m.pkl", s))
    assert out["slots"][0]["component"] == 0
    assert out["slots"][1]["component"] == 0
    assert out["slots"][2]["component"] == 0
    assert out["slots"][2]["zone"] == "color"
