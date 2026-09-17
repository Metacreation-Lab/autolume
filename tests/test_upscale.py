import numpy as np
import pytest
import spandrel
import torch
from spandrel.architectures.Compact import Compact

import upscale.core as core
from upscale.core import (
    MODELS, PERFORM_MODEL, PREPARE_MODEL, _tiled_forward, _upscale_once,
    load_upscaler, needs_upscale, required_weights, upscale_passes,
    upscale_to_target)
from upscale.weights import WEIGHTS

# The real tile is 1024, which would make the multi tile tests run on huge
# arrays. They shrink it instead: the tiling math does not depend on the size.
TEST_TILE = 64
TEST_TILE_OVERLAP = 8


@pytest.fixture
def tile(monkeypatch):
    """Run the tiling path on a small tile and return its size."""
    monkeypatch.setattr(core, "TILE", TEST_TILE)
    monkeypatch.setattr(core, "TILE_OVERLAP", TEST_TILE_OVERLAP)
    return TEST_TILE


def test_needs_upscale_short_side_rule():
    assert needs_upscale(400, 400, 1024)
    assert needs_upscale(2000, 500, 1024)   # short side below target
    assert needs_upscale(500, 2000, 1024)
    assert not needs_upscale(1024, 1024, 1024)
    assert not needs_upscale(1600, 1200, 1024)


def test_needs_upscale_missing_dims():
    assert not needs_upscale(None, None, 1024)
    assert not needs_upscale(0, 512, 1024)


def test_upscale_passes():
    assert upscale_passes(1024, 1024, 1024) == 0
    assert upscale_passes(400, 400, 1024) == 1
    assert upscale_passes(200, 4000, 1024) == 2
    assert upscale_passes(60, 60, 1024) == 3       # capped at MAX_PASSES
    assert upscale_passes(1, 1, 1024) == 3


class Fake4x(torch.nn.Module):
    """Stand-in for the real upscaler: 4x nearest upsample with one param."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest") * self.scale


def test_upscale_to_target_shapes():
    model = Fake4x().eval()
    img = np.full((100, 150, 3), 128, dtype=np.uint8)
    out = upscale_to_target(img, model, 1024)
    # short side 100 -> 400 -> 1600, two passes
    assert out.shape == (1600, 2400, 3)
    assert out.dtype == np.uint8


def test_upscale_to_target_noop_above_target():
    model = Fake4x().eval()
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    out = upscale_to_target(img, model, 1024)
    assert out.shape == (1024, 1024, 3)


class CountingFake4x(Fake4x):
    """Fake4x that records how many forward passes it was asked for."""
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return super().forward(x)


def _rand_input(h, w):
    torch.manual_seed(h * 10000 + w)
    return torch.rand(1, 3, h, w)


def test_tile_overlap_fits_inside_the_tile():
    assert 0 < core.TILE_OVERLAP < core.TILE


def test_tiled_forward_matches_plain_forward_on_a_multi_tile_grid(tile):
    model = Fake4x().eval()
    inp = _rand_input(tile * 3, tile * 4 + 20)
    assert torch.equal(_tiled_forward(model, inp), model(inp))


def test_tiled_forward_matches_on_sizes_not_divisible_by_tile(tile):
    model = Fake4x().eval()
    for h, w in [(tile + 1, tile + 1), (tile * 2 + 5, tile + 7),
                 (tile * 2, tile * 3), (tile - 20, tile)]:
        inp = _rand_input(h, w)
        assert torch.equal(_tiled_forward(model, inp), model(inp)), (h, w)


def test_tiled_forward_handles_thin_edge_slivers(tile):
    model = Fake4x().eval()
    for h, w in [(tile + 1, tile * 2 + 1), (tile * 2 + 2, tile + 3)]:
        inp = _rand_input(h, w)
        assert torch.equal(_tiled_forward(model, inp), model(inp)), (h, w)


def test_tiled_forward_uses_a_single_pass_when_input_fits_a_tile(tile):
    model = CountingFake4x().eval()
    inp = _rand_input(tile, tile)
    out = _tiled_forward(model, inp)
    assert model.calls == 1
    assert torch.equal(out, Fake4x().eval()(inp))


def test_tiled_forward_tiles_when_one_side_exceeds_the_tile(tile):
    model = CountingFake4x().eval()
    _tiled_forward(model, _rand_input(tile, tile + 1))
    assert model.calls > 1


def test_tiled_forward_output_shape(tile):
    model = Fake4x().eval()
    for h, w in [(16, 16), (tile, tile), (tile + 44, tile + 44),
                 (tile * 2, tile * 3), (tile - 7, tile * 4)]:
        out = _tiled_forward(model, _rand_input(h, w))
        assert out.shape == (1, 3, h * 4, w * 4), (h, w)


def test_tiled_forward_overlap_is_bounded_by_the_image(tile):
    """Tiles at the borders take context only from inside the image."""
    model = Fake4x().eval()
    inp = _rand_input(tile + 5, tile + 5)
    assert torch.equal(_tiled_forward(model, inp), model(inp))


def test_upscale_once_end_to_end_above_tile_size(tile):
    model = Fake4x().eval()
    img = np.random.default_rng(0).integers(
        0, 256, (tile + 40, tile + 90, 3), dtype=np.uint8)
    out = _upscale_once(img, model)
    assert out.shape == ((tile + 40) * 4, (tile + 90) * 4, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[:4, :4, 0], np.full((4, 4), img[0, 0, 0]))


def _compact_state_dict():
    """Weights of a small Compact architecture, randomly initialised."""
    return Compact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                   upscale=4, act_type="prelu").state_dict()


class _FakeDescriptor:
    """Stand-in for spandrel's ImageModelDescriptor."""
    def __init__(self, supports_half=True):
        self.supports_half = supports_half
        self.model = Fake4x()
        self.halved = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def half(self):
        self.halved = True
        return self


def _stub_spandrel(monkeypatch, supports_half=True):
    """Replace the spandrel loader so file loads never touch real weights."""
    loaded, descriptor = [], _FakeDescriptor(supports_half)

    class FakeLoader:
        def load_from_file(self, path):
            loaded.append(path)
            return descriptor

    monkeypatch.setattr(core, "spandrel", type("S", (), {"ModelLoader": FakeLoader}))
    return loaded, descriptor


def _stub_weight_loading(monkeypatch):
    """Serve fake weights so load_upscaler runs without touching the network."""
    calls = []

    def fake_ensure(name, progress_cb=None, cancel_event=None):
        calls.append(name)
        return name

    monkeypatch.setattr(core, "ensure_weight", fake_ensure)
    monkeypatch.setattr(core, "get_device", lambda: torch.device("cpu"))
    return calls


def test_load_upscaler_loads_the_single_weight_of_the_model(monkeypatch):
    calls = _stub_weight_loading(monkeypatch)
    loaded, descriptor = _stub_spandrel(monkeypatch)
    model = load_upscaler("CompactC3")
    assert calls == ["CompactC3"]
    assert loaded == ["CompactC3"]
    assert model is descriptor.model


def test_load_upscaler_defaults_to_the_prepare_model(monkeypatch):
    calls = _stub_weight_loading(monkeypatch)
    _stub_spandrel(monkeypatch)
    load_upscaler()
    assert calls == [MODELS[PREPARE_MODEL]["weights"]]


def test_load_upscaler_returns_none_when_weight_missing(monkeypatch):
    monkeypatch.setattr(core, "ensure_weight",
                        lambda name, progress_cb=None, cancel_event=None: None)
    assert load_upscaler() is None


def test_loader_stays_fp32_on_cpu(monkeypatch):
    _stub_weight_loading(monkeypatch)
    _, descriptor = _stub_spandrel(monkeypatch)
    load_upscaler()
    assert not descriptor.halved


def test_loader_uses_fp16_on_gpu(monkeypatch):
    _stub_weight_loading(monkeypatch)
    monkeypatch.setattr(core, "get_device", lambda: torch.device("cuda"))
    _, descriptor = _stub_spandrel(monkeypatch)
    load_upscaler()
    assert descriptor.halved


def test_loader_skips_fp16_when_model_rejects_it(monkeypatch):
    _stub_weight_loading(monkeypatch)
    monkeypatch.setattr(core, "get_device", lambda: torch.device("cuda"))
    _, descriptor = _stub_spandrel(monkeypatch, supports_half=False)
    load_upscaler()
    assert not descriptor.halved


def test_spandrel_detects_compact_arch():
    sd = _compact_state_dict()
    descriptor = spandrel.ModelLoader().load_from_state_dict(sd)
    assert descriptor.scale == 4
    loaded = descriptor.model.state_dict()
    assert loaded.keys() == sd.keys()
    for key, value in sd.items():
        assert torch.allclose(loaded[key], value)


def test_weights_registry():
    assert set(WEIGHTS) == {"RealPLKSR", "CompactC3"}
    for filename, url in WEIGHTS.values():
        assert url.startswith("https://")
        assert url.endswith(filename)


def test_models_reference_known_weights():
    assert set(MODELS) == {"RealPLKSR", "CompactC3"}
    for spec in MODELS.values():
        assert spec["weights"] in WEIGHTS


def test_each_screen_has_one_known_model():
    assert PREPARE_MODEL == "RealPLKSR"
    assert PERFORM_MODEL == "CompactC3"
    assert PREPARE_MODEL in MODELS
    assert PERFORM_MODEL in MODELS


def test_model_names_are_unique_and_non_empty():
    names = [spec["name"] for spec in MODELS.values()]
    assert all(names)
    assert len(set(names)) == len(names)


def test_required_weights():
    for key in MODELS:
        assert required_weights(key) == [MODELS[key]["weights"]]
