import numpy as np
import spandrel
import torch
from spandrel.architectures.Compact import Compact

import upscale.core as core
from upscale.core import (
    MODELS, PERFORM_DEFAULT_MODEL, PERFORM_LABELS, PERFORM_MODELS,
    PREPARE_DEFAULT_MODEL, PREPARE_LABELS, PREPARE_MODELS, load_upscaler,
    needs_upscale, required_weights, upscale_passes, upscale_to_target)
from upscale.weights import WEIGHTS


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
    model = load_upscaler("DAT2")
    assert calls == ["DAT2"]
    assert loaded == ["DAT2"]
    assert model is descriptor.model


def test_load_upscaler_defaults_to_the_prepare_model(monkeypatch):
    calls = _stub_weight_loading(monkeypatch)
    _stub_spandrel(monkeypatch)
    load_upscaler()
    assert calls == [MODELS[PREPARE_DEFAULT_MODEL]["weights"]]


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
    assert set(WEIGHTS) == {"RealPLKSR", "DAT2", "CompactC3"}
    for filename, url in WEIGHTS.values():
        assert url.startswith("https://")
        assert url.endswith(filename)


def test_models_reference_known_weights():
    assert set(MODELS) == {"RealPLKSR", "DAT2", "CompactC3"}
    for spec in MODELS.values():
        assert spec["weights"] in WEIGHTS


def test_model_rosters_are_subsets():
    for roster in (PREPARE_MODELS, PERFORM_MODELS):
        assert set(roster) <= set(MODELS)


def test_prepare_roster_is_standard_and_restore():
    assert PREPARE_MODELS == ["RealPLKSR", "DAT2"]
    assert PREPARE_LABELS == {"RealPLKSR": "Standard", "DAT2": "Restore"}
    assert PREPARE_DEFAULT_MODEL == "RealPLKSR"
    assert PREPARE_DEFAULT_MODEL in PREPARE_MODELS


def test_perform_roster_is_fast_and_standard():
    assert PERFORM_MODELS == ["CompactC3", "RealPLKSR"]
    assert PERFORM_LABELS == {"CompactC3": "Fast", "RealPLKSR": "Standard"}
    assert PERFORM_DEFAULT_MODEL == "CompactC3"
    assert PERFORM_DEFAULT_MODEL in PERFORM_MODELS


def test_model_names_are_unique_and_non_empty():
    names = [spec["name"] for spec in MODELS.values()]
    assert all(names)
    assert len(set(names)) == len(names)


def test_required_weights():
    for key in MODELS:
        assert required_weights(key) == [MODELS[key]["weights"]]
