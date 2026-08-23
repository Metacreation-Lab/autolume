import numpy as np
import spandrel
import torch
from spandrel.architectures.Compact import Compact

import upscale.core as core
from upscale.core import (
    blend_state_dicts, load_upscaler, needs_upscale, required_weights,
    upscale_passes, upscale_to_target)
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


def test_blend_state_dicts():
    a = {"w": torch.ones(2, 2)}
    b = {"w": torch.zeros(2, 2)}
    out = blend_state_dicts(a, b, 0.25)
    assert torch.allclose(out["w"], torch.full((2, 2), 0.25))
    full = blend_state_dicts(a, b, 1.0)
    assert torch.allclose(full["w"], a["w"])


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
    """Weights of the realesr-general-x4v3 architecture, randomly initialised."""
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
    state_dicts = {"Balance": _compact_state_dict(), "BalanceWDN": _compact_state_dict()}
    calls = []

    def fake_ensure(name, progress_cb=None, cancel_event=None):
        calls.append(name)
        return name

    monkeypatch.setattr(core, "ensure_weight", fake_ensure)
    monkeypatch.setattr(core.torch, "load",
                        lambda path, map_location=None: {"params": state_dicts[path]})
    monkeypatch.setattr(core, "get_device", lambda: torch.device("cpu"))
    return state_dicts, calls


def test_load_upscaler_skips_wdn_at_full_denoise(monkeypatch):
    _, calls = _stub_weight_loading(monkeypatch)
    loaded, descriptor = _stub_spandrel(monkeypatch)
    model = load_upscaler(denoise=1.0)
    assert calls == ["Balance"]
    assert loaded == ["Balance"]
    assert model is descriptor.model


def test_load_upscaler_pure_wdn_at_zero_denoise(monkeypatch):
    _, calls = _stub_weight_loading(monkeypatch)
    loaded, descriptor = _stub_spandrel(monkeypatch)
    model = load_upscaler(denoise=0.0)
    assert calls == ["BalanceWDN"]
    assert loaded == ["BalanceWDN"]
    assert model is descriptor.model


def test_load_upscaler_blends_wdn_below_full_denoise(monkeypatch):
    state_dicts, calls = _stub_weight_loading(monkeypatch)
    model = load_upscaler(denoise=0.5)
    assert calls == ["Balance", "BalanceWDN"]
    assert not model.training     # spandrel hands back a training-mode module
    loaded = model.state_dict()
    assert loaded.keys() == state_dicts["Balance"].keys()
    for key, value in state_dicts["Balance"].items():
        expected = 0.5 * value + 0.5 * state_dicts["BalanceWDN"][key]
        assert torch.allclose(loaded[key], expected)


def test_load_upscaler_returns_none_when_weight_missing(monkeypatch):
    monkeypatch.setattr(core, "ensure_weight",
                        lambda name, progress_cb=None, cancel_event=None: None)
    assert load_upscaler() is None


def test_load_upscaler_returns_none_when_blend_partner_missing(monkeypatch):
    monkeypatch.setattr(core, "ensure_weight",
                        lambda name, progress_cb=None, cancel_event=None:
                        None if name == "BalanceWDN" else name)
    assert load_upscaler(denoise=0.5) is None


def test_load_upscaler_quality_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "ensure_weight",
                        lambda name, progress_cb=None, cancel_event=None:
                        (calls.append(name) or "weights.pth"))
    monkeypatch.setattr(core, "get_device", lambda: torch.device("cpu"))
    loaded, descriptor = _stub_spandrel(monkeypatch)
    model = load_upscaler(denoise=0.3, model_type="Quality")
    assert calls == ["Quality"]
    assert loaded == ["weights.pth"]
    assert model is descriptor.model


def test_load_upscaler_quality_none_when_weight_missing(monkeypatch):
    monkeypatch.setattr(core, "ensure_weight",
                        lambda name, progress_cb=None, cancel_event=None: None)
    assert load_upscaler(model_type="Quality") is None


def test_loader_stays_fp32_on_cpu(monkeypatch):
    _stub_weight_loading(monkeypatch)
    _, descriptor = _stub_spandrel(monkeypatch)
    load_upscaler(denoise=1.0)
    assert not descriptor.halved


def test_loader_uses_fp16_on_gpu(monkeypatch):
    _stub_weight_loading(monkeypatch)
    monkeypatch.setattr(core, "get_device", lambda: torch.device("cuda"))
    _, descriptor = _stub_spandrel(monkeypatch)
    load_upscaler(denoise=1.0)
    assert descriptor.halved


def test_loader_skips_fp16_when_model_rejects_it(monkeypatch):
    _stub_weight_loading(monkeypatch)
    monkeypatch.setattr(core, "get_device", lambda: torch.device("cuda"))
    _, descriptor = _stub_spandrel(monkeypatch, supports_half=False)
    load_upscaler(denoise=1.0)
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
    assert set(WEIGHTS) == {"Balance", "BalanceWDN", "Quality"}
    assert WEIGHTS["Balance"][0] == "realesr-general-x4v3.pth"
    assert WEIGHTS["BalanceWDN"][0] == "realesr-general-wdn-x4v3.pth"
    assert WEIGHTS["Quality"][0] == "RealESRGAN_x4plus.pth"
    for _, url in WEIGHTS.values():
        assert url.startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/")


def test_required_weights():
    assert required_weights("Quality", 0.5) == ["Quality"]
    assert required_weights("Balance", 1.0) == ["Balance"]
    assert required_weights("Balance", 0.0) == ["BalanceWDN"]
    assert required_weights("Balance", 0.5) == ["Balance", "BalanceWDN"]
