import numpy as np
import torch

import super_res.dataset_upscale as dataset_upscale
from super_res.dataset_upscale import (
    blend_state_dicts, load_upscaler, needs_upscale, required_weights,
    upscale_passes, upscale_to_target)
from super_res.net_base import SRVGGNetCompact
from super_res.super_res import SR_WEIGHTS


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
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                           upscale=4, act_type="prelu").state_dict()


def _stub_weight_loading(monkeypatch):
    """Serve fake weights so load_upscaler runs without touching the network."""
    state_dicts = {"Balance": _compact_state_dict(), "BalanceWDN": _compact_state_dict()}
    calls = []

    def fake_ensure(model_type, progress_cb=None, cancel_event=None):
        calls.append(model_type)
        return model_type

    monkeypatch.setattr(dataset_upscale, "ensure_sr_weight", fake_ensure)
    monkeypatch.setattr(dataset_upscale.torch, "load",
                        lambda path, map_location=None: {"params": state_dicts[path]})
    monkeypatch.setattr(dataset_upscale, "get_device", lambda: torch.device("cpu"))
    return state_dicts, calls


def test_load_upscaler_skips_wdn_at_full_denoise(monkeypatch):
    state_dicts, calls = _stub_weight_loading(monkeypatch)
    model = load_upscaler(denoise=1.0)
    assert calls == ["Balance"]
    loaded = model.state_dict()
    for key, value in state_dicts["Balance"].items():
        assert torch.allclose(loaded[key], value)


def test_load_upscaler_blends_wdn_below_full_denoise(monkeypatch):
    state_dicts, calls = _stub_weight_loading(monkeypatch)
    model = load_upscaler(denoise=0.5)
    assert calls == ["Balance", "BalanceWDN"]
    loaded = model.state_dict()
    for key, value in state_dicts["Balance"].items():
        expected = 0.5 * value + 0.5 * state_dicts["BalanceWDN"][key]
        assert torch.allclose(loaded[key], expected)


def test_load_upscaler_returns_none_when_weight_missing(monkeypatch):
    monkeypatch.setattr(dataset_upscale, "ensure_sr_weight",
                        lambda model_type, progress_cb=None, cancel_event=None: None)
    assert load_upscaler() is None


def test_wdn_weight_registered():
    assert "BalanceWDN" in SR_WEIGHTS
    filename, url = SR_WEIGHTS["BalanceWDN"]
    assert filename == "realesr-general-wdn-x4v3.pth"
    assert url.startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/")


def test_load_upscaler_pure_wdn_at_zero_denoise(monkeypatch):
    state_dicts, calls = _stub_weight_loading(monkeypatch)
    model = load_upscaler(denoise=0.0)
    assert calls == ["BalanceWDN"]
    loaded = model.state_dict()
    for key, value in state_dicts["BalanceWDN"].items():
        assert torch.allclose(loaded[key], value)


def test_load_upscaler_quality_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(dataset_upscale, "ensure_sr_weight",
                        lambda model_type, progress_cb=None, cancel_event=None:
                        (calls.append(model_type) or "weights.pth"))
    fake = Fake4x()
    monkeypatch.setattr(dataset_upscale, "load_model", lambda choice, path: fake)
    model = load_upscaler(denoise=0.3, model_type="Quality")
    assert calls == ["Quality"]
    assert model is fake


def test_load_upscaler_quality_none_when_weight_missing(monkeypatch):
    monkeypatch.setattr(dataset_upscale, "ensure_sr_weight",
                        lambda model_type, progress_cb=None, cancel_event=None: None)
    assert load_upscaler(model_type="Quality") is None


def test_required_weights():
    assert required_weights("Quality", 0.5) == ["Quality"]
    assert required_weights("Balance", 1.0) == ["Balance"]
    assert required_weights("Balance", 0.0) == ["BalanceWDN"]
    assert required_weights("Balance", 0.5) == ["Balance", "BalanceWDN"]
