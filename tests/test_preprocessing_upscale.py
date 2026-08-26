import queue

import numpy as np
import PIL.Image
import torch

import upscale
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils


class Fake4x(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest") * self.scale


def _make_settings(tmp_path, image_size, target, ai_upscale):
    src = tmp_path / "src.png"
    PIL.Image.fromarray(
        np.full((image_size, image_size, 3), 100, dtype=np.uint8)).save(src)
    settings = DatasetPreprocessingUtils()
    settings.images = [str(src)]
    settings.size = target
    settings.resizeMode = 1
    settings.nonSquare = False
    settings.output_path = str(tmp_path / "out")
    settings.upscaleSettings = {"aiUpscale": ai_upscale, "model": "RealPLKSR"}
    return settings


def _run(settings):
    q, reply = queue.Queue(), queue.Queue()
    q.put(settings)
    DatasetPreprocessingUtils.create_training_dataset(q, reply)
    messages = []
    while not reply.empty():
        messages.append(reply.get())
    return messages


def test_default_settings_have_upscale_fields():
    settings = DatasetPreprocessingUtils()
    assert settings.upscaleSettings == {"aiUpscale": False, "model": "RealPLKSR"}


def test_upscaler_called_for_below_target_image(tmp_path, monkeypatch):
    calls = []
    fake = Fake4x().eval()
    monkeypatch.setattr(upscale, "load_upscaler", lambda model: fake)
    real_upscale = upscale.upscale_to_target
    def spy(image, model, target_size):
        calls.append(image.shape)
        return real_upscale(image, model, target_size)
    monkeypatch.setattr(upscale, "upscale_to_target", spy)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    messages = _run(settings)

    assert calls == [(64, 64, 3)]
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))
    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (128, 128)


def test_upscaler_not_loaded_when_disabled(tmp_path, monkeypatch):
    def boom(model):
        raise AssertionError("load_upscaler must not be called when aiUpscale is off")
    monkeypatch.setattr(upscale, "load_upscaler", boom)
    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=False)
    messages = _run(settings)
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))


def test_at_target_image_skips_upscaler(tmp_path, monkeypatch):
    fake = Fake4x().eval()
    monkeypatch.setattr(upscale, "load_upscaler", lambda model: fake)
    calls = []
    monkeypatch.setattr(upscale, "upscale_to_target",
                        lambda image, model, target_size: (calls.append(1) or image))
    settings = _make_settings(tmp_path, image_size=128, target=128, ai_upscale=True)
    _run(settings)
    assert calls == []


def test_upscale_failure_keeps_the_image(tmp_path, monkeypatch):
    fake = Fake4x().eval()
    monkeypatch.setattr(upscale, "load_upscaler", lambda model: fake)

    def boom(image, model, target_size):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(upscale, "upscale_to_target", boom)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    messages = _run(settings)

    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))
    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (128, 128)


def test_load_upscaler_failure_falls_back_to_resizing(tmp_path, monkeypatch):
    def boom(model):
        raise RuntimeError("download failed")
    monkeypatch.setattr(upscale, "load_upscaler", boom)

    def never(image, model, target_size):
        raise AssertionError("upscale_to_target must not run without an upscaler")
    monkeypatch.setattr(upscale, "upscale_to_target", never)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    messages = _run(settings)

    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))
    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (128, 128)


def test_settings_without_upscale_field_still_work(tmp_path):
    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=False)
    del settings.upscaleSettings
    messages = _run(settings)
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))


def test_stale_denoise_setting_is_ignored(tmp_path, monkeypatch):
    seen = {}
    fake = Fake4x().eval()

    def fake_load(model):
        seen["model"] = model
        return fake
    monkeypatch.setattr(upscale, "load_upscaler", fake_load)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    # Settings saved before the denoise slider was removed must not break.
    settings.upscaleSettings["denoise"] = 0.25
    messages = _run(settings)

    assert seen["model"] == "RealPLKSR"
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))


def test_settings_without_model_key_use_the_default(tmp_path, monkeypatch):
    seen = {}
    fake = Fake4x().eval()

    def fake_load(model):
        seen["model"] = model
        return fake
    monkeypatch.setattr(upscale, "load_upscaler", fake_load)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    del settings.upscaleSettings["model"]
    messages = _run(settings)

    assert seen["model"] == "RealPLKSR"
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))


def test_unknown_model_key_falls_back_to_the_default(tmp_path, monkeypatch):
    seen = {}
    fake = Fake4x().eval()

    def fake_load(model):
        seen["model"] = model
        return fake
    monkeypatch.setattr(upscale, "load_upscaler", fake_load)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    settings.upscaleSettings["model"] = "NoSuchModel"
    messages = _run(settings)

    assert seen["model"] == "RealPLKSR"
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))


def test_selected_model_forwarded_to_loader(tmp_path, monkeypatch):
    seen = {}
    fake = Fake4x().eval()

    def fake_load(model):
        seen["model"] = model
        return fake
    monkeypatch.setattr(upscale, "load_upscaler", fake_load)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    settings.upscaleSettings["model"] = "DAT2"
    messages = _run(settings)

    assert seen["model"] == "DAT2"
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))


def test_required_weights_for_prepare_models():
    for model in upscale.PREPARE_MODELS:
        assert upscale.required_weights(model) == [upscale.MODELS[model]["weights"]]


def test_prepare_models_are_known_and_start_with_the_default():
    assert upscale.PREPARE_MODELS[0] == upscale.PREPARE_DEFAULT_MODEL
    assert all(key in upscale.MODELS for key in upscale.PREPARE_MODELS)
