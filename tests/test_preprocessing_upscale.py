import queue

import numpy as np
import PIL.Image
import torch

import super_res.dataset_upscale as dataset_upscale
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
    settings.upscaleSettings = {"aiUpscale": ai_upscale, "denoise": 1.0, "model": "Balance"}
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
    assert settings.upscaleSettings == {"aiUpscale": False, "denoise": 0.0, "model": "Balance"}


def test_upscaler_called_for_below_target_image(tmp_path, monkeypatch):
    calls = []
    fake = Fake4x().eval()
    monkeypatch.setattr(dataset_upscale, "load_upscaler", lambda denoise, model_type: fake)
    real_upscale = dataset_upscale.upscale_to_target
    def spy(image, model, target_size):
        calls.append(image.shape)
        return real_upscale(image, model, target_size)
    monkeypatch.setattr(dataset_upscale, "upscale_to_target", spy)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    messages = _run(settings)

    assert calls == [(64, 64, 3)]
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))
    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (128, 128)


def test_upscaler_not_loaded_when_disabled(tmp_path, monkeypatch):
    def boom(denoise, model_type):
        raise AssertionError("load_upscaler must not be called when aiUpscale is off")
    monkeypatch.setattr(dataset_upscale, "load_upscaler", boom)
    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=False)
    messages = _run(settings)
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))


def test_at_target_image_skips_upscaler(tmp_path, monkeypatch):
    fake = Fake4x().eval()
    monkeypatch.setattr(dataset_upscale, "load_upscaler", lambda denoise, model_type: fake)
    calls = []
    monkeypatch.setattr(dataset_upscale, "upscale_to_target",
                        lambda image, model, target_size: (calls.append(1) or image))
    settings = _make_settings(tmp_path, image_size=128, target=128, ai_upscale=True)
    _run(settings)
    assert calls == []


def test_upscale_failure_keeps_the_image(tmp_path, monkeypatch):
    fake = Fake4x().eval()
    monkeypatch.setattr(dataset_upscale, "load_upscaler", lambda denoise, model_type: fake)

    def boom(image, model, target_size):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(dataset_upscale, "upscale_to_target", boom)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    messages = _run(settings)

    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))
    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (128, 128)


def test_load_upscaler_failure_falls_back_to_resizing(tmp_path, monkeypatch):
    def boom(denoise, model_type):
        raise RuntimeError("download failed")
    monkeypatch.setattr(dataset_upscale, "load_upscaler", boom)

    def never(image, model, target_size):
        raise AssertionError("upscale_to_target must not run without an upscaler")
    monkeypatch.setattr(dataset_upscale, "upscale_to_target", never)

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


def test_model_and_denoise_forwarded_to_loader(tmp_path, monkeypatch):
    seen = {}
    fake = Fake4x().eval()

    def fake_load(denoise, model_type):
        seen["args"] = (denoise, model_type)
        return fake
    monkeypatch.setattr(dataset_upscale, "load_upscaler", fake_load)

    settings = _make_settings(tmp_path, image_size=64, target=128, ai_upscale=True)
    settings.upscaleSettings["model"] = "Quality"
    settings.upscaleSettings["denoise"] = 0.25
    messages = _run(settings)

    assert seen["args"] == (0.25, "Quality")
    assert any(m.get("type") == "completed" for m in messages if isinstance(m, dict))
