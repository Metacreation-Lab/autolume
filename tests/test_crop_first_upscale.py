"""Dataset preparation upscales only the region the final resize keeps."""

import logging
import queue

import numpy as np
import PIL.Image
import pytest
import torch

import upscale
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils


class Fake4x(torch.nn.Module):
    """Nearest neighbour 4x, standing in for a real upscaler."""
    def __init__(self):
        super().__init__()
        # upscale reads the device and dtype off the first parameter.
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest") * self.scale


@pytest.fixture(autouse=True)
def no_silent_fallback(caplog):
    """The pipeline swallows processing errors, so make them fail the test."""
    yield
    errors = [r.getMessage() for r in caplog.get_records("call")
              if r.levelno >= logging.ERROR]
    assert not errors, errors


def _content(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _settings(tmp_path, width, height, target, resize_mode, ai_upscale=True,
              non_square=False, out_name="out", seed=0):
    src = tmp_path / f"src_{width}x{height}_{seed}.png"
    PIL.Image.fromarray(_content(width, height, seed)).save(src)
    settings = DatasetPreprocessingUtils()
    settings.images = [str(src)]
    settings.size = target
    settings.resizeMode = resize_mode
    settings.nonSquare = non_square
    settings.output_path = str(tmp_path / out_name)
    settings.upscaleSettings = {"aiUpscale": ai_upscale, "model": "RealPLKSR"}
    return settings


def _run(settings):
    q, reply = queue.Queue(), queue.Queue()
    q.put(settings)
    DatasetPreprocessingUtils.create_training_dataset(q, reply)


def _install_fake_upscaler(monkeypatch):
    """Record every shape handed to the upscaler, run a real nearest 4x pass."""
    calls = []
    fake = Fake4x().eval()
    monkeypatch.setattr(upscale, "load_upscaler", lambda model: fake)
    real = upscale.upscale_to_target

    def spy(image, model, target_size):
        calls.append(image.shape[:2])
        return real(image, model, target_size)

    monkeypatch.setattr(upscale, "upscale_to_target", spy)
    return calls


# width, height, target
SHAPES = [
    (64, 64, 128),      # square, below target
    (64, 200, 128),     # portrait
    (200, 64, 128),     # landscape
    (300, 4000, 512),   # extreme panorama
    (128, 128, 128),    # exactly target
    (300, 300, 128),    # above target
    (120, 500, 128),    # short side just below target
]


@pytest.mark.parametrize("width,height,target", SHAPES)
@pytest.mark.parametrize("resize_mode", [0, 1])
def test_upscaler_input_stays_within_target(tmp_path, monkeypatch, width, height,
                                            target, resize_mode):
    calls = _install_fake_upscaler(monkeypatch)
    _run(_settings(tmp_path, width, height, target, resize_mode))

    assert bool(calls) == (min(width, height) < target)
    assert all(h <= target and w <= target for h, w in calls), calls


@pytest.mark.parametrize("width,height,target", SHAPES)
@pytest.mark.parametrize("resize_mode", [0, 1])
def test_square_output_is_exactly_the_target(tmp_path, monkeypatch, width, height,
                                             target, resize_mode):
    _install_fake_upscaler(monkeypatch)
    settings = _settings(tmp_path, width, height, target, resize_mode)
    _run(settings)

    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (target, target)


@pytest.mark.parametrize("resize_mode", [0, 1])
def test_image_needing_no_upscale_is_untouched_by_the_upscaler(tmp_path, monkeypatch,
                                                               resize_mode):
    plain = _settings(tmp_path, 300, 300, 128, resize_mode, ai_upscale=False,
                      out_name="plain")
    _run(plain)

    _install_fake_upscaler(monkeypatch)
    upscaled = _settings(tmp_path, 300, 300, 128, resize_mode, ai_upscale=True,
                         out_name="ai")
    _run(upscaled)

    assert (tmp_path / "plain" / "image_00000.png").read_bytes() == \
        (tmp_path / "ai" / "image_00000.png").read_bytes()


def test_crop_against_the_border_takes_no_margin_there(tmp_path, monkeypatch):
    # The crop spans the full height, so there is no room above or below it.
    calls = _install_fake_upscaler(monkeypatch)
    _run(_settings(tmp_path, 200, 64, 128, resize_mode=1))

    assert calls == [(64, 64 + 2 * 16)]


def test_margin_follows_what_each_side_has(tmp_path, monkeypatch):
    # An odd width offsets the crop by 8 on the left and 9 on the right.
    calls = _install_fake_upscaler(monkeypatch)
    settings = _settings(tmp_path, 81, 64, 128, resize_mode=1)
    _run(settings)

    assert calls == [(64, 81)]
    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (128, 128)


def test_margin_never_pushes_the_crop_past_the_target(tmp_path, monkeypatch):
    # A 120px crop for a 128px target leaves room for 4 margin pixels a side.
    calls = _install_fake_upscaler(monkeypatch)
    _run(_settings(tmp_path, 120, 500, 128, resize_mode=1))

    assert calls == [(120 + 2 * 4, 120)]


def test_square_crop_needs_no_margin_when_it_is_the_whole_image(tmp_path, monkeypatch):
    calls = _install_fake_upscaler(monkeypatch)
    _run(_settings(tmp_path, 64, 64, 128, resize_mode=1))

    assert calls == [(64, 64)]


def test_center_crop_keeps_the_same_framing_as_upscaling_the_whole_image(tmp_path,
                                                                        monkeypatch):
    _install_fake_upscaler(monkeypatch)
    settings = _settings(tmp_path, 96, 64, 128, resize_mode=1)
    _run(settings)
    new = np.array(PIL.Image.open(tmp_path / "out" / "image_00000.png"))

    # The old order: upscale everything, then crop and resize.
    source = DatasetPreprocessingUtils().load_images(settings.images[0])
    old = DatasetPreprocessingUtils.resize_image_np(
        upscale.upscale_to_target(source, Fake4x().eval(), settings.size), settings)

    assert new.shape == old.shape
    # The two resamplings reach different distances past the crop border, so
    # only the interior is expected to match pixel for pixel.
    assert np.array_equal(new[8:-8, 8:-8], old[8:-8, 8:-8])


def test_stretch_shrinks_oversized_axes_before_upscaling(tmp_path, monkeypatch):
    calls = _install_fake_upscaler(monkeypatch)
    _run(_settings(tmp_path, 300, 64, 128, resize_mode=0))

    assert calls == [(64, 128)]


def test_non_square_upscales_only_the_aspect_region(tmp_path, monkeypatch):
    calls = _install_fake_upscaler(monkeypatch)
    settings = _settings(tmp_path, 400, 64, 128, resize_mode=1, non_square=True)
    settings.nonSquareSettings = {"widthRatio": 16, "heightRatio": 9,
                                  "paddingMode": 0}
    _run(settings)

    # 16:9 of a 64px tall source is 113px wide, plus 16 margin pixels a side.
    assert calls == [(64, 113 + 2 * 16)]
    out = PIL.Image.open(tmp_path / "out" / "image_00000.png")
    assert out.size == (128, 128)


def test_augmented_images_reuse_a_single_upscale(tmp_path, monkeypatch):
    calls = _install_fake_upscaler(monkeypatch)
    settings = _settings(tmp_path, 96, 64, 128, resize_mode=1)
    settings.augmentationSettings = {"xFlip": True, "yFlip": True}
    _run(settings)

    assert len(calls) == 1
    for name in ("image_00000.png", "image_00000_augmented1.png",
                 "image_00000_augmented2.png"):
        assert PIL.Image.open(tmp_path / "out" / name).size == (128, 128)
