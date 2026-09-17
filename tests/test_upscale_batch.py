import queue

import cv2
import numpy as np
import torch

from dnnlib import EasyDict
from upscale import batch


class Fake4x(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest") * self.scale


def _args(**overrides):
    args = EasyDict(result_path="", input_path=[], model_type=batch.BATCH_DEFAULT_MODEL)
    args.update(overrides)
    return args


def test_video_output_name():
    name = batch.output_name("clip.mp4", "RealPLKSR", 1920, 1080, ".mp4")
    assert name == "clip_result_Quality_1920x1080.mp4"


def test_image_output_name():
    name = batch.output_name("shot.png", "CompactC3", 2048, 2048, ".jpg")
    assert name == "shot_result_Fast_2048x2048.jpg"


def test_output_name_keeps_dots_in_the_stem():
    name = batch.output_name("take.01.jpeg", "CompactC3", 512, 512, ".jpg")
    assert name == "take.01_result_Fast_512x512.jpg"


def test_output_name_falls_back_to_the_model_key():
    name = batch.output_name("a.png", "NoSuchModel", 8, 8, ".jpg")
    assert name == "a_result_NoSuchModel_8x8.jpg"


def test_batch_labels_cover_every_model():
    assert [batch.BATCH_LABELS[key] for key in batch.BATCH_MODELS] == \
        ["Fast", "Quality"]


def test_batch_models_are_known_upscalers():
    import upscale
    assert all(key in upscale.MODELS for key in batch.BATCH_MODELS)


def test_default_selection_is_quality():
    assert batch.BATCH_DEFAULT_MODEL == "RealPLKSR"
    assert batch.BATCH_LABELS[batch.BATCH_DEFAULT_MODEL] == "Quality"


def test_required_weights_resolve_for_every_batch_model():
    import upscale
    for key in batch.BATCH_MODELS:
        assert upscale.required_weights(key) == [upscale.MODELS[key]["weights"]]


def _write_source_image(tmp_path, size=32):
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), np.full((size, size, 3), 100, dtype=np.uint8))
    return src


def test_image_run_writes_a_4x_result(tmp_path):
    src = _write_source_image(tmp_path)
    args = _args(result_path=str(tmp_path), model_type="RealPLKSR")
    reply = queue.Queue()

    batch._sr_image(Fake4x().eval(), args, str(src), src.name, 0, reply)

    out = tmp_path / "src_result_Quality_128x128.jpg"
    assert out.exists()
    assert cv2.imread(str(out)).shape == (128, 128, 3)


def test_image_run_preserves_uniform_pixel_values(tmp_path):
    src = _write_source_image(tmp_path)
    args = _args(result_path=str(tmp_path), model_type="CompactC3")
    reply = queue.Queue()

    batch._sr_image(Fake4x().eval(), args, str(src), src.name, 0, reply)

    out = cv2.imread(str(tmp_path / "src_result_Fast_128x128.jpg"))
    assert np.abs(out.astype(int) - 100).max() <= 2


def test_image_run_reports_start_and_finish(tmp_path):
    src = _write_source_image(tmp_path)
    args = _args(result_path=str(tmp_path))
    reply = queue.Queue()

    batch._sr_image(Fake4x().eval(), args, str(src), src.name, 3, reply)

    messages = []
    while not reply.empty():
        messages.append(reply.get())
    assert messages == [[3, 0, 1, -1, False], [3, 1, 1, -1, False]]


def test_run_batch_upscale_reports_done_for_an_empty_selection(monkeypatch):
    monkeypatch.setattr(batch, "load_upscaler", lambda model: Fake4x().eval())
    q, reply = queue.Queue(), queue.Queue()
    q.put(_args(input_path=[]))

    batch.run_batch_upscale(q, reply)

    assert reply.get() == [0, 0, 1, -1, True]
