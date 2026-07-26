import logging

import pytest
import torch

from autolume.live.core.superres import MAX_SHORT_SIDE, SuperRes
from utils import resource_paths


class RecordingModel(torch.nn.Module):
    """Stands in for SRVGGNetPlus: records every device it is moved to."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.moves: list[torch.device] = []

    def load_state_dict(self, state_dict, *args, **kwargs):
        return None

    def to(self, device, *args, **kwargs):
        self.moves.append(torch.device(device))
        return self

    def forward(self, x):
        _, c, h, w = x.shape
        return torch.zeros(1, c, h * 4, w * 4)


class MoveFailsModel(torch.nn.Module):
    """`.to()` always raises, like an out-of-memory device move would."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def load_state_dict(self, state_dict, *args, **kwargs):
        return None

    def to(self, device, *args, **kwargs):
        raise RuntimeError("device move boom")

    def forward(self, x):
        raise AssertionError("forward should not run when the move already failed")


class AlwaysFailsForwardModel(torch.nn.Module):
    """The forward pass always raises, like a CUDA OOM would."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def load_state_dict(self, state_dict, *args, **kwargs):
        return None

    def forward(self, x):
        raise RuntimeError("forward pass boom")


class FlakyForwardModel(torch.nn.Module):
    """Fails the first forward call, then succeeds every call after."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calls = 0

    def load_state_dict(self, state_dict, *args, **kwargs):
        return None

    def forward(self, x):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient boom")
        _, c, h, w = x.shape
        return torch.zeros(1, c, h * 4, w * 4)


@pytest.fixture
def install_model(monkeypatch, tmp_path):
    """Factory fixture: install_model(cls) makes a real weight file exist and
    swaps SRVGGNetPlus for the given fake class."""

    def _install(model_cls):
        weight_path = tmp_path / "Fast.pt"
        torch.save({}, weight_path)
        monkeypatch.setattr(resource_paths, "resource_path", lambda *parts: weight_path)
        import super_res.net_base as net_base

        monkeypatch.setattr(net_base, "SRVGGNetPlus", model_cls)
        return weight_path

    return _install


@pytest.fixture
def fake_weights(install_model):
    """A weight file that exists, and a device-move-recording fake network."""
    return install_model(RecordingModel)


def test_apply_moves_only_when_device_changes(fake_weights):
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    sr.apply(image, "cpu")
    sr.apply(image, "cpu")
    assert len(sr._model.moves) == 1

    sr.apply(image, "meta")
    assert len(sr._model.moves) == 2


def test_apply_upscales_using_the_loaded_model(fake_weights):
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    output = sr.apply(image, "cpu")

    assert output.shape == (3, 32, 32)
    assert not sr.disabled


def test_missing_weights_disables_cleanly(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        resource_paths, "resource_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    with caplog.at_level(logging.WARNING):
        first = sr.apply(image, "cpu")
        second = sr.apply(image, "cpu")

    assert first is image
    assert second is image
    assert sr.disabled
    assert sr.disabled_reason is not None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_missing_weights_never_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        resource_paths, "resource_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    for _ in range(3):
        sr.apply(image, "cpu")


def test_corrupt_weights_disable_cleanly_instead_of_raising(monkeypatch, tmp_path):
    weight_path = tmp_path / "Fast.pt"
    weight_path.write_bytes(b"not a real checkpoint")
    monkeypatch.setattr(resource_paths, "resource_path", lambda *parts: weight_path)
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    output = sr.apply(image, "cpu")

    assert output is image
    assert sr.disabled
    assert sr.disabled_reason is not None


def test_short_side_guard_skips_oversized_frames(monkeypatch, caplog):
    def _boom(*parts):
        raise AssertionError("weights should not be loaded when the guard trips")

    monkeypatch.setattr(resource_paths, "resource_path", _boom)
    sr = SuperRes()
    tall_short_side = MAX_SHORT_SIDE + 100
    image = torch.zeros(3, tall_short_side, tall_short_side + 500)

    with caplog.at_level(logging.WARNING):
        first = sr.apply(image, "cpu")
        second = sr.apply(image, "cpu")

    assert first is image
    assert second is image
    assert sr._model is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_guard_boundary_allows_exactly_max_short_side(fake_weights):
    sr = SuperRes()
    image = torch.zeros(3, MAX_SHORT_SIDE, MAX_SHORT_SIDE)

    output = sr.apply(image, "cpu")

    assert output.shape == (3, MAX_SHORT_SIDE * 4, MAX_SHORT_SIDE * 4)


def test_device_move_failure_returns_original_image(install_model):
    install_model(MoveFailsModel)
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    output = sr.apply(image, "cpu")

    assert output is image
    assert not sr.disabled
    assert sr.last_error is not None


def test_forward_failure_returns_original_image(install_model):
    install_model(AlwaysFailsForwardModel)
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    output = sr.apply(image, "cpu")

    assert output is image
    assert not sr.disabled
    assert sr.last_error is not None


def test_forward_failure_logs_once_across_repeated_calls(install_model, caplog):
    install_model(AlwaysFailsForwardModel)
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            output = sr.apply(image, "cpu")
            assert output is image

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert not sr.disabled


def test_forward_failure_recovers_on_next_success(install_model):
    install_model(FlakyForwardModel)
    sr = SuperRes()
    image = torch.zeros(3, 8, 8)

    first = sr.apply(image, "cpu")
    assert first is image
    assert sr.last_error is not None
    assert not sr.disabled

    second = sr.apply(image, "cpu")
    assert second.shape == (3, 32, 32)
    assert sr.last_error is None


_REAL_WEIGHTS = resource_paths.resource_path("sr_models", "Fast.pt")


@pytest.mark.skipif(
    not _REAL_WEIGHTS.exists(), reason="sr_models/Fast.pt not present in this checkout"
)
def test_real_weights_forward_upscales_4x():
    sr = SuperRes()
    image = torch.rand(3, 16, 16)

    output = sr.apply(image, "cpu")

    assert output.shape == (3, 64, 64)
    assert not sr.disabled
