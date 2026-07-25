import numpy as np
import pytest

import autolume.audio.engine as engine_mod
from autolume.audio.engine import AudioEngine
from autolume.audio.features import FEATURE_NAMES


class FakeStream:
    """Stand-in for AudioStream that needs no hardware."""

    def __init__(self, device, window_size=2048):
        self.device_name = "Fake Device"
        self.sample_rate = 44100
        self._active = True

    def read(self):
        rng = np.random.default_rng(0)
        return rng.uniform(-0.5, 0.5, 2048).astype(np.float32)

    @property
    def active(self):
        return self._active

    def close(self):
        self._active = False


class Recorder:
    """Records the feature maps the engine publishes."""

    def __init__(self):
        self.published = []

    def __call__(self, features):
        self.published.append(dict(features))


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_input_devices", lambda: [(3, "Fake Device")])
    monkeypatch.setattr(engine_mod, "AudioStream", FakeStream)


def test_engine_starts_disabled(patched):
    eng = AudioEngine(Recorder())
    assert not eng.enabled
    assert eng.devices == [(3, "Fake Device")]
    assert eng.sample_rate == 0


def test_enable_opens_stream(patched):
    eng = AudioEngine(Recorder())
    eng.enable()
    assert eng.enabled
    assert eng.sample_rate == 44100
    assert eng.error is None


def test_update_publishes_all_features(patched):
    recorder = Recorder()
    eng = AudioEngine(recorder)
    eng.enable()
    eng.update()
    assert len(recorder.published) == 1
    features = recorder.published[0]
    assert set(features) == set(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features.values())


def test_update_while_disabled_publishes_nothing(patched):
    recorder = Recorder()
    eng = AudioEngine(recorder)
    eng.update()
    assert recorder.published == []


def test_inactive_stream_self_disables(patched):
    eng = AudioEngine(Recorder())
    eng.enable()
    eng.stream.close()  # active -> False
    eng.update()
    assert not eng.enabled
    assert "stopped" in eng.error


def test_disable_clears_features(patched):
    eng = AudioEngine(Recorder())
    eng.enable()
    eng.update()
    eng.disable()
    assert not eng.enabled
    assert all(value == 0.0 for value in eng.features.values())


def test_refresh_preserves_selected_device(monkeypatch):
    devices = [(1, "Mic A"), (2, "Mic B")]
    monkeypatch.setattr(engine_mod, "refresh_devices", lambda: None)
    monkeypatch.setattr(engine_mod, "list_input_devices", lambda: list(devices))
    eng = AudioEngine(Recorder())
    eng.select_device(1)  # Mic B
    devices[:] = [(5, "Mic B"), (9, "Mic A")]  # re-enumeration reshuffles indices
    eng.refresh()
    assert eng.devices[eng.device_pos][1] == "Mic B"


def test_onset_sensitivity_forwards_and_clamps(patched):
    eng = AudioEngine(Recorder())
    eng.set_onset_sensitivity(0.8)
    eng.enable()
    assert eng.extractor.onset_sensitivity == 0.8
    eng.set_onset_sensitivity(0.2)
    assert eng.extractor.onset_sensitivity == 0.2
    eng.set_onset_sensitivity(5.0)
    assert eng.onset_sensitivity == 1.0
    eng.set_onset_sensitivity(-1.0)
    assert eng.onset_sensitivity == 0.0
