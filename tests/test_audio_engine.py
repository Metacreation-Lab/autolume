import numpy as np
import pytest
from pythonosc.osc_message import OscMessage

import audio.engine as engine_mod
from audio.engine import AudioEngine
from audio.features import FEATURE_NAMES


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


class FakeDispatcher:
    """Records the OSC messages the publisher loops back."""

    def __init__(self):
        self.messages = []

    def call_handlers_for_packet(self, dgram, addr):
        msg = OscMessage(dgram)
        self.messages.append((msg.address, msg.params[0]))


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_input_devices", lambda: [(3, "Fake Device")])
    monkeypatch.setattr(engine_mod, "AudioStream", FakeStream)


def test_engine_starts_disabled(patched):
    eng = AudioEngine(FakeDispatcher())
    assert not eng.enabled
    assert eng.devices == [(3, "Fake Device")]
    assert eng.sample_rate == 0


def test_enable_opens_stream(patched):
    eng = AudioEngine(FakeDispatcher())
    eng.enable()
    assert eng.enabled
    assert eng.sample_rate == 44100
    assert eng.error is None


def test_update_publishes_all_features(patched):
    disp = FakeDispatcher()
    eng = AudioEngine(disp)
    eng.enable()
    eng.update()
    published = {addr for addr, _ in disp.messages}
    assert published == {f"/audio/{name}" for name in FEATURE_NAMES}
    assert all(isinstance(value, float) for _, value in disp.messages)


def test_update_while_disabled_publishes_nothing(patched):
    disp = FakeDispatcher()
    eng = AudioEngine(disp)
    eng.update()
    assert disp.messages == []


def test_inactive_stream_self_disables(patched):
    eng = AudioEngine(FakeDispatcher())
    eng.enable()
    eng.stream.close()  # active -> False
    eng.update()
    assert not eng.enabled
    assert "stopped" in eng.error


def test_disable_clears_features(patched):
    eng = AudioEngine(FakeDispatcher())
    eng.enable()
    eng.update()
    eng.disable()
    assert not eng.enabled
    assert all(value == 0.0 for value in eng.features.values())
