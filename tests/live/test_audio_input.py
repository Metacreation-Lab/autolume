import threading
import time

import numpy as np
import pytest

from autolume.live.io import audio as audio_module
from autolume.live.io.audio import AudioInput, AudioStatus


class FakeEngine:
    """Stands in for AudioEngine with the same surface the thread touches."""

    def __init__(self, enabled=False, features=None):
        self.enabled = enabled
        self.features = features if features is not None else {"level": 0.25, "bass": 0.5}
        self.spectrum = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.error = None
        self.devices = ((3, "mic"), (7, "line in"))
        self.device_pos = 0
        self.onset_sensitivity = 0.65
        self.sample_rate = 48000
        self.calls = []
        self.update_raises = None

    def enable(self):
        self.calls.append(("enable",))
        self.enabled = True

    def disable(self):
        self.calls.append(("disable",))
        self.enabled = False

    def select_device(self, pos):
        self.calls.append(("select_device", pos))
        self.device_pos = pos

    def set_onset_sensitivity(self, value):
        self.calls.append(("set_onset_sensitivity", value))
        self.onset_sensitivity = value

    def refresh(self):
        self.calls.append(("refresh",))

    def update(self):
        self.calls.append(("update",))
        if self.update_raises is not None:
            raise self.update_raises

    def close(self):
        self.disable()


class StuckThread:
    """A thread that never finishes joining, to force the stop() timeout path."""

    def __init__(self):
        self.joins = []

    def join(self, timeout=None):
        self.joins.append(timeout)

    def is_alive(self):
        return True


def make_input(engine, **kwargs):
    events = []
    return AudioInput(events.append, engine=engine, **kwargs), events


def test_disabled_tick_submits_nothing():
    engine = FakeEngine(enabled=False)
    audio, events = make_input(engine)

    assert audio.status().enabled is False
    audio.tick()

    assert events == []
    assert ("update",) in engine.calls


def test_enabled_tick_submits_one_event_per_feature():
    engine = FakeEngine(enabled=True, features={"level": 0.25, "onset": 1.0})
    audio, events = make_input(engine, clock=lambda: 1234.5)

    audio.tick()

    assert [event.address for event in events] == ["/audio/level", "/audio/onset"]
    assert [event.value for event in events] == [0.25, 1.0]
    assert {event.source for event in events} == {"audio"}
    assert [event.timestamp for event in events] == [1234.5, 1234.5]


def test_commands_apply_on_the_tick_not_at_call_time():
    engine = FakeEngine()
    audio, _ = make_input(engine)

    audio.enable()
    audio.select_device(1)
    audio.set_onset_sensitivity(0.25)
    audio.refresh()
    audio.disable()

    assert engine.calls == []

    audio.tick()

    assert engine.calls == [
        ("enable",),
        ("select_device", 1),
        ("set_onset_sensitivity", 0.25),
        ("refresh",),
        ("disable",),
        ("update",),
    ]


def test_status_reflects_the_engine_and_is_a_snapshot():
    engine = FakeEngine(enabled=True)
    engine.error = "Could not open mic"
    audio, _ = make_input(engine)

    audio.tick()
    status = audio.status()

    assert status.enabled is True
    assert status.devices == ((3, "mic"), (7, "line in"))
    assert status.device_pos == 0
    assert status.features == {"level": 0.25, "bass": 0.5}
    assert np.array_equal(status.spectrum, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert status.error == "Could not open mic"
    assert status.onset_sensitivity == 0.65
    assert status.sample_rate == 48000

    engine.features["level"] = 9.0
    engine.spectrum[0] = 9.0
    engine.enabled = False

    assert status.features["level"] == 0.25
    assert status.spectrum[0] == 1.0
    assert status.enabled is True
    with pytest.raises(TypeError):
        status.features["level"] = 9.0


def test_status_can_be_compared_and_hashed():
    engine = FakeEngine(enabled=True)
    audio, _ = make_input(engine)
    audio.tick()
    status = audio.status()

    # A generated __eq__ raises on the ndarray field and a generated __hash__
    # raises on the mapping, so both of these are about not raising.
    assert status == status
    assert status != AudioStatus()
    assert hash(status) == hash(status)


def test_update_error_surfaces_and_the_loop_continues():
    engine = FakeEngine(enabled=True)
    engine.update_raises = RuntimeError("device exploded")
    audio, events = make_input(engine)

    audio.tick()

    assert events == []
    assert "device exploded" in audio.status().error

    engine.update_raises = None
    audio.tick()

    assert len(events) == len(engine.features)
    assert audio.status().error is None


def test_start_and_stop_are_idempotent_and_release_the_device():
    engine = FakeEngine(enabled=True)
    audio, _ = make_input(engine)

    audio.start()
    audio.start()
    try:
        threads = [thread for thread in threading.enumerate() if thread.name == "audio"]
        # Named so a crash dump or profiler says which thread this is, and a
        # daemon so a stalled device cannot hold the process open at exit.
        assert len(threads) == 1
        assert threads[0].daemon is True
    finally:
        audio.stop()
    audio.stop()

    assert engine.enabled is False
    assert ("disable",) in engine.calls
    assert audio.status().enabled is False


def test_stop_without_start_is_safe():
    engine = FakeEngine()
    audio, _ = make_input(engine)

    audio.stop()

    assert ("disable",) in engine.calls
    assert engine.enabled is False


def test_engine_is_built_lazily_on_the_first_tick(monkeypatch):
    engine = FakeEngine()
    built = []

    def build():
        built.append(engine)
        return engine

    monkeypatch.setattr(audio_module, "_build_default_engine", build)
    audio = AudioInput(lambda event: None)

    assert built == []

    audio.tick()
    audio.tick()

    assert built == [engine]


def test_a_failing_engine_build_is_not_retried(monkeypatch):
    attempts = []

    def build():
        attempts.append(1)
        raise RuntimeError("no portaudio")

    monkeypatch.setattr(audio_module, "_build_default_engine", build)
    audio = AudioInput(lambda event: None)

    audio.tick()
    audio.tick()

    assert len(attempts) == 1
    assert "no portaudio" in audio.status().error


def test_refresh_retries_a_failed_engine_build_and_applies_the_command(monkeypatch):
    engine = FakeEngine()
    attempts = []

    def build():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("no portaudio")
        return engine

    monkeypatch.setattr(audio_module, "_build_default_engine", build)
    audio = AudioInput(lambda event: None)

    audio.tick()
    assert "no portaudio" in audio.status().error

    audio.refresh()
    audio.tick()

    assert len(attempts) == 2
    assert ("refresh",) in engine.calls
    assert audio.status().error is None


def test_a_stuck_thread_is_reported_and_still_blocks_a_restart():
    engine = FakeEngine(enabled=True)
    audio, _ = make_input(engine)
    stuck = StuckThread()
    audio._thread = stuck

    audio.stop()

    assert stuck.joins == [2.0]
    assert "did not stop" in audio.status().error
    # The engine belongs to the live thread, so nothing here may touch it.
    assert ("disable",) not in engine.calls

    audio.start()

    assert audio._thread is stuck


def test_the_thread_survives_a_tick_that_raises_outside_its_own_guard(monkeypatch):
    engine = FakeEngine()
    audio, _ = make_input(engine, rate_hz=200.0)
    calls = []

    def broken_tick():
        calls.append(1)
        raise RuntimeError("guard bypassed")

    monkeypatch.setattr(audio, "tick", broken_tick)

    audio.start()
    try:
        time.sleep(0.2)
    finally:
        audio.stop()

    assert len(calls) > 1


def test_threaded_run_submits_at_about_the_requested_rate():
    engine = FakeEngine(enabled=True, features={"level": 0.5})
    audio, events = make_input(engine, rate_hz=60.0)

    audio.start()
    try:
        time.sleep(0.5)
    finally:
        audio.stop()

    # Loose bound on purpose: 60 Hz for 0.5 s is ~30 ticks, and a loaded
    # machine may deliver far fewer without the thread being broken.
    assert len(events) >= 10
