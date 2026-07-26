import time

import numpy as np
import pytest

from autolume.live.core import presets
from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import ModelHost, ModelInfo
from autolume.live.core.params import BINDING_SET, Binding
from autolume.live.core.presets import PRESET_APPLY
from autolume.live.io.osc import OscEmitter
from autolume.live.runtime import build_runtime


class FakeModel:
    def __init__(self, path):
        self.pkl_path = path
        self.z_dim = 4
        self.num_ws = 2

    def render_frame(self, params, frame_index):
        value = int(abs(params.latent_x) * 10) % 256
        return np.full((8, 8, 3), value, dtype=np.uint8)


class FakeAudioEngine:
    """Stands in for AudioEngine, guards included.

    `select_device` ignores a change while enabled exactly as the real engine
    does, so a test that switches devices cannot pass here while doing nothing
    in production.
    """

    def __init__(self):
        self.enabled = False
        self.devices = ((0, "fake mic"), (1, "fake line in"))
        self.device_pos = 0
        self.features = {"level": 0.0}
        self.spectrum = np.zeros(4, dtype=np.float32)
        self.error = None
        self.onset_sensitivity = 0.65
        self.sample_rate = 48000
        self.disabled_count = 0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.disabled_count += 1

    def select_device(self, pos):
        if not self.enabled:
            self.device_pos = pos

    def set_onset_sensitivity(self, value):
        self.onset_sensitivity = value

    def refresh(self):
        pass

    def update(self):
        if self.enabled:
            self.features = {"level": 0.75}


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_end_to_end_headless_flow():
    host = ModelHost(loader=FakeModel)
    runtime = build_runtime(model_host=host, start_osc=False, start_audio=False)
    runtime.start()
    try:
        runtime.submit(ControlEvent("/model/path", "/tmp/fake.pkl", source="ui"))
        assert wait_for(lambda: host.current() is not None)
        assert wait_for(lambda: runtime.preview.latest()[1] is not None)

        runtime.submit(ControlEvent("/latent/x", 5.0, source="ui"))
        assert wait_for(
            lambda: runtime.preview.latest()[1] is not None
            and runtime.preview.latest()[1][0, 0, 0] == 50
        )

        runtime.submit(ControlEvent("/anim/playing", 1.0, source="ui"))
        runtime.submit(ControlEvent("/anim/speed/x", 4.0, source="ui"))
        x0 = runtime.control_store.snapshot().latent_x
        time.sleep(0.3)
        assert runtime.control_store.snapshot().latent_x > x0
    finally:
        runtime.stop()


def test_runtime_exposes_model_info_store_and_the_control_loop_sees_it():
    host = ModelHost(loader=FakeModel)
    runtime = build_runtime(model_host=host, start_osc=False, start_audio=False)
    assert runtime.model_info_store is host.info_store
    assert runtime.control_loop.model_info is None
    runtime.start()
    try:
        runtime.submit(ControlEvent("/model/path", "/tmp/fake.pkl", source="ui"))
        expected = ModelInfo(pkl_path="/tmp/fake.pkl", z_dim=4, num_ws=2)
        assert wait_for(lambda: runtime.model_info_store.snapshot() == expected)
        assert wait_for(lambda: runtime.control_loop.model_info == expected)
    finally:
        runtime.stop()


def test_stop_is_clean_and_idempotent():
    runtime = build_runtime(
        model_host=ModelHost(loader=FakeModel), start_osc=False, start_audio=False
    )
    runtime.start()
    runtime.stop()
    runtime.stop()


def make_runtime(**kwargs):
    kwargs.setdefault("model_host", ModelHost(loader=FakeModel))
    kwargs.setdefault("start_osc", False)
    kwargs.setdefault("start_audio", False)
    return build_runtime(**kwargs)


def test_source_store_fills_as_events_arrive():
    runtime = make_runtime()
    runtime.start()
    try:
        assert runtime.source_store.snapshot().entries == {}

        runtime.submit(ControlEvent("/knob/one", 0.5))

        assert wait_for(
            lambda: runtime.source_store.snapshot().get("/knob/one") is not None
        )
        assert runtime.source_store.snapshot().get("/knob/one").value == 0.5
        assert "/knob/one" in runtime.source_store.snapshot().recent(time.monotonic())
    finally:
        runtime.stop()


def test_binding_submitted_through_the_runtime_drives_its_target():
    runtime = make_runtime()
    runtime.start()
    try:
        runtime.submit(
            ControlEvent(
                BINDING_SET,
                Binding("truncation_psi", "/knob/one", "x * 2"),
                source="ui",
            )
        )
        assert wait_for(lambda: runtime.control_store.snapshot().bindings != ())

        runtime.submit(ControlEvent("/knob/one", 0.25))

        assert wait_for(
            lambda: runtime.control_store.snapshot().truncation_psi == 0.5
        )
        assert runtime.render_store.snapshot().truncation_psi == 0.5
    finally:
        runtime.stop()


def test_audio_runs_with_the_runtime_and_reaches_the_source_table():
    engine = FakeAudioEngine()
    runtime = make_runtime(start_audio=True, audio_engine=engine)
    runtime.start()
    try:
        runtime.audio.enable()

        assert wait_for(lambda: runtime.audio.status().enabled)
        assert wait_for(
            lambda: runtime.source_store.snapshot().get("/audio/level") is not None
        )
        assert runtime.source_store.snapshot().get("/audio/level").value == 0.75
    finally:
        runtime.stop()
    assert engine.enabled is False
    assert engine.disabled_count >= 1


def test_preset_saved_from_a_running_runtime_is_restored_by_apply(tmp_path):
    runtime = make_runtime()
    runtime.start()
    try:
        runtime.submit(ControlEvent("/trunc/psi", 1.25, source="ui"))
        runtime.submit(
            ControlEvent(
                BINDING_SET,
                Binding("global_noise", "/knob/two", "x / 2"),
                source="ui",
            )
        )
        assert wait_for(
            lambda: runtime.control_store.snapshot().truncation_psi == 1.25
            and runtime.control_store.snapshot().bindings != ()
        )

        path = tmp_path / "look.json"
        presets.save(runtime.control_store.snapshot(), path)
        assert presets.list_presets(tmp_path) == ["look"]

        runtime.submit(ControlEvent("/trunc/psi", 0.0, source="ui"))
        assert wait_for(lambda: runtime.control_store.snapshot().truncation_psi == 0.0)

        runtime.submit(
            ControlEvent(PRESET_APPLY, presets.load(path), source="ui")
        )

        assert wait_for(lambda: runtime.control_store.snapshot().truncation_psi == 1.25)
        state = runtime.control_store.snapshot()
        assert [b.target for b in state.bindings] == ["global_noise"]
        assert state.bindings[0].expression == "x / 2"
    finally:
        runtime.stop()


class FailingOsc:
    """An OSC transport that cannot bind, the way a taken port range behaves."""

    port = None

    def __init__(self):
        self.stopped = 0

    def start(self):
        raise OSError("No OSC port available in 1338-1357")

    def stop(self):
        self.stopped += 1


def test_a_failed_osc_start_does_not_leave_the_audio_thread_running():
    engine = FakeAudioEngine()
    runtime = make_runtime(start_audio=True, audio_engine=engine, start_osc=True)
    runtime.osc = FailingOsc()

    with pytest.raises(OSError):
        runtime.start()

    # The engine is only disabled once the audio thread has been joined, so a
    # disabled engine is what proves the thread went with it and released the
    # device.
    assert engine.disabled_count == 1
    assert runtime.audio._thread is None
    # The failed start left nothing running, so stopping again is still a no-op.
    runtime.stop()
    assert engine.disabled_count == 1


def test_runtime_wires_a_real_osc_emitter_by_default():
    """Nothing here ever sends: `pulse_address` stays empty, so `OscEmitter`
    is constructed but its lazy client never is, and no socket opens.
    """
    runtime = make_runtime()
    assert isinstance(runtime.control_loop._emit.__self__, OscEmitter)


def test_runtime_emits_a_pulse_through_an_injected_emit():
    emitted = []
    runtime = make_runtime(
        emit=lambda ip, port, address, value: emitted.append(
            (ip, port, address, value)
        )
    )
    runtime.start()
    try:
        runtime.submit(
            ControlEvent("/loop/pulse/address", "/pulse", source="ui")
        )
        runtime.submit(ControlEvent("/loop/anim", 1.0, source="ui"))
        assert wait_for(lambda: emitted != [])
        assert emitted[0] == ("127.0.0.1", 5005, "/pulse", 2.0)
    finally:
        runtime.stop()


def test_stop_is_clean_and_idempotent_with_audio_running():
    engine = FakeAudioEngine()
    runtime = make_runtime(start_audio=True, audio_engine=engine)
    runtime.start()
    runtime.start()
    runtime.audio.enable()
    assert wait_for(lambda: runtime.audio.status().enabled)
    runtime.stop()
    runtime.stop()
    assert engine.enabled is False
    assert engine.disabled_count == 1
