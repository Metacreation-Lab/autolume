import time

import numpy as np
import pytest

from autolume.live.core import presets
from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import DeviceStatus, ModelHost, ModelInfo
from autolume.live.core.params import (
    BINDING_SET,
    Binding,
    ControlState,
    to_render_params,
)
from autolume.live.core.presets import PRESET_APPLY
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.io.osc import OscEmitter
from autolume.live.runtime import OscStatus, _ModelWatchingControlLoop, build_runtime


class DeviceAwareFakeModel:
    """Like FakeModel, but records the device it was built with, the way a
    real LoadedModel does, so device-switch tests can check it."""

    def __init__(self, path, device=None):
        self.pkl_path = path
        self.z_dim = 4
        self.num_ws = 2
        self.device = device

    def render_frame(self, params, frame_index):
        return np.zeros((8, 8, 3), dtype=np.uint8)


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


# --- device switching ------------------------------------------------------


def test_device_change_reloads_the_current_pkl_on_the_new_device():
    import torch

    host = ModelHost(loader=DeviceAwareFakeModel)
    runtime = make_runtime(model_host=host)
    runtime.start()
    try:
        runtime.submit(ControlEvent("/model/path", "/tmp/fake.pkl", source="ui"))
        assert wait_for(lambda: host.current() is not None)

        runtime.submit(ControlEvent("/render/device", "cpu", source="ui"))
        assert wait_for(
            lambda: host.current() is not None
            and host.current().device == torch.device("cpu")
        )
        assert wait_for(lambda: runtime.control_store.snapshot().device == "cpu")
        assert host.device_store.snapshot().error is None
    finally:
        runtime.stop()


def test_device_change_to_an_unavailable_device_reverts_to_the_previous_value():
    host = ModelHost(loader=DeviceAwareFakeModel)
    runtime = make_runtime(model_host=host)
    runtime.start()
    try:
        runtime.submit(ControlEvent("/model/path", "/tmp/fake.pkl", source="ui"))
        assert wait_for(lambda: host.current() is not None)

        runtime.submit(ControlEvent("/render/device", "cpu", source="ui"))
        assert wait_for(lambda: runtime.control_store.snapshot().device == "cpu")
        running = host.current()

        # This machine never has CUDA, so this is a genuine failure, not a
        # mocked one. The revert must land back on "cpu", the value that was
        # actually working, not on the registry's "auto" default: that is
        # the only way this test can tell "reverted" from "never changed".
        runtime.submit(ControlEvent("/render/device", "cuda", source="ui"))
        assert wait_for(lambda: host.device_store.snapshot().error is not None)
        assert wait_for(lambda: runtime.control_store.snapshot().device == "cpu")
        assert host.current() is running
    finally:
        runtime.stop()


class FakeDeviceHost:
    """Enough of ModelHost's surface for `_ModelWatchingControlLoop` to
    drive, with `device_store` publishable by hand so a test can control
    the interleaving deterministically instead of racing real threads."""

    def __init__(self):
        self.device_store = LatestValueStore(DeviceStatus())
        self.requests = []

    def request_device(self, name):
        self.requests.append(name)

    def request_load(self, path):
        pass


def test_two_rapid_device_changes_revert_to_the_last_confirmed_value():
    """Regression: the fallback used to be whatever the previous *request*
    was, not a value a status had actually confirmed working. Two switches
    close enough together that the first request's status never lands
    (superseded before the loader thread gets to it, the way the reviewer's
    "cuda" then "tpu" repro triggered it 7 times in 12) used to make that
    unvalidated first request the permanent revert target instead of the
    device that was actually running.

    Driven directly through `_ModelWatchingControlLoop.tick()` with a fake
    host, so the interleaving is exact rather than left to real thread
    timing.
    """
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    host = FakeDeviceHost()
    loop = _ModelWatchingControlLoop(control_store, render_store, source_store, host)
    # A tick with nothing queued, standing in for however many ticks a real
    # session runs before anyone ever touches /render/device (a model is
    # already loaded and settled long before a performer reaches for this
    # control). Isolates this test from the separate first-tick-adoption
    # fix: what is under test here is the fallback value a *later* failure
    # reverts to, not whether the very first tick forwards a change.
    loop.tick()

    loop.submit(ControlEvent("/render/device", "cuda", source="ui"))
    loop.tick()
    assert host.requests == ["cuda"]
    # "cuda"'s status never arrives: it is superseded before the (fake)
    # loader thread would ever get to it, exactly as the flaky real-thread
    # repro sometimes did.

    loop.submit(ControlEvent("/render/device", "tpu", source="ui"))
    loop.tick()
    assert host.requests == ["cuda", "tpu"]

    host.device_store.set(
        DeviceStatus(active=None, requested="tpu", error="no such device")
    )
    # One tick to notice the failure and queue the revert control event,
    # one more to actually apply it: `submit` only queues, `tick` drains
    # the queue at its own start, before `_watch_device` runs.
    loop.tick()
    loop.tick()

    assert control_store.snapshot().device == "auto"


# --- OSC port restart --------------------------------------------------


class FakeOscTransport:
    """A stand-in transport whose start/stop are observable and whose
    failure is controllable, so a restart test never opens a real socket."""

    def __init__(self, port, fail=False):
        self.requested_port = port
        self.fail = fail
        self.started = 0
        self.stopped = 0
        self.port = None

    def start(self):
        self.started += 1
        if self.fail:
            raise OSError(f"port {self.requested_port} is taken")
        # Simulates the scan-upward behavior: the bound port is not always
        # the one requested.
        self.port = self.requested_port + 1
        return self.port

    def stop(self):
        self.stopped += 1


def test_osc_port_change_restarts_the_transport_once():
    transports = []

    def factory(port):
        transport = FakeOscTransport(port)
        transports.append(transport)
        return transport

    runtime = make_runtime(start_osc=True, osc_factory=factory)
    runtime.start()
    try:
        assert len(transports) == 1
        first = transports[0]
        assert first.started == 1
        assert runtime.osc_status_store.snapshot().bound_port == first.port

        runtime.submit(ControlEvent("/osc/port", 6000, source="ui"))
        assert wait_for(lambda: len(transports) == 2)
        second = transports[1]
        assert wait_for(lambda: second.started == 1)
        assert wait_for(lambda: first.stopped == 1)
        assert second.started == 1
        assert runtime.osc is second
        assert runtime.osc_status_store.snapshot() == OscStatus(
            bound_port=second.port, error=None
        )
    finally:
        runtime.stop()


def test_a_failed_osc_rebind_keeps_the_old_transport_serving():
    transports = []

    def factory(port):
        transport = FakeOscTransport(port, fail=(port == 6000))
        transports.append(transport)
        return transport

    runtime = make_runtime(start_osc=True, osc_factory=factory)
    runtime.start()
    try:
        first = transports[0]
        bound_before = runtime.osc_status_store.snapshot().bound_port

        runtime.submit(ControlEvent("/osc/port", 6000, source="ui"))
        assert wait_for(lambda: len(transports) == 2)
        assert wait_for(lambda: runtime.osc_status_store.snapshot().error is not None)

        assert runtime.osc is first
        assert first.stopped == 0
        status = runtime.osc_status_store.snapshot()
        assert status.bound_port == bound_before
        assert status.error is not None
    finally:
        runtime.stop()
