"""The NDI output sink.

`NDIlib` is faked wholesale: the point under test is the thread lifecycle,
the latest-frame mailbox and what happens when a send fails, none of which
needs a real NDI runtime (and none of which could be driven deterministically
against one).
"""

import threading
import time
import types

import numpy as np
import pytest

from autolume.live.io import ndi as ndi_module
from autolume.live.io.ndi import NdiSink, available


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def frame(value=1, width=4, height=2):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (value, value + 1, value + 2)
    image.flags.writeable = False
    return image


class FakeSender:
    def __init__(self, name):
        self.name = name
        self.sent = []
        self.destroyed = False


class FakeNDIlib:
    """Stands in for the `NDIlib` module, one instance per test."""

    FOURCC_VIDEO_TYPE_BGRX = "bgrx"

    def __init__(self):
        self.senders = []
        self.initialized = 0
        self.initialize_result = True
        self.create_result = True
        self.send_gate = None
        self.send_errors = []

    def initialize(self):
        self.initialized += 1
        return self.initialize_result

    def SendCreate(self):
        return types.SimpleNamespace(ndi_name="")

    def send_create(self, settings):
        if not self.create_result:
            return None
        sender = FakeSender(settings.ndi_name)
        self.senders.append(sender)
        return sender

    def VideoFrameV2(self):
        return types.SimpleNamespace(data=None, FourCC=None)

    def send_send_video_v2(self, sender, video):
        if self.send_gate is not None:
            self.send_gate.wait(3.0)
        if self.send_errors:
            raise self.send_errors.pop(0)
        sender.sent.append((np.array(video.data, copy=True), video.FourCC))

    def send_destroy(self, sender):
        sender.destroyed = True


@pytest.fixture
def ndilib(monkeypatch):
    fake = FakeNDIlib()
    monkeypatch.setattr(ndi_module, "NDIlib", fake)
    yield fake
    if fake.send_gate is not None:
        fake.send_gate.set()


@pytest.fixture
def sink():
    made = NdiSink()
    yield made
    made.stop()


# --- the library may simply not be there ----------------------------------


def test_available_follows_the_library(monkeypatch, ndilib):
    assert available() is True
    monkeypatch.setattr(ndi_module, "NDIlib", None)
    assert available() is False


def test_starting_without_the_library_reports_instead_of_raising(monkeypatch, sink):
    monkeypatch.setattr(ndi_module, "NDIlib", None)
    before = threading.active_count()
    sink.start("Autolume Live")
    status = sink.status()
    assert status.sending is False
    assert status.error is not None
    assert threading.active_count() == before


# --- lifecycle ------------------------------------------------------------


def test_enable_creates_a_sender_and_sends_converted_frames(ndilib, sink):
    sink.start("Autolume Live")
    assert sink.status().sending is True
    sink.on_frame(frame(10), 0)
    assert wait_for(lambda: ndilib.senders and ndilib.senders[0].sent)

    sender = ndilib.senders[0]
    assert sender.name == "Autolume Live"
    assert ndilib.initialized == 1
    data, fourcc = sender.sent[0]
    assert fourcc == FakeNDIlib.FOURCC_VIDEO_TYPE_BGRX
    # RGB (10, 11, 12) arrives as BGRX, converted on the sink thread.
    assert data.shape == (2, 4, 4)
    assert data[0, 0].tolist() == [12, 11, 10, 255]


def test_the_sender_exists_before_any_frame_arrives(ndilib, sink):
    """A viewer must be able to find the source on an idle machine."""
    sink.start("Autolume Live")
    assert wait_for(lambda: len(ndilib.senders) == 1)
    assert ndilib.senders[0].sent == []


def test_disable_destroys_the_sender_and_stops_the_thread(ndilib, sink):
    before = threading.active_count()
    sink.start("Autolume Live")
    assert wait_for(lambda: len(ndilib.senders) == 1)
    sink.stop()
    assert ndilib.senders[0].destroyed is True
    assert sink.status().sending is False
    assert wait_for(lambda: threading.active_count() == before)


def test_frames_outside_a_session_are_ignored(ndilib, sink):
    sink.on_frame(frame(1), 0)
    sink.start("Autolume Live")
    assert wait_for(lambda: len(ndilib.senders) == 1)
    sink.stop()
    sink.on_frame(frame(2), 1)
    time.sleep(0.05)
    assert ndilib.senders[0].sent == []


def test_a_name_change_recreates_the_sender(ndilib, sink):
    sink.start("Autolume Live")
    sink.on_frame(frame(1), 0)
    assert wait_for(lambda: ndilib.senders and ndilib.senders[0].sent)

    sink.set_name("Second Stage")
    sink.on_frame(frame(2), 1)
    assert wait_for(lambda: len(ndilib.senders) == 2 and ndilib.senders[1].sent)

    assert ndilib.senders[0].destroyed is True
    assert ndilib.senders[1].name == "Second Stage"
    assert sink.status().name == "Second Stage"
    assert sink.status().sending is True


def test_a_name_change_while_disabled_does_nothing(ndilib, sink):
    sink.set_name("Nobody Is Listening")
    time.sleep(0.05)
    assert ndilib.senders == []
    assert sink.status().sending is False


def test_stop_can_be_asked_not_to_wait_at_all(ndilib, sink):
    """What the control thread calls: it may never block on a send."""
    ndilib.send_gate = threading.Event()
    sink.start("Autolume Live")
    sink.on_frame(frame(1), 0)
    assert wait_for(lambda: len(ndilib.senders) == 1)

    started = time.monotonic()
    sink.stop(timeout=0.0)
    assert time.monotonic() - started < 0.5
    assert sink.status().sending is False
    ndilib.send_gate.set()
    assert wait_for(lambda: ndilib.senders[0].destroyed)


def test_a_restart_while_the_previous_thread_is_still_going_is_refused(ndilib, sink):
    ndilib.send_gate = threading.Event()
    sink.start("Autolume Live")
    sink.on_frame(frame(1), 0)
    assert wait_for(lambda: len(ndilib.senders) == 1)
    sink.stop(timeout=0.0)

    sink.start("Autolume Live")
    assert sink.status().sending is False
    assert sink.status().error is not None
    ndilib.send_gate.set()
    assert wait_for(lambda: ndilib.senders[0].destroyed)
    assert len(ndilib.senders) == 1


# --- failures -------------------------------------------------------------


def test_a_send_failure_disables_the_sink_without_raising(ndilib, sink):
    ndilib.send_errors = [RuntimeError("ndi send exploded")]
    sink.start("Autolume Live")
    sink.on_frame(frame(1), 0)
    assert wait_for(lambda: sink.status().sending is False)

    status = sink.status()
    assert "ndi send exploded" in status.error
    assert ndilib.senders[0].destroyed is True
    # The render thread keeps handing frames over and keeps not caring.
    sink.on_frame(frame(2), 1)
    assert ndilib.senders[0].sent == []


def test_a_sender_that_cannot_be_created_reports_and_stops(ndilib, sink):
    ndilib.create_result = False
    sink.start("Autolume Live")
    assert wait_for(lambda: sink.status().sending is False)
    assert sink.status().error is not None


def test_a_runtime_that_will_not_initialise_reports_and_stops(ndilib, sink):
    ndilib.initialize_result = False
    sink.start("Autolume Live")
    assert wait_for(lambda: sink.status().sending is False)
    assert sink.status().error is not None
    assert ndilib.senders == []


def test_send_failures_log_once_per_cause_not_once_per_number(ndilib, sink, caplog):
    """The dedup key normalises digits, so a counter in the message is one cause.

    A key built from the raw message degrades to a log line per failure the
    moment the driver puts a frame number or a byte count in it.
    """
    with caplog.at_level("WARNING"):
        for index in range(3):
            ndilib.send_errors = [RuntimeError(f"send failed on frame {index * 17}")]
            sink.start("Autolume Live")
            sink.on_frame(frame(index), index)
            assert wait_for(lambda: sink.status().sending is False)
            sink.stop()
    failures = [
        record for record in caplog.records if "send failed on frame" in record.message
    ]
    assert len(failures) == 1


def test_a_new_cause_is_logged_again(ndilib, sink, caplog):
    with caplog.at_level("WARNING"):
        ndilib.send_errors = [RuntimeError("first cause")]
        sink.start("Autolume Live")
        sink.on_frame(frame(1), 0)
        assert wait_for(lambda: sink.status().sending is False)
        sink.stop()

        ndilib.send_errors = [ValueError("second cause")]
        sink.start("Autolume Live")
        sink.on_frame(frame(2), 1)
        assert wait_for(lambda: sink.status().sending is False)
    assert any("first cause" in record.message for record in caplog.records)
    assert any("second cause" in record.message for record in caplog.records)


# --- the render thread never waits ----------------------------------------


def test_the_render_thread_never_waits_on_a_send(ndilib, sink):
    """The mailbox holds one frame, so a slow receiver costs frames, not fps.

    The old app sent from the UI thread, which put the whole show behind
    whatever the network was doing.
    """
    ndilib.send_gate = threading.Event()
    sink.start("Autolume Live")
    assert wait_for(lambda: len(ndilib.senders) == 1)
    sink.on_frame(frame(1), 0)
    time.sleep(0.05)

    started = time.monotonic()
    for value in range(2, 40):
        sink.on_frame(frame(value), value)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5

    ndilib.send_gate.set()
    sender = ndilib.senders[0]
    assert wait_for(lambda: len(sender.sent) >= 2)
    time.sleep(0.05)
    # The first frame was in flight, then everything queued behind it
    # collapsed to the newest one: no backlog was kept anywhere.
    assert len(sender.sent) <= 3
    assert sender.sent[-1][0][0, 0, 2] == 39
