"""The video recorder and the screenshot worker, both off the render thread.

Every test here stands a fake `cv2.VideoWriter` in for the real one: the codec
is not what is under test, the queue discipline and the thread lifecycle are.
"""

import threading
import time

import cv2
import numpy as np
import pytest

from autolume.live.io.recorder import (
    DEFAULT_FPS,
    QUEUE_CAPACITY,
    Recorder,
    ScreenshotWorker,
    capture_basename,
    capture_path,
)


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def frame(value=1, width=4, height=2):
    """A frame the way the render loop hands one out: RGB uint8, read only."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (value, value + 1, value + 2)
    image.flags.writeable = False
    return image


class FakeWriter:
    """Stands in for `cv2.VideoWriter`, recording what it was asked to write."""

    created = []
    open_gate = None
    write_gate = None
    opened = True
    write_error = None

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        FakeWriter.created.append(self)
        if FakeWriter.open_gate is not None:
            FakeWriter.open_gate.wait(3.0)

    def isOpened(self):
        return FakeWriter.opened

    def write(self, image):
        if FakeWriter.write_gate is not None:
            FakeWriter.write_gate.wait(3.0)
        if FakeWriter.write_error is not None:
            raise FakeWriter.write_error
        self.frames.append(np.array(image, copy=True))

    def release(self):
        self.released = True


@pytest.fixture
def writers(monkeypatch):
    FakeWriter.created = []
    FakeWriter.open_gate = None
    FakeWriter.write_gate = None
    FakeWriter.opened = True
    FakeWriter.write_error = None
    monkeypatch.setattr(cv2, "VideoWriter", FakeWriter)
    yield FakeWriter.created
    # Releases any gate a failing test left closed, so a stuck encoder thread
    # cannot outlive the test that created it.
    for gate in (FakeWriter.open_gate, FakeWriter.write_gate):
        if gate is not None:
            gate.set()


# --- the recorder ---------------------------------------------------------


def test_the_recorder_writes_what_it_drains(writers, tmp_path):
    path = str(tmp_path / "take.mp4")
    recorder = Recorder()
    recorder.start(path, 24)
    try:
        for value in range(5):
            recorder.on_frame(frame(value), value)
        assert wait_for(lambda: writers and len(writers[0].frames) == 5)
    finally:
        recorder.stop()

    writer = writers[0]
    assert writer.path == path
    assert writer.fps == 24
    assert writer.size == (4, 2)
    assert writer.released is True
    status = recorder.status()
    assert (status.recording, status.frames_written, status.frames_dropped) == (
        False,
        5,
        0,
    )


def test_frames_are_converted_from_rgb_to_bgr(writers, tmp_path):
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        recorder.on_frame(frame(10), 0)
        assert wait_for(lambda: writers and writers[0].frames)
    finally:
        recorder.stop()
    assert writers[0].frames[0][0, 0].tolist() == [12, 11, 10]


def test_the_writer_is_created_only_once_the_first_frame_arrives(writers, tmp_path):
    """Size comes from the frame, so there is nothing to open before one."""
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    time.sleep(0.05)
    assert writers == []
    recorder.on_frame(frame(1), 0)
    assert wait_for(lambda: len(writers) == 1)
    recorder.stop()


def test_an_uncapped_render_rate_records_at_the_default_fps(writers, tmp_path):
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 0)
    try:
        recorder.on_frame(frame(1), 0)
        assert wait_for(lambda: len(writers) == 1)
    finally:
        recorder.stop()
    assert writers[0].fps == DEFAULT_FPS


def test_frames_before_start_and_after_stop_are_ignored(writers, tmp_path):
    recorder = Recorder()
    recorder.on_frame(frame(1), 0)
    recorder.start(str(tmp_path / "take.mp4"), 30)
    recorder.on_frame(frame(2), 1)
    assert wait_for(lambda: writers and len(writers[0].frames) == 1)
    recorder.stop()
    recorder.on_frame(frame(3), 2)
    time.sleep(0.05)
    assert len(writers[0].frames) == 1


def test_frames_beyond_the_queue_drop_the_oldest_and_are_counted(writers, tmp_path):
    """The queue is bounded, so a slow encoder loses frames, not memory.

    The writer is held closed in its constructor, so exactly one frame has
    left the queue when the rest arrive. That makes the arithmetic exact:
    everything past the capacity is a drop, and what survives is the newest
    tail, not the oldest head.
    """
    FakeWriter.open_gate = threading.Event()
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    total = QUEUE_CAPACITY + 60
    try:
        recorder.on_frame(frame(0), 0)
        assert wait_for(lambda: len(writers) == 1)
        for value in range(1, total):
            recorder.on_frame(frame(value), value)
        assert recorder.status().frames_dropped == total - 1 - QUEUE_CAPACITY
        FakeWriter.open_gate.set()
        assert wait_for(lambda: len(writers[0].frames) == QUEUE_CAPACITY + 1)
    finally:
        FakeWriter.open_gate.set()
        recorder.stop()

    # Channel 2 of a written (BGR) frame is the red channel the fake frame put
    # its value in. The first frame got through before the queue filled;
    # everything after it is the tail of the run, which is what "drop the
    # oldest" means.
    written = writers[0].frames
    assert written[0][0, 0, 2] == 0
    assert written[1][0, 0, 2] == total - QUEUE_CAPACITY
    assert written[-1][0, 0, 2] == total - 1
    status = recorder.status()
    assert status.frames_written == QUEUE_CAPACITY + 1
    assert status.frames_dropped == total - 1 - QUEUE_CAPACITY


def test_stop_is_idempotent_and_bounded_in_time(writers, tmp_path):
    """Stop never becomes the app's problem, however stuck the encoder is.

    The old app started the encoder at Stop and joined it on the UI thread,
    so quitting a long take froze the window. Here the encoder is already
    running, and a stop that cannot complete gives up on its own deadline
    instead of holding the caller.
    """
    FakeWriter.write_gate = threading.Event()
    recorder = Recorder(stop_timeout=0.2)
    recorder.start(str(tmp_path / "take.mp4"), 30)
    for value in range(5):
        recorder.on_frame(frame(value), value)
    assert wait_for(lambda: len(writers) == 1)

    started = time.monotonic()
    recorder.stop()
    first = time.monotonic() - started
    started = time.monotonic()
    recorder.stop()
    second = time.monotonic() - started
    FakeWriter.write_gate.set()

    assert first < 1.0
    assert second < 1.0


def test_stop_can_be_asked_not_to_wait_at_all(writers, tmp_path):
    """What the control thread calls: signal the encoder, never join on it.

    The tail of a take can be a hundred frames of mp4 encoding. The control
    thread carries the whole show's heartbeat and cannot wait for that, so it
    hands the flush to the encoder thread and moves on.
    """
    FakeWriter.write_gate = threading.Event()
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    recorder.on_frame(frame(1), 0)
    assert wait_for(lambda: len(writers) == 1)

    started = time.monotonic()
    recorder.stop(timeout=0.0)
    assert time.monotonic() - started < 0.5
    FakeWriter.write_gate.set()
    assert wait_for(lambda: writers[0].released)


def test_a_resolution_change_stops_the_take(writers, tmp_path, caplog):
    """VideoWriter cannot change size, so the take ends rather than corrupts.

    Toggling super-res or switching model mid recording changes the frame
    size. Writing the new size into a file opened for the old one produces a
    file that plays as garbage, with nothing anywhere saying why.
    """
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        recorder.on_frame(frame(1, width=4, height=2), 0)
        assert wait_for(lambda: writers and len(writers[0].frames) == 1)
        with caplog.at_level("WARNING"):
            recorder.on_frame(frame(2, width=8, height=4), 1)
            assert wait_for(lambda: recorder.status().recording is False)
        recorder.on_frame(frame(3, width=8, height=4), 2)
        time.sleep(0.05)
    finally:
        recorder.stop()

    assert len(writers) == 1
    assert len(writers[0].frames) == 1
    assert writers[0].released is True
    status = recorder.status()
    assert status.frames_written == 1
    assert status.error is not None
    assert "size" in status.error.lower()
    assert any("size" in record.message.lower() for record in caplog.records)


def test_a_writer_that_will_not_open_ends_the_take(writers, tmp_path):
    FakeWriter.opened = False
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        recorder.on_frame(frame(1), 0)
        assert wait_for(lambda: recorder.status().recording is False)
    finally:
        recorder.stop()
    assert recorder.status().error is not None
    assert writers[0].frames == []


def test_a_write_failure_ends_the_take_without_raising(writers, tmp_path):
    FakeWriter.write_error = RuntimeError("disk went away")
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        recorder.on_frame(frame(1), 0)
        assert wait_for(lambda: recorder.status().recording is False)
    finally:
        recorder.stop()
    assert "disk went away" in recorder.status().error


def test_the_end_of_a_take_logs_what_was_written_and_dropped(writers, tmp_path, caplog):
    recorder = Recorder()
    with caplog.at_level("INFO"):
        recorder.start(str(tmp_path / "take.mp4"), 30)
        recorder.on_frame(frame(1), 0)
        recorder.on_frame(frame(2), 1)
        assert wait_for(lambda: writers and len(writers[0].frames) == 2)
        recorder.stop()
    assert any(
        "2" in record.getMessage() and "dropped" in record.getMessage().lower()
        for record in caplog.records
    )


def test_a_second_start_while_the_first_is_still_flushing_is_refused(writers, tmp_path):
    """Two encoders on one queue would interleave two takes into one file."""
    FakeWriter.write_gate = threading.Event()
    recorder = Recorder()
    recorder.start(str(tmp_path / "one.mp4"), 30)
    recorder.on_frame(frame(1), 0)
    assert wait_for(lambda: len(writers) == 1)
    recorder.stop(timeout=0.0)

    recorder.start(str(tmp_path / "two.mp4"), 30)
    assert recorder.status().recording is False
    assert recorder.status().error is not None
    FakeWriter.write_gate.set()
    assert wait_for(lambda: writers[0].released)
    assert len(writers) == 1


def test_a_take_can_be_started_again_once_the_previous_one_finished(writers, tmp_path):
    recorder = Recorder()
    recorder.start(str(tmp_path / "one.mp4"), 30)
    recorder.on_frame(frame(1), 0)
    assert wait_for(lambda: writers and len(writers[0].frames) == 1)
    recorder.stop()

    recorder.start(str(tmp_path / "two.mp4"), 30)
    try:
        recorder.on_frame(frame(2), 0)
        assert wait_for(lambda: len(writers) == 2 and len(writers[1].frames) == 1)
    finally:
        recorder.stop()
    assert recorder.status().frames_written == 1


def test_stopping_a_recorder_that_never_started_is_a_no_op(writers):
    recorder = Recorder()
    recorder.stop()
    assert recorder.status().recording is False


# --- the screenshot worker ------------------------------------------------


def test_save_png_writes_off_the_calling_thread(tmp_path):
    worker = ScreenshotWorker()
    path = tmp_path / "shot.png"
    caller = threading.current_thread()
    worker.save_png(str(path), frame(9))
    try:
        assert wait_for(path.exists)
    finally:
        worker.stop()
    written = cv2.imread(str(path))
    assert written is not None
    assert written[0, 0].tolist() == [11, 10, 9]
    assert threading.current_thread() is caller


def test_save_png_creates_the_folder_it_writes_into(tmp_path):
    worker = ScreenshotWorker()
    path = tmp_path / "captures" / "shot.png"
    worker.save_png(str(path), frame(1))
    try:
        assert wait_for(path.exists)
    finally:
        worker.stop()


def test_a_failed_screenshot_never_reaches_the_caller(tmp_path, caplog):
    worker = ScreenshotWorker()
    with caplog.at_level("WARNING"):
        worker.save_png(str(tmp_path / "nope" / "\0bad.png"), frame(1))
        time.sleep(0.1)
        worker.stop()
    assert any("screenshot" in record.getMessage().lower() for record in caplog.records)


def test_stop_flushes_what_is_still_queued(tmp_path):
    worker = ScreenshotWorker()
    paths = [tmp_path / f"shot{index}.png" for index in range(4)]
    for index, path in enumerate(paths):
        worker.save_png(str(path), frame(index))
    worker.stop()
    assert all(path.exists() for path in paths)


def test_a_request_after_stop_is_ignored(tmp_path):
    worker = ScreenshotWorker()
    worker.save_png(str(tmp_path / "first.png"), frame(1))
    worker.stop()
    late = tmp_path / "late.png"
    worker.save_png(str(late), frame(2))
    time.sleep(0.1)
    assert late.exists() is False


def test_no_thread_is_started_until_something_is_captured():
    worker = ScreenshotWorker()
    before = threading.active_count()
    worker.stop()
    assert threading.active_count() == before


# --- capture naming -------------------------------------------------------


def test_capture_basename_uses_the_model_name_and_a_timestamp():
    name = capture_basename("/models/My Model.pkl")
    assert name.startswith("My-Model_")
    assert len(name) == len("My-Model_") + len("2026-07-26_12-00-00")


def test_capture_basename_without_a_model_falls_back():
    assert capture_basename(None).startswith("autolume_")
    assert capture_basename("").startswith("autolume_")


def test_capture_path_lands_under_captures_and_is_a_string():
    path = capture_path("/models/thing.pkl", ".mp4")
    assert isinstance(path, str)
    assert path.endswith(".mp4")
    assert "captures" in path
    assert "thing_" in path
