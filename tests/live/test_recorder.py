"""The video recorder and the screenshot worker, both off the render thread.

Every test here stands a fake `cv2.VideoWriter` in for the real one: the codec
is not what is under test, the queue discipline and the thread lifecycle are.

The recorder times the file against a clock, so a test that feeds frames in a
tight loop is feeding a take that lasted microseconds, and the file it gets is
one frame long however many it fed. The tests about the queue and the encoder
hand it a `StepClock` at the file's own rate instead, which is the take those
tests were written about: one rendered frame per frame of the file.
"""

import datetime
import threading
import time

import cv2
import numpy as np
import pytest

from autolume.live.io.recorder import (
    DEFAULT_FPS,
    MAX_GAP_SECONDS,
    MIN_QUEUE_FRAMES,
    QUEUE_CAPACITY,
    SCREENSHOT_CAPACITY,
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


class StepClock:
    """A monotonic clock advancing a fixed step per reading, first read at 0.

    The recorder reads it once per queued frame, on the thread that queues
    one, so a step is the interval between two rendered frames.
    """

    def __init__(self, step):
        self._step = float(step)
        self._now = -float(step)

    def __call__(self):
        self._now += self._step
        return self._now


class ListClock:
    """A clock reading off a script, holding the last reading once spent."""

    def __init__(self, readings):
        self._readings = list(readings)
        self._index = 0

    def __call__(self):
        value = self._readings[min(self._index, len(self._readings) - 1)]
        self._index += 1
        return value


class FakeWriter:
    """Stands in for `cv2.VideoWriter`, recording what it was asked to write."""

    created = []
    open_gate = None
    write_gate = None
    write_delay = 0.0
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
        if FakeWriter.write_delay:
            time.sleep(FakeWriter.write_delay)
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
    FakeWriter.write_delay = 0.0
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
    recorder = Recorder(clock=StepClock(1.0 / 24))
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


def test_a_take_fed_slower_than_the_cap_still_lasts_as_long_as_it_did(
    writers, tmp_path
):
    """The contract is the take's duration, not the number it declares.

    `VideoWriter` is constant frame rate only, so a file that declares the
    render cap and holds one frame per rendered frame plays back at
    cap/achieved times too fast: a 1024 model at 17 fps under the default cap
    of 60 replays a three second take in under a second. Fed here in real
    time at half the cap, so a file that ignored the clock would be exactly
    twice too short.
    """
    fps = 30
    feed_fps = 15
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), fps)
    try:
        started = time.monotonic()
        for value in range(15):
            recorder.on_frame(frame(value), value)
            time.sleep(1.0 / feed_fps)
        elapsed = time.monotonic() - started - 1.0 / feed_fps
        assert wait_for(lambda: writers and len(writers[0].frames) >= elapsed * fps)
    finally:
        recorder.stop()
    expected = elapsed * fps
    assert abs(len(writers[0].frames) - expected) <= 0.15 * expected


def test_a_slow_take_fills_the_gaps_between_its_frames(writers, tmp_path):
    """The same contract as the real time test above, counted exactly.

    Frames arrive at a fifth of the file's rate, so every one of them has to
    be held for five of the file's frames. The last frame lands at second 7,
    which is slot 210, so the file is 211 frames long and plays for the seven
    seconds the take lasted.
    """
    recorder = Recorder(clock=StepClock(1.0 / 6))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        for value in range(43):
            recorder.on_frame(frame(value), value)
        assert wait_for(lambda: writers and len(writers[0].frames) == 211)
    finally:
        recorder.stop()
    assert recorder.status().frames_written == 211


def test_a_take_fed_faster_than_the_files_rate_is_thinned_to_it(writers, tmp_path):
    """The error the other way: an uncapped loop outrunning the file's rate.

    A small model with the render loop uncapped renders far faster than the
    30 fps the file falls back to. Holding one file frame per rendered frame
    would stretch a two second take over eight.
    """
    recorder = Recorder(clock=StepClock(1.0 / 120))
    recorder.start(str(tmp_path / "take.mp4"), 0)
    try:
        for value in range(241):
            recorder.on_frame(frame(value), value)
        assert wait_for(lambda: writers and len(writers[0].frames) == 61)
        time.sleep(0.05)
    finally:
        recorder.stop()
    assert len(writers[0].frames) == 61


def test_a_stall_mid_take_is_filled_only_up_to_the_gap_ceiling(writers, tmp_path):
    """A long stall must not turn into minutes of duplicated frames.

    Loading a model can hold the render loop for far longer than anyone wants
    written out one held frame at a time, so the gap a single frame can fill
    is bounded. The take drifts by whatever the stall was over the ceiling,
    which is the only thing left to trade. Ten minutes apart here, so the file
    is the two frames themselves plus the ceiling between them.
    """
    expected = 2 + int(30 * MAX_GAP_SECONDS)
    recorder = Recorder(clock=ListClock([0.0, 600.0]))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        recorder.on_frame(frame(1), 0)
        recorder.on_frame(frame(2), 1)
        assert wait_for(lambda: writers and len(writers[0].frames) == expected)
        time.sleep(0.05)
    finally:
        recorder.stop()
    assert len(writers[0].frames) == expected


def test_an_uncapped_render_rate_records_at_the_default_nominal_fps(writers, tmp_path):
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

    A drop costs the take its picture, never its length. The frames fed here
    are one file frame apart, so the file is still `total` frames long and the
    stretch nothing survived for holds the last picture that did.
    """
    FakeWriter.open_gate = threading.Event()
    recorder = Recorder(clock=StepClock(1.0 / 30))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    total = QUEUE_CAPACITY + 60
    lost = total - 1 - QUEUE_CAPACITY
    try:
        recorder.on_frame(frame(0), 0)
        assert wait_for(lambda: len(writers) == 1)
        for value in range(1, total):
            recorder.on_frame(frame(value), value)
        assert recorder.status().frames_dropped == lost
        FakeWriter.open_gate.set()
        assert wait_for(lambda: len(writers[0].frames) == total)
    finally:
        FakeWriter.open_gate.set()
        recorder.stop()

    # Channel 2 of a written (BGR) frame is the red channel the fake frame put
    # its value in. The first frame got through before the queue filled;
    # everything after the gap is the tail of the run, which is what "drop the
    # oldest" means.
    written = writers[0].frames
    assert written[0][0, 0, 2] == 0
    assert written[lost][0, 0, 2] == 0
    assert written[lost + 1][0, 0, 2] == total - QUEUE_CAPACITY
    assert written[-1][0, 0, 2] == total - 1
    status = recorder.status()
    assert status.frames_written == total
    assert status.frames_dropped == lost


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
    recorder = Recorder(clock=StepClock(1.0 / 30))
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


def test_a_flood_of_requests_drops_the_oldest_and_says_so_once(tmp_path, caplog):
    """`/capture/screenshot` is reachable from OSC, so a fader can point at it.

    A warning per dropped request would be a formatted log line on the render
    thread at frame rate, which is the sort of thing that turns a mistake in
    a mapping into dropped frames.
    """
    worker = ScreenshotWorker()
    with caplog.at_level("WARNING"):
        # The worker thread starts on the first request, so it may drain a
        # few. Enough requests arrive here that the queue overflows whatever
        # it manages to take.
        for index in range(SCREENSHOT_CAPACITY * 6):
            worker.save_png(str(tmp_path / f"shot{index}.png"), frame(index))
        worker.stop()
    drops = [
        record for record in caplog.records if "faster than they can be saved" in record.message
    ]
    assert len(drops) == 1
    assert len(list(tmp_path.glob("*.png"))) < SCREENSHOT_CAPACITY * 6


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


# --- the queue is bounded in bytes, not in frames --------------------------


def big_frame(value=1, side=64):
    image = np.zeros((side, side, 3), dtype=np.uint8)
    image[:, :] = (value, value + 1, value + 2)
    image.flags.writeable = False
    return image


def test_the_queue_allowance_falls_as_the_frame_size_rises(writers, tmp_path):
    """120 frames is a frame count, and a frame count is not a memory bound.

    Super-res 4x on a 1024 model renders 4096x4096, which is 50 MB a frame:
    120 of those is 6 GB of host RAM alongside two StyleGAN models. The
    budget is what is bounded; the frame count is only its ceiling.
    """
    budget = 4 * 1024 * 1024
    small = Recorder(byte_budget=budget)
    small.start(str(tmp_path / "small.mp4"), 30)
    small.on_frame(big_frame(1, side=32), 0)
    assert wait_for(lambda: small.queue_allowance() is not None)
    small_allowance = small.queue_allowance()
    small.stop()

    large = Recorder(byte_budget=budget)
    large.start(str(tmp_path / "large.mp4"), 30)
    large.on_frame(big_frame(1, side=512), 0)
    assert wait_for(lambda: large.queue_allowance() is not None)
    large_allowance = large.queue_allowance()
    large.stop()

    assert small_allowance > large_allowance
    assert small_allowance == QUEUE_CAPACITY
    assert large_allowance == budget // (512 * 512 * 3)


def test_the_allowance_never_falls_below_a_usable_floor(writers, tmp_path):
    recorder = Recorder(byte_budget=1)
    recorder.start(str(tmp_path / "huge.mp4"), 30)
    try:
        recorder.on_frame(big_frame(1, side=256), 0)
        assert wait_for(lambda: recorder.queue_allowance() is not None)
    finally:
        recorder.stop()
    assert recorder.queue_allowance() == MIN_QUEUE_FRAMES


def test_the_budget_bounds_what_the_encoder_holds_as_well_as_the_queue(
    monkeypatch, tmp_path
):
    """IO-6: the budget must bound every frame the recorder holds, not the
    deque alone. Draining the whole deque let the encoder hold a full batch
    for the length of its flush while the render thread refilled the deque
    to the full allowance again: measured allowance 8, peak 16 simultaneous
    frames, 2x the byte budget (604 MB, not 320, in the documented 4096
    super-res case). Every write here needs a permit, so the encoder is
    provably mid flush while the queue refills, exactly the window the old
    drain doubled the footprint in."""
    permits = threading.Semaphore(0)

    class OnePermitWriter:
        attempts = 0

        def __init__(self, path, fourcc, fps, size):
            pass

        def isOpened(self):
            return True

        def write(self, image):
            OnePermitWriter.attempts += 1
            permits.acquire()

        def release(self):
            pass

    OnePermitWriter.attempts = 0
    monkeypatch.setattr(cv2, "VideoWriter", OnePermitWriter)
    allowance = 4
    nbytes = frame(0).nbytes
    recorder = Recorder(byte_budget=allowance * nbytes, clock=StepClock(1.0 / 30))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        # The encoder takes the first frame and blocks mid write.
        recorder.on_frame(frame(0), 0)
        assert wait_for(lambda: OnePermitWriter.attempts == 1)
        assert recorder.queue_allowance() == allowance

        # The render thread refills the queue to its full allowance.
        for value in range(1, 1 + allowance):
            recorder.on_frame(frame(value), value)

        # One write completes; the encoder is now holding whatever it took
        # off the queue while the render thread fills it up again.
        permits.release()
        assert wait_for(
            lambda: OnePermitWriter.attempts == 2
            and recorder.status().frames_written == 1
        )
        fed = 1 + 2 * allowance
        for value in range(1 + allowance, fed):
            recorder.on_frame(frame(value), value)

        status = recorder.status()
        held = fed - status.frames_written - status.frames_dropped
        assert held <= allowance + 1
    finally:
        # Enough permits for every write the flush still owes, so the
        # encoder can finish and `stop()`'s join is not left waiting on a
        # gate this test holds shut.
        for _ in range(200):
            permits.release()
        recorder.stop()


def test_the_byte_budget_is_what_actually_bounds_the_queue(writers, tmp_path):
    """The allowance is not just reported, it is the drop threshold."""
    FakeWriter.open_gate = threading.Event()
    allowance = 4
    total = 20
    recorder = Recorder(byte_budget=allowance * 64 * 64 * 3, clock=StepClock(1.0 / 30))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    try:
        recorder.on_frame(big_frame(0), 0)
        assert wait_for(lambda: len(writers) == 1)
        for value in range(1, total):
            recorder.on_frame(big_frame(value), value)
        assert recorder.queue_allowance() == allowance
        assert recorder.status().frames_dropped == total - 1 - allowance
        FakeWriter.open_gate.set()
        # Only the allowance survived the queue, and the file is still the
        # length of the take: the dropped stretch holds the last picture.
        assert wait_for(lambda: len(writers[0].frames) == total)
    finally:
        FakeWriter.open_gate.set()
        recorder.stop()


# --- shutdown always finalizes the file ------------------------------------


def test_a_stop_that_runs_out_of_time_still_releases_the_writer(writers, tmp_path):
    """An abandoned encoder means no `release()`, and no moov atom.

    A file whose header was never written is not a short recording, it is a
    lost performance: nothing can open it at all. Dropping the backlog and
    finalizing is strictly better.
    """
    # A backlog that needs a second to flush, against a stop that has a fifth
    # of one. Measured mp4v cost at 4096x4096 is 118 ms a frame, so this is
    # the shape of a real shutdown during a high resolution take.
    FakeWriter.write_delay = 0.05
    recorder = Recorder(clock=StepClock(1.0 / 30))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    for value in range(20):
        recorder.on_frame(frame(value), value)
    assert wait_for(lambda: len(writers) == 1)

    started = time.monotonic()
    recorder.stop(timeout=0.2, abort_on_timeout=True)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert writers[0].released is True
    assert 0 < len(writers[0].frames) < 20
    assert recorder.status().recording is False
    assert recorder.status().error is not None


def test_an_ordinary_stop_never_abandons_the_backlog(writers, tmp_path):
    """Only a shutdown deadline aborts. A normal Stop flushes everything."""
    FakeWriter.write_delay = 0.01
    recorder = Recorder(clock=StepClock(1.0 / 30))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    for value in range(20):
        recorder.on_frame(frame(value), value)
    recorder.stop()
    assert len(writers[0].frames) == 20
    assert recorder.status().error is None


# --- the refusal message outlives the take that caused it ------------------


def test_the_refusal_survives_the_finishing_take(writers, tmp_path):
    """Otherwise the reason vanishes and Record is simply dead for a while.

    The take that was still saving publishes its own clean finish a moment
    later. If that clears the refusal, the performer is left with an unlit
    Record button, no second file, and nothing on screen that says why.
    """
    FakeWriter.write_gate = threading.Event()
    recorder = Recorder()
    recorder.start(str(tmp_path / "one.mp4"), 30)
    recorder.on_frame(frame(1), 0)
    assert wait_for(lambda: len(writers) == 1)
    recorder.stop(timeout=0.0)

    recorder.start(str(tmp_path / "two.mp4"), 30)
    refusal = recorder.status().error
    assert refusal is not None

    FakeWriter.write_gate.set()
    assert wait_for(lambda: writers[0].released)
    time.sleep(0.1)
    assert recorder.status().error == refusal

    recorder.start(str(tmp_path / "three.mp4"), 30)
    assert recorder.status().error is None
    recorder.stop()


def test_frames_written_keeps_moving_while_a_take_is_flushing(writers, tmp_path):
    """The counter a performer watches at Stop must not be the one that freezes."""
    FakeWriter.write_gate = threading.Event()
    recorder = Recorder(clock=StepClock(1.0 / 30))
    recorder.start(str(tmp_path / "take.mp4"), 30)
    for value in range(6):
        recorder.on_frame(frame(value), value)
    assert wait_for(lambda: len(writers) == 1)
    recorder.stop(timeout=0.0)

    FakeWriter.write_gate.set()
    assert wait_for(lambda: recorder.status().frames_written >= 3)
    assert wait_for(lambda: recorder.status().frames_written == 6)
    recorder.stop()


# --- screenshot naming and failure logging ---------------------------------


def test_two_captures_in_the_same_second_do_not_overwrite(tmp_path, monkeypatch):
    from utils import user_data

    monkeypatch.setattr(user_data, "_prefs", {"version": 1, "data_root": str(tmp_path)})
    monkeypatch.setattr(user_data, "_data_root", str(tmp_path))
    frozen = datetime.datetime(2026, 7, 26, 12, 0, 0)

    first = capture_path("/models/thing.pkl", ".png", now=frozen)
    second = capture_path("/models/thing.pkl", ".png", now=frozen)
    third = capture_path("/models/thing.pkl", ".png", now=frozen)

    assert len({first, second, third}) == 3
    assert first.endswith("thing_2026-07-26_12-00-00.png")


def test_a_name_already_on_disk_is_not_reused(tmp_path, monkeypatch):
    from utils import user_data

    monkeypatch.setattr(user_data, "_prefs", {"version": 1, "data_root": str(tmp_path)})
    monkeypatch.setattr(user_data, "_data_root", str(tmp_path))
    frozen = datetime.datetime(2026, 7, 26, 13, 0, 0)
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "thing_2026-07-26_13-00-00.mp4").write_bytes(b"old take")

    path = capture_path("/models/thing.pkl", ".mp4", now=frozen)
    assert path.endswith("thing_2026-07-26_13-00-00-2.mp4")


def test_repeated_screenshot_failures_log_once_per_cause(tmp_path, caplog):
    """An unwritable folder under an OSC sweep must not flood the log."""
    worker = ScreenshotWorker()
    target = tmp_path / "wall"
    target.write_bytes(b"not a folder")
    with caplog.at_level("WARNING"):
        for index in range(4):
            worker.save_png(str(target / f"shot{index}.png"), frame(index))
            time.sleep(0.05)
        worker.stop()
    failures = [
        record
        for record in caplog.records
        if "Could not save the screenshot" in record.getMessage()
    ]
    assert len(failures) == 1


class _ParkOnceLock:
    """Wraps a real lock and parks the first entrant, then gets out of the way.

    `threading.Lock` instances refuse attribute assignment (`acquire` is
    read-only on the builtin type), so the interception has to be a
    replacement object rather than a monkeypatched method. Every caller in
    `recorder.py` uses `with self._lock:`, never bare `acquire`/`release`, so
    wrapping the context manager protocol is enough.
    """

    def __init__(self, real_lock, parked, resume):
        self._real = real_lock
        self._parked = parked
        self._resume = resume
        self._armed = True

    def __enter__(self):
        if self._armed:
            self._armed = False
            self._parked.set()
            self._resume.wait(2.0)
        self._real.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._real.release()
        return False


def test_a_frame_that_races_the_end_of_a_take_is_not_stranded(writers, tmp_path):
    """The `_active` check outside the lock can be passed just as a take ends.

    Nothing drains the queue after the encoder's last clear, so a frame that
    lands there is held until the next take starts. That is 48 MiB at
    4096x4096, which the byte budget makes reachable rather than theoretical.

    The window is between `on_frame`'s outer check (active is still True) and
    its lock acquisition. To hit that window deterministically rather than by
    timing luck, the racer is parked exactly there with `_ParkOnceLock`, and
    only let through after the encoder has genuinely finished the take
    (through a real `stop()`, not a simulated one) and released the lock
    itself. A version without the recheck inside the lock appends the
    racer's frame into a deque nothing will ever drain again.
    """
    recorder = Recorder()
    recorder.start(str(tmp_path / "take.mp4"), 30)
    recorder.on_frame(frame(1), 0)
    assert wait_for(lambda: writers and writers[0].frames)

    parked = threading.Event()
    resume = threading.Event()
    recorder._lock = _ParkOnceLock(recorder._lock, parked, resume)
    racer = threading.Thread(target=lambda: recorder.on_frame(frame(2), 1))
    racer.start()
    # The racer has passed the outer `_active` check (still True here) and is
    # parked immediately before entering the lock.
    assert parked.wait(2.0)

    # Run the take to completion for real, through the same wrapped lock:
    # `_finish` sets `_active` False and clears the (still empty) deque.
    recorder.stop(timeout=2.0)
    assert recorder.status().recording is False
    assert list(recorder._frames) == []

    # Only now does the racer's on_frame proceed into the locked section,
    # exactly as if the take had ended out from under it.
    resume.set()
    racer.join(2.0)

    assert list(recorder._frames) == []
    recorder.stop()
