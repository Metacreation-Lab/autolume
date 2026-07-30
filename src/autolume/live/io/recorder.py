"""Video recording and screenshot capture, both fed by the render fan-out.

The old app captured frames at the rate the UI happened to change them, wrote
them at a hard coded 30 fps, and only started the encoder thread at Stop,
joining it on the UI thread while it flushed the whole queue (constraints.md
legacy bugs 3, 4 and 8). Here the encoder runs for the whole take on its own
thread behind a bounded queue and every frame comes from the render loop's
fan-out, so what is written is what was shown.

When it was shown is a separate promise, and the one legacy bug 4 was really
about. `cv2.VideoWriter` is constant frame rate: it names one rate for the
whole file and every frame handed to it lasts exactly 1/rate. Naming the
render cap and writing one frame per rendered frame therefore replays the
take at cap/achieved times its real speed, which for a 1024 model at 17 fps
under the default cap of 60 is three and a half times too fast. So every
frame is stamped as it is queued, and the encoder holds each one for as many
of the file's frames as the clock says it was on screen for, dropping the
ones that arrive faster than the file's rate. The declared rate is nominal:
what it fixes is the file's resolution in time, not its speed.

`cv2` is imported inside the worker threads rather than at module scope, so
importing this module (which the control plane does) never pays for OpenCV.
"""

import collections
import dataclasses
import datetime
import logging
import os
import re
import threading
import time
from collections.abc import Callable

import numpy as np

from autolume.live.core.store import LatestValueStore
from autolume.live.errors import describe, safe_describe

logger = logging.getLogger(__name__)

# The one structured address reachable from raw OSC: it carries no state, it
# asks for one file. Everything else structured is a value object the network
# cannot express (see constraints.md).
SCREENSHOT_ADDRESS = "/capture/screenshot"

# Ceiling on the queue depth, not the bound that matters: two seconds of
# frames at 60 fps, which is a deep enough buffer for any frame small enough
# that 120 of them are affordable.
QUEUE_CAPACITY = 120
# What actually bounds the queue. A frame count is not a memory bound: 4x
# super-res on a 1024 model renders 4096x4096, 50 MB a frame, and 120 of
# those is 6 GB of host RAM sitting next to two StyleGAN models. The
# allowance is derived from the first frame's size against this budget, so a
# small frame still gets a deep queue and a huge one gets a shallow one.
BYTE_BUDGET = 320 * 1024 * 1024
# Floor, so even a frame bigger than the whole budget gets some buffering:
# without it a single slow write would drop every frame behind it.
MIN_QUEUE_FRAMES = 4
# What the file records at when the render loop is uncapped: an mp4 has to
# name a rate, and the old app's hard coded 30 is a sane one to inherit.
DEFAULT_FPS = 30
# The longest gap between two rendered frames a single frame may be held
# across. Loading a model can stall the render loop for a minute, and filling
# that honestly would write a minute of duplicates one at a time, which the
# performer waits through at Stop. Past the ceiling the take loses the excess:
# a recording that drifts by the length of a stall beats one that spends
# minutes saving.
MAX_GAP_SECONDS = 5.0
STOP_TIMEOUT = 5.0
# How long the encoder gets to finalize the file after it has been told to
# drop the backlog. Only one write and a release remain at that point.
ABORT_GRACE = 2.0
SCREENSHOT_CAPACITY = 8
_IDLE_WAIT = 0.05
_DROP_LOG_INTERVAL = 100
# Mirrors ndi.py and superres.py: past this many distinct causes the log
# stops growing and says so once.
_LOG_ONCE_CAP = 64
_DIGIT_RUN = re.compile(r"\d+")

_SIZE_CHANGED = "The frame size changed. The recording was stopped."
# Past tense on purpose. It stays on the status until a new take starts, and
# by then the take it refers to has usually finished saving, so a present
# tense sentence would have turned into a lie while the performer read it.
_STILL_SAVING = (
    "The previous recording was still saving. Press Record again to start a "
    "new take."
)
_ABORTED = "Autolume was closing. The recording was cut short and saved."


@dataclasses.dataclass(frozen=True)
class RecorderStatus:
    """What the recorder is doing, published for the performance panel.

    `recording` is the truth to show, not the `recording` parameter: a take
    that ended itself (a resolution change, a writer that would not open)
    clears it with `error` saying why, and the runtime follows by putting the
    parameter back.
    """

    recording: bool = False
    path: str | None = None
    frames_written: int = 0
    frames_dropped: int = 0
    error: str | None = None


def capture_basename(pkl_path: str | None, now: datetime.datetime | None = None) -> str:
    """`<model>_<timestamp>`, the old app's capture naming (display_capture_widget)."""
    name = ""
    if pkl_path:
        name = os.path.splitext(os.path.basename(str(pkl_path)))[0]
        name = re.sub(r"\s+", "-", name.strip())
    stamp = (now or datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{name or 'autolume'}_{stamp}"


def capture_path(
    pkl_path: str | None, extension: str, now: datetime.datetime | None = None
) -> str:
    """Where a capture lands: `<data root>/captures/<basename><extension>`.

    The basename is timestamped to the second, so two captures inside one
    second would otherwise be one file. This address is reachable from a
    hardware button, where two taps in a second is a normal thing to do, so a
    taken name gets `-2`, `-3` and so on appended.

    Both an on-disk check and a short memory of what was just handed out are
    needed: the writes are asynchronous, so the first file may well not exist
    yet when the second name is asked for.

    A string, never a `Path`: this crosses into a status snapshot, a log line
    and OpenCV, and a `WindowsPath` in any of those is a recurring bug in this
    repo. `data_path` is imported here rather than at module scope to keep the
    control plane's import chain off the legacy flat-root modules, matching
    `presets.py`.
    """
    from utils.user_data import data_path

    base = capture_basename(pkl_path, now)
    folder = data_path("captures")
    with _naming_lock:
        name = f"{base}{extension}"
        index = 1
        while index < _MAX_NAME_ATTEMPTS and _is_taken(folder, name):
            index += 1
            name = f"{base}-{index}{extension}"
        _handed_out.append(name)
    return str(data_path("captures", name))


def _is_taken(folder, name: str) -> bool:
    if name in _handed_out:
        return True
    try:
        return os.path.exists(os.path.join(str(folder), name))
    except OSError:
        return False


# The last few names handed out, so a capture whose file has not been
# written yet still takes its name out of circulation. Bounded, because this
# address can be swept from OSC.
_handed_out: collections.deque[str] = collections.deque(maxlen=64)
_naming_lock = threading.Lock()
_MAX_NAME_ATTEMPTS = 100


class Recorder:
    """Records the render fan-out to an mp4, encoding on its own thread.

    `on_frame` runs on the render thread and does nothing but append to a
    bounded deque: no conversion, no encode, no disk. Everything else happens
    on the encoder thread, which lives for exactly one take.
    """

    def __init__(
        self,
        capacity: int = QUEUE_CAPACITY,
        stop_timeout: float = STOP_TIMEOUT,
        byte_budget: int = BYTE_BUDGET,
        abort_grace: float = ABORT_GRACE,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._capacity = int(capacity)
        self._byte_budget = int(byte_budget)
        self._stop_timeout = float(stop_timeout)
        self._abort_grace = float(abort_grace)
        # Read once per queued frame, on the render thread. Monotonic, because
        # a take spanning a clock adjustment must not run backwards.
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        # Bounded by `_allowance` rather than by the deque's own maxlen: the
        # allowance is only knowable once a frame's size is, and a deque
        # cannot be re-bounded after construction.
        self._frames: collections.deque[tuple[float, np.ndarray]] = collections.deque()
        self._allowance: int | None = None
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._abort = threading.Event()
        self._thread: threading.Thread | None = None
        # Read by the render thread on every frame, so it is a plain flag and
        # not a lock acquisition. Cleared by whichever of the two threads ends
        # the take first.
        self._active = False
        self._path: str | None = None
        self._fps = DEFAULT_FPS
        # When the take's first frame was queued, which is where the file
        # begins. Taken here rather than from the first frame the encoder
        # gets to see: an encoder that falls behind from the very first frame
        # drops the head of the queue, and timing the file from what survived
        # would silently cut that much off the front of the take.
        self._origin: float | None = None
        self._written = 0
        self._dropped = 0
        # A refused start, kept until a new take actually begins. The take
        # that was still saving publishes its own clean finish a moment
        # later, and that must not wipe the only explanation the performer
        # has for why their second take never happened.
        self._refusal: str | None = None
        self._store: LatestValueStore[RecorderStatus] = LatestValueStore(
            RecorderStatus()
        )

    def status(self) -> RecorderStatus:
        return self._store.snapshot()

    def queue_allowance(self) -> int | None:
        """How many frames this take may hold, or None before the first frame."""
        return self._allowance

    def start(self, path: str, fps: int) -> None:
        """Begin a take. Called from the control thread: never blocks, never raises.

        `fps` is the render fps cap as it stands now, and it is the file's
        nominal rate for the whole take, because a `VideoWriter` names its
        rate once and cannot be told otherwise later. Nominal: frames are
        held or dropped against their own timestamps to fill it, so the take
        plays back at the speed it was performed at whatever the render loop
        actually managed.
        """
        if self._active:
            return
        thread = self._thread
        if thread is not None and thread.is_alive():
            # The previous encoder still owns the queue and its own writer.
            # Starting here would interleave two takes into one file, so the
            # request is refused with a reason the panel can show instead.
            logger.info("Ignoring a record request while the previous take is saving")
            self._refusal = _STILL_SAVING
            self._publish()
            return
        self._refusal = None
        with self._lock:
            self._frames.clear()
            self._allowance = None
            self._origin = None
            self._written = 0
            self._dropped = 0
        self._path = str(path)
        self._fps = _positive_fps(fps)
        self._stopping.clear()
        self._abort.clear()
        self._wake.clear()
        self._active = True
        self._publish()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None, abort_on_timeout: bool = False) -> None:
        """End the take and let the encoder flush what is queued.

        `timeout` bounds the join. The control thread passes 0.0: the tail of
        a take can be a hundred frames of encoding and the show's heartbeat
        cannot wait for it, so the encoder finishes, releases the writer and
        publishes its final counts on its own thread.

        `abort_on_timeout` is what shutdown passes, and it is the difference
        between a short recording and a lost one. The encoder is a daemon
        thread, so a process that exits while it is still flushing abandons it
        mid write: `release()` never runs, the mp4 header is never finalized,
        and the whole take is unopenable rather than merely short. Past the
        deadline the encoder is told to drop whatever is left and finalize.
        """
        # Cleared first and unconditionally, so a take started in the window
        # between another thread reading `_thread` and this line cannot leave
        # the sink accepting frames forever.
        self._active = False
        thread = self._thread
        if thread is None:
            return
        self._stopping.set()
        self._wake.set()
        thread.join(self._stop_timeout if timeout is None else float(timeout))
        if thread.is_alive() and abort_on_timeout:
            logger.warning(
                "The recording did not finish saving in time. Cutting it short "
                "so the file is playable"
            )
            self._abort.set()
            self._wake.set()
            thread.join(self._abort_grace)
        if thread.is_alive():
            logger.info("The recording is still saving in the background")
            return
        self._thread = None

    def on_frame(self, frame: np.ndarray, seq: int) -> None:
        """Queue one rendered frame. Runs on the render thread."""
        if not self._active:
            return
        # Before the lock, so a contended queue cannot backdate the frame it
        # delayed. This is the take's only record of when the frame was shown.
        stamp = self._clock()
        with self._lock:
            # Checked again, under the lock this time. A render thread
            # already past the check above when the take ends appends to a
            # deque the encoder has just finished clearing, and nothing
            # drains it afterwards, so that frame is held until the next
            # take starts: 48 MiB of it at 4096x4096.
            if not self._active:
                return
            if self._allowance is None:
                self._allowance = _queue_allowance(
                    getattr(frame, "nbytes", 0), self._capacity, self._byte_budget
                )
            if self._origin is None:
                self._origin = stamp
            dropped = len(self._frames) >= self._allowance
            if dropped:
                self._frames.popleft()
                self._dropped += 1
            self._frames.append((stamp, frame))
        self._wake.set()
        if dropped:
            self._publish()

    def _run(self) -> None:
        try:
            import cv2
        except Exception as exc:
            self._finish(f"Recording needs OpenCV. {describe(exc)}")
            return
        writer = None
        size: tuple[int, int] | None = None
        reason: str | None = None
        # Where the file's clock starts, read once from the take's first
        # queued frame and then moved on by whatever a stall gave up. Not the
        # moment Record was pressed: the wait for the first frame is the
        # pipeline coming up, not part of the take.
        origin: float | None = None
        # The last image written, held across the file's frames that no
        # rendered frame arrived for.
        held = None
        max_gap = max(1, int(round(self._fps * MAX_GAP_SECONDS)))
        try:
            while True:
                self._wake.clear()
                drained = False
                while (queued := self._take_one()) is not None:
                    drained = True
                    stamp, frame = queued
                    if self._abort.is_set():
                        reason = _ABORTED
                        break
                    frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                    if writer is None:
                        size = frame_size
                        writer, reason = self._open(cv2, size)
                        if writer is None:
                            break
                    elif frame_size != size:
                        logger.warning(
                            "Recording stopped: the frame size changed from %dx%d "
                            "to %dx%d mid take",
                            size[0],
                            size[1],
                            frame_size[0],
                            frame_size[1],
                        )
                        reason = _SIZE_CHANGED
                        break
                    if origin is None:
                        origin = self._origin if self._origin is not None else stamp
                    # Which of the file's frames this one belongs in.
                    # `_written` is also how many of them are already filled,
                    # so the difference is the gap this frame arrived after:
                    # negative if it arrived faster than the file's rate, in
                    # which case its slot is taken and it is dropped.
                    slot = int(round((stamp - origin) * self._fps))
                    gap = slot - self._written
                    if gap > max_gap:
                        # The take gives up the excess rather than writing it
                        # out. `origin` moves with it so the frames after the
                        # stall are timed against where the file actually is,
                        # instead of every one of them re-owing the same gap.
                        origin += (gap - max_gap) / self._fps
                        gap = max_gap
                    if gap < 0:
                        continue
                    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # The gap holds the picture that was on screen through it,
                    # so a frame lands at the time it was rendered rather than
                    # appearing early to cover for the frames before it.
                    batch = [held if held is not None else image] * gap + [image]
                    held = image
                    for one in batch:
                        if self._abort.is_set():
                            reason = _ABORTED
                            break
                        try:
                            writer.write(one)
                        except Exception as exc:
                            logger.exception("Writing a recorded frame failed")
                            reason = f"Recording failed. {describe(exc)}"
                            break
                        with self._lock:
                            self._written += 1
                        # Per frame, not per drained batch: a batch can be a
                        # hundred frames and several seconds of encoding, and a
                        # counter that freezes for the whole flush freezes at
                        # exactly the moment somebody is watching it to see
                        # whether their recording is going anywhere.
                        self._publish()
                    if reason is not None:
                        break
                if reason is not None:
                    break
                if self._stopping.is_set() and not self._pending():
                    break
                if not drained:
                    self._wake.wait(_IDLE_WAIT)
        except Exception as exc:
            logger.exception("The recorder's encoder thread failed")
            reason = f"Recording failed. {describe(exc)}"
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    logger.exception("Releasing the recording writer failed")
            self._finish(reason)

    def _open(self, cv2, size: tuple[int, int]) -> tuple[object | None, str | None]:
        try:
            directory = os.path.dirname(self._path or "")
            if directory:
                os.makedirs(directory, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self._path, fourcc, self._fps, size)
        except Exception as exc:
            logger.exception("Could not open the recording file %s", self._path)
            return None, f"Could not start recording. {describe(exc)}"
        if not writer.isOpened():
            logger.warning("Could not open the recording file %s", self._path)
            try:
                writer.release()
            except Exception:
                logger.exception("Releasing an unopened recording writer failed")
            return None, f"Could not open {self._path} for recording."
        return writer, None

    def _take_one(self) -> tuple[float, np.ndarray] | None:
        """The oldest queued frame, or None. One at a time on purpose:
        draining the whole deque let the encoder hold a full batch for the
        length of its flush while the render thread refilled the deque to
        the full allowance again, so the real peak was 2x the byte budget
        (measured: allowance 8, 16 simultaneous frames, 48 MB against a
        24 MB budget). Popping one keeps everything the recorder holds
        inside the allowance, plus the single frame being encoded.
        """
        with self._lock:
            if not self._frames:
                return None
            return self._frames.popleft()

    def _pending(self) -> bool:
        with self._lock:
            return bool(self._frames)

    def _publish(self, error: str | None = None) -> None:
        """Publish the current counters.

        `recording` is `_active` rather than an argument: the flag is what
        says whether frames are still being taken, and a status that
        disagreed with it would be the panel's truth. `error` falls back to a
        pending refusal, which is the one message that has to outlive the
        take it is about.
        """
        with self._lock:
            written, dropped = self._written, self._dropped
        self._store.set(
            RecorderStatus(
                recording=self._active,
                path=self._path,
                frames_written=written,
                frames_dropped=dropped,
                error=error or self._refusal,
            )
        )

    def _finish(self, reason: str | None) -> None:
        """Close the take out from the encoder thread, whatever ended it."""
        self._active = False
        with self._lock:
            self._frames.clear()
            written, dropped = self._written, self._dropped
        self._publish(reason)
        if reason is None:
            logger.info(
                "Recording saved to %s, %d frames written, %d dropped",
                self._path,
                written,
                dropped,
            )
        else:
            logger.warning(
                "Recording of %s ended. %s %d frames written, %d dropped",
                self._path,
                reason,
                written,
                dropped,
            )


class ScreenshotWorker:
    """Writes single frames to PNG, off whatever thread asked for one.

    The render thread hands a frame over and returns to rendering; encoding
    and the file write happen here. The thread starts on the first request, so
    a session that never captures anything never spins one up.
    """

    def __init__(
        self, capacity: int = SCREENSHOT_CAPACITY, stop_timeout: float = STOP_TIMEOUT
    ) -> None:
        self._stop_timeout = float(stop_timeout)
        self._lock = threading.Lock()
        self._pending: collections.deque[tuple[str, np.ndarray]] = collections.deque(
            maxlen=int(capacity)
        )
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._stopped = False
        self._drops = 0
        self._logged_errors: set[str] = set()
        self._log_cap_warned = False

    def save_png(self, path: str, frame: np.ndarray) -> None:
        """Queue one PNG write. Called from the render thread: never blocks."""
        with self._lock:
            if self._stopped:
                logger.debug("Ignoring a screenshot request after shutdown")
                return
            if len(self._pending) == self._pending.maxlen:
                self._report_drop()
            self._pending.append((str(path), frame))
            if self._thread is None:
                self._running = True
                self._thread = threading.Thread(
                    target=self._run, name="screenshot", daemon=True
                )
                self._thread.start()
        self._wake.set()

    def stop(self, timeout: float | None = None) -> None:
        """Finish the queued writes, then let the thread go."""
        with self._lock:
            self._stopped = True
            thread = self._thread
        if thread is None:
            return
        self._running = False
        self._wake.set()
        thread.join(self._stop_timeout if timeout is None else float(timeout))
        if thread.is_alive():
            logger.info("A screenshot is still being written in the background")
            return
        with self._lock:
            self._thread = None

    def _report_drop(self) -> None:
        """Say a request was dropped, throttled: this is a render-thread call.

        `/capture/screenshot` is reachable from OSC, so a fader pointed at it
        arrives every frame, and an unthrottled warning would put a formatted
        log line on the render thread at frame rate. Must be called with the
        lock held.
        """
        self._drops += 1
        if self._drops == 1 or self._drops % _DROP_LOG_INTERVAL == 0:
            logger.warning(
                "Screenshots are arriving faster than they can be saved, "
                "%d dropped so far",
                self._drops,
            )

    def _run(self) -> None:
        while True:
            self._wake.clear()
            jobs = self._drain()
            for path, frame in jobs:
                self._write(path, frame)
            if not self._running and not self._queued():
                return
            if not jobs:
                self._wake.wait(_IDLE_WAIT)

    def _drain(self) -> list[tuple[str, np.ndarray]]:
        with self._lock:
            jobs = list(self._pending)
            self._pending.clear()
        return jobs

    def _queued(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def _write(self, path: str, frame: np.ndarray) -> None:
        try:
            import cv2

            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            written = cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            self._record_failure(path, exc)
            return
        if written:
            logger.info("Screenshot saved to %s", path)
        else:
            self._record_failure(path, None)

    def _record_failure(self, path: str, exc: Exception | None) -> None:
        """Log a failed write once per cause, following `ndi.py`'s shape.

        A captures folder that cannot be written to fails identically for
        every request, and this address can be swept from OSC, so an
        unthrottled traceback per failure buries the log under one mistake.
        The key normalises digit runs so a message carrying an errno or a
        frame number is still one cause.
        """
        text = safe_describe(exc) if exc is not None else "OpenCV would not write it"
        key = f"{type(exc).__name__ if exc is not None else 'refused'}:{_DIGIT_RUN.sub('N', text)}"
        if key in self._logged_errors:
            return
        if len(self._logged_errors) >= _LOG_ONCE_CAP:
            if not self._log_cap_warned:
                self._log_cap_warned = True
                logger.warning(
                    "Reached %d distinct screenshot failure causes, further "
                    "distinct causes will not be logged",
                    _LOG_ONCE_CAP,
                )
            return
        self._logged_errors.add(key)
        logger.warning("Could not save the screenshot %s. %s", path, text)


def _queue_allowance(nbytes: int, capacity: int, budget: int) -> int:
    """How many frames of `nbytes` fit the budget, within the ceiling and floor.

    The ceiling keeps a small frame's queue at the depth it always had; the
    floor keeps a frame bigger than the whole budget from having no queue at
    all.
    """
    if nbytes <= 0:
        return capacity
    return max(MIN_QUEUE_FRAMES, min(capacity, budget // int(nbytes)))


def _positive_fps(fps: object) -> int:
    """The take's frame rate, defaulting an uncapped or unusable one.

    `fps_cap` is 0 when the render loop is uncapped, and this is reached from
    the control thread, so anything that is not a usable rate resolves rather
    than raises.
    """
    try:
        value = int(fps)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    return value if value > 0 else DEFAULT_FPS
