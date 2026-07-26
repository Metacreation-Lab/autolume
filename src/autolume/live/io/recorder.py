"""Video recording and screenshot capture, both fed by the render fan-out.

The old app captured frames at the rate the UI happened to change them, wrote
them at a hard coded 30 fps, and only started the encoder thread at Stop,
joining it on the UI thread while it flushed the whole queue (constraints.md
legacy bugs 3, 4 and 8). Here the encoder runs for the whole take on its own
thread behind a bounded queue, the writer's fps is the render cap the take
started at, and every frame comes from the render loop's fan-out, so what is
written is what was shown.

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

import numpy as np

from autolume.live.core.store import LatestValueStore
from autolume.live.errors import describe

logger = logging.getLogger(__name__)

# The one structured address reachable from raw OSC: it carries no state, it
# asks for one file. Everything else structured is a value object the network
# cannot express (see constraints.md).
SCREENSHOT_ADDRESS = "/capture/screenshot"

# Two seconds of frames at 60 fps. Long enough to ride out an encoder stall,
# short enough that the queue is not a second copy of the whole take.
QUEUE_CAPACITY = 120
# What the file records at when the render loop is uncapped: an mp4 has to
# name a rate, and the old app's hard coded 30 is a sane one to inherit.
DEFAULT_FPS = 30
STOP_TIMEOUT = 5.0
SCREENSHOT_CAPACITY = 8
_IDLE_WAIT = 0.05

_SIZE_CHANGED = "The frame size changed. The recording was stopped."
_STILL_SAVING = "The previous recording is still saving. Try again in a moment."


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

    A string, never a `Path`: this crosses into a status snapshot, a log line
    and OpenCV, and a `WindowsPath` in any of those is a recurring bug in this
    repo. `data_path` is imported here rather than at module scope to keep the
    control plane's import chain off the legacy flat-root modules, matching
    `presets.py`.
    """
    from utils.user_data import data_path

    return str(data_path("captures", f"{capture_basename(pkl_path, now)}{extension}"))


class Recorder:
    """Records the render fan-out to an mp4, encoding on its own thread.

    `on_frame` runs on the render thread and does nothing but append to a
    bounded deque: no conversion, no encode, no disk. Everything else happens
    on the encoder thread, which lives for exactly one take.
    """

    def __init__(
        self, capacity: int = QUEUE_CAPACITY, stop_timeout: float = STOP_TIMEOUT
    ) -> None:
        self._capacity = int(capacity)
        self._stop_timeout = float(stop_timeout)
        self._lock = threading.Lock()
        self._frames: collections.deque[np.ndarray] = collections.deque(
            maxlen=self._capacity
        )
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        # Read by the render thread on every frame, so it is a plain flag and
        # not a lock acquisition. Cleared by whichever of the two threads ends
        # the take first.
        self._active = False
        self._path: str | None = None
        self._fps = DEFAULT_FPS
        self._written = 0
        self._dropped = 0
        self._store: LatestValueStore[RecorderStatus] = LatestValueStore(
            RecorderStatus()
        )

    def status(self) -> RecorderStatus:
        return self._store.snapshot()

    def start(self, path: str, fps: int) -> None:
        """Begin a take. Called from the control thread: never blocks, never raises.

        `fps` is the render fps cap as it stands now. It is what the file
        records at for the whole take, because a `VideoWriter` names its rate
        once and cannot be told otherwise later.
        """
        if self._active:
            return
        thread = self._thread
        if thread is not None and thread.is_alive():
            # The previous encoder still owns the queue and its own writer.
            # Starting here would interleave two takes into one file, so the
            # request is refused with a reason the panel can show instead.
            logger.info("Ignoring a record request while the previous take is saving")
            self._publish(_STILL_SAVING)
            return
        with self._lock:
            self._frames.clear()
            self._written = 0
            self._dropped = 0
        self._path = str(path)
        self._fps = _positive_fps(fps)
        self._stopping.clear()
        self._wake.clear()
        self._active = True
        self._publish()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """End the take and let the encoder flush what is queued.

        `timeout` bounds the join. The control thread passes 0.0: the tail of
        a take can be a hundred frames of encoding and the show's heartbeat
        cannot wait for it, so the encoder finishes, releases the writer and
        publishes its final counts on its own thread. Shutdown passes the full
        timeout, because there the flush is the point.
        """
        thread = self._thread
        if thread is None:
            return
        self._active = False
        self._stopping.set()
        self._wake.set()
        thread.join(self._stop_timeout if timeout is None else float(timeout))
        if thread.is_alive():
            logger.info("The recording is still saving in the background")
            return
        self._thread = None

    def on_frame(self, frame: np.ndarray, seq: int) -> None:
        """Queue one rendered frame. Runs on the render thread."""
        if not self._active:
            return
        with self._lock:
            dropped = len(self._frames) == self._capacity
            if dropped:
                self._dropped += 1
            self._frames.append(frame)
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
        try:
            while True:
                self._wake.clear()
                frames = self._drain()
                for frame in frames:
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
                    try:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    except Exception as exc:
                        logger.exception("Writing a recorded frame failed")
                        reason = f"Recording failed. {describe(exc)}"
                        break
                    with self._lock:
                        self._written += 1
                if reason is not None:
                    break
                # Only while the take is live: once `stop` has cleared
                # `_active` this thread is flushing, and republishing here
                # would erase whatever the status is now carrying (a refused
                # second take says so through the same channel).
                if self._active:
                    self._publish()
                if self._stopping.is_set() and not self._pending():
                    break
                if not frames:
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

    def _drain(self) -> list[np.ndarray]:
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
        return frames

    def _pending(self) -> bool:
        with self._lock:
            return bool(self._frames)

    def _publish(self, error: str | None = None) -> None:
        """Publish the current counters. `recording` is `_active`, not an
        argument: the flag is what says whether frames are still being taken,
        and a status that disagreed with it would be the panel's truth."""
        with self._lock:
            written, dropped = self._written, self._dropped
        self._store.set(
            RecorderStatus(
                recording=self._active,
                path=self._path,
                frames_written=written,
                frames_dropped=dropped,
                error=error,
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

    def save_png(self, path: str, frame: np.ndarray) -> None:
        """Queue one PNG write. Called from the render thread: never blocks."""
        with self._lock:
            if self._stopped:
                logger.debug("Ignoring a screenshot request after shutdown")
                return
            if len(self._pending) == self._pending.maxlen:
                logger.warning("Screenshot queue is full, dropping the oldest request")
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
        except Exception:
            logger.exception("Could not save the screenshot %s", path)
            return
        if written:
            logger.info("Screenshot saved to %s", path)
        else:
            logger.warning("Could not save the screenshot %s", path)


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
