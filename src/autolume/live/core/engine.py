"""The render loop: derives each frame from one atomic RenderParams snapshot.

Self-paces against the fps cap with absolute-deadline scheduling so
post-render sink work is absorbed into the next frame's sleep instead of
stacking on the cap (pacing strategy ported from balagan engine.py).
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from autolume.live.core.params import RenderParams
from autolume.live.core.sinks import FrameSink
from autolume.live.core.store import LatestValueStore
from autolume.live.errors import safe_describe

logger = logging.getLogger(__name__)

_IDLE_SLEEP = 0.05
# Past this many distinct failure causes the model has bigger problems than a
# missing log line, and the set stops growing (mirrors superres.py's cap,
# which says why it went quiet).
_LOG_ONCE_CAP = 64
_DIGIT_RUN = re.compile(r"\d+")


@dataclass(frozen=True)
class RenderStatus:
    """What the render thread has to report, published for the preview.

    Every other subsystem already has a channel like this (`RecorderStatus`,
    `NdiStatus`, `OscStatus`, `ModelHost.error()`), and the render path was
    the one whose failure the performer could see without being able to read
    why: a model that raises on every frame leaves the preview holding the
    last good frame, indefinitely, while the loop retries the same params.

    `error` is set while frames are failing and cleared by the first frame
    that renders. `failed_frames` counts the current streak. `last_ok_seq`
    names the frame the picture is stuck on, or -1 when no frame has ever
    rendered.
    """

    error: str | None = None
    failed_frames: int = 0
    last_ok_seq: int = -1


class RenderLoop:
    def __init__(
        self,
        render_store: LatestValueStore[RenderParams],
        model_host,
        sinks: list[FrameSink],
        screenshot: Callable[[str, np.ndarray], None] | None = None,
    ) -> None:
        self._render_store = render_store
        self._model_host = model_host
        self._sinks = sinks
        # Where a latched screenshot goes. The writer owns a thread of its
        # own (io/recorder.py's ScreenshotWorker), so this call hands a frame
        # over and returns; the render thread never touches disk.
        self._screenshot = screenshot
        self._screenshot_path: str | None = None
        self._screenshot_lock = threading.Lock()
        self.status_store: LatestValueStore[RenderStatus] = LatestValueStore(
            RenderStatus()
        )
        # Written only on the render thread; a plain flag so the healthy
        # steady state costs no store read or write per frame.
        self._failing = False
        self._logged_errors: set[str] = set()
        self._log_cap_warned = False
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._next_deadline: float | None = None
        self._fps = 0.0
        self._fps_window_start = time.perf_counter()
        self._fps_window_frames = 0

    def render_one(self) -> bool:
        params = self._render_store.snapshot()
        model = self._model_host.current()
        if model is None:
            self._next_deadline = None
            return False
        # One number names this frame everywhere: the index the noise animation
        # advances with is the sequence the sinks are told about.
        seq = self._seq
        try:
            frame = model.render_frame(params, seq)
        except Exception as exc:
            self._record_render_failure(exc)
            self._next_deadline = None
            return False
        self._seq = seq + 1
        if self._failing:
            self._failing = False
            self.status_store.set(
                RenderStatus(error=None, failed_frames=0, last_ok_seq=seq)
            )
        # Every sink is about to be handed this one array, so nothing may write
        # to it. Marked here, once, before the fan-out, so a sink added later is
        # covered by construction rather than by whoever adds it remembering.
        # A consumer that tinted its frame in place would tint the show and the
        # recording too, and this turns that into a ValueError in the code that
        # did it instead of a picture nobody can account for. Anything that
        # genuinely needs to change a frame, super-res and the recorder's
        # encoder being the likely ones, copies it first. Copying here instead
        # would be megabytes of memcpy per sink per frame on the render thread,
        # paid every frame, against a fault that should never happen.
        frame.flags.writeable = False
        for sink in self._sinks:
            try:
                sink.on_frame(frame, seq)
            except Exception:
                logger.exception("Frame sink %r failed", sink)
        self._serve_screenshot(frame)
        self._track_fps()
        self._limit_framerate(params.fps_cap)
        return True

    def _record_render_failure(self, exc: Exception) -> None:
        """Publish the failure for the UI and log it once per cause.

        `_run` retries the same params every `_IDLE_SLEEP`, so a model that
        raises on every frame used to write a full traceback ~18 times a
        second, indefinitely, while every status line stayed green. The dedup
        key normalises digit runs the way `superres.py`'s does, so an OOM
        message carrying varying byte counts is one cause. The status itself
        is set unconditionally: it is the current state, not a log.
        """
        text = safe_describe(exc)
        previous = self.status_store.snapshot()
        last_ok = previous.last_ok_seq if self._failing else self._seq - 1
        self._failing = True
        self.status_store.set(
            RenderStatus(
                error=text,
                failed_frames=previous.failed_frames + 1,
                last_ok_seq=last_ok,
            )
        )
        key = f"{type(exc).__name__}:{_DIGIT_RUN.sub('N', text)}"
        if key in self._logged_errors:
            return
        if len(self._logged_errors) >= _LOG_ONCE_CAP:
            if not self._log_cap_warned:
                self._log_cap_warned = True
                logger.warning(
                    "Reached %d distinct render failure causes, further "
                    "distinct causes will not be logged",
                    _LOG_ONCE_CAP,
                )
            return
        self._logged_errors.add(key)
        logger.exception("Frame render failed")

    def request_screenshot(self, path: str) -> None:
        """Latch a screenshot for the next frame this loop fans out.

        Called from the control thread, on `/capture/screenshot`. One slot,
        latest wins: a second request before a frame has been rendered
        replaces the first rather than queuing, the same way the preview
        mailbox holds one frame. A request made while nothing is rendering
        keeps waiting until a model produces one.
        """
        with self._screenshot_lock:
            self._screenshot_path = str(path)

    def _serve_screenshot(self, frame: np.ndarray) -> None:
        """Hand `frame` to the writer if a screenshot is latched.

        Runs on the render thread after the fan-out, so the screenshot is the
        same array the show and the recording just got, and the writer is
        expected to return immediately.
        """
        with self._screenshot_lock:
            path = self._screenshot_path
            self._screenshot_path = None
        if path is None:
            return
        if self._screenshot is None:
            logger.warning("Ignoring a screenshot request, nothing is wired to save it")
            return
        try:
            self._screenshot(path, frame)
        except Exception:
            logger.exception("Handing over the screenshot %s failed", path)

    def fps(self) -> float:
        return self._fps

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while self._running.is_set():
            produced = self.render_one()
            if not produced:
                time.sleep(_IDLE_SLEEP)

    def _track_fps(self) -> None:
        self._fps_window_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps = self._fps_window_frames / elapsed
            self._fps_window_frames = 0
            self._fps_window_start = now

    def _limit_framerate(self, fps_cap: int) -> None:
        if fps_cap <= 0:
            self._next_deadline = None
            return
        period = 1.0 / fps_cap
        if self._next_deadline is None:
            self._next_deadline = time.perf_counter() + period
        remaining = self._next_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        self._next_deadline += period
        now = time.perf_counter()
        if self._next_deadline < now:
            self._next_deadline = now + period
