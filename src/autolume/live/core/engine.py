"""The render loop: derives each frame from one atomic RenderParams snapshot.

Self-paces against the fps cap with absolute-deadline scheduling so
post-render sink work is absorbed into the next frame's sleep instead of
stacking on the cap (pacing strategy ported from balagan engine.py).
"""

import logging
import threading
import time

from autolume.live.core.params import RenderParams
from autolume.live.core.sinks import FrameSink
from autolume.live.core.store import LatestValueStore

logger = logging.getLogger(__name__)

_IDLE_SLEEP = 0.05


class RenderLoop:
    def __init__(
        self,
        render_store: LatestValueStore[RenderParams],
        model_host,
        sinks: list[FrameSink],
    ) -> None:
        self._render_store = render_store
        self._model_host = model_host
        self._sinks = sinks
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
        try:
            frame = model.render_frame(params, self._seq)
        except Exception:
            logger.exception("Frame render failed")
            self._next_deadline = None
            return False
        self._seq += 1
        for sink in self._sinks:
            try:
                sink.on_frame(frame, self._seq)
            except Exception:
                logger.exception("Frame sink %r failed", sink)
        self._track_fps()
        self._limit_framerate(params.fps_cap)
        return True

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
