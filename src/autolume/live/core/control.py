"""The control thread: the runtime's fixed-rate heartbeat.

All control producers submit events here. Each tick drains the queue,
applies events through the mapping, integrates motion with measured dt,
and publishes fresh immutable snapshots. Render and UI rates never gate
this loop.
"""

import collections
import dataclasses
import threading
import time
from typing import Callable

from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.motion import integrate
from autolume.live.core.params import ControlState, RenderParams, to_render_params
from autolume.live.core.store import LatestValueStore

_QUEUE_LIMIT = 1024


class ControlLoop:
    def __init__(
        self,
        control_store: LatestValueStore[ControlState],
        render_store: LatestValueStore[RenderParams],
        tick_hz: float = 125.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._control_store = control_store
        self._render_store = render_store
        self._period = 1.0 / tick_hz
        self._clock = clock
        self._queue: collections.deque[ControlEvent] = collections.deque(
            maxlen=_QUEUE_LIMIT
        )
        self._queue_lock = threading.Lock()
        self._last_tick: float | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def submit(self, event: ControlEvent) -> None:
        if event.timestamp is None:
            event = dataclasses.replace(event, timestamp=self._clock())
        with self._queue_lock:
            self._queue.append(event)

    def tick(self) -> RenderParams:
        now = self._clock()
        dt = now - self._last_tick if self._last_tick is not None else 0.0
        self._last_tick = now

        with self._queue_lock:
            events = list(self._queue)
            self._queue.clear()

        state = self._control_store.snapshot()
        for event in events:
            state = apply_event(state, event)
        state = integrate(state, dt)
        self._control_store.set(state)
        render_params = to_render_params(state)
        self._render_store.set(render_params)
        return render_params

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        deadline = time.monotonic() + self._period
        while self._running.is_set():
            self.tick()
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            deadline += self._period
            now = time.monotonic()
            if deadline < now:
                deadline = now + self._period
