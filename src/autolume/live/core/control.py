"""The control thread: the runtime's fixed-rate heartbeat.

All control producers submit events here. Each tick drains the queue,
applies events through the mapping, drives the bindings they feed,
integrates motion with measured dt, and publishes fresh immutable
snapshots. Render and UI rates never gate this loop.

Nothing here may take the thread down: a performance does not stop
because one event was malformed or one mapping had a typo.
"""

import collections
import dataclasses
import logging
import threading
import time
from typing import Callable

from autolume.live.core.events import ControlEvent
from autolume.live.core.expr import ExpressionError, compile_expression
from autolume.live.core.mapping import apply_event
from autolume.live.core.motion import integrate
from autolume.live.core.params import (
    REGISTRY,
    ControlState,
    RenderParams,
    apply_value,
    to_render_params,
)
from autolume.live.core.sources import SourceTable, as_float, canonical_address
from autolume.live.core.store import LatestValueStore
from autolume.live.core.touch import TOUCH_BEGIN, TOUCH_END, TouchTracker

logger = logging.getLogger(__name__)

_QUEUE_LIMIT = 1024
_GUARD_REPEAT_INTERVAL = 1000
_EXPRESSION_LIMIT = 128
_TOUCH_ADDRESSES = (TOUCH_BEGIN, TOUCH_END)


class ControlLoop:
    def __init__(
        self,
        control_store: LatestValueStore[ControlState],
        render_store: LatestValueStore[RenderParams],
        source_store: LatestValueStore[SourceTable],
        tick_hz: float = 125.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._control_store = control_store
        self._render_store = render_store
        self._source_store = source_store
        self._period = 1.0 / tick_hz
        self._clock = clock
        self.touch = TouchTracker()
        self._queue: collections.deque[ControlEvent] = collections.deque(
            maxlen=_QUEUE_LIMIT
        )
        self._queue_lock = threading.Lock()
        self._expressions: dict[str, Callable[[float], float] | str] = {}
        self._logged_errors: set[tuple[str, str]] = set()
        self._guard_hits: dict[tuple[str, str], int] = {}
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
        published_sources = self._source_store.snapshot()
        sources = published_sources
        for event in events:
            # Last resort guard. Validation lives in the mapping, but an event
            # that still manages to raise must not take the control thread with
            # it, so it is dropped and the rest of the drain goes through.
            try:
                state, sources = self._apply(state, sources, event, now)
            except Exception as exc:
                key = (event.address, type(exc).__name__)
                self._report_guard(key, "Dropping control event %r", event)
        state = integrate(state, dt)

        self._control_store.set(state)
        if sources is not published_sources:
            self._source_store.set(sources)
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
            try:
                self.tick()
            except Exception as exc:
                self._report_guard(("tick", type(exc).__name__), "Control tick failed")
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            deadline += self._period
            now = time.monotonic()
            if deadline < now:
                deadline = now + self._period

    def _apply(
        self, state: ControlState, sources: SourceTable, event: ControlEvent, now: float
    ) -> tuple[ControlState, SourceTable]:
        # Canonicalized once, here, so the table, the parameter lookup and the
        # bindings all match on the same spelling. An address canonical on one
        # side of a comparison only is a mapping that silently never fires.
        address = canonical_address(event.address)
        if address != event.address:
            event = dataclasses.replace(event, address=address)
        if address in _TOUCH_ADDRESSES:
            self._track_touch(event, now)
            return state, sources
        number = as_float(event.value)
        if number is None:
            return apply_event(state, event), sources
        stamp = now if event.timestamp is None else event.timestamp
        sources = sources.observe(address, number, stamp)
        state = apply_event(state, event)
        return self._drive_bindings(state, address, number, now), sources

    def _report_guard(self, key: tuple[str, str], message: str, *args: object) -> None:
        """Log a last resort guard hit, throttled to protect the tick budget.

        The first hit of a `key` carries the full traceback, then only every
        `_GUARD_REPEAT_INTERVAL`-th one reports, with the count it stands for.
        Formatting a traceback costs hundreds of microseconds, so a poisonous
        input arriving at 200 Hz would otherwise eat the heartbeat that these
        guards exist to keep alive.

        Must be called from an `except` block: the first report uses the
        exception currently being handled.
        """
        count = self._guard_hits.get(key, 0) + 1
        if count == 1 and len(self._guard_hits) >= _EXPRESSION_LIMIT:
            self._guard_hits.clear()
        self._guard_hits[key] = count
        if count == 1:
            logger.exception(message, *args)
        elif count % _GUARD_REPEAT_INTERVAL == 0:
            logger.error(
                message + " (%d more suppressed)", *args, _GUARD_REPEAT_INTERVAL - 1
            )

    def _track_touch(self, event: ControlEvent, now: float) -> None:
        name = event.value
        # Touch is a UI concept. Accepting it from anywhere else would let an
        # open OSC port wedge a parameter by beginning a touch it never ends.
        if event.source != "ui":
            logger.debug("Ignoring %s from %s", event.address, event.source)
            return
        # Only registry parameters can be held, which also bounds the tracker.
        if not isinstance(name, str) or name not in REGISTRY:
            logger.debug("Ignoring touch event for %r", name)
            return
        if event.address == TOUCH_BEGIN:
            self.touch.begin(name, now)
        else:
            self.touch.end(name, now)

    def _drive_bindings(
        self, state: ControlState, address: str, value: float, now: float
    ) -> ControlState:
        for binding in state.bindings:
            if not binding.enabled or binding.source != address:
                continue
            if self.touch.is_held(binding.target, now):
                continue
            try:
                result = self._compile(binding.expression)(value)
            except ExpressionError as exc:
                state = self._record_error(state, binding.target, str(exc))
                continue
            state = apply_value(state, binding.target, result)
            state = self._record_error(state, binding.target, None)
        return state

    def _compile(self, source: str) -> Callable[[float], float]:
        """Compile a mapping expression, memoizing failures as well as hits.

        `compile_expression` caches what it compiles but not the errors it
        raises, so without this a broken expression would be parsed again on
        every tick for as long as the binding exists.
        """
        compiled = self._expressions.get(source)
        if compiled is None:
            try:
                compiled = compile_expression(source)
            except ExpressionError as exc:
                compiled = str(exc)
            if len(self._expressions) >= _EXPRESSION_LIMIT:
                self._expressions.clear()
            self._expressions[source] = compiled
        if isinstance(compiled, str):
            raise ExpressionError(compiled)
        return compiled

    def _record_error(
        self, state: ControlState, target: str, error: str | None
    ) -> ControlState:
        """Put `error` (or None to clear it) on the binding driving `target`.

        Logged once per target and error text, never once per message: a broken
        expression fires as often as its source does.
        """
        if error is not None and (target, error) not in self._logged_errors:
            if len(self._logged_errors) >= _EXPRESSION_LIMIT:
                self._logged_errors.clear()
            self._logged_errors.add((target, error))
            logger.warning("Binding for %s failed: %s", target, error)
        bindings = list(state.bindings)
        for index, binding in enumerate(bindings):
            if binding.target != target:
                continue
            if binding.error == error:
                return state
            bindings[index] = dataclasses.replace(binding, error=error)
            return dataclasses.replace(state, bindings=tuple(bindings))
        return state
