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
import math
import threading
import time
from typing import Callable

import numpy as np

from autolume.live.core.events import ControlEvent
from autolume.live.core.expr import ExpressionError, compile_expression
from autolume.live.core.generator import ModelInfo
from autolume.live.core.loop import LoopStep, advance
from autolume.live.core.mapping import apply_event
from autolume.live.core.models import ModelFolder
from autolume.live.core.motion import WalkState, integrate
from autolume.live.core.noiseloop import NoiseLoopTableBuilder
from autolume.live.core.params import (
    BY_ADDRESS,
    REGISTRY,
    VECTOR_RANDOMIZE,
    Binding,
    ControlState,
    ParamKind,
    RenderParams,
    apply_value,
    listens_on,
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
# np.random.RandomState only accepts a uint32 seed; an OSC int can be anything.
_SEED_MASK = (1 << 32) - 1


class ControlLoop:
    def __init__(
        self,
        control_store: LatestValueStore[ControlState],
        render_store: LatestValueStore[RenderParams],
        source_store: LatestValueStore[SourceTable],
        tick_hz: float = 125.0,
        clock: Callable[[], float] = time.monotonic,
        models: ModelFolder | None = None,
        model_info_store: LatestValueStore[ModelInfo | None] | None = None,
        walk_rng: np.random.RandomState | None = None,
        noise_table_builder: NoiseLoopTableBuilder | None = None,
        emit: Callable[[str, int, str, float], None] | None = None,
    ) -> None:
        self._control_store = control_store
        self._render_store = render_store
        self._source_store = source_store
        self._period = 1.0 / tick_hz
        self._clock = clock
        # What a row on a text parameter resolves against. Held rather than
        # looked up per message: it caches its listing, which is what keeps a
        # swept fader from reading the folder once per value.
        self._models = models or ModelFolder()
        # Read-only; ModelHost is the sole writer. Refreshed once per tick so
        # later motion writers (Plan 3+) see a stable value for the whole tick
        # rather than racing the loader thread mid computation.
        self._model_info_store = model_info_store or LatestValueStore(None)
        self._model_info: ModelInfo | None = self._model_info_store.snapshot()
        self.touch = TouchTracker()
        # Seeded once at construction, never persisted: the vector walk's
        # target is runtime-only state, not part of the show (design.md).
        self.walk = WalkState(walk_rng or np.random.RandomState())
        # The loop step from the most recent tick, including `started` and
        # `wrapped` resolved against the tick before it. Task 7 reads this to
        # decide whether the outbound pulse fires.
        self._last_loop_step = LoopStep(
            alpha=0.0, index=0, wrapped=False, started=False
        )
        # The noise loop's table is (re)built off this thread by the builder
        # (noiseloop.py); requested only when the key it was built from
        # changes. While a build is in flight the previously published table
        # keeps serving, so a radius change never stalls or freezes motion.
        self._noise_table_builder = noise_table_builder or NoiseLoopTableBuilder()
        self._noise_table_key: tuple[int, float, int] | None = None
        # Injected so tests never open a socket; the runtime binds an
        # `OscEmitter().send`. None means the pulse is not wired at all.
        self._emit = emit
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

    @property
    def model_info(self) -> ModelInfo | None:
        """The loaded model's dimensions, as of the most recent tick."""
        return self._model_info

    @property
    def last_loop_step(self) -> LoopStep:
        """The loop's `started`/`wrapped` result from the most recent tick.

        Exposed for introspection and tests. The outbound pulse (Task 7) does
        not read it: deciding whether a `wrapped` edge is trustworthy needs
        `prior_alpha` and the touch state, which only exist inside
        `_integrate_loop` while the step is still fresh, so `_emit_pulse` runs
        there instead of re-deriving that context from this property later.
        """
        return self._last_loop_step

    @property
    def noise_table_key(self) -> tuple[int, float, int] | None:
        """The `(seed, radius, z_dim)` the published noise table was built from.

        None before any table has finished building. Read-only, through the
        same `LatestValueStore.snapshot()` every other cross-thread read in
        this module already goes through, so a UI can compare it against the
        key the current state and model would build and show a rebuild
        pending indicator (Task 9) without reaching into a private field or
        opening a new channel.
        """
        table = self._noise_table_builder.store.snapshot()
        return None if table is None else table.key

    def tick(self) -> RenderParams:
        now = self._clock()
        dt = now - self._last_tick if self._last_tick is not None else 0.0
        self._last_tick = now
        self._model_info = self._model_info_store.snapshot()

        with self._queue_lock:
            events = list(self._queue)
            self._queue.clear()

        state = self._control_store.snapshot()
        was_loop_active = state.loop_active
        prior_loop_alpha = state.loop_alpha
        prior_loop_index = state.loop_index
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
        state = integrate(
            state, dt, self.touch, now, model_info=self._model_info, walk=self.walk
        )
        state, self._last_loop_step = self._integrate_loop(
            state, dt, now, was_loop_active, prior_loop_alpha, prior_loop_index
        )

        self._control_store.set(state)
        if sources is not published_sources:
            self._source_store.set(sources)
        render_params = to_render_params(state)
        if state.loop_active and state.noise_loop:
            vector = self._noise_latent_vector(state)
            if vector is not None:
                render_params = dataclasses.replace(render_params, latent_vec=vector)
        self._render_store.set(render_params)
        return render_params

    def _integrate_loop(
        self,
        state: ControlState,
        dt: float,
        now: float,
        was_active: bool,
        prior_alpha: float,
        prior_index: int,
    ) -> tuple[ControlState, LoopStep]:
        """Advance the keyframe or noise loop and land the result in `state`.

        `advance` is pure and never sets `started`, since it cannot see the
        tick before this one; this is the one place that can, by comparing
        `was_active` (captured before this tick's events ran) to `loop_active`
        now.

        `prior_alpha`/`prior_index` are the same before-this-tick values, so a
        manual write to `/loop/alpha` or `/loop/index` earlier this tick is
        detected as "already changed" and integration leaves it alone,
        exactly as a manual write outruns a binding within one tick.

        `alpha_is_integrated` is also what decides whether the outbound pulse
        trusts `step.wrapped` (see `_emit_pulse`). `advance` computes it from
        `state.loop_alpha` as it stands here, which is this tick's value after
        events but before this check: while a performer holds `loop_alpha`
        near a segment boundary, that value never moves, but the tiny
        integration rate added to it can still tip `divmod` over the edge
        every tick, and the same is true the tick a manual write lands
        elsewhere in-bounds. Neither is a completed cycle, so the pulse must
        not read `wrapped` on its own; the alpha write below is gated on the
        same condition for the same reason, and reusing it keeps the two
        decisions from drifting apart.

        This is also why a binding that writes `loop_alpha` every tick
        permanently suppresses both the pulse and the perfect-loop stop:
        `alpha_is_integrated` is false on any tick an event changed the
        value, and a binding firing every tick means every tick is such a
        tick. Deliberate, per the same scrub ruling above: a value under
        continuous outside control is being scrubbed by definition, and must
        not stop the show or fire a pulse out from under whatever is driving
        it. Do not change this without revisiting that ruling.
        """
        step = advance(state, dt)
        if state.loop_active and not was_active:
            step = dataclasses.replace(step, started=True)
        alpha_is_integrated = (
            state.loop_alpha == prior_alpha
            and not self.touch.is_held("loop_alpha", now)
        )
        if alpha_is_integrated:
            state = apply_value(state, "loop_alpha", step.alpha)
        if state.loop_index == prior_index and not self.touch.is_held(
            "loop_index", now
        ):
            state = apply_value(state, "loop_index", step.index)
        # Gated on the same `alpha_is_integrated` as the pulse: a scrub-induced
        # `wrapped` is not a completed cycle, so it must not silently stop
        # playback either. Stopping on a false wrap is the worse failure of
        # the two (a stray pulse is noise, a stopped show is a mistake with
        # nothing to explain it), so this shares the guard rather than only
        # the pulse getting it.
        if step.wrapped and alpha_is_integrated and state.perfect_loop:
            state = apply_value(state, "loop_active", False)
        self._emit_pulse(state, step, alpha_is_integrated)
        return state, step

    def _emit_pulse(
        self, state: ControlState, step: LoopStep, alpha_is_integrated: bool
    ) -> None:
        """Fire the outbound sync pulse for this tick's play/wrap edges.

        One message per event, never a stream: `step.started` and
        `step.wrapped` are each true on at most the one tick their edge
        happened. `wrapped` only trusts an integration-driven crossing
        (`alpha_is_integrated`, see `_integrate_loop`) so a hand scrubbing
        `loop_alpha` across a boundary never reads to downstream gear as a
        completed cycle. `started` has no such hazard: it comes from
        comparing `loop_active` across the tick, which a scrub never touches.
        """
        if self._emit is None:
            return
        address = canonical_address(state.pulse_address)
        if not address:
            return
        if step.started:
            self._safe_emit(state, address, 2.0)
        if step.wrapped and alpha_is_integrated:
            self._safe_emit(state, address, 1.0)

    def _safe_emit(self, state: ControlState, address: str, value: float) -> None:
        try:
            self._emit(state.pulse_ip, state.pulse_port, address, value)
        except Exception as exc:
            self._report_guard(("pulse", type(exc).__name__), "Outbound pulse failed")

    def _noise_latent_vector(self, state: ControlState) -> tuple[float, ...] | None:
        """The noise loop's vector for this tick, or None if it cannot be built.

        Reads whatever table `_noise_table_builder` currently has published;
        never builds one itself and never waits on the builder's thread. None
        means "no model yet", "a non positive radius", or "no table has
        finished building yet" — in every case `render_params.latent_vec` is
        left at `state.latent_vec` rather than raising, since a live show
        cannot stop for any of them.
        """
        info = self._model_info
        if info is None:
            return None
        radius = state.noise_radius
        if not math.isfinite(radius) or radius <= 0.0:
            return None
        key = (state.noise_loop_seed, radius, info.z_dim)
        if key != self._noise_table_key:
            self._noise_table_key = key
            self._noise_table_builder.request_build(key)
        table = self._noise_table_builder.store.snapshot()
        if table is None:
            return None
        return table.vector(state.loop_alpha)

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._noise_table_builder.stop()

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
        remote = event.source != "ui"
        # The source table is the picker's list of available inputs, so what
        # this app writes out has no business in it. Recording UI events would
        # fill it with the parameters' own transport addresses and put binding
        # a parameter to itself two clicks away.
        #
        # Recorded before the gate below, and this ordering is now load
        # bearing rather than tidy: with remote input off until a row says
        # otherwise, every address a performer is trying to set up arrives
        # blocked. Record only what was accepted and the picker would be empty
        # for exactly the controller they are pointing at us, and the gutter
        # would have nothing to show them either.
        if number is not None and remote:
            stamp = now if event.timestamp is None else event.timestamp
            sources = sources.observe(address, number, stamp)
        if self._accepts_direct(address, remote):
            if address == VECTOR_RANDOMIZE:
                state = self._randomize_vector(state, number)
            else:
                state = apply_event(state, event)
        # The raw value travels alongside the number it may not be. A text
        # value carries no number and used to stop here, which left a text
        # parameter with no remote path at all once the gate closed.
        state = self._drive_bindings(state, address, event.value, number, now, remote)
        return state, sources

    def _accepts_direct(self, address: str, remote: bool) -> bool:
        """Whether an event may write the parameter its own address names.

        The performer's hand is never gated: a UI event passes here whatever
        the mapping says, because a mapping row stops the network, not the
        mouse.

        Nothing remote passes. A parameter is written from outside only through
        its mapping row, and a parameter with no row has not been offered to
        the network at all, so a controller that finds the port cannot move a
        show nobody pointed at it. `_drive_bindings` is the one remote path,
        and it applies the row's switch, its expression and the touch grace.

        Everything that is not a parameter address passes: the structured
        addresses carry their own validation, and an unknown one is dropped
        downstream where it is logged.
        """
        if not remote:
            return True
        return BY_ADDRESS.get(address) is None

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

    def _randomize_vector(
        self, state: ControlState, seed_number: float | None
    ) -> ControlState:
        """Materialize `/vector/randomize` now that `ModelInfo` can be reached.

        `mapping.apply_event` recognizes this address but leaves state
        untouched: it has no way to reach the loaded model's `z_dim`. This is
        where the event actually does something, so a `latent_vec` that never
        changes after this call is a real no-op, not the mapping's inert one.
        """
        info = self._model_info
        if info is None:
            logger.info("Ignoring %s, no model is loaded yet", VECTOR_RANDOMIZE)
            return state
        if seed_number is None or not math.isfinite(seed_number):
            logger.warning("Ignoring non numeric seed on %s", VECTOR_RANDOMIZE)
            return state
        seed = int(round(seed_number)) & _SEED_MASK
        vec = tuple(np.random.RandomState(seed).randn(info.z_dim).tolist())
        return dataclasses.replace(state, latent_vec=vec)

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
        self,
        state: ControlState,
        address: str,
        value: object,
        number: float | None,
        now: float,
        remote: bool,
    ) -> ControlState:
        """Run every enabled row listening on `address`.

        This is the only way anything remote reaches a parameter, so a
        parameter with no row here is one nothing outside can move.

        A row with no source listens on its parameter's own address, and only
        for remote input: its expression is there to shape what arrives from
        outside, and running it over the value the performer just set by hand
        would turn their own control against them.

        `number` is `value` as a float where it is one, and None where it is
        not. Both are passed because a text parameter is driven by either.
        """
        for binding in state.bindings:
            if not binding.enabled:
                continue
            if not binding.source and not remote:
                continue
            if listens_on(binding) != address:
                continue
            if self.touch.is_held(binding.target, now):
                continue
            spec = REGISTRY.get(binding.target)
            if spec is not None and spec.kind is ParamKind.STR:
                state = self._drive_reference(state, binding, value, number)
                continue
            if number is None:
                continue
            try:
                result = self._compile(binding.expression)(number)
            except ExpressionError as exc:
                state = self._record_error(state, binding.target, str(exc))
                continue
            state = apply_value(state, binding.target, result)
            state = self._record_error(state, binding.target, None)
        return state

    def _drive_reference(
        self,
        state: ControlState,
        binding: Binding,
        value: object,
        number: float | None,
    ) -> ControlState:
        """Resolve a model reference onto a text parameter, or leave it alone.

        A number is an index into the models folder and the row's expression
        applies to it, which is the whole of why this is worth having: nearly
        every controller a performer owns sends numbers, and `x*4` over a fader
        is what turns one of them into a selector across five models.

        A text value names a model and no expression applies, because an
        expression yields a number and there is nothing it could do to a name.
        The mapping row says so beside the field rather than leaving a live
        looking field that quietly does nothing.

        A reference that resolves to nothing is logged where it is resolved and
        ignored here. It is not recorded as a binding error: an index off the
        end is what the top of a wrongly scaled fader sends on the way past, so
        marking the row failing would flash a red row through every sweep.

        Loading is not started here and cannot be: this writes the path into
        the state, and the runtime notices the change on its own tick and hands
        it to the model host, which loads on its own thread. The control thread
        never waits on a file.
        """
        if isinstance(value, str):
            resolved = self._models.named(value)
        elif number is not None:
            try:
                index = self._compile(binding.expression)(number)
            except ExpressionError as exc:
                return self._record_error(state, binding.target, str(exc))
            resolved = self._models.at_index(index)
        else:
            return state
        if resolved is None:
            return state
        state = apply_value(state, binding.target, resolved)
        return self._record_error(state, binding.target, None)

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
