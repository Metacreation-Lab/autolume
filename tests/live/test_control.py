import itertools
import logging
import time

from autolume.live.core import control as control_module
from autolume.live.core.control import ControlLoop
from autolume.live.core.events import ControlEvent
from autolume.live.core.expr import compile_expression
from autolume.live.core.params import (
    BINDING_SET,
    Binding,
    ControlState,
    RenderParams,
    to_render_params,
)
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.core.touch import (
    TOUCH_BEGIN,
    TOUCH_END,
    TOUCH_GRACE,
    TOUCH_HOLD_LIMIT,
)

_real_apply_event = control_module.apply_event


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class CountingStore(LatestValueStore):
    def __init__(self, initial):
        super().__init__(initial)
        self.sets = 0

    def set(self, value):
        self.sets += 1
        super().set(value)


def make_loop(clock=None, store=LatestValueStore):
    control_store = store(ControlState())
    render_store = store(to_render_params(ControlState()))
    source_store = store(SourceTable())
    loop = ControlLoop(
        control_store, render_store, source_store, clock=clock or FakeClock()
    )
    return loop, control_store, render_store, source_store


def bind(loop, target, source, expression="x", enabled=True):
    loop.submit(
        ControlEvent(
            BINDING_SET,
            Binding(
                target=target, source=source, expression=expression, enabled=enabled
            ),
        )
    )


def binding_for(state, target):
    return next(binding for binding in state.bindings if binding.target == target)


def test_tick_applies_events_in_order():
    loop, control_store, _, _ = make_loop()
    loop.submit(ControlEvent("/latent/x", 1.0))
    loop.submit(ControlEvent("/latent/x", 2.0))
    loop.tick()
    assert control_store.snapshot().latent_x == 2.0


def test_tick_publishes_render_params():
    loop, _, render_store, _ = make_loop()
    loop.submit(ControlEvent("/trunc/psi", 1.2))
    result = loop.tick()
    assert isinstance(result, RenderParams)
    assert render_store.snapshot().truncation_psi == 1.2


def test_tick_integrates_motion_with_measured_dt():
    clock = FakeClock()
    loop, control_store, _, _ = make_loop(clock)
    loop.submit(ControlEvent("/anim/playing", True))
    loop.submit(ControlEvent("/anim/speed/x", 2.0))
    loop.tick()
    clock.now = 0.5
    loop.tick()
    assert abs(control_store.snapshot().latent_x - 1.0) < 1e-9


def test_first_tick_has_zero_dt():
    clock = FakeClock()
    clock.now = 100.0
    loop, control_store, _, _ = make_loop(clock)
    loop.submit(ControlEvent("/anim/playing", True))
    loop.tick()
    assert control_store.snapshot().latent_x == 0.0


def test_submit_overflow_drops_oldest():
    loop, control_store, _, _ = make_loop()
    for i in range(2000):
        loop.submit(ControlEvent("/latent/x", float(i)))
    loop.tick()
    assert control_store.snapshot().latent_x == 1999.0


def test_submit_stamps_timestamp():
    clock = FakeClock()
    clock.now = 42.0
    loop, _, _, _ = make_loop(clock)
    loop.submit(ControlEvent("/latent/x", 1.0))
    event = loop._queue[0]
    assert event.timestamp == 42.0


def test_thread_start_stop():
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    loop = ControlLoop(control_store, render_store, source_store, tick_hz=500.0)
    loop.start()
    loop.submit(ControlEvent("/latent/x", 7.0))
    deadline = time.monotonic() + 2.0
    while control_store.snapshot().latent_x != 7.0 and time.monotonic() < deadline:
        time.sleep(0.005)
    loop.stop()
    assert control_store.snapshot().latent_x == 7.0


def test_unbound_address_only_updates_the_source_table():
    clock = FakeClock()
    clock.now = 5.0
    loop, control_store, _, source_store = make_loop(clock)
    loop.submit(ControlEvent("/audio/level", 0.4))
    loop.tick()
    entry = source_store.snapshot().get("/audio/level")
    assert entry.value == 0.4
    assert entry.timestamp == 5.0
    assert control_store.snapshot() == ControlState()


def test_source_table_is_published_only_when_it_changed():
    loop, _, _, source_store = make_loop()
    loop.tick()
    unchanged = source_store.snapshot()
    loop.tick()
    assert source_store.snapshot() is unchanged
    loop.submit(ControlEvent("/audio/level", 0.4))
    loop.tick()
    assert source_store.snapshot() is not unchanged


def test_binding_drives_its_target():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "x*2")
    loop.submit(ControlEvent("/audio/level", 0.25))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.5


def test_disabled_binding_does_not_fire_until_it_is_enabled():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "x*2", enabled=False)
    loop.submit(ControlEvent("/audio/level", 0.25))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.7

    bind(loop, "truncation_psi", "/audio/level", "x*2", enabled=True)
    loop.submit(ControlEvent("/audio/level", 0.25))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.5


def test_failing_expression_keeps_the_value_and_records_the_error():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "1/x")
    loop.submit(ControlEvent("/audio/level", 0.0))
    loop.tick()
    state = control_store.snapshot()
    assert state.truncation_psi == 0.7
    assert binding_for(state, "truncation_psi").error is not None


def test_successful_evaluation_clears_a_previous_error():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "1/x")
    loop.submit(ControlEvent("/audio/level", 0.0))
    loop.tick()
    loop.submit(ControlEvent("/audio/level", 2.0))
    loop.tick()
    state = control_store.snapshot()
    assert state.truncation_psi == 0.5
    assert binding_for(state, "truncation_psi").error is None


def test_failing_expression_is_logged_once_per_error(caplog):
    loop, _, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "1/x")
    with caplog.at_level(logging.WARNING, logger=control_module.__name__):
        for _ in range(5):
            loop.submit(ControlEvent("/audio/level", 0.0))
            loop.tick()
    logged = [r for r in caplog.records if r.name == control_module.__name__]
    assert len(logged) == 1


def test_held_target_ignores_binding_writes_and_resumes_after_the_grace():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.7

    loop.submit(ControlEvent(TOUCH_END, "truncation_psi"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.7

    clock.now += TOUCH_GRACE
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_held_target_still_accepts_direct_events_on_its_own_address():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi"))
    loop.submit(ControlEvent("/trunc/psi", 1.2))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.2


def test_an_unheld_target_lets_the_binding_overwrite_an_earlier_ui_event():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent("/trunc/psi", 1.2))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_touch_events_from_outside_the_ui_are_ignored():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi", source="osc"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_a_remote_touch_end_cannot_release_a_ui_hold():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi"))
    loop.submit(ControlEvent(TOUCH_END, "truncation_psi", source="osc"))
    loop.tick()
    clock.now += TOUCH_GRACE * 2
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.7


def test_a_hold_that_is_never_ended_lapses_and_the_binding_resumes():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.7

    clock.now += TOUCH_HOLD_LIMIT
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_malformed_touch_event_is_ignored():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, 3.0))
    loop.submit(ControlEvent(TOUCH_BEGIN, "not_a_parameter"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_event_burst_coalesces_into_one_publication():
    clock = FakeClock()
    loop, control_store, render_store, source_store = make_loop(
        clock, store=CountingStore
    )
    bind(loop, "truncation_psi", "/audio/level")
    for i in range(60):
        loop.submit(ControlEvent("/audio/level", i / 100.0))
    loop.tick()
    assert control_store.sets == 1
    assert render_store.sets == 1
    assert source_store.sets == 1
    assert source_store.snapshot().get("/audio/level").value == 0.59
    assert control_store.snapshot().truncation_psi == 0.59


def test_broken_expression_is_compiled_once_not_on_every_tick(monkeypatch):
    compiles = itertools.count()

    def counting_compile(source):
        next(compiles)
        return compile_expression(source)

    monkeypatch.setattr(control_module, "compile_expression", counting_compile)
    loop, _, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "x*")
    for _ in range(10):
        loop.submit(ControlEvent("/audio/level", 0.5))
        loop.tick()
    assert next(compiles) == 1


def test_fixing_a_broken_expression_takes_effect_immediately():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "x*")
    loop.submit(ControlEvent("/audio/level", 0.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.7

    bind(loop, "truncation_psi", "/audio/level", "x*2")
    loop.submit(ControlEvent("/audio/level", 0.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.0


def _poisoned_apply_event(state, event):
    if event.address.startswith("/boom"):
        raise TypeError("unhashable type: 'list'")
    return _real_apply_event(state, event)


def test_event_that_fails_to_apply_is_dropped_and_the_drain_continues(
    monkeypatch, caplog
):
    monkeypatch.setattr(control_module, "apply_event", _poisoned_apply_event)
    loop, control_store, _, _ = make_loop()
    with caplog.at_level(logging.ERROR, logger=control_module.__name__):
        loop.submit(ControlEvent("/boom", 1.0))
        loop.submit(ControlEvent("/latent/x", 3.0))
        loop.tick()
    assert control_store.snapshot().latent_x == 3.0
    assert any("/boom" in record.getMessage() for record in caplog.records)


def test_a_repeatedly_poisoned_event_is_logged_once_not_on_every_tick(
    monkeypatch, caplog
):
    monkeypatch.setattr(control_module, "apply_event", _poisoned_apply_event)
    loop, _, _, _ = make_loop()
    with caplog.at_level(logging.ERROR, logger=control_module.__name__):
        for _ in range(50):
            loop.submit(ControlEvent("/boom", 1.0))
            loop.tick()
    logged = [r for r in caplog.records if r.name == control_module.__name__]
    assert len(logged) == 1
    assert logged[0].exc_info is not None


def test_each_poisoned_address_and_error_is_reported_once(monkeypatch, caplog):
    monkeypatch.setattr(control_module, "apply_event", _poisoned_apply_event)
    loop, _, _, _ = make_loop()
    with caplog.at_level(logging.ERROR, logger=control_module.__name__):
        for _ in range(5):
            loop.submit(ControlEvent("/boom", 1.0))
            loop.submit(ControlEvent("/boom/other", 1.0))
            loop.tick()
    logged = [r for r in caplog.records if r.name == control_module.__name__]
    assert len(logged) == 2


def test_suppressed_event_failures_are_counted_periodically(monkeypatch, caplog):
    monkeypatch.setattr(control_module, "apply_event", _poisoned_apply_event)
    loop, _, _, _ = make_loop()
    with caplog.at_level(logging.ERROR, logger=control_module.__name__):
        for _ in range(control_module._GUARD_REPEAT_INTERVAL):
            loop.submit(ControlEvent("/boom", 1.0))
        loop.tick()
    logged = [r for r in caplog.records if r.name == control_module.__name__]
    assert len(logged) == 2
    assert str(control_module._GUARD_REPEAT_INTERVAL - 1) in logged[1].getMessage()


def test_a_tick_that_keeps_failing_is_logged_once(caplog):
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    loop = ControlLoop(control_store, render_store, source_store, tick_hz=500.0)
    ticks = itertools.count()

    def always_failing_tick():
        next(ticks)
        raise RuntimeError("boom")

    loop.tick = always_failing_tick
    with caplog.at_level(logging.ERROR, logger=control_module.__name__):
        loop.start()
        time.sleep(0.1)
        loop.stop()
    logged = [r for r in caplog.records if r.name == control_module.__name__]
    assert next(ticks) > 5
    assert len(logged) == 1
    assert logged[0].exc_info is not None


def test_control_thread_survives_a_raising_tick():
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    loop = ControlLoop(control_store, render_store, source_store, tick_hz=500.0)
    ticks = itertools.count()
    real_tick = loop.tick

    def flaky_tick():
        if next(ticks) < 3:
            raise RuntimeError("boom")
        return real_tick()

    loop.tick = flaky_tick
    loop.start()
    loop.submit(ControlEvent("/latent/x", 7.0))
    deadline = time.monotonic() + 2.0
    while control_store.snapshot().latent_x != 7.0 and time.monotonic() < deadline:
        time.sleep(0.005)
    loop.stop()
    assert control_store.snapshot().latent_x == 7.0
