import itertools
import logging
import time

import numpy as np

from autolume.live.core import control as control_module
from autolume.live.core.control import ControlLoop
from autolume.live.core.events import ControlEvent
from autolume.live.core.expr import compile_expression
from autolume.live.core.generator import ModelInfo
from autolume.live.core.noiseloop import NoiseLoop
from autolume.live.core.params import (
    BINDING_SET,
    VECTOR_RANDOMIZE,
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

# What truncation_psi reads as when nothing has written to it. These tests use
# that parameter as a stand-in for "the write did not land", so the value has to
# follow the registry: pinning a number here would turn every future change of
# the default into a wall of unrelated failures.
UNWRITTEN = ControlState().truncation_psi

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


def make_loop_with_state(clock, state):
    control_store = LatestValueStore(state)
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    loop = ControlLoop(control_store, render_store, source_store, clock=clock)
    return loop, control_store, render_store, source_store


def make_loop_with_state_and_model(clock, state, info):
    control_store = LatestValueStore(state)
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    model_info_store = LatestValueStore(info)
    loop = ControlLoop(
        control_store,
        render_store,
        source_store,
        clock=clock,
        model_info_store=model_info_store,
    )
    return loop, control_store, render_store, source_store


def make_loop_with_model(clock, info):
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    model_info_store = LatestValueStore(info)
    loop = ControlLoop(
        control_store,
        render_store,
        source_store,
        clock=clock,
        model_info_store=model_info_store,
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
    loop.submit(ControlEvent("/latent/x", 1.0, source="ui"))
    loop.submit(ControlEvent("/latent/x", 2.0, source="ui"))
    loop.tick()
    assert control_store.snapshot().latent_x == 2.0


def test_tick_publishes_render_params():
    loop, _, render_store, _ = make_loop()
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="ui"))
    result = loop.tick()
    assert isinstance(result, RenderParams)
    assert render_store.snapshot().truncation_psi == 1.2


def test_tick_integrates_motion_with_measured_dt():
    clock = FakeClock()
    loop, control_store, _, _ = make_loop(clock)
    loop.submit(ControlEvent("/anim/playing", True, source="ui"))
    loop.submit(ControlEvent("/anim/speed/x", 2.0, source="ui"))
    loop.tick()
    clock.now = 0.5
    loop.tick()
    assert abs(control_store.snapshot().latent_x - 1.0) < 1e-9


def test_motion_leaves_alone_the_parameter_the_hand_is_holding():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    loop.submit(ControlEvent("/anim/playing", True, source="ui"))
    loop.submit(ControlEvent("/anim/speed/x", 2.0, source="ui"))
    loop.submit(ControlEvent(TOUCH_BEGIN, "latent_x", source="ui"))
    loop.tick()
    clock.now += 0.5
    loop.tick()
    assert control_store.snapshot().latent_x == 0.0

    loop.submit(ControlEvent(TOUCH_END, "latent_x", source="ui"))
    loop.tick()
    clock.now += TOUCH_GRACE + 0.5
    loop.tick()
    assert control_store.snapshot().latent_x > 0.0


def test_motion_leaves_alone_a_parameter_a_binding_is_driving():
    clock = FakeClock()
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "latent_x", "/audio/level")
    loop.submit(ControlEvent("/anim/playing", True, source="ui"))
    loop.submit(ControlEvent("/anim/speed/x", 2.0, source="ui"))
    loop.submit(ControlEvent("/audio/level", 1.5, source="osc"))
    loop.tick()
    clock.now = 0.5
    loop.tick()
    assert control_store.snapshot().latent_x == 1.5


def test_first_tick_has_zero_dt():
    clock = FakeClock()
    clock.now = 100.0
    loop, control_store, _, _ = make_loop(clock)
    loop.submit(ControlEvent("/anim/playing", True, source="ui"))
    loop.tick()
    assert control_store.snapshot().latent_x == 0.0


def test_submit_overflow_drops_oldest():
    loop, control_store, _, _ = make_loop()
    for i in range(2000):
        loop.submit(ControlEvent("/latent/x", float(i), source="ui"))
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
    loop.submit(ControlEvent("/latent/x", 7.0, source="ui"))
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


def test_touching_a_control_does_not_offer_it_as_an_input_source():
    """The picker is a list of inputs, not an echo of the app's own output.

    Every parameter has a transport address, so recording UI events would fill
    the picker with the very addresses the app writes, and binding a parameter
    to itself would be two clicks away.
    """
    clock = FakeClock()
    clock.now = 5.0
    loop, _, _, source_store = make_loop(clock)
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="ui"))
    loop.tick()
    assert source_store.snapshot().recent(clock.now) == []

    loop.submit(ControlEvent("/trunc/psi", 1.2, source="osc"))
    loop.tick()
    assert source_store.snapshot().recent(clock.now) == ["/trunc/psi"]


def test_a_slash_less_address_reaches_the_binding_it_is_listed_as():
    """The picker offers what the table stored, so a binding on it must fire.

    An inbound address without a leading slash is listed as `/audio/level`,
    picked as `/audio/level`, and stored as `/audio/level`. If the loop drives
    bindings with the raw text the mapping is dead with nothing to show for it.
    """
    loop, control_store, _, source_store = make_loop()
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent("audio/level", 1.5, source="osc"))
    loop.tick()
    assert source_store.snapshot().get("/audio/level").value == 1.5
    assert control_store.snapshot().truncation_psi == 1.5


def test_a_slash_less_address_still_reaches_its_own_parameter():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "", enabled=True)
    loop.submit(ControlEvent("trunc/psi", 1.2, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.2


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
    assert control_store.snapshot().truncation_psi == UNWRITTEN

    bind(loop, "truncation_psi", "/audio/level", "x*2", enabled=True)
    loop.submit(ControlEvent("/audio/level", 0.25))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.5


def test_a_stream_moves_nothing_until_a_row_is_switched_on():
    """The reported bug, in the terms it was hit in.

    TouchDesigner streaming /anim/playing at a runtime nobody had configured,
    the Animate box ticking itself on and refusing to stay off, and no way in
    the app to refuse it. A parameter now sits still until it is asked to
    move, and the row is where the asking happens.
    """
    loop, control_store, _, _ = make_loop()
    loop.submit(ControlEvent("/anim/playing", 1.0, source="osc"))
    loop.tick()
    assert control_store.snapshot().anim_playing is False

    bind(loop, "anim_playing", "", enabled=True)
    loop.submit(ControlEvent("/anim/playing", 1.0, source="osc"))
    loop.tick()
    assert control_store.snapshot().anim_playing is True


def test_a_row_switched_off_stops_input_on_the_parameters_own_address():
    # Off again is off again, whether it was never on or was just turned back.
    loop, control_store, _, _ = make_loop()
    bind(loop, "anim_playing", "", enabled=True)
    loop.submit(ControlEvent("/anim/playing", 1.0, source="osc"))
    loop.tick()
    assert control_store.snapshot().anim_playing is True

    bind(loop, "anim_playing", "", enabled=False)
    loop.submit(ControlEvent("/anim/playing", 0.0, source="ui"))
    loop.tick()
    assert control_store.snapshot().anim_playing is False

    loop.submit(ControlEvent("/anim/playing", 1.0, source="osc"))
    loop.tick()
    assert control_store.snapshot().anim_playing is False


def test_a_row_switched_off_still_answers_the_hand():
    # The switch stops the network, never the mouse. A performer who cannot
    # move a control they have just taken off the network has lost it entirely.
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "", enabled=False)
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="ui"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.2


def test_a_row_switched_on_lets_input_through_the_parameters_own_address():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "", enabled=True)
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.2


def test_an_unmapped_parameter_ignores_its_own_address():
    """The default, and the point of the whole change.

    A controller that finds the port cannot move a parameter nobody offered
    it. On a shared network at a venue that is the difference between a show
    and someone else's stray traffic.
    """
    loop, control_store, _, _ = make_loop()
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN


def test_an_unmapped_parameter_still_answers_the_hand():
    # The default takes the parameter off the network, not away from the
    # performer. Nothing about the switch is allowed to reach the mouse.
    loop, control_store, _, _ = make_loop()
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="ui"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.2


def test_a_row_with_no_source_runs_its_expression_on_remote_input():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "", "x*2")
    loop.submit(ControlEvent("/trunc/psi", 0.25, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.5


def test_a_row_with_no_source_leaves_what_the_hand_sets_alone():
    # The expression shapes what arrives from outside. Running it over the
    # value the performer just set would turn their own control against them.
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "", "x*2")
    loop.submit(ControlEvent("/trunc/psi", 0.25, source="ui"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.25


def test_a_row_pointed_elsewhere_closes_the_parameters_own_address():
    # One driver per parameter: the row names the one address that reaches it,
    # and a bound control is read only to the hand for the same reason.
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN


def test_a_held_parameter_ignores_remote_writes_to_its_own_address():
    """Touch grace, on the one writer that can now reach a live control.

    A bound control is drawn read only, so the hand and a binding never share
    one. A remote writer on the parameter's own address does share it, by
    design, and without this the drag would fight the stream frame by frame.
    """
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "", enabled=True)
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi", source="ui"))
    loop.submit(ControlEvent("/trunc/psi", 1.5, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN

    loop.submit(ControlEvent(TOUCH_END, "truncation_psi", source="ui"))
    loop.submit(ControlEvent("/trunc/psi", 1.5, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN

    clock.now += TOUCH_GRACE
    loop.submit(ControlEvent("/trunc/psi", 1.5, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_a_held_parameter_still_takes_the_value_the_hand_is_dragging():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi", source="ui"))
    loop.submit(ControlEvent("/trunc/psi", 1.5, source="osc"))
    loop.submit(ControlEvent("/trunc/psi", 0.3, source="ui"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 0.3


def test_a_blocked_remote_write_is_still_offered_as_an_input_source():
    # The picker lists what arrived, not what was accepted, or an address a row
    # has switched off could never be picked as another parameter's source.
    clock = FakeClock()
    clock.now = 5.0
    loop, control_store, _, source_store = make_loop(clock)
    bind(loop, "truncation_psi", "", enabled=False)
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN
    assert source_store.snapshot().recent(clock.now) == ["/trunc/psi"]


def test_traffic_at_a_parameter_with_no_row_is_still_recorded_as_a_source():
    """The one thing that must not be gated along with the write.

    Every parameter starts deaf, so every address a performer is trying to set
    up arrives blocked. If recording followed acceptance the picker would be
    empty for exactly the controller they are pointing at us, and the gutter
    would have nothing to show them either. They would be left telling a wrong
    port from a wrong address from a switch they have not found, with no
    evidence at all.
    """
    clock = FakeClock()
    clock.now = 5.0
    loop, control_store, _, source_store = make_loop(clock)
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="osc"))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN
    table = source_store.snapshot()
    assert table.recent(clock.now) == ["/trunc/psi"]
    assert table.active("/trunc/psi", clock.now)


def test_failing_expression_keeps_the_value_and_records_the_error():
    loop, control_store, _, _ = make_loop()
    bind(loop, "truncation_psi", "/audio/level", "1/x")
    loop.submit(ControlEvent("/audio/level", 0.0))
    loop.tick()
    state = control_store.snapshot()
    assert state.truncation_psi == UNWRITTEN
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
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi", source="ui"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN

    loop.submit(ControlEvent(TOUCH_END, "truncation_psi", source="ui"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN

    clock.now += TOUCH_GRACE
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_held_target_still_accepts_direct_events_on_its_own_address():
    # From the ui, which is the only writer the hold does not stand against.
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi", source="ui"))
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="ui"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.2


def test_an_unheld_target_lets_the_binding_overwrite_an_earlier_ui_event():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent("/trunc/psi", 1.2, source="ui"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_a_touch_event_that_does_not_declare_a_source_is_ignored():
    """Privilege has to be asked for, never inherited from a default.

    The touch gate is the one thing standing between an open OSC port and a
    parameter wedged for the rest of the show, so the next producer someone
    adds must not pass it by forgetting a keyword.
    """
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi"))
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
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi", source="ui"))
    loop.submit(ControlEvent(TOUCH_END, "truncation_psi", source="osc"))
    loop.tick()
    clock.now += TOUCH_GRACE * 2
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN


def test_a_hold_that_is_never_ended_lapses_and_the_binding_resumes():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, "truncation_psi", source="ui"))
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == UNWRITTEN

    clock.now += TOUCH_HOLD_LIMIT
    loop.submit(ControlEvent("/audio/level", 1.5))
    loop.tick()
    assert control_store.snapshot().truncation_psi == 1.5


def test_malformed_touch_event_is_ignored():
    clock = FakeClock()
    clock.now = 10.0
    loop, control_store, _, _ = make_loop(clock)
    bind(loop, "truncation_psi", "/audio/level")
    loop.submit(ControlEvent(TOUCH_BEGIN, 3.0, source="ui"))
    loop.submit(ControlEvent(TOUCH_BEGIN, "not_a_parameter", source="ui"))
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
    assert control_store.snapshot().truncation_psi == UNWRITTEN

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
        loop.submit(ControlEvent("/latent/x", 3.0, source="ui"))
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
    loop.submit(ControlEvent("/latent/x", 7.0, source="ui"))
    deadline = time.monotonic() + 2.0
    while control_store.snapshot().latent_x != 7.0 and time.monotonic() < deadline:
        time.sleep(0.005)
    loop.stop()
    assert control_store.snapshot().latent_x == 7.0


def test_vector_randomize_is_not_left_inert_by_the_mapping_layer():
    """The regression this task exists to fix.

    `mapping.apply_event` recognizes `/vector/randomize` but deliberately
    leaves state untouched, because it cannot reach `ModelInfo`. If the
    control loop's own handling of this address were ever removed, the event
    would fall back to that inert path and this would fail.
    """
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)
    loop, control_store, _, _ = make_loop_with_model(clock, info)
    loop.submit(ControlEvent(VECTOR_RANDOMIZE, 7, source="ui"))
    loop.tick()
    vec = control_store.snapshot().latent_vec
    assert vec == tuple(np.random.RandomState(7).randn(4).tolist())


def test_vector_randomize_from_osc_also_materializes():
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=3, num_ws=8)
    loop, control_store, _, _ = make_loop_with_model(clock, info)
    loop.submit(ControlEvent(VECTOR_RANDOMIZE, 42, source="osc"))
    loop.tick()
    vec = control_store.snapshot().latent_vec
    assert vec == tuple(np.random.RandomState(42).randn(3).tolist())


def test_vector_randomize_without_a_model_logs_and_changes_nothing(caplog):
    loop, control_store, _, _ = make_loop()
    with caplog.at_level(logging.INFO, logger=control_module.__name__):
        loop.submit(ControlEvent(VECTOR_RANDOMIZE, 7, source="ui"))
        loop.tick()
    assert control_store.snapshot().latent_vec == ()
    logged = [r for r in caplog.records if r.name == control_module.__name__]
    assert len(logged) == 1
    assert VECTOR_RANDOMIZE in logged[0].getMessage()


def test_vector_randomize_with_a_non_numeric_seed_is_ignored():
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=2, num_ws=8)
    loop, control_store, _, _ = make_loop_with_model(clock, info)
    loop.submit(ControlEvent(VECTOR_RANDOMIZE, "not a seed", source="ui"))
    loop.tick()
    assert control_store.snapshot().latent_vec == ()


def test_vector_randomize_with_a_non_finite_seed_is_ignored_cleanly(caplog):
    """A non finite seed must take the log-and-ignore path, not the guard.

    `int(round(seed_number))` raises on `nan` (ValueError) and `inf`
    (OverflowError). Without an explicit `math.isfinite` check, the event
    still leaves the vector unchanged, but only because the last resort guard
    in `tick()` catches the exception and drops it, logging an ERROR with a
    full traceback instead of this function's own clean warning. Asserting
    only "vector unchanged" would pass on that buggy path too, so this checks
    the log record as well: a guard hit is unmistakably an ERROR from
    `_report_guard`, while the intended path is a WARNING from
    `_randomize_vector`.
    """
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=2, num_ws=8)
    loop, control_store, _, _ = make_loop_with_model(clock, info)
    with caplog.at_level(logging.WARNING, logger=control_module.__name__):
        loop.submit(ControlEvent(VECTOR_RANDOMIZE, float("nan"), source="ui"))
        loop.submit(ControlEvent(VECTOR_RANDOMIZE, float("inf"), source="ui"))
        loop.tick()
    assert control_store.snapshot().latent_vec == ()
    logged = [r for r in caplog.records if r.name == control_module.__name__]
    assert len(logged) == 2
    assert all(record.levelno == logging.WARNING for record in logged)
    assert all(record.exc_info is None for record in logged)


def test_vector_randomize_updates_when_the_model_changes_between_ticks():
    """`model_info` is refreshed once per tick, not read at construction."""
    clock = FakeClock()
    small = ModelInfo(pkl_path="small.pkl", z_dim=2, num_ws=8)
    loop, control_store, _, _ = make_loop_with_model(clock, small)
    loop.submit(ControlEvent(VECTOR_RANDOMIZE, 1, source="ui"))
    loop.tick()
    assert len(control_store.snapshot().latent_vec) == 2

    loop._model_info_store.set(ModelInfo(pkl_path="big.pkl", z_dim=6, num_ws=8))
    loop.submit(ControlEvent(VECTOR_RANDOMIZE, 1, source="ui"))
    loop.tick()
    assert len(control_store.snapshot().latent_vec) == 6


def test_the_vector_walk_is_wired_into_the_control_loop():
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=2, num_ws=8)
    control_store = LatestValueStore(
        ControlState(vector_mode=True, anim_playing=True, anim_speed_x=1.0)
    )
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    model_info_store = LatestValueStore(info)
    loop = ControlLoop(
        control_store,
        render_store,
        source_store,
        clock=clock,
        model_info_store=model_info_store,
        walk_rng=np.random.RandomState(0),
    )
    loop.submit(ControlEvent(VECTOR_RANDOMIZE, 1, source="ui"))
    loop.tick()
    first = control_store.snapshot().latent_vec
    clock.now += 0.1
    loop.tick()
    second = control_store.snapshot().latent_vec
    assert second != first


# --- loop integration (Task 4) ----------------------------------------------


def test_loop_integration_advances_alpha_in_time_mode():
    clock = FakeClock()
    state = ControlState(loop_active=True, loop_uses_time=True, loop_time=4.0)
    loop, control_store, _, _ = make_loop_with_state(clock, state)
    loop.tick()  # first tick, dt == 0
    clock.now = 0.1
    loop.tick()
    result = control_store.snapshot()
    assert abs(result.loop_alpha - 6 * 0.1 / 4.0) < 1e-9
    assert result.loop_index == 0


def test_negative_loop_speed_wraps_the_index_backwards():
    clock = FakeClock()
    state = ControlState(
        loop_active=True, loop_uses_time=False, loop_speed=-1.0, loop_alpha=0.05
    )
    loop, control_store, _, _ = make_loop_with_state(clock, state)
    loop.tick()
    clock.now = 0.1
    loop.tick()
    assert control_store.snapshot().loop_index == 5


def test_loop_inactive_integrates_nothing():
    clock = FakeClock()
    state = ControlState(loop_active=False, loop_uses_time=False, loop_speed=5.0)
    loop, control_store, _, _ = make_loop_with_state(clock, state)
    loop.tick()
    clock.now = 1.0
    loop.tick()
    result = control_store.snapshot()
    assert result.loop_alpha == 0.0
    assert result.loop_index == 0


def test_a_held_loop_alpha_does_not_advance_and_resumes_after_the_grace():
    clock = FakeClock()
    clock.now = 10.0
    state = ControlState(loop_active=True, loop_uses_time=False, loop_speed=1.0)
    loop, control_store, _, _ = make_loop_with_state(clock, state)
    loop.submit(ControlEvent(TOUCH_BEGIN, "loop_alpha", source="ui"))
    loop.tick()
    clock.now += 0.5
    loop.tick()
    assert control_store.snapshot().loop_alpha == 0.0

    loop.submit(ControlEvent(TOUCH_END, "loop_alpha", source="ui"))
    loop.tick()
    clock.now += TOUCH_GRACE + 0.5
    loop.tick()
    assert control_store.snapshot().loop_alpha > 0.0


def test_manual_loop_alpha_write_wins_over_integration_this_tick():
    """Same rule as bindings vs UI, applied to the loop's own scrub address."""
    clock = FakeClock()
    state = ControlState(loop_active=True, loop_uses_time=False, loop_speed=5.0)
    loop, control_store, _, _ = make_loop_with_state(clock, state)
    loop.tick()
    clock.now = 1.0
    loop.submit(ControlEvent("/loop/alpha", 0.42, source="ui"))
    loop.tick()
    assert control_store.snapshot().loop_alpha == 0.42


def test_manual_loop_index_write_wins_over_integration_this_tick():
    clock = FakeClock()
    state = ControlState(
        loop_active=True, loop_uses_time=True, loop_time=4.0, loop_alpha=0.99
    )
    loop, control_store, _, _ = make_loop_with_state(clock, state)
    loop.tick()
    clock.now = 1.0
    loop.submit(ControlEvent("/loop/index", 4, source="ui"))
    loop.tick()
    assert control_store.snapshot().loop_index == 4


def test_perfect_loop_deactivates_on_wrap_and_only_then():
    clock = FakeClock()
    state = ControlState(
        loop_active=True, perfect_loop=True, loop_uses_time=True, loop_time=4.0
    )
    loop, control_store, _, _ = make_loop_with_state(clock, state)
    loop.tick()
    clock.now = 0.1
    loop.tick()
    assert control_store.snapshot().loop_active is True

    clock.now += 10.0
    loop.tick()
    assert control_store.snapshot().loop_active is False


def test_started_flag_fires_only_on_the_tick_playback_begins():
    clock = FakeClock()
    state = ControlState(loop_active=False, loop_uses_time=False, loop_speed=1.0)
    loop, _, _, _ = make_loop_with_state(clock, state)
    loop.tick()
    assert loop.last_loop_step.started is False

    loop.submit(ControlEvent("/loop/anim", True, source="ui"))
    loop.tick()
    assert loop.last_loop_step.started is True

    clock.now = 0.1
    loop.tick()
    assert loop.last_loop_step.started is False


def test_wrapped_flag_is_exposed_on_a_completed_cycle():
    clock = FakeClock()
    state = ControlState(loop_active=True, loop_uses_time=True, loop_time=4.0)
    loop, _, _, _ = make_loop_with_state(clock, state)
    loop.tick()
    assert loop.last_loop_step.wrapped is False

    clock.now = 4.0
    loop.tick()
    assert loop.last_loop_step.wrapped is True


# --- noise loop vector (Task 5) ----------------------------------------------


def test_noise_loop_vector_is_published_as_the_render_latent_vec():
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)
    state = ControlState(
        loop_active=True, noise_loop=True, noise_loop_seed=3, noise_radius=2.0
    )
    loop, control_store, _, _ = make_loop_with_state_and_model(clock, state, info)
    result = loop.tick()
    expected = NoiseLoop(3, 2.0, 4).vector(0.0)
    assert result.latent_vec == expected
    assert result.mode == "vec"
    # The user's own vector, untouched by the noise loop.
    assert control_store.snapshot().latent_vec == ()


def test_noise_loop_never_overwrites_the_users_own_latent_vec():
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)
    users_vector = (1.0, 2.0, 3.0, 4.0)
    state = ControlState(
        loop_active=True,
        noise_loop=True,
        noise_loop_seed=0,
        noise_radius=1.0,
        latent_vec=users_vector,
    )
    loop, control_store, render_store, _ = make_loop_with_state_and_model(
        clock, state, info
    )
    loop.tick()
    assert control_store.snapshot().latent_vec == users_vector
    assert render_store.snapshot().latent_vec != users_vector


def test_noise_loop_rebuilds_only_when_seed_radius_or_z_dim_changes(monkeypatch):
    builds = []

    class CountingNoiseLoop(NoiseLoop):
        def __init__(self, seed, radius, dim):
            builds.append((seed, radius, dim))
            super().__init__(seed, radius, dim)

    monkeypatch.setattr(control_module, "NoiseLoop", CountingNoiseLoop)
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)
    state = ControlState(
        loop_active=True, noise_loop=True, noise_loop_seed=1, noise_radius=1.0
    )
    loop, _, _, _ = make_loop_with_state_and_model(clock, state, info)
    for _ in range(5):
        clock.now += 0.01
        loop.tick()
    assert len(builds) == 1

    loop.submit(ControlEvent("/loop/seed", 2, source="ui"))
    clock.now += 0.01
    loop.tick()
    assert len(builds) == 2

    loop.submit(ControlEvent("/loop/radius", 5.0, source="ui"))
    clock.now += 0.01
    loop.tick()
    assert len(builds) == 3

    loop._model_info_store.set(ModelInfo(pkl_path="model.pkl", z_dim=6, num_ws=8))
    clock.now += 0.01
    loop.tick()
    assert len(builds) == 4


def test_noise_vector_is_resampled_only_when_alpha_moves(monkeypatch):
    calls = []

    class CountingNoiseLoop(NoiseLoop):
        def vector(self, alpha):
            calls.append(alpha)
            return super().vector(alpha)

    monkeypatch.setattr(control_module, "NoiseLoop", CountingNoiseLoop)
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)
    state = ControlState(
        loop_active=True, noise_loop=True, noise_loop_seed=0, noise_radius=1.0
    )
    loop, _, _, _ = make_loop_with_state_and_model(clock, state, info)
    loop.tick()  # dt == 0, alpha stays at 0.0
    loop.tick()  # dt == 0 again, alpha still 0.0: no resample
    assert len(calls) == 1

    clock.now += 1.0
    loop.tick()  # alpha moved: resamples once
    assert len(calls) == 2


def test_noise_loop_without_a_model_does_not_raise_and_keeps_latent_vec():
    clock = FakeClock()
    state = ControlState(loop_active=True, noise_loop=True)
    loop, control_store, render_store, _ = make_loop_with_state(clock, state)
    result = loop.tick()
    assert result.mode == "vec"
    assert result.latent_vec == ()
    assert control_store.snapshot().latent_vec == ()


def test_noise_loop_with_a_non_positive_radius_does_not_raise():
    clock = FakeClock()
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)
    state = ControlState(
        loop_active=True, noise_loop=True, noise_loop_seed=0, noise_radius=0.0
    )
    loop, _, render_store, _ = make_loop_with_state_and_model(clock, state, info)
    result = loop.tick()
    assert result.latent_vec == ()
