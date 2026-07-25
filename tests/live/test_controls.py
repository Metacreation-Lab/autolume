"""Unit tests for the binder's pure decisions.

The widgets themselves cannot be exercised: imgui-bundle's null backend
asserts at runtime, so everything below stops at the module boundary.
"""

import dataclasses
import struct

import pytest

from autolume.live.core.control import ControlLoop
from autolume.live.core.events import ControlEvent
from autolume.live.core.params import (
    REGISTRY,
    Binding,
    ControlState,
    ParamKind,
    to_render_params,
)
from autolume.live.core.presets import PRESET_APPLY, to_payload
from autolume.live.core.sources import LIVE_WINDOW, SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.core.touch import TOUCH_BEGIN, TOUCH_END
from autolume.live.ui.controls import (
    _HOLD_FRAMES,
    _UNBOUND_TIP,
    BINDING_COLOR,
    ERROR_COLOR,
    MOTION_COLOR,
    Marker,
    Override,
    binding_for,
    displayed_value,
    drag_bounds,
    gutter_for,
    next_override,
    remote_writer,
    require_spec,
    slider_bounds,
    values_agree,
    widget_events,
)


def _as_float32(value: float) -> float:
    """The value as imgui would hand it back, rounded through a C float."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _events(**flags):
    return widget_events(
        REGISTRY["truncation_psi"],
        0.5,
        activated=flags.get("activated", False),
        changed=flags.get("changed", False),
        deactivated=flags.get("deactivated", False),
    )


def test_require_spec_returns_the_registry_entry():
    spec = require_spec("truncation_psi", ParamKind.FLOAT)
    assert spec is REGISTRY["truncation_psi"]


def test_require_spec_rejects_an_unknown_parameter():
    with pytest.raises(KeyError):
        require_spec("not_a_parameter", ParamKind.FLOAT)


def test_require_spec_rejects_a_kind_mismatch():
    with pytest.raises(TypeError):
        require_spec("noise_enabled", ParamKind.FLOAT)


def test_slider_bounds_come_from_the_spec():
    assert slider_bounds(REGISTRY["truncation_psi"]) == (-1.0, 2.0)


def test_slider_bounds_reject_an_unbounded_parameter():
    with pytest.raises(ValueError):
        slider_bounds(REGISTRY["latent_x"])


def test_drag_bounds_leave_an_unbounded_parameter_free():
    assert drag_bounds(REGISTRY["latent_x"]) == (0.0, 0.0)


def test_drag_bounds_use_the_spec_when_it_has_them():
    assert drag_bounds(REGISTRY["noise_seed"]) == (0, 2**31 - 1)


def test_binding_for_finds_the_binding_on_that_target():
    first = Binding(target="latent_x", source="/audio/level")
    second = Binding(target="latent_y", source="/audio/bass")
    assert binding_for((first, second), "latent_y") is second


def test_binding_for_returns_none_when_nothing_drives_the_parameter():
    assert binding_for((Binding(target="latent_x", source="/a"),), "latent_y") is None


NOW = 100.0


def sending(address, when=NOW):
    """A source table holding one address, seen at `when`."""
    return SourceTable().observe(address, 0.5, when)


def test_an_undriven_parameter_is_unmarked_and_stays_playable():
    gutter = gutter_for(ControlState(), "latent_x")
    assert gutter.marker is Marker.NONE
    assert gutter.color is None
    assert not gutter.read_only


def test_an_enabled_binding_makes_the_control_read_only():
    state = ControlState(bindings=(Binding("latent_x", "/audio/level"),))
    gutter = gutter_for(state, "latent_x")
    assert gutter.marker is Marker.BINDING
    assert gutter.color == BINDING_COLOR
    # The next value from the source erases a drag, so a live widget here would
    # be a control that visibly does nothing.
    assert gutter.read_only


def test_a_failing_binding_still_holds_the_control_it_claimed():
    # The integrator skips an enabled binding whether or not it evaluates, so a
    # playable widget here would fight a parameter nothing else can move.
    binding = Binding("latent_x", "/audio/level", "nope(", error="bad expression")
    gutter = gutter_for(ControlState(bindings=(binding,)), "latent_x")
    assert gutter.marker is Marker.BINDING
    assert gutter.color == ERROR_COLOR
    assert gutter.read_only


def test_a_row_switched_off_carries_no_marker_at_all():
    """It drives nothing, so it names nothing. This used to draw a grey square.

    That was one appearance the performer never acted on: whether a parked row
    exists is what the mapping panel is for, and a marker for it competed with
    the ones that mean something is happening right now.
    """
    binding = Binding("latent_x", "/audio/level", enabled=False)
    gutter = gutter_for(ControlState(bindings=(binding,)), "latent_x")
    assert gutter.marker is Marker.NONE
    assert gutter.color is None
    assert not gutter.read_only
    assert "/audio/level" in gutter.tooltip


def test_an_animated_parameter_is_marked_and_stays_playable():
    # Motion is relative, so dragging an animated parameter is scrubbing: it
    # carries on from wherever the hand left it.
    gutter = gutter_for(ControlState(anim_playing=True), "latent_x")
    assert gutter.marker is Marker.MOTION
    assert gutter.color == MOTION_COLOR
    assert not gutter.read_only


def test_animation_marks_only_what_motion_actually_writes():
    playing = ControlState(anim_playing=True)
    assert gutter_for(playing, "truncation_psi").marker is Marker.NONE
    assert gutter_for(playing, "anim_speed_x").marker is Marker.NONE


def test_a_binding_beats_motion_on_the_same_parameter():
    state = ControlState(
        anim_playing=True, bindings=(Binding("latent_x", "/audio/level"),)
    )
    gutter = gutter_for(state, "latent_x")
    assert gutter.marker is Marker.BINDING
    assert gutter.read_only
    assert gutter_for(state, "latent_y").marker is Marker.MOTION


def test_motion_takes_the_marker_back_from_a_binding_switched_off():
    # A binding that is off drives nothing, and motion does, so showing the
    # parked binding here would name the wrong driver.
    binding = Binding("latent_x", "/audio/level", enabled=False)
    state = ControlState(anim_playing=True, bindings=(binding,))
    assert gutter_for(state, "latent_x").marker is Marker.MOTION


LISTENING = ControlState(bindings=(Binding("latent_x", ""),))


def test_remote_input_that_is_on_and_receiving_is_filled():
    gutter = gutter_for(LISTENING, "latent_x", sending("/latent/x"), NOW)
    assert gutter.marker is Marker.BINDING
    assert gutter.color == BINDING_COLOR
    assert gutter.filled
    assert "/latent/x" in gutter.tooltip


def test_remote_input_that_is_on_with_nothing_arriving_is_an_outline():
    """Fill is liveness, and this is the state it was worth adding for.

    A row on /audio/bass drawn as an outline says the audio module is off or
    the room is silent, which otherwise takes another panel to work out.
    """
    state = ControlState(bindings=(Binding("latent_x", "/audio/bass"),))
    gutter = gutter_for(state, "latent_x", SourceTable(), NOW)
    assert gutter.marker is Marker.BINDING
    assert gutter.color == BINDING_COLOR
    assert not gutter.filled
    assert "Nothing is arriving" in gutter.tooltip


def test_an_implicit_and_an_explicit_source_look_exactly_alike():
    """The distinction the performer does not act on, and so is not drawn.

    Which address a row listens on is what the mapping row is for. Across
    twelve rows, "something remote drives this" is one fact and gets one
    appearance.
    """
    implicit = gutter_for(LISTENING, "latent_x", sending("/latent/x"), NOW)
    explicit_state = ControlState(bindings=(Binding("latent_x", "/td/knob"),))
    explicit = gutter_for(explicit_state, "latent_x", sending("/td/knob"), NOW)
    assert implicit.marker is explicit.marker
    assert implicit.color == explicit.color
    assert implicit.filled and explicit.filled


def test_the_fill_goes_out_once_the_sender_stops():
    stale = sending("/latent/x", NOW - LIVE_WINDOW - 0.1)
    assert not gutter_for(LISTENING, "latent_x", stale, NOW).filled
    fresh = sending("/latent/x", NOW - LIVE_WINDOW)
    assert gutter_for(LISTENING, "latent_x", fresh, NOW).filled


def test_a_parameter_something_remote_is_writing_stays_playable():
    # The whole point of showing it: a misbehaving controller is exactly when
    # the performer has to be able to grab the parameter back, and read only
    # here would leave them watching it move.
    gutter = gutter_for(LISTENING, "latent_x", sending("/latent/x"), NOW)
    assert not gutter.read_only


def test_a_receiving_row_is_named_ahead_of_the_animation():
    # It is genuinely writing the parameter, and it is the one of the two that
    # is invisible everywhere else in the app.
    state = ControlState(anim_playing=True, bindings=(Binding("latent_x", ""),))
    table = sending("/latent/x")
    gutter = gutter_for(state, "latent_x", table, NOW)
    assert gutter.marker is Marker.BINDING
    assert gutter.filled
    assert gutter_for(state, "latent_y", table, NOW).marker is Marker.MOTION


def test_an_idle_row_yields_the_marker_to_the_animation():
    # Between messages the integrator is what is advancing the value, so
    # naming the row would name a writer that is not writing.
    state = ControlState(anim_playing=True, bindings=(Binding("latent_x", ""),))
    assert gutter_for(state, "latent_x", SourceTable(), NOW).marker is Marker.MOTION


def test_a_caller_with_no_source_table_shows_nothing_live():
    assert not gutter_for(LISTENING, "latent_x").filled


def test_traffic_on_one_address_marks_only_the_parameter_it_names():
    both = ControlState(bindings=(Binding("latent_x", ""), Binding("latent_y", "")))
    table = sending("/latent/x")
    assert gutter_for(both, "latent_x", table, NOW).filled
    assert not gutter_for(both, "latent_y", table, NOW).filled


def test_traffic_at_a_parameter_nobody_switched_on_is_not_shown_at_all():
    """The Perform gutter answers one question: what drives this right now.

    Remote input is off until a row says otherwise, so most traffic most of
    the time is reaching a parameter that refuses it, and an ambient signal
    for that would be noise on a panel meant for playing. Discovery is a
    deliberate act in the Mapping panel, which is why it matters that the
    control loop records a blocked write as a source.
    """
    gutter = gutter_for(ControlState(), "latent_x", sending("/latent/x"), NOW)
    assert gutter.marker is Marker.NONE
    assert gutter.color is None
    assert gutter.tooltip == _UNBOUND_TIP


def test_remote_writer_ignores_a_parameter_with_no_row():
    # Nothing remote can write it, so nothing may be drawn as writing it.
    assert remote_writer(ControlState(), "latent_x", sending("/latent/x"), NOW) is None


def test_a_row_with_no_source_and_a_broken_expression_shows_the_failure():
    # The row is on, so nothing is knocking. What it needs to say is why the
    # value arriving is not landing.
    binding = Binding("latent_x", "", "nope(", error="bad expression")
    state = ControlState(bindings=(binding,))
    gutter = gutter_for(state, "latent_x", sending("/latent/x"), NOW)
    assert gutter.marker is Marker.BINDING
    assert gutter.color == ERROR_COLOR
    assert not gutter.read_only


def test_a_row_with_no_source_never_takes_the_control_away_from_the_hand():
    # It writes only when a message happens to arrive, exactly like the
    # unmapped default, so it has no claim on the widget between messages.
    state = ControlState(bindings=(Binding("latent_x", "", "x*2"),))
    assert not gutter_for(state, "latent_x").read_only


def test_a_row_with_no_source_leaves_the_axis_to_the_animation():
    state = ControlState(anim_playing=True, bindings=(Binding("latent_x", "", "x*2"),))
    assert gutter_for(state, "latent_x").marker is Marker.MOTION


def test_a_row_with_no_source_names_the_address_it_governs():
    on = Binding("latent_x", "")
    off = dataclasses.replace(on, enabled=False)
    assert "/latent/x" in gutter_for(ControlState(bindings=(on,)), "latent_x").tooltip
    assert "/latent/x" in gutter_for(ControlState(bindings=(off,)), "latent_x").tooltip


def test_every_tooltip_says_something_the_bundled_font_can_draw():
    # The bundled font has no symbol glyphs, so a non ascii character renders
    # as a question mark.
    binding = Binding("latent_x", "/audio/level")
    sourceless = Binding("latent_x", "")
    states = (
        ControlState(),
        ControlState(anim_playing=True),
        ControlState(bindings=(binding,)),
        ControlState(bindings=(dataclasses.replace(binding, enabled=False),)),
        ControlState(bindings=(dataclasses.replace(binding, error="boom"),)),
        ControlState(bindings=(sourceless,)),
        ControlState(bindings=(dataclasses.replace(sourceless, enabled=False),)),
    )
    gutters = [gutter_for(state, "latent_x") for state in states]
    # And the one the source table is needed to reach at all.
    listening = ControlState(bindings=(sourceless,))
    gutters.append(gutter_for(listening, "latent_x", sending("/latent/x"), NOW))
    assert {gutter.marker for gutter in gutters} == set(Marker)
    assert {gutter.filled for gutter in gutters} == {True, False}
    for gutter in gutters:
        tooltip = gutter.tooltip
        assert tooltip.isascii()
        assert ";" not in tooltip
        assert " - " not in tooltip
        assert tooltip.endswith(".")


def test_a_bound_tooltip_names_the_source_it_is_bound_to():
    state = ControlState(bindings=(Binding("latent_x", "/audio/level"),))
    assert "/audio/level" in gutter_for(state, "latent_x").tooltip


def test_displayed_value_is_the_snapshot_when_nothing_is_held():
    assert displayed_value({}, "latent_x", 0.5) == 0.5


def test_displayed_value_is_the_local_one_while_the_widget_is_held():
    held = {"latent_x": Override(0.9, 0.5, 1)}
    assert displayed_value(held, "latent_x", 0.5) == 0.9


def test_displayed_value_of_one_parameter_does_not_leak_to_another():
    held = {"latent_x": Override(0.9, 0.5, 1)}
    assert displayed_value(held, "latent_y", 0.5) == 0.5


def test_values_agree_at_the_precision_the_widget_works_in():
    # imgui rounds through a C float, so the value coming back from a slider
    # can differ from the stored double in the last bits and still be the very
    # same value the performer asked for.
    assert values_agree(0.1, _as_float32(0.1))
    assert not values_agree(0.1, 0.2)


def test_values_agree_exactly_on_whole_numbers():
    # A seed two thousand apart is a different seed, however large it is.
    assert not values_agree(2**31 - 1, 2**31 - 2001)
    assert values_agree(7, 7)
    assert values_agree(True, True)
    assert not values_agree(True, False)


def _hold(override, snapshot, value, **flags):
    return next_override(
        override,
        snapshot,
        value,
        changed=flags.get("changed", False),
        active=flags.get("active", False),
        live=flags.get("live", True),
        frame=flags.get("frame", 1),
    )


def test_a_change_holds_the_value_against_the_stored_one():
    assert _hold(None, 0.5, 0.9, changed=True, active=True, frame=7) == Override(
        0.9, 0.5, 7
    )


def test_a_change_that_releases_in_the_same_frame_still_holds():
    # Every checkbox click. The value only reaches the store on the next
    # control tick, so dropping the hold here draws the old state for a frame.
    assert _hold(None, False, True, changed=True) == Override(True, False, 1)


def test_the_hold_is_dropped_once_the_store_agrees():
    held = Override(0.9, 0.5, 1)
    assert _hold(held, 0.9, 0.9, active=True) is None


def test_the_hold_outlasts_a_binding_writing_underneath_a_held_widget():
    held = Override(0.9, 0.5, 1)
    assert _hold(held, 0.2, 0.9, active=True, frame=1 + _HOLD_FRAMES * 10) == held


def test_the_hold_outlasts_a_release_while_the_value_is_still_in_flight():
    held = Override(True, False, 1)
    assert _hold(held, False, True) == held


def test_the_hold_is_dropped_when_the_store_moves_on_after_release():
    # Nothing else clears it: the widget may never be drawn again, and a
    # binding driving this parameter would otherwise show a frozen number.
    held = Override(0.9, 0.5, 1)
    assert _hold(held, 0.2, 0.9) is None


def test_nothing_held_stays_nothing():
    assert _hold(None, 0.5, 0.5) is None


def test_a_control_the_hand_cannot_act_on_holds_nothing():
    # A binding taking the parameter over is the way this happens: the widget
    # goes read only between one frame and the next, and from then on it can
    # neither report a change nor ever clear what it was holding.
    held = Override(False, True, 1)
    assert _hold(held, True, False, live=False, frame=2) is None


def test_a_control_the_hand_cannot_act_on_cannot_open_a_hold():
    assert _hold(None, True, False, changed=True, live=False) is None


def test_the_hold_lapses_rather_than_wait_for_a_value_that_already_went_by():
    # A bool the store wrote and overwrote inside one control tick: no frame
    # ever sees the value the hold waits for, and there is no third value it
    # could move on to either.
    held = Override(False, True, 1)
    assert _hold(held, True, False, frame=_HOLD_FRAMES) == held
    assert _hold(held, True, False, frame=1 + _HOLD_FRAMES) is None


def test_an_untouched_widget_emits_nothing():
    assert _events() == ()


def test_activation_begins_a_touch_on_the_parameter_name():
    (event,) = _events(activated=True)
    assert (event.address, event.value, event.source) == (
        TOUCH_BEGIN,
        "truncation_psi",
        "ui",
    )


def test_deactivation_ends_the_touch():
    (event,) = _events(deactivated=True)
    assert (event.address, event.value, event.source) == (
        TOUCH_END,
        "truncation_psi",
        "ui",
    )


def test_a_change_goes_to_the_address_from_the_registry():
    (event,) = _events(changed=True)
    spec = REGISTRY["truncation_psi"]
    assert (event.address, event.value, event.source) == (spec.address, 0.5, "ui")


def test_a_touch_brackets_the_value_it_protects():
    events = _events(activated=True, changed=True, deactivated=True)
    assert [event.address for event in events] == [
        TOUCH_BEGIN,
        REGISTRY["truncation_psi"].address,
        TOUCH_END,
    ]


def test_touch_events_are_sourced_from_the_ui():
    # The control loop honors touch only from the ui, so this is load bearing.
    events = _events(activated=True, changed=True, deactivated=True)
    assert {event.source for event in events} == {"ui"}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Session:
    """A binder driving a real control loop, one frame at a time.

    imgui-bundle's null backend asserts, so the widget loop is restated here
    rather than driven: the gutter decides whether the control is live, a
    widget inside `begin_disabled` reports neither a change nor any activity,
    and the events and the hold both come off the same `changed` flag. Only the
    imgui calls are missing, and what runs underneath is the real control
    thread, so a value takes the same round trip it takes in the app.
    """

    def __init__(self, state: ControlState) -> None:
        self.clock = _Clock()
        self.store = LatestValueStore(state)
        self.loop = ControlLoop(
            self.store,
            LatestValueStore(to_render_params(state)),
            LatestValueStore(SourceTable()),
            clock=self.clock,
        )
        self.local: dict[str, Override] = {}
        self.frame = 0

    def tick(self, count: int = 1) -> None:
        for _ in range(count):
            self.clock.now += 0.008
            self.loop.tick()

    def submit(self, address: str, value: object, source: str = "ui") -> None:
        self.loop.submit(ControlEvent(address, value, source=source))

    def draw(self, name: str, *, clicked: bool = False, enabled: bool = True):
        """Draw one widget for one frame and return what it shows."""
        self.frame += 1
        state = self.store.snapshot()
        gutter = gutter_for(state, name)
        live = enabled and not gutter.read_only
        stored = getattr(state, name)
        shown = displayed_value(self.local, name, stored)
        # imgui swallows a click inside a disabled block entirely.
        changed = clicked and live
        value = (not shown) if changed else shown
        override = next_override(
            self.local.get(name),
            stored,
            value,
            changed=changed,
            active=False,
            live=live,
            frame=self.frame,
        )
        if override is None:
            self.local.pop(name, None)
        else:
            self.local[name] = override
        for event in widget_events(
            REGISTRY[name],
            value,
            activated=changed,
            changed=changed,
            deactivated=changed,
        ):
            self.loop.submit(event)
        return shown, gutter


def test_a_binding_taking_a_parameter_over_cannot_freeze_the_control_that_lost_it():
    """The reported bug: Animate drawn unchecked while the animation runs.

    The performer clicks Animate off while the control thread is between ticks,
    then recalls a preset that turns animation back on and binds the parameter.
    Both drain in the same tick, so no frame ever sees the store hold the value
    the hold is waiting for, and the checkbox is now read only and can never
    report a change again. Waiting on a value the store has already moved past
    is what makes the hold outlive its purpose, and a bool has no third value to
    fall back on, so nothing else can end it either.
    """
    session = _Session(ControlState(anim_playing=True, anim_speed_x=-0.03))
    shown, gutter = session.draw("anim_playing", clicked=True)
    assert shown is True and not gutter.read_only

    payload = to_payload(
        ControlState(
            anim_playing=True,
            anim_speed_x=-0.03,
            bindings=(Binding("anim_playing", "/audio/level"),),
        )
    )
    session.submit(PRESET_APPLY, payload)
    session.tick()
    assert session.store.snapshot().anim_playing is True

    # Every frame after that the performer clicks the box and nothing moves.
    for _ in range(200):
        session.submit("/audio/level", 0.31, source="audio")
        session.tick(2)
        shown, gutter = session.draw("anim_playing", clicked=True)

    state = session.store.snapshot()
    # The animation is running and the gutter says so, so the box beside it
    # cannot be drawn unchecked. That disagreement is the whole bug.
    assert gutter_for(state, "latent_x").marker is Marker.MOTION
    assert state.latent_x != 0.0
    assert shown is True
    assert not session.local
