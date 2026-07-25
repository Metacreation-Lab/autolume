"""Unit tests for the binder's pure decisions.

The widgets themselves cannot be exercised: imgui-bundle's null backend
asserts at runtime, so everything below stops at the module boundary.
"""

import struct

import pytest

from autolume.live.core.params import REGISTRY, Binding, ParamKind
from autolume.live.core.touch import TOUCH_BEGIN, TOUCH_END
from autolume.live.ui.controls import (
    BINDING_COLOR,
    BINDING_ERROR_COLOR,
    BINDING_OFF_COLOR,
    Override,
    binding_for,
    displayed_value,
    drag_bounds,
    indicator_color,
    next_override,
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


def test_indicator_is_absent_without_a_binding():
    assert indicator_color(None) is None


def test_indicator_marks_an_active_binding():
    assert indicator_color(Binding(target="latent_x", source="/a")) == BINDING_COLOR


def test_indicator_marks_a_disabled_binding_apart():
    binding = Binding(target="latent_x", source="/a", enabled=False)
    assert indicator_color(binding) == BINDING_OFF_COLOR


def test_indicator_marks_an_error_even_when_the_binding_is_disabled():
    binding = Binding(target="latent_x", source="/a", enabled=False, error="boom")
    assert indicator_color(binding) == BINDING_ERROR_COLOR


def test_displayed_value_is_the_snapshot_when_nothing_is_held():
    assert displayed_value({}, "latent_x", 0.5) == 0.5


def test_displayed_value_is_the_local_one_while_the_widget_is_held():
    held = {"latent_x": Override(0.9, 0.5)}
    assert displayed_value(held, "latent_x", 0.5) == 0.9


def test_displayed_value_of_one_parameter_does_not_leak_to_another():
    held = {"latent_x": Override(0.9, 0.5)}
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


def test_a_change_holds_the_value_against_the_stored_one():
    assert next_override(None, 0.5, 0.9, changed=True, active=True) == Override(
        0.9, 0.5
    )


def test_a_change_that_releases_in_the_same_frame_still_holds():
    # Every checkbox click. The value only reaches the store on the next
    # control tick, so dropping the hold here draws the old state for a frame.
    assert next_override(None, False, True, changed=True, active=False) == Override(
        True, False
    )


def test_the_hold_is_dropped_once_the_store_agrees():
    held = Override(0.9, 0.5)
    assert next_override(held, 0.9, 0.9, changed=False, active=True) is None


def test_the_hold_outlasts_a_binding_writing_underneath_a_held_widget():
    held = Override(0.9, 0.5)
    assert next_override(held, 0.2, 0.9, changed=False, active=True) == held


def test_the_hold_outlasts_a_release_while_the_value_is_still_in_flight():
    held = Override(True, False)
    assert next_override(held, False, True, changed=False, active=False) == held


def test_the_hold_is_dropped_when_the_store_moves_on_after_release():
    # Nothing else clears it: the widget may never be drawn again, and a
    # binding driving this parameter would otherwise show a frozen number.
    held = Override(0.9, 0.5)
    assert next_override(held, 0.2, 0.9, changed=False, active=False) is None


def test_nothing_held_stays_nothing():
    assert next_override(None, 0.5, 0.5, changed=False, active=False) is None


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
