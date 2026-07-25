"""Unit tests for the binder's pure decisions.

The widgets themselves cannot be exercised: imgui-bundle's null backend
asserts at runtime, so everything below stops at the module boundary.
"""

import pytest

from autolume.live.core.params import REGISTRY, Binding, ParamKind
from autolume.live.core.touch import TOUCH_BEGIN, TOUCH_END
from autolume.live.ui.controls import (
    BINDING_COLOR,
    BINDING_ERROR_COLOR,
    BINDING_OFF_COLOR,
    binding_for,
    displayed_value,
    drag_bounds,
    indicator_color,
    require_spec,
    slider_bounds,
    widget_events,
)


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
    assert displayed_value({"latent_x": 0.9}, "latent_x", 0.5) == 0.9


def test_displayed_value_of_one_parameter_does_not_leak_to_another():
    assert displayed_value({"latent_x": 0.9}, "latent_y", 0.5) == 0.5


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
