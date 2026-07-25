import logging

import pytest

from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    Binding,
    ClearBinding,
    ControlState,
)

MAPPING_LOGGER = "autolume.live.core.mapping"


def warnings_from(caplog, logger_name):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelname == "WARNING"
    ]


def test_float_event_applies():
    state = apply_event(ControlState(), ControlEvent("/latent/x", 4.2))
    assert state.latent_x == 4.2


def test_int_coercion_and_clamp():
    state = apply_event(ControlState(), ControlEvent("/render/fps", 999.9))
    assert state.fps_cap == 240
    state = apply_event(state, ControlEvent("/render/fps", -5))
    assert state.fps_cap == 0


def test_bool_coercion_from_osc_float():
    state = apply_event(ControlState(), ControlEvent("/anim/playing", 1.0))
    assert state.anim_playing is True
    state = apply_event(state, ControlEvent("/anim/playing", 0.0))
    assert state.anim_playing is False


def test_str_param_applies():
    state = apply_event(ControlState(), ControlEvent("/model/path", "/tmp/m.pkl"))
    assert state.pkl_path == "/tmp/m.pkl"


def test_float_clamped_to_bounds():
    state = apply_event(ControlState(), ControlEvent("/trunc/psi", 5.0))
    assert state.truncation_psi == 2.0


def test_unknown_address_ignored():
    before = ControlState()
    after = apply_event(before, ControlEvent("/nope", 1.0))
    assert after == before


def test_uncoercible_value_ignored():
    before = ControlState()
    after = apply_event(before, ControlEvent("/latent/x", "not a number"))
    assert after == before


def test_uncoercible_value_warning_names_the_wire_address(caplog):
    with caplog.at_level(logging.WARNING):
        apply_event(ControlState(), ControlEvent("/latent/x", "not a number"))
    messages = warnings_from(caplog, "autolume.live.core.params")
    assert messages
    assert any("/latent/x" in message for message in messages)


def set_binding(state, binding):
    return apply_event(state, ControlEvent(BINDING_SET, binding))


def clear_binding(state, target):
    return apply_event(state, ControlEvent(BINDING_CLEAR, ClearBinding(target)))


def test_binding_set_appends_then_replaces_in_place():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    state = set_binding(state, Binding("truncation_psi", "/audio/bass", "x*2"))
    assert [b.target for b in state.bindings] == ["latent_x", "truncation_psi"]

    state = set_binding(state, Binding("latent_x", "/ctl/1", "x+1"))
    assert [b.target for b in state.bindings] == ["latent_x", "truncation_psi"]
    assert state.bindings[0].source == "/ctl/1"
    assert state.bindings[0].expression == "x+1"


def test_binding_set_with_non_binding_value_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BINDING_SET, 1.0))
    assert after == before
    assert any("non binding value" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_binding_set_with_unknown_target_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding("nope", "/audio/level"))
    assert after == before
    assert any("unknown parameter" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("expression", [None, 3.0, b"x", ["x"]])
def test_binding_set_with_non_str_expression_ignored(caplog, expression):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding("latent_x", "/audio/level", expression))
    assert after == before
    assert any("malformed binding" in m for m in warnings_from(caplog, MAPPING_LOGGER))


@pytest.mark.parametrize("target", [["latent_x"], {"latent_x"}, None, 1.0])
def test_binding_set_with_non_str_target_ignored(caplog, target):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding(target, "/audio/level"))
    assert after == before
    assert any("malformed binding" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_binding_set_with_non_str_source_ignored(caplog):
    before = ControlState()
    with caplog.at_level(logging.WARNING):
        after = set_binding(before, Binding("latent_x", None))
    assert after == before
    assert any("malformed binding" in m for m in warnings_from(caplog, MAPPING_LOGGER))


def test_binding_set_with_bad_expression_stores_error():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level", "x +* 2"))
    binding = state.bindings[0]
    assert binding.expression == "x +* 2"
    assert binding.error is not None


def test_binding_set_with_valid_expression_clears_error():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level", "nope(x)"))
    assert state.bindings[0].error is not None
    state = set_binding(state, Binding("latent_x", "/audio/level", "x*2", error="old"))
    assert state.bindings[0].error is None


def test_binding_clear_removes_only_that_target():
    state = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    state = set_binding(state, Binding("truncation_psi", "/audio/bass"))
    state = clear_binding(state, "latent_x")
    assert [b.target for b in state.bindings] == ["truncation_psi"]


def test_binding_clear_unbound_target_is_noop():
    before = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    assert clear_binding(before, "truncation_psi") == before


@pytest.mark.parametrize("value", [1.0, "latent_x", None, ["latent_x"]])
def test_binding_clear_with_non_clear_binding_value_ignored(caplog, value):
    before = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BINDING_CLEAR, value))
    assert after == before
    assert any(
        "non clear binding value" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )


@pytest.mark.parametrize("target", [None, 1.0, ["latent_x"]])
def test_binding_clear_with_non_str_target_ignored(caplog, target):
    before = set_binding(ControlState(), Binding("latent_x", "/audio/level"))
    with caplog.at_level(logging.WARNING):
        after = apply_event(before, ControlEvent(BINDING_CLEAR, ClearBinding(target)))
    assert after == before
    assert any(
        "malformed clear binding" in m for m in warnings_from(caplog, MAPPING_LOGGER)
    )
