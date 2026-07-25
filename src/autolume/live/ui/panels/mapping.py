"""Mapping panel: which source drives which parameter, and through what.

One row per bindable parameter, so a performer reads the whole patch at a
glance instead of hunting through right click menus. The same row is what
`ControlBinder` shows in its right click popup, so both entry points edit a
mapping the same way and neither can drift from the other.

Every edit leaves as a control event. Nothing here writes state directly, and
a clear travels as a `ClearBinding` object rather than a target string, so no
OSC peer can express one.
"""

import time
from typing import Callable

from imgui_bundle import imgui

from autolume.live.core.events import ControlEvent
from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    REGISTRY,
    Binding,
    ClearBinding,
    ParamKind,
    ParamSpec,
)
from autolume.live.ui.controls import BINDING_ERROR_COLOR, binding_for

# Widths in multiples of the font size, so the rows keep their proportions on
# every display scale instead of cramping at 200 percent.
_SOURCE_EMS = 11.0
_EXPRESSION_EMS = 16.0
_PICKER_EMS = (16.0, 14.0)
_ENTER = imgui.InputTextFlags_.enter_returns_true

_NO_SOURCES = "No input detected"
_HINT = (
    "Every parameter can be mapped here. "
    "Pick lists the addresses arriving right now. "
    "Send OSC or turn audio on to fill it."
)


def bindable_specs() -> list[ParamSpec]:
    """The parameters a source can drive, in registry order.

    Text parameters are left out: an expression yields a number, so a model
    path has nothing a binding could give it.
    """
    return [spec for spec in REGISTRY.values() if spec.kind is not ParamKind.STR]


def display_label(name: str) -> str:
    """A readable name for a registry parameter, derived rather than restated."""
    return name.replace("_", " ").capitalize()


def canonical_address(text: str) -> str:
    """Normalize a typed address the way the source table stores one."""
    address = text.strip()
    if not address:
        return ""
    return address if address.startswith("/") else "/" + address


class MappingPanel:
    def __init__(self, runtime, clock: Callable[[], float] = time.monotonic) -> None:
        self._runtime = runtime
        self._clock = clock
        # Text held only while a field is being edited, so a preset load or an
        # incoming binding still shows through every field the user is not in.
        self._drafts: dict[tuple[str, str], str] = {}
        self._label_width = 0.0
        self._measured_at = 0.0

    def _label_column(self) -> float:
        """Where the fields start, past the longest parameter name.

        Measured from the font rather than fixed, because a name wider than the
        column would be overlapped by the field placed at it, and a display
        scale change makes any pixel constant wrong.
        """
        size = imgui.get_font_size()
        if size != self._measured_at:
            self._measured_at = size
            widest = max(
                imgui.calc_text_size(display_label(spec.name)).x
                for spec in bindable_specs()
            )
            self._label_width = widest + imgui.get_style().item_spacing.x * 2.0
        return self._label_width

    def gui(self) -> None:
        state = self._runtime.control_store.snapshot()
        imgui.text_wrapped(_HINT)
        imgui.separator()
        for spec in bindable_specs():
            self._row(spec.name, state)

    def popup(self, name: str) -> None:
        """The editor for one parameter, drawn inside an already open popup."""
        spec = REGISTRY.get(name)
        if spec is None or spec.kind is ParamKind.STR:
            imgui.text_disabled("This control cannot be mapped.")
            return
        self._row(name, self._runtime.control_store.snapshot())

    def _row(self, name: str, state) -> None:
        binding = binding_for(state.bindings, name)
        column = self._label_column()
        imgui.push_id(name)
        imgui.text(display_label(name))
        imgui.same_line(column)
        self._source_field(name, binding)
        imgui.same_line()
        self._picker(name, binding)
        imgui.same_line()
        self._enable_box(name, binding)
        imgui.same_line()
        self._clear_button(name, binding)
        imgui.indent(column)
        self._expression_field(name, binding)
        if binding is not None and binding.error:
            imgui.push_style_color(
                imgui.Col_.text, imgui.ImVec4(*BINDING_ERROR_COLOR)
            )
            imgui.text_wrapped(binding.error)
            imgui.pop_style_color()
        imgui.unindent(column)
        imgui.pop_id()

    def _source_field(self, name: str, binding: Binding | None) -> None:
        current = binding.source if binding is not None else ""
        imgui.set_next_item_width(imgui.get_font_size() * _SOURCE_EMS)
        entered, text = imgui.input_text(
            "##source", self._shown(name, "source", current), _ENTER
        )
        self._keep_draft(name, "source", text, current)
        if entered or imgui.is_item_deactivated_after_edit():
            self._commit(name, binding, source=text)

    def _expression_field(self, name: str, binding: Binding | None) -> None:
        current = binding.expression if binding is not None else "x"
        imgui.set_next_item_width(imgui.get_font_size() * _EXPRESSION_EMS)
        entered, text = imgui.input_text(
            "##expression", self._shown(name, "expression", current), _ENTER
        )
        self._keep_draft(name, "expression", text, current)
        if entered or imgui.is_item_deactivated_after_edit():
            self._commit(name, binding, expression=text)

    def _picker(self, name: str, binding: Binding | None) -> None:
        if imgui.button("Pick"):
            imgui.open_popup("sources")
        if not imgui.begin_popup("sources"):
            return
        addresses = self._runtime.source_store.snapshot().recent(self._clock())
        if not addresses:
            # An empty picker is a normal state at the start of a show, so it
            # says what is missing rather than showing nothing at all.
            imgui.text_disabled(_NO_SOURCES)
        else:
            em = imgui.get_font_size()
            imgui.begin_child(
                "##addresses", imgui.ImVec2(em * _PICKER_EMS[0], em * _PICKER_EMS[1])
            )
            for address in addresses:
                if imgui.selectable(address, False)[0]:
                    self._commit(name, binding, source=address)
                    imgui.close_current_popup()
            imgui.end_child()
        imgui.end_popup()

    def _enable_box(self, name: str, binding: Binding | None) -> None:
        if binding is None:
            imgui.begin_disabled()
        changed, enabled = imgui.checkbox(
            "On", binding.enabled if binding is not None else False
        )
        if binding is None:
            imgui.end_disabled()
        elif changed:
            self._commit(name, binding, enabled=enabled)

    def _clear_button(self, name: str, binding: Binding | None) -> None:
        if binding is None:
            imgui.begin_disabled()
        clicked = imgui.button("Clear")
        if binding is None:
            imgui.end_disabled()
        elif clicked:
            self._submit(BINDING_CLEAR, ClearBinding(name))
            self._forget(name)

    def _commit(
        self,
        name: str,
        binding: Binding | None,
        source: str | None = None,
        expression: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Send the edited binding, or clear it once its source is emptied."""
        if source is None:
            source = self._shown(name, "source", binding.source if binding else "")
        if expression is None:
            expression = self._shown(
                name, "expression", binding.expression if binding else "x"
            )
        if enabled is None:
            enabled = binding.enabled if binding is not None else True
        address = canonical_address(source)
        if not address:
            if binding is not None:
                self._submit(BINDING_CLEAR, ClearBinding(name))
                self._forget(name)
            return
        self._submit(
            BINDING_SET,
            Binding(name, address, expression.strip() or "x", bool(enabled)),
        )
        self._forget(name)

    def _submit(self, address: str, value: object) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _shown(self, name: str, field: str, current: str) -> str:
        return self._drafts.get((name, field), current)

    def _keep_draft(self, name: str, field: str, text: str, current: str) -> None:
        if text != self._shown(name, field, current):
            self._drafts[(name, field)] = text

    def _forget(self, name: str) -> None:
        self._drafts.pop((name, "source"), None)
        self._drafts.pop((name, "expression"), None)
