"""Mapping panel: which source drives which parameter, and through what.

One row per bindable parameter, so a performer reads the whole patch at a
glance instead of hunting through right click menus. The same row is what
`ControlBinder` shows in its right click popup, so both entry points edit a
mapping the same way and neither can drift from the other.

The row governs every remote writer of its parameter, which is the whole point
of there being one row per parameter rather than one per configured mapping.
Nothing outside reaches a parameter until its row is switched on, so an
untouched row is an inert one and the switch beside it is the one honest answer
to what may write this parameter from outside. It never governs the hand.

An empty source means the parameter's own address, so a row that says only On
opens the address the registry gives the parameter without anyone typing it.

Every parameter has a row, including the model path. A value means something
different there, a position in the models folder or a name, so that row says
what it means underneath itself rather than leaving the performer to find out
by sending something.

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
    binding_for,
)
from autolume.live.core.sources import canonical_address
from autolume.live.ui.theme import ERROR_COLOR

# Widths in multiples of the font size, so the rows keep their proportions on
# every display scale instead of cramping at 200 percent.
_SOURCE_EMS = 11.0
_EXPRESSION_EMS = 16.0
_PICKER_EMS = (16.0, 14.0)
_ENTER = imgui.InputTextFlags_.enter_returns_true

_NO_SOURCES = "No input detected"
_HINT = (
    "Nothing outside can move a parameter until you tick On for it. "
    "A row with no source then listens on the parameter's own address, shown "
    "in grey. "
    "Pick lists the addresses arriving right now, which is how you find out "
    "what your controller is sending."
)
_REFERENCE_NOTE = (
    "A number picks a model by its position in your models folder, counting "
    "from 0, and the expression above scales it. Text picks a model by name "
    "or by part of a name, and the expression does not apply to it."
)


def bindable_specs() -> list[ParamSpec]:
    """The parameters a source can drive, in registry order.

    Every one of them. A text parameter used to be left out, on the grounds
    that an expression yields a number and a model path has nothing to do with
    one, which made switching models the single thing a controller could not
    do. What it needed was not an exception to the rule but a row like any
    other: what a value means on it differs, and that is the row's business to
    explain, not the registry's to withhold.
    """
    return list(REGISTRY.values())


def reference_note(spec: ParamSpec) -> str | None:
    """The line under `spec`'s row explaining what a value means on it, or None.

    Only a text parameter needs one, and it needs one because its row is the
    only place where the expression field applies to some values and not
    others. A number is an index into the models folder and the expression
    scales it, which is what makes a fader a model selector. Text names a model
    and no expression touches it, since an expression yields a number.

    Leaving that unsaid would make the field the misleading kind: editable,
    and silently doing nothing to half of what arrives.
    """
    return _REFERENCE_NOTE if spec.kind is ParamKind.STR else None


def display_label(name: str) -> str:
    """A readable name for a registry parameter, derived rather than restated."""
    return name.replace("_", " ").capitalize()


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
        if REGISTRY.get(name) is None:
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
        note = reference_note(REGISTRY[name])
        if note is not None:
            self._note(note)
        if binding is not None and binding.error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ERROR_COLOR))
            imgui.text_wrapped(binding.error)
            imgui.pop_style_color()
        imgui.unindent(column)
        imgui.pop_id()

    def _source_field(self, name: str, binding: Binding | None) -> None:
        current = binding.source if binding is not None else ""
        imgui.set_next_item_width(imgui.get_font_size() * _SOURCE_EMS)
        # An empty field is not "nothing", it is the address this row will
        # listen on once it is switched on, so the field says which one rather
        # than leaving the performer to work out what an empty row means.
        entered, text = imgui.input_text_with_hint(
            "##source",
            REGISTRY[name].address,
            self._shown(name, "source", current),
            _ENTER,
        )
        self._keep_draft(name, "source", text)
        if entered or imgui.is_item_deactivated_after_edit():
            self._commit(name, binding, source=text)

    def _expression_field(self, name: str, binding: Binding | None) -> None:
        current = binding.expression if binding is not None else "x"
        imgui.set_next_item_width(imgui.get_font_size() * _EXPRESSION_EMS)
        entered, text = imgui.input_text(
            "##expression", self._shown(name, "expression", current), _ENTER
        )
        self._keep_draft(name, "expression", text)
        if entered or imgui.is_item_deactivated_after_edit():
            self._commit(name, binding, expression=text)

    def _note(self, text: str) -> None:
        """Draw a greyed line under a row, in the colour the rest of the UI
        already means "present but not something you act on" with.

        Written on the row rather than hidden in a tooltip: the thing it
        explains is a field the performer is about to type into, and a tooltip
        is read after the mistake rather than before it.
        """
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(text)
        imgui.pop_style_color()

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
        """The row's switch, which governs every remote writer of the parameter.

        Always live, and off by default: a parameter nobody has switched on is
        deaf to the network, so this box is the whole of the opt in and has to
        be reachable before a source is typed. Ticking it with the source left
        empty is the shortest way to open a parameter, on its own address.
        """
        changed, enabled = imgui.checkbox(
            "On", binding.enabled if binding is not None else False
        )
        if changed:
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
        """Send the edited row, or clear it when it says nothing anymore.

        A row with no source, switched off, passing the value through is
        exactly what a parameter with no row does, so it is cleared rather than
        stored. That keeps the default state out of presets and makes switching
        a row on and off again land back where it started. Any other row is
        kept, including one with nothing but its switch ticked, which is how
        "remote input on this parameter" is recorded and persisted.
        """
        if source is None:
            source = self._shown(name, "source", binding.source if binding else "")
        if expression is None:
            expression = self._shown(
                name, "expression", binding.expression if binding else "x"
            )
        address = canonical_address(source)
        expression = expression.strip() or "x"
        if enabled is None:
            # Naming a source is the ask, so the first one switches the row on
            # rather than leaving a mapping the performer picked doing nothing
            # until they find the box beside it. Off by default protects them
            # from traffic they did not configure, and an address they just
            # typed is not that. An existing row keeps whatever it was set to.
            enabled = binding.enabled if binding is not None else bool(address)
        enabled = bool(enabled)
        if not address and not enabled and expression == "x":
            if binding is not None:
                self._submit(BINDING_CLEAR, ClearBinding(name))
                self._forget(name)
            return
        self._submit(BINDING_SET, Binding(name, address, expression, enabled))
        self._forget(name)

    def _submit(self, address: str, value: object) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _shown(self, name: str, field: str, current: str) -> str:
        return self._drafts.get((name, field), current)

    def _keep_draft(self, name: str, field: str, text: str) -> None:
        """Hold the typed buffer only while the field is the active item.

        A field that stops being drawn never reports deactivation, so an edit
        abandoned by switching tabs would otherwise keep its draft forever and
        hide every later change to the binding. Activity is the one condition
        that resolves even then.
        """
        key = (name, field)
        if imgui.is_item_active():
            self._drafts[key] = text
        else:
            self._drafts.pop(key, None)

    def _forget(self, name: str) -> None:
        self._drafts.pop((name, "source"), None)
        self._drafts.pop((name, "expression"), None)
