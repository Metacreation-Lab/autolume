"""Binding aware imgui controls.

Every performable widget goes through `ControlBinder` so that all of them
behave the same way. Bounds, kind and transport address come from the
registry and the value from the store, so a panel names a parameter and
restates nothing about it, which is why no panel can name one control and
hand it another one's value. The
performer's hand wins over a binding while a widget is held. A parameter that
something else drives says so.

Every parameter has exactly one driver, and the chip in the gutter left of the
widget names it. A binding writes an absolute value, so it takes the control
away from the hand and the widget is drawn read only. Motion is relative and
carries on from wherever the value is, so an animated control stays live and
dragging it is scrubbing. The chip is clickable in all three states and opens
the mapping editor, which is what a read only control has instead of its own
right click menu.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, TypeVar

from imgui_bundle import imgui

from autolume.live.core import motion
from autolume.live.core.events import ControlEvent
from autolume.live.core.params import (
    REGISTRY,
    Binding,
    ControlState,
    ParamKind,
    ParamSpec,
)
from autolume.live.core.touch import TOUCH_BEGIN, TOUCH_END

BINDING_COLOR = (0.35, 0.75, 1.0, 1.0)
BINDING_OFF_COLOR = (0.5, 0.5, 0.5, 1.0)
MOTION_COLOR = (0.55, 0.8, 0.55, 0.9)
# Every failure the UI shows is this colour, wherever it is shown, so that one
# definition here is what makes a broken binding, a missing device and a preset
# that would not save look alike.
ERROR_COLOR = (1.0, 0.3, 0.3, 1.0)

# Wide enough to click at, since the chip in it is the way back from a control
# a binding has taken over.
_GUTTER_RATIO = 0.7
_MARKER_RATIO = 0.3
_PLATE_INSET = 0.15
# A widget hands its value back through a C float, so it cannot express a
# difference finer than this. Anything closer is the same value.
_FLOAT_TOLERANCE = 1e-6

T = TypeVar("T")
Color = tuple[float, float, float, float]
Value = float | int | bool


def require_spec(name: str, kind: ParamKind) -> ParamSpec:
    """Look up the spec for `name` and refuse it if it is not a `kind`.

    Raising here rather than drawing something plausible is deliberate. A
    checkbox on a float parameter would emit values the control thread then
    coerces, and the mismatch would only ever surface as a puzzling value.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"{name} is not a registry parameter")
    if spec.kind is not kind:
        raise TypeError(f"{name} is a {spec.kind.value} parameter, not {kind.value}")
    return spec


def slider_bounds(spec: ParamSpec) -> tuple[float, float]:
    """The range a slider spans, refusing a parameter that declares none."""
    if spec.minimum is None or spec.maximum is None:
        raise ValueError(f"{spec.name} has no bounds to slide between")
    return spec.minimum, spec.maximum


def drag_bounds(spec: ParamSpec) -> tuple[float, float]:
    """The range a drag clamps to. Equal values mean imgui leaves it free."""
    if spec.minimum is None or spec.maximum is None:
        return 0.0, 0.0
    return spec.minimum, spec.maximum


def binding_for(bindings: tuple[Binding, ...], name: str) -> Binding | None:
    for binding in bindings:
        if binding.target == name:
            return binding
    return None


def indicator_color(binding: Binding | None) -> Color | None:
    """The colour of the binding marker, or None when nothing drives this."""
    if binding is None:
        return None
    if binding.error is not None:
        return ERROR_COLOR
    if not binding.enabled:
        return BINDING_OFF_COLOR
    return BINDING_COLOR


class Marker(Enum):
    """What the gutter left of a control shows."""

    NONE = "none"
    MOTION = "motion"
    BINDING = "binding"


@dataclass(frozen=True)
class Gutter:
    """Everything one control needs to know about who is driving it."""

    marker: Marker
    color: Color | None
    read_only: bool
    tooltip: str


_UNBOUND_TIP = "Nothing is driving this. Click to map it to a source."
_MOTION_TIP = (
    "Animation is driving this. Drag it and the animation carries on from there."
)


def _binding_tip(binding: Binding) -> str:
    if binding.error is not None:
        return "This binding is failing. Click to fix it."
    if not binding.enabled:
        return f"Bound to {binding.source}. The binding is off. Click to edit it."
    return f"Bound to {binding.source}. Click to edit it."


def gutter_for(state: ControlState, name: str) -> Gutter:
    """Who drives `name`, in the terms the control is drawn in.

    A binding writes an absolute value, so the next message from its source
    erases anything the hand did in between and a control it drives is read
    only rather than futile. Motion is relative and advances from wherever the
    value is, so dragging an animated parameter is scrubbing and stays live.

    Precedence is the control loop's own: an enabled binding beats motion, and
    a binding switched off drives nothing, so motion takes the marker back and
    the parked binding is only shown once nothing is driving the parameter.
    Whether motion has the parameter is `motion.drives`' answer rather than
    ours, so this cannot disagree with the integrator.

    No touch tracker is passed, which is the one skip of the integrator's this
    does not follow: a hand on the widget pauses motion for the drag and its
    grace, but it does not take the parameter away from motion, and a marker
    blinking off for the length of every drag would be noise.
    """
    binding = binding_for(state.bindings, name)
    if binding is not None and binding.enabled:
        return Gutter(
            Marker.BINDING, indicator_color(binding), True, _binding_tip(binding)
        )
    if motion.drives(state, name):
        return Gutter(Marker.MOTION, MOTION_COLOR, False, _MOTION_TIP)
    if binding is not None:
        return Gutter(
            Marker.BINDING, indicator_color(binding), False, _binding_tip(binding)
        )
    return Gutter(Marker.NONE, None, False, _UNBOUND_TIP)


@dataclass(frozen=True)
class Override:
    """A value the performer set, held until the store catches up with it.

    `stored` is what the store held at the moment the value was set, which is
    how a store that moves on to something else can be told from a store that
    has simply not applied the value yet.
    """

    value: Value
    stored: Value


def values_agree(one: Value, other: Value) -> bool:
    """Whether two values of a parameter are the same value.

    Floats are compared at the precision the widget itself works in: imgui
    rounds through a C float, so the value that comes back from a slider and
    the double the store keeps can differ in the last bits while being the
    very same value the performer asked for. Whole numbers are compared
    exactly, because a seed two thousand away is a different seed however
    large it is, and a relative tolerance would swallow that.
    """
    if isinstance(one, float) or isinstance(other, float):
        return math.isclose(
            float(one), float(other), rel_tol=_FLOAT_TOLERANCE, abs_tol=1e-9
        )
    return one == other


def displayed_value(overrides: Mapping[str, Override], name: str, snapshot: T) -> T:
    """The value to draw: the local one while the performer holds the widget.

    imgui re-asserts the drag value every frame while the mouse is down, and a
    binding writing the same parameter overwrites it in between, so drawing the
    snapshot makes a bound control jitter under the hand. The touch grace stops
    the binding, this stops the round trip through the control thread.
    """
    override = overrides.get(name)
    if override is None:
        return snapshot
    return override.value  # type: ignore[return-value]


def next_override(
    override: Override | None,
    snapshot: Value,
    value: Value,
    *,
    changed: bool,
    active: bool,
) -> Override | None:
    """The local hold after one frame of a widget, or None once it is done.

    A hold is not released on deactivation. The value submitted on that frame
    only reaches the store on the next control tick, so dropping the hold there
    draws the previous state for a frame, which a checkbox shows as a visible
    flick back. It ends when the store agrees instead.

    A store that moves to a third value once the widget is released is a
    binding that won the parameter back. The hold ends there too, because
    showing what drives the parameter beats a number frozen at what the hand
    last asked for. While the widget is still held that same disagreement is
    the round trip the hold exists to hide, so it is ignored.

    Between them the two endings also bound the lifetime of a hold left behind
    by a widget that stopped being drawn mid drag, a panel closed or a section
    collapsed. Nothing reads it while it is not drawn, and the first frame the
    widget comes back settles it either way.
    """
    if changed:
        return Override(value, snapshot)
    if override is None:
        return None
    if values_agree(override.value, snapshot):
        return None
    if not active and not values_agree(override.stored, snapshot):
        return None
    return override


def widget_events(
    spec: ParamSpec,
    value: object,
    *,
    activated: bool,
    changed: bool,
    deactivated: bool,
) -> tuple[ControlEvent, ...]:
    """The events one frame of interaction with a widget produces.

    Order carries meaning because the control loop drains the queue in order.
    The touch begins before the first value it protects and ends after the
    last, and a widget that changes and releases in the same frame, which is
    every checkbox, still brackets its value correctly.
    """
    events = []
    if activated:
        events.append(ControlEvent(TOUCH_BEGIN, spec.name, source="ui"))
    if changed:
        events.append(ControlEvent(spec.address, value, source="ui"))
    if deactivated:
        events.append(ControlEvent(TOUCH_END, spec.name, source="ui"))
    return tuple(events)


class ControlBinder:
    """Draws registry parameters as widgets wired to the control thread.

    `mapping_popup` is called inside the right click popup of every control,
    with the parameter name. It is injected so this module does not depend on
    the mapping panel.
    """

    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._mapping_popup = mapping_popup
        self._local: dict[str, Override] = {}
        self._frame = -1
        self._state: ControlState | None = None

    def state(self) -> ControlState:
        """The control state this frame, for a panel that needs more than values.

        The same snapshot every widget is drawn from, so a panel reading it
        cannot disagree with the controls it draws beside.
        """
        return self._snapshot()

    def value(self, name: str) -> Value:
        """What the named control is showing right now.

        A panel greying a row out of a checkbox must follow this rather than
        the store, or the row lags the box it belongs to by a frame.
        """
        return displayed_value(self._local, name, getattr(self._snapshot(), name))

    def slider_float(self, name: str, label: str, *, enabled: bool = True) -> None:
        spec = require_spec(name, ParamKind.FLOAT)
        minimum, maximum = slider_bounds(spec)
        self._widget(
            spec,
            lambda shown: imgui.slider_float(label, float(shown), minimum, maximum),
            enabled,
        )

    def drag_float(
        self, name: str, label: str, speed: float = 0.01, *, enabled: bool = True
    ) -> None:
        spec = require_spec(name, ParamKind.FLOAT)
        minimum, maximum = drag_bounds(spec)
        self._widget(
            spec,
            lambda shown: imgui.drag_float(
                label, float(shown), speed, minimum, maximum
            ),
            enabled,
        )

    def slider_int(self, name: str, label: str, *, enabled: bool = True) -> None:
        spec = require_spec(name, ParamKind.INT)
        minimum, maximum = slider_bounds(spec)
        self._widget(
            spec,
            lambda shown: imgui.slider_int(
                label, int(shown), int(minimum), int(maximum)
            ),
            enabled,
        )

    def drag_int(
        self, name: str, label: str, speed: float = 1.0, *, enabled: bool = True
    ) -> None:
        spec = require_spec(name, ParamKind.INT)
        minimum, maximum = drag_bounds(spec)
        self._widget(
            spec,
            lambda shown: imgui.drag_int(
                label, int(shown), speed, int(minimum), int(maximum)
            ),
            enabled,
        )

    def checkbox(self, name: str, label: str, *, enabled: bool = True) -> None:
        spec = require_spec(name, ParamKind.BOOL)
        self._widget(spec, lambda shown: imgui.checkbox(label, bool(shown)), enabled)

    def _widget(
        self,
        spec: ParamSpec,
        draw: Callable[[Value], tuple[bool, Value]],
        enabled: bool,
    ) -> None:
        name = spec.name
        state = self._snapshot()
        stored = getattr(state, name)
        gutter = gutter_for(state, name)
        # A control a binding drives is drawn read only. The next value from
        # the source erases a drag, so the hand cannot hold it anyway, and a
        # widget that visibly does nothing is worse than one that says so.
        live = enabled and not gutter.read_only
        imgui.push_id(name)
        self._indicator(name, gutter)
        if not live:
            imgui.begin_disabled()
        changed, value = draw(displayed_value(self._local, name, stored))
        if not live:
            imgui.end_disabled()
        activated = imgui.is_item_activated()
        deactivated = imgui.is_item_deactivated()
        events = widget_events(
            spec,
            value,
            activated=activated,
            changed=changed,
            deactivated=deactivated,
        )
        override = next_override(
            self._local.get(name),
            stored,
            value,
            changed=changed,
            active=imgui.is_item_active(),
        )
        if override is None:
            self._local.pop(name, None)
        else:
            self._local[name] = override
        for event in events:
            self._runtime.submit(event)
        self._mapping_menu(name)
        imgui.pop_id()

    def _indicator(self, name: str, gutter: Gutter) -> None:
        """Draw the chip left of the widget: who drives it, and a way to change it.

        The gutter is reserved whether or not anything drives the parameter, so
        a binding appearing does not shift the control under the performer's
        cursor. The markers are drawn shapes because the bundled font has no
        symbol glyphs, a square for a binding and a dot for motion, so the two
        differ in shape as well as colour.

        The chip is a real clickable item and it is drawn before any disabled
        block, which is what makes disabling a bound control safe: imgui
        suppresses hover inside a disabled block, so a read only control cannot
        open its own right click menu, and without this the Mapping panel would
        be the only way to take a parameter back mid show.
        """
        height = imgui.get_frame_height()
        width = round(height * _GUTTER_RATIO)
        origin = imgui.get_cursor_screen_pos()
        clickable = self._mapping_popup is not None
        if clickable:
            if imgui.invisible_button("##chip", imgui.ImVec2(width, height)):
                imgui.open_popup("chip")
            hovered = imgui.is_item_hovered()
            # The plate answers the cursor at once, the words wait for the
            # cursor to settle, so crossing the column does not fire a row of
            # tooltips at a performer reaching for a slider.
            explain = imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal)
        else:
            imgui.dummy(imgui.ImVec2(width, height))
            hovered = explain = False
        self._chip_shape(origin, width, height, gutter, hovered)
        if explain:
            imgui.set_tooltip(gutter.tooltip)
        if clickable and imgui.begin_popup("chip"):
            self._mapping_popup(name)
            imgui.end_popup()
        imgui.same_line()

    def _chip_shape(
        self, origin, width: int, height: float, gutter: Gutter, hovered: bool
    ) -> None:
        """Paint the chip: a hover plate under the marker for the current driver.

        The plate is what tells a performer the chip can be clicked, including
        on the parameters nothing drives, where there is no marker to hover.
        """
        draw_list = imgui.get_window_draw_list()
        middle = imgui.ImVec2(origin.x + width * 0.5, origin.y + height * 0.5)
        if hovered:
            inset = height * _PLATE_INSET
            draw_list.add_rect_filled(
                imgui.ImVec2(origin.x, origin.y + inset),
                imgui.ImVec2(origin.x + width, origin.y + height - inset),
                imgui.get_color_u32(imgui.Col_.button_hovered),
                imgui.get_style().frame_rounding,
            )
        if gutter.color is None:
            return
        color = imgui.get_color_u32(imgui.ImVec4(*gutter.color))
        side = round(height * _MARKER_RATIO)
        if gutter.marker is Marker.MOTION:
            draw_list.add_circle_filled(middle, side * 0.5, color)
        else:
            draw_list.add_rect_filled(
                imgui.ImVec2(middle.x - side * 0.5, middle.y - side * 0.5),
                imgui.ImVec2(middle.x + side * 0.5, middle.y + side * 0.5),
                color,
            )

    def _mapping_menu(self, name: str) -> None:
        if self._mapping_popup is None:
            return
        if imgui.begin_popup_context_item():
            self._mapping_popup(name)
            imgui.end_popup()

    def _snapshot(self) -> ControlState:
        """The control state this frame, read once however many widgets ask.

        Read here rather than passed in so a widget cannot be drawn from one
        snapshot and marked from another, and so a panel cannot forget to hand
        it over and silently lose every binding marker.
        """
        frame = imgui.get_frame_count()
        if self._state is None or frame != self._frame:
            self._frame = frame
            self._state = self._runtime.control_store.snapshot()
        return self._state
