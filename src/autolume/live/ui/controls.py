"""Binding aware imgui controls.

Every performable widget goes through `ControlBinder` so that all of them
behave the same way. Bounds, kind and transport address come from the
registry and the value from the store, so a panel names a parameter and
restates nothing about it, which is why no panel can name one control and
hand it another one's value. The
performer's hand wins over a binding while a widget is held. A parameter that
something else drives says so.
"""

import math
from dataclasses import dataclass
from typing import Callable, Mapping, TypeVar

from imgui_bundle import imgui

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
BINDING_ERROR_COLOR = (1.0, 0.3, 0.3, 1.0)
BINDING_OFF_COLOR = (0.5, 0.5, 0.5, 1.0)

_GUTTER_RATIO = 0.4
_MARKER_RATIO = 0.3
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
        return BINDING_ERROR_COLOR
    if not binding.enabled:
        return BINDING_OFF_COLOR
    return BINDING_COLOR


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
        stored = getattr(self._snapshot(), name)
        imgui.push_id(name)
        self._indicator(name)
        if not enabled:
            imgui.begin_disabled()
        changed, value = draw(displayed_value(self._local, name, stored))
        if not enabled:
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

    def _indicator(self, name: str) -> None:
        """Reserve a gutter left of the widget and mark it when bound.

        The gutter is reserved whether or not a binding exists, so a binding
        appearing does not shift the control under the performer's cursor. The
        marker is a drawn square because the bundled font has no symbol glyphs.
        """
        color = indicator_color(binding_for(self._snapshot().bindings, name))
        height = imgui.get_frame_height()
        width = round(height * _GUTTER_RATIO)
        origin = imgui.get_cursor_screen_pos()
        if color is not None:
            side = round(height * _MARKER_RATIO)
            left = origin.x + (width - side) * 0.5
            top = origin.y + (height - side) * 0.5
            imgui.get_window_draw_list().add_rect_filled(
                imgui.ImVec2(left, top),
                imgui.ImVec2(left + side, top + side),
                imgui.get_color_u32(imgui.ImVec4(*color)),
            )
        imgui.dummy(imgui.ImVec2(width, height))
        imgui.same_line()

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
