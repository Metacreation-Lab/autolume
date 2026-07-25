"""Binding aware imgui controls.

Every performable widget goes through `ControlBinder` so that all of them
behave the same way. Bounds, kind and transport address come from the
registry, so a panel names a parameter and never restates its range. The
performer's hand wins over a binding while a widget is held. A parameter that
something else drives says so.
"""

from typing import Callable, Mapping, TypeVar

from imgui_bundle import imgui

from autolume.live.core.events import ControlEvent
from autolume.live.core.params import REGISTRY, Binding, ParamKind, ParamSpec
from autolume.live.core.touch import TOUCH_BEGIN, TOUCH_END

BINDING_COLOR = (0.35, 0.75, 1.0, 1.0)
BINDING_ERROR_COLOR = (1.0, 0.3, 0.3, 1.0)
BINDING_OFF_COLOR = (0.5, 0.5, 0.5, 1.0)

_GUTTER_RATIO = 0.4
_MARKER_RATIO = 0.3

T = TypeVar("T")
Color = tuple[float, float, float, float]


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


def displayed_value(overrides: Mapping[str, T], name: str, snapshot: T) -> T:
    """The value to draw: the local one while the performer holds the widget.

    imgui re-asserts the drag value every frame while the mouse is down, and a
    binding writing the same parameter overwrites it in between, so drawing the
    snapshot makes a bound control jitter under the hand. The touch grace stops
    the binding, this stops the round trip through the control thread.
    """
    return overrides.get(name, snapshot)


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
        self._local: dict[str, float | int | bool] = {}
        self._frame = -1
        self._state = runtime.control_store.snapshot()

    def slider_float(
        self, name: str, label: str, value: float, *, enabled: bool = True
    ) -> None:
        spec = require_spec(name, ParamKind.FLOAT)
        minimum, maximum = slider_bounds(spec)
        shown = float(displayed_value(self._local, name, value))
        self._widget(
            spec, lambda: imgui.slider_float(label, shown, minimum, maximum), enabled
        )

    def drag_float(
        self,
        name: str,
        label: str,
        value: float,
        speed: float = 0.01,
        *,
        enabled: bool = True,
    ) -> None:
        spec = require_spec(name, ParamKind.FLOAT)
        minimum, maximum = drag_bounds(spec)
        shown = float(displayed_value(self._local, name, value))
        self._widget(
            spec,
            lambda: imgui.drag_float(label, shown, speed, minimum, maximum),
            enabled,
        )

    def slider_int(
        self, name: str, label: str, value: int, *, enabled: bool = True
    ) -> None:
        spec = require_spec(name, ParamKind.INT)
        minimum, maximum = slider_bounds(spec)
        shown = int(displayed_value(self._local, name, value))
        self._widget(
            spec,
            lambda: imgui.slider_int(label, shown, int(minimum), int(maximum)),
            enabled,
        )

    def drag_int(
        self,
        name: str,
        label: str,
        value: int,
        speed: float = 1.0,
        *,
        enabled: bool = True,
    ) -> None:
        spec = require_spec(name, ParamKind.INT)
        minimum, maximum = drag_bounds(spec)
        shown = int(displayed_value(self._local, name, value))
        self._widget(
            spec,
            lambda: imgui.drag_int(label, shown, speed, int(minimum), int(maximum)),
            enabled,
        )

    def checkbox(
        self, name: str, label: str, value: bool, *, enabled: bool = True
    ) -> None:
        spec = require_spec(name, ParamKind.BOOL)
        shown = bool(displayed_value(self._local, name, value))
        self._widget(spec, lambda: imgui.checkbox(label, shown), enabled)

    def _widget(
        self,
        spec: ParamSpec,
        draw: Callable[[], tuple[bool, float | int | bool]],
        enabled: bool,
    ) -> None:
        name = spec.name
        imgui.push_id(name)
        self._indicator(name)
        if not enabled:
            imgui.begin_disabled()
        changed, value = draw()
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
        if changed:
            self._local[name] = value
        if deactivated:
            self._local.pop(name, None)
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

    def _snapshot(self):
        """The control state this frame, read once however many widgets ask.

        Read here rather than passed in so a panel cannot forget to hand it
        over and silently lose every binding marker.
        """
        frame = imgui.get_frame_count()
        if frame != self._frame:
            self._frame = frame
            self._state = self._runtime.control_store.snapshot()
        return self._state
