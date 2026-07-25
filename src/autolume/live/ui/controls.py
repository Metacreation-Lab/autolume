"""Binding aware imgui controls.

Every performable widget goes through `ControlBinder` so that all of them
behave the same way. Bounds, kind and transport address come from the
registry and the value from the store, so a panel names a parameter and
restates nothing about it, which is why no panel can name one control and
hand it another one's value. The
performer's hand wins over a binding while a widget is held. A parameter that
something else drives says so.

The chip in the gutter left of the widget names what is driving the parameter.
A binding writes an absolute value, so it takes the control away from the hand
and the widget is drawn read only. Motion is relative and carries on from
wherever the value is, so an animated control stays live and dragging it is
scrubbing. A control a remote writer is playing stays live too, because the
moment a controller misbehaves is the moment the performer must be able to take
it back. The chip is clickable in every state and opens the mapping editor,
which is what a read only control has instead of its own right click menu.

The vocabulary is one rectangle, four colours and two fills.

    grey            nothing drives it
    green           motion drives it
    blue, filled    remote input on and receiving
    blue, outline   remote input on, nothing arriving right now
    red             the expression is failing

One rectangle in every row, whatever drives the parameter, which is what tells
the performer the column is there to click without making them find it with the
mouse first. Earlier passes drew an empty container instead, in the frame
colour, and that could not work in this theme by construction: darcula_darker
puts `frame_bg` at (0.145, 0.122, 0.122) against a `window_bg` of (0.138,
0.142, 0.149), so an empty widget is the background. The idle rectangle is the
theme's disabled text colour instead, which is the one grey a theme guarantees
reads as inactive and still legible, in a light theme as well as a dark one.

With one shape, colour carries who, so the four differ in brightness as well as
hue rather than in hue alone. Grey is the dimmest, because it is the state that
should be quiet on eleven rows at once, and green the brightest, because grey
against green is the pair that collapses for a red green colour deficiency and
a difference in value is what survives it. Fill is the second channel and it
carries the state that changes during a show: whether anything is arriving.

Blue means remote input is on, whether the row carries a source of its own or
falls back to the parameter's address. That difference changes what the mapping
row says and whether the control is read only, and both are already visible
where they matter, so drawing it again here would spend the performer's glance
on a distinction they never act on. Across twelve rows, "something remote
drives this" is one fact.

Fill is liveness, on the same window the marker uses, which turns the marker
into a diagnostic: a blue outline on a parameter bound to /audio/bass says the
audio module is off or the room is silent, which otherwise takes another panel
to work out.

Traffic arriving at a parameter nobody switched on is not shown here at all.
Remote input is off until a row says otherwise, so that is most traffic most of
the time, and this gutter answers one question only: what is driving this
parameter right now. Discovering that a controller is reaching Autolume is a
deliberate act in the Mapping panel, where the address picker lists everything
that has arrived. That is what makes it load bearing that the control loop
records a blocked write as a source, since the picker is the only place it
surfaces.
"""

import dataclasses
import math
import time
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
    binding_for,
    listens_on,
)
from autolume.live.core.sources import SourceTable
from autolume.live.core.touch import TOUCH_BEGIN, TOUCH_END

# Remote input, on. Filled when something is arriving, outlined when it is not.
BINDING_COLOR = (0.35, 0.75, 1.0, 1.0)
# Brighter and more saturated than the pale green this used to be. Every marker
# is the same rectangle now, so motion and the idle grey are told apart by
# colour alone, and a desaturated green sits at almost exactly a mid grey's
# brightness, which is the one pair a red green deficiency turns into one
# colour. Raising its value puts a step between them that hue loss cannot take.
MOTION_COLOR = (0.45, 0.92, 0.45, 1.0)
# Every failure the UI shows is this colour, wherever it is shown, so that one
# definition here is what makes a broken binding, a missing device and a preset
# that would not save look alike.
ERROR_COLOR = (1.0, 0.3, 0.3, 1.0)

# Wide enough to click at, since the chip in it is the way back from a control
# a binding has taken over.
_GUTTER_RATIO = 0.7
# Was 0.3 while every marker was solid. An outline needs a hole to be read as
# one: at 0.3 the square is about six pixels at 100 percent display scale and a
# stroke thick enough to survive rounding leaves nothing inside it, so the
# outline and the fill would differ by a pixel of ink and the liveness channel
# would be lost. At 0.4 the square is about eight pixels with a five pixel
# hole, still well inside the fifteen pixel slot around it.
_MARKER_RATIO = 0.4
_PLATE_INSET = 0.15
# Stroke weight of an outline, as a fraction of the frame height.
_OUTLINE_RATIO = 0.07
# What the idle grey keeps of the theme's disabled text colour. Under 1.0 so
# the quietest state is also the dimmest, which is what puts a brightness step
# between it and the green beside it, and still near four to one against the
# panel so it is plainly visible without a hover.
_IDLE_MARKER_ALPHA = 0.7
# What an outlined marker keeps of its colour. The fill difference already
# carries most of the signal, and this makes it degrade to dim against bright
# rather than to nothing when a display scale rounds the hole away.
_IDLE_ALPHA = 0.8
# A widget hands its value back through a C float, so it cannot express a
# difference finer than this. Anything closer is the same value.
_FLOAT_TOLERANCE = 1e-6
# How long a local hold may wait for the store, in frames. The round trip it
# hides is one frame plus one control tick, so this is generous by two orders
# of magnitude and still short enough that a hold the store never answers is
# gone before the next thing the performer reaches for.
_HOLD_FRAMES = 30
# Enter commits a text field, the same way it does in the mapping panel.
_ENTER = imgui.InputTextFlags_.enter_returns_true
# The narrowest a text field is ever drawn, in multiples of the font size. It
# exists only so the width can never reach zero, where imgui reads an item
# width as a distance from the right edge instead and the field would grow as
# the panel shrinks. Everything on the row keeps its size while the field gives
# up its own, so the chip and the button beside it are the last things to go.
_FIELD_MIN_EMS = 1.0

T = TypeVar("T")
Color = tuple[float, float, float, float]
Value = float | int | bool | str | None


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


class Marker(Enum):
    """Who drives the parameter the gutter sits beside.

    Not a shape any more: every state is the same rectangle, and colour is what
    separates them. This still names the driver rather than the colour, so the
    decision and the palette can be argued about one at a time.
    """

    NONE = "none"
    MOTION = "motion"
    BINDING = "binding"


@dataclass(frozen=True)
class Gutter:
    """Everything one control needs to know about who is driving it.

    `filled` is the marker's liveness. It is a field rather than two more
    markers because it is one fact about one driver, and because outline
    against fill is the only thing fill ever means in this gutter.
    """

    marker: Marker
    color: Color | None
    read_only: bool
    tooltip: str
    filled: bool = True


_UNBOUND_TIP = "Nothing is driving this. Click to map it to a source."
_ERROR_TIP = "This binding is failing. Click to fix it."
_MOTION_TIP = (
    "Animation is driving this. Drag it and the animation carries on from there."
)


def idle_color() -> Color:
    """The grey a parameter nothing drives is drawn in.

    Read from the style rather than stated as a constant so it keeps its
    polarity in a light theme and stays the grey the rest of the interface
    already means "present but inactive" with. A function so the drawing and
    anything checking it cannot pick different colours.
    """
    disabled = imgui.get_style_color_vec4(imgui.Col_.text_disabled)
    return (
        disabled.x,
        disabled.y,
        disabled.z,
        disabled.w * _IDLE_MARKER_ALPHA,
    )


def _binding_tip(binding: Binding, live: bool) -> str:
    """What an enabled row says, told apart by whether anything is arriving.

    The idle wording is the whole point of drawing liveness at all: a row on
    /audio/bass with nothing coming in means the audio module is off or the
    room is silent, and reading that off the chip beats going to look.
    """
    address = listens_on(binding)
    if live:
        return f"{address} is driving this right now. Click to edit it."
    return (
        f"Remote input is on for {address}. "
        "Nothing is arriving on it. "
        "Click to edit it."
    )


def _off_tip(binding: Binding) -> str:
    return f"Remote input on {listens_on(binding)} is off. Click to turn it on."


def _live_address(
    state: ControlState,
    name: str,
    sources: SourceTable | None,
    now: float,
) -> str | None:
    """The address `name`'s row listens on, if something is sending on it now.

    The address a row would take, whether or not the row lets it through: a
    parameter with no row at all still has the one the registry gives it, which
    is what a controller aimed at Autolume will be sending on.
    """
    if sources is None:
        return None
    spec = REGISTRY.get(name)
    if spec is None:
        return None
    binding = binding_for(state.bindings, name)
    address = listens_on(binding) if binding is not None else spec.address
    if not address or not sources.active(address, now):
        return None
    return address


def remote_writer(
    state: ControlState,
    name: str,
    sources: SourceTable | None,
    now: float,
) -> str | None:
    """The address something remote is writing `name` on right now, or None.

    An enabled row only, on whichever address it listens on. Whether that is a
    source the performer picked or the parameter's own is not a difference they
    act on from the gutter, so it is not one the gutter draws.
    """
    binding = binding_for(state.bindings, name)
    if binding is None or not binding.enabled or binding.error is not None:
        return None
    return _live_address(state, name, sources, now)


def gutter_for(
    state: ControlState,
    name: str,
    sources: SourceTable | None = None,
    now: float = 0.0,
) -> Gutter:
    """Who drives `name` right now, in the terms the control is drawn in.

    A binding writes an absolute value, so the next message from its source
    erases anything the hand did in between and a control it drives is read
    only rather than futile. Motion is relative and advances from wherever the
    value is, so dragging an animated parameter is scrubbing and stays live.

    A row with no source of its own stays playable too, and that is deliberate
    rather than an oversight: a misbehaving controller is exactly the moment
    the performer has to be able to grab the parameter back, and the control
    loop's touch grace is what makes that grab stick.

    Precedence over the collapsed vocabulary. An enabled row comes first when
    something is arriving on it, because then it is genuinely writing the
    parameter and motion is not: a stream at 60 Hz and an integrator adding to
    the same value cannot both be named, and the one the performer cannot see
    anywhere else wins. An enabled row with nothing arriving yields to motion,
    which is the honest answer between messages, and is why the blue outline
    only shows on a parameter motion is leaving alone. A row that is off drives
    nothing and gets no marker at all, only its own tooltip, because whether a
    parked row exists is what the Mapping panel is for.

    `motion.drives` is called rather than restated, so this cannot disagree
    with the integrator, and it already stands down for an enabled row with a
    source of its own.

    `sources` may be omitted by a caller that has no source table, and then
    nothing is ever drawn as live.

    No touch tracker is passed, which is the one skip of the integrator's this
    does not follow: a hand on the widget pauses motion for the drag and its
    grace, but it does not take the parameter away from motion, and a marker
    blinking off for the length of every drag would be noise.
    """
    binding = binding_for(state.bindings, name)
    if binding is not None and binding.enabled:
        read_only = bool(binding.source)
        if binding.error is not None:
            return Gutter(Marker.BINDING, ERROR_COLOR, read_only, _ERROR_TIP)
        live = remote_writer(state, name, sources, now) is not None
        if live or not motion.drives(state, name):
            return Gutter(
                Marker.BINDING,
                BINDING_COLOR,
                read_only,
                _binding_tip(binding, live),
                filled=live,
            )
    if motion.drives(state, name):
        return Gutter(Marker.MOTION, MOTION_COLOR, False, _MOTION_TIP)
    if binding is not None:
        return Gutter(Marker.NONE, None, False, _off_tip(binding))
    return Gutter(Marker.NONE, None, False, _UNBOUND_TIP)


@dataclass(frozen=True)
class Override:
    """A value the performer set, held until the store catches up with it.

    `stored` is what the store held at the moment the value was set, which is
    how a store that moves on to something else can be told from a store that
    has simply not applied the value yet. `frame` is when it was set, which is
    what bounds how long it may wait.
    """

    value: Value
    stored: Value
    frame: int


def values_agree(one: Value, other: Value) -> bool:
    """Whether two values of a parameter are the same value.

    Floats are compared at the precision the widget itself works in: imgui
    rounds through a C float, so the value that comes back from a slider and
    the double the store keeps can differ in the last bits while being the
    very same value the performer asked for. Whole numbers are compared
    exactly, because a seed two thousand away is a different seed however
    large it is, and a relative tolerance would swallow that.

    Text is compared exactly too, and the tolerance is reached only when both
    sides are numbers. A path one character different is a different model, and
    a text parameter is also the one that can hold nothing at all, so the
    numeric branch has to be unreachable from a value that is not a number
    rather than merely unlikely to be handed one.
    """
    numeric = isinstance(one, (int, float)) and isinstance(other, (int, float))
    if numeric and (isinstance(one, float) or isinstance(other, float)):
        return math.isclose(
            float(one), float(other), rel_tol=_FLOAT_TOLERANCE, abs_tol=1e-9
        )
    return one == other


def text_value(value: object) -> str:
    """A text parameter as something a field can hold.

    A parameter holding nothing is an empty field rather than the word None,
    which is what makes the model row read as empty and open rather than as
    broken. The field's hint is what fills that emptiness in.
    """
    return "" if value is None else str(value)


def text_submission(text: str, stored: object) -> str | None:
    """What committing a text field sends, or None when it sends nothing.

    The space around it goes. A model path arrives pasted at least as often as
    it is typed, and a trailing newline is not part of a filename.

    An empty field sends nothing. There is no value that means "no model": the
    render loop keeps rendering whatever it already holds, so an empty commit
    would only put the state out of step with what is on screen. The field goes
    back to showing the store instead.

    Committing what is already there sends nothing either, so pressing Enter on
    an untouched field is not a reload, and neither is the escape that puts the
    original text back before the field reports it.
    """
    typed = text.strip()
    if not typed or typed == text_value(stored):
        return None
    return typed


def text_hold(
    text: str, submitted: str | None, *, active: bool, committed: bool
) -> str | None:
    """What a text field shows after one frame, or None to show the store.

    While the field is under the hand the hold is the buffer in it, so a remote
    write landing mid edit cannot pull half typed text out from under the
    performer. On the frame it commits, the hold becomes what was submitted
    rather than what was typed: they differ by the space that was stripped, and
    holding the untrimmed text would mean holding a value the store can never
    agree with, which then has to lapse instead of releasing.

    A commit that submits nothing lets go at once. The edit was abandoned or it
    said nothing new, so the field belongs to the store again on that frame
    rather than showing text nothing will ever come back to confirm.
    """
    if committed:
        return submitted
    return text if active else None


def field_width(available: float, reserve: float, minimum: float) -> float:
    """How wide a text field is drawn when the row keeps space to its right.

    The field takes everything left over, because the value in it is a path and
    the panel is narrow. `reserve` is what the row draws after it, so the field
    gives up its width first and the button beyond it stays on the row down to
    a panel width where nothing else in the panel works either.
    """
    return max(available - reserve, minimum)


def fitted_width(
    default: float, available: float, reserve: float, minimum: float
) -> float:
    """How wide a widget is drawn so that its whole row fits the panel.

    Its natural width wherever the row already fits, which is every ordinary
    case, and only what is left where it does not. Nothing about the panel
    changes until the alternative is a row running off the edge of it.

    imgui's own default is a fixed sixteen times the font size, and a widget's
    label is drawn to the right of that rather than inside it, so a row costs
    the widget plus the label however narrow the panel is. That is a width that
    grows with the font size while the panel does not, and past the edge it
    grows the window's content instead. Nothing shows the overflow directly:
    what shows is every separator in the panel, which spans the content region
    and so stops well short of rows that have run past it.
    """
    return max(min(default, available - reserve), minimum)


def label_reserve(label: str) -> float:
    """The width a widget's label takes to the right of it, its spacing too.

    Zero for a hidden label, since imgui adds no spacing for a label it does
    not draw.
    """
    width = imgui.calc_text_size(label, None, True).x
    if width <= 0.0:
        return 0.0
    return width + imgui.get_style().item_inner_spacing.x


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
    live: bool,
    frame: int,
) -> Override | None:
    """The local hold after one frame of a widget, or None once it is done.

    A hold is not released on deactivation. The value submitted on that frame
    only reaches the store on the next control tick, so dropping the hold there
    draws the previous state for a frame, which a checkbox shows as a visible
    flick back. It ends when the store agrees instead.

    A control the hand cannot act on has no hold at all. A read only or
    disabled widget can never report a change, so it can neither open one nor
    ever be the thing that clears one, and a hold left on it would draw a value
    the performer has no way left to correct.

    A store that moves to a third value once the widget is released is a
    binding that won the parameter back. The hold ends there too, because
    showing what drives the parameter beats a number frozen at what the hand
    last asked for. While the widget is still held that same disagreement is
    the round trip the hold exists to hide, so it is ignored.

    And a hold lapses. Every other ending waits on a passing state of the
    store, and the UI only looks at the store once a frame: a value written and
    overwritten within one control tick, or while the window is not drawing, is
    a value no frame ever sees. A bool has no third value to fall back on when
    that happens, so a hold that only waited would wait for the rest of the
    session. This is the same ceiling `TouchTracker` puts on a touch, for the
    same reason: state cleared only by an event that may never arrive needs a
    condition that resolves on its own.
    """
    if not live:
        return None
    if changed:
        return Override(value, snapshot, frame)
    if override is None:
        return None
    if values_agree(override.value, snapshot):
        return None
    if active:
        return override
    if not values_agree(override.stored, snapshot):
        return None
    if frame - override.frame >= _HOLD_FRAMES:
        return None
    return override


def next_text_override(
    override: Override | None,
    stored: Value,
    held: str | None,
    *,
    committed: bool,
    active: bool,
    live: bool,
    frame: int,
) -> Override | None:
    """The local hold after one frame of a text field, or None once it is done.

    `next_override` for everything a text field shares with a slider, which is
    every ending: the store catching up, a source winning the parameter back
    once the hand is off it, and the lapse behind both.

    What a slider has no version of is an edit that ends without submitting
    anything, because a slider cannot be released without having changed
    something. Nothing will ever arrive for that hold to agree with, so it ends
    on the frame it commits rather than showing the abandoned text for the half
    second it would otherwise take to lapse.
    """
    if committed and held is None:
        return None
    return next_override(
        override,
        stored,
        held if held is not None else text_value(stored),
        changed=held is not None,
        active=active,
        live=live,
        frame=frame,
    )


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
        self,
        runtime,
        mapping_popup: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runtime = runtime
        self._mapping_popup = mapping_popup
        # The same clock the control loop stamps source values with, or the
        # remote marker would compare two unrelated timelines and either never
        # light or never go out.
        self._clock = clock
        self._local: dict[str, Override] = {}
        self._frame = -1
        self._state: ControlState | None = None
        self._sources = SourceTable()

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
            label,
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
            label,
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
            label,
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
            label,
            lambda shown: imgui.drag_int(
                label, int(shown), speed, int(minimum), int(maximum)
            ),
            enabled,
        )

    def checkbox(self, name: str, label: str, *, enabled: bool = True) -> None:
        spec = require_spec(name, ParamKind.BOOL)
        self._widget(
            spec, label, lambda shown: imgui.checkbox(label, bool(shown)), enabled
        )

    def input_text(
        self,
        name: str,
        label: str,
        *,
        hint: str = "",
        reserve: float = 0.0,
        enabled: bool = True,
    ) -> bool:
        """Draw `name` as a text field, and say whether the hand may use it.

        `reserve` is the width the panel wants for whatever it puts after the
        field on the same row, so the field can take the rest.

        The returned flag is whether the field is live. A panel that draws a
        button beside it disables it on the same flag, because an explicit
        source drives the whole row or none of it: a button that still worked
        beside a read only field would write a value the source's next message
        erases, which is the one thing the read only field is there to prevent.
        """
        spec = require_spec(name, ParamKind.STR)

        def draw(shown: str) -> tuple[bool, str]:
            imgui.set_next_item_width(
                field_width(
                    imgui.get_content_region_avail().x,
                    reserve,
                    imgui.get_font_size() * _FIELD_MIN_EMS,
                )
            )
            return imgui.input_text_with_hint(label, hint, shown, _ENTER)

        return self._text_widget(spec, draw, enabled)

    def _widget(
        self,
        spec: ParamSpec,
        label: str,
        draw: Callable[[Value], tuple[bool, Value]],
        enabled: bool,
    ) -> None:
        name = spec.name
        state = self._snapshot()
        stored = getattr(state, name)
        gutter = gutter_for(state, name, self._sources, self._clock())
        # A control a binding drives is drawn read only. The next value from
        # the source erases a drag, so the hand cannot hold it anyway, and a
        # widget that visibly does nothing is worse than one that says so.
        live = enabled and not gutter.read_only
        imgui.push_id(name)
        self._indicator(name, gutter)
        if not live:
            imgui.begin_disabled()
        self._fit(label)
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
            live=live,
            # `_snapshot` has already read the frame count this frame.
            frame=self._frame,
        )
        if override is None:
            self._local.pop(name, None)
        else:
            self._local[name] = override
        for event in events:
            self._runtime.submit(event)
        self._mapping_menu(name)
        imgui.pop_id()

    def _text_widget(
        self,
        spec: ParamSpec,
        draw: Callable[[str], tuple[bool, str]],
        enabled: bool,
    ) -> bool:
        """One frame of a text field, wired like every other control.

        The same contract as `_widget`: the chip first and outside the disabled
        block, a touch around the edit, the local hold while it is open, and
        read only when an explicit source drives the parameter. What differs is
        when a value leaves, because a text field has no gesture that ends. It
        commits on Enter and on losing focus, which is what the mapping panel's
        fields already do, so both places in the app that take typed text take
        it the same way.

        The touch spans the whole edit rather than the commit, so a source
        writing this parameter cannot pull the text out from under a performer
        who is halfway through typing a path.
        """
        name = spec.name
        state = self._snapshot()
        stored = getattr(state, name)
        gutter = gutter_for(state, name, self._sources, self._clock())
        live = enabled and not gutter.read_only
        imgui.push_id(name)
        self._indicator(name, gutter)
        shown = text_value(displayed_value(self._local, name, stored))
        if not live:
            imgui.begin_disabled()
        entered, text = draw(shown)
        if not live:
            imgui.end_disabled()
        activated = imgui.is_item_activated()
        active = imgui.is_item_active()
        deactivated = imgui.is_item_deactivated()
        committed = live and (entered or imgui.is_item_deactivated_after_edit())
        # Read now, because the mapping popup below draws items of its own and
        # every is_item_ query answers about the last one drawn.
        hovered = imgui.is_item_hovered(
            imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.allow_when_disabled
        )
        submitted = text_submission(text, stored) if committed else None
        held = text_hold(text, submitted, active=active, committed=committed)
        events = widget_events(
            spec,
            submitted,
            activated=activated,
            changed=submitted is not None,
            deactivated=deactivated,
        )
        override = next_text_override(
            self._local.get(name),
            stored,
            held,
            committed=committed,
            active=active,
            live=live,
            # `_snapshot` has already read the frame count this frame.
            frame=self._frame,
        )
        if override is None:
            self._local.pop(name, None)
        else:
            self._local[name] = override
        for event in events:
            self._runtime.submit(event)
        self._mapping_menu(name)
        # An inactive field renders from its first character, so a path wider
        # than the column shows its front and hides the filename, which is the
        # end a performer reads. The tooltip is where the whole of it is
        # legible, including on a field a source has taken read only.
        if hovered and not active and shown:
            imgui.set_tooltip(shown)
        imgui.pop_id()
        return live

    def _fit(self, label: str) -> None:
        """Keep the row inside the panel, whatever its label and the font size.

        imgui's default width is a fixed sixteen times the font size and the
        label sits outside it, so a row costs more than the panel has as soon
        as the font is scaled up or the column is dragged narrow, and what runs
        past the edge grows the window's content instead. Nothing draws the
        overflow: what shows is the separators, which span the content region
        and so stop short of every row that has run past it.

        Only ever narrower, never wider, so a panel that already fits looks
        exactly as it did.
        """
        imgui.set_next_item_width(
            fitted_width(
                imgui.calc_item_width(),
                imgui.get_content_region_avail().x,
                label_reserve(label),
                imgui.get_font_size() * _FIELD_MIN_EMS,
            )
        )

    def _indicator(self, name: str, gutter: Gutter) -> None:
        """Draw the chip left of the widget: who drives it, and a way to change it.

        The gutter is reserved whether or not anything drives the parameter, so
        a binding appearing does not shift the control under the performer's
        cursor. The marker is a drawn rectangle rather than a glyph because the
        bundled font has no symbols.

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
        self,
        origin,
        width: int,
        height: float,
        gutter: Gutter,
        hovered: bool,
    ) -> None:
        """Paint the chip: the slot, a hover plate, then the driver's marker.

        One rectangle on every row, because the chip is the only way into the
        mapping editor from a control a binding has taken read only, and a
        column that only appears once the cursor is already on it is a column
        nobody finds.

        Colour says who drives it and the four differ in brightness as well as
        hue, because with one shape there is nothing else left to differ in.
        Against darcula_darker the idle grey lands near four to one on the
        panel, the blue near eight and the green near eleven, so the pair a red
        green deficiency flattens is also the pair furthest apart in value.

        Fill is the second channel and it says one thing: whether input is
        arriving right now. The outline also drops a little colour, so where a
        display scale rounds the hole away the difference degrades into dim
        against bright instead of into nothing.

        Nothing here changes the layout: the gutter is reserved by the item in
        `_indicator` and every shape is painted inside what it already took.
        """
        draw_list = imgui.get_window_draw_list()
        middle = imgui.ImVec2(origin.x + width * 0.5, origin.y + height * 0.5)
        inset = height * _PLATE_INSET
        top_left = imgui.ImVec2(origin.x, origin.y + inset)
        bottom_right = imgui.ImVec2(origin.x + width, origin.y + height - inset)
        rounding = imgui.get_style().frame_rounding
        stroke = max(1.0, height * _OUTLINE_RATIO)
        if hovered:
            draw_list.add_rect_filled(
                top_left,
                bottom_right,
                imgui.get_color_u32(imgui.Col_.button_hovered),
                rounding,
            )
        red, green, blue, alpha = gutter.color if gutter.color else idle_color()
        if not gutter.filled:
            alpha *= _IDLE_ALPHA
        color = imgui.get_color_u32(imgui.ImVec4(red, green, blue, alpha))
        side = round(height * _MARKER_RATIO)
        corner_min = imgui.ImVec2(middle.x - side * 0.5, middle.y - side * 0.5)
        corner_max = imgui.ImVec2(middle.x + side * 0.5, middle.y + side * 0.5)
        # Outline is the arriving state of remote input and nothing else, so
        # every other colour, which carries no liveness, is always solid.
        if gutter.filled:
            draw_list.add_rect_filled(corner_min, corner_max, color)
        else:
            # Scaled from the frame so the outline keeps its weight at 200
            # percent, floored at a pixel so it never disappears at 100.
            draw_list.add_rect(corner_min, corner_max, color, 0.0, stroke)

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
        it over and silently lose every binding marker. The source table is
        read on the same frame boundary, for the same reason: the marker saying
        a parameter is being written from outside has to be as old as the value
        it is drawn beside.
        """
        frame = imgui.get_frame_count()
        if self._state is None or frame != self._frame:
            self._frame = frame
            self._state = self._runtime.control_store.snapshot()
            self._sources = self._runtime.source_store.snapshot()
        return self._state
