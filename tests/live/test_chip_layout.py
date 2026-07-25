"""The chip column, drawn for real against the real theme.

imgui cannot be driven with a window here, but it can be driven without one:
a context, a display size and the textures backend flag are enough to run a
frame and let the layout resolve. That is worth the setup for exactly one
question, because it is the question a screenshot raised and nothing else can
answer. A performer reported the chip appearing on some rows and not others,
which would have meant a rule that quietly applied to five parameters and not
six. It did not, but nothing in the suite could say so, and the only reason it
could be checked at all was by hand at the end of a build.

What is pinned here is everything that only exists once something is painted:
that every bindable parameter reserves the same gutter, that a rectangle lands
in it whatever drives the parameter, that a solid rectangle and an outlined one
really are different, and that the palette separates in brightness against the
theme the app runs under. Which driver wins and what a tooltip says are
`test_controls.py`, which needs no context at all.
"""

import itertools

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui, immvision

from autolume.live.core.params import Binding, ControlState
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.ui.controls import (
    BINDING_COLOR,
    ERROR_COLOR,
    MOTION_COLOR,
    ControlBinder,
    Marker,
    idle_color,
)
from autolume.live.ui.panels.mapping import bindable_specs
from autolume.live.ui.panels.perform import PerformPanel, button_width
from autolume.live.ui.panels.preview import _DIM_ALPHA, PreviewPanel

NOW = 100.0
IDLE = ControlState()
SILENT = SourceTable()


class FakeRuntime:
    def __init__(self, state, sources):
        self.control_store = LatestValueStore(state)
        self.source_store = LatestValueStore(sources)

    def submit(self, event):
        pass


@pytest.fixture
def frame():
    """One imgui frame, with the theme the app actually runs under.

    The theme matters: `darcula_darker` puts the frame colour within a hair of
    the window colour, which is what made an earlier chip drawn in it invisible.
    Anything asserting about what a performer can see has to run under it.
    """
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        # Destroying a context flushes settings to disk, so without this the
        # suite drops an imgui.ini wherever it was run from.
        io.set_ini_filename(None)
        io.display_size = imgui.ImVec2(900.0, 700.0)
        io.delta_time = 1.0 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        hello_imgui.apply_tweaked_theme(
            hello_imgui.RunnerParams().imgui_window_params.tweaked_theme
        )
        imgui.new_frame()
        imgui.begin("Controls")
        yield
        imgui.end()
        imgui.render()
    finally:
        imgui.destroy_context(context)


def painted(state, sources, name):
    """How many vertices the chip for `name` puts in the draw list.

    Zero means nothing was painted. Otherwise the count separates a solid shape
    from an outline of the same shape, which is the one visual distinction in
    this gutter that no pure function can be asked about. Compared against each
    other rather than against a literal, so an imgui that tessellates
    differently does not fail the suite.
    """
    counts = {}
    original = ControlBinder._chip_shape
    try:

        def measure(self, origin, width, height, gutter, hovered):
            draw_list = imgui.get_window_draw_list()
            before = draw_list.vtx_buffer.size()
            original(self, origin, width, height, gutter, hovered)
            counts[measure.name] = draw_list.vtx_buffer.size() - before

        ControlBinder._chip_shape = measure
        original_indicator = ControlBinder._indicator

        def label(self, parameter, gutter):
            measure.name = parameter
            original_indicator(self, parameter, gutter)

        ControlBinder._indicator = label
        try:
            draw_every_bindable(state, sources)
        finally:
            ControlBinder._indicator = original_indicator
    finally:
        ControlBinder._chip_shape = original
    return counts[name]


def widget_specs():
    """The bindable parameters, every one of which is drawn with a gutter.

    The model path used to be left out, because it was a button and a label
    drawn by hand rather than a control. It is a text field through the binder
    now, so the row it draws is the same row as every other one and the
    assertions below cover it.
    """
    return bindable_specs()


def draw_every_bindable(state=IDLE, sources=SILENT):
    """Draw one widget per bindable parameter, and report each one's gutter.

    Driven off the registry rather than off a copy of the perform panel, so a
    parameter added later is covered here the day it is added.
    """
    binder = ControlBinder(
        FakeRuntime(state, sources), mapping_popup=lambda name: None, clock=lambda: NOW
    )
    draw = {
        "float": binder.drag_float,
        "int": binder.drag_int,
        "bool": binder.checkbox,
        "str": binder.input_text,
    }
    rows = []
    original = ControlBinder._indicator
    try:

        def record(self, name, gutter):
            start = imgui.get_cursor_screen_pos().x
            original(self, name, gutter)
            rows.append((name, imgui.get_cursor_screen_pos().x - start, gutter))

        ControlBinder._indicator = record
        for index, spec in enumerate(widget_specs()):
            # A separator every few rows, because the reported split fell either
            # side of one and a layout fault could plausibly have lived there.
            if index % 4 == 0:
                imgui.separator_text(f"Group {index}")
            draw[spec.kind.value](spec.name, spec.name)
    finally:
        ControlBinder._indicator = original
    return rows


def test_every_bindable_parameter_reserves_the_same_gutter(frame):
    rows = draw_every_bindable()
    assert len(rows) == len(widget_specs())
    assert len({round(width, 3) for _, width, _ in rows}) == 1
    assert all(width > 0.0 for _, width, _ in rows)


def test_every_row_carries_a_marker_even_when_nothing_drives_it(frame):
    # The column has to advertise itself without being hovered, which is the
    # whole reason the idle state is a grey marker and not an empty container.
    rows = draw_every_bindable()
    assert all(gutter.marker is Marker.NONE for _, _, gutter in rows)


def luminance(color):
    """Relative luminance of `color` composited onto the panel background.

    The chip is painted straight onto the panel, so a colour carrying alpha is
    only as bright as what shows through, and comparing raw tuples would flatter
    every translucent one.
    """
    window = imgui.get_style_color_vec4(imgui.Col_.window_bg)
    alpha = color[3] if len(color) > 3 else 1.0
    over = [
        channel * alpha + background * (1.0 - alpha)
        for channel, background in zip(color[:3], (window.x, window.y, window.z))
    ]
    red, green, blue = (channel**2.2 for channel in over)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(one, other):
    return (max(one, other) + 0.05) / (min(one, other) + 0.05)


def test_the_idle_marker_stands_off_the_panel_it_is_drawn_on(frame):
    """The grey has to be visible without a hover, in the real theme.

    An earlier pass drew the idle chip in the frame colour, which in this theme
    is (0.145, 0.122, 0.122) against a window of (0.138, 0.142, 0.149). That is
    the background, so it could not have worked whatever the alpha. Asserting a
    real separation is what stops the next choice from going the same way, and
    it goes through `idle_color` so the check cannot drift from the drawing.
    """
    window = imgui.get_style_color_vec4(imgui.Col_.window_bg)
    panel = luminance((window.x, window.y, window.z))
    assert contrast(luminance(idle_color()), panel) > 3.0


def test_the_idle_grey_and_the_motion_green_differ_in_brightness(frame):
    """Every marker is the same rectangle, so colour alone says who drives it.

    Grey against green is the pair a red green deficiency flattens, and hue is
    exactly what such an eye loses, so the difference has to survive in value.
    A pale green at a mid grey's brightness, which is what this was, is the one
    combination that does not.
    """
    assert contrast(luminance(MOTION_COLOR), luminance(idle_color())) > 2.0


def test_no_two_marker_colours_are_the_same_brightness(frame):
    palette = (idle_color(), MOTION_COLOR, BINDING_COLOR, ERROR_COLOR)
    values = sorted(luminance(color) for color in palette)
    assert all(
        contrast(low, high) > 1.2 for low, high in itertools.pairwise(values)
    )


def test_an_idle_row_actually_paints_something(frame):
    # The affordance is the marker being there, so "nothing drives it" has to
    # put ink in the gutter rather than take the early return it used to.
    assert painted(IDLE, SILENT, "latent_x") > 0


def test_an_idle_remote_row_is_drawn_hollow_and_a_receiving_one_solid(frame):
    """Fill is liveness, and this is the only place it can be checked.

    A solid rectangle and an outline of the same rectangle are the same colour
    and the same bounds, so nothing above the draw list can tell them apart.
    """
    sources = SourceTable().observe("/trunc/psi", 0.5, NOW)
    state = ControlState(bindings=(Binding("truncation_psi", ""),))
    receiving = painted(state, sources, "truncation_psi")
    silent = painted(state, SILENT, "truncation_psi")
    assert silent > receiving


def test_a_driven_parameter_is_marked_apart_from_an_idle_one(frame):
    sources = SourceTable().observe("/trunc/psi", 0.5, NOW)
    state = ControlState(bindings=(Binding("truncation_psi", ""),))
    rows = {name: gutter for name, _, gutter in draw_every_bindable(state, sources)}
    assert rows["truncation_psi"].marker is Marker.BINDING
    assert rows["truncation_psi"].filled
    assert rows["latent_x"].marker is Marker.NONE


class FakeModelHost:
    def __init__(self, pending=None, error=None, current=None):
        self._pending = pending
        self._error = error
        self._current = current

    def pending(self):
        return self._pending

    def error(self):
        return self._error

    def current(self):
        return self._current


class FakePreview:
    def latest(self):
        return -1, None


class FakeRenderLoop:
    def fps(self):
        return 60.0


class FakeOsc:
    port = 1338


class PanelRuntime(FakeRuntime):
    """A runtime with the parts the two panels read, and nothing behind them."""

    def __init__(self, state=IDLE, sources=SILENT, host=None):
        super().__init__(state, sources)
        self.model_host = host or FakeModelHost()
        self.preview = FakePreview()
        self.render_loop = FakeRenderLoop()
        self.osc = FakeOsc()


def model_field(state, **kwargs):
    """Draw the model field for one frame and report whether it is live."""
    binder = ControlBinder(
        FakeRuntime(state, SILENT), mapping_popup=lambda name: None, clock=lambda: NOW
    )
    return binder.input_text("pkl_path", "##model", **kwargs)


def test_a_source_driving_the_model_takes_the_field_and_the_button_with_it(frame):
    """The ownership rule, on the row where it now has two things to say.

    The returned flag is what the panel disables Browse on, so a field drawn
    read only beside a button that still opened a dialog would be the one
    inconsistency this rule exists to prevent: the next message from the source
    erases whatever the dialog picked.
    """
    driven = ControlState(bindings=(Binding("pkl_path", "/td/model"),))
    assert not model_field(driven)
    assert model_field(IDLE)
    # A row that is on with no source of its own is a row the hand keeps, the
    # same as every other parameter.
    assert model_field(ControlState(bindings=(Binding("pkl_path", ""),)))
    assert not model_field(IDLE, enabled=False)


def test_a_button_is_as_wide_as_the_row_reserves_for_it(frame):
    # The row hands the field everything except this, so a measurement that
    # disagreed with the button would put Browse over the panel edge.
    reserved = button_width("Browse")
    imgui.button("Browse")
    assert imgui.get_item_rect_size().x == pytest.approx(reserved)


LONG_PATH = "/Users/vj/Documents/autolume/models/wikiart-1024.pkl"


def row_edges(width: float, font_scale: float) -> list[float]:
    """Every row's right edge in the perform panel, against the content edge.

    Zero is flush with the edge of the content region, which is where the
    separators end and as far as anything may go. Positive is a row that has
    run off the panel and grown the window's content behind it.

    A window of its own rather than the shared frame, because width and font
    size are the two things that make a row overflow and neither can be changed
    inside a frame that has already begun.
    """
    context = imgui.create_context()
    edges: list[float] = []
    right = [0.0]
    original_widget = ControlBinder._widget
    original_row = PerformPanel._model_row
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = imgui.ImVec2(1280.0, 800.0)
        io.delta_time = 1.0 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        hello_imgui.apply_tweaked_theme(
            hello_imgui.RunnerParams().imgui_window_params.tweaked_theme
        )
        imgui.get_style().font_scale_main = font_scale

        def measure_widget(self, spec, label, draw, enabled):
            original_widget(self, spec, label, draw, enabled)
            edges.append(imgui.get_item_rect_max().x - right[0])

        def measure_row(self):
            # After the whole row, so the last item measured is Browse and not
            # the field, which is the half that could be pushed off the edge.
            original_row(self)
            edges.append(imgui.get_item_rect_max().x - right[0])

        ControlBinder._widget = measure_widget
        PerformPanel._model_row = measure_row
        panel = PerformPanel(
            PanelRuntime(ControlState(pkl_path=LONG_PATH)),
            mapping_popup=lambda name: None,
        )
        # Three frames, because a scrollbar is decided from the frame before
        # it and the row has to fit the panel it is actually drawn in.
        for _ in range(3):
            imgui.new_frame()
            imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
            # Short on purpose: the panel scrolls in the dock it ships in, and
            # a vertical scrollbar is what takes the width the rows measure.
            imgui.set_next_window_size(imgui.ImVec2(width, 300.0))
            imgui.begin("Controls")
            edges.clear()
            right[0] = (
                imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
            )
            panel.gui()
            imgui.end()
            imgui.render()
    finally:
        ControlBinder._widget = original_widget
        PerformPanel._model_row = original_row
        imgui.destroy_context(context)
    return edges


@pytest.mark.parametrize("font_scale", (1.0, 1.5, 2.0))
@pytest.mark.parametrize("width", (448.0, 360.0, 280.0))
def test_no_row_runs_past_the_panel_it_is_drawn_in(width, font_scale):
    """The panel's content may never be wider than the panel.

    imgui's default item width is a fraction of the window and the label is
    drawn outside it, so a row costs more than it declares and grows with the
    font while the panel does not. What overflows is not itself visible: what
    shows is every separator in the panel, which spans the content region and
    so stops short of the rows that have run past it.

    448 is the docked width at the shipped window size, and the font scale is
    the UI font size preference, which is why the worst case is a fair test
    rather than a contrived one.
    """
    edges = row_edges(width, font_scale)
    assert len(edges) == len(bindable_specs())
    assert max(edges) <= 0.0


def test_the_model_row_uses_the_width_it_is_given_and_no_more():
    """The field takes what is left, so the row is flush with the edge.

    Being under is a field that gave up width it could have used for a path.
    Being over is the row growing the panel's content. The row measures the
    button beside it, so it can be exactly right rather than close.
    """
    for width in (448.0, 360.0, 280.0):
        assert row_edges(width, 1.0)[0] == pytest.approx(0.0, abs=1.0)


def dimmed_white(alpha):
    """The brightest frame a model can produce, under a dim of `alpha`."""
    return (1.0 - alpha, 1.0 - alpha, 1.0 - alpha)


def test_the_dim_carries_the_status_over_the_brightest_frame(frame):
    """The words land on arbitrary imagery, so no text colour is safe alone.

    The dim has to be set for the worst case the model can produce, which is
    white, and not for whichever painting happens to be loaded. The red is what
    binds rather than the white: it is a mid luminance colour, so it has the
    least room of anything drawn here, and it carries the message that most
    needs reading.
    """
    dimmed = luminance(dimmed_white(_DIM_ALPHA))
    text = imgui.get_style_color_vec4(imgui.Col_.text)
    assert contrast(luminance((text.x, text.y, text.z)), dimmed) > 7.0
    assert contrast(luminance(ERROR_COLOR), dimmed) > 4.0


def test_a_lighter_dim_would_not_carry_the_failure_colour(frame):
    # The check above only means something if it could fail. At three fifths
    # the red is under two to one, which is a red no one can read.
    assert contrast(luminance(ERROR_COLOR), luminance(dimmed_white(0.6))) < 2.0


def test_the_dim_leaves_the_frame_underneath_visible(frame):
    """The reason the previous model is not unloaded is that it keeps showing.

    A dim that took the picture to black would give the same legibility and
    throw away what it was protecting, which is the performer watching what is
    still rendering while the next model loads.
    """
    assert _DIM_ALPHA <= 0.85


class FramePreview:
    """A mailbox with a frame in it, so the preview has something to be quiet
    about. What it holds never reaches immvision, which needs a GL context."""

    def latest(self):
        return 7, np.zeros((8, 8, 3), dtype=np.uint8)


def overlay_paint(runtime):
    """What the status overlay costs: vertices drawn, and layout taken.

    Measured around the overlay itself, inside the child it paints into, which
    is the only place either question can be asked. Zero vertices really is
    nothing drawn, and a cursor that has not moved really is nothing placed.

    Everything else runs for real. Only immvision is stood in for, because it
    uploads a texture and there is no GL context here, and what it was handed
    is recorded so the frame can be asked about afterwards.
    """
    result = []
    shown = []
    original_overlay = PreviewPanel._overlay
    original_display = immvision.image_display
    try:

        def measure(self, status, origin, area, has_frame):
            draw_list = imgui.get_window_draw_list()
            before = draw_list.vtx_buffer.size()
            cursor = imgui.get_cursor_pos()
            original_overlay(self, status, origin, area, has_frame)
            moved = imgui.get_cursor_pos() != cursor
            result.append((draw_list.vtx_buffer.size() - before, moved))

        def display(label, image, size=None, refresh_image=False, **kwargs):
            shown.append((image, size, refresh_image))
            # immvision places an item of the size it was asked for, and the
            # cursor having been moved to centre it is only legal because
            # something is then submitted there.
            imgui.dummy(imgui.ImVec2(float(size[0]), float(size[1])))
            return (0.0, 0.0)

        PreviewPanel._overlay = measure
        immvision.image_display = display
        # A panel with room in it. The window this fixture opens is auto sized
        # and has next to none, which is the zero area case rather than this
        # one.
        imgui.begin_child("##panel", imgui.ImVec2(400.0, 300.0))
        PreviewPanel(runtime).gui()
        imgui.end_child()
    finally:
        PreviewPanel._overlay = original_overlay
        immvision.image_display = original_display
    return result[0] + (shown,)


def test_a_preview_with_nothing_to_say_paints_nothing_at_all(frame):
    """No plate, no glyph, no ink, over a frame that is doing fine.

    The status sits on the image now, so "nothing to say" has to mean nothing
    drawn rather than an empty line the way it did when it had a row of its
    own. A plate over a running preview would be a permanent grey smudge in
    the middle of the picture.
    """
    running = PanelRuntime(host=FakeModelHost(current=object()))
    running.preview = FramePreview()
    ink, moved, shown = overlay_paint(running)
    assert (ink, moved) == (0, False)
    # And the frame itself did go out, so this is a quiet preview rather than
    # an empty one.
    assert len(shown) == 1


def test_the_status_is_painted_over_the_frame_rather_than_placed_above_it(frame):
    # It takes no layout in either state, which is what lets it sit on the
    # image instead of pushing it down the way the old line did.
    failing = PanelRuntime(host=FakeModelHost(error="Could not load the model"))
    failing.preview = FramePreview()
    ink, moved, _ = overlay_paint(failing)
    assert ink > 0
    assert not moved


def test_only_a_preview_with_a_frame_in_it_is_dimmed(frame):
    """With nothing rendered there is nothing to dim.

    The words already stand out on an empty panel, so a dim there would be a
    grey rectangle explaining itself. Measured as ink rather than asserted,
    because the rectangle is the only difference between the two states.
    """
    host = FakeModelHost(error="Could not load the model")
    empty = PanelRuntime(host=host)
    over_frame = PanelRuntime(host=host)
    over_frame.preview = FramePreview()
    assert overlay_paint(over_frame)[0] > overlay_paint(empty)[0]


def test_the_dim_never_reaches_the_frame_the_sinks_are_handed(frame):
    """The dim is presentation and belongs to this panel only.

    The render loop fans the same frames out to every sink, and the parity plan
    adds NDI, a recorder and a fullscreen output, so a dim that reached the
    frame would dim the show and the recording on every model switch. It is a
    rectangle on a draw list, and this is what says the array is not part of
    it.
    """
    original = np.full((8, 8, 3), 210, dtype=np.uint8)
    untouched = original.copy()

    class OneFrame:
        def latest(self):
            return 7, original

    runtime = PanelRuntime(host=FakeModelHost(error="Could not load the model"))
    runtime.preview = OneFrame()
    ink, _, shown = overlay_paint(runtime)
    assert ink > 0
    # Dimmed on screen, and the very same array, byte for byte, on its way to
    # the drawing. Nothing copied it and nothing wrote to it.
    (image, _size, _refresh), = shown
    assert image is original
    assert np.array_equal(original, untouched)


def test_the_preview_never_makes_its_panel_scroll(frame):
    """The status takes no layout, at any panel size, however long it is.

    It is painted at a position rather than placed, which is what lets it sit
    on the image without having pushed it down first. A message wider than the
    panel is the case that would show it up, so that is the one drawn here.
    """
    failure = FakeModelHost(error="Could not load the model. " * 6)
    for index, size in enumerate(((900.0, 300.0), (200.0, 600.0), (60.0, 60.0))):
        imgui.begin_child(f"##host{index}", imgui.ImVec2(*size))
        PreviewPanel(PanelRuntime(host=failure)).gui()
        assert imgui.get_scroll_max_x() == 0.0
        assert imgui.get_scroll_max_y() == 0.0
        imgui.end_child()


def test_the_preview_draws_its_status_line_in_every_state(frame):
    """Each state pushes a colour it then has to pop.

    An unbalanced style stack is not a wrong pixel, it is imgui asserting at
    the end of the frame, so drawing all three inside one is the check.
    """
    for host in (
        FakeModelHost(),
        FakeModelHost(pending="/models/wikiart.pkl"),
        FakeModelHost(error="No such file"),
    ):
        PreviewPanel(PanelRuntime(host=host)).gui()
