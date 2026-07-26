"""The driver marker column, drawn for real against the real theme.

imgui cannot be driven with a window here, but it can be driven without one:
a context, a display size and the textures backend flag are enough to run a
frame and let the layout resolve. That is worth the setup for exactly one
question, because it is the question a screenshot raised and nothing else can
answer. A performer reported the marker appearing on some rows and not others,
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

import collections
import dataclasses
import itertools
from typing import Callable

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui, immvision

from autolume.live.core.generator import ModelInfo
from autolume.live.core.params import Binding, ControlState
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.ui import theme, window
from autolume.live.ui.theme import BINDING_COLOR, ERROR_COLOR, MOTION_COLOR
from autolume.live.ui.controls import (
    ControlBinder,
    Marker,
    idle_color,
)
from autolume.live.ui.panels import preview as preview_module
from autolume.live.ui.panels.loop import LoopPanel
from autolume.live.ui.panels.mapping import bindable_specs
from autolume.live.ui.panels.perform import PerformPanel, button_width
from autolume.live.ui.panels.preview import (
    _DIM_ALPHA,
    DisplayMode,
    PreviewPanel,
)

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
    the window colour, which is what made an earlier marker drawn in it invisible.
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
        theme.apply_theme()
        imgui.new_frame()
        imgui.begin("Controls")
        yield
        imgui.end()
        imgui.render()
    finally:
        imgui.destroy_context(context)


def painted(state, sources, name):
    """How many vertices the driver marker for `name` puts in the draw list.

    Zero means nothing was painted. Otherwise the count separates a solid shape
    from an outline of the same shape, which is the one visual distinction in
    this gutter that no pure function can be asked about. Compared against each
    other rather than against a literal, so an imgui that tessellates
    differently does not fail the suite.
    """
    counts = {}
    original = ControlBinder._driver_marker_shape
    try:

        def measure(self, origin, width, height, gutter, hovered):
            draw_list = imgui.get_window_draw_list()
            before = draw_list.vtx_buffer.size()
            original(self, origin, width, height, gutter, hovered)
            counts[measure.name] = draw_list.vtx_buffer.size() - before

        ControlBinder._driver_marker_shape = measure
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
        ControlBinder._driver_marker_shape = original
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

    The marker is painted straight onto the panel, so a colour carrying alpha is
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

    An earlier pass drew the idle marker in the frame colour, which in this theme
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
    assert all(contrast(low, high) > 1.2 for low, high in itertools.pairwise(values))


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


class FakeControlLoop:
    """Stands in for the one thing the Loop panel reads off `ControlLoop`."""

    def __init__(self, noise_table_key=None):
        self.noise_table_key = noise_table_key


class PanelRuntime(FakeRuntime):
    """A runtime with the parts the panels read, and nothing behind them."""

    def __init__(self, state=IDLE, sources=SILENT, host=None):
        super().__init__(state, sources)
        self.model_host = host or FakeModelHost()
        self.preview = FakePreview()
        self.render_loop = FakeRenderLoop()
        self.osc = FakeOsc()
        self.model_info_store = LatestValueStore(None)
        self.control_loop = FakeControlLoop()


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
        theme.apply_theme()
        imgui.get_style().font_scale_main = font_scale

        def measure_widget(self, spec, label, draw, enabled, **kwargs):
            original_widget(self, spec, label, draw, enabled, **kwargs)
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

    The row count is not checked against the full registry here: this panel
    draws its own fixed set of rows, one per bindable parameter it has chosen
    to show, and that set is smaller than the registry once parameters exist
    that live in another panel. It is checked against a literal instead
    (below), which is the guard `max(edges) <= 0.0` alone cannot stand in
    for: a row dropped from `PerformPanel.gui` draws one fewer edge, all of
    them still `<= 0.0`, and this parametrization would keep passing. Update
    the literal, deliberately, whenever a row is added to or removed from
    this panel.
    """
    edges = row_edges(width, font_scale)
    assert max(edges) <= 0.0
    # 8 latent rows (vector mode, project, latent x/y, animate, speed x/y,
    # truncation) + 4 noise rows + 1 render row, drawn through `_widget`, plus
    # the model row, measured separately since it is a text field.
    assert len(edges) == 14


def bound_rows(gui: Callable[[], None]) -> int:
    """How many `ControlBinder` rows one call to `gui` draws.

    Wraps both `_widget` (every numeric or bool row) and `_text_widget`
    (every text row), so this counts a panel's bound rows whichever kind
    they are, the same purpose `row_edges` serves for `PerformPanel` but
    without needing a docked window to measure against.
    """
    count = 0
    original_widget = ControlBinder._widget
    original_text = ControlBinder._text_widget

    def widget(self, *args, **kwargs):
        nonlocal count
        count += 1
        return original_widget(self, *args, **kwargs)

    def text_widget(self, *args, **kwargs):
        nonlocal count
        count += 1
        return original_text(self, *args, **kwargs)

    ControlBinder._widget = widget
    ControlBinder._text_widget = text_widget
    try:
        gui()
    finally:
        ControlBinder._widget = original_widget
        ControlBinder._text_widget = original_text
    return count


def test_no_loop_panel_row_is_silently_dropped(frame):
    """The same guard as the perform panel's, for the Loop panel's own rows.

    A row quietly removed from `LoopPanel.gui` still paints whatever is left
    with `max(edges) <= 0.0` unbothered, since fewer rows is not a wider one;
    only a count pinned to what the panel actually draws catches that, which
    is what task 9's review found missing here too.
    """
    panel = LoopPanel(PanelRuntime(), mapping_popup=lambda name: None)
    # Transport (5) + keyframe count (1) + scrub (2) + noise loop (3) +
    # pulse (3, all text fields). The six default keyframe rows are plain
    # widgets, not `ControlBinder` rows, and are not part of this count.
    assert bound_rows(panel.gui) == 14


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


def rendered(value=0, shape=(8, 8, 3)):
    """A frame as the render loop hands it out: read-only.

    Anything standing in for the mailbox has to produce one of these, or it is
    testing a preview that no longer exists.
    """
    frame = np.full(shape, value, dtype=np.uint8)
    frame.flags.writeable = False
    return frame


class FramePreview:
    """A mailbox with a frame in it, so the preview has something to be quiet
    about. What it holds never reaches the GPU, which is not here."""

    def latest(self):
        return 7, rendered()


class FakeTexture:
    """Stands in for immvision.GlTexture, which needs a GL context.

    Everything around it runs for real, including the copy the panel makes and
    the quad imgui places, so what is faked is exactly the upload and nothing
    else. It refuses a read-only array the way the real binding does: it
    converts through a mutable cv::Mat, and a stand-in that accepted one would
    let the panel pass here and fail on the first frame in the app.
    """

    texture_id = 1

    def __init__(self):
        self.uploads = []

    def update_from_image(self, image, is_color_order_bgr=False):
        if not image.flags.writeable:
            raise TypeError("update_from_image(): incompatible function arguments")
        self.uploads.append((image, is_color_order_bgr, image.copy()))


class FakeEnlarger:
    """Stands in for the framebuffer blit, which needs a GL context.

    Calling into GL without one does not raise, it takes the process down, so
    this is a stand-in rather than a guard inside the panel. A guard would be a
    branch that only ever runs here and never in the app, which is the shape
    that let a broken magnification ship silently once already.

    It records what it was asked for, which is worth more than a no-op: the
    only headless evidence that a magnified frame is resampled at all, and at
    the size the quad will be drawn at.
    """

    def __init__(self):
        self.calls = []

    def enlarge(self, source, frame, size):
        self.calls.append((source, frame, size))
        return 2


Painted = collections.namedtuple("Painted", "ink moved panel quads enlarged")

# Two panels drawn in one frame must not share a child id, or the second one
# resolves to nothing and its assertions pass on an empty layout.
_panel_ids = itertools.count()


def draw_preview(runtime, panel=None, sizes=((400.0, 300.0),)):
    """Draw a preview panel once per size, and report what it cost.

    The overlay's ink and whether it took any layout, from the first pass, the
    panel itself so the uploads it made can be asked about, and the size of
    every quad the frame was drawn as.
    """
    result = []
    quads = []
    original_overlay = PreviewPanel._overlay
    original_texture = immvision.GlTexture
    original_enlarger = preview_module.Enlarger
    original_image = imgui.image
    try:

        def image(tex_ref, image_size, *args, **kwargs):
            quads.append((image_size.x, image_size.y))
            original_image(tex_ref, image_size, *args, **kwargs)

        def measure(self, status, origin, area, has_frame):
            draw_list = imgui.get_window_draw_list()
            before = draw_list.vtx_buffer.size()
            cursor = imgui.get_cursor_pos()
            original_overlay(self, status, origin, area, has_frame)
            moved = imgui.get_cursor_pos() != cursor
            result.append((draw_list.vtx_buffer.size() - before, moved))

        PreviewPanel._overlay = measure
        immvision.GlTexture = FakeTexture
        preview_module.Enlarger = FakeEnlarger
        imgui.image = image
        panel = panel or PreviewPanel(runtime)
        for size in sizes:
            # A panel with room in it. The window this fixture opens is auto
            # sized and has next to none, which is the zero area case rather
            # than this one.
            imgui.begin_child(f"##panel{next(_panel_ids)}", imgui.ImVec2(*size))
            panel.gui()
            imgui.end_child()
    finally:
        PreviewPanel._overlay = original_overlay
        immvision.GlTexture = original_texture
        preview_module.Enlarger = original_enlarger
        imgui.image = original_image
    return Painted(
        result[0][0],
        result[0][1],
        panel,
        quads,
        panel._enlarger.calls if panel._enlarger is not None else [],
    )


def uploads_of(panel):
    return panel._texture.uploads if panel._texture is not None else []


def test_a_preview_with_nothing_to_say_paints_nothing_at_all(frame):
    """No dim, no glyph, no ink, over a frame that is doing fine.

    The status sits on the image now, so "nothing to say" has to mean nothing
    drawn rather than an empty line the way it did when it had a row of its
    own. A dim over a running preview would be a permanent grey wash over the
    middle of the picture.
    """
    running = PanelRuntime(host=FakeModelHost(current=object()))
    running.preview = FramePreview()
    painted = draw_preview(running)
    assert (painted.ink, painted.moved) == (0, False)
    # And the frame itself did go up, so this is a quiet preview rather than an
    # empty one.
    assert len(uploads_of(painted.panel)) == 1


def test_the_status_is_painted_over_the_frame_rather_than_placed_above_it(frame):
    # It takes no layout in either state, which is what lets it sit on the
    # image instead of pushing it down the way the old line did.
    failing = PanelRuntime(host=FakeModelHost(error="Could not load the model"))
    failing.preview = FramePreview()
    painted = draw_preview(failing)
    assert painted.ink > 0
    assert not painted.moved


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
    assert draw_preview(over_frame).ink > draw_preview(empty).ink


def test_the_dim_never_reaches_the_frame_the_sinks_are_handed(frame):
    """The dim is presentation and belongs to this panel only.

    The render loop fans the same frames out to every sink, and the parity plan
    adds NDI, a recorder and a fullscreen output, so a dim that reached the
    frame would dim the show and the recording on every model switch. It is a
    rectangle on a draw list, and this is what says the array is not part of
    it.
    """
    original = rendered(210)
    untouched = original.copy()

    class OneFrame:
        def latest(self):
            return 7, original

    runtime = PanelRuntime(host=FakeModelHost(error="Could not load the model"))
    runtime.preview = OneFrame()
    painted = draw_preview(runtime)
    assert painted.ink > 0
    # Dimmed on screen, and byte for byte the frame that arrived on its way to
    # the GPU. The panel copies because the uploader will not take a read-only
    # array and for no other reason, so the copy carries the picture and not
    # the dim, and the frame the other sinks hold is the one it came in as.
    ((image, _bgr, _pixels),) = uploads_of(painted.panel)
    assert np.array_equal(image, original)
    assert np.array_equal(original, untouched)
    assert not np.shares_memory(image, original)


def test_the_preview_draws_a_frame_it_is_not_allowed_to_write_to(frame):
    """End to end on the array the render loop actually hands out.

    Read-only is what keeps one sink from corrupting every other, and it is
    also the one thing the uploader will not accept, since it converts through
    a mutable cv::Mat. So the panel copies, and this is what says the copy is
    there and that the picture survived it.
    """
    runtime = PanelRuntime(host=FakeModelHost(current=object()))
    runtime.preview = type(
        "OneFrame", (), {"latest": lambda self: (3, rendered(140))}
    )()
    ((image, bgr, _pixels),) = uploads_of(draw_preview(runtime).panel)
    assert image.flags.writeable
    assert np.array_equal(image, rendered(140))
    # The frames are RGB. Leaving this to immvision's global colour order would
    # be a silent swap of red and blue in the only place anyone would see it.
    assert bgr is False


def test_a_still_preview_uploads_once_rather_than_every_gui_pass(frame):
    """The whole reason the mailbox carries a sequence number.

    A megabyte image copied and uploaded on every UI pass would be paying for a
    frame that has not changed, several times per rendered frame, on the UI
    thread.
    """
    runtime = PanelRuntime(host=FakeModelHost(current=object()))
    runtime.preview = FramePreview()
    panel = draw_preview(runtime, sizes=((400.0, 300.0),) * 3).panel
    assert len(uploads_of(panel)) == 1
    assert np.array_equal(uploads_of(panel)[0][2], rendered())


def test_resizing_the_panel_does_not_upload_the_frame_again(frame):
    """The texture holds the frame, the quad carries the size.

    An earlier pass uploaded on a size change too, because immvision built its
    texture at the size it was asked for. Drawing the quad here is what made
    that unnecessary, and a dock split being dragged is a stream of new sizes.
    """
    runtime = PanelRuntime(host=FakeModelHost(current=object()))
    runtime.preview = FramePreview()
    panel = draw_preview(
        runtime, sizes=((400.0, 300.0), (250.0, 300.0), (250.0, 180.0))
    ).panel
    assert len(uploads_of(panel)) == 1


def test_immvision_would_letterbox_and_left_align_what_it_is_given(frame):
    """Why the frame is drawn as a quad instead of by `image_display`.

    `image_display` fits what it is handed into the size it is asked for with
    the aspect ratio kept, and pins the result to the top left of that box. A
    square frame asked to fill a wide box therefore comes back scaled by the
    short axis and starting at the left, which is Fit's shape in Stretch's
    place and the alignment of neither. Asserted on the matrix immvision
    exposes, so that if it ever centres and stretches, the reason for owning
    the draw is known to have gone.
    """
    matrix = immvision.make_zoom_pan_matrix_full_view((1024, 1024), (800, 400))
    horizontal, vertical = matrix[0][0], matrix[1][1]
    assert horizontal == vertical == 400 / 1024
    assert (matrix[0][2], matrix[1][2]) == (0.0, 0.0)


def test_a_viewport_window_gives_its_panel_the_whole_window(frame):
    """The preview is a viewport, not a form, and wants no padding.

    Padding is read once, by `Begin`, so there is nowhere inside the panel to
    drop it from. This is what says the window opens itself in order to push
    the style in front of that call, and that the push actually lands.
    """
    measured = {}

    def record(key):
        def gui():
            measured[key] = (
                imgui.get_cursor_screen_pos().x - imgui.get_window_pos().x,
                imgui.get_content_region_avail().x - imgui.get_window_size().x,
            )

        return gui

    window._viewport_body("Viewport", record("viewport"))
    # A form window, opened the way hello_imgui opens every other panel.
    imgui.begin("Form")
    record("form")()
    imgui.end()

    assert measured["viewport"] == (0.0, 0.0)
    assert measured["form"][0] > 0.0
    assert measured["form"][1] < 0.0


def test_only_the_preview_opens_itself_and_every_form_keeps_its_padding():
    """The viewport treatment is one window's, and the forms are untouched.

    `call_begin_end` is what hands the `Begin` to us, so a viewport that lost
    it would quietly get its padding back, and a form that gained it would draw
    nothing at all.
    """
    params = window._build_runner_params(PanelRuntime())
    opens_itself = {
        dockable.label: dockable.call_begin_end
        for dockable in params.docking_params.dockable_windows
    }
    assert opens_itself["Preview"] is False
    assert set(opens_itself) == {
        "Controls",
        "Loop",
        "Audio",
        "Mapping",
        "Presets",
        "Preview",
    }
    forms = {label for label, own in opens_itself.items() if own}
    assert forms == {"Controls", "Loop", "Audio", "Mapping", "Presets"}


def test_the_quad_is_drawn_at_the_size_the_mode_asked_for(frame):
    """End to end, from the geometry to the item imgui actually places.

    The quad carries all the scaling now, so its size is the only place the
    modes differ once a frame is on screen, and a quad drawn at the frame's own
    size would be a preview that ignored both of them.
    """
    runtime = PanelRuntime(host=FakeModelHost(current=object()))
    runtime.preview = FramePreview()

    fitted = draw_preview(runtime)
    # An 8 by 8 frame in a panel with room to spare, drawn at 8 by 8. Fit does
    # not magnify past native size, and on this unscaled context native size is
    # the frame's own. So nothing is resampled either.
    assert fitted.quads == [(8.0, 8.0)]
    assert fitted.enlarged == []

    panel = PreviewPanel(runtime)
    panel._mode = DisplayMode.STRETCH
    stretched = draw_preview(runtime, panel=panel)
    ((width, height),) = stretched.quads
    # Stretch is the mode that magnifies, and that is the whole of what it
    # does differently. A square frame stays square: a quad that took the
    # panel's shape would be the distortion this mode used to do.
    assert (width, height) > (8.0, 8.0)
    assert width == height
    # And a magnified frame reaches the quad through the resample, at exactly
    # the size the quad is drawn at. Drawing the 8 by 8 texture over a larger
    # quad instead is the interpolated result this exists to avoid, and it
    # would leave this list empty while every size assertion above still held.
    ((_, source, target),) = stretched.enlarged
    assert source == (8, 8)
    assert target == (int(width), int(height))


def test_the_perform_panel_draws_in_vector_mode_without_raising(frame):
    """The one branch no other test in this file puts the panel through.

    Every other test here draws `PerformPanel` against the default
    `ControlState`, which is seed mode. This is what stands between that and
    an exception in the vector row, which nothing else in the suite can
    reach: there is no headless way to click Randomize, Load or Save, only to
    prove the row they are on draws at all.
    """
    state = ControlState(vector_mode=True, latent_vec=(1.0, 2.0, 3.0))
    panel = PerformPanel(PanelRuntime(state=state), mapping_popup=lambda name: None)
    panel.gui()


def loop_panel_height(panel: LoopPanel) -> float:
    """How far `panel.gui()` moves the cursor down, in one frame.

    A stand in for "how many lines it drew" that needs nothing from imgui
    beyond the cursor it already tracks, the same measurement the preview
    panel's tests already take of a conditional overlay.
    """
    start = imgui.get_cursor_pos().y
    panel.gui()
    return imgui.get_cursor_pos().y - start


def test_the_noise_pending_note_only_draws_when_the_table_is_stale(frame):
    """`_noise_pending_row`, the whole of item 2's surfacing, end to end.

    Two panels, differing only in whether the published table's key matches
    what the state now asks for, and the stale one has to draw more, which is
    the pending note and nothing else changed between them.
    """
    state = ControlState(
        loop_active=True, noise_loop=True, noise_loop_seed=3, noise_radius=2.0
    )
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)

    stale = PanelRuntime(state=state)
    stale.model_info_store = LatestValueStore(info)
    stale.control_loop = FakeControlLoop(noise_table_key=(1, 1.0, 4))

    fresh = PanelRuntime(state=state)
    fresh.model_info_store = LatestValueStore(info)
    fresh.control_loop = FakeControlLoop(noise_table_key=(3, 2.0, 4))

    stale_height = loop_panel_height(LoopPanel(stale, mapping_popup=lambda name: None))
    imgui.new_line()  # separates the two panels on the same cursor column
    fresh_height = loop_panel_height(LoopPanel(fresh, mapping_popup=lambda name: None))
    assert stale_height > fresh_height


def test_the_noise_pending_note_is_silent_while_the_loop_is_stopped(frame):
    """No build was ever requested while stopped, so nothing should claim one.

    The control loop only calls `request_build` under `loop_active and
    noise_loop` (control.py `tick`), so a stale key with the loop stopped is
    not a rebuild in progress, only one that has not been asked for yet. This
    stopped, mismatched panel has to come out the same height as a playing,
    matched one: neither has anything pending to report.
    """
    info = ModelInfo(pkl_path="model.pkl", z_dim=4, num_ws=8)

    stopped = ControlState(
        loop_active=False, noise_loop=True, noise_loop_seed=3, noise_radius=2.0
    )
    stopped_runtime = PanelRuntime(state=stopped)
    stopped_runtime.model_info_store = LatestValueStore(info)
    # Stale on purpose: this is exactly the key mismatch the playing case
    # reads as pending, and here it must not be.
    stopped_runtime.control_loop = FakeControlLoop(noise_table_key=(1, 1.0, 4))

    playing = dataclasses.replace(stopped, loop_active=True)
    playing_runtime = PanelRuntime(state=playing)
    playing_runtime.model_info_store = LatestValueStore(info)
    playing_runtime.control_loop = FakeControlLoop(noise_table_key=(3, 2.0, 4))

    stopped_height = loop_panel_height(
        LoopPanel(stopped_runtime, mapping_popup=lambda name: None)
    )
    imgui.new_line()
    playing_height = loop_panel_height(
        LoopPanel(playing_runtime, mapping_popup=lambda name: None)
    )
    assert stopped_height == pytest.approx(playing_height)
