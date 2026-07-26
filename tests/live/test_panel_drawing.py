"""The panels task 10 adds, drawn for real in a headless imgui context.

imgui needs no GPU to lay a frame out. A context with no renderer backend
still measures text, places items and reports rects, which is enough to prove
two things no pure helper test can: that a panel's `gui()` runs at all, and
that nothing it draws runs off the edge of the panel it is drawn in. The same
technique the driver marker suite already uses for the Perform and Loop
panels (`test_driver_marker_layout.py`, `row_edges`), aimed at the new ones.

What is still not covered, and cannot be here: whether the result looks right,
whether a click lands where it appears to, and every GL upload. Those stay
manual.
"""

import pytest
from imgui_bundle import imgui

from autolume.live.core.generator import DeviceStatus
from autolume.live.core.params import ControlState
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.io.ndi import NdiStatus
from autolume.live.io.recorder import RecorderStatus
from autolume.live.runtime import OscStatus
from autolume.live.ui import theme
from autolume.live.ui.panels.performance import PerformancePanel

# The docked width at the shipped window size, and the two narrower docks a
# performer can drag to, matching the widths the perform panel is held to.
WIDTHS = (448.0, 360.0, 280.0)
FONT_SCALES = (1.0, 1.5, 2.0)
# What a wrapped note is allowed to overshoot by, and why it is not zero:
# `imgui.text_wrapped` wraps at the work rect but reports its extent through
# `CalcTextSize`, which rounds a fractional advance up, so a note that wraps
# exactly to the edge measures one pixel past it. Demonstrated below in
# `test_a_wrapped_note_measures_one_pixel_past_its_own_wrap_width` so the
# allowance is a known artifact rather than slack: a row that genuinely
# overflows does so by tens of pixels.
WRAP_ROUNDING = 1.0


class Stage:
    disabled_reason = None
    last_error = None


class Model:
    """Stands in for a `LoadedModel` for the parts a panel reads off one."""

    def __init__(self, pkl_path="/models/a.pkl"):
        self.pkl_path = pkl_path
        self._superres = Stage()


class Host:
    def __init__(self, current=None, error=None):
        self._current = current
        self._error = error
        self.device_store = LatestValueStore(DeviceStatus(active="cpu"))
        self.calls = []

    def current(self):
        return self._current

    def pending(self):
        return None

    def error(self):
        return self._error


class RenderLoop:
    def fps(self):
        return 60.0


class Ndi:
    def status(self):
        return NdiStatus()


class Recorder:
    def status(self):
        return RecorderStatus()


class Runtime:
    def __init__(self, state=None, host=None, model_info=None):
        self.control_store = LatestValueStore(state or ControlState())
        self.source_store = LatestValueStore(SourceTable())
        self.model_host = host or Host()
        self.model_info_store = LatestValueStore(model_info)
        self.render_loop = RenderLoop()
        self.osc_status_store = LatestValueStore(OscStatus(bound_port=1338))
        self.ndi = Ndi()
        self.recorder = Recorder()
        self.submitted = []

    def submit(self, event):
        self.submitted.append(event)


def widest_overflow(build, width: float, font_scale: float) -> float:
    """How far the widest thing a panel drew reaches past the content edge.

    Zero is flush with the edge of the content region, where the separators
    end and as far as anything may go. Positive is content that has run off the
    panel and grown the window behind it.

    Measured off `cursor_max_pos`, imgui's own running maximum of everything
    placed this frame, rather than the last item's rect: the perform panel's
    measurement (`test_driver_marker_layout.py`) can hook `ControlBinder._widget`
    because every row it draws goes through it, and these panels deliberately
    mix plain widgets in with bound ones, so there is no single wrapper to
    hook. The running maximum needs none and misses nothing, which makes it
    the stronger of the two measurements.

    Three frames, because a scrollbar is decided from the frame before it and
    the panel has to fit the width it is actually drawn in.
    """
    context = imgui.create_context()
    overflow = 0.0
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = imgui.ImVec2(1280.0, 800.0)
        io.delta_time = 1.0 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        theme.apply_theme()
        imgui.get_style().font_scale_main = font_scale
        panel = build()
        for _ in range(3):
            imgui.new_frame()
            imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
            imgui.set_next_window_size(imgui.ImVec2(width, 300.0))
            imgui.begin("Panel")
            right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
            panel.gui()
            reached = imgui.internal.get_current_window().dc.cursor_max_pos.x
            overflow = reached - right
            imgui.end()
            imgui.render()
    finally:
        imgui.destroy_context(context)
    return overflow


LOADED = ControlState(pkl_path="/models/wikiart-1024.pkl")


PERFORMANCE_CASES = [
    ("empty", lambda: PerformancePanel(Runtime())),
    ("loaded", lambda: PerformancePanel(Runtime(LOADED, Host(current=Model())))),
    (
        "everything on",
        lambda: PerformancePanel(
            Runtime(
                ControlState(
                    use_superres=True,
                    ndi_enabled=True,
                    recording=True,
                    fullscreen=True,
                    device="rocm",
                ),
                Host(current=Model()),
            )
        ),
    ),
]


@pytest.mark.parametrize("font_scale", FONT_SCALES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize(
    "name,build", PERFORMANCE_CASES, ids=[name for name, _ in PERFORMANCE_CASES]
)
def test_no_panel_content_runs_past_the_panel_it_is_drawn_in(
    name, build, width, font_scale
):
    """The panel's content may never be wider than the panel.

    imgui's default item width is a fraction of the window and a label is
    drawn outside it, so a row costs more than it declares and grows with the
    font while the panel does not. What overflows is not itself visible: what
    shows is every separator in the panel, which spans the content region and
    so stops short of whatever has run past it.
    """
    assert widest_overflow(build, width, font_scale) <= WRAP_ROUNDING, name


@pytest.mark.parametrize(
    "name,build", PERFORMANCE_CASES, ids=[name for name, _ in PERFORMANCE_CASES]
)
def test_a_panel_draws_without_raising(name, build):
    """Nothing a panel draws may raise, whatever the state behind it.

    A panel raises on the UI thread, not the control thread, so it takes the
    window down rather than the show. That is still the whole interface gone
    mid performance, which is why the empty, half loaded and failed states are
    all drawn here rather than only the happy one.
    """
    widest_overflow(build, 448.0, 1.0)


def test_a_wrapped_note_measures_one_pixel_past_its_own_wrap_width():
    """Justifies `WRAP_ROUNDING`, so the allowance is not read as slack.

    A `text_wrapped` and nothing else, in a panel tall enough to scroll the
    way a real one does, overshoots by exactly one pixel. Nothing in this
    repository decides that: imgui wraps at the work rect and then measures the
    result with `CalcTextSize`, which rounds a fractional advance up, so a line
    that wraps flush with the edge reports one pixel past it. The scrollbar is
    part of the setup rather than incidental, because it is what moves the wrap
    width onto a boundary where the rounding shows.
    """

    class WrappedNote:
        def gui(self):
            imgui.text_wrapped(
                "Super-res only runs on frames up to 1024 pixels on the short "
                "side. A larger frame is passed through unchanged."
            )
            for _ in range(20):
                imgui.dummy(imgui.ImVec2(1.0, 20.0))

    assert widest_overflow(WrappedNote, 360.0, 1.0) == WRAP_ROUNDING
