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

from autolume.live.core.generator import (
    DeviceStatus,
    LayerInfo,
    MixSaveStatus,
    ModelInfo,
)
from autolume.live.core.params import BY_ADDRESS, ControlState, Transform
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.io.ndi import NdiStatus
from autolume.live.io.recorder import RecorderStatus
from autolume.live.runtime import OscStatus
from autolume.live.ui import theme
from autolume.live.ui.panels.bending import BendingPanel
from autolume.live.ui.panels.mixing import MixingPanel
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


class Network:
    def __init__(self, names):
        self._names = names

    def named_parameters(self):
        return [(name, None) for name in self._names]


class Model:
    """Stands in for a `LoadedModel` for the parts a panel reads off one.

    `G` carries the parameter names `mixing.conv_names` walks, so the mixing
    panel's row derivation runs against a real name list rather than a mocked
    out one.
    """

    def __init__(self, pkl_path="/models/a.pkl", names=()):
        self.pkl_path = pkl_path
        self.G = Network(names)
        self._superres = Stage()


def block_names(resolutions, per_block=2):
    """Parameter names shaped like a StyleGAN generator's, plus the mapping."""
    names = ["mapping.fc0.weight"]
    for resolution in resolutions:
        for index in range(per_block):
            names.append(f"synthesis.b{resolution}.conv{index}.weight")
        names.append(f"synthesis.b{resolution}.torgb.weight")
    return tuple(names)


class Host:
    def __init__(self, current=None, current_b=None, error=None, mixing=False):
        self._current = current
        self._current_b = current_b
        self._error = error
        self._mixing = mixing
        self.device_store = LatestValueStore(DeviceStatus(active="cpu"))
        self.mix_save_store = LatestValueStore(MixSaveStatus())
        self.calls = []

    def current(self):
        return self._current

    def current_b(self):
        return self._current_b

    def pending(self):
        return None

    def pending_b(self):
        return None

    def error(self):
        return self._error

    def mixing_enabled(self):
        return self._mixing

    def request_load_b(self, path):
        self.calls.append(("load_b", path))

    def request_mix(self, entries):
        self.calls.append(("mix", tuple(entries)))

    def set_mixing_enabled(self, enabled):
        self._mixing = bool(enabled)
        self.calls.append(("enable", enabled))

    def request_save_mix(self, name):
        self.calls.append(("save", name))


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

CATALOG = ModelInfo(
    pkl_path="/models/wikiart-1024.pkl",
    z_dim=512,
    num_ws=18,
    layers=(
        LayerInfo("b4.conv1", 512, 4, 4),
        LayerInfo("b4.torgb", 3, 4, 4),
        LayerInfo("b8.conv0", 512, 8, 8),
        LayerInfo("b8.conv1", 512, 8, 8),
        LayerInfo("b8.torgb", 3, 8, 8),
        LayerInfo("b16.conv0", 256, 16, 16),
        LayerInfo("b16.torgb", 3, 16, 16),
        LayerInfo("output", 3, 16, 16),
    ),
)

# The worst case the bending panel can be asked to draw: a chain on the layer it
# opens on, a torgb layer's greyed noise rows, directions loaded so the adjuster
# is live, and a capture layer the catalog does not contain.
BENT = ControlState(
    pkl_path="/models/wikiart-1024.pkl",
    transforms=(
        Transform("translate", "b4.conv1", (0.5, -0.5), (0, 1, 2)),
        Transform("erode", "b4.conv1", (3.0,), (0,)),
        Transform("ablate", "b16.conv0", (1.0,), tuple(range(64))),
    ),
    layer_noise=(("b4.conv1", 0.4),),
    layer_ratios=(("b4.conv1", 2.0, 0.5),),
    directions=((0.1,) * 512, (0.2,) * 512),
    adjust_w1=1.5,
    capture_layer="b256.conv0",
    base_channel=64,
)


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

BENDING_CASES = [
    ("bending empty", lambda: BendingPanel(Runtime())),
    (
        "bending catalog",
        lambda: BendingPanel(Runtime(LOADED, Host(current=Model()), CATALOG)),
    ),
    (
        "bending bent",
        lambda: BendingPanel(Runtime(BENT, Host(current=Model()), CATALOG)),
    ),
]

# Raw parameter names, mapping network included, which is what a real `G`
# reports. `conv_names` drops the mapping entries, so a selection is one shorter
# than these are.
PARAMS_A = block_names((4, 8, 16))
PARAMS_B = block_names((4, 8, 16, 32, 64, 128, 256, 512, 1024))

PAIRED = ControlState(pkl_path="/models/a.pkl", pkl2="/models/b.pkl")

MIXING_CASES = [
    ("mixing empty", lambda: MixingPanel(Runtime())),
    (
        "mixing one slot",
        lambda: MixingPanel(
            Runtime(LOADED, Host(current=Model(names=PARAMS_A)))
        ),
    ),
    (
        # A deep second model, so the widest resolution label the panel can draw
        # is on screen and every row it produces is measured.
        "mixing both slots",
        lambda: MixingPanel(
            Runtime(
                PAIRED,
                Host(
                    current=Model("/models/a.pkl", PARAMS_A),
                    current_b=Model("/models/b.pkl", PARAMS_B),
                ),
            )
        ),
    ),
    (
        "mixing failed",
        lambda: MixingPanel(
            Runtime(
                PAIRED,
                Host(
                    current=Model("/models/a.pkl", PARAMS_A),
                    current_b=Model("/models/b.pkl", PARAMS_B),
                    error=(
                        "These models are incompatible. Compressed models "
                        "generally can not be used for mixing."
                    ),
                ),
            )
        ),
    ),
]

CASES = PERFORMANCE_CASES + BENDING_CASES + MIXING_CASES


@pytest.mark.parametrize("font_scale", FONT_SCALES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("name,build", CASES, ids=[name for name, _ in CASES])
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


@pytest.mark.parametrize("name,build", CASES, ids=[name for name, _ in CASES])
def test_a_panel_draws_without_raising(name, build):
    """Nothing a panel draws may raise, whatever the state behind it.

    A panel raises on the UI thread, not the control thread, so it takes the
    window down rather than the show. That is still the whole interface gone
    mid performance, which is why the empty, half loaded and failed states are
    all drawn here rather than only the happy one.
    """
    widest_overflow(build, 448.0, 1.0)


def test_a_pair_with_no_selection_yet_is_given_the_default_and_asks_for_it():
    """The one thing the mixing panel does without being clicked.

    A pair whose selection does not fit them cannot be drawn against, so the
    panel adopts the default. Adopting it has to reach both the state and the
    host, or a performer's first click lands on a selection the host has never
    heard of. It also has to happen once rather than every frame, since
    `request_mix` builds a whole generator on the loader thread.
    """
    host = Host(
        current=Model("/models/a.pkl", PARAMS_A),
        current_b=Model("/models/b.pkl", PARAMS_B),
    )
    runtime = Runtime(PAIRED, host)
    widest_overflow(lambda: MixingPanel(runtime), 448.0, 1.0)
    requests = [entries for name, entries in host.calls if name == "mix"]
    assert len(requests) == 1
    assert set(requests[0]) == {"A", "B"}
    # One entry per synthesis parameter: the mapping network is not part of a
    # selection, so the single mapping entry in each fake is not counted.
    assert len(requests[0]) == max(len(PARAMS_A), len(PARAMS_B)) - 1
    addresses = [event.address for event in runtime.submitted]
    assert addresses.count("/mix/layers") == 1


def test_a_second_model_path_is_turned_into_a_load_exactly_once():
    """Nothing else does this.

    `pkl_path` has a watcher on the control thread that calls `request_load`;
    `pkl2` has none, so slot B is only ever loaded because this panel asks. Once
    per change, because the loader coalesces but a request per frame would still
    be a request per frame.
    """
    host = Host()
    runtime = Runtime(ControlState(pkl2="/models/b.pkl"), host)
    widest_overflow(lambda: MixingPanel(runtime), 448.0, 1.0)
    assert [call for call in host.calls if call[0] == "load_b"] == [
        ("load_b", "/models/b.pkl")
    ]


def test_no_second_model_asks_for_no_load():
    host = Host()
    widest_overflow(lambda: MixingPanel(Runtime(LOADED, host)), 448.0, 1.0)
    assert not [call for call in host.calls if call[0] == "load_b"]


def test_every_address_these_panels_write_by_hand_is_a_real_one():
    """The three addresses the new panels emit as literals.

    Every other control goes through `ControlBinder`, which looks its address up
    in the registry and cannot get one wrong. These three sit beside a combo or
    a file dialog, which `ControlBinder` has no version of, so the panel writes
    the address itself. A typo there would submit an event the control loop logs
    and drops, and nothing else would say so.
    """
    for address in ("/render/device", "/mix/model", "/image/layer"):
        assert address in BY_ADDRESS, address


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
