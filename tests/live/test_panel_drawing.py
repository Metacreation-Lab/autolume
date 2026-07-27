"""The panels task 10 adds, drawn for real in a headless imgui context.

imgui needs no GPU to lay a frame out. A context with no renderer backend
still measures text, places items and reports rects, which is enough to prove
two things no pure helper test can: that a panel's `gui()` runs at all, and
that nothing it draws runs off the edge of the panel it is drawn in. The same
technique the driver marker suite already uses for the Perform and Loop
panels (`test_driver_marker_layout.py`, `row_edges`), aimed at the new ones.

A press is real too. `click_button` puts a mouse position and a button event
on `io`, and imgui hit tests them against the rect it placed the button at, so
what runs is the panel's own click path. That is not a nicety: two bugs in
these panels fire only on a press, and no amount of passive rendering reaches
either.

What is still not covered, and cannot be here: whether the result looks right,
and every GL upload. Those stay manual.
"""

import dataclasses

import pytest
from imgui_bundle import imgui

from autolume.live.core.generator import (
    DeviceStatus,
    LayerInfo,
    MixSaveStatus,
    ModelInfo,
)
from autolume.live.core.mapping import apply_event
from autolume.live.core.mixing import conv_names
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
    def __init__(
        self, current=None, current_b=None, error=None, mixing=False, mixed=None
    ):
        self._current = current
        self._current_b = current_b
        self._error = error
        self._mixing = mixing
        self._mixed = mixed
        self.device_store = LatestValueStore(DeviceStatus(active="cpu"))
        self.mix_save_store = LatestValueStore(MixSaveStatus())
        self.calls = []

    def current(self):
        # The mix while one is on, exactly like the real host, so a panel that
        # reads this where it wants model A is caught here rather than by hand.
        if self._mixing and self._mixed is not None:
            return self._mixed
        return self._current

    def current_a(self):
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
    """The parts of a runtime the panels read, and nothing behind them.

    `live` runs every submitted event through the **real** `mapping.apply_event`
    and puts the result back on the store, which is what the control thread does
    a tick later in the app. Without it the store never moves, so a panel that
    edits state and reads it back on the next frame is drawn against a snapshot
    that has forgotten the edit, and a click test would be measuring the fake
    rather than the panel. Off by default, because the drawing tests deliberately
    hold one fixed state, some of it deliberately malformed.
    """

    def __init__(self, state=None, host=None, model_info=None, live=False):
        self.control_store = LatestValueStore(state or ControlState())
        self.source_store = LatestValueStore(SourceTable())
        self.model_host = host or Host()
        self.model_info_store = LatestValueStore(model_info)
        self.render_loop = RenderLoop()
        self.osc_status_store = LatestValueStore(OscStatus(bound_port=1338))
        self.ndi = Ndi()
        self.recorder = Recorder()
        self.submitted = []
        self._live = live

    def submit(self, event):
        self.submitted.append(event)
        if self._live:
            self.control_store.set(
                apply_event(self.control_store.snapshot(), event)
            )


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


def click_button(build, label: str, *, occurrence: int = 0, width: float = 448.0):
    """Draw a panel and press a real button in it with a real mouse.

    imgui decides the press, not the test. The mouse position and the button
    events go through `io`, imgui hit tests them against the rect it placed the
    button at, and `imgui.button` returns True on its own. `imgui.button` is
    wrapped only to *record* where each one landed, never to change what it
    returns, so what runs is the panel's real click path.

    This exists because the passive draw tests cannot reach it. A panel bug that
    only fires on a press (a list mutated mid-iteration, a cache never seeded)
    is invisible to every test that only renders, and both kinds have already
    happened here.

    The press and the release are separate frames because a button fires on
    release over itself, and the panel keeps being drawn afterwards so the frame
    that has to survive the click's own consequences is drawn too.
    """
    context = imgui.create_context()
    original = imgui.button
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = imgui.ImVec2(1280.0, 800.0)
        io.delta_time = 1.0 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        theme.apply_theme()
        seen: list[tuple[float, float]] = []
        target: list[tuple[float, float] | None] = [None]

        def recording_button(text, *args, **kwargs):
            pressed = original(text, *args, **kwargs)
            if text == label:
                low = imgui.get_item_rect_min()
                high = imgui.get_item_rect_max()
                seen.append(((low.x + high.x) * 0.5, (low.y + high.y) * 0.5))
            return pressed

        imgui.button = recording_button
        panel = build()
        for frame in range(8):
            if target[0] is not None and frame in (4, 5):
                io.add_mouse_pos_event(*target[0])
                io.add_mouse_button_event(0, frame == 4)
            imgui.new_frame()
            imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
            imgui.set_next_window_size(imgui.ImVec2(width, 900.0))
            imgui.begin("Panel")
            seen.clear()
            panel.gui()
            imgui.end()
            imgui.render()
            if frame == 2:
                if occurrence >= len(seen):
                    raise AssertionError(
                        f"no button {label!r} at occurrence {occurrence}"
                    )
                target[0] = seen[occurrence]
    finally:
        imgui.button = original
        imgui.destroy_context(context)


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


def test_a_pair_with_no_selection_yet_is_given_the_default_once():
    """The one thing the mixing panel does without being clicked.

    A pair whose selection does not fit them cannot be drawn against, so the
    panel adopts the default. It writes that to state and nothing else: the host
    is driven from `_watch_mixing` on the control thread, so a selection reaches
    the loader whether or not this tab is the one showing. Once rather than every
    frame, because the store only catches up a tick later and each submission
    would cost another whole-generator build.
    """
    host = Host(
        current=Model("/models/a.pkl", PARAMS_A),
        current_b=Model("/models/b.pkl", PARAMS_B),
    )
    runtime = Runtime(PAIRED, host)
    widest_overflow(lambda: MixingPanel(runtime), 448.0, 1.0)
    sent = [
        event.value.entries
        for event in runtime.submitted
        if event.address == "/mix/layers"
    ]
    assert len(sent) == 1
    assert set(sent[0]) == {"A", "B"}
    # One entry per synthesis parameter: the mapping network is not part of a
    # selection, so the single mapping entry in each fake is not counted.
    assert len(sent[0]) == max(len(PARAMS_A), len(PARAMS_B)) - 1


def test_the_panel_never_reaches_past_state_into_the_host():
    """The whole of the tab-gating fix, as one assertion.

    Every host call the panel used to make (`request_load_b`, `request_mix`,
    `set_mixing_enabled`) is now the control loop's, because a dockable window's
    gui function does not run while its tab is hidden and a host driven from
    here left `/mix/enabled`, `/mix/model` and a restored preset waiting on a
    click. The panel may still *read* the host, and does.
    """
    host = Host(
        current=Model("/models/a.pkl", PARAMS_A),
        current_b=Model("/models/b.pkl", PARAMS_B),
    )
    state = dataclasses.replace(
        PAIRED, mixing_enabled=True, combined_layers=("A",) * 9 + ("B",) * 18
    )
    widest_overflow(lambda: MixingPanel(Runtime(state, host)), 448.0, 1.0)
    assert host.calls == []


def test_a_restored_preset_still_reaches_the_host():
    """Important 1, from the panel's side.

    A preset saved from this same pair restores a selection that already fits,
    so the panel's own "adopt the default" branch never runs and it submits
    nothing at all. That used to mean no build was ever queued. It is correct
    now precisely *because* the panel is not the thing that queues one: the
    state it was restored into is, through `_watch_mixing`, which
    `test_runtime.py` covers.
    """
    host = Host(
        current=Model("/models/a.pkl", PARAMS_A),
        current_b=Model("/models/b.pkl", PARAMS_B),
    )
    state = dataclasses.replace(
        PAIRED, mixing_enabled=True, combined_layers=("A",) * 9 + ("B",) * 18
    )
    runtime = Runtime(state, host)
    widest_overflow(lambda: MixingPanel(runtime), 448.0, 1.0)
    assert [event for event in runtime.submitted if event.address == "/mix/layers"] == []


def test_the_rows_come_from_model_a_even_while_a_mix_is_rendering():
    """Important 3.

    `current()` returns the mix once one is built, and its layer names are the
    ones the selection *produced*, not the ones it applies to. Gating the read
    on `mixing_enabled()` instead does not work: retiring a mix leaves that flag
    set. A mix here is deliberately given a name list of a different length, so
    reading the wrong slot changes the row count visibly.
    """
    truncated = Model("/models/a.pkl", block_names((4, 8)))
    host = Host(
        current=Model("/models/a.pkl", PARAMS_A),
        current_b=Model("/models/b.pkl", PARAMS_B),
        mixing=True,
        mixed=truncated,
    )
    panel = MixingPanel(Runtime(PAIRED, host))
    widest_overflow(lambda: panel, 448.0, 1.0)
    names_a, _ = panel._names()
    assert len(names_a) == len(conv_names(Network(PARAMS_A)))




def hostile_states(count: int, seed: int = 0):
    """States a preset or an OSC message could genuinely put a panel in.

    Not arbitrary noise: every field is filled with something the control loop
    will actually accept and hold. Unknown operators and layer names, indices
    past a tensor, direction vectors of mismatched lengths, a `combined_layers`
    of the wrong length carrying an origin that is not A, B or X, a capture layer
    from another model, a negative base channel. A preset written on a different
    machine, against a different model, produces most of these.
    """
    random = __import__("random").Random(seed)
    ops = [
        "translate", "rotate", "scale", "erode", "dilate", "invert",
        "flip-h", "flip-v", "binary-thresh", "scalar-multiply", "ablate",
        "sobel", "not-an-op", "",
    ]
    layers = ["b4.conv1", "b8.conv1", "output", "b999.conv0", "", "weird.name"]
    for _ in range(count):
        yield ControlState(
            pkl_path=random.choice(["", "/m/a.pkl"]),
            pkl2=random.choice([None, "", "/m/b.pkl"]),
            transforms=tuple(
                Transform(
                    random.choice(ops),
                    random.choice(layers),
                    tuple(random.uniform(-5, 5) for _ in range(random.randint(0, 3))),
                    tuple(
                        random.randint(-5, 2000) for _ in range(random.randint(0, 5))
                    ),
                )
                for _ in range(random.randint(0, 4))
            ),
            layer_noise=tuple(
                (random.choice(layers), random.uniform(-2, 2))
                for _ in range(random.randint(0, 3))
            ),
            layer_ratios=tuple(
                (random.choice(layers), random.uniform(-2, 4), random.uniform(-2, 4))
                for _ in range(random.randint(0, 3))
            ),
            directions=tuple(
                tuple(random.uniform(-1, 1) for _ in range(random.choice([0, 4, 512])))
                for _ in range(random.randint(0, 10))
            ),
            combined_layers=tuple(
                random.choice(["A", "B", "X", "Q"])
                for _ in range(random.randint(0, 20))
            ),
            capture_layer=random.choice(["", "output", "b999.conv0"]),
            base_channel=random.randint(-10, 9999),
            device=random.choice(["auto", "cuda", "rocm", ""]),
            mixing_enabled=random.choice([True, False]),
            use_superres=random.choice([True, False]),
            osc_port=random.randint(-5, 99999),
        )


HOSTS = [
    lambda: Host(),
    lambda: Host(current=Model("/m/a.pkl", PARAMS_A)),
    lambda: Host(
        current=Model("/m/a.pkl", PARAMS_A), current_b=Model("/m/b.pkl", PARAMS_B)
    ),
    # Parameter names with no resolution in them at all, which `layer_resolution`
    # refuses. The panel has to survive a generator it cannot group.
    lambda: Host(
        current=Model("/m/a.pkl", ("bad",)), current_b=Model("/m/b.pkl", ("also.bad",))
    ),
]

CATALOGS = [
    None,
    CATALOG,
    ModelInfo("/m/a.pkl", 0, 0, ()),
    ModelInfo("/m/a.pkl", 512, 18, (LayerInfo("", 0, 0, 0),)),
]


@pytest.mark.parametrize("state", list(hostile_states(8)))
def test_no_state_a_preset_can_carry_makes_a_panel_raise(state):
    """The UI thread, unlike the control thread, has no error channel.

    A panel that raises takes the whole window down mid performance, and the
    states below are the ones that reach a panel without ever being clicked into
    it: a preset written against another model, an OSC message, a mix whose
    sources changed underneath it. Every combination of state, host and catalog
    is drawn for three frames.
    """
    for make_host in HOSTS:
        for catalog in CATALOGS:
            for panel in (BendingPanel, MixingPanel, PerformancePanel):
                runtime = Runtime(state, make_host(), catalog)
                widest_overflow(lambda: panel(runtime), 360.0, 1.0)


CHAINED = ControlState(
    pkl_path="/models/wikiart-1024.pkl",
    transforms=(
        Transform("ablate", "b4.conv1", (1.0,), (0,)),
        Transform("invert", "b4.conv1", (1.0,), (1,)),
        Transform("rotate", "b4.conv1", (30.0,), (2,)),
    ),
)


def test_removing_a_transform_that_is_not_the_last_one_does_not_raise():
    """The bug this harness was built for.

    Remove takes that row's selection editor out of the panel's own list
    straight away, while `state.transforms` is still the snapshot the frame was
    drawn from. Every later row would then index a list one shorter than the
    indices it was drawn with, which is an `IndexError` on the UI thread and the
    whole window gone. Removing the *last* transform never hits it, which is why
    the passive tests and a casual click both miss it.
    """
    runtime = Runtime(CHAINED, Host(), CATALOG)
    click_button(lambda: BendingPanel(runtime), "Remove", occurrence=0)
    removals = [
        event.value.index
        for event in runtime.submitted
        if event.address == "/bend/remove"
    ]
    assert removals == [0]


def test_removing_the_middle_transform_of_a_chain_does_not_raise():
    runtime = Runtime(CHAINED, Host(), CATALOG)
    click_button(lambda: BendingPanel(runtime), "Remove", occurrence=1)
    removals = [
        event.value.index
        for event in runtime.submitted
        if event.address == "/bend/remove"
    ]
    assert removals == [1]


def test_adding_a_transform_sends_one_that_the_control_thread_accepts():
    runtime = Runtime(LOADED, Host(), CATALOG)
    click_button(lambda: BendingPanel(runtime), "Add transform")
    sets = [event for event in runtime.submitted if event.address == "/bend/set"]
    assert len(sets) == 1
    applied = apply_event(LOADED, sets[0])
    assert len(applied.transforms) == 1
    assert applied.transforms[0].layer == "b4.conv1"


def test_the_first_cut_of_a_session_can_be_recovered():
    """The other bug this harness was built for.

    The panel's cache starts empty and is only ever written by a cut, so before
    this was fixed the very first cut had nothing to hold on to and Recover put
    nothing back. Reachable whenever a preset's selection already fits the pair,
    which is every preset saved and reloaded on the same machine.
    """
    host = Host(
        current=Model("/models/a.pkl", PARAMS_A),
        current_b=Model("/models/b.pkl", PARAMS_B),
    )
    # A selection that already fits the pair, which is what a preset saved and
    # reloaded on the same machine carries. That is the case that reaches the
    # bug: the panel only ever seeds its cache in the branch that replaces a
    # selection which does *not* fit, so a fitting one leaves the cache empty.
    runtime = Runtime(
        ControlState(
            pkl_path="/models/a.pkl",
            pkl2="/models/b.pkl",
            combined_layers=("A",) * 9 + ("B",) * 18,
        ),
        host,
        live=True,
    )
    panel = MixingPanel(runtime)

    def selections():
        return [
            event.value.entries
            for event in runtime.submitted
            if event.address == "/mix/layers"
        ]

    # Through the panel's own cache rather than the pure helpers, which is the
    # whole point: the helpers were always handed a seeded cache by their tests,
    # and the panel's starts empty.
    click_button(lambda: panel, "X", occurrence=1)
    after_cut = selections()[-1]
    assert "X" in after_cut
    cut_at = after_cut.index("X")
    click_button(lambda: panel, "Recover", occurrence=0)
    after_recover = selections()[-1]
    assert after_recover != after_cut
    assert after_recover[:cut_at] == after_cut[:cut_at]
    # The resolution that was cut is back, and only that one.
    assert "X" not in after_recover[cut_at : cut_at + 3]
    assert after_recover[-1] == "X"


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
