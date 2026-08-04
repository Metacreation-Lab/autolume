"""Headless draw test for the diffusion panel.

Renders the widget in a real imgui context with no backend, which catches
pyimgui API misuse (wrong return-value unpacking, unbalanced begin/end,
missing pops) that unit tests on the params alone cannot see.
"""
import os

import pytest

imgui = pytest.importorskip("imgui")

import dnnlib
from widgets.diffusion_widget import DiffusionWidget


class FakeDispatcher:
    def streaming_addresses(self):
        return []

    def map_val_to_func(self, *args, **kwargs):
        pass

    def map(self, *args, **kwargs):
        pass

    def unmap(self, *args, **kwargs):
        pass


def make_viz():
    app = dnnlib.EasyDict(button_w=80.0, spacing=4.0, label_w=100.0, font_size=14.0,
                          content_width=1200, content_height=800)
    viz = dnnlib.EasyDict(app=app, pane_w=1000.0, result=dnnlib.EasyDict(),
                          args=dnnlib.EasyDict(), osc_dispatcher=FakeDispatcher())
    viz.cleared = []
    viz.clear_result = lambda: viz.cleared.append(True)
    return viz


@pytest.fixture
def imgui_frame():
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1200, 800
    io.fonts.get_tex_data_as_rgba32()
    io.fonts.texture_id = 1  # a backend would own this
    yield
    imgui.destroy_context(context)


def draw(widget, viz):
    """Draw one frame and return how far down the panel pushed the cursor.

    Vertex counts are always zero without a render backend, so layout height is
    the signal that rows actually drew: a section whose body is skipped leaves
    the cursor nearly where it started.
    """
    imgui.new_frame()
    # size the window like the real pane, or every width computed from the
    # available content region is measured against a window nobody will see
    imgui.set_next_window_size(viz.pane_w, 800)
    imgui.begin("test")
    start = imgui.get_cursor_pos_y()
    height = 0.0
    try:
        widget(show=True)
        height = imgui.get_cursor_pos_y() - start
    finally:
        # an unbalanced begin() makes imgui abort natively when the context is
        # destroyed, which would hide the real error behind a crashed process
        imgui.end()
        imgui.render()
    return height


@pytest.fixture
def widget_factory(monkeypatch, tmp_path):
    from utils import session_state, model_dir
    monkeypatch.setattr(session_state, "cache_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(session_state, "_state", None)
    # the panel picks its starting checkpoint from this folder, so tests must
    # not read whatever the developer happens to have installed
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setattr(model_dir, "diffusion_checkpoints_dir", lambda: str(checkpoints))
    return DiffusionWidget


def indicators(widget, status="", loaded=None, loading=False):
    return {name: (color, tip)
            for name, color, tip in widget.status_indicators(status, loaded, loading)}


def test_panel_draws_every_control(imgui_frame, widget_factory):
    viz = make_viz()
    widget = widget_factory(viz)
    height = draw(widget, viz)
    # enable + status dots + message, prompt, strength/seed, separator, checkpoint,
    # resolution, lora, weight, tensorrt and the osc child: a skipped section is far shorter
    assert height > 150
    assert viz.args.use_diffusion is False
    assert "model" in viz.args.diffusion


def test_panel_draws_while_enabled_and_erroring(imgui_frame, widget_factory):
    viz = make_viz()
    widget = widget_factory(viz)
    widget.unavailable = ''
    widget.enabled = True
    widget.use_lora = True
    widget.lora_path = "x.safetensors"
    viz.result.diffusion_status = "Error: something went wrong"
    draw(widget, viz)
    assert viz.args.use_diffusion is True
    assert viz.args.diffusion["lora_path"] == "x.safetensors"


def test_every_control_row_shares_one_column(imgui_frame, widget_factory):
    viz = make_viz()
    widget = widget_factory(viz)
    widget.unavailable = ''
    widget.enabled = True
    widget.use_lora = True
    columns = []
    move = widget.control_column

    def spy(v):
        move(v)
        columns.append(round(imgui.get_cursor_pos_x(), 1))

    widget.control_column = spy
    draw(widget, viz)
    # a wide label (TensorRT, Checkpoint) used to push its control past the column
    # and overlap the label text; every row must land on the same x
    assert len(columns) >= 6
    assert len(set(columns)) == 1, f"rows misaligned: {sorted(set(columns))}"
    assert columns[0] > viz.app.label_w  # clears the label and its help icon


def test_status_is_gray_while_the_module_is_off(imgui_frame, widget_factory):
    from widgets.diffusion_widget import GRAY
    widget = widget_factory(make_viz())
    widget.unavailable = ''
    dots = indicators(widget)
    assert [color for color, _tip in dots.values()] == [GRAY, GRAY, GRAY]


def test_checkpoint_and_lora_only_go_green_once_actually_live(imgui_frame, widget_factory):
    from widgets.diffusion_widget import AMBER, GREEN
    widget = widget_factory(make_viz())
    widget.unavailable = ''
    widget.enabled = True
    widget.params.model = "stabilityai/sd-turbo"
    widget.use_lora = True
    widget.lora_path = "age.safetensors"
    widget.lora_scale = 3.0

    # selected but nothing loaded yet
    dots = indicators(widget, loading=True)
    assert dots["Checkpoint"][0] is AMBER
    assert dots["LoRA"][0] is AMBER

    live = dict(model="stabilityai/sd-turbo", lora_path="age.safetensors",
                lora_scale=3.0, acceleration="none")
    dots = indicators(widget, loaded=live)
    assert dots["Checkpoint"][0] is GREEN
    assert dots["LoRA"][0] is GREEN
    assert "3" in dots["LoRA"][1]

    # a weight change reloads the pipeline, so the live one no longer matches
    widget.lora_scale = 4.0
    assert indicators(widget, loaded=live)["LoRA"][0] is AMBER


def test_tensorrt_reports_built_versus_running(imgui_frame, widget_factory, monkeypatch):
    from widgets import diffusion_widget
    from widgets.diffusion_widget import AMBER, GRAY, GREEN, RED
    widget = widget_factory(make_viz())
    widget.unavailable = ''
    widget.enabled = True
    assert indicators(widget)["TensorRT"][0] is GRAY

    widget.params.acceleration = "tensorrt"
    monkeypatch.setattr(diffusion_widget.trt, "engines_ready", lambda params: False)
    widget._ready_key = None
    # the stage runs unaccelerated meanwhile, so the label has to say why
    assert indicators(widget)["TensorRT (unbuilt)"][0] is RED

    monkeypatch.setattr(diffusion_widget.trt, "engines_ready", lambda params: True)
    widget._ready_key = None
    # built is not the same as loaded: the pipeline swap takes seconds
    assert indicators(widget)["TensorRT"][0] is AMBER
    live = dict(model=widget.params.model, lora_path="", lora_scale=1.0,
                acceleration="tensorrt")
    assert indicators(widget, loaded=live)["TensorRT"][0] is GREEN


def test_status_line_only_reports_loading_and_ready(imgui_frame, widget_factory, monkeypatch):
    from widgets import diffusion_widget
    from widgets.diffusion_widget import GREEN
    widget = widget_factory(make_viz())
    widget.unavailable = ''
    widget.enabled = True
    widget.params.model = "some/checkpoint"
    assert widget.status_line("Loading pipeline (4 s)")[0] == "Loading pipeline (4 s)"
    # unbuilt engines are the TensorRT indicator's business, not this line's
    widget.params.acceleration = "tensorrt"
    monkeypatch.setattr(diffusion_widget.trt, "engines_ready", lambda params: False)
    widget._ready_key = None
    assert widget.status_line("") == ("Ready", GREEN)


def test_the_status_line_names_why_the_stage_cannot_run(imgui_frame, widget_factory):
    """A broken install and a machine with no card are different problems, and
    only one of them is worth acting on."""
    from diffusion import engine as diffusion_engine
    from widgets.diffusion_widget import GRAY, RED
    widget = widget_factory(make_viz())

    widget.unavailable = diffusion_engine.NOT_INSTALLED
    assert widget.status_line("") == (diffusion_engine.NOT_INSTALLED, RED)

    widget.unavailable = diffusion_engine.NO_GPU
    assert widget.status_line("") == (diffusion_engine.NO_GPU, GRAY)


def test_the_panel_stays_off_while_the_stage_cannot_run(imgui_frame, widget_factory):
    from diffusion import engine as diffusion_engine
    viz = make_viz()
    widget = widget_factory(viz)
    widget.unavailable = diffusion_engine.NO_GPU
    widget.enabled = True
    draw(widget, viz)
    assert viz.args.use_diffusion is False


def test_a_finished_build_wakes_the_render_worker(imgui_frame, widget_factory):
    """A build changes nothing in viz.args, and the worker only renders on an
    args change, so without this the new engines sit unused until restart."""
    import queue
    viz = make_viz()
    widget = widget_factory(viz)
    widget.build_state = 'building'
    widget.build_reply = queue.Queue()
    widget.build_reply.put({'progress': 'Compiling unet (2 of 3)'})
    widget.poll_build()
    assert widget.build_message == 'Compiling unet (2 of 3)'
    assert viz.cleared == []  # nothing to pick up yet
    widget.build_reply = queue.Queue()
    widget.build_reply.put({'done': True})
    widget.poll_build()
    assert widget.build_state == 'idle'
    assert viz.cleared == [True]


def test_a_failed_build_does_not_wake_the_render_worker(imgui_frame, widget_factory):
    import queue
    viz = make_viz()
    widget = widget_factory(viz)
    widget.build_state = 'building'
    widget.build_reply = queue.Queue()
    widget.build_reply.put({'error': 'RuntimeError: out of memory'})
    widget.poll_build()
    assert widget.build_state == 'error'
    assert viz.cleared == []  # there is nothing new to load


def test_strength_and_smoothing_share_the_row_evenly_with_a_small_seed(
        imgui_frame, widget_factory, monkeypatch):
    viz = make_viz()
    viz.pane_w = 720.0  # the real pane is font_size * 45, not the test default
    widget = widget_factory(viz)
    sizes = {}
    real_slider, real_input = imgui.slider_float, imgui.input_int

    def record(call):
        def wrapper(label, *args, **kwargs):
            result = call(label, *args, **kwargs)
            sizes[label] = imgui.get_item_rect_size()[0]
            return result
        return wrapper

    monkeypatch.setattr(imgui, "slider_float", record(real_slider))
    monkeypatch.setattr(imgui, "input_int", record(real_input))
    draw(widget, viz)
    strength = sizes['##diffusion_strength']
    smoothing = sizes['##diffusion_smoothing']
    seed = sizes['##diffusion_seed']
    assert abs(strength - smoothing) < 1.0, (strength, smoothing)
    assert seed < strength * 0.7, (seed, strength)  # a number, not a range
    assert strength > 60, strength  # still usable, not squeezed to nothing


def test_an_osc_preset_saved_before_a_control_existed_still_loads(imgui_frame, widget_factory):
    """Adding an OSC control must not break presets that predate it: the menu
    draw reads every key in funcs directly and would raise KeyError."""
    viz = make_viz()
    widget = widget_factory(viz)
    old_style = ({'Prompt': True, 'Strength': True, 'Seed': True},
                 {'Prompt': False, 'Strength': True, 'Seed': False},
                 {'Prompt': '...', 'Strength': 'str', 'Seed': '...'},
                 {'Prompt': '...', 'Strength': '...', 'Seed': '...'},
                 {'Prompt': 'x', 'Strength': 'x', 'Seed': 'x'})
    widget.osc_menu.set_params(old_style)
    for key in widget.osc_menu.funcs:
        assert key in widget.osc_menu.use_osc, key
        assert key in widget.osc_menu.osc_addresses, key
        assert key in widget.osc_menu.mappings, key
    assert widget.osc_menu.use_osc['Strength'] is True  # saved values still win
    draw(widget, viz)  # the panel must render without a KeyError


def test_header_help_covers_every_control():
    """One (?) in the header, like every other module: it has to explain the lot."""
    from widgets.help_icon_widget import HelpIconWidget
    texts, _urls = HelpIconWidget().load_help_texts("visualizer")
    help_text = texts["diffusion"]
    for label in ("Enable", "Status", "Prompt", "Strength", "Seed", "Smoothing",
                  "Checkpoint", "Resolution", "LoRA", "Weight", "TensorRT"):
        assert f"{label}:" in help_text, f"header help does not mention {label}"


def test_lora_checkbox_off_clears_the_path_without_losing_it(imgui_frame, monkeypatch, tmp_path):
    from utils import session_state
    monkeypatch.setattr(session_state, "cache_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(session_state, "_state", None)
    viz = make_viz()
    widget = DiffusionWidget(viz)
    widget.lora_path = "keep/me.safetensors"
    widget.lora_scale = 3.5
    widget.use_lora = False
    draw(widget, viz)
    assert viz.args.diffusion["lora_path"] == ""
    assert widget.lora_path == "keep/me.safetensors"  # remembered for re-enabling
    widget.use_lora = True
    draw(widget, viz)
    assert viz.args.diffusion["lora_path"] == "keep/me.safetensors"
    assert viz.args.diffusion["lora_scale"] == 3.5


def test_no_checkpoint_is_offered_when_none_is_installed(imgui_frame, widget_factory):
    """The field used to open on stabilityai/sd-turbo whether or not the user
    had it, which meant Enable silently downloaded gigabytes with no progress
    anywhere."""
    viz = make_viz()
    widget = widget_factory(viz)
    assert widget.params.model == ""
    draw(widget, viz)
    assert viz.args.diffusion["model"] == ""
    assert "No checkpoint" in widget.status_line("")[0]


def test_the_first_installed_checkpoint_is_offered(imgui_frame, widget_factory, monkeypatch):
    from utils import model_dir
    open(os.path.join(model_dir.diffusion_checkpoints_dir(), "a.safetensors"), "wb").close()
    widget = widget_factory(make_viz())
    assert os.path.basename(widget.params.model) == "a.safetensors"


def test_the_last_used_checkpoint_comes_back(imgui_frame, widget_factory, monkeypatch):
    from utils import model_dir, session_state
    ck = model_dir.diffusion_checkpoints_dir()
    for name in ("a.safetensors", "z.safetensors"):
        open(os.path.join(ck, name), "wb").close()
    viz = make_viz()
    widget = widget_factory(viz)
    widget.params.model = os.path.join(ck, "z.safetensors")
    draw(widget, viz)  # committing happens on the frame that sees the change
    assert session_state.get("diffusion", "model", "").endswith("z.safetensors")
    # a new panel in a new session starts where the last one left off, not at
    # the alphabetically first file
    assert widget_factory(make_viz()).params.model.endswith("z.safetensors")
