"""Headless draw test for the diffusion panel.

Renders the widget in a real imgui context with no backend, which catches
pyimgui API misuse (wrong return-value unpacking, unbalanced begin/end,
missing pops) that unit tests on the params alone cannot see.
"""
import pytest

imgui = pytest.importorskip("imgui")

import dnnlib
from widgets.diffusion_widget import DiffusionWidget


class FakeDispatcher:
    def streaming_addresses(self):
        return []

    def map_val_to_func(self, *args, **kwargs):
        pass

    def unmap(self, *args, **kwargs):
        pass


def make_viz():
    app = dnnlib.EasyDict(button_w=80.0, spacing=4.0, label_w=100.0, font_size=14.0,
                          content_width=1200, content_height=800)
    return dnnlib.EasyDict(app=app, pane_w=1000.0, result=dnnlib.EasyDict(),
                           args=dnnlib.EasyDict(), osc_dispatcher=FakeDispatcher())


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


def test_panel_draws_every_control(imgui_frame, monkeypatch, tmp_path):
    from utils import session_state
    monkeypatch.setattr(session_state, "cache_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(session_state, "_state", None)
    viz = make_viz()
    widget = DiffusionWidget(viz)
    height = draw(widget, viz)
    # enable + status, prompt, strength/seed, separator, checkpoint, resolution,
    # lora, weight, tensorrt and the osc child: a skipped section is far shorter
    assert height > 150
    assert viz.args.use_diffusion is False
    assert viz.args.diffusion["model"]


def test_panel_draws_while_enabled_and_erroring(imgui_frame, monkeypatch, tmp_path):
    from utils import session_state
    monkeypatch.setattr(session_state, "cache_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(session_state, "_state", None)
    viz = make_viz()
    widget = DiffusionWidget(viz)
    widget.available = True
    widget.enabled = True
    widget.use_lora = True
    widget.lora_path = "x.safetensors"
    viz.result.diffusion_status = "Error: something went wrong"
    draw(widget, viz)
    assert viz.args.use_diffusion is True
    assert viz.args.diffusion["lora_path"] == "x.safetensors"


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
