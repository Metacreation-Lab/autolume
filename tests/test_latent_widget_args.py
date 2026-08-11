import imgui
import pytest
import torch

import dnnlib
from modules.renderloop import compare_args
from widgets.latent_widget import LatentWidget


@pytest.fixture(autouse=True)
def no_imgui_ids(monkeypatch):
    monkeypatch.setattr(imgui, "push_id", lambda *_: None)
    monkeypatch.setattr(imgui, "pop_id", lambda: None)


def make_widget():
    w = LatentWidget.__new__(LatentWidget)
    w.latent = dnnlib.EasyDict(vec=torch.randn(1, 512), next=torch.randn(1, 512),
                               x=0, y=0, frac_x=0., frac_y=0.,
                               update_mode=1, speed=0.25, mode=False, project=True)
    w.viz = dnnlib.EasyDict(args=dnnlib.EasyDict(),
                            app=dnnlib.EasyDict(frame_delta=1 / 60))
    return w


def test_vec_anim_step_is_visible_to_render_args_gate():
    w = make_widget()
    w(show=False)
    frame1 = dict(w.viz.args)
    w.update_vec()
    w(show=False)
    frame2 = dict(w.viz.args)
    assert not compare_args(frame1, frame2)


def test_published_vec_does_not_alias_widget_state():
    w = make_widget()
    w(show=False)
    published = w.viz.args.vec
    w.latent.vec += 1.0
    assert not torch.equal(published, w.latent.vec)
