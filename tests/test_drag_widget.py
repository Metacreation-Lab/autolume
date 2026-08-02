import queue
import types

import numpy as np
import torch

from widgets.drag_widget import DragWidget


class _Latent:
    def __init__(self):
        self.mode = True
        self.project = 'untouched'
        self.vec = None


def _viz():
    return types.SimpleNamespace(
        args={'pkl': None, 'device': 'cpu'},
        result=None,
        latent_widget=types.SimpleNamespace(latent=_Latent()),
        app=types.SimpleNamespace(font_size=12, button_w=100, spacing=4),
        print_error=lambda e: None,
    )


def _in_flight_widget(num_ws=8, w_dim=512):
    widget = DragWidget(_viz())
    rng = np.random.RandomState(0)
    widget._w = rng.randn(1, num_ws, w_dim).astype(np.float32)
    widget._d0 = rng.randn(w_dim).astype(np.float32)
    widget._cmd = queue.Queue()
    widget.dragging = True
    return widget


def test_commit_recomposes_to_the_dragged_latent():
    widget = _in_flight_widget()
    worker_w = widget._w.copy()
    direction = widget._d0.copy()
    latent = widget.viz.latent_widget.latent

    widget.stop_drag(commit=True)

    # The renderer adds the adjuster direction back on top of the served vector,
    # so the composition must land exactly on the last dragged latent (no jump).
    composed = latent.vec + torch.from_numpy(direction)
    assert torch.allclose(composed, torch.from_numpy(worker_w[0]), atol=1e-6)
    assert latent.vec.shape == (8, 512)
    assert latent.mode is False
    assert latent.project == 'untouched'   # shared with seed mode, must not change
    assert widget.dragging is False


def test_commit_without_direction_serves_the_worker_latent():
    widget = _in_flight_widget()
    widget._d0 = None
    worker_w = widget._w.copy()

    widget.stop_drag(commit=True)

    latent = widget.viz.latent_widget.latent
    assert torch.allclose(latent.vec, torch.from_numpy(worker_w[0]))


def test_wants_input_is_false_after_a_worker_error():
    widget = DragWidget(_viz())
    widget.armed = True
    assert widget.wants_input()
    widget._error = 'worker died'
    assert not widget.wants_input()
