import time

import numpy as np

from autolume.live.core.engine import RenderLoop
from autolume.live.core.params import ControlState, to_render_params
from autolume.live.core.sinks import PreviewMailbox
from autolume.live.core.store import LatestValueStore


class FakeModel:
    pkl_path = "/tmp/fake.pkl"

    def __init__(self):
        self.calls = []

    def render_frame(self, latent_x, latent_y, truncation_psi):
        self.calls.append((latent_x, latent_y, truncation_psi))
        return np.full((4, 4, 3), len(self.calls) % 256, dtype=np.uint8)


class FakeHost:
    def __init__(self, model=None):
        self.model = model

    def current(self):
        return self.model


def make_store(**changes):
    state = ControlState(**changes)
    return LatestValueStore(to_render_params(state))


def test_render_one_without_model_produces_nothing():
    mailbox = PreviewMailbox()
    loop = RenderLoop(make_store(), FakeHost(None), [mailbox])
    assert loop.render_one() is False
    assert mailbox.latest() == (0, None)


def test_render_one_feeds_sinks_with_increasing_seq():
    mailbox = PreviewMailbox()
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(FakeModel()), [mailbox])
    assert loop.render_one() is True
    seq1, frame1 = mailbox.latest()
    assert loop.render_one() is True
    seq2, frame2 = mailbox.latest()
    assert seq2 == seq1 + 1
    assert frame2 is not None and frame2.shape == (4, 4, 3)


def test_render_uses_current_params():
    model = FakeModel()
    store = make_store(latent_x=3.5, truncation_psi=1.2, fps_cap=0)
    loop = RenderLoop(store, FakeHost(model), [])
    loop.render_one()
    assert model.calls[-1] == (3.5, 0.0, 1.2)


def test_sink_error_does_not_kill_loop():
    class BadSink:
        def on_frame(self, frame, seq):
            raise RuntimeError("sink exploded")

    mailbox = PreviewMailbox()
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(FakeModel()), [BadSink(), mailbox])
    assert loop.render_one() is True
    assert mailbox.latest()[1] is not None


def test_thread_start_stop_produces_frames():
    mailbox = PreviewMailbox()
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(FakeModel()), [mailbox])
    loop.start()
    deadline = time.monotonic() + 2.0
    while mailbox.latest()[1] is None and time.monotonic() < deadline:
        time.sleep(0.005)
    loop.stop()
    assert mailbox.latest()[1] is not None
    assert loop.fps() >= 0.0
