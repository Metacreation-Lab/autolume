import time

import numpy as np
import pytest

from autolume.live.core.engine import RenderLoop
from autolume.live.core.params import ControlState, to_render_params
from autolume.live.core.sinks import PreviewMailbox
from autolume.live.core.store import LatestValueStore


class FakeModel:
    pkl_path = "/tmp/fake.pkl"

    def __init__(self):
        self.calls = []

    def render_frame(self, params, frame_index):
        self.calls.append((params, frame_index))
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
    seq1, _ = mailbox.latest()
    assert loop.render_one() is True
    seq2, frame2 = mailbox.latest()
    assert seq2 == seq1 + 1
    assert frame2 is not None and frame2.shape == (4, 4, 3)


def test_render_uses_current_params():
    model = FakeModel()
    store = make_store(latent_x=3.5, truncation_psi=1.2, fps_cap=0)
    loop = RenderLoop(store, FakeHost(model), [])
    loop.render_one()
    params, _ = model.calls[-1]
    assert params is store.snapshot()
    assert (params.latent_x, params.latent_y, params.truncation_psi) == (3.5, 0.0, 1.2)


def test_render_passes_increasing_frame_index():
    model = FakeModel()
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(model), [])
    loop.render_one()
    loop.render_one()
    loop.render_one()
    assert [frame_index for _, frame_index in model.calls] == [0, 1, 2]


def test_sink_seq_is_the_index_the_frame_was_rendered_with():
    model = FakeModel()
    mailbox = PreviewMailbox()
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(model), [mailbox])
    for _ in range(3):
        loop.render_one()
        assert mailbox.latest()[0] == model.calls[-1][1]


def test_a_sink_cannot_write_to_the_frame_it_is_handed():
    """Every sink gets the same array, so none of them may write to it.

    Today the preview is the only sink and nothing has been corrupted. The
    parity plan adds NDI, a recorder and a fullscreen output, and at that point
    one consumer tinting its frame in place tints the show and the recording
    with it. The flag makes that a ValueError in the code that did it, at the
    moment it does it, instead of a picture nobody can explain.
    """
    captured = []

    class RecordingSink:
        def on_frame(self, frame, seq):
            captured.append(frame)

    loop = RenderLoop(make_store(fps_cap=0), FakeHost(FakeModel()), [RecordingSink()])
    assert loop.render_one() is True
    frame = captured[0]
    assert frame.flags.writeable is False
    with pytest.raises(ValueError):
        frame[0, 0, 0] = 0
    with pytest.raises(ValueError):
        frame //= 2


def test_every_sink_is_covered_without_having_to_remember():
    """The flag is set once before the fan-out, not by each sink in turn.

    A sink added later inherits the guarantee by construction. The first sink
    in the list is as protected as the last, which is what says the mark
    happens before the loop rather than inside it.
    """
    seen = []

    class Watcher:
        def __init__(self, name):
            self.name = name

        def on_frame(self, frame, seq):
            seen.append((self.name, frame.flags.writeable))

    loop = RenderLoop(
        make_store(fps_cap=0),
        FakeHost(FakeModel()),
        [Watcher("first"), Watcher("second"), Watcher("third")],
    )
    loop.render_one()
    assert seen == [("first", False), ("second", False), ("third", False)]


def test_a_sink_that_writes_anyway_does_not_take_the_loop_down():
    """The error is loud in the offender and survivable everywhere else.

    A consumer that has not learned to copy yet must not be able to stop the
    frames, because the render loop feeding the show outranks any one output.
    """

    class WritingSink:
        def on_frame(self, frame, seq):
            frame[:] = 0

    mailbox = PreviewMailbox()
    loop = RenderLoop(
        make_store(fps_cap=0), FakeHost(FakeModel()), [WritingSink(), mailbox]
    )
    assert loop.render_one() is True
    assert mailbox.latest()[1] is not None


def test_sink_error_does_not_kill_loop():
    class BadSink:
        def on_frame(self, frame, seq):
            raise RuntimeError("sink exploded")

    mailbox = PreviewMailbox()
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(FakeModel()), [BadSink(), mailbox])
    assert loop.render_one() is True
    assert mailbox.latest()[1] is not None


def test_render_error_does_not_kill_loop():
    class FlakyModel:
        pkl_path = "/tmp/flaky.pkl"

        def __init__(self):
            self.calls = 0

        def render_frame(self, params, frame_index):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthesis exploded")
            return np.full((4, 4, 3), 7, dtype=np.uint8)

    mailbox = PreviewMailbox()
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(FlakyModel()), [mailbox])
    assert loop.render_one() is False
    assert mailbox.latest() == (0, None)
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
