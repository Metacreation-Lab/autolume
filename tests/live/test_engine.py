import logging
import time

import numpy as np
import pytest

from autolume.live.core.engine import RenderLoop, RenderStatus
from autolume.live.core.params import ControlState, to_render_params
from autolume.live.core.sinks import PreviewMailbox
from autolume.live.core.store import LatestValueStore

# The bending tests below make the generator import the operator library, and
# merely importing kornia trips a torch FutureWarning from its lightglue
# submodule. Matched by message so it cannot mask anything else.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*torch\.cuda\.amp\.custom_fwd.*:FutureWarning"
)


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


# --- the render status channel --------------------------------------------
#
# Every other subsystem has one (RecorderStatus, NdiStatus, OscStatus, the
# host's error()). The render path was the sole exception, and the only one
# whose failure the performer can see, as a picture that stops moving, without
# being able to read why.


class BrokenModel:
    """A model that raises on every frame, like C1's size-mismatch did."""

    pkl_path = "/tmp/broken.pkl"

    def __init__(self, message=None):
        self.calls = 0
        self._message = message

    def render_frame(self, params, frame_index):
        self.calls += 1
        if self._message is not None:
            raise RuntimeError(self._message(self.calls))
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")


def test_a_render_failure_is_published_for_the_ui():
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(BrokenModel()), [])
    assert loop.render_one() is False
    status = loop.status_store.snapshot()
    assert status.error == "mat1 and mat2 shapes cannot be multiplied"
    assert status.failed_frames == 1
    assert loop.render_one() is False
    assert loop.status_store.snapshot().failed_frames == 2


def test_no_model_is_idle_not_a_failure():
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(None), [])
    assert loop.render_one() is False
    assert loop.status_store.snapshot() == RenderStatus()


def test_the_status_names_the_frame_the_picture_is_stuck_on():
    host = FakeHost(FakeModel())
    loop = RenderLoop(make_store(fps_cap=0), host, [])
    for _ in range(3):
        assert loop.render_one() is True
    host.model = BrokenModel()
    loop.render_one()
    loop.render_one()
    status = loop.status_store.snapshot()
    assert status.last_ok_seq == 2
    assert status.failed_frames == 2
    assert status.error is not None


def test_the_first_good_frame_clears_the_failure():
    class Flaky:
        pkl_path = "/tmp/flaky.pkl"

        def __init__(self):
            self.calls = 0

        def render_frame(self, params, frame_index):
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("transient")
            return np.zeros((4, 4, 3), dtype=np.uint8)

    loop = RenderLoop(make_store(fps_cap=0), FakeHost(Flaky()), [])
    loop.render_one()
    loop.render_one()
    assert loop.status_store.snapshot().failed_frames == 2
    assert loop.render_one() is True
    assert loop.status_store.snapshot() == RenderStatus(
        error=None, failed_frames=0, last_ok_seq=0
    )


def test_a_failure_repeating_with_varying_numbers_is_logged_once(caplog):
    # A CUDA OOM message embeds varying byte counts. Keyed on the raw text it
    # filled the whole dedup set in seconds; normalised, it is one cause.
    model = BrokenModel(lambda n: f"Tried to allocate {n * 20} MiB")
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(model), [])
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            loop.render_one()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    # The status still carries the current message, not the first one.
    assert loop.status_store.snapshot().error == "Tried to allocate 100 MiB"


def test_the_render_failure_log_cannot_grow_without_bound(caplog):
    model = BrokenModel(lambda n: "cause " + "x" * n)
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(model), [])
    with caplog.at_level(logging.WARNING):
        for _ in range(70):
            loop.render_one()
    tracebacks = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(tracebacks) == 64
    cap_notices = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "not be logged" in r.getMessage()
    ]
    assert len(cap_notices) == 1
    # The status channel is not capped: it is state, not a log.
    assert loop.status_store.snapshot().failed_frames == 70


def test_a_broken_error_str_still_reaches_the_status():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("broken __str__")

    class Model:
        pkl_path = "/tmp/unprintable.pkl"

        def render_frame(self, params, frame_index):
            raise Unprintable()

    loop = RenderLoop(make_store(fps_cap=0), FakeHost(Model()), [])
    assert loop.render_one() is False
    assert loop.status_store.snapshot().error == "Unprintable"


# --- screenshots ----------------------------------------------------------
#
# The loop latches a request and hands the next frame it fans out to a writer
# that is somebody else's thread. Nothing here touches disk.


def test_a_screenshot_request_latches_exactly_one_frame():
    shots = []
    loop = RenderLoop(
        make_store(fps_cap=0),
        FakeHost(FakeModel()),
        [],
        screenshot=lambda path, frame: shots.append((path, frame)),
    )
    loop.request_screenshot("/captures/one.png")
    loop.render_one()
    loop.render_one()
    loop.render_one()
    assert len(shots) == 1
    path, frame = shots[0]
    assert path == "/captures/one.png"
    assert frame.shape == (4, 4, 3)


def test_a_screenshot_takes_the_frame_that_came_after_the_request():
    shots = []
    loop = RenderLoop(
        make_store(fps_cap=0),
        FakeHost(FakeModel()),
        [],
        screenshot=lambda path, frame: shots.append((path, frame)),
    )
    loop.render_one()
    loop.request_screenshot("/captures/one.png")
    loop.render_one()
    # FakeModel paints each frame with its call count, so this names the frame.
    assert shots[0][1][0, 0, 0] == 2


def test_a_request_with_nothing_rendering_waits_for_a_frame():
    """No model means no frame, and the request keeps rather than loses it."""
    shots = []
    host = FakeHost(None)
    loop = RenderLoop(
        make_store(fps_cap=0),
        host,
        [],
        screenshot=lambda path, frame: shots.append((path, frame)),
    )
    loop.request_screenshot("/captures/one.png")
    assert loop.render_one() is False
    assert shots == []
    host.model = FakeModel()
    loop.render_one()
    assert len(shots) == 1


def test_a_second_request_replaces_one_that_has_not_been_served():
    shots = []
    loop = RenderLoop(
        make_store(fps_cap=0),
        FakeHost(FakeModel()),
        [],
        screenshot=lambda path, frame: shots.append((path, frame)),
    )
    loop.request_screenshot("/captures/one.png")
    loop.request_screenshot("/captures/two.png")
    loop.render_one()
    loop.render_one()
    assert [path for path, _ in shots] == ["/captures/two.png"]


def test_a_screenshot_writer_that_fails_does_not_stop_the_loop():
    def explode(path, frame):
        raise RuntimeError("no disk")

    mailbox = PreviewMailbox()
    loop = RenderLoop(
        make_store(fps_cap=0), FakeHost(FakeModel()), [mailbox], screenshot=explode
    )
    loop.request_screenshot("/captures/one.png")
    assert loop.render_one() is True
    assert loop.render_one() is True
    assert mailbox.latest()[1] is not None


def test_a_request_with_no_writer_wired_is_dropped():
    loop = RenderLoop(make_store(fps_cap=0), FakeHost(FakeModel()), [])
    loop.request_screenshot("/captures/one.png")
    assert loop.render_one() is True


def test_a_screenshot_gets_the_same_read_only_frame_the_sinks_got():
    shots = []
    mailbox = PreviewMailbox()
    loop = RenderLoop(
        make_store(fps_cap=0),
        FakeHost(FakeModel()),
        [mailbox],
        screenshot=lambda path, frame: shots.append((path, frame)),
    )
    loop.request_screenshot("/captures/one.png")
    loop.render_one()
    assert shots[0][1] is mailbox.latest()[1]
    assert shots[0][1].flags.writeable is False


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


# --- the render loop against a real generator -----------------------------
#
# Everything above stands a fake model in for the generator. These drive the
# whole plan 4 render path (bending hooks, layer capture, image derivation)
# through a real LoadedModel, which is the only place the two halves meet.


def _bendable_model():
    import torch
    import torch.nn as nn

    from autolume.live.core.generator import LoadedModel

    class _Conv(nn.Module):
        def forward(self, x):
            return x * 0.5

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _Conv()

        def forward(self, ws, noise_mode="const"):
            return self.conv1(torch.full([1, 3, 2, 4], 0.5))

    class _Mapping:
        def __init__(self):
            self.w_avg = torch.zeros([8])
            self.c_dim = 0

        def __call__(self, z, c, truncation_psi):
            return torch.zeros([z.shape[0], 2, 8])

    class _G:
        z_dim = 4
        num_ws = 2

        def __init__(self):
            self.mapping = _Mapping()
            self.synthesis = _Synthesis()

    return LoadedModel("/tmp/bendable.pkl", _G(), torch.device("cpu"))


def test_the_loop_fans_out_a_bent_frame():
    from autolume.live.core.params import Transform

    mailbox = PreviewMailbox()
    store = make_store(
        fps_cap=0, transforms=(Transform("ablate", "conv1", (1.0,), (0, 1, 2)),)
    )
    loop = RenderLoop(store, FakeHost(_bendable_model()), [mailbox])

    assert loop.render_one() is True
    frame = mailbox.latest()[1]
    assert frame.shape == (2, 4, 3)
    assert frame[0, 0].tolist() == [128, 128, 128]
    assert frame.flags.writeable is False


def test_the_loop_keeps_going_when_a_transform_cannot_be_applied():
    from autolume.live.core.params import Transform

    mailbox = PreviewMailbox()
    # Channel 99 does not exist on this layer, so the transform is dropped
    # before it can reach the operator. It never becomes an exception at all,
    # which is what keeps a CUDA context alive rather than merely catching a
    # device side assert that has already poisoned it.
    store = make_store(
        fps_cap=0, transforms=(Transform("ablate", "conv1", (1.0,), (99,)),)
    )
    loop = RenderLoop(store, FakeHost(_bendable_model()), [mailbox])

    assert loop.render_one() is True
    assert loop.render_one() is True
    # 0.5 through conv1 is 0.25, the unbent value.
    assert mailbox.latest()[1][0, 0].tolist() == [159, 159, 159]


def test_the_loop_fans_out_a_captured_layer_at_its_own_size():
    mailbox = PreviewMailbox()
    store = make_store(fps_cap=0, capture_layer="output", grayscale=True)
    loop = RenderLoop(store, FakeHost(_bendable_model()), [mailbox])

    assert loop.render_one() is True
    frame = mailbox.latest()[1]
    assert frame.shape == (2, 4, 3)
    assert frame[0, 0].tolist() == [159, 159, 159]
