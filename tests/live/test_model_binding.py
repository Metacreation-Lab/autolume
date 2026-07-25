"""Switching models from a controller.

The model path is the one parameter a row drives with something that is not a
number, and the one whose remote value has to be resolved before it means
anything. Two things are pinned here: what a value resolves to, and that
nothing about resolving it can stall or crash the control thread.

The numeric path carries the most weight. A fader, a button and an encoder all
send numbers and almost no controller sends text, so a string only
implementation would satisfy "bindable" and be useless on stage. That is why
the expression running on the numeric path has its own test: it is what turns a
0 to 1 fader into a selector across the folder.
"""

import threading
import time

import pytest

from autolume.live.core.control import ControlLoop
from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import ModelHost
from autolume.live.core.models import ModelFolder
from autolume.live.core.params import (
    BINDING_SET,
    Binding,
    ControlState,
    to_render_params,
)
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.runtime import _ModelWatchingControlLoop

MODEL_ADDRESS = "/model/path"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeModel:
    def __init__(self, path):
        self.pkl_path = path


@pytest.fixture
def models(tmp_path):
    """A models folder, named so that sorted order is not creation order.

    The capital letter is deliberate. Sorted is what the index counts through
    and what the first substring match walks, so a folder where sorting does
    something is the only one that can tell a real ordering from an accident.
    """
    names = ("beach-256.pkl", "Wikiart-1024.pkl", "abstract-512.pkl")
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return [
        str(tmp_path / "Wikiart-1024.pkl"),
        str(tmp_path / "abstract-512.pkl"),
        str(tmp_path / "beach-256.pkl"),
    ]


def folder_of(paths, clock=None, interval=1.0):
    # Handed back in the wrong order on purpose. The index counts through the
    # sorted listing and the first substring match walks the same one, so
    # whether the folder sorts is load bearing for every test in this file.
    return ModelFolder(
        lister=lambda: list(reversed(paths)),
        clock=clock or FakeClock(),
        interval=interval,
    )


def make_loop(paths, clock=None, folder=None):
    clock = clock or FakeClock()
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    loop = ControlLoop(
        control_store,
        render_store,
        source_store,
        clock=clock,
        models=folder if folder is not None else folder_of(paths, clock),
    )
    return loop, control_store


def bind(loop, expression="x", enabled=True, source=""):
    loop.submit(
        ControlEvent(
            BINDING_SET,
            Binding("pkl_path", source, expression, enabled),
            source="ui",
        )
    )
    loop.tick()


def send(loop, value, source="osc"):
    loop.submit(ControlEvent(MODEL_ADDRESS, value, source=source))
    loop.tick()


def test_a_number_selects_the_model_at_that_position(models):
    loop, store = make_loop(models)
    bind(loop)

    send(loop, 2)

    assert store.snapshot().pkl_path == models[2]


def test_the_expression_scales_a_fader_across_the_folder(models):
    """The point of the whole numeric path.

    A controller sends 0 to 1 and the folder holds three models, so without the
    expression the fader is a switch between the first model and the second.
    """
    loop, store = make_loop(models)
    bind(loop, expression="x * 2")

    send(loop, 1.0)
    assert store.snapshot().pkl_path == models[2]

    send(loop, 0.5)
    assert store.snapshot().pkl_path == models[1]


def test_a_path_that_exists_is_taken_as_it_is(models, tmp_path):
    outside = tmp_path / "elsewhere" / "one-off.pkl"
    outside.parent.mkdir()
    outside.write_bytes(b"")
    loop, store = make_loop(models)
    bind(loop)

    send(loop, str(outside))

    assert store.snapshot().pkl_path == str(outside)


def test_a_filename_resolves_inside_the_models_folder(models):
    loop, store = make_loop(models)
    bind(loop)

    send(loop, "beach-256.pkl")

    assert store.snapshot().pkl_path == models[2]


@pytest.mark.parametrize("typed", ["wikiart", "WIKIART", "WikiArt", "Wikiart-1024"])
def test_part_of_a_name_is_enough_and_case_does_not_matter(models, typed):
    """Nobody types `Wikiart-1024.pkl` into a controller mid set.

    Ignoring case is a small, deliberate departure from the old app, which
    compared exactly. What is being compared is a fragment a performer typed
    into a controller or a cue list, and no one wants two models told apart by
    the capitalisation of their filenames.
    """
    loop, store = make_loop(models)
    bind(loop)

    send(loop, typed)

    assert store.snapshot().pkl_path == models[0]


def test_a_fragment_matching_several_models_takes_the_first_in_the_listing(tmp_path):
    """Deterministic, and deterministic in the order the index already uses.

    One listing then explains both halves of the row, so a performer who knows
    what index 1 is also knows which model an ambiguous name will pick.
    """
    for name in ("late-wikiart.pkl", "Wikiart-1024.pkl"):
        (tmp_path / name).write_bytes(b"")
    paths = [str(tmp_path / "Wikiart-1024.pkl"), str(tmp_path / "late-wikiart.pkl")]
    loop, store = make_loop(paths)
    bind(loop)

    send(loop, "wikiart")

    assert store.snapshot().pkl_path == paths[0]


def test_a_whole_filename_beats_a_fragment_of_a_longer_one(tmp_path):
    # `big-wikiart.pkl` sorts first and contains the whole of the other name,
    # so naming a model exactly has to win or it could never be selected.
    for name in ("big-wikiart.pkl", "wikiart.pkl"):
        (tmp_path / name).write_bytes(b"")
    paths = [str(tmp_path / "big-wikiart.pkl"), str(tmp_path / "wikiart.pkl")]
    loop, store = make_loop(paths)
    bind(loop)

    send(loop, "wikiart.pkl")

    assert store.snapshot().pkl_path == paths[1]


def test_the_expression_is_not_applied_to_a_name(models):
    """A name is not a number, so there is nothing for an expression to do.

    `x * 2` on the numeric path moves the selection two places. If it reached
    the text path at all, this would either raise or resolve to something else.
    """
    loop, store = make_loop(models)
    bind(loop, expression="x * 2")

    send(loop, "abstract")

    assert store.snapshot().pkl_path == models[1]


def test_an_index_off_the_end_is_ignored(models):
    """Ignored, not clamped, and never an error on the row.

    A performer sweeping a wrongly scaled fader sends the far end of it on the
    way past, so a row that went red there would flash red through every
    gesture. `_guard_hits` is the control loop's record of an event that raised
    at it, and it stays empty because nothing here may raise on that thread.
    """
    loop, store = make_loop(models)
    bind(loop)
    send(loop, 1)

    send(loop, 7)
    send(loop, -1)

    assert store.snapshot().pkl_path == models[1]
    assert store.snapshot().bindings[0].error is None
    assert loop._guard_hits == {}


def test_an_index_that_is_not_a_number_selects_nothing(models):
    """The folder is asked directly here, since the loop cannot deliver this.

    A non finite expression result is refused by the expression itself and
    marks the row failing, exactly as it does on every other parameter. This is
    the guard underneath that, for anything else that reaches the folder.
    """
    folder = folder_of(models)
    assert folder.at_index(float("nan")) is None
    assert folder.at_index(float("inf")) is None
    assert folder.at_index("second") is None
    assert folder.at_index(1) == models[1]


def test_a_name_matching_nothing_is_ignored(models):
    loop, store = make_loop(models)
    bind(loop)
    send(loop, 1)

    send(loop, "portraits")
    send(loop, "")

    assert store.snapshot().pkl_path == models[1]
    assert store.snapshot().bindings[0].error is None
    assert loop._guard_hits == {}


def test_a_models_folder_that_cannot_be_read_is_not_an_error():
    """A performer with no models folder is a normal state, not a crash.

    Both shapes of it: a folder that lists nothing, and one whose listing
    raises, which is what an unmounted data root does.
    """

    def unreadable():
        raise OSError("data root is gone")

    for lister in (list, unreadable):
        loop, store = make_loop([], folder=ModelFolder(lister=lister, interval=0.0))
        bind(loop)

        send(loop, 0)
        send(loop, "anything")

        assert store.snapshot().pkl_path is None
        # The loop's own guard would have caught anything raised at it and
        # dropped the event, which looks identical from the state. It has to
        # be empty, or "not an error" is only true of the outcome.
        assert loop._guard_hits == {}


def test_a_row_that_is_off_lets_nothing_through(models):
    loop, store = make_loop(models)
    bind(loop, enabled=False)

    send(loop, 1)
    send(loop, "beach")

    assert store.snapshot().pkl_path is None


def test_a_parameter_with_no_row_lets_nothing_through(models):
    """The default. A controller that finds the port cannot swap the model."""
    loop, store = make_loop(models)

    send(loop, 1)
    send(loop, "beach")

    assert store.snapshot().pkl_path is None


@pytest.mark.parametrize("enabled", [True, False])
def test_the_hand_opens_a_model_whatever_the_row_says(models, enabled):
    """The row governs the network and never the mouse.

    The path is taken verbatim rather than resolved, because the performer
    picked it in a file dialog and it is already the answer.
    """
    loop, store = make_loop(models)
    bind(loop, enabled=enabled)

    send(loop, "/from/the/dialog.pkl", source="ui")

    assert store.snapshot().pkl_path == "/from/the/dialog.pkl"


def test_the_folder_is_listed_once_for_a_burst_of_values(models):
    """Resolving happens on the control thread, so it may not read the folder
    per message. A swept fader sends as fast as its controller does, and
    resolving the data root parses the preferences file on every call.
    """
    reads = []
    clock = FakeClock()

    def counting():
        reads.append(clock.now)
        return list(models)

    folder = ModelFolder(lister=counting, clock=clock, interval=2.0)
    loop, _ = make_loop(models, clock=clock, folder=folder)
    bind(loop)

    for value in (0, 1, 2, 1, 0, 2, 1):
        send(loop, value)
    assert len(reads) == 1

    clock.now += 2.5
    send(loop, 0)
    assert len(reads) == 2


def test_a_model_added_to_the_folder_becomes_selectable(models, tmp_path):
    # The listing is cached, not frozen. A model dropped in mid show has to
    # arrive on its own rather than needing a restart.
    clock = FakeClock()
    folder = ModelFolder(
        lister=lambda: sorted_pkls(tmp_path), clock=clock, interval=2.0
    )
    loop, store = make_loop(models, clock=clock, folder=folder)
    bind(loop)
    send(loop, 0)

    (tmp_path / "0-new.pkl").write_bytes(b"")
    clock.now += 2.5
    send(loop, 0)

    assert store.snapshot().pkl_path == str(tmp_path / "0-new.pkl")


def sorted_pkls(directory):
    return sorted(str(path) for path in directory.glob("*.pkl"))


def watching_loop(host, paths):
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    source_store = LatestValueStore(SourceTable())
    return _ModelWatchingControlLoop(
        control_store,
        render_store,
        source_store,
        host,
        clock=FakeClock(),
        models=folder_of(paths),
    )


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_the_control_thread_never_waits_for_a_load(models):
    """Loading a model takes seconds. A tick takes eight milliseconds.

    The remote path writes the resolved path into the state and nothing else.
    The runtime notices the change on its own tick and hands it to the model
    host, which loads on its own thread, so a bound fader cannot park every
    other parameter in the app behind a file read.
    """
    release = threading.Event()
    entered = threading.Event()

    def slow(path):
        entered.set()
        release.wait(timeout=5.0)
        return FakeModel(path)

    host = ModelHost(loader=slow)
    loop = watching_loop(host, models)
    try:
        bind(loop)
        loop.submit(ControlEvent(MODEL_ADDRESS, 1, source="osc"))

        started = time.perf_counter()
        loop.tick()
        elapsed = time.perf_counter() - started

        assert elapsed < 0.5
        assert entered.wait(1.0)
        assert host.current() is None
        # And the parameter moved, so the tick did the work rather than
        # skipping it.
        assert loop._control_store.snapshot().pkl_path == models[1]
    finally:
        release.set()
        host.stop()


def test_a_swept_fader_coalesces_to_the_model_it_lands_on(models):
    """Every intermediate value the performer passed through is skipped.

    A fader crossing the folder asks for every model on the way. Loading each
    one would queue seconds of work for a gesture that took a moment, so the
    host only ever finishes the one that was asked for last.
    """
    release = threading.Event()
    started = threading.Event()
    loaded = []

    def slow(path):
        loaded.append(path)
        started.set()
        release.wait(timeout=5.0)
        return FakeModel(path)

    host = ModelHost(loader=slow)
    loop = watching_loop(host, models)
    try:
        bind(loop)
        loop.submit(ControlEvent(MODEL_ADDRESS, 0, source="osc"))
        loop.tick()
        assert started.wait(1.0)

        for value in (1, 2, 1, 0, 2):
            loop.submit(ControlEvent(MODEL_ADDRESS, value, source="osc"))
            loop.tick()
        release.set()

        assert wait_for(
            lambda: host.current() is not None
            and host.current().pkl_path == models[2]
        )
        assert loaded == [models[0], models[2]]
    finally:
        release.set()
        host.stop()
