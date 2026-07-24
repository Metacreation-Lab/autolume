import threading

import pytest

from autolume.live.core.params import ControlState
from autolume.live.core.store import LatestValueStore


def test_snapshot_returns_initial():
    store = LatestValueStore(ControlState())
    assert store.snapshot() == ControlState()


def test_update_swaps_new_snapshot():
    store = LatestValueStore(ControlState())
    before = store.snapshot()
    after = store.update(latent_x=3.0)
    assert after.latent_x == 3.0
    assert before.latent_x == 0.0
    assert store.snapshot() is after


def test_update_unknown_field_raises():
    store = LatestValueStore(ControlState())
    with pytest.raises(TypeError):
        store.update(nope=1)


def test_set_replaces_wholesale():
    store = LatestValueStore(ControlState())
    replacement = ControlState(truncation_psi=1.5)
    store.set(replacement)
    assert store.snapshot() is replacement


def test_concurrent_updates_never_tear():
    store = LatestValueStore(ControlState())

    def hammer(field, stop):
        while not stop.is_set():
            store.update(**{field: 1.0})

    stop = threading.Event()
    threads = [
        threading.Thread(target=hammer, name=f, args=(f, stop))
        for f in ("latent_x", "latent_y", "anim_speed_x")
    ]
    for t in threads:
        t.start()
    for _ in range(2000):
        snap = store.snapshot()
        assert isinstance(snap, ControlState)
    stop.set()
    for t in threads:
        t.join()
    final = store.snapshot()
    assert final.latent_x == 1.0 and final.latent_y == 1.0
