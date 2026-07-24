import threading
import time

import pytest

from autolume.live.core.generator import ModelHost, corner_seeds


def test_corner_seeds_integer_position_is_single_seed():
    corners = corner_seeds(3.0, 2.0, step_y=100)
    assert corners == [(3 + 2 * 100, 1.0)]


def test_corner_seeds_weights_sum_to_one():
    corners = corner_seeds(1.25, 7.5, step_y=100)
    assert len(corners) == 4
    assert abs(sum(w for _, w in corners) - 1.0) < 1e-9


def test_corner_seeds_negative_positions_wrap_to_uint32():
    corners = corner_seeds(-1.5, 0.0, step_y=100)
    assert all(0 <= seed < 2**32 for seed, _ in corners)


def test_corner_seeds_deterministic():
    assert corner_seeds(0.3, 0.7) == corner_seeds(0.3, 0.7)


class FakeModel:
    def __init__(self, path):
        self.pkl_path = path


def test_model_host_loads_in_background():
    host = ModelHost(loader=FakeModel)
    assert host.current() is None
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.current().pkl_path == "/tmp/a.pkl"
    assert host.error() is None
    host.stop()


def test_model_host_surfaces_loader_error():
    def failing(path):
        raise RuntimeError("bad pkl")

    host = ModelHost(loader=failing)
    host.request_load("/tmp/bad.pkl")
    deadline = time.monotonic() + 2.0
    while host.error() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert "bad pkl" in host.error()
    assert host.current() is None
    host.stop()


def test_model_host_coalesces_to_newest_request():
    release = threading.Event()
    loaded = []

    def slow(path):
        release.wait(timeout=2.0)
        loaded.append(path)
        return FakeModel(path)

    host = ModelHost(loader=slow)
    host.request_load("/tmp/a.pkl")
    host.request_load("/tmp/b.pkl")
    host.request_load("/tmp/c.pkl")
    release.set()
    deadline = time.monotonic() + 2.0
    while (host.current() is None or host.current().pkl_path != "/tmp/c.pkl") and \
            time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.current().pkl_path == "/tmp/c.pkl"
    assert "/tmp/b.pkl" not in loaded
    host.stop()
