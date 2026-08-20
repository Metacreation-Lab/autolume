import time

import pytest

from utils.dataset_preprocessing_utils import VideoDurationProber


def poll_until(prober, expected_count, timeout=5.0):
    """Collect poll() results until expected_count paths arrived or timeout."""
    results = {}
    deadline = time.monotonic() + timeout
    while len(results) < expected_count and time.monotonic() < deadline:
        results.update(prober.poll())
        time.sleep(0.01)
    return results


@pytest.fixture
def prober(request):
    def make(probe_fn):
        p = VideoDurationProber(probe_fn=probe_fn)
        request.addfinalizer(p.shutdown)
        return p
    return make


def test_results_arrive_via_poll(prober):
    p = prober(lambda path: {"a.mp4": 10.0, "b.mp4": 2.5}[path])
    p.start(["a.mp4", "b.mp4"])
    assert poll_until(p, 2) == {"a.mp4": 10.0, "b.mp4": 2.5}


def test_poll_with_nothing_pending_returns_empty(prober):
    p = prober(lambda path: 1.0)
    assert p.poll() == {}
    p.start(["a.mp4"])
    poll_until(p, 1)
    assert p.poll() == {}


def test_cancel_discards_in_flight_results(prober):
    p = prober(lambda path: 1.0)
    p.start(["a.mp4"])
    # Let the probe finish before cancelling so its result is queued but stale.
    time.sleep(0.2)
    p.cancel()
    assert poll_until(p, 1, timeout=0.3) == {}


def test_restart_discards_previous_generation(prober):
    p = prober(lambda path: 1.0 if path == "old.mp4" else 2.0)
    p.start(["old.mp4"])
    time.sleep(0.2)
    p.start(["new.mp4"])
    assert poll_until(p, 1) == {"new.mp4": 2.0}


def test_failing_probe_reports_zero_duration(prober):
    def probe(path):
        raise RuntimeError("ffprobe exploded")
    p = prober(probe)
    p.start(["broken.mp4"])
    assert poll_until(p, 1) == {"broken.mp4": 0}
