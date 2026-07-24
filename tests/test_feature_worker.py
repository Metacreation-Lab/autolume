import queue
import time

import pytest
import torch

from features.extraction import ExtractionConfig
from features import store, worker
from features.worker import FeatureExtractionJob


class FakeMapping(torch.nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.lin = torch.nn.Linear(z_dim, w_dim, bias=False)

    def forward(self, z, c=None):
        w = self.lin(z)
        return torch.stack([w, w], dim=1)


class FakeGenerator(torch.nn.Module):
    def __init__(self, z_dim=16, w_dim=16):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.mapping = FakeMapping(z_dim, w_dim)


def drain(q):
    messages = []
    while not q.empty():
        messages.append(q.get())
    return messages


def test_job_fn_extracts_and_saves(tmp_path, monkeypatch):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"fake-model")
    root = tmp_path / "features"
    monkeypatch.setattr(worker, "_load_generator",
                        lambda path, device: FakeGenerator())
    monkeypatch.setattr(worker, "_default_config",
                        lambda: ExtractionConfig(n_features=4, n_samples=1024,
                                                 batch_size=256))
    q = queue.Queue()
    worker._job_fn(str(model), q, root=str(root))
    messages = drain(q)
    assert messages[-1][0] == "done"
    assert any(m[0] == "progress" for m in messages)
    fs = store.lookup(model, root=root)
    assert fs is not None
    assert fs.directions.shape == (4, 16)


def test_job_fn_reports_error(tmp_path, monkeypatch):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"fake-model")

    def boom(path, device):
        raise RuntimeError("no mapping network")

    monkeypatch.setattr(worker, "_load_generator", boom)
    q = queue.Queue()
    worker._job_fn(str(model), q, root=str(tmp_path / "features"))
    messages = drain(q)
    assert messages[-1][0] == "error"
    assert "no mapping network" in messages[-1][1]


class StubProcess:
    def __init__(self, *args, **kwargs):
        self.exitcode = None
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.exitcode = -15

    def join(self, timeout=None):
        pass


def poll_until(job, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job.poll()
        if predicate(job):
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_job_poll_progress_then_done():
    job = FeatureExtractionJob("model.pkl", process_factory=StubProcess)
    assert job._process.started
    job._queue.put(("progress", 0.5, "Sampling"))
    poll_until(job, lambda j: j.progress == 0.5)
    assert job.message == "Sampling"
    assert job.error is None and not job.done
    job._queue.put(("done", "/tmp/features/x.npz"))
    poll_until(job, lambda j: j.done)
    assert job.result_path == "/tmp/features/x.npz"


def test_job_poll_error_message():
    job = FeatureExtractionJob("model.pkl", process_factory=StubProcess)
    job._queue.put(("error", "exploded"))
    poll_until(job, lambda j: j.error is not None)
    assert job.error == "exploded"
    assert not job.done


def test_dead_process_without_message_is_an_error():
    job = FeatureExtractionJob("model.pkl", process_factory=StubProcess)
    job._process.exitcode = 1
    job.poll()
    assert job.error is not None


def test_cancel_terminates():
    job = FeatureExtractionJob("model.pkl", process_factory=StubProcess)
    job.cancel()
    assert job._process.exitcode == -15
