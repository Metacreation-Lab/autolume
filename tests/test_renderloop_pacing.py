"""The render worker renders on demand, so a stage that finishes loading in the
background needs frames to keep coming or its result never reaches the screen.
"""
import queue
import threading
import time

import dnnlib
from modules import renderloop


class FakeRenderer:
    """Reports a background load in flight for the first few frames."""

    def __init__(self, loading_frames=3):
        self.calls = 0
        self.loading_frames = loading_frames

    def render(self, **args):
        self.calls += 1
        return dnnlib.EasyDict(diffusion_loading=self.calls < self.loading_frames)


def run_loop(monkeypatch, fake):
    monkeypatch.setattr(renderloop.renderer, "Renderer", lambda: fake)
    args_queue, result_queue = queue.Queue(), queue.Queue()
    thread = threading.Thread(target=renderloop.AsyncRenderer._process_fn,
                              args=(args_queue, result_queue), daemon=True)
    thread.start()
    return args_queue, result_queue


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_frames_keep_coming_while_a_stage_is_loading(monkeypatch):
    fake = FakeRenderer(loading_frames=4)
    args_queue, _result_queue = run_loop(monkeypatch, fake)
    args_queue.put([dict(use_diffusion=True), 0])
    # one args change must yield several frames: enabling on a still image has
    # to reach the screen without the user nudging a control
    assert wait_for(lambda: fake.calls >= 4)


def test_loop_goes_idle_once_loading_finishes(monkeypatch):
    fake = FakeRenderer(loading_frames=2)
    args_queue, _result_queue = run_loop(monkeypatch, fake)
    args_queue.put([dict(use_diffusion=True), 0])
    assert wait_for(lambda: fake.calls >= 2)
    settled = fake.calls
    time.sleep(0.3)
    # no busy loop: with nothing loading and no new args, rendering stops
    assert fake.calls == settled


def test_new_args_still_render_when_nothing_is_loading(monkeypatch):
    fake = FakeRenderer(loading_frames=0)
    args_queue, _result_queue = run_loop(monkeypatch, fake)
    args_queue.put([dict(seed=1), 0])
    assert wait_for(lambda: fake.calls >= 1)
    args_queue.put([dict(seed=2), 0])
    assert wait_for(lambda: fake.calls >= 2)
