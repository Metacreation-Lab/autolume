import time

import numpy as np

from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import ModelHost
from autolume.live.runtime import build_runtime


class FakeModel:
    def __init__(self, path):
        self.pkl_path = path

    def render_frame(self, params, frame_index):
        value = int(abs(params.latent_x) * 10) % 256
        return np.full((8, 8, 3), value, dtype=np.uint8)


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_end_to_end_headless_flow():
    host = ModelHost(loader=FakeModel)
    runtime = build_runtime(model_host=host, start_osc=False)
    runtime.start()
    try:
        runtime.submit(ControlEvent("/model/path", "/tmp/fake.pkl"))
        assert wait_for(lambda: host.current() is not None)
        assert wait_for(lambda: runtime.preview.latest()[1] is not None)

        runtime.submit(ControlEvent("/latent/x", 5.0))
        assert wait_for(
            lambda: runtime.preview.latest()[1] is not None
            and runtime.preview.latest()[1][0, 0, 0] == 50
        )

        runtime.submit(ControlEvent("/anim/playing", 1.0))
        runtime.submit(ControlEvent("/anim/speed/x", 4.0))
        x0 = runtime.control_store.snapshot().latent_x
        time.sleep(0.3)
        assert runtime.control_store.snapshot().latent_x > x0
    finally:
        runtime.stop()


def test_stop_is_clean_and_idempotent():
    runtime = build_runtime(model_host=ModelHost(loader=FakeModel), start_osc=False)
    runtime.start()
    runtime.stop()
    runtime.stop()
