import time

from autolume.live.core.control import ControlLoop
from autolume.live.core.events import ControlEvent
from autolume.live.core.params import ControlState, RenderParams, to_render_params
from autolume.live.core.store import LatestValueStore


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_loop(clock=None):
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    loop = ControlLoop(control_store, render_store, clock=clock or FakeClock())
    return loop, control_store, render_store


def test_tick_applies_events_in_order():
    loop, control_store, _ = make_loop()
    loop.submit(ControlEvent("/latent/x", 1.0))
    loop.submit(ControlEvent("/latent/x", 2.0))
    loop.tick()
    assert control_store.snapshot().latent_x == 2.0


def test_tick_publishes_render_params():
    loop, _, render_store = make_loop()
    loop.submit(ControlEvent("/trunc/psi", 1.2))
    result = loop.tick()
    assert isinstance(result, RenderParams)
    assert render_store.snapshot().truncation_psi == 1.2


def test_tick_integrates_motion_with_measured_dt():
    clock = FakeClock()
    loop, control_store, _ = make_loop(clock)
    loop.submit(ControlEvent("/anim/playing", True))
    loop.submit(ControlEvent("/anim/speed/x", 2.0))
    loop.tick()
    clock.now = 0.5
    loop.tick()
    assert abs(control_store.snapshot().latent_x - 1.0) < 1e-9


def test_first_tick_has_zero_dt():
    clock = FakeClock()
    clock.now = 100.0
    loop, control_store, _ = make_loop(clock)
    loop.submit(ControlEvent("/anim/playing", True))
    loop.tick()
    assert control_store.snapshot().latent_x == 0.0


def test_submit_overflow_drops_oldest():
    loop, control_store, _ = make_loop()
    for i in range(2000):
        loop.submit(ControlEvent("/latent/x", float(i)))
    loop.tick()
    assert control_store.snapshot().latent_x == 1999.0


def test_submit_stamps_timestamp():
    clock = FakeClock()
    clock.now = 42.0
    loop, _, _ = make_loop(clock)
    loop.submit(ControlEvent("/latent/x", 1.0))
    event = loop._queue[0]
    assert event.timestamp == 42.0


def test_thread_start_stop():
    control_store = LatestValueStore(ControlState())
    render_store = LatestValueStore(to_render_params(ControlState()))
    loop = ControlLoop(control_store, render_store, tick_hz=500.0)
    loop.start()
    loop.submit(ControlEvent("/latent/x", 7.0))
    deadline = time.monotonic() + 2.0
    while control_store.snapshot().latent_x != 7.0 and time.monotonic() < deadline:
        time.sleep(0.005)
    loop.stop()
    assert control_store.snapshot().latent_x == 7.0
