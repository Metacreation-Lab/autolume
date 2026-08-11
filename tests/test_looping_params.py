import dnnlib
from widgets.looping_widget import LoopingWidget


class QueueStub:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def make_widget():
    w = LoopingWidget.__new__(LoopingWidget)
    w.params = dnnlib.EasyDict(num_keyframes=6, mode=True, anim=False, index=0,
                               looptime=4, perfect_loop=False)
    w.keyframes = ["kf"] * 6
    w.alpha = 0
    w.expand_vec = False
    w.seeds = [[i, 0] for i in range(6)]
    w.modes = [0] * 6
    w.project = [True] * 6
    w.paths = [""] * 6
    w.loop_type = True
    w.radius = 1.0
    w.noise_seed = 0
    w.args_queue = QueueStub()
    return w


def test_roundtrip_preserves_noise_loop_state():
    src = make_widget()
    src.params.anim = True
    src.loop_type = False  # NoiseLoop mode
    src.radius = 10.0
    src.noise_seed = 7
    dst = make_widget()
    dst.set_params(src.get_params())
    assert dst.loop_type is False
    assert dst.radius == 10.0
    assert dst.noise_seed == 7
    assert dst.params.anim is True


def test_load_resyncs_noise_loop_worker():
    src = make_widget()
    src.loop_type = False
    src.radius = 10.0
    src.noise_seed = 7
    dst = make_widget()
    dst.set_params(src.get_params())
    assert dst.args_queue.items == [(7, 10.0)]


def test_legacy_12_tuple_still_loads():
    legacy = (6, ["kf"] * 6, 0, 0, True, False, 4, False,
              [[i, 0] for i in range(6)], [0] * 6, [True] * 6, [""] * 6)
    dst = make_widget()
    dst.set_params(legacy)
    assert dst.loop_type is True
    assert dst.radius == 1.0
    assert dst.noise_seed == 0
    assert dst.args_queue.items == []
