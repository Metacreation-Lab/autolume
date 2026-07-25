import threading
import time

import pytest

from autolume.live.core.generator import (
    ModelHost,
    corner_seeds,
    effective_noise_seed,
    noise_mode,
)
from autolume.live.core.params import ControlState, to_render_params


def render_params(**changes):
    return to_render_params(ControlState(**changes))


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


class _FakeMapping:
    def __init__(self, w_avg, num_ws):
        self.w_avg = w_avg
        self.c_dim = 0
        self._num_ws = num_ws

    def __call__(self, z, c, truncation_psi):
        import torch

        return torch.zeros([z.shape[0], self._num_ws, self.w_avg.shape[0]])


class _FakeG:
    z_dim = 4

    def __init__(self, synthesis, modules=()):
        import torch

        self.mapping = _FakeMapping(torch.zeros([8]), num_ws=2)
        self.synthesis = synthesis
        self._modules = list(modules)
        self.module_walks = 0

    def modules(self):
        self.module_walks += 1
        return list(self._modules)


class _NoisyModule:
    def __init__(self):
        self.global_noise = 1.0


class _PlainModule:
    pass


def _fake_model(synthesis, modules=()):
    import torch

    from autolume.live.core.generator import LoadedModel

    return LoadedModel(
        "/tmp/fake.pkl", _FakeG(synthesis, modules), torch.device("cpu")
    )


def _zeros_synthesis(ws, noise_mode):
    import torch

    return torch.zeros([1, 3, 8, 8])


def test_render_frame_with_tensor_synthesis_output():
    model = _fake_model(_zeros_synthesis)
    frame = model.render_frame(render_params(), 0)
    assert frame.shape == (8, 8, 3)
    assert frame.dtype.name == "uint8"


def test_render_frame_with_tuple_synthesis_output():
    def synthesis(ws, noise_mode):
        import torch

        return torch.zeros([1, 3, 8, 8]), []

    model = _fake_model(synthesis)
    frame = model.render_frame(render_params(), 0)
    assert frame.shape == (8, 8, 3)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"noise_enabled": False}, "none"),
        ({"noise_enabled": False, "noise_seed": 9, "noise_anim": True}, "none"),
        ({"noise_seed": 0, "noise_anim": False}, "const"),
        ({"noise_seed": 9, "noise_anim": False}, "random"),
        ({"noise_seed": 0, "noise_anim": True}, "random"),
    ],
)
def test_noise_mode_truth_table(changes, expected):
    assert noise_mode(render_params(**changes)) == expected


def test_effective_noise_seed_static_ignores_frame_index():
    params = render_params(noise_seed=42, noise_anim=False)
    assert effective_noise_seed(params, 0) == 42
    assert effective_noise_seed(params, 137) == 42


def test_effective_noise_seed_animated_advances_with_frame_index():
    params = render_params(noise_seed=42, noise_anim=True)
    assert effective_noise_seed(params, 0) == 42
    assert effective_noise_seed(params, 3) == 45


def test_effective_noise_seed_stays_within_32_bits():
    params = render_params(noise_seed=2**31 - 1, noise_anim=True)
    # 2**34 is a whole number of 32 bit wraps, so the seed comes back untouched.
    assert effective_noise_seed(params, 2**34) == 2**31 - 1
    # One step past the sign bit, which a 31 bit mask would fold back to zero.
    assert effective_noise_seed(params, 1) == 2**31
    assert effective_noise_seed(params, 2**32 + 1) == 2**31


def test_render_frame_passes_noise_mode_to_synthesis():
    seen = []

    def synthesis(ws, noise_mode):
        import torch

        seen.append(noise_mode)
        return torch.zeros([1, 3, 8, 8])

    model = _fake_model(synthesis)
    model.render_frame(render_params(noise_seed=5), 0)
    model.render_frame(render_params(noise_enabled=False), 1)
    assert seen == ["random", "none"]


def test_render_frame_seeds_torch_with_effective_seed(monkeypatch):
    import torch

    seeds = []
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    model = _fake_model(_zeros_synthesis)
    model.render_frame(render_params(noise_seed=11, noise_anim=True), 4)
    assert seeds == [15]


@pytest.mark.parametrize(
    "changes",
    [
        {"noise_enabled": False},
        {"noise_seed": 0, "noise_anim": False},
    ],
)
def test_render_frame_leaves_the_global_rng_alone_outside_random_noise(
    monkeypatch, changes
):
    # The network never samples in "none" or "const" mode, so reseeding there
    # only pins torch's global stream for whatever else runs on this thread.
    import torch

    seeds = []
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    model = _fake_model(_zeros_synthesis)
    model.render_frame(render_params(**changes), 3)
    assert seeds == []


def test_global_noise_applied_only_to_modules_that_declare_it():
    noisy = _NoisyModule()
    plain = _PlainModule()
    model = _fake_model(_zeros_synthesis, modules=[noisy, plain])
    model.render_frame(render_params(global_noise=0.25), 0)
    assert noisy.global_noise == 0.25
    assert not hasattr(plain, "global_noise")


def test_global_noise_walk_is_skipped_when_value_is_unchanged():
    noisy = _NoisyModule()
    model = _fake_model(_zeros_synthesis, modules=[noisy])
    model.render_frame(render_params(global_noise=0.25), 0)
    assert model.G.module_walks == 1
    model.render_frame(render_params(global_noise=0.25), 1)
    assert model.G.module_walks == 1
    model.render_frame(render_params(global_noise=0.75), 2)
    assert model.G.module_walks == 2
    assert noisy.global_noise == 0.75
