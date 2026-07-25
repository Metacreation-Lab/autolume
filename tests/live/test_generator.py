import logging
import threading
import time

import pytest

from autolume.live.core.generator import (
    ModelHost,
    ModelInfo,
    corner_seeds,
    effective_noise_seed,
    noise_mode,
    slerp,
)
from autolume.live.core.params import ControlState, Keyframe, to_render_params


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
        self.z_dim = 4
        self.num_ws = 2


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


def test_model_host_publishes_model_info_on_successful_load():
    host = ModelHost(loader=FakeModel)
    assert host.info_store.snapshot() is None
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.info_store.snapshot() == ModelInfo(
        pkl_path="/tmp/a.pkl", z_dim=4, num_ws=2
    )
    host.stop()


def test_model_host_clears_model_info_on_load_failure():
    should_fail = threading.Event()

    def loader(path):
        if should_fail.is_set():
            raise RuntimeError("bad pkl")
        return FakeModel(path)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.info_store.snapshot() is not None

    should_fail.set()
    host.request_load("/tmp/bad.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is not None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.info_store.snapshot() is None
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
    num_ws = 2

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


# --- vec and loop modes -----------------------------------------------------
#
# These fixtures use z_dim == w_dim (unlike the fixtures above, which keep
# them different to prove shape routing does not care), so the vec/W tests
# below can exercise realistic single-vector-broadcast behavior.


class _RecordingMapping:
    def __init__(self, w_dim, num_ws):
        import torch

        self.w_avg = torch.zeros([w_dim])
        self.c_dim = 0
        self._num_ws = num_ws
        self.calls = []

    def __call__(self, z, c, truncation_psi):
        self.calls.append((z.clone(), truncation_psi))
        return z.unsqueeze(1).repeat(1, self._num_ws, 1) * truncation_psi


class _VecG:
    def __init__(self, z_dim, num_ws, synthesis):
        self.z_dim = z_dim
        self.num_ws = num_ws
        self.mapping = _RecordingMapping(z_dim, num_ws)
        self.synthesis = synthesis

    def modules(self):
        return []


def _recording_synthesis(sink):
    def synthesis(ws, noise_mode):
        import torch

        sink.append(ws.clone())
        return torch.zeros([1, 3, 4, 4])

    return synthesis


def _vec_model(z_dim=4, num_ws=3, synthesis=None):
    import torch

    from autolume.live.core.generator import LoadedModel

    return LoadedModel(
        "/tmp/vec.pkl", _VecG(z_dim, num_ws, synthesis or _zeros_synthesis), torch.device("cpu")
    )


def test_vec_projected_passes_truncation_into_mapping():
    model = _vec_model(z_dim=4, num_ws=3)
    params = render_params(
        vector_mode=True,
        latent_project=True,
        latent_vec=(1.0, 2.0, 3.0, 4.0),
        truncation_psi=0.5,
    )
    model.render_frame(params, 0)
    assert len(model.G.mapping.calls) == 1
    z, psi = model.G.mapping.calls[0]
    assert psi == 0.5
    assert z.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_vec_unprojected_exact_length_passes_rows_through():
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    rows = (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
    params = render_params(
        vector_mode=True, latent_project=False, latent_vec=rows
    )
    model.render_frame(params, 0)
    assert sink[-1][0].tolist() == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]


def test_vec_unprojected_too_many_rows_truncates_to_num_ws():
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    rows = (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
    params = render_params(
        vector_mode=True, latent_project=False, latent_vec=rows
    )
    model.render_frame(params, 0)
    assert sink[-1][0].tolist() == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]


def test_vec_unprojected_too_few_rows_repeats_last_row():
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    rows = (1, 1, 1, 1, 2, 2, 2, 2)
    params = render_params(
        vector_mode=True, latent_project=False, latent_vec=rows
    )
    model.render_frame(params, 0)
    assert sink[-1][0].tolist() == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ]


def test_empty_vector_fallback_is_deterministic_and_logged_once(caplog):
    import numpy as np
    import torch

    model = _vec_model(z_dim=4, num_ws=2)
    params = render_params(vector_mode=True, latent_project=True, latent_vec=())
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 0)
        model.render_frame(params, 1)
    calls = model.G.mapping.calls
    assert len(calls) == 2
    expected = np.random.RandomState(0).randn(4).astype(np.float32)
    assert torch.allclose(calls[0][0][0], torch.from_numpy(expected))
    assert torch.allclose(calls[1][0][0], torch.from_numpy(expected))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_loop_endpoint_cache_reuses_mapping_per_endpoint():
    model = _vec_model(z_dim=4, num_ws=3)
    kf0 = Keyframe("vec", vec=(1.0, 2.0, 3.0, 4.0), project=True)
    kf1 = Keyframe("vec", vec=(5.0, 6.0, 7.0, 8.0), project=True)
    params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=0.5,
        truncation_psi=0.6,
    )
    model.render_frame(params, 0)
    model.render_frame(params, 1)
    assert len(model.G.mapping.calls) == 2


def test_loop_alpha_zero_matches_previous_keyframe():
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("seed", seed_x=1.0, seed_y=0.0)
    loop_params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=0.0,
        truncation_psi=0.7,
    )
    model.render_frame(loop_params, 0)
    loop_ws = sink[-1]

    seed_params = render_params(latent_x=0.0, latent_y=0.0, truncation_psi=0.7)
    model.render_frame(seed_params, 0)
    seed_ws = sink[-1]

    assert torch.allclose(loop_ws, seed_ws, atol=1e-5)


def test_loop_alpha_one_matches_current_keyframe():
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("seed", seed_x=1.0, seed_y=0.0)
    loop_params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=1.0,
        truncation_psi=0.7,
    )
    model.render_frame(loop_params, 0)
    loop_ws = sink[-1]

    seed_params = render_params(latent_x=1.0, latent_y=0.0, truncation_psi=0.7)
    model.render_frame(seed_params, 0)
    seed_ws = sink[-1]

    assert torch.allclose(loop_ws, seed_ws, atol=1e-5)


def test_loop_index_zero_wraps_to_last_keyframe():
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("seed", seed_x=1.0, seed_y=0.0)
    kf2 = Keyframe("seed", seed_x=2.0, seed_y=0.0)
    loop_params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1, kf2),
        loop_index=0,
        loop_alpha=0.0,
        truncation_psi=0.7,
    )
    model.render_frame(loop_params, 0)
    loop_ws = sink[-1]

    seed_params = render_params(latent_x=2.0, latent_y=0.0, truncation_psi=0.7)
    model.render_frame(seed_params, 0)
    seed_ws = sink[-1]

    assert torch.allclose(loop_ws, seed_ws, atol=1e-5)


def test_slerp_of_orthogonal_unit_vectors_has_unit_norm():
    import torch

    w0 = torch.tensor([1.0, 0.0])
    w1 = torch.tensor([0.0, 1.0])
    result = slerp(0.5, w0, w1)
    assert abs(float(result.norm()) - 1.0) < 1e-5


def test_slerp_falls_back_to_lerp_when_near_colinear():
    import torch

    w0 = torch.tensor([1.0, 0.0])
    w1 = torch.tensor([1.0, 1e-5])
    alpha = 0.3
    result = slerp(alpha, w0, w1)
    expected = w0 + alpha * (w1 - w0)
    assert torch.allclose(result, expected, atol=1e-6)


def test_slerp_alpha_bounds_reproduce_endpoints():
    import torch

    w0 = torch.tensor([1.0, 0.0, 0.0])
    w1 = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(slerp(0.0, w0, w1), w0, atol=1e-6)
    assert torch.allclose(slerp(1.0, w0, w1), w1, atol=1e-6)
