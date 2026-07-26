import logging
import threading
import time

import pytest

from autolume.live.core.generator import (
    LayerInfo,
    ModelHost,
    ModelInfo,
    channel_window,
    corner_seeds,
    derive_float_image,
    effective_noise_seed,
    noise_mode,
    slerp,
    to_uint8_frame,
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


def test_vec_unprojected_empty_vector_fallback_is_deterministic_and_logged_once(
    caplog,
):
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    params = render_params(vector_mode=True, latent_project=False, latent_vec=())
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 0)
        model.render_frame(params, 1)
    first, second = sink[-2].tolist(), sink[-1].tolist()
    assert first == second
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


def test_loop_endpoint_cache_is_bounded_to_the_cap():
    model = _vec_model(z_dim=4, num_ws=3)
    keyframes = tuple(
        Keyframe("vec", vec=(float(i),) * 4, project=True) for i in range(6)
    )
    for index in range(len(keyframes)):
        params = render_params(
            loop_active=True,
            keyframes=keyframes,
            loop_index=index,
            loop_alpha=0.5,
            truncation_psi=0.6,
        )
        model.render_frame(params, index)
    assert len(model._keyframe_w_cache) <= 4


def test_loop_vec_keyframe_wrong_length_falls_back_without_raising(caplog):
    model = _vec_model(z_dim=4, num_ws=3)
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("vec", vec=(1.0, 2.0), project=True)
    params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=0.5,
        truncation_psi=0.6,
    )
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 0)
    assert len(model.G.mapping.calls) == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


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


# --- layer catalog enumeration ------------------------------------------


def _block(channels, height, width):
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def forward(self, ws):
            return torch.zeros(ws.shape[0], channels, height, width)

    return _Block()


def _fake_synthesis_module():
    import torch.nn as nn

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            # conv1 is deliberately non-square: a width/height transposition
            # bug would still pass a square-only fixture.
            self.conv1 = _block(channels=8, height=4, width=6)
            self.torgb = _block(channels=3, height=8, width=8)

        def forward(self, ws, noise_mode="const"):
            self.conv1(ws)
            return self.torgb(ws)

    return _Synthesis()


def _failing_synthesis_module():
    import torch.nn as nn

    class _Boom(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _block(channels=8, height=4, width=6)

        def forward(self, ws, noise_mode="const"):
            raise RuntimeError("exotic architecture")

    return _Boom()


def _group_cnn_synthesis_module():
    """A single child whose output is 5D: `(N, C, G, H, W)`, the group-CNN
    layout the 4D/5D filter is meant to also accept."""
    import torch
    import torch.nn as nn

    class _GroupBlock(nn.Module):
        def forward(self, ws):
            return torch.zeros(ws.shape[0], 6, 3, 4, 5)

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            self.group_conv = _GroupBlock()

        def forward(self, ws, noise_mode="const"):
            return self.group_conv(ws)

    return _Synthesis()


def test_enumerate_layers_records_names_shapes_and_order():
    model = _fake_model(_fake_synthesis_module())

    layers = model.enumerate_layers()

    assert layers == (
        LayerInfo(name="conv1", channels=8, width=6, height=4),
        LayerInfo(name="torgb", channels=3, width=8, height=8),
        LayerInfo(name="output", channels=3, width=8, height=8),
    )


def test_enumerate_layers_5d_output_channels_read_axis_1_not_axis_minus_3():
    # 5D layout is (N, C, G, H, W): axis -3 is the group count (3), not the
    # channel count (6). channels must come from axis 1.
    model = _fake_model(_group_cnn_synthesis_module())

    layers = model.enumerate_layers()

    assert layers == (
        LayerInfo(name="group_conv", channels=6, width=5, height=4),
        LayerInfo(name="output", channels=6, width=5, height=4),
    )


def test_enumerate_layers_removes_hooks_afterward():
    import torch

    synthesis = _fake_synthesis_module()
    model = _fake_model(synthesis)

    model.enumerate_layers()

    for module in synthesis.modules():
        assert len(module._forward_hooks) == 0

    # A second call to the same (now unhooked) synthesis must not raise.
    synthesis(torch.zeros(1, 2, 8))


def test_enumerate_layers_failure_yields_empty_catalog_and_logs_once(caplog):
    synthesis = _failing_synthesis_module()
    model = _fake_model(synthesis)

    with caplog.at_level(logging.WARNING):
        layers = model.enumerate_layers()

    assert layers == ()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    # The safety-critical half of the finally contract: a raising dry pass
    # must not leave hooks on a model about to render.
    for module in synthesis.modules():
        assert len(module._forward_hooks) == 0


def test_model_host_publishes_empty_layer_catalog_when_enumeration_fails():
    import torch

    from autolume.live.core.generator import LoadedModel

    def loader(path):
        return LoadedModel(
            path, _FakeG(_failing_synthesis_module()), torch.device("cpu")
        )

    host = ModelHost(loader=loader)
    host.request_load("/tmp/exotic.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    info = host.info_store.snapshot()
    assert info is not None
    assert info.layers == ()
    assert host.current() is not None
    assert host.error() is None
    host.stop()


def test_model_host_publishes_empty_layer_catalog_when_wrapper_enumeration_raises(
    caplog,
):
    """`_model_info`'s own guard, not `LoadedModel.enumerate_layers`'s.

    A future wrapper generator (a mixed-model network, say) need not be a
    `LoadedModel`: it only has to duck-type `z_dim`/`num_ws`/`enumerate_layers`.
    If its `enumerate_layers` raises, `_model_info` must still publish an
    empty catalog rather than failing the whole load.
    """

    class _WrapperModel:
        def __init__(self, path):
            self.pkl_path = path
            self.z_dim = 4
            self.num_ws = 2

        def enumerate_layers(self):
            raise RuntimeError("wrapper enumeration exploded")

    host = ModelHost(loader=_WrapperModel)
    with caplog.at_level(logging.WARNING):
        host.request_load("/tmp/wrapper.pkl")
        deadline = time.monotonic() + 2.0
        while host.info_store.snapshot() is None and time.monotonic() < deadline:
            time.sleep(0.005)

    info = host.info_store.snapshot()
    assert info is not None
    assert info.layers == ()
    assert host.current() is not None
    assert host.error() is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
    host.stop()


def test_model_host_publishes_layer_catalog_on_successful_load():
    import torch

    from autolume.live.core.generator import LoadedModel

    def loader(path):
        return LoadedModel(
            path, _FakeG(_fake_synthesis_module()), torch.device("cpu")
        )

    host = ModelHost(loader=loader)
    host.request_load("/tmp/good.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    info = host.info_store.snapshot()
    assert info is not None
    assert [layer.name for layer in info.layers] == ["conv1", "torgb", "output"]
    host.stop()


# --- image derivation ----------------------------------------------------


def _activation(channels, height=1, width=2):
    import torch

    count = channels * height * width
    return torch.arange(count, dtype=torch.float32).reshape(channels, height, width)


def test_channel_window_takes_three_consecutive_channels():
    window = channel_window(_activation(5), base_channel=1, grayscale=False)
    assert window.tolist() == [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]]


def test_channel_window_clamps_the_base_to_keep_three_channels_in_range():
    window = channel_window(_activation(5), base_channel=9, grayscale=False)
    assert window.tolist() == [[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]


def test_channel_window_grayscale_replicates_one_channel():
    window = channel_window(_activation(5), base_channel=3, grayscale=True)
    assert window.tolist() == [[[6.0, 7.0]], [[6.0, 7.0]], [[6.0, 7.0]]]


def test_channel_window_grayscale_clamps_the_base_to_the_last_channel():
    window = channel_window(_activation(5), base_channel=99, grayscale=True)
    assert window.tolist() == [[[8.0, 9.0]], [[8.0, 9.0]], [[8.0, 9.0]]]


@pytest.mark.parametrize("channels", [1, 2])
def test_channel_window_falls_back_to_one_channel_when_three_do_not_exist(channels):
    window = channel_window(_activation(channels), base_channel=1, grayscale=False)
    last = _activation(channels)[channels - 1].tolist()
    assert window.tolist() == [last, last, last]


def _signed_activation():
    import torch

    return torch.tensor([[[1.0, -2.0]], [[0.0, 0.0]], [[3.0, 4.0]]])


def test_derive_normalizes_each_channel_by_its_own_max_absolute_value():
    image = derive_float_image(_signed_activation(), render_params(img_normalize=True))
    assert image.tolist() == [[[0.5, -1.0]], [[0.0, 0.0]], [[0.75, 1.0]]]


def test_derive_scales_by_decibels():
    image = derive_float_image(_signed_activation(), render_params(img_scale_db=20.0))
    assert image.tolist() == [[[10.0, -20.0]], [[0.0, 0.0]], [[30.0, 40.0]]]


def test_derive_normalizes_before_scaling():
    # Normalization is scale invariant, so the other order would swallow the
    # decibel gain entirely and leave the normalized values behind.
    image = derive_float_image(
        _signed_activation(), render_params(img_normalize=True, img_scale_db=20.0)
    )
    assert image.tolist() == [[[5.0, -10.0]], [[0.0, 0.0]], [[7.5, 10.0]]]


def test_derive_leaves_a_flat_channel_alone_instead_of_dividing_by_zero():
    import torch

    image = derive_float_image(torch.zeros([3, 1, 2]), render_params(img_normalize=True))
    assert image.tolist() == [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
    assert bool(torch.isfinite(image).all())


def test_uint8_frame_maps_the_signed_unit_range_onto_the_byte_range():
    import torch

    image = torch.tensor([[[-1.0, -0.5, 0.0, 0.5, 1.0]]]).repeat(3, 1, 1)
    frame = to_uint8_frame(image)
    assert frame.shape == (1, 5, 3)
    assert frame.dtype.name == "uint8"
    assert frame[0, :, 0].tolist() == [0, 64, 128, 191, 255]


def test_uint8_frame_clamps_out_of_range_values():
    import torch

    image = torch.tensor([[[-4.0, 4.0]]]).repeat(3, 1, 1)
    frame = to_uint8_frame(image)
    assert frame[0, :, 0].tolist() == [0, 255]


def test_uint8_frame_is_contiguous_after_the_channel_permute():
    import torch

    frame = to_uint8_frame(torch.zeros([3, 2, 4]))
    assert frame.flags["C_CONTIGUOUS"]


def _channel_ramp_synthesis(channels=5):
    def synthesis(ws, noise_mode):
        import torch

        ramp = torch.arange(channels, dtype=torch.float32) / channels
        return ramp.reshape(1, channels, 1, 1).repeat(1, 1, 1, 2)

    return synthesis


def test_render_frame_derives_from_the_selected_channels():
    model = _fake_model(_channel_ramp_synthesis())
    frame = model.render_frame(render_params(base_channel=1), 0)
    # Channels 1, 2 and 3 of the ramp: 0.2, 0.4 and 0.6.
    assert frame[0, 0].tolist() == [153, 179, 204]


def test_render_frame_grayscale_replicates_one_channel():
    model = _fake_model(_channel_ramp_synthesis())
    frame = model.render_frame(render_params(base_channel=1, grayscale=True), 0)
    assert frame.shape == (1, 2, 3)
    assert frame[0, 0].tolist() == [153, 153, 153]
