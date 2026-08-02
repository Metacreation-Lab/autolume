import pytest
import torch

from drag.engine import DragEngine, motion_loss, track_points


@pytest.fixture(scope='module')
def tiny_g():
    from architectures import custom_stylegan2
    torch.manual_seed(0)
    G = custom_stylegan2.Generator(
        z_dim=64, w_dim=64, c_dim=0, img_resolution=64, img_channels=3,
        mapping_kwargs={'num_layers': 2},
        synthesis_kwargs={'channel_base': 1024, 'channel_max': 64, 'num_fp16_res': 0})
    return G.eval().requires_grad_(False)


@pytest.fixture
def w0(tiny_g):
    with torch.no_grad():
        return tiny_g.mapping(torch.randn(1, tiny_g.z_dim), None)


def make_engine(tiny_g, w0):
    return DragEngine(tiny_g, w0, device='cpu')


def test_track_points_finds_shifted_feature():
    feat = torch.zeros(1, 8, 64, 64)
    ref = torch.arange(8, dtype=torch.float32)
    feat[0, :, 23, 22] = ref
    refs = [ref.reshape(1, 8)]
    assert track_points(feat, refs, [[20, 20]], r2_px=6) == [[23, 22]]


def test_motion_loss_gradients_and_convergence():
    feat = torch.randn(1, 4, 32, 32, requires_grad=True)
    loss, converged = motion_loss(feat, [[10, 10]], [[20, 10]], r1_px=3, stop_px=2.0)
    assert not converged
    assert loss.item() > 0 or loss.requires_grad
    loss.backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0

    feat2 = torch.randn(1, 4, 32, 32, requires_grad=True)
    loss2, converged2 = motion_loss(feat2, [[10, 10]], [[10, 10]], r1_px=3, stop_px=2.0)
    assert converged2
    assert loss2.item() == 0


def test_step_updates_only_trainable_layers(tiny_g, w0):
    engine = make_engine(tiny_g, w0)
    try:
        points = [[32, 32]]
        for _ in range(3):
            points, converged = engine.step(points, [[44, 32]])
        n = engine.trainable_ws
        assert not torch.equal(engine.w[:, :n], w0[:, :n])
        assert torch.equal(engine.w[:, n:], w0[:, n:])
        cw = engine.current_w()
        assert cw.shape == w0.shape
        assert torch.equal(cw[:, n:], w0[:, n:])
        assert all(isinstance(v, int) for p in points for v in p)
    finally:
        engine.close()


def test_step_converged_leaves_w_unchanged(tiny_g, w0):
    engine = make_engine(tiny_g, w0)
    try:
        _, converged = engine.step([[32, 32]], [[32, 32]])
        assert converged
        assert torch.equal(engine.w, w0)
    finally:
        engine.close()


def test_mask_guard(tiny_g, w0):
    # An untouched (all ones) mask must not constrain the optimization.
    engine_a = make_engine(tiny_g, w0)
    engine_b = make_engine(tiny_g, w0)
    try:
        pa, _ = engine_a.step([[32, 32]], [[44, 32]], mask=None)
        pb, _ = engine_b.step([[32, 32]], [[44, 32]], mask=torch.ones(64, 64))
        assert torch.equal(engine_a.w, engine_b.w)
        assert pa == pb
    finally:
        engine_a.close()
        engine_b.close()
