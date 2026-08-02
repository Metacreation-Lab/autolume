import numpy as np
import pytest
import torch

from features.extraction import ExtractionCancelled, ExtractionConfig, extract


class FakeMapping(torch.nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.lin = torch.nn.Linear(z_dim, w_dim, bias=False)

    def forward(self, z, c=None):
        w = self.lin(z)
        return torch.stack([w, w], dim=1)  # (B, num_ws=2, w_dim)


class FakeGenerator(torch.nn.Module):
    def __init__(self, z_dim=16, w_dim=16):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.mapping = FakeMapping(z_dim, w_dim)


class FakeConditionalMapping(torch.nn.Module):
    def __init__(self, z_dim, w_dim, c_dim):
        super().__init__()
        self.c_dim = c_dim
        self.lin = torch.nn.Linear(z_dim, w_dim, bias=False)

    def forward(self, z, c=None):
        # Mimic StyleGAN's assert_shape: touching c.ndim raises on None.
        assert c.ndim == 2 and c.shape[0] == z.shape[0] and c.shape[1] == self.c_dim
        w = self.lin(z)
        return torch.stack([w, w], dim=1)


class FakeConditionalGenerator(torch.nn.Module):
    def __init__(self, z_dim=16, w_dim=16, c_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.mapping = FakeConditionalMapping(z_dim, w_dim, c_dim)


CFG = ExtractionConfig(n_components=4, n_samples=2048, batch_size=256)


def make_G():
    torch.manual_seed(0)  # deterministic Linear init
    return FakeGenerator()


def test_shapes_dtypes_and_unit_norm():
    dirs, sigmas = extract(make_G(), CFG)
    assert dirs.shape == (4, 16)
    assert dirs.dtype == np.float32
    assert sigmas.shape == (4,)
    assert sigmas.dtype == np.float32
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-5)


def test_sigmas_positive_and_descending():
    _dirs, sigmas = extract(make_G(), CFG)
    assert (sigmas > 0).all()
    assert np.array_equal(sigmas, np.sort(sigmas)[::-1])


def test_deterministic_across_runs():
    d1, s1 = extract(make_G(), CFG)
    d2, s2 = extract(make_G(), CFG)
    assert np.array_equal(d1, d2)
    assert np.array_equal(s1, s2)


def test_different_seed_differs():
    other = ExtractionConfig(n_components=4, n_samples=2048, batch_size=256, seed=1)
    assert not np.array_equal(extract(make_G(), CFG)[0], extract(make_G(), other)[0])


def test_n_components_clamped_to_w_dim():
    cfg = ExtractionConfig(n_components=99, n_samples=2048, batch_size=256)
    dirs, sigmas = extract(make_G(), cfg)
    assert dirs.shape == (16, 16)
    assert sigmas.shape == (16,)


def test_sign_stabilization():
    dirs, _ = extract(make_G(), CFG)
    largest = dirs[np.arange(len(dirs)), np.abs(dirs).argmax(axis=1)]
    assert (largest > 0).all()


def test_progress_is_monotonic_and_completes():
    fracs = []
    extract(make_G(), CFG, progress_cb=lambda f, m: fracs.append(f))
    assert fracs == sorted(fracs)
    assert fracs[-1] == 1.0


def test_cancel_raises():
    with pytest.raises(ExtractionCancelled):
        extract(make_G(), CFG, cancel_check=lambda: True)


def test_extract_restores_numpy_rng_state():
    np.random.seed(12345)
    expected = np.random.random_sample(4)
    np.random.seed(12345)
    extract(make_G(), CFG)
    assert np.array_equal(np.random.random_sample(4), expected)


def test_conditional_model_receives_zero_labels():
    torch.manual_seed(0)
    dirs, _ = extract(FakeConditionalGenerator(), CFG)
    assert dirs.shape == (4, 16)
