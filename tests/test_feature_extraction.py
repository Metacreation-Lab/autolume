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


CFG = ExtractionConfig(n_features=4, n_samples=2048, batch_size=256)


def make_G():
    torch.manual_seed(0)  # deterministic Linear init
    return FakeGenerator()


def test_shape_dtype_and_unit_norm():
    d = extract(make_G(), CFG)
    assert d.shape == (4, 16)
    assert d.dtype == np.float32
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-5)


def test_deterministic_across_runs():
    assert np.array_equal(extract(make_G(), CFG), extract(make_G(), CFG))


def test_different_seed_differs():
    other = ExtractionConfig(n_features=4, n_samples=2048, batch_size=256, seed=1)
    assert not np.array_equal(extract(make_G(), CFG), extract(make_G(), other))


def test_n_features_clamped_to_w_dim():
    cfg = ExtractionConfig(n_features=99, n_samples=2048, batch_size=256)
    assert extract(make_G(), cfg).shape == (16, 16)


def test_sign_stabilization():
    d = extract(make_G(), CFG)
    largest = d[np.arange(len(d)), np.abs(d).argmax(axis=1)]
    assert (largest > 0).all()


def test_progress_is_monotonic_and_completes():
    fracs = []
    extract(make_G(), CFG, progress_cb=lambda f, m: fracs.append(f))
    assert fracs == sorted(fracs)
    assert fracs[-1] == 1.0


def test_cancel_raises():
    with pytest.raises(ExtractionCancelled):
        extract(make_G(), CFG, cancel_check=lambda: True)


def test_batch_estimator_path():
    cfg = ExtractionConfig(estimator="ipca", n_features=4, n_samples=2048, batch_size=256)
    assert extract(make_G(), cfg).shape == (4, 16)


def test_extract_restores_numpy_rng_state():
    np.random.seed(12345)
    expected = np.random.random_sample(4)
    np.random.seed(12345)
    extract(make_G(), CFG)
    assert np.array_equal(np.random.random_sample(4), expected)


def test_conditional_model_receives_zero_labels():
    torch.manual_seed(0)
    d = extract(FakeConditionalGenerator(), CFG)
    assert d.shape == (4, 16)
