import numpy as np
import pytest

from features.estimators import get_estimator

NAMES = ["pca", "ipca", "fbpca", "ica", "spca"]


def make_data(n=400, dim=8):
    # Gaussian cloud whose dominant variance axis is coordinate 0.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X[:, 0] *= 10.0
    return X - X.mean(axis=0, keepdims=True)


def make_ica_data(n=1000, dim=8):
    # Uniform (non-Gaussian) sources so ICA is identifiable: independence is
    # not equivalent to uncorrelatedness here, unlike for Gaussian data.
    # A larger sample than the shared Gaussian fixture is needed for FastICA
    # to reliably converge on the scaled axis.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(n, dim)).astype(np.float32)
    X[:, 0] *= 10.0
    return X - X.mean(axis=0, keepdims=True)


@pytest.mark.parametrize("name", NAMES)
def test_estimator_contract(name):
    est = get_estimator(name, 3, 1.0)
    X = make_data()
    if est.batch_support:
        assert est.fit_partial(X)
    else:
        est.fit(X)
    comps, stdev, var_ratio = est.get_components()
    assert comps.shape == (3, 8)
    assert stdev.shape == (3,)
    # ICA is unidentifiable on Gaussian data (independence == uncorrelatedness),
    # so it has no reason to recover the scaled axis; checked separately below.
    if name == "ica":
        return
    top = comps[0] / np.linalg.norm(comps[0])
    assert abs(top[0]) > 0.9


def test_ica_alignment_on_non_gaussian_data():
    est = get_estimator("ica", 3, 1.0)
    X = make_ica_data()
    est.fit(X)
    comps, stdev, var_ratio = est.get_components()
    assert comps.shape == (3, 8)
    assert stdev.shape == (3,)
    top = comps[0] / np.linalg.norm(comps[0])
    assert abs(top[0]) > 0.9


def test_unknown_estimator_raises():
    with pytest.raises(RuntimeError):
        get_estimator("nope", 3, 1.0)
