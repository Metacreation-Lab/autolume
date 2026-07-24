"""Deterministic latent direction extraction.

Derived from GANSpace (Erik Härkönen et al., Apache-2.0):
https://github.com/harskish/ganspace
"""
from dataclasses import dataclass

import numpy as np
import torch

from features import estimators


class ExtractionCancelled(Exception):
    pass


@dataclass(frozen=True)
class ExtractionConfig:
    estimator: str = "fbpca"
    n_features: int = 8
    seed: int = 0
    n_samples: int = 300_000
    batch_size: int = 1024
    sparsity: float = 1.0


def extract(G, config=None, progress_cb=None, cancel_check=None):
    """Sample the mapping network and fit an estimator to find salient directions.

    Returns a (n_features, w_dim) float32 array of unit-norm directions in w
    space, sorted by explained variance, with each row's largest-magnitude
    coordinate forced positive so repeated runs cannot flip slider polarity.
    """
    config = config or ExtractionConfig()

    def progress(fraction, message):
        if progress_cb is not None:
            progress_cb(fraction, message)

    def check_cancel():
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled()

    device = next(G.mapping.parameters()).device
    w_dim = int(G.w_dim)
    n_features = min(config.n_features, w_dim)
    est = estimators.get_estimator(config.estimator, n_features, config.sparsity)

    B = config.batch_size
    n_batches = max(1, config.n_samples // B)
    gen = torch.Generator().manual_seed(config.seed)
    samples = np.empty((n_batches * B, w_dim), dtype=np.float32)

    # Conditional models require labels; use zero labels like the renderer's
    # seed path (StyleGAN's mapping crashes on c=None when c_dim > 0).
    c_dim = int(getattr(G, "c_dim", 0) or 0)
    c = torch.zeros(B, c_dim, device=device) if c_dim > 0 else None

    with torch.no_grad():
        for i in range(n_batches):
            check_cancel()
            progress(0.7 * i / n_batches, f"Sampling latents {i + 1} of {n_batches}")
            z = torch.randn(B, G.z_dim, generator=gen).to(device)
            w = G.mapping(z, c)[:, 0]
            samples[i * B:(i + 1) * B] = w.cpu().numpy()

    check_cancel()
    # fbpca has no seed parameter and consumes numpy's global RNG; restore the
    # caller's RNG state afterwards so extraction has no side effects on it.
    state = np.random.get_state()
    np.random.seed(config.seed)
    try:
        if est.batch_support:
            chunk = 10 * B
            for j in range(0, len(samples), chunk):
                check_cancel()
                progress(0.7 + 0.3 * j / len(samples), f"Fitting {config.estimator}")
                if not est.fit_partial(samples[j:j + chunk]):
                    break
        else:
            progress(0.75, f"Fitting {config.estimator}")
            samples -= samples.mean(axis=0, keepdims=True)
            est.fit(samples)

        comps, _stdev, _var_ratio = est.get_components()
        dirs = np.array(comps[:n_features], dtype=np.float32)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        flip = dirs[np.arange(len(dirs)), np.abs(dirs).argmax(axis=1)] < 0
        dirs[flip] *= -1
    finally:
        np.random.set_state(state)
    progress(1.0, "Done")
    return dirs
