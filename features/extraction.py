"""Deterministic latent direction extraction.

Method from GANSpace (Erik Härkönen et al.): PCA over w-space samples of the
mapping network. https://github.com/harskish/ganspace
The fbpca call parameters follow their reference implementation.
"""
from dataclasses import dataclass

import fbpca
import numpy as np
import torch


class ExtractionCancelled(Exception):
    pass


@dataclass(frozen=True)
class ExtractionConfig:
    n_components: int = 64
    seed: int = 0
    n_samples: int = 300_000
    batch_size: int = 1024


def extract(G, config=None, progress_cb=None, cancel_check=None):
    """Sample the mapping network and return its principal direction bank.

    Returns (directions, sigmas): a (k, w_dim) float32 array of unit-norm
    directions in w space sorted by explained variance, each row's
    largest-magnitude coordinate forced positive so repeated runs cannot flip
    slider polarity, and a (k,) float32 array of the standard deviation of
    the samples along each direction (the natural per-component scale).
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
    k = min(config.n_components, w_dim)

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
    progress(0.75, "Fitting directions")
    samples -= samples.mean(axis=0, keepdims=True)

    # fbpca has no seed parameter and consumes numpy's global RNG; restore the
    # caller's RNG state afterwards so extraction has no side effects on it.
    state = np.random.get_state()
    np.random.seed(config.seed)
    try:
        _u, _s, dirs = fbpca.pca(samples, k=k, raw=True, n_iter=2, l=2 * k)
    finally:
        np.random.set_state(state)

    # SVD returns rows in descending singular value order, which is exactly
    # descending explained variance of the centered samples.
    dirs = np.array(dirs, dtype=np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    flip = dirs[np.arange(len(dirs)), np.abs(dirs).argmax(axis=1)] < 0
    dirs[flip] *= -1
    sigmas = (samples @ dirs.T).std(axis=0).astype(np.float32)
    progress(1.0, "Done")
    return dirs, sigmas
