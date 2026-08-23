"""Real-ESRGAN upscaling for dataset preprocessing.

The model is realesr-general-x4v3 with the official denoise blend:
strength 1.0 is the plain model (strongest denoise), lower values
interpolate toward the weak-denoise wdn weights, which keep more grain.

Network architectures come from spandrel, which detects them from the
weights, so this package carries no model definitions of its own.
"""

import logging

import numpy as np
import spandrel
import torch

from upscale.weights import ensure_weight
from utils.device_utils import get_device

logger = logging.getLogger(__name__)

MODEL_KEY = "Balance"
WDN_MODEL_KEY = "BalanceWDN"
MAX_PASSES = 3


def required_weights(denoise):
    """WEIGHTS keys the given settings need on disk."""
    if denoise >= 1.0:
        return [MODEL_KEY]
    if denoise <= 0.0:
        return [WDN_MODEL_KEY]
    return [MODEL_KEY, WDN_MODEL_KEY]


def needs_upscale(width, height, target_size):
    """True when an image this size gets enlarged to reach target_size.

    The short side is the limit in both resize modes: stretch enlarges both
    axes to the target and center crop resizes the short side to the target.
    """
    if not width or not height:
        return False
    return min(width, height) < target_size


def upscale_passes(width, height, target_size):
    """Number of 4x passes for the short side to reach target_size (capped)."""
    if not needs_upscale(width, height, target_size):
        return 0
    short = min(width, height)
    passes = 0
    while short < target_size and passes < MAX_PASSES:
        short *= 4
        passes += 1
    return passes


def blend_state_dicts(sd_a, sd_b, alpha):
    """alpha * sd_a + (1 - alpha) * sd_b, key by key."""
    return {k: alpha * sd_a[k] + (1.0 - alpha) * sd_b[k] for k in sd_a}


def _finalize(descriptor):
    """Move a spandrel descriptor to the inference device, return its module.

    On GPU the forward pass runs in fp16 when the architecture supports it,
    which roughly halves activation memory and keeps the model inside the
    VRAM budget instead of spilling to system RAM (a ~13x slowdown on
    Windows/CUDA). CPU stays fp32, where fp16 buys nothing.
    """
    device = get_device()
    descriptor = descriptor.to(device).eval()
    if device.type in ("cuda", "mps") and descriptor.supports_half:
        descriptor = descriptor.half()
    return descriptor.model


def load_upscaler(denoise=0.0, progress_cb=None, cancel_event=None):
    """Build the dataset upscaler. Returns None if a download was cancelled."""
    # Blend of the x4v3 endpoints, loading only the files the denoise value
    # actually uses.
    if denoise >= 1.0:
        path = ensure_weight(MODEL_KEY, progress_cb, cancel_event)
        if path is None:
            return None
        return _finalize(spandrel.ModelLoader().load_from_file(path))
    if denoise <= 0.0:
        wdn_path = ensure_weight(WDN_MODEL_KEY, progress_cb, cancel_event)
        if wdn_path is None:
            return None
        return _finalize(spandrel.ModelLoader().load_from_file(wdn_path))
    path = ensure_weight(MODEL_KEY, progress_cb, cancel_event)
    if path is None:
        return None
    wdn_path = ensure_weight(WDN_MODEL_KEY, progress_cb, cancel_event)
    if wdn_path is None:
        return None
    sd = torch.load(path, map_location="cpu")["params"]
    wdn_sd = torch.load(wdn_path, map_location="cpu")["params"]
    blended = blend_state_dicts(sd, wdn_sd, denoise)
    return _finalize(spandrel.ModelLoader().load_from_state_dict(blended))


def _upscale_once(image, model):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inp = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.inference_mode():
        out = model(inp.to(dtype)).float().clamp_(0.0, 1.0)
    return (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def upscale_to_target(image, model, target_size):
    """Apply as many 4x passes as the short side needs to reach target_size.

    image: RGB uint8 HxWx3 array. The caller's resize step handles the exact
    final dimensions; this only guarantees the short side is at target or the
    pass cap was hit.
    """
    h, w = image.shape[:2]
    for _ in range(upscale_passes(w, h, target_size)):
        image = _upscale_once(image, model)
    return image
