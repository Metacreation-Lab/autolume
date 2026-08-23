"""Real-ESRGAN upscaling for dataset preprocessing.

Two model choices. Balance is realesr-general-x4v3 with the official
denoise blend: strength 1.0 is the plain model (strongest denoise), lower
values interpolate toward the weak-denoise wdn weights, which keep more
grain. Quality is RealESRGAN_x4plus, much slower, no denoise control.
"""

import logging

import numpy as np
import torch

from super_res.net_base import SRVGGNetCompact
from super_res.super_res import ensure_sr_weight, load_model
from utils.device_utils import get_device

logger = logging.getLogger(__name__)

MODEL_KEY = "Balance"
WDN_MODEL_KEY = "BalanceWDN"
QUALITY_MODEL_KEY = "Quality"
MAX_PASSES = 3


def required_weights(model_type, denoise):
    """SR_WEIGHTS keys the given settings need on disk."""
    if model_type == QUALITY_MODEL_KEY:
        return [QUALITY_MODEL_KEY]
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


def load_upscaler(denoise=0.0, model_type=MODEL_KEY, progress_cb=None, cancel_event=None):
    """Build the dataset upscaler. Returns None if a download was cancelled."""
    if model_type == QUALITY_MODEL_KEY:
        path = ensure_sr_weight(QUALITY_MODEL_KEY, progress_cb, cancel_event)
        if path is None:
            return None
        return load_model(QUALITY_MODEL_KEY, path).eval()
    # Balance: blend of the x4v3 endpoints, loading only the files the
    # denoise value actually uses.
    sd = None
    if denoise > 0.0:
        path = ensure_sr_weight(MODEL_KEY, progress_cb, cancel_event)
        if path is None:
            return None
        sd = torch.load(path, map_location="cpu")["params"]
    if denoise < 1.0:
        wdn_path = ensure_sr_weight(WDN_MODEL_KEY, progress_cb, cancel_event)
        if wdn_path is None:
            return None
        wdn_sd = torch.load(wdn_path, map_location="cpu")["params"]
        sd = wdn_sd if sd is None else blend_state_dicts(sd, wdn_sd, denoise)
    device = get_device()
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                            upscale=4, act_type="prelu").eval().to(device)
    model.load_state_dict(sd)
    # Same fp16 rationale as load_model in super_res.py.
    if device.type in ("cuda", "mps"):
        model = model.half()
    return model


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
