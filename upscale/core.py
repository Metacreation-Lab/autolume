"""Image upscaling for dataset preparation and live output.

Three models are available. Dataset preparation offers Standard
(4xNomosWebPhoto_RealPLKSR) and Restore (4xRealWebPhoto_v4_dat2); the live
display offers Fast (4xLSDIRCompactC3) and the same Standard. Both screens
name their models by what they do rather than by weight file.

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

MAX_PASSES = 3

# Selectable upscalers, by the WEIGHTS key each one loads. Every model is 4x.
MODELS = {
    "RealPLKSR": {"name": "4xNomosWebPhoto_RealPLKSR", "weights": "RealPLKSR"},
    "DAT2":      {"name": "4xRealWebPhoto_v4_dat2",    "weights": "DAT2"},
    "CompactC3": {"name": "4xLSDIRCompactC3",          "weights": "CompactC3"},
}
PREPARE_MODELS = ["RealPLKSR", "DAT2"]
PREPARE_LABELS = {"RealPLKSR": "Standard", "DAT2": "Restore"}
PREPARE_DEFAULT_MODEL = "RealPLKSR"
PERFORM_MODELS = ["CompactC3", "RealPLKSR"]
PERFORM_LABELS = {"CompactC3": "Fast", "RealPLKSR": "Standard"}
PERFORM_DEFAULT_MODEL = "CompactC3"


def required_weights(model):
    """WEIGHTS keys the given model needs on disk."""
    return [MODELS[model]["weights"]]


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


def load_upscaler(model=PREPARE_DEFAULT_MODEL, progress_cb=None, cancel_event=None):
    """Build the upscaler. Returns None if a download was cancelled."""
    path = ensure_weight(MODELS[model]["weights"], progress_cb, cancel_event)
    if path is None:
        return None
    return _finalize(spandrel.ModelLoader().load_from_file(path))


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
