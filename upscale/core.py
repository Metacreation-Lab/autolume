"""Image upscaling for dataset preparation and live output.

Two models are available, one per screen. Dataset preparation always uses
4xNomosWebPhoto_RealPLKSR, the live display always uses 4xLSDIRCompactC3.
Neither screen offers a choice; the Upscaling tool is the one place
both are selectable.

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

# Inference is tiled so peak VRAM follows the tile, not the image. The 256px
# tile was what DAT2's windowed attention needed; the remaining conv nets run a
# 1024px input in about 2 GB (measured), so ordinary inputs take the single
# pass path and tiling is a safety net for oversized ones such as the batch
# tool's video frames.
TILE = 1024
TILE_OVERLAP = 32

# Selectable upscalers, by the WEIGHTS key each one loads. Every model is 4x.
MODELS = {
    "RealPLKSR": {"name": "4xNomosWebPhoto_RealPLKSR", "weights": "RealPLKSR"},
    "CompactC3": {"name": "4xLSDIRCompactC3",          "weights": "CompactC3"},
}
PREPARE_MODEL = "RealPLKSR"
PERFORM_MODEL = "CompactC3"


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


def load_upscaler(model=PREPARE_MODEL, progress_cb=None, cancel_event=None):
    """Build the upscaler. Returns None if a download was cancelled."""
    path = ensure_weight(MODELS[model]["weights"], progress_cb, cancel_event)
    if path is None:
        return None
    return _finalize(spandrel.ModelLoader().load_from_file(path))


def _tiled_forward(model, inp):
    """Run model(inp) tile by tile. Drop-in for a single forward pass.

    Each tile is fed with up to TILE_OVERLAP pixels of neighbouring context and
    the result is cropped back to the tile itself, so the seam artifacts a hard
    cut would leave fall in the discarded border (overlap-discard, no blending,
    as in Real-ESRGAN).
    """
    _, _, h, w = inp.shape
    if h <= TILE and w <= TILE:
        return model(inp)

    out, scale = None, 1
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y1, x1 = min(y + TILE, h), min(x + TILE, w)
            ey, ex = max(y - TILE_OVERLAP, 0), max(x - TILE_OVERLAP, 0)
            ey1, ex1 = min(y1 + TILE_OVERLAP, h), min(x1 + TILE_OVERLAP, w)
            tile = model(inp[:, :, ey:ey1, ex:ex1])
            if out is None:
                scale = tile.shape[2] // (ey1 - ey)
                out = tile.new_empty((inp.shape[0], tile.shape[1], h * scale, w * scale))
            top, left = (y - ey) * scale, (x - ex) * scale
            out[:, :, y * scale:y1 * scale, x * scale:x1 * scale] = tile[
                :, :, top:top + (y1 - y) * scale, left:left + (x1 - x) * scale]
    return out


def _upscale_once(image, model):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inp = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.inference_mode():
        out = _tiled_forward(model, inp.to(dtype)).float().clamp_(0.0, 1.0)
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
