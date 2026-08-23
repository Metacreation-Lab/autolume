"""Image upscaling for dataset preparation (Real-ESRGAN)."""

from upscale.core import (load_upscaler, needs_upscale, required_weights,
                          upscale_passes, upscale_to_target)
from upscale.weights import WEIGHTS, ensure_weight, weight_path
