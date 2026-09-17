"""Image upscaling for dataset preparation and live output."""

from upscale.core import (MODELS, PERFORM_MODEL, PREPARE_MODEL, load_upscaler,
                          needs_upscale, required_weights, upscale_passes,
                          upscale_to_target)
from upscale.weights import WEIGHTS, ensure_weight, weight_path
