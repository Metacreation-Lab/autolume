"""Lazy super-resolution stage: 4x upscale applied on the render side.

The legacy super-res module was loaded once at import time and never moved
when the model's device changed (constraints.md legacy bug 5). `SuperRes` is
owned by the render path instead: it loads its weights on first use and
re-homes itself to whatever device `apply` is told the model now lives on.

Torch and the network class are imported inside methods, not at module
level, so importing this module never pulls torch into the control plane.
"""

import contextlib
import logging

from utils import resource_paths

logger = logging.getLogger(__name__)

MAX_SHORT_SIDE = 1024


class SuperRes:
    """4x SRVGGNetPlus ("Fast") upscaler for a ``[C, H, W]`` float image tensor."""

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._disabled = False
        self.disabled_reason: str | None = None
        self._guard_logged = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def apply(self, image, device):
        """Return ``image`` upscaled 4x, or unchanged if disabled or too large.

        ``device`` is whatever the caller's model currently lives on, so a
        device switch is picked up on the next call rather than requiring a
        new ``SuperRes`` instance. Never raises: missing or broken weights
        and an oversized frame just skip the stage.
        """
        import torch

        if self._disabled:
            return image
        short_side = min(int(image.shape[-2]), int(image.shape[-1]))
        if short_side > MAX_SHORT_SIDE:
            if not self._guard_logged:
                self._guard_logged = True
                logger.warning(
                    "Super-res skipped: frame short side %d exceeds the %d guard",
                    short_side,
                    MAX_SHORT_SIDE,
                )
            return image
        if self._model is None:
            self._model = self._load()
            if self._model is None:
                return image
        device = torch.device(device)
        if self._device != device:
            self._model = self._model.to(device)
            self._device = device
        with torch.inference_mode(), self._autocast(device):
            batch = image.unsqueeze(0).to(device)
            output = self._model(batch).float()
        return output[0]

    def _load(self):
        weight_path = resource_paths.resource_path("sr_models", "Fast.pt")
        if not weight_path.exists():
            self._disabled = True
            self.disabled_reason = f"Super-res weights not found at {weight_path}"
            logger.warning(self.disabled_reason)
            return None
        try:
            import torch
            from super_res.net_base import SRVGGNetPlus  # heavy import, deferred to first use

            model = SRVGGNetPlus(num_in_ch=3, num_out_ch=3, num_feat=48, upscale=4, act_type="prelu")
            state_dict = torch.load(str(weight_path), map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
        except Exception:
            # A render-thread call site: a bad weight file disables the stage
            # the same way a missing one does, rather than raising.
            self._disabled = True
            self.disabled_reason = f"Failed to load super-res weights from {weight_path}"
            logger.exception(self.disabled_reason)
            return None
        return model

    @staticmethod
    def _autocast(device):
        import torch

        if device.type == "cuda":
            return torch.autocast("cuda")
        # MPS has no autocast support, and CPU autocast would corrupt the
        # dtype of ops that fall back to CPU on MPS, so both stay full precision.
        return contextlib.nullcontext()
