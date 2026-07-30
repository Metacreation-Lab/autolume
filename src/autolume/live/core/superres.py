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
import re

from autolume.live.errors import safe_describe
from utils import resource_paths

logger = logging.getLogger(__name__)

MAX_SHORT_SIDE = 1024

# Past this many distinct failure causes, the model has bigger problems than
# a missing log line, and the set stops growing (mirrors generator.py's
# _LOG_ONCE_CAP so a session that goes quiet says why instead of just
# falling silent).
_LOG_ONCE_CAP = 64
_DIGIT_RUN = re.compile(r"\d+")


class SuperRes:
    """4x SRVGGNetPlus ("Fast") upscaler for a ``[C, H, W]`` float image tensor."""

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._disabled = False
        self.disabled_reason: str | None = None
        self._guard_logged = False
        self.last_error: str | None = None
        self._logged_errors: set[str] = set()
        self._log_cap_warned = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def apply(self, image, device):
        """Return ``image`` upscaled 4x, or unchanged if disabled or too large.

        ``device`` is whatever the caller's model currently lives on, so a
        device switch is picked up on the next call rather than requiring a
        new ``SuperRes`` instance. Never raises: missing or broken weights,
        an oversized frame, a failed device move, and a failed forward pass
        all just skip the stage for that call instead.

        A load failure (missing or corrupt weights) is permanent: it is a
        fact about this install, not this frame, so it sets ``disabled`` and
        stays that way. A device-move or forward failure is transient: it
        depends on frame size and current memory pressure, so it only sets
        ``last_error`` for that call, cleared again on the next success,
        rather than permanently killing the stage over one bad frame.
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
            try:
                self._model = self._model.to(device)
            except Exception as exc:
                self._record_forward_failure(f"Super-res failed to move to {device}", exc)
                return image
            self._device = device
        try:
            with torch.inference_mode(), self._autocast(device):
                batch = image.unsqueeze(0).to(device)
                output = self._model(batch).float()
        except Exception as exc:
            self._record_forward_failure("Super-res forward pass failed", exc)
            return image
        self.last_error = None
        return output[0]

    def _record_forward_failure(self, context: str, exc: Exception) -> None:
        """Note a transient (input- or memory-dependent) failure, logged once per cause.

        Unlike `_load`'s permanent sentinel, this never sets `disabled`: the
        very next frame, at a smaller size or with memory freed up, may work.
        `last_error` itself is set unconditionally (it is the current status,
        not a log), only the warning line is deduplicated by cause. `exc` is
        stringified exactly once here, defensively, and that text is used for
        both the message and the dedup key rather than formatting `exc`
        directly at each call site.
        """
        text = safe_describe(exc)
        message = f"{context}: {text}"
        self.last_error = message
        key = f"{type(exc).__name__}:{_DIGIT_RUN.sub('N', text)}"
        if key in self._logged_errors:
            return
        if len(self._logged_errors) >= _LOG_ONCE_CAP:
            if not self._log_cap_warned:
                self._log_cap_warned = True
                logger.warning(
                    "Reached %d distinct super-res failure causes, further "
                    "distinct causes will not be logged",
                    _LOG_ONCE_CAP,
                )
            return
        self._logged_errors.add(key)
        logger.warning(message)

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
        if device.type == "mps":
            # MPS has no autocast support of its own, and enabling CPU
            # autocast here would corrupt the dtype of ops MPS silently
            # falls back to running on CPU. Full precision only.
            return contextlib.nullcontext()
        # CPU (or any other backend): this network is small enough that
        # autocast buys no measurable speedup, so skip the dtype juggling.
        return contextlib.nullcontext()
