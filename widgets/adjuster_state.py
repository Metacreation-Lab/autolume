"""Versioned serialization for the Adjust Input widget preset state."""
import torch

STATE_VERSION = 1

_KEYS = ("model_pkl", "dirs", "base_dirs", "weights",
         "use_osc", "addresses", "mappings", "base_is_feature")


def pack_state(model_pkl, dirs, base_dirs, weights, use_osc, addresses, mappings,
               base_is_feature):
    return {
        "version": STATE_VERSION,
        "model_pkl": str(model_pkl) if model_pkl else None,
        "dirs": dirs.detach().cpu(),
        "base_dirs": base_dirs.detach().cpu(),
        "weights": weights.detach().cpu(),
        "use_osc": list(use_osc),
        "addresses": list(addresses),
        "mappings": list(mappings),
        "base_is_feature": list(base_is_feature),
    }


def unpack_state(data):
    """Validated state dict, or None for unknown or inconsistent formats."""
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return None
    if any(key not in data for key in _KEYS):
        return None
    dirs, base_dirs, weights = data["dirs"], data["base_dirs"], data["weights"]
    if not all(isinstance(t, torch.Tensor) for t in (dirs, base_dirs, weights)):
        return None
    if dirs.dim() != 2:
        return None
    try:
        n = len(dirs)
        if base_dirs.shape != dirs.shape or len(weights) != n:
            return None
        list_keys = ("use_osc", "addresses", "mappings", "base_is_feature")
        if any(len(data[key]) != n for key in list_keys):
            return None
        return {key: data[key] for key in _KEYS}
    except (TypeError, AttributeError):
        return None
