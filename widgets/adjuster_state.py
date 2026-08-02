"""Versioned serialization for the Feature Mixer slot rack."""
import torch

from features.zones import CUSTOM_ZONE, ZONES

STATE_VERSION = 2
NUM_SLOTS = 8

SLOT_KEYS = ("component", "direction", "sigma", "zone", "weight", "name",
             "use_osc", "address", "mapping")


def make_slot(direction, component=None, sigma=1.0, zone="all", weight=0.0,
              name="", use_osc=False, address="", mapping="x", layers=None):
    return {
        "component": component,
        "direction": direction.detach().cpu().to(torch.float32),
        "sigma": float(sigma),
        "zone": zone,
        "weight": float(weight),
        "name": name,
        "use_osc": bool(use_osc),
        "address": address,
        "mapping": mapping,
        "layers": [bool(b) for b in layers] if layers is not None else None,
    }


def pack_state(model_pkl, slots):
    return {
        "version": STATE_VERSION,
        "model_pkl": str(model_pkl) if model_pkl else None,
        "slots": [dict(slot, direction=slot["direction"].detach().cpu())
                  for slot in slots],
    }


def _valid_slot(slot, w_dim):
    if not isinstance(slot, dict) or any(key not in slot for key in SLOT_KEYS):
        return False
    direction = slot["direction"]
    if not torch.is_tensor(direction) or direction.dim() != 1:
        return False
    if w_dim is not None and direction.shape[0] != w_dim:
        return False
    if slot["zone"] not in ZONES and slot["zone"] != CUSTOM_ZONE:
        return False
    if slot["component"] is not None and not isinstance(slot["component"], int):
        return False
    return True


def _slot_layers(slot):
    layers = slot.get("layers")
    if isinstance(layers, list) and layers:
        return layers
    return None


def unpack_state(data):
    """Validated state dict normalized to NUM_SLOTS slots, or None."""
    try:
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            return None
        slots = data.get("slots")
        if not isinstance(slots, list) or not slots:
            return None
        first = slots[0]
        if not _valid_slot(first, None):
            return None
        w_dim = first["direction"].shape[0]
        if not all(_valid_slot(slot, w_dim) for slot in slots):
            return None
        slots = [make_slot(**{key: slot[key] for key in SLOT_KEYS},
                           layers=_slot_layers(slot))
                 for slot in slots[:NUM_SLOTS]]
        for slot in slots:
            if slot["zone"] == CUSTOM_ZONE and slot["layers"] is None:
                slot["zone"] = "all"
        while len(slots) < NUM_SLOTS:
            direction = torch.randn(w_dim)
            norm = float(direction.norm())
            if norm > 0:
                direction = direction / norm
            slots.append(make_slot(direction, sigma=float(w_dim) ** 0.5))
        return {"model_pkl": data.get("model_pkl"), "slots": slots}
    except (TypeError, AttributeError, KeyError):
        return None
