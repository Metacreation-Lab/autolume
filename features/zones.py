"""Perceptual layer zones for direction edits.

GANSpace shows a direction restricted to early synthesis layers changes
geometry while the same direction on late layers changes appearance. Zones
select whole resolution blocks (layers come in pairs per resolution) so they
align with the model's coarse to fine structure at every resolution.
"""
import torch

ZONES = ("all", "form", "texture", "color")
CUSTOM_ZONE = "custom"
ZONE_LABELS = {"all": "All", "form": "Form", "texture": "Texture",
               "color": "Color", CUSTOM_ZONE: "Custom"}


def block_count(num_ws):
    """Resolution blocks in a model: layers come in pairs per resolution."""
    return max(1, (int(num_ws) + 1) // 2)


def block_labels(num_ws):
    """One label per block: 4x4, 8x8, doubling up to the model resolution."""
    labels = []
    res = 4
    for _ in range(block_count(num_ws)):
        labels.append(f"{res}x{res}")
        res *= 2
    return labels


def zone_blocks(zone, num_ws):
    """Per block booleans for a named zone."""
    if zone not in ZONES:
        raise ValueError(f"Unknown zone {zone!r}")
    n = block_count(num_ws)
    if zone == "all":
        return [True] * n
    form_end = max(1, round(0.3 * n))
    texture_end = min(n - 1, max(form_end + 1, round(0.6 * n)))
    if zone == "form":
        start, end = 0, form_end
    elif zone == "texture":
        start, end = form_end, texture_end
    else:
        start, end = texture_end, n
    return [start <= b < end for b in range(n)]


def blocks_to_mask(blocks, num_ws):
    """Expand per block booleans to a per layer boolean mask."""
    mask = torch.zeros(int(num_ws), dtype=torch.bool)
    for layer in range(int(num_ws)):
        block = layer // 2
        if block < len(blocks) and blocks[block]:
            mask[layer] = True
    return mask


def layer_mask(zone, num_ws):
    """Boolean (num_ws,) tensor of the synthesis layers a zone covers."""
    return blocks_to_mask(zone_blocks(zone, num_ws), num_ws)


def match_zone(layers, num_ws):
    """Name of the zone covering exactly these layers, or the custom zone."""
    n = int(num_ws)
    padded = ([bool(v) for v in (layers or [])] + [False] * n)[:n]
    for zone in ZONES:
        if padded == layer_mask(zone, n).tolist():
            return zone
    return CUSTOM_ZONE
