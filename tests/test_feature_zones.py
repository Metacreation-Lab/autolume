import pytest
import torch

from features.zones import (CUSTOM_ZONE, ZONES, ZONE_LABELS, block_count,
                            block_labels, blocks_to_mask, layer_mask,
                            zone_blocks)


def test_zone_names():
    assert ZONES == ("all", "form", "texture", "color")
    assert set(ZONE_LABELS) == set(ZONES) | {CUSTOM_ZONE}


def test_all_zone_covers_everything():
    assert layer_mask("all", 18).all()
    assert layer_mask("all", 18).shape == (18,)


def test_named_zones_partition_layers():
    for num_ws in range(3, 19):
        form = layer_mask("form", num_ws)
        texture = layer_mask("texture", num_ws)
        color = layer_mask("color", num_ws)
        combined = form | texture | color
        assert combined.all(), num_ws
        assert (form & texture).sum() == 0
        assert (texture & color).sum() == 0
        assert (form & color).sum() == 0
        assert form.any() and color.any(), num_ws


def test_zones_align_to_resolution_blocks():
    # 1024px StyleGAN2: 9 blocks; form 4-16px, texture 32-64px, color 128px up
    assert layer_mask("form", 18).nonzero().flatten().tolist() == list(range(0, 6))
    assert layer_mask("texture", 18).nonzero().flatten().tolist() == list(range(6, 10))
    assert layer_mask("color", 18).nonzero().flatten().tolist() == list(range(10, 18))


def test_block_count_pairs_layers():
    assert block_count(18) == 9
    assert block_count(14) == 7
    assert block_count(3) == 2
    assert block_count(1) == 1


def test_block_labels_double_resolution():
    assert block_labels(18) == ["4x4", "8x8", "16x16", "32x32", "64x64",
                                "128x128", "256x256", "512x512", "1024x1024"]
    assert block_labels(4) == ["4x4", "8x8"]


def test_blocks_to_mask_expands_pairs():
    mask = blocks_to_mask([True, False, True], 6)
    assert mask.tolist() == [True, True, False, False, True, True]


def test_blocks_to_mask_handles_short_and_long_lists():
    assert blocks_to_mask([True], 6).tolist() == [True, True, False, False,
                                                  False, False]
    assert blocks_to_mask([True] * 10, 4).all()


def test_zone_blocks_roundtrip_through_mask():
    for zone in ZONES:
        blocks = zone_blocks(zone, 18)
        assert torch.equal(blocks_to_mask(blocks, 18), layer_mask(zone, 18))


def test_unknown_zone_raises():
    with pytest.raises(ValueError):
        layer_mask("bass", 18)
