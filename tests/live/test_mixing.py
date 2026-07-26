"""Merge maths for live network mixing.

Every test here runs against real `custom_stylegan2.Generator`s with real
`state_dict`s rather than mocks. The whole risk in this file is a merge that
silently takes the wrong tensor: a mocked `state_dict` would let that pass
while the real one produces a model that loads fine and renders wrong.
The generators are built tiny (16x16, 4 to 8 channels a block) so a whole
suite of them costs less than a second.
"""

import pytest

import torch

from autolume.live.core.mixing import (
    INCOMPATIBLE_MODELS,
    combine,
    conv_names,
    layer_resolution,
    mapping_names,
    selection_length,
)


# `architectures.custom_stylegan2` drags in kornia and pkg_resources, each of
# which warns once on first import. Imported inside the helper below rather
# than at module scope so that first import happens under these filters
# instead of during collection, where a mark cannot reach it.
pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:.*torch\.cuda\.amp\.custom_fwd.*:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:pkg_resources is deprecated.*:DeprecationWarning"
    ),
]


def generator(seed=0, img_resolution=16, channel_max=8, w_dim=8, z_dim=8):
    """A tiny but genuine custom stylegan2 generator.

    `synthesis_kwargs` is always passed explicitly: `Generator.__init__`
    declares it as a mutable default and then calls `.update()` on it, so a
    call that leaves it out permanently pollutes the default for every later
    call in the process.
    """
    from architectures import custom_stylegan2

    torch.manual_seed(seed)
    return custom_stylegan2.Generator(
        z_dim=z_dim,
        c_dim=0,
        w_dim=w_dim,
        img_channels=3,
        img_resolution=img_resolution,
        synthesis_kwargs={"channel_base": 64, "channel_max": channel_max},
    )


def same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(a, b)


def test_conv_names_leaves_out_the_mapping_network():
    G = generator()
    names = conv_names(G)
    assert names
    assert all("mapping" not in name for name in names)
    assert all(name.startswith("synthesis.") for name in names)


def test_mapping_names_are_only_the_mapping_network():
    G = generator()
    names = mapping_names(G)
    assert names
    assert all("mapping" in name for name in names)


def test_conv_and_mapping_names_partition_the_parameters():
    G = generator()
    assert set(conv_names(G)) | set(mapping_names(G)) == {
        name for name, _ in G.named_parameters()
    }
    assert not set(conv_names(G)) & set(mapping_names(G))


def test_layer_resolution_reads_the_block_number():
    assert layer_resolution("synthesis.b4.conv1.weight") == 4
    assert layer_resolution("synthesis.b16.torgb.affine.bias") == 16


def test_layer_resolution_rejects_a_name_without_one():
    with pytest.raises(ValueError):
        layer_resolution("synthesis.const")


def test_selection_length_is_the_longer_models_layer_count():
    small = generator(seed=1, img_resolution=8)
    big = generator(seed=2, img_resolution=32)
    assert selection_length(small, big) == len(conv_names(big))
    assert selection_length(big, small) == len(conv_names(big))


def test_all_a_reproduces_model_a_weights():
    a = generator(seed=1)
    b = generator(seed=2)
    mixed = combine(a, b, ["A"] * selection_length(a, b))
    state_a = a.state_dict()
    state_mixed = mixed.state_dict()
    for name in conv_names(a) + mapping_names(a):
        assert same(state_mixed[name], state_a[name]), name


def test_all_b_reproduces_model_b_weights():
    a = generator(seed=1)
    b = generator(seed=2)
    mixed = combine(a, b, ["B"] * selection_length(a, b))
    state_b = b.state_dict()
    state_mixed = mixed.state_dict()
    for name in conv_names(b) + mapping_names(b):
        assert same(state_mixed[name], state_b[name]), name


def test_all_a_and_all_b_are_genuinely_different_models():
    """Guards the two tests above from passing on identical weights."""
    a = generator(seed=1)
    b = generator(seed=2)
    name = conv_names(a)[1]
    assert not same(a.state_dict()[name], b.state_dict()[name])


def test_a_boundary_split_takes_early_layers_from_a_and_late_from_b():
    a = generator(seed=1)
    b = generator(seed=2)
    names = conv_names(a)
    entries = ["A" if layer_resolution(n) <= 8 else "B" for n in names]
    assert "A" in entries and "B" in entries
    mixed = combine(a, b, entries)
    state_a, state_b, state_mixed = a.state_dict(), b.state_dict(), mixed.state_dict()
    for name, entry in zip(names, entries):
        source = state_a if entry == "A" else state_b
        assert same(state_mixed[name], source[name]), name


def test_the_mapping_network_follows_the_first_entry():
    a = generator(seed=1)
    b = generator(seed=2)
    names = conv_names(a)
    # Everything from A except the mapping, which the first entry decides.
    entries = ["B"] + ["A"] * (len(names) - 1)
    mixed = combine(a, b, entries)
    state_b, state_mixed = b.state_dict(), mixed.state_dict()
    for name in mapping_names(b):
        assert same(state_mixed[name], state_b[name]), name


def test_x_truncates_the_output_resolution():
    a = generator(seed=1, img_resolution=16)
    b = generator(seed=2, img_resolution=16)
    names = conv_names(a)
    entries = ["X" if layer_resolution(n) == 16 else "A" for n in names]
    mixed = combine(a, b, entries)
    assert mixed.img_resolution == 8
    assert all(layer_resolution(n) <= 8 for n in conv_names(mixed))


def test_x_truncation_still_renders_at_the_truncated_size():
    a = generator(seed=1, img_resolution=16)
    b = generator(seed=2, img_resolution=16)
    names = conv_names(a)
    entries = ["X" if layer_resolution(n) == 16 else "A" for n in names]
    mixed = combine(a, b, entries).eval()
    ws = torch.zeros([1, mixed.num_ws, mixed.w_dim])
    with torch.no_grad():
        img, _ = mixed.synthesis(ws, noise_mode="const")
    assert img.shape[-2:] == (8, 8)


def test_a_deeper_b_extends_the_mixed_model_past_a():
    a = generator(seed=1, img_resolution=8)
    b = generator(seed=2, img_resolution=16)
    names_a, names_b = conv_names(a), conv_names(b)
    entries = ["A"] * len(names_a) + ["B"] * (len(names_b) - len(names_a))
    mixed = combine(a, b, entries)
    assert mixed.img_resolution == 16
    state_a, state_b, state_mixed = a.state_dict(), b.state_dict(), mixed.state_dict()
    for index, entry in enumerate(entries):
        if entry == "A":
            assert same(state_mixed[names_a[index]], state_a[names_a[index]])
        else:
            assert same(state_mixed[names_b[index]], state_b[names_b[index]])


def test_incompatible_models_raise_the_documented_message():
    # Different block widths, split at a boundary: the mixed block's input
    # width comes from A while the weight copied into it comes from B.
    a = generator(seed=1, channel_max=8)
    b = generator(seed=2, channel_max=16)
    names = conv_names(a)
    entries = ["A" if layer_resolution(n) == 4 else "B" for n in names]
    with pytest.raises(ValueError) as caught:
        combine(a, b, entries)
    assert str(caught.value) == INCOMPATIBLE_MODELS


def test_the_documented_message_is_the_legacy_text():
    assert INCOMPATIBLE_MODELS == (
        "These models are incompatible. Compressed models generally can not "
        "be used for mixing."
    )


def test_a_selection_of_the_wrong_length_raises():
    a = generator(seed=1)
    b = generator(seed=2)
    with pytest.raises(ValueError) as caught:
        combine(a, b, ["A"] * (selection_length(a, b) - 1))
    assert "entries" in str(caught.value)


def test_an_unknown_origin_raises():
    a = generator(seed=1)
    b = generator(seed=2)
    entries = ["A"] * selection_length(a, b)
    entries[2] = "C"
    with pytest.raises(ValueError) as caught:
        combine(a, b, entries)
    assert "'C'" in str(caught.value)


def test_a_layer_the_chosen_model_does_not_have_raises():
    a = generator(seed=1, img_resolution=8)
    b = generator(seed=2, img_resolution=16)
    names_a, names_b = conv_names(a), conv_names(b)
    # The tail belongs to B alone, so asking for A there is out of range.
    entries = ["A"] * len(names_b)
    with pytest.raises(ValueError) as caught:
        combine(a, b, entries)
    assert str(len(names_a)) in str(caught.value)


def test_an_all_x_selection_raises():
    a = generator(seed=1)
    b = generator(seed=2)
    with pytest.raises(ValueError) as caught:
        combine(a, b, ["X"] * selection_length(a, b))
    assert "A or B" in str(caught.value)


def test_a_gap_in_the_middle_of_the_selection_raises():
    a = generator(seed=1, img_resolution=16)
    b = generator(seed=2, img_resolution=16)
    names = conv_names(a)
    # X in the middle rather than as a trailing truncation: the mixed model
    # would be missing a block the resolution it keeps still needs.
    entries = ["X" if layer_resolution(n) == 8 else "A" for n in names]
    with pytest.raises(ValueError) as caught:
        combine(a, b, entries)
    assert "8" in str(caught.value)


def test_combine_does_not_alias_the_source_weights():
    a = generator(seed=1)
    b = generator(seed=2)
    mixed = combine(a, b, ["A"] * selection_length(a, b))
    name = conv_names(a)[1]
    with torch.no_grad():
        mixed.state_dict()[name].add_(1.0)
    assert not same(mixed.state_dict()[name], a.state_dict()[name])


def test_combine_leaves_the_sources_on_their_own_device():
    a = generator(seed=1)
    b = generator(seed=2)
    mixed = combine(a, b, ["A"] * selection_length(a, b))
    assert next(mixed.parameters()).device == torch.device("cpu")
    assert next(a.parameters()).device == torch.device("cpu")


def test_a_mix_does_not_inherit_the_sources_buffers():
    """Ported behavior, pinned deliberately rather than left implicit.

    The merge copies parameters and not buffers, because that is what the
    two legacy paths do. A mixed model therefore starts with a zero `w_avg`
    and its own freshly drawn constant noise, so an all-A mix diverges from
    A exactly where those two are read: truncation below 1, and the "const"
    noise mode. Flagged in the task report for the maintainer to decide on,
    not changed here.
    """
    a = generator(seed=1)
    b = generator(seed=2)
    # A trained model's w_avg is whatever training left there, never zero.
    with torch.no_grad():
        a.mapping.w_avg.fill_(0.5)
    mixed = combine(a, b, ["A"] * selection_length(a, b))
    assert torch.equal(mixed.mapping.w_avg, torch.zeros_like(mixed.mapping.w_avg))
    noise = "synthesis.b8.conv0.noise_const"
    assert not torch.equal(mixed.state_dict()[noise], a.state_dict()[noise])
