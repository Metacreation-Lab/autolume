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


def randomize(model, seed):
    """Give every parameter and stateful buffer a value distinct from any
    other model's.

    A freshly constructed generator zero-initialises 29 of its 54
    parameters (every bias, every affine bias, every noise strength) and
    leaves `w_avg` at zero, so two of them are bit identical across more
    than half their tensors. Merge assertions against such a pair pass just
    as happily on a merge that takes the wrong source.

    Perturbs the real initialisation rather than replacing it: drawing
    every weight afresh at any scale drives the synthesis output far past
    the uint8 range, so every model renders a uniform white frame and the
    tests that compare frames stop meaning anything. `resample_filter` is
    left alone deliberately: it is a real filter kernel, construction
    deterministic and identical in both models by definition, so it is the
    one tensor a merge cannot get wrong.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1 + seed * 0.02)
        for name, buffer in model.named_buffers():
            if not name.endswith("resample_filter"):
                buffer.add_(torch.randn_like(buffer) * 0.1 + seed * 0.02)
    return model


def generator(
    seed=0, img_resolution=16, channel_max=8, w_dim=8, z_dim=8, mapping_layers=None
):
    """A tiny but genuine custom stylegan2 generator, fully randomised.

    `synthesis_kwargs` is always passed explicitly: `Generator.__init__`
    declares it as a mutable default and then calls `.update()` on it, so a
    call that leaves it out permanently pollutes the default for every later
    call in the process.
    """
    from architectures import custom_stylegan2

    mapping_kwargs = {} if mapping_layers is None else {"num_layers": mapping_layers}
    torch.manual_seed(seed)
    return randomize(
        custom_stylegan2.Generator(
            z_dim=z_dim,
            c_dim=0,
            w_dim=w_dim,
            img_channels=3,
            img_resolution=img_resolution,
            mapping_kwargs=mapping_kwargs,
            synthesis_kwargs={"channel_base": 64, "channel_max": channel_max},
        ),
        seed,
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


def test_an_all_a_mix_is_bit_identical_to_model_a():
    """The plan's acceptance criterion, as a test: every entry of the mixed
    state dict, parameters and buffers alike, equal to A's."""
    a = generator(seed=1)
    b = generator(seed=2)
    mixed = combine(a, b, ["A"] * selection_length(a, b))
    state_a, state_mixed = a.state_dict(), mixed.state_dict()
    assert set(state_mixed) == set(state_a)
    for name in state_a:
        assert same(state_mixed[name], state_a[name]), name


def test_an_all_b_mix_is_bit_identical_to_model_b():
    a = generator(seed=1)
    b = generator(seed=2)
    mixed = combine(a, b, ["B"] * selection_length(a, b))
    state_b, state_mixed = b.state_dict(), mixed.state_dict()
    assert set(state_mixed) == set(state_b)
    for name in state_b:
        assert same(state_mixed[name], state_b[name]), name


def test_a_mix_inherits_the_buffers_of_the_weights_they_serve():
    """`w_avg` follows the mapping network it is a statistic of, and each
    `noise_const` follows the block whose weights it is added to."""
    a = generator(seed=1)
    b = generator(seed=2)
    names = conv_names(a)
    entries = ["A" if layer_resolution(n) <= 8 else "B" for n in names]
    mixed = combine(a, b, entries)
    state_a, state_b, state_mixed = a.state_dict(), b.state_dict(), mixed.state_dict()

    assert same(state_mixed["mapping.w_avg"], state_a["mapping.w_avg"])
    assert same(
        state_mixed["synthesis.b8.conv0.noise_const"],
        state_a["synthesis.b8.conv0.noise_const"],
    )
    assert same(
        state_mixed["synthesis.b16.conv0.noise_const"],
        state_b["synthesis.b16.conv0.noise_const"],
    )


def test_the_mapping_buffer_follows_the_mapping_network_not_model_a():
    a = generator(seed=1)
    b = generator(seed=2)
    names = conv_names(a)
    entries = ["B"] + ["A"] * (len(names) - 1)
    mixed = combine(a, b, entries)
    assert same(mixed.state_dict()["mapping.w_avg"], b.state_dict()["mapping.w_avg"])


def test_an_unusable_mapping_argument_from_a_source_raises_the_documented_message():
    """`mapping_kwargs` comes out of a user's pkl and goes into a
    constructor with no `**kwargs`, so it is a way a source file can reach
    this module. It must still come back as the one documented sentence,
    not as a raw `TypeError` about a keyword argument.
    """
    a = generator(seed=1)
    b = generator(seed=2)
    # What an unpickled model restores verbatim. The public `init_kwargs` is
    # a deep copying property, so writing through it would not stick.
    a._init_kwargs = dict(a._init_kwargs, mapping_kwargs={"not_a_real_argument": 1})
    with pytest.raises(ValueError) as caught:
        combine(a, b, ["A"] * selection_length(a, b))
    assert str(caught.value) == INCOMPATIBLE_MODELS


def test_a_split_module_gives_its_buffers_to_its_first_parameter():
    """The selection is per parameter, so one module's weights can be split
    across both models. The buffer follows the module's first parameter in
    network order, which for a synthesis layer is the `weight` a
    `noise_const` is added alongside.
    """
    a = generator(seed=1)
    b = generator(seed=2)
    names = conv_names(a)
    weight = names.index("synthesis.b8.conv0.weight")
    bias = names.index("synthesis.b8.conv0.bias")
    assert weight < bias
    noise = "synthesis.b8.conv0.noise_const"

    entries = ["A"] * selection_length(a, b)
    entries[bias] = "B"
    assert same(combine(a, b, entries).state_dict()[noise], a.state_dict()[noise])

    entries = ["A"] * selection_length(a, b)
    entries[weight] = "B"
    assert same(combine(a, b, entries).state_dict()[noise], b.state_dict()[noise])


def test_a_source_mapping_depth_is_carried_into_the_mix():
    """A model trained with a non default mapping depth contributes its
    whole mapping, not its first eight layers with the rest left random."""
    a = generator(seed=1, mapping_layers=2)
    b = generator(seed=2, mapping_layers=2)
    mixed = combine(a, b, ["A"] * selection_length(a, b))
    assert set(mapping_names(mixed)) == set(mapping_names(a))
    for name in mapping_names(a):
        assert same(mixed.state_dict()[name], a.state_dict()[name]), name


def test_a_mix_repeated_from_the_same_pair_is_reproducible():
    """The save path assembles a second time, so two builds from one pair
    must agree or the file written is not the mix being previewed."""
    a = generator(seed=1)
    b = generator(seed=2)
    entries = ["A" if layer_resolution(n) <= 8 else "B" for n in conv_names(a)]
    first = combine(a, b, entries).state_dict()
    second = combine(a, b, entries).state_dict()
    assert set(first) == set(second)
    for name in first:
        assert same(first[name], second[name]), name
