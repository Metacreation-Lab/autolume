"""Network mixing: one generator assembled from two loaded models.

Ported from `modules/network_mixing.py`'s `combine_models`, with the per
resolution channel merge the live path needs so a pair whose blocks are
different widths still assembles into a network that can be built at all.

Everything here is a pure function over the two source generators and the
selection. No threading, no device handling, no host state: `ModelHost`
calls `combine` on the loader thread and owns what happens to the result.

The selection is one entry per name in `conv_names`, aligned by index, and
padded to whichever model has more layers. "A" and "B" name the model a
layer's weights come from, and "X" removes the layer, which in practice
only ever truncates the tail and so lowers the mixed model's output
resolution.
"""

import logging
import re

logger = logging.getLogger(__name__)

ORIGIN_A = "A"
ORIGIN_B = "B"
ORIGIN_X = "X"

# The legacy wording, kept verbatim: this is what the mixing panel shows a
# performer whose two models cannot be assembled, and it names the usual
# cause rather than the shape mismatch underneath.
INCOMPATIBLE_MODELS = (
    "These models are incompatible. Compressed models generally can not "
    "be used for mixing."
)

_DIGIT_RUN = re.compile(r"\d+")


def conv_names(G) -> tuple[str, ...]:
    """Every synthesis parameter name, in network order."""
    return tuple(name for name, _ in G.named_parameters() if "mapping" not in name)


def mapping_names(G) -> tuple[str, ...]:
    """Every mapping network parameter name, in network order."""
    return tuple(name for name, _ in G.named_parameters() if "mapping" in name)


def layer_resolution(name: str) -> int:
    """The block resolution a parameter name belongs to.

    Synthesis parameters are named `synthesis.b<res>....`, so the first run
    of digits is the resolution. This is how the legacy mixing UI grouped
    layers and how the merged channel table is keyed.
    """
    match = _DIGIT_RUN.search(name)
    if match is None:
        raise ValueError(f"Layer name {name!r} carries no resolution")
    return int(match.group())


def selection_length(G_a, G_b) -> int:
    """How many entries a selection over this pair must have.

    The longer model decides: a selection covers every layer either model
    has, so the extra tail of a deeper model is addressable.
    """
    return max(len(conv_names(G_a)), len(conv_names(G_b)))


def _name_at(names: tuple[str, ...], index: int, origin: str) -> str:
    if index >= len(names):
        raise ValueError(
            f"Layer {index + 1} is set to model {origin} but that model "
            f"only has {len(names)} layers"
        )
    return names[index]


def _channels_of(G, origin: str) -> dict:
    channels = getattr(getattr(G, "synthesis", None), "channels_dict", None)
    if not isinstance(channels, dict):
        raise ValueError(
            f"Model {origin} does not publish a channel table, so it cannot "
            "be mixed"
        )
    return channels


def _output_resolution(
    entries: tuple[str, ...], names_a: tuple[str, ...], names_b: tuple[str, ...]
) -> int:
    """The mixed model's output resolution: the last layer still kept.

    Trailing "X" entries are what truncate the network, so the resolution is
    read off the last entry that is neither removed nor blank.
    """
    last_index = 0
    for index, entry in enumerate(entries):
        if entry not in ("", ORIGIN_X):
            last_index = index
    last = entries[last_index] if entries else ""
    if last == ORIGIN_A:
        return layer_resolution(_name_at(names_a, last_index, ORIGIN_A))
    if last == ORIGIN_B:
        return layer_resolution(_name_at(names_b, last_index, ORIGIN_B))
    raise ValueError(
        f"The last layer kept must come from A or B, not {last!r}"
    )


def _merged_channels(
    entries: tuple[str, ...],
    names_a: tuple[str, ...],
    names_b: tuple[str, ...],
    G_a,
    G_b,
    img_resolution: int,
) -> dict:
    """Each kept resolution's channel count, taken from the model it came from.

    Without this the mixed model would be built on the default channel
    schedule and a pair trained with a custom width would not load at all.
    """
    channels: dict[int, int] = {}
    for index, entry in enumerate(entries):
        if entry == ORIGIN_A:
            resolution = layer_resolution(_name_at(names_a, index, ORIGIN_A))
            channels[resolution] = _channels_of(G_a, ORIGIN_A)[resolution]
        elif entry == ORIGIN_B:
            resolution = layer_resolution(_name_at(names_b, index, ORIGIN_B))
            channels[resolution] = _channels_of(G_b, ORIGIN_B)[resolution]
        elif entry in ("", ORIGIN_X):
            continue
        else:
            raise ValueError(f"Layer origin must be A, B or X, not {entry!r}")
    expected = []
    resolution = 4
    while resolution <= img_resolution:
        expected.append(resolution)
        resolution *= 2
    missing = [res for res in expected if res not in channels]
    if missing:
        raise ValueError(
            "The layer selection removes resolution "
            f"{missing[0]} but keeps a larger one, so the mixed model has a gap"
        )
    return channels


def combine(G_a, G_b, combined_layers):
    """Assemble a new generator from `G_a` and `G_b` per `combined_layers`.

    Returned unmoved and untrained: a plain module on whatever device it was
    constructed on (CPU), holding copies of the chosen weights. The caller
    puts it in eval mode and moves it. Nothing here is shared with either
    source, so releasing the result never touches them.

    Raises `ValueError` for every rejection, including the two the sources
    themselves cause: a selection that does not line up with the pair, and a
    pair whose chosen layers cannot be assembled at all.
    """
    from architectures import custom_stylegan2

    entries = tuple(combined_layers)
    names_a = conv_names(G_a)
    names_b = conv_names(G_b)
    expected = max(len(names_a), len(names_b))
    if len(entries) != expected:
        raise ValueError(
            f"The layer selection has {len(entries)} entries but this pair "
            f"of models has {expected}"
        )
    img_resolution = _output_resolution(entries, names_a, names_b)
    channels = _merged_channels(
        entries, names_a, names_b, G_a, G_b, img_resolution
    )
    mixed = custom_stylegan2.Generator(
        z_dim=G_a.z_dim,
        c_dim=G_a.c_dim,
        w_dim=G_a.w_dim,
        img_channels=G_a.img_channels,
        img_resolution=img_resolution,
        # Passed explicitly because Generator declares this as a mutable
        # default and then updates it in place.
        synthesis_kwargs={"channels_dict": channels},
    )
    destination = mixed.state_dict()
    state_a = G_a.state_dict()
    state_b = G_b.state_dict()
    # The mapping network is not per layer, so the first entry decides which
    # model's whole mapping the mix inherits.
    if entries and entries[0] == ORIGIN_A:
        for name in mapping_names(G_a):
            destination[name] = state_a[name]
    elif entries and entries[0] == ORIGIN_B:
        for name in mapping_names(G_b):
            destination[name] = state_b[name]
    for index, entry in enumerate(entries):
        if entry == ORIGIN_A:
            name = names_a[index]
            destination[name] = state_a[name]
        elif entry == ORIGIN_B:
            name = names_b[index]
            destination[name] = state_b[name]
    try:
        mixed.load_state_dict(destination)
    except Exception as exc:
        logger.warning("Could not assemble the mixed model: %s", exc)
        raise ValueError(INCOMPATIBLE_MODELS) from exc
    return mixed
