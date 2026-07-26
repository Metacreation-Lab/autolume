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


def _mapping_kwargs(G) -> dict:
    """The mapping network arguments `G` was built with.

    Forwarded to the mixed model so a source trained with a non-default
    mapping depth contributes its whole mapping rather than its first eight
    layers with the rest left freshly random. `init_kwargs` is what the
    persistence decorator records at construction; a generator without one
    falls back to the defaults, which is what the legacy merge always used.
    """
    init_kwargs = getattr(G, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        return {}
    mapping_kwargs = init_kwargs.get("mapping_kwargs")
    return dict(mapping_kwargs) if isinstance(mapping_kwargs, dict) else {}


def _origin_by_module(
    entries: tuple[str, ...], names_a: tuple[str, ...], names_b: tuple[str, ...]
) -> dict[str, str]:
    """Which model each synthesis submodule's weights came from.

    Keyed by every ancestor module path of every chosen parameter, first
    match in network order winning, so a lookup can walk up from a buffer's
    own module until it finds the weights that buffer serves.
    """
    origins: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if entry == ORIGIN_A:
            name = names_a[index]
        elif entry == ORIGIN_B:
            name = names_b[index]
        else:
            continue
        parts = name.split(".")[:-1]
        while parts:
            origins.setdefault(".".join(parts), entry)
            parts = parts[:-1]
    return origins


def _buffer_origin(
    key: str, module_origins: dict[str, str], mapping_origin: str | None
) -> str | None:
    """The model a non-parameter state entry should be taken from.

    A buffer belongs with the weights it serves. `mapping.w_avg` is a
    statistic of one mapping network's own W distribution, so it follows
    whichever model provided the mapping; a synthesis buffer follows its
    own block's weights. Anything that cannot be placed keeps the value it
    was constructed with.
    """
    if key.startswith("mapping"):
        return mapping_origin
    parts = key.split(".")[:-1]
    while parts:
        origin = module_origins.get(".".join(parts))
        if origin is not None:
            return origin
        parts = parts[:-1]
    return None


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
    # The mapping network is not per layer, so the first entry decides which
    # model's whole mapping the mix inherits, its build arguments included.
    mapping_origin = entries[0] if entries and entries[0] in (ORIGIN_A, ORIGIN_B) else None
    mapping_source = G_b if mapping_origin == ORIGIN_B else G_a
    mixed = custom_stylegan2.Generator(
        z_dim=G_a.z_dim,
        c_dim=G_a.c_dim,
        w_dim=G_a.w_dim,
        img_channels=G_a.img_channels,
        img_resolution=img_resolution,
        mapping_kwargs=_mapping_kwargs(mapping_source),
        # Passed explicitly because Generator declares this as a mutable
        # default and then updates it in place.
        synthesis_kwargs={"channels_dict": channels},
    )
    destination = mixed.state_dict()
    state_a = G_a.state_dict()
    state_b = G_b.state_dict()
    state_by_origin = {ORIGIN_A: state_a, ORIGIN_B: state_b}
    if mapping_origin is not None:
        for name in mapping_names(mapping_source):
            destination[name] = state_by_origin[mapping_origin][name]
    for index, entry in enumerate(entries):
        if entry == ORIGIN_A:
            name = names_a[index]
            destination[name] = state_a[name]
        elif entry == ORIGIN_B:
            name = names_b[index]
            destination[name] = state_b[name]
    # Buffers are not parameters, so nothing above has touched them, and a
    # mixed model left with its own freshly constructed ones is not the model
    # it was assembled from: `w_avg` would be zero, which moves every frame
    # rendered below truncation 1, and each `noise_const` would be a fresh
    # draw, which changes the picture outright wherever noise strength is not
    # zero. Routed by the weights they serve, so all-A is bit identical to A.
    module_origins = _origin_by_module(entries, names_a, names_b)
    parameter_names = {name for name, _ in mixed.named_parameters()}
    for key in destination:
        if key in parameter_names:
            continue
        origin = _buffer_origin(key, module_origins, mapping_origin)
        if origin is None:
            continue
        source = state_by_origin[origin]
        if key in source:
            destination[key] = source[key]
    try:
        mixed.load_state_dict(destination)
    except Exception as exc:
        logger.warning("Could not assemble the mixed model: %s", exc)
        raise ValueError(INCOMPATIBLE_MODELS) from exc
    return mixed
