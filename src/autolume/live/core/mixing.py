"""Network mixing: one generator assembled from two loaded models.

Ported from `modules/network_mixing.py`'s `combine_models`, with the per
resolution channel merge the live path needs so a pair whose blocks are
different widths still assembles into a network that can be built at all.

Everything here is a pure function over the two source generators and the
selection. No threading, no device handling, no host state: `ModelHost`
calls `combine` on the loader thread and owns what happens to the result.

The selection is one entry per name in `conv_names`, aligned by index, and
padded to whichever model has more layers. "A" and "B" name the model a
layer's weights come from, and "X" removes the layer, which may only
truncate the tail and so lowers the mixed model's output resolution. The
first entry carries more than its own layer: the mapping network is not
per layer, so the mix inherits the whole of it from the model that entry
names, and it may not be an "X".
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


def _check_truncation_only(entries: tuple[str, ...]) -> None:
    """Enforce what `_output_resolution` documents: X only ever truncates.

    A removed layer with a kept layer after it is not a shorter network, it
    is the same network with one tensor left at its construction draw, and
    that builds, renders and saves without a word. The panel's own X always
    cuts a trailing run (`ui/panels/mixing.py::cut`), so this only ever
    refuses a selection that came from a preset file or from OSC.

    Called after the channel merge rather than from inside
    `_output_resolution`, so a whole block removed from the middle keeps the
    merge's own sentence, which names the resolution the performer sees on
    the row instead of a layer number they have no way to count.
    """
    kept = [index for index, entry in enumerate(entries) if entry not in ("", ORIGIN_X)]
    if not kept:
        return
    for index, entry in enumerate(entries[: kept[-1]]):
        if entry in ("", ORIGIN_X):
            raise ValueError(
                f"Layer {index + 1} is removed but a later layer is kept. "
                "Removing a layer can only shorten the end of the network."
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

    Keyed by every ancestor module path of every chosen parameter, so a
    lookup can walk up from a buffer's own module until it finds the weights
    that buffer serves.

    A module's parameters can be split across the two models, since the
    selection is per parameter and not per module. `setdefault` resolves
    that: the module goes to its **first parameter in network order**, which
    for a synthesis layer is its `weight`, the one a `noise_const` is
    actually added alongside.
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
    key: str, module_origins: dict[str, str], mapping_origin: str
) -> str | None:
    """The model a non-parameter state entry should be taken from.

    A buffer belongs with the weights it serves. `mapping.w_avg` is a
    statistic of one mapping network's own W distribution, so it follows
    whichever model provided the mapping; a synthesis buffer follows the
    weights of the nearest enclosing module that has any, which for a split
    module is decided by `_origin_by_module`'s first-parameter rule.
    Anything that cannot be placed keeps the value it was constructed with.
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
    # The mapping network is not per layer, so the first entry decides which
    # model's whole mapping the mix inherits, its build arguments included.
    # A first entry naming neither model leaves nothing to inherit it from,
    # and the mix then carries a freshly drawn mapping, `w_avg` at zero and a
    # random `b4.const`, which builds and renders and says nothing. Refused
    # here rather than repaired, because there is no repair a caller could
    # have meant: the first layer is the one layer that is not optional.
    first = entries[0] if entries else ""
    mapping_origin = first if first in (ORIGIN_A, ORIGIN_B) else None
    if mapping_origin is None:
        raise ValueError(
            f"The first layer must come from A or B, not {first!r}. The mixed "
            "model takes its whole mapping network from it."
        )
    mapping_source = G_b if mapping_origin == ORIGIN_B else G_a
    img_resolution = _output_resolution(entries, names_a, names_b)
    channels = _merged_channels(
        entries, names_a, names_b, G_a, G_b, img_resolution
    )
    _check_truncation_only(entries)
    # Construction is inside the guard, not just the load: the mapping
    # arguments come out of a user's pkl and go straight into a constructor
    # with no `**kwargs`, so an unexpected key there is a way for a source
    # file to reach this module. Everything from here on reports as one
    # documented sentence rather than a raw Python error.
    try:
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
        for name in mapping_names(mapping_source):
            destination[name] = state_by_origin[mapping_origin][name]
        for index, entry in enumerate(entries):
            if entry == ORIGIN_A:
                name = names_a[index]
                destination[name] = state_a[name]
            elif entry == ORIGIN_B:
                name = names_b[index]
                destination[name] = state_b[name]
        # Buffers are not parameters, so nothing above has touched them, and
        # a mixed model left with its own freshly constructed ones is not the
        # model it was assembled from: `w_avg` would be zero, which moves
        # every frame rendered below truncation 1, and each `noise_const`
        # would be a fresh draw, which changes the picture outright wherever
        # noise strength is not zero. Routed by the weights they serve, so
        # all-A is bit identical to A.
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
        mixed.load_state_dict(destination)
    except Exception as exc:
        logger.warning("Could not assemble the mixed model: %s", exc)
        raise ValueError(INCOMPATIBLE_MODELS) from exc
    return mixed
