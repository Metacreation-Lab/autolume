"""Generate demo presets that show off the live runtime's bending and mixing
features against a real model.

    uv run tools/make_demo_presets.py <model.pkl> [model_b.pkl]

Layer names and channel counts depend on the loaded model's resolution (a
256px model and a 1024px model have a different block ladder), so a
handwritten preset could name a layer that does not exist in a given model.
This script loads the real model instead, reads its actual layer catalog,
and builds presets matched to it.

Every preset is one half of a pair that isolates a single variable, so
loading the two side by side teaches what that variable does:

- bend-position-1/2: the same bend, once early (low resolution) and once
  late, since where you bend matters more than what you bend.
- bend-operator-<op>: one preset per exposed bending operator, all at the
  same layer, directly comparable.
- bend-channels-1/2: the same bend on every channel versus a small subset.
- capture-layer-1/2: an intermediate activation viewed directly, once in
  color and once in grayscale at a non-zero base channel.
- mix-1..4 (only with a second model): all-A, all-B, a split at a middle
  resolution, and a trailing "X" depth truncation.

Every generated name starts with "demo-" so the set is obvious in the
preset list and easy to delete. Presets are read into shape and validated
before anything is written, and an existing preset is only ever overwritten
with a printed notice, never silently.
"""

import argparse
import sys
from pathlib import Path

# tools/ is not part of the installed package and this file is invoked
# directly (`uv run tools/make_demo_presets.py`), not via `-m`: Python sets
# sys.path[0] to this file's own directory rather than the repo root.
# autolume itself resolves fine either way (it is installed editable), but
# presets.preset_dir() reaches into the legacy root-level `utils` package,
# which is only importable if the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from autolume.live.core import mixing, presets
from autolume.live.core.generator import LayerInfo, load_model
from autolume.live.core.params import ControlState, Transform

PREFIX = "demo-"

# Sensible, visible-but-not-destroying values for the operator survey.
# Arities and guards mirror mapping.py's `_OPERATOR_ARITY`,
# `_MIN_SCALE_MAGNITUDE` and the erode/dilate integral-kernel check:
# translate takes two params, every other exposed operator takes one;
# scale's factor must not sit near zero, and erode/dilate's kernel must be a
# whole number of 1 or more.
_ROTATE_DEGREES = 25.0
_SCALE_FACTOR = 1.5
_ERODE_KERNEL = 3.0
_DILATE_KERNEL = 3.0
_BINARY_THRESHOLD = 0.0
_SCALAR_MULTIPLY = 1.5
_TRUTHY = 1.0  # invert / flip-h / flip-v / ablate read params[0] as a flag

# The eleven operators mapping.py's `_validate_transform` accepts. sobel,
# canny and resize are deliberately unexposed and stay out of this survey.
OPERATORS = (
    "translate",
    "rotate",
    "scale",
    "erode",
    "dilate",
    "invert",
    "flip-h",
    "flip-v",
    "binary-thresh",
    "scalar-multiply",
    "ablate",
)


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_catalog(path: str, device=None):
    """Load `path` and return `(LoadedModel, layer catalog)`, or fail clearly."""
    try:
        model = load_model(path, device=device) if device is not None else load_model(path)
    except Exception as exc:
        fail(
            f"Could not load {path}: {exc}\n"
            "Check the path is correct and that it is a StyleGAN .pkl checkpoint."
        )
    layers = model.enumerate_layers()
    if not layers:
        fail(
            f"{path} loaded, but its layer catalog came back empty, so there is "
            "nothing to bend or capture. This usually means the architecture is "
            "not one enumerate_layers understands."
        )
    return model, layers


def conv_layers(layers: tuple[LayerInfo, ...]) -> list[LayerInfo]:
    """Every addressable synthesis layer except the final "output" alias."""
    return [layer for layer in layers if layer.name != "output"]


def pick_position_pair(layers: tuple[LayerInfo, ...]) -> tuple[LayerInfo, LayerInfo]:
    """The earliest and latest synthesis blocks, for the layer position demo."""
    convs = conv_layers(layers)
    if len(convs) < 2:
        fail(
            "This model has fewer than two synthesis layers, there is no "
            "early/late pair to demonstrate layer position with."
        )
    return convs[0], convs[-1]


def pick_survey_layer(layers: tuple[LayerInfo, ...]) -> LayerInfo:
    """One well-chosen mid-resolution layer, shared by the operator survey,
    the channel selection pair and the capture layer demo, so those stay
    directly comparable to each other.
    """
    convs = conv_layers(layers)
    widths = sorted({layer.width for layer in convs})
    middle_width = widths[len(widths) // 2]
    candidates = [layer for layer in convs if layer.width == middle_width]
    # Richer channel count first: every operator gets more to visibly act on.
    return max(candidates, key=lambda layer: layer.channels)


def all_indices(layer: LayerInfo) -> tuple[int, ...]:
    return tuple(range(layer.channels))


def subset_indices(layer: LayerInfo, count: int = 3) -> tuple[int, ...]:
    return tuple(range(min(count, layer.channels)))


def translate_offset(width: int) -> float:
    """A pixel shift visible at this layer's own resolution, never zero."""
    return float(max(1, round(width * 0.2)))


def operator_transform(op: str, layer: LayerInfo, indices: tuple[int, ...]) -> Transform:
    """One demo `Transform` for `op` at `layer`, with sensible parameters."""
    if op == "translate":
        params = (translate_offset(layer.width), 0.0)
    elif op == "rotate":
        params = (_ROTATE_DEGREES,)
    elif op == "scale":
        params = (_SCALE_FACTOR,)
    elif op in ("erode", "dilate"):
        params = (_ERODE_KERNEL if op == "erode" else _DILATE_KERNEL,)
    elif op == "binary-thresh":
        params = (_BINARY_THRESHOLD,)
    elif op == "scalar-multiply":
        params = (_SCALAR_MULTIPLY,)
    else:  # invert, flip-h, flip-v, ablate
        params = (_TRUTHY,)
    return Transform(op=op, layer=layer.name, params=params, indices=indices)


def all_selection(origin: str, names: tuple[str, ...], total: int) -> tuple[str, ...]:
    """Every layer from `origin`, padded with a trailing "X" run for any
    depth the other model has beyond it. `combine()` rejects an origin past
    its own catalog length, so this can never claim more than `names` holds,
    which is what keeps an all-A/all-B preset a true control.
    """
    return (origin,) * len(names) + (mixing.ORIGIN_X,) * (total - len(names))


def split_selection(
    names_a: tuple[str, ...], names_b: tuple[str, ...], total: int
) -> tuple[str, ...]:
    """A boundary split at the middle shared resolution: A up to it, B from
    it onward, padded with a trailing "X" run for any depth beyond the
    shorter of the two models.
    """
    shared = min(len(names_a), len(names_b))
    if shared == 0:
        fail("One of the two models has no synthesis layers, there is nothing to split.")
    resolutions = [mixing.layer_resolution(name) for name in names_a[:shared]]
    distinct = sorted(set(resolutions))
    if len(distinct) < 2:
        fail(
            f"These two models share only one resolution stage in their first "
            f"{shared} layers, there is no middle to split at."
        )
    middle = distinct[len(distinct) // 2]
    split = next(i for i, res in enumerate(resolutions) if res == middle)
    entries = ["A"] * split + ["B"] * (shared - split) + ["X"] * (total - shared)
    return tuple(entries)


def truncated_selection(names_a: tuple[str, ...], total: int) -> tuple[str, ...]:
    """Model A with its own last resolution stage dropped via a trailing "X"
    run, distinct from `all_selection`'s mere depth padding: this one always
    removes real layers A actually has.
    """
    resolutions = [mixing.layer_resolution(name) for name in names_a]
    last_resolution = resolutions[-1]
    keep = sum(1 for res in resolutions if res != last_resolution)
    if keep == 0:
        fail("Model A only has one resolution stage, there is nothing to truncate.")
    return tuple(["A"] * keep + ["X"] * (total - keep))


def build_bending_presets(
    model_a_path: str, layers: tuple[LayerInfo, ...]
) -> list[tuple[str, ControlState]]:
    built: list[tuple[str, ControlState]] = []
    early, late = pick_position_pair(layers)
    survey_layer = pick_survey_layer(layers)

    # Bending, layer position: the same operator and parameters, once on the
    # earliest (lowest resolution) block and once on the latest. The single
    # most instructive pair, since it shows that where you bend matters more
    # than what you bend.
    built.append((
        f"{PREFIX}bend-position-1-early-{early.name}",
        ControlState(
            pkl_path=model_a_path,
            transforms=(operator_transform("rotate", early, all_indices(early)),),
        ),
    ))
    built.append((
        f"{PREFIX}bend-position-2-late-{late.name}",
        ControlState(
            pkl_path=model_a_path,
            transforms=(operator_transform("rotate", late, all_indices(late)),),
        ),
    ))

    # Bending, operator survey: one preset per operator family, all at the
    # same well-chosen layer, so they are directly comparable. Ablate on
    # every channel of the layer would zero it out entirely and take the
    # rest of the network down with it, so it alone gets a subset.
    for op in OPERATORS:
        indices = subset_indices(survey_layer) if op == "ablate" else all_indices(survey_layer)
        built.append((
            f"{PREFIX}bend-operator-{op}",
            ControlState(
                pkl_path=model_a_path,
                transforms=(operator_transform(op, survey_layer, indices),),
            ),
        ))

    # Bending, channel selection: the same operator, layer and parameters,
    # once on every channel and once on a small subset.
    built.append((
        f"{PREFIX}bend-channels-1-all",
        ControlState(
            pkl_path=model_a_path,
            transforms=(operator_transform("scale", survey_layer, all_indices(survey_layer)),),
        ),
    ))
    built.append((
        f"{PREFIX}bend-channels-2-subset",
        ControlState(
            pkl_path=model_a_path,
            transforms=(
                operator_transform("scale", survey_layer, subset_indices(survey_layer)),
            ),
        ),
    ))

    # Capture layer: view an intermediate activation directly instead of the
    # final image, since that is the clearest way to see what the network is
    # actually building. One plain color view, one grayscale at a non-zero
    # base channel so the channel window itself is demonstrated too.
    built.append((
        f"{PREFIX}capture-layer-1-color",
        ControlState(pkl_path=model_a_path, capture_layer=survey_layer.name),
    ))
    stepped_channel = min(survey_layer.channels - 1, max(0, survey_layer.channels // 2))
    built.append((
        f"{PREFIX}capture-layer-2-grayscale",
        ControlState(
            pkl_path=model_a_path,
            capture_layer=survey_layer.name,
            grayscale=True,
            base_channel=stepped_channel,
        ),
    ))
    return built


def build_mixing_presets(
    model_a_path: str, model_b_path: str, model_a, model_b
) -> list[tuple[str, ControlState]]:
    names_a = mixing.conv_names(model_a.G)
    names_b = mixing.conv_names(model_b.G)
    total = mixing.selection_length(model_a.G, model_b.G)

    # Mixing: all-A and all-B are the controls that make the split legible,
    # a split at a middle resolution, and a trailing "X" truncation.
    variants = (
        ("mix-1-all-a", all_selection("A", names_a, total)),
        ("mix-2-all-b", all_selection("B", names_b, total)),
        ("mix-3-split-middle", split_selection(names_a, names_b, total)),
        ("mix-4-truncate-x", truncated_selection(names_a, total)),
    )
    built: list[tuple[str, ControlState]] = []
    for name, entries in variants:
        # Validated against the real pair before anything is written: a
        # selection that does not line up, or a pair that cannot be
        # assembled at all, must fail here with the documented reason
        # rather than as a broken preset the app discovers later.
        try:
            mixing.combine(model_a.G, model_b.G, entries)
        except ValueError as exc:
            fail(f"The {name!r} mixing preset would not load: {exc}")
        built.append((
            f"{PREFIX}{name}",
            ControlState(
                pkl_path=model_a_path,
                pkl2=model_b_path,
                mixing_enabled=True,
                combined_layers=entries,
            ),
        ))
    return built


def write_presets(directory: Path, built: list[tuple[str, ControlState]]) -> None:
    for name, state in built:
        path = directory / f"{name}.json"
        if path.exists():
            print(f"  overwriting existing preset {name!r}")
        presets.save(state, str(path))
        print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate demo presets that show off the live runtime's bending "
            "and mixing features, matched against a real model's layer catalog."
        )
    )
    parser.add_argument("model_a", help="Path to a .pkl model to bend and capture from.")
    parser.add_argument(
        "model_b",
        nargs="?",
        default=None,
        help="Optional second .pkl model, needed only for the mixing presets.",
    )
    args = parser.parse_args()

    model_a_path = str(Path(args.model_a).expanduser().resolve())
    model_a, layers_a = load_catalog(model_a_path)

    built = build_bending_presets(model_a_path, layers_a)

    if args.model_b:
        model_b_path = str(Path(args.model_b).expanduser().resolve())
        # Slot B is never rendered, only a weight source for combine(), the
        # same convention the live ModelHost keeps it under: CPU, always.
        model_b, _layers_b = load_catalog(model_b_path, device=torch.device("cpu"))
        built.extend(build_mixing_presets(model_a_path, model_b_path, model_a, model_b))

    directory = presets.preset_dir()
    print(f"Writing {len(built)} demo presets to {directory}")
    write_presets(directory, built)
    print(
        f"Done. Open the preset list in the live runtime, every name here "
        f"starts with {PREFIX!r}."
    )


if __name__ == "__main__":
    main()
