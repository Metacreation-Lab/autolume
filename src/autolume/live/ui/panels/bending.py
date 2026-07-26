"""Bending panel: editing the network, not performing it.

Four things live here, and they share one thing: they are all edits to how a
frame is derived rather than parameters a performer plays. The layer list and
its transform chain, the per layer noise, the adjuster's directions, and the
image derivation.

The transform chain is structured state (`ControlState.transforms`), reached
through `/bend/set` and `/bend/remove` value objects rather than through
addresses of its own. That is deliberate and settled: the legacy app gave
every transform parameter an OSC address whose handler ran `eval` on a mapping
string, and reproducing that shape is what this rebuild exists to avoid. A
transform parameter is edited here or driven through the adjuster.

Channel selections resolve to explicit indices when Apply is pressed, and stay
put until Apply is pressed again. The old app re-rolled a random selection
every frame, which read as a shimmer nobody asked for and made a saved look
unrepeatable. The selection editor is a UI affordance on top of the indices,
so a cluster YAML file is read here, at pick time, and never by anything under
`core/`.

The adjuster's weights are eight fixed registry parameters and so are ordinary
bound rows. Its directions are structured state. A slot with no direction
behind it is greyed, because the weights beyond `len(directions)` are zeroed
once, when directions load, and nothing re-zeroes them afterwards: a later
write to `/adjust/6` with three directions loaded puts a stale weight back
that no direction is left to use.
"""

import logging
import random
from typing import Callable, Sequence

import numpy as np
from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import LayerInfo
from autolume.live.core.params import (
    ADJUST_DIRECTIONS,
    BEND_NOISE,
    BEND_RATIO,
    BEND_REMOVE,
    BEND_SET,
    ControlState,
    RemoveTransform,
    SetDirections,
    SetLayerNoise,
    SetLayerRatio,
    SetTransform,
    Transform,
)
from autolume.live.errors import describe
from autolume.live.ui.controls import ControlBinder
from autolume.live.ui.panels.perform import (
    combo_index,
    fit_item,
    paired_control_width,
    same_line_if_it_fits,
    string_combo,
    trailing_width,
)
from autolume.live.ui.theme import ERROR_COLOR

logger = logging.getLogger(__name__)

# Every operator the UI exposes, with how many parameters it takes and what
# kind each one is. Restated here rather than imported from `core/mapping.py`,
# whose table is private, and held to that table by a test
# (`test_bending_panel.py`) so the two cannot drift: a UI offering an operator
# the validator rejects would draw a row that silently never applies.
#
# The kinds come from `bending/transform_layers.py`'s own reads of `params[0]`.
# "switch" operators test it for truth and nothing else, so they are drawn as
# checkboxes. `erode` and `dilate` size a `torch.ones` kernel with it, so it is
# a whole number of at least one. Everything else reads it as a number.
OPERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("translate", ("float", "float")),
    ("rotate", ("float",)),
    ("scale", ("float",)),
    ("erode", ("kernel",)),
    ("dilate", ("kernel",)),
    ("invert", ("switch",)),
    ("flip-h", ("switch",)),
    ("flip-v", ("switch",)),
    ("binary-thresh", ("float",)),
    ("scalar-multiply", ("float",)),
    ("ablate", ("switch",)),
)
OPERATOR_NAMES = tuple(name for name, _ in OPERATORS)
_PARAM_KINDS = dict(OPERATORS)

# What a freshly added operator starts at. A switch starts on, so adding one
# does something visible. `scale` starts at 1 and `scalar-multiply` at 1
# because those are their neutral values, and `scale` is refused outright below
# a magnitude the validator enforces, so zero is never a safe start.
_PARAM_DEFAULTS: dict[str, tuple[float, ...]] = {
    "translate": (0.0, 0.0),
    "rotate": (0.0,),
    "scale": (1.0,),
    "erode": (3.0,),
    "dilate": (3.0,),
    "invert": (1.0,),
    "flip-h": (1.0,),
    "flip-v": (1.0,),
    "binary-thresh": (0.0,),
    "scalar-multiply": (1.0,),
    "ablate": (1.0,),
}

# What a parameter field is labelled, per operator. Two entries only, because
# only `translate` has more than one parameter and only a few operators have a
# name for theirs worth reading.
_PARAM_LABELS: dict[str, tuple[str, ...]] = {
    "translate": ("x", "y"),
    "rotate": ("degrees",),
    "scale": ("factor",),
    "erode": ("kernel",),
    "dilate": ("kernel",),
    "binary-thresh": ("threshold",),
    "scalar-multiply": ("factor",),
}

# The selection modes the index editor offers.
MODE_ALL = "all"
MODE_RANDOM = "random"
MODE_RANGE = "range"
MODE_CLUSTER = "cluster"
SELECTION_MODES = (MODE_ALL, MODE_RANDOM, MODE_RANGE, MODE_CLUSTER)
SELECTION_LABELS = ("All channels", "Random percent", "Range", "Cluster file")

OUTPUT_LAYER = "output"
_CAPTURE_OUTPUT_LABEL = "Final output"
_ADJUST_SLOTS = 8
_CLUSTER_FILTER = ["Cluster files", "*.yaml *.yml"]
_DIRECTION_FILTER = ["Direction files", "*.npy"]
_NO_MODEL = "Load a model to see its layers."
_NO_LAYERS = "This model published no layer catalog."
_NO_TRANSFORMS = "No transforms on this layer yet."
_NO_DIRECTIONS = "No directions loaded. Load a file or randomize."
_TORGB_NOTE = "A torgb layer carries no noise of its own."
_REMOVE = "Remove"
_REROLL = "Re-roll"
_LOAD = "Load"
_LOAD_ALL = "Load all"
_RANDOMIZE = "Randomize"
# Sized in multiples of the font size so the panel keeps its proportions at
# every display scale.
_NUMBER_EMS = 5.0
_PERCENT_EMS = 4.5
# A random selection is drawn from this generator, so the same seed gives the
# same channels. Re-rolling is an explicit press, never a frame.
_SEED_CEILING = 2**31 - 1


def is_bendable(name: str) -> bool:
    """Whether a catalog layer is one the panel offers.

    The same filter the old widget used: the synthesis output, the convolution
    layers, and the tRGB layers. Everything else a `named_modules()` walk turns
    up (activations, upsamplers, the blocks themselves) is either not a tensor
    a transform can act on or is the same tensor under another name.
    """
    if name == OUTPUT_LAYER:
        return True
    leaf = name.rsplit(".", 1)[-1]
    return leaf.startswith("conv") or leaf.startswith("torgb")


def is_torgb(name: str) -> bool:
    return name.rsplit(".", 1)[-1].startswith("torgb")


def visible_layers(
    layers: Sequence[LayerInfo], simple: bool
) -> tuple[LayerInfo, ...]:
    """The rows the layer list draws, filtered and optionally thinned.

    Advanced is every layer the filter keeps, in catalog order, which is
    network order. Simple is one row per resolution, the first layer at that
    resolution, plus the output: a performer who wants to bend "the 32s" does
    not want to pick between three of them, and the resolutions are what the
    picture actually reads as.

    Keyed on the layer's own width rather than on a resolution ladder counted
    up from four, which is what the legacy widget did and which silently
    dropped every row past a gap in the ladder.
    """
    kept = [layer for layer in layers if is_bendable(layer.name)]
    if not simple:
        return tuple(kept)
    seen: set[int] = set()
    rows: list[LayerInfo] = []
    for layer in kept:
        if layer.name == OUTPUT_LAYER:
            rows.append(layer)
            continue
        if layer.width in seen:
            continue
        seen.add(layer.width)
        rows.append(layer)
    return tuple(rows)


def layer_marks(
    name: str,
    transforms: Sequence[Transform],
    layer_noise: Sequence[tuple[str, float]],
    layer_ratios: Sequence[tuple[str, float, float]],
) -> tuple[bool, bool]:
    """`(carries a transform, carries noise or a ratio)` for one layer's row.

    Two facts, drawn as two shapes in the row's gutter, because both are edits
    a performer makes on one layer and then goes looking for again from a list
    of twenty.
    """
    bent = any(transform.layer == name for transform in transforms)
    noisy = any(layer == name and strength for layer, strength in layer_noise)
    stretched = any(
        layer == name and (rx, ry) != (1.0, 1.0) for layer, rx, ry in layer_ratios
    )
    return bent, noisy or stretched


def layer_noise_value(
    layer_noise: Sequence[tuple[str, float]], name: str
) -> float:
    """One layer's noise strength. Absent means neutral, which is zero."""
    for layer, strength in layer_noise:
        if layer == name:
            return float(strength)
    return 0.0


def layer_ratio_value(
    layer_ratios: Sequence[tuple[str, float, float]], name: str
) -> tuple[float, float]:
    """One layer's noise ratio pair. Absent means neutral, which is (1, 1)."""
    for layer, rx, ry in layer_ratios:
        if layer == name:
            return float(rx), float(ry)
    return 1.0, 1.0


def param_kinds(op: str) -> tuple[str, ...]:
    """What kind of field each of `op`'s parameters is drawn as."""
    return _PARAM_KINDS.get(op, ())


def param_label(op: str, index: int) -> str:
    """A parameter's label, or an empty one when it has no name worth reading."""
    labels = _PARAM_LABELS.get(op, ())
    return labels[index] if index < len(labels) else ""


def default_params(op: str) -> tuple[float, ...]:
    return _PARAM_DEFAULTS.get(op, ())


def params_for_op(op: str, previous: Sequence[float]) -> tuple[float, ...]:
    """`op`'s parameters, keeping what the previous operator's still fit.

    Switching between two operators of the same arity keeps the number that
    was already dialled in, which is what makes trying dilate against erode a
    single click. Growing or shrinking the arity fills from the defaults rather
    than leaving a slot uninitialised, and a switch operator never inherits a
    number, since its parameter is a truth value and any nonzero number reads
    as on.
    """
    kinds = param_kinds(op)
    defaults = default_params(op)
    values = []
    for index, kind in enumerate(kinds):
        if kind == "switch" or index >= len(previous):
            values.append(defaults[index])
        else:
            values.append(float(previous[index]))
    return tuple(values)


def clamp_param(op: str, index: int, value: float) -> float:
    """A typed parameter, brought inside what the validator will accept.

    Done here rather than left to the validator because the validator's answer
    to an unacceptable value is to drop the whole transform, silently. A kernel
    of zero and a scale of zero are both easy to type on the way to something
    else, so both are corrected in the field instead of costing the edit.
    """
    kinds = param_kinds(op)
    kind = kinds[index] if index < len(kinds) else "float"
    if kind == "kernel":
        return float(max(1, round(value)))
    if op == "scale" and abs(value) < 1e-6:
        return 1e-6 if value >= 0.0 else -1e-6
    return float(value)


class IndexSelection:
    """The channel selection editor's own state for one transform.

    Not part of `ControlState`: what the control thread and a preset carry is
    the resolved `indices` tuple, and this is only how a performer arrived at
    it. That split is what makes the resolution happen once, at Apply, rather
    than every frame.
    """

    def __init__(self) -> None:
        self.mode = MODE_ALL
        self.percent = 50.0
        self.low = 0
        self.high = 0
        self.cluster_path = ""
        self.cluster_id = 0
        self.seed = 0
        # Parsed at pick time and held, so resolving does not reread the file.
        # None means nothing has been loaded, which greys Apply.
        self.cluster_config: dict | None = None
        self.error: str | None = None


def resolve_indices(
    selection: IndexSelection, layer: str, channels: int
) -> tuple[int, ...]:
    """The explicit channels a selection means, for a layer this wide.

    Pure, and the whole point of the editor: the answer is computed once, when
    Apply is pressed, and stored on the transform. Every mode clamps to the
    layer, so a selection authored against a 512 channel layer and applied to a
    128 channel one narrows rather than reaching past the end of a tensor.

    An empty result is a legitimate answer and is returned as one. A transform
    with no channels selected applies to nothing, which is what "zero percent"
    and "an empty range" both mean.
    """
    channels = max(0, int(channels))
    if selection.mode == MODE_ALL:
        return tuple(range(channels))
    if selection.mode == MODE_RANDOM:
        percent = min(max(float(selection.percent), 0.0), 100.0)
        count = int(channels * percent / 100.0)
        count = min(count, channels)
        if count <= 0:
            return ()
        picked = random.Random(selection.seed).sample(range(channels), count)
        return tuple(sorted(picked))
    if selection.mode == MODE_RANGE:
        low = min(max(int(selection.low), 0), channels)
        high = min(max(int(selection.high), 0), channels)
        return tuple(range(low, high))
    return cluster_indices(
        selection.cluster_config, layer, selection.cluster_id, channels
    )


def cluster_indices(
    config: dict | None, layer: str, cluster_id: int, channels: int
) -> tuple[int, ...]:
    """The channels a cluster file assigns to `cluster_id` on `layer`.

    The file shape is the old app's: a mapping of layer name to a list of
    entries, each carrying a `cluster_index` and a `feature_index`. Anything
    that does not match that shape contributes nothing rather than raising,
    because this runs on a file a performer picked and a malformed one is not
    worth losing the panel over.
    """
    if not isinstance(config, dict):
        return ()
    entries = config.get(layer)
    if not isinstance(entries, list):
        return ()
    picked = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            if int(entry.get("cluster_index")) != int(cluster_id):
                continue
            index = int(entry.get("feature_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < channels:
            picked.append(index)
    return tuple(sorted(set(picked)))


def cluster_ids(config: dict | None, layer: str) -> tuple[int, ...]:
    """Every cluster a file names for `layer`, so the picker can offer them."""
    if not isinstance(config, dict):
        return ()
    entries = config.get(layer)
    if not isinstance(entries, list):
        return ()
    found = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            found.add(int(entry.get("cluster_index")))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(found))


def load_cluster_file(path: str) -> dict:
    """Read a cluster YAML file. Raises, so the caller can report it.

    `yaml.safe_load`, never `load`: this is a file a performer picked, and the
    unsafe loader constructs arbitrary Python objects from it. `yaml` is
    imported here rather than at module scope so a session that never opens a
    cluster file never pays for it.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("A cluster file must be a mapping of layer to clusters.")
    return config


def load_direction_file(path: str) -> tuple[tuple[float, ...], ...]:
    """Read one or more direction vectors from a `.npy` file.

    A 1D file is one direction, a 2D file is a batch of them, which is the
    same pair of shapes the old adjuster accepted. Anything deeper is refused
    by name rather than reshaped into something that would silently be wrong.
    """
    array = np.asarray(np.load(path), dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            "A direction file must hold one vector or a list of vectors."
        )
    return tuple(tuple(float(value) for value in row) for row in array)


def merged_directions(
    current: Sequence[Sequence[float]],
    loaded: Sequence[Sequence[float]],
    slot: int | None,
) -> tuple[tuple[float, ...], ...] | None:
    """The direction set after a load, batch or into one slot, or None.

    `slot` None replaces the whole set, capped at eight. A slot replaces just
    that one, and appends when the slot is the next free one, which is how a
    performer fills the list one file at a time. A slot beyond the next free
    one cannot be reached: the UI does not draw a Load for it.

    None is a refusal, and the caller says why. Refused: an empty file, a file
    holding more than one vector aimed at a single slot, and a vector of a
    different length from the ones already loaded, since they are summed
    against one W row. A refusal is its own answer rather than "the set came
    back unchanged", which a performer loading the same file twice would
    otherwise be told was a failure.
    """
    existing = tuple(tuple(float(v) for v in row) for row in current)
    incoming = tuple(tuple(float(v) for v in row) for row in loaded)
    if not incoming:
        return None
    if slot is None:
        return incoming[:_ADJUST_SLOTS]
    if len(incoming) != 1:
        return None
    vector = incoming[0]
    if existing and len(vector) != len(existing[0]):
        return None
    if slot == len(existing) and slot < _ADJUST_SLOTS:
        return existing + (vector,)
    if 0 <= slot < len(existing):
        return existing[:slot] + (vector,) + existing[slot + 1 :]
    return None


def randomized_directions(
    current: Sequence[Sequence[float]], width: int, seed: int
) -> tuple[tuple[float, ...], ...]:
    """Fresh normal directions, as many as there are now, or one if there are none.

    Width comes from the model rather than from a constant, so a randomize on a
    512 wide model and one on a 1024 wide model each produce vectors the sum
    against W can actually use.
    """
    count = max(1, len(current))
    generator = random.Random(seed)
    return tuple(
        tuple(generator.gauss(0.0, 1.0) for _ in range(width)) for _ in range(count)
    )


def capture_choices(
    layers: Sequence[LayerInfo],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(values, labels)` for the capture layer combo.

    The empty string comes first and means the final output, which is what
    `capture_layer` defaults to and what the render path reads an empty value
    as. `output` is dropped from the rest of the list because it is the same
    tensor under a different name and offering it twice invites the question of
    how they differ.
    """
    values = [""]
    labels = [_CAPTURE_OUTPUT_LABEL]
    for layer in layers:
        if not is_bendable(layer.name) or layer.name == OUTPUT_LAYER:
            continue
        values.append(layer.name)
        labels.append(f"{layer.name} ({layer.channels} x {layer.width})")
    return tuple(values), tuple(labels)


def selected_layer(layers: Sequence[LayerInfo], name: str) -> LayerInfo | None:
    """The catalog entry `name` refers to, or the first row when it is gone.

    A layer selection outlives the model it was made against, so a model swap
    can leave the panel pointing at a name the new network has no version of.
    Falling back to the first row keeps the sections below the list populated
    instead of blanking them.
    """
    for layer in layers:
        if layer.name == name:
            return layer
    return layers[0] if layers else None


class BendingPanel:
    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        self._simple = True
        self._layer = ""
        # One editor per transform, kept aligned to `ControlState.transforms`
        # by position. Rebuilt to defaults whenever the two disagree, which is
        # what a preset load looks like from here.
        self._selections: list[IndexSelection] = []
        self._cluster_dialog: tuple[int, pfd.open_file] | None = None
        self._direction_dialog: tuple[int | None, pfd.open_file] | None = None
        self._direction_error: str | None = None
        self._direction_slot = 0

    def gui(self) -> None:
        state = self._binder.state()
        info = self._runtime.model_info_store.snapshot()
        layers = visible_layers(info.layers if info else (), self._simple)
        self._layer_rows(state, layers, info is not None)
        layer = selected_layer(layers, self._layer)
        self._transform_rows(state, layer)
        self._noise_rows(state, layer)
        self._adjuster_rows(state, info)
        self._image_rows(info)
        # Drained from `gui` rather than from the section that opens them: a
        # file dialog is not app modal, so the performer can change the layer,
        # the mode or the whole model while one is open, and the result still
        # has to land on the transform it was opened for.
        self._take_cluster_file(state)
        self._take_direction_file(state)

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _layer_rows(
        self, state: ControlState, layers: Sequence[LayerInfo], loaded: bool
    ) -> None:
        """The layer list, with Simple and Advanced above it.

        Every row carries two marks in its gutter, one for a transform and one
        for noise, drawn as shapes rather than glyphs because the bundled font
        has no symbols. The gutter is reserved on every row whether or not it
        has anything in it, so a mark appearing does not shift the row under
        the cursor.
        """
        imgui.separator_text("Layers")
        self._detail_row()
        if not layers:
            self._note(_NO_LAYERS if loaded else _NO_MODEL)
            return
        if self._layer == "":
            self._layer = layers[0].name
        for layer in layers:
            self._layer_row(state, layer)

    def _detail_row(self) -> None:
        """Simple or Advanced, on one line where it fits and two where it does not.

        A radio pair, not a checkbox: this is a choice between two ways of
        listing the same layers, and a checkbox reading "Advanced" would look
        like a setting rather than the other half of a pair. Neither radio's
        width can be narrowed, since a label is drawn outside the item it
        belongs to, so the row reflows below the width it needs rather than
        running off the edge. Same treatment, and the same reason, as the
        keyframe entry in the Loop panel.
        """
        if imgui.radio_button("Simple", self._simple):
            self._simple = True
        same_line_if_it_fits(
            imgui.get_style().item_spacing.x + paired_control_width("Advanced")
        )
        if imgui.radio_button("Advanced", not self._simple):
            self._simple = False

    def _layer_row(self, state: ControlState, layer: LayerInfo) -> None:
        imgui.push_id(layer.name)
        bent, noisy = layer_marks(
            layer.name, state.transforms, state.layer_noise, state.layer_ratios
        )
        self._marks(bent, noisy)
        label = f"{layer.name}  {layer.channels} x {layer.width}"
        # Sized explicitly to what is left of the row. A selectable left to
        # itself takes the width of its own label, so a long layer name in a
        # narrow dock at a scaled up font ran past the panel edge instead of
        # being clipped inside it.
        size = imgui.ImVec2(imgui.get_content_region_avail().x, 0.0)
        if imgui.selectable(label, self._layer == layer.name, 0, size)[0]:
            self._layer = layer.name
        imgui.pop_id()

    def _marks(self, bent: bool, noisy: bool) -> None:
        """Two small squares left of a layer row: transforms, then noise.

        Drawn into space an invisible item already reserved, so nothing here
        changes the layout. Colours are the driver marker's, reused rather than
        reinvented: the same green for "something is acting on this" the
        control gutter already means it with.
        """
        height = imgui.get_frame_height()
        width = round(height * 0.9)
        origin = imgui.get_cursor_screen_pos()
        imgui.dummy(imgui.ImVec2(width, height))
        imgui.same_line()
        draw_list = imgui.get_window_draw_list()
        side = round(height * 0.3)
        gap = max(1.0, side * 0.4)
        top = origin.y + (height - side) * 0.5
        for index, on in enumerate((bent, noisy)):
            if not on:
                continue
            left = origin.x + index * (side + gap)
            draw_list.add_rect_filled(
                imgui.ImVec2(left, top),
                imgui.ImVec2(left + side, top + side),
                imgui.get_color_u32(imgui.Col_.check_mark),
            )

    def _transform_rows(self, state: ControlState, layer: LayerInfo | None) -> None:
        """The chain on the selected layer, then Add.

        Indices into `ControlState.transforms` are the whole chain's, not this
        layer's, because that is what `/bend/set` and `/bend/remove` address. A
        row therefore carries the global index it was drawn from and edits go
        straight back to it.
        """
        imgui.separator_text("Transforms")
        if layer is None:
            self._note(_NO_MODEL)
            return
        self._sync_selections(state)
        rows = [
            (index, transform)
            for index, transform in enumerate(state.transforms)
            if transform.layer == layer.name
        ]
        if not rows:
            self._note(_NO_TRANSFORMS)
        for index, transform in rows:
            self._transform_row(index, transform, layer)
        if imgui.button("Add transform"):
            op = OPERATOR_NAMES[0]
            self._selections.append(IndexSelection())
            self._emit(
                BEND_SET,
                SetTransform(
                    len(state.transforms),
                    Transform(
                        op,
                        layer.name,
                        default_params(op),
                        tuple(range(layer.channels)),
                    ),
                ),
            )

    def _sync_selections(self, state: ControlState) -> None:
        """Keep one editor per transform, by position.

        A preset replaces the whole chain in one step, so the editors can be
        the wrong length or describe the wrong transforms entirely. They are
        rebuilt to defaults then, which is honest rather than lossy: the
        resolved indices live on the transform and are untouched, and the row
        shows how many channels are actually selected, so a default editor
        beside a loaded selection does not misreport anything. Nothing is
        applied until Apply is pressed.
        """
        while len(self._selections) < len(state.transforms):
            self._selections.append(IndexSelection())
        del self._selections[len(state.transforms) :]

    def _transform_row(
        self, index: int, transform: Transform, layer: LayerInfo
    ) -> None:
        imgui.push_id(index)
        # The combo gives its width up so Remove stays on the row beside it.
        reserve = trailing_width(_REMOVE)
        picked = string_combo(
            "##op", transform.op, OPERATOR_NAMES, reserve=reserve
        )
        if picked is not None and picked != transform.op:
            self._set(
                index,
                transform,
                op=picked,
                params=params_for_op(picked, transform.params),
            )
        imgui.same_line()
        if imgui.button(_REMOVE):
            self._emit(BEND_REMOVE, RemoveTransform(index))
            if 0 <= index < len(self._selections):
                del self._selections[index]
            imgui.pop_id()
            return
        self._param_fields(index, transform)
        self._selection_editor(index, transform, layer)
        imgui.pop_id()

    def _param_fields(self, index: int, transform: Transform) -> None:
        kinds = param_kinds(transform.op)
        values = list(transform.params)
        changed = False
        for slot, kind in enumerate(kinds):
            if slot >= len(values):
                break
            imgui.push_id(slot)
            label = param_label(transform.op, slot)
            if kind == "switch":
                edited, on = imgui.checkbox(label or "On", bool(values[slot]))
                if edited:
                    values[slot], changed = (1.0 if on else 0.0), True
            else:
                fit_item(label, ems=_NUMBER_EMS)
                edited, number = imgui.input_float(
                    label or "##value", float(values[slot]), 0.0, 0.0, "%.3f",
                    imgui.InputTextFlags_.enter_returns_true,
                )
                if edited:
                    values[slot] = clamp_param(transform.op, slot, number)
                    changed = True
            imgui.pop_id()
        if changed:
            self._set(index, transform, params=tuple(values))

    def _selection_editor(
        self, index: int, transform: Transform, layer: LayerInfo
    ) -> None:
        """Where a transform's channels come from, and the Apply that fixes them.

        Every mode's controls are drawn every frame, the ones the mode does not
        use greyed, so switching modes never changes how tall the row is.
        """
        selection = self._selections[index]
        self._note(f"{len(transform.indices)} of {layer.channels} channels")
        picked = string_combo(
            "##mode", selection.mode, SELECTION_MODES, SELECTION_LABELS
        )
        if picked is not None:
            selection.mode = picked
        self._random_controls(selection)
        self._range_controls(selection, layer)
        self._cluster_controls(index, selection, layer)
        ready = selection.mode != MODE_CLUSTER or selection.cluster_config is not None
        if not ready:
            imgui.begin_disabled()
        if imgui.button("Apply"):
            self._set(
                index,
                transform,
                indices=resolve_indices(selection, layer.name, layer.channels),
            )
        if not ready:
            imgui.end_disabled()
        if selection.error:
            self._error(selection.error)

    def _random_controls(self, selection: IndexSelection) -> None:
        live = selection.mode == MODE_RANDOM
        if not live:
            imgui.begin_disabled()
        fit_item("percent", ems=_PERCENT_EMS, reserve=trailing_width(_REROLL))
        changed, percent = imgui.input_float(
            "percent", float(selection.percent), 0.0, 0.0, "%.0f",
            imgui.InputTextFlags_.enter_returns_true,
        )
        if changed:
            selection.percent = min(max(percent, 0.0), 100.0)
        imgui.same_line()
        # Re-rolling is a press. The old app re-rolled every frame, which is
        # the shimmer this whole editor exists to remove.
        if imgui.button(_REROLL):
            selection.seed = random.randint(0, _SEED_CEILING)
        if not live:
            imgui.end_disabled()

    def _range_controls(self, selection: IndexSelection, layer: LayerInfo) -> None:
        live = selection.mode == MODE_RANGE
        if not live:
            imgui.begin_disabled()
        fit_item("first", ems=_NUMBER_EMS)
        changed_low, low = imgui.input_int(
            "first", int(selection.low), 0, 0, imgui.InputTextFlags_.enter_returns_true
        )
        fit_item("last", ems=_NUMBER_EMS)
        changed_high, high = imgui.input_int(
            "last", int(selection.high), 0, 0, imgui.InputTextFlags_.enter_returns_true
        )
        if changed_low:
            selection.low = min(max(low, 0), layer.channels)
        if changed_high:
            selection.high = min(max(high, 0), layer.channels)
        if not live:
            imgui.end_disabled()

    def _cluster_controls(
        self, index: int, selection: IndexSelection, layer: LayerInfo
    ) -> None:
        live = selection.mode == MODE_CLUSTER
        if not live:
            imgui.begin_disabled()
        if imgui.button("Cluster file"):
            self._cluster_dialog = (
                index,
                pfd.open_file("Choose a cluster file", "", _CLUSTER_FILTER),
            )
        imgui.same_line()
        ids = cluster_ids(selection.cluster_config, layer.name)
        labels = [str(value) for value in ids]
        if labels:
            fit_item("cluster", ems=_NUMBER_EMS)
            changed, chosen = imgui.combo(
                "cluster", combo_index(str(selection.cluster_id), labels), labels
            )
            if changed and 0 <= chosen < len(ids):
                selection.cluster_id = ids[chosen]
        else:
            self._note("No clusters loaded")
        if not live:
            imgui.end_disabled()

    def _take_cluster_file(self, state: ControlState) -> None:
        if self._cluster_dialog is None:
            return
        index, dialog = self._cluster_dialog
        if not dialog.ready():
            return
        self._cluster_dialog = None
        result = dialog.result()
        # Checked against both lists, not just the editors: the dialog resolves
        # on whatever later frame the performer picks a file on, and by then a
        # preset or a Remove can have shortened the chain out from under the
        # index it was opened for. `_sync_selections` only runs while a model is
        # loaded, so the two lists can also disagree outright.
        if not result:
            return
        if not (0 <= index < min(len(self._selections), len(state.transforms))):
            return
        selection = self._selections[index]
        selection.error = None
        try:
            selection.cluster_config = load_cluster_file(str(result[0]))
        except Exception as exc:
            logger.exception("Could not read the cluster file %s", result[0])
            selection.cluster_config = None
            selection.error = f"Could not read the cluster file. {describe(exc)}"
            return
        selection.cluster_path = str(result[0])
        ids = cluster_ids(selection.cluster_config, state.transforms[index].layer)
        selection.cluster_id = ids[0] if ids else 0

    def _set(self, index: int, transform: Transform, **changes) -> None:
        self._emit(
            BEND_SET,
            SetTransform(
                index,
                Transform(
                    changes.get("op", transform.op),
                    changes.get("layer", transform.layer),
                    changes.get("params", transform.params),
                    changes.get("indices", transform.indices),
                ),
            ),
        )

    def _noise_rows(self, state: ControlState, layer: LayerInfo | None) -> None:
        """One layer's noise strength and its x/y ratio.

        Both are greyed on a tRGB layer, which carries no noise of its own: the
        render side only writes these onto a module that declares them, so a
        live control there would do nothing and say nothing. Greyed rather than
        hidden, so picking a tRGB layer does not change the panel's height.
        """
        imgui.separator_text("Layer noise")
        if layer is None:
            self._note(_NO_MODEL)
            return
        live = not is_torgb(layer.name)
        if not live:
            imgui.begin_disabled()
        strength = layer_noise_value(state.layer_noise, layer.name)
        fit_item("Strength")
        changed, value = imgui.slider_float("Strength", strength, 0.0, 2.0)
        if changed:
            self._emit(BEND_NOISE, SetLayerNoise(layer.name, float(value)))
        rx, ry = layer_ratio_value(state.layer_ratios, layer.name)
        fit_item("Ratio x")
        changed_x, new_x = imgui.slider_float("Ratio x", rx, 0.0, 4.0)
        fit_item("Ratio y")
        changed_y, new_y = imgui.slider_float("Ratio y", ry, 0.0, 4.0)
        if changed_x or changed_y:
            self._emit(
                BEND_RATIO,
                SetLayerRatio(layer.name, float(new_x), float(new_y)),
            )
        if not live:
            imgui.end_disabled()
            self._note(_TORGB_NOTE)

    def _adjuster_rows(self, state: ControlState, info) -> None:
        """Eight weights, and the directions they scale.

        The weights are ordinary bound rows, so each carries its driver marker
        and can be played from a controller. A slot past the loaded directions
        is greyed: the weights beyond `len(directions)` are zeroed once when
        directions load and nothing re-zeroes them after, so an ungreyed slot
        there is a control that writes a value no direction is left to use.

        Loading a single direction goes through a slot picker under the sliders
        rather than a Load button on each one. Two reasons, and the second is
        the deciding one: a button beside a slider takes width the slider needs
        in a narrow dock, and the driver marker column only reads as a column
        while every row in it starts at the same place.
        """
        imgui.separator_text("Adjuster")
        loaded = len(state.directions)
        if not loaded:
            self._note(_NO_DIRECTIONS)
        for slot in range(_ADJUST_SLOTS):
            self._binder.slider_float(
                f"adjust_w{slot + 1}", f"{slot + 1}", enabled=slot < loaded
            )
        self._direction_load_row(loaded)
        self._direction_action_row(state, info)
        if self._direction_error:
            self._error(self._direction_error)

    def _direction_load_row(self, loaded: int) -> None:
        """Pick a slot, load one vector into it.

        The slots offered are the ones that exist plus the next free one, so the
        list fills one file at a time and never has a hole in it.
        """
        slots = tuple(range(min(loaded + 1, _ADJUST_SLOTS)))
        labels = [str(slot + 1) for slot in slots]
        fit_item("Slot", ems=_NUMBER_EMS, reserve=trailing_width(_LOAD))
        changed, chosen = imgui.combo(
            "Slot", min(self._direction_slot, len(slots) - 1), labels
        )
        if changed and 0 <= chosen < len(slots):
            self._direction_slot = chosen
        imgui.same_line()
        if imgui.button(_LOAD):
            self._direction_dialog = (
                min(self._direction_slot, len(slots) - 1),
                pfd.open_file("Load a direction", "", _DIRECTION_FILTER),
            )

    def _direction_action_row(self, state: ControlState, info) -> None:
        if imgui.button(_LOAD_ALL):
            self._direction_dialog = (
                None,
                pfd.open_file("Load directions", "", _DIRECTION_FILTER),
            )
        same_line_if_it_fits(trailing_width(_RANDOMIZE))
        # Randomize needs the model's W width to make a vector the sum against
        # W can use, so it is greyed until a model publishes one.
        width = info.z_dim if info else 0
        if not width:
            imgui.begin_disabled()
        if imgui.button(_RANDOMIZE):
            self._emit(
                ADJUST_DIRECTIONS,
                SetDirections(
                    randomized_directions(
                        state.directions, width, random.randint(0, _SEED_CEILING)
                    )
                ),
            )
        if not width:
            imgui.end_disabled()

    def _take_direction_file(self, state: ControlState) -> None:
        if self._direction_dialog is None:
            return
        slot, dialog = self._direction_dialog
        if not dialog.ready():
            return
        self._direction_dialog = None
        result = dialog.result()
        if not result:
            return
        self._direction_error = None
        try:
            loaded = load_direction_file(str(result[0]))
        except Exception as exc:
            logger.exception("Could not load directions from %s", result[0])
            self._direction_error = f"Could not load the directions. {describe(exc)}"
            return
        merged = merged_directions(state.directions, loaded, slot)
        if merged is None:
            self._direction_error = (
                "That file does not fit this slot. "
                "A slot takes one vector the same length as the others."
            )
            return
        self._emit(ADJUST_DIRECTIONS, SetDirections(merged))

    def _image_rows(self, info) -> None:
        """How the captured activation becomes the picture.

        The capture layer is a combo over the same catalog the layer list
        draws, with the final output first. It is the one control here that
        changes the frame's size, which is why it sits with the rest of the
        image derivation rather than in the layer list it reads from.
        """
        imgui.separator_text("Image")
        self._binder.checkbox("grayscale", "Grayscale")
        self._binder.slider_float("img_scale_db", "Contrast dB")
        self._binder.checkbox("img_normalize", "Normalize")
        self._binder.drag_int("base_channel", "Base channel")
        state = self._binder.state()
        values, labels = capture_choices(info.layers if info else ())
        picked = string_combo("Capture", state.capture_layer, values, labels)
        if picked is not None:
            self._emit("/image/layer", picked)
        if state.capture_layer and combo_index(state.capture_layer, values) < 0:
            self._note(f"{state.capture_layer} is not a layer in this model.")

    def _note(self, text: str) -> None:
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(text)
        imgui.pop_style_color()

    def _error(self, text: str) -> None:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ERROR_COLOR))
        imgui.text_wrapped(text)
        imgui.pop_style_color()
