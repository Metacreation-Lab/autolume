"""Mixing panel: one generator assembled from two models.

The selection is one entry per synthesis parameter, `"A"`, `"B"` or `"X"`,
aligned to `mixing.conv_names` of the loaded pair and padded to the longer of
the two. That is per parameter, not per block, so a 1024 model has hundreds of
them and no performer is going to click through the list. The rows here are
per resolution instead: picking A on a resolution sets every parameter at that
resolution to A, which is how the old widget's collapsed view behaved and is
the level the picture actually reads at.

`"X"` removes layers, and the only shape the merge can be built around is a
trailing run of them: the mixed model's output resolution is read off the last
layer still kept, so a gap in the middle is a network with no way to get from
one side of it to the other. The X button therefore always cuts from a row to
the end, and Recover puts one resolution back at a time. A mid-selection `"X"`
inside a kept block is still expressible in state, so a hand edited preset can
produce one, and the merge rejects it with its own sentence rather than
building something wrong.

Nothing here blocks. Loading model B, building the mix and saving the merged
file are all loader thread jobs, and all three report back through stores this
panel polls.
"""

import logging
from typing import Callable, Sequence

from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.core.mixing import (
    ORIGIN_A,
    ORIGIN_B,
    ORIGIN_X,
    conv_names,
    layer_resolution,
)
from autolume.live.core.params import MIX_LAYERS, SetCombinedLayers
from autolume.live.ui.controls import ControlBinder
from autolume.live.ui.panels.perform import (
    draw_error,
    fit_item,
    draw_note,
    trailing_width,
)

logger = logging.getLogger(__name__)

ORIGIN_MIXED = "Mixed"

_BROWSE = "Browse"
_NO_MODEL2 = "No second model loaded"
_CUT = "X"
_RECOVER = "Recover"
_SAVE = "Save"
_NO_PAIR = "Load a model in both slots to mix them."
_LOADING_B = "Loading the second model."
_NAME_EMS = 12.0
_SAVE_HINT = "merged"


class MixRow:
    """One resolution's row: which parameter indices it covers, from where.

    `a_indices` and `b_indices` are the subsets model A and model B actually
    have a layer for. They differ when the two models have different depths,
    and an index only one of them has is why a checkbox is greyed rather than
    simply absent.
    """

    def __init__(
        self,
        resolution: int,
        indices: tuple[int, ...],
        a_indices: tuple[int, ...],
        b_indices: tuple[int, ...],
    ) -> None:
        self.resolution = resolution
        self.indices = indices
        self.a_indices = a_indices
        self.b_indices = b_indices

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MixRow):
            return NotImplemented
        return (
            self.resolution,
            self.indices,
            self.a_indices,
            self.b_indices,
        ) == (other.resolution, other.indices, other.a_indices, other.b_indices)

    def __repr__(self) -> str:
        return f"MixRow({self.resolution}, {self.indices})"


def _resolution_of(names: Sequence[str], index: int) -> int | None:
    if index >= len(names):
        return None
    try:
        return layer_resolution(names[index])
    except ValueError:
        return None


def resolution_rows(
    names_a: Sequence[str], names_b: Sequence[str]
) -> tuple[MixRow, ...]:
    """The rows the panel draws, one per resolution, in network order.

    An index belongs to a row if either model's layer at that index sits at that
    resolution, which is how the two are lined up when their depths differ. A
    row is created the first time a resolution is seen walking the pair
    together, so the order is the network's rather than numeric, and a model
    whose blocks are not a doubling ladder still gets one row per block.
    """
    length = max(len(names_a), len(names_b))
    order: list[int] = []
    covered: dict[int, list[int]] = {}
    from_a: dict[int, list[int]] = {}
    from_b: dict[int, list[int]] = {}
    for index in range(length):
        res_a = _resolution_of(names_a, index)
        res_b = _resolution_of(names_b, index)
        for resolution, owner in ((res_a, from_a), (res_b, from_b)):
            if resolution is None:
                continue
            if resolution not in covered:
                order.append(resolution)
                covered[resolution] = []
                from_a.setdefault(resolution, [])
                from_b.setdefault(resolution, [])
            if index not in covered[resolution]:
                covered[resolution].append(index)
            owner[resolution].append(index)
    return tuple(
        MixRow(
            resolution,
            tuple(covered[resolution]),
            tuple(from_a.get(resolution, ())),
            tuple(from_b.get(resolution, ())),
        )
        for resolution in order
    )


def default_selection(
    names_a: Sequence[str], names_b: Sequence[str]
) -> tuple[str, ...]:
    """Model A throughout, with model B's extra depth on the end.

    The selection the legacy widget built when a pair was first seen, and still
    the right default: all-A is bit identical to model A, so a fresh mix looks
    exactly like what was already on screen and every click from there is a
    visible change.
    """
    entries = [ORIGIN_A] * len(names_a)
    entries += [ORIGIN_B] * max(0, len(names_b) - len(names_a))
    return tuple(entries)


def fits_pair(
    entries: Sequence[str], names_a: Sequence[str], names_b: Sequence[str]
) -> bool:
    """Whether a held selection still lines up with the pair now loaded.

    A selection of the wrong length is the "the model changed underneath it"
    case. The host rejects it with its own sentence, but the panel has to notice
    first so it can offer the default instead of drawing rows against a
    selection that cannot be applied.
    """
    return len(entries) == max(len(names_a), len(names_b))


def row_origin(entries: Sequence[str], row: MixRow) -> str:
    """What a resolution's row displays: A, B, X or Mixed.

    Mixed is the honest answer for a row whose parameters do not agree, which
    the per resolution view can produce from a preset or from a selection made
    while a different pair was loaded. It is not a value: no click ever writes
    it, and picking A or B on such a row makes the whole row agree.
    """
    values = {entries[index] for index in row.indices if index < len(entries)}
    if not values:
        return ORIGIN_MIXED
    if values == {ORIGIN_A}:
        return ORIGIN_A
    if values == {ORIGIN_B}:
        return ORIGIN_B
    if values == {ORIGIN_X}:
        return ORIGIN_X
    return ORIGIN_MIXED


def cascade(entries: Sequence[str], row: MixRow, origin: str) -> tuple[str, ...]:
    """Set every parameter at `row`'s resolution to `origin`.

    The cascade the collapsed legacy row performed, made whole: it walked
    forward from the row's first index and set every later parameter at the same
    resolution, which for a row is every parameter in it. Only the indices the
    chosen model actually has a layer for are written, because a parameter set
    to a model that does not have it is rejected by the merge.
    """
    reachable = row.a_indices if origin == ORIGIN_A else row.b_indices
    updated = list(entries)
    for index in reachable:
        if index < len(updated):
            updated[index] = origin
    return tuple(updated)


def cut(
    entries: Sequence[str], cached: Sequence[str], row: MixRow
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Remove `row` and everything after it. Returns `(entries, cached)`.

    Always a trailing run, never a single row: the output resolution is read
    off the last layer still kept, so removing a block while keeping a larger
    one leaves the mixed model a gap it cannot be built around. The legacy X
    button enforced the same thing the same way.

    What is kept before the cut is copied into `cached`, which is what Recover
    reads back. Nothing after the cut is copied, so `cached` never holds an `X`
    of its own and a Recover can always restore something applicable.

    A `cached` shorter than `entries` is filled out from `entries` rather than
    left short, and that is the case that matters rather than a defensive
    nicety: the panel's cache starts empty and is only ever written by a cut, so
    the very first cut of a session had nothing to hold on to and Recover put
    nothing back. Filling from `entries` is exactly right, because a layer that
    has not been cut yet is holding its own current value.
    """
    start = row.indices[0] if row.indices else len(entries)
    kept = tuple(entries[:start])
    held = tuple(cached) + tuple(entries[len(cached) :])
    return kept + (ORIGIN_X,) * (len(entries) - start), kept + held[start:]


def recover(
    entries: Sequence[str],
    cached: Sequence[str],
    rows: Sequence[MixRow],
    row_index: int,
) -> tuple[str, ...]:
    """Put one resolution back, from what was held when it was cut.

    One resolution at a time rather than all of them, which is what the legacy
    Recover did: it restored up to the first index of the next resolution and
    left everything deeper removed. That makes truncating and then walking back
    up a sequence of single clicks instead of an all or nothing switch.
    """
    if not (0 <= row_index < len(rows)):
        return tuple(entries)
    following = rows[row_index + 1 :]
    end = len(entries)
    for row in following:
        if row.indices:
            end = row.indices[0]
            break
    restored = list(entries)
    for index in range(min(end, len(restored), len(cached))):
        restored[index] = cached[index]
    return tuple(restored)


def save_note(status) -> str | None:
    """What to say beside Save. A failure, or where the file went."""
    if status.error:
        return status.error
    if status.path:
        return f"Saved to {status.path}."
    return None


def model_label(path: str | None) -> str:
    """The bare file name of a model, for a column heading."""
    if not path:
        return _NO_MODEL2
    return str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


class MixingPanel:
    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        self._open_dialog: pfd.open_file | None = None
        self._name = ""
        # Model A's parameter names, cached against the generator they were read
        # from. Walking `named_parameters()` is cheap but not free, and it is the
        # same answer every frame for the same network.
        self._names_a: tuple[str, ...] = ()
        self._names_a_source = None
        self._names_b: tuple[str, ...] = ()
        self._names_b_source = None
        # What the selection held before the most recent cut, which is what
        # Recover reads back.
        self._cached: tuple[str, ...] = ()
        # The selection most recently sent. Both `request_mix` and the event are
        # edge triggered by contract, and `request_mix` rebuilds a whole
        # generator on the loader thread, so a panel that sent one every frame
        # while the store caught up would queue a build per frame.
        self._requested: tuple[str, ...] | None = None

    def gui(self) -> None:
        state = self._binder.state()
        names_a, names_b = self._names()
        self._model_row()
        self._enable_row(names_a, names_b)
        self._layer_rows(state, names_a, names_b)
        self._save_row(names_a, names_b)
        self._status_row()
        self._take_dialog_result()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _names(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The parameter names of both slots, cached per network.

        Slot A comes from `current_a()`, never `current()`. `current()` answers
        "what is on screen", which is the mix whenever one is built, and these
        names are what the selection *applies to* rather than what it produced.

        An earlier version read `current()` and skipped the read while
        `mixing_enabled()`, arguing that a model A swap retires the mix and
        reopens the gate. It does retire the mix, but it leaves `mixing_enabled`
        set, so the gate stayed shut and the panel went on deriving rows,
        `fits_pair` and cascades from a model that was no longer loaded, until
        somebody happened to toggle Enable off and on.

        Cached against the generator object itself rather than the path, since
        the path is unchanged by a device switch that rebuilds the model.
        """
        host = self._runtime.model_host
        self._names_a, self._names_a_source = self._slot_names(
            host.current_a(), self._names_a, self._names_a_source
        )
        self._names_b, self._names_b_source = self._slot_names(
            host.current_b(), self._names_b, self._names_b_source
        )
        return self._names_a, self._names_b

    def _slot_names(self, model, held: tuple[str, ...], source):
        """One slot's names, re-read only when its generator has changed."""
        network = getattr(model, "G", None)
        if network is None:
            return (), None
        if network is source:
            return held, source
        return self._conv_names(network), network

    def _conv_names(self, network) -> tuple[str, ...]:
        """`conv_names`, wrapped. A generator that cannot be walked is not worth
        taking the window down over, and every section below reads an empty name
        list as "not ready" already.
        """
        try:
            return conv_names(network)
        except Exception:
            logger.exception("Could not read the layer names of a mixing model")
            return ()

    def _model_row(self) -> None:
        """The second model, as the same row the first one has in Perform.

        A path field that can be typed or pasted into, and a Browse beside it,
        both drawn through the binder so the row carries its driver marker and
        greys itself when a source is writing `pkl2`.
        """
        imgui.separator_text("Second model")
        reserve = trailing_width(_BROWSE)
        live = self._binder.input_text(
            "pkl2", "##model2", hint=_NO_MODEL2, reserve=reserve
        )
        imgui.same_line()
        if not live:
            imgui.begin_disabled()
        if imgui.button(_BROWSE):
            self._open_dialog = pfd.open_file(
                "Choose the second model", "", ["Model files", "*.pkl"]
            )
        if not live:
            imgui.end_disabled()
        if self._runtime.model_host.pending_b():
            draw_note(_LOADING_B)

    def _take_dialog_result(self) -> None:
        if self._open_dialog is None or not self._open_dialog.ready():
            return
        result = self._open_dialog.result()
        self._open_dialog = None
        if result:
            self._emit("/mix/model", str(result[0]))

    def _enable_row(self, names_a, names_b) -> None:
        """Enable mixing, as an ordinary bound checkbox and nothing else.

        The host is driven from `_ModelWatchingControlLoop._watch_mixing`, not
        from here. A panel is a dockable window and its gui function does not
        run while its tab is hidden, so a panel that drove the host would leave
        `/mix/enabled` over OSC, and a preset restored while another tab was
        showing, waiting on somebody to click this tab.
        """
        ready = bool(names_a) and bool(names_b)
        self._binder.checkbox("mixing_enabled", "Enable mixing", enabled=ready)

    def _layer_rows(self, state, names_a, names_b) -> None:
        """One row per resolution: A, B, the cut, and the heading above them.

        A selection that does not fit the pair is replaced with the default
        rather than drawn against: the merge would reject it, and a row drawn
        from an index the selection does not have says nothing true.
        """
        imgui.separator_text("Layers")
        if not names_a or not names_b:
            draw_note(_NO_PAIR)
            return
        entries = tuple(state.combined_layers)
        if not fits_pair(entries, names_a, names_b):
            entries = default_selection(names_a, names_b)
            self._apply(entries)
        # A cache of the wrong length belongs to a pair that is no longer
        # loaded, and restoring from it would put another model's origins back.
        # A fitting selection reaches this without ever going through the branch
        # above, which is exactly the preset-restore case, so the seeding cannot
        # live in there.
        if len(self._cached) != len(entries):
            self._cached = entries
        rows = resolution_rows(names_a, names_b)
        draw_note(f"A is {model_label(state.pkl_path)}")
        draw_note(f"B is {model_label(state.pkl2)}")
        for row_index, row in enumerate(rows):
            self._layer_row(entries, rows, row_index, row)

    def _layer_row(self, entries, rows, row_index: int, row: MixRow) -> None:
        imgui.push_id(row.resolution)
        origin = row_origin(entries, row)
        cut_here = origin == ORIGIN_X
        imgui.text(f"{row.resolution}")
        imgui.same_line()
        self._origin_checkbox(entries, row, ORIGIN_A, origin, cut_here)
        self._origin_checkbox(entries, row, ORIGIN_B, origin, cut_here)
        self._cut_button(entries, rows, row_index, row, cut_here)
        imgui.pop_id()

    def _origin_checkbox(
        self, entries, row: MixRow, origin: str, shown: str, cut_here: bool
    ) -> None:
        """One of the row's two boxes. Ticked when the whole row is that model.

        Greyed when the model has no layer at this resolution, which is what a
        pair of different depths produces, and greyed while the row is cut,
        because picking a model for a removed layer is not a thing to offer.
        A Mixed row shows neither box ticked, so one click makes it agree.
        """
        reachable = row.a_indices if origin == ORIGIN_A else row.b_indices
        live = bool(reachable) and not cut_here
        if not live:
            imgui.begin_disabled()
        clicked, _ = imgui.checkbox(origin, shown == origin)
        if not live:
            imgui.end_disabled()
        if clicked and live:
            self._apply(cascade(entries, row, origin))
        imgui.same_line()

    def _cut_button(
        self, entries, rows, row_index: int, row: MixRow, cut_here: bool
    ) -> None:
        """X to remove this resolution and everything deeper, Recover to put it
        back.

        One button, two labels, and they are not the same width: the row does
        change width when a resolution is cut. That is accepted rather than
        overlooked. It is the last item on the row, so nothing shifts under a
        hand reaching for something else, and the alternative is padding X out
        to Recover's width, which makes a one-character button look like a
        misdrawn one.
        """
        if cut_here:
            if imgui.button(_RECOVER):
                self._apply(recover(entries, self._cached, rows, row_index))
            return
        # The last row cannot be cut: something has to be left to read the
        # mixed model's output resolution off.
        last = row_index == len(rows) - 1
        if last:
            imgui.begin_disabled()
        if imgui.button(_CUT):
            updated, cached = cut(entries, self._cached, row)
            self._cached = cached
            self._apply(updated)
        if last:
            imgui.end_disabled()

    def _apply(self, entries: tuple[str, ...]) -> None:
        """Store the selection. The build follows from the state change.

        One write, to state, and `_watch_mixing` on the control thread turns it
        into `request_mix`. The panel used to call the host itself alongside the
        event, which made the host's copy of the selection reachable only while
        this tab was showing.

        Edge triggered against what was last sent rather than against the store,
        which only catches up on the next control tick: the frames in between
        would each submit the same selection again, and each one costs a whole
        generator build on the loader thread.
        """
        if entries == self._requested:
            return
        self._requested = entries
        self._emit(MIX_LAYERS, SetCombinedLayers(entries))

    def _save_row(self, names_a, names_b) -> None:
        """A name and Save. The write happens on the loader thread.

        Save needs a name and both slots. The host reports all three if it gets
        them anyway, but a greyed button says it before the click rather than
        after.
        """
        imgui.separator_text("Merged model")
        fit_item("##name", ems=_NAME_EMS, reserve=trailing_width(_SAVE))
        _, self._name = imgui.input_text_with_hint("##name", _SAVE_HINT, self._name)
        imgui.same_line()
        ready = bool(self._name.strip()) and bool(names_a) and bool(names_b)
        if not ready:
            imgui.begin_disabled()
        if imgui.button(_SAVE):
            self._runtime.model_host.request_save_mix(self._name.strip())
        if not ready:
            imgui.end_disabled()
        status = self._runtime.model_host.mix_save_store.snapshot()
        note = save_note(status)
        if status.error:
            draw_error(note)
        else:
            draw_note(note)

    def _status_row(self) -> None:
        """Why a mix is not showing, when there is a reason.

        The host's error channel, which the preview overlay also reads: a failed
        mix leaves model A rendering, so the picture looks fine and this line is
        the only place the reason exists next to the controls that caused it.
        """
        error = self._runtime.model_host.error()
        if error:
            draw_error(error)
