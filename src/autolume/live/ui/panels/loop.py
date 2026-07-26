"""Loop panel: keyframe and noise loop playback, and the outbound sync pulse.

Everything here is either a bindable registry parameter, drawn through
`ControlBinder` like any other performable row, or a keyframe edit, which is
structured state with no registry address of its own (design.md). Keyframe
rows are plain widgets for that reason: there is no binding to mark and no
mapping row to open, the same argument that leaves the vector row in
`perform.py` without a driver marker.

The noise loop's radius and seed rebuild a background table that can take a
few seconds at the far end of the radius range (`core/noiseloop.py`; it used
to be tens of seconds, until the range's top was lowered for exactly this).
Motion keeps running on the previous table while that happens, which is
correct and would otherwise look broken: nothing on screen says a rebuild is
under way. `_noise_pending_row` is the whole of what says so, read off the
same published snapshot the control loop itself reads rather than a new
channel of its own, and `_pending_note` is what puts a rough ETA on it.

Keyframes and the noise loop are mutually exclusive, not two independent
features: `derive_mode` plays one or the other, never both, off the same
`noise_loop` flag this panel now exposes as a mode radio in Control rather
than a checkbox buried in the Noise loop section. Only the section matching
the selected mode is drawn; see `gui`'s own comment for why that is the one
place in this panel allowed to hide rather than grey.
"""

import dataclasses
import logging
import random
from typing import Callable

from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import ModelInfo
from autolume.live.core.noiseloop import estimated_build_seconds
from autolume.live.core.params import (
    KEYFRAME_REMOVE,
    KEYFRAME_SET,
    ControlState,
    Keyframe,
    RemoveKeyframe,
    SetKeyframe,
    default_keyframe,
)
from autolume.live.errors import describe
from autolume.live.ui.controls import ControlBinder
from autolume.live.ui.panels.perform import button_width, load_vector_file
from autolume.live.ui.theme import ERROR_COLOR

logger = logging.getLogger(__name__)

# Wide enough to read a typical seed value, e.g. "123456.78", comfortably:
# the field was 1.4 down to fit a two row entry into a 280px panel (task 9
# review), which is the mistake item 8 undoes. See `_KEYFRAME_ROW_COMBOS` in
# the test suite for the width this and the rest of the row were measured
# against.
_SEED_WIDTH_EMS = 5.5
_VECTOR_FILTER = ["Vector files", "*.npy *.pt"]
# Trimmed from an earlier version that also said "Project in Perform also
# applies to this loop": that half only holds while `loop_active`, the one
# time a noise loop actually puts the generator in `"vec"` mode and visibly
# ungreys the checkbox in Perform. Selecting Noise loop mode without playing
# it does not, so this note is only drawn then too (`_noise_rows`); showing
# it, and this claim, while Project sits greyed in Perform would have the
# UI assert a coupling it is simultaneously denying. What is left, once
# it does show, is the one thing the greying alone cannot say: what the
# checkbox actually changes about a noise loop's frames.
_PROJECT_NOTE = "Off, each step is read as a raw W row instead of a latent."
_IP_NOTE = "This must be an IP address. A host name will not work."


def desired_noise_key(
    state: ControlState, model_info: ModelInfo | None
) -> tuple[int, float, int] | None:
    """The `(seed, radius, z_dim)` the noise table should be built from now.

    None without a loaded model: there is no `z_dim` to build against, and no
    table can be pending in that case either.
    """
    if model_info is None:
        return None
    return (state.noise_loop_seed, state.noise_radius, model_info.z_dim)


def noise_table_pending(
    desired: tuple[int, float, int] | None, built: tuple[int, float, int] | None
) -> bool:
    """Whether the published table is stale against what the state now asks for."""
    return desired is not None and desired != built


def _paired_control_width(label: str) -> float:
    """Width imgui gives a checkbox or a radio button: its box, then the label.

    Both draw the same shape internally: a box or circle sized to
    `get_frame_height()`, followed, when the label is non-empty, by the
    style's inner item spacing and the label's own text width. Read off the
    live style and font rather than assumed, so a theme or font change is
    already accounted for with no refit.
    """
    box = imgui.get_frame_height()
    return box + imgui.get_style().item_inner_spacing.x + imgui.calc_text_size(label).x


def keyframe_row_required_width(index: int) -> float:
    """The one-line layout's real width need, measured against the row it is
    about to draw rather than predicted from a fitted constant.

    A fitted slope and intercept shipped here twice and was wrong both
    times: fitted against font sizes (13, 20, 26pt) the UI font size
    preference this app carries does not actually have (the whole live
    runtime renders at one fixed size, `theme.FONT_SIZE`), it drifted from
    what the row truly needs the moment a control changed, which is exactly
    what a fitted number cannot see coming. This instead mirrors
    `_keyframe_row_one_line`'s own draw order item for item: every widget it
    places on the line, so a widget added there is a width added here, with
    no interpolation between measured points and nothing to refit later.

    Two of the ten items are not derived from a label at all, because they
    are not sized from one: the seed fields carry an explicit width of their
    own (`_SEED_WIDTH_EMS`, `_keyframe_seed_fields`'s `imgui.set_next_item_width`),
    so that number is used directly rather than re-derived. `index`'s own
    text width varies with how many digits it prints, unlike every other
    item on the row, so it is measured off the real value being drawn
    instead of assumed fixed.
    """
    seed_field = imgui.get_font_size() * _SEED_WIDTH_EMS
    items = (
        imgui.calc_text_size(str(index)).x,
        _paired_control_width("Seed"),
        _paired_control_width("Vector"),
        _paired_control_width("Project"),
        seed_field,
        seed_field,
        button_width("Load"),
        button_width("Randomize"),
        button_width("Snap"),
        button_width("Remove"),
    )
    return sum(items) + imgui.get_style().item_spacing.x * (len(items) - 1)


def keyframe_row_fits_one_line(index: int, available: float) -> bool:
    """Whether `available` is wide enough for the keyframe row's one-line layout.

    A deliberate, width-driven reflow: this is the one thing allowed to
    change the row's height, unlike the kind switch, which must never
    (`_keyframe_row`'s own docstring). It is checked once per row, before
    anything is drawn, so the whole row commits to one layout rather than
    part of it drawing wide and part narrow. `keyframe_row_required_width`
    computes the same real widths imgui is about to draw the row at, so
    there is no separate number here to drift from what actually gets
    drawn (verified equal, within a rounding pixel, at several font sizes
    including the app's own: `test_keyframe_row_required_width_matches_
    what_is_actually_drawn`).
    """
    return available >= keyframe_row_required_width(index)


def _pending_note(radius: float) -> str:
    """The rebuilding note, with an ETA for `radius`, the one now being built.

    The estimate is `estimated_build_seconds`'s straight line between two
    measured points (`noiseloop.py`), stated as approximate rather than as a
    countdown: it is a rough guide for whether to wait it out or lower the
    radius, not a promise this machine will match the one it was measured on.
    """
    seconds = round(estimated_build_seconds(radius))
    unit = "second" if seconds == 1 else "seconds"
    return (
        "Noise table is rebuilding. "
        "Motion stays on the previous table until it is ready. "
        f"Estimated build time is about {seconds} {unit}. "
        "This is a rough estimate and can be off on a different machine."
    )


def captured_keyframe(keyframe: Keyframe, state: ControlState) -> Keyframe:
    """`keyframe` replaced with an exact copy of the current navigation state.

    The kind follows `state.vector_mode`, the navigation mode itself, not
    whatever kind the keyframe already happened to be: Snap is "capture what
    I am looking at right now", and what is on screen is decided by how the
    performer is navigating, not by the row's previous content. Snapping
    onto a seed keyframe while navigating by vector used to leave it a seed
    keyframe filled from `latent_x`/`latent_y`, which are stale and do not
    correspond to the frame on screen at all in vector mode; keeping the
    previous kind meant Snap could capture something other than what was
    actually being looked at.

    `vector_mode`, not `derive_mode`: the latter also folds in `loop_active`,
    answering "what is rendering this frame" rather than "how is the
    performer navigating", and a running loop drives the frame regardless of
    which mode Snap would otherwise read. Whether Snap itself should greyed
    out while a loop plays is a separate, open question, not decided here.

    The field the new kind does not use is cleared, not left stale: a vector
    snap zeroes `seed_x`/`seed_y`, a seed snap empties `vec`. A leftover
    value there is invisible in the UI right up until someone flips the
    row's kind back, at which point it reappears as a position nobody chose.
    `project` (vector mode only) is copied from `latent_project` too, since
    it is part of what produced the frame being captured, deciding whether
    `vec` is read as a `z` through the mapping network or already a `w`; in
    seed mode it is left alone, since `_keyframe_to_w` never reads it for a
    seed keyframe.
    """
    if state.vector_mode:
        return dataclasses.replace(
            keyframe,
            kind="vec",
            vec=state.latent_vec,
            project=state.latent_project,
            seed_x=0.0,
            seed_y=0.0,
        )
    return dataclasses.replace(
        keyframe,
        kind="seed",
        seed_x=state.latent_x,
        seed_y=state.latent_y,
        vec=(),
    )


class LoopPanel:
    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        # Which row a Load dialog was opened for, so the frame it resolves on
        # can write back to that keyframe rather than whichever one happens
        # to be under the cursor by then.
        self._vector_dialog: tuple[int, pfd.open_file] | None = None
        self._vector_error: str | None = None

    def gui(self) -> None:
        state = self._binder.state()
        self._control_rows()
        # The displayed value, not `state.noise_loop`: the radio drawn just
        # above holds its own local override for the frame it is clicked on,
        # the same reason `perform.py` reads `vector_mode` the same way, and
        # reading the raw snapshot here would show the old section for one
        # frame after the click.
        noise_mode = bool(self._binder.value("noise_loop"))
        self._scrub_rows(state, noise_mode)
        # Drained here, unconditionally, rather than from inside
        # `_keyframe_rows`: `portable_file_dialogs` windows are not
        # app-modal, so a performer can switch to Noise loop mode, which
        # hides the Keyframes section entirely (`_keyframe_rows` is not even
        # called below), while a Load dialog opened before the switch is
        # still open. Draining only from `_keyframe_rows` would leave the
        # result stuck in `self._vector_dialog` for as long as Noise loop
        # stays selected, then apply it the instant Keyframes is reselected,
        # to whatever keyframe now sits at the stored index, which may no
        # longer be the one Load was opened for. Applying it here, the
        # first frame it is ready on regardless of which section is showing,
        # closes that window: the result still lands on the exact keyframe
        # Load was opened for (`_take_vector_load`'s own index re-check),
        # the same as it always has, just without waiting on the section
        # that keyframe happens to live in being visible.
        self._take_vector_load(state)
        # Hiding a whole section, not greying it: this is the one exception
        # to the stable-footprint rule in this panel, and it is a deliberate
        # one, not an oversight. That rule exists so a control does not shift
        # out from under a hand reaching for it mid-drag; a section swap here
        # follows the mode radio the performer just clicked, settles inside
        # the same frame, and only ever removes the section the design
        # (`derive_mode`) already made inert, since Keyframes and Noise loop
        # never both play at once. Every per-control case in this panel
        # (Project on a keyframe row, the seed fields, the vector controls,
        # the index scrub below) still greys and never hides.
        if noise_mode:
            self._noise_rows(state)
        else:
            self._keyframe_rows(state)
        self._pulse_rows()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _control_rows(self) -> None:
        """Play, the time or speed switch, Stop at cycle end, and the loop mode.

        Both the seconds and the speed control are always drawn, one of them
        greyed, the same no-reflow rule the latent section follows: a mode
        switch never changes which rows are on screen, only whether the hand
        can reach them. Seconds is a typed field rather than a slider: a
        loop's length is a quantity a performer sets, not a position worth
        sweeping through and watching change frame by frame the way alpha is.

        The mode radio at the bottom, Keyframes or Noise loop, is `noise_loop`
        itself, the same BOOL `derive_mode` already reads to choose which kind
        of loop plays while `loop_active`: this panel used to draw it as a
        checkbox buried in the Noise loop section, which read as an
        independent feature rather than the other half of a choice this
        section makes for the whole panel (`gui`'s own comment on what that
        choice then hides).
        """
        imgui.separator_text("Control")
        self._binder.checkbox("loop_active", "Play")
        self._binder.bool_radio("loop_uses_time", "Speed", "Time")
        uses_time = bool(self._binder.value("loop_uses_time"))
        self._binder.input_float(
            "loop_time", "Seconds", enabled=uses_time, format="%.2f"
        )
        self._binder.slider_float("loop_speed", "Speed", enabled=not uses_time)
        # Label only: the registry name and the OSC address (/loop/perfect)
        # are unchanged, and so is a preset's key for it.
        self._binder.checkbox("perfect_loop", "Stop at cycle end")
        self._binder.bool_radio("noise_loop", "Keyframes", "Noise loop")

    def _keyframe_rows(self, state: ControlState) -> None:
        """Every entry, then Add keyframe.

        No count field and no header Remove (item 10-11 of the manual
        review, superseding what item 2 first asked for: a typed field for
        the count was built, then dropped again at the maintainer's
        request). The row indices already say how many keyframes there
        are, so a count is redundant, and editing a list by typing its
        length is a stranger action than adding or removing one at a time.
        A header Remove was ambiguous about which keyframe it would take;
        per-row Remove (`_keyframe_actions`) is the only one now.

        There is no `keyframe_count` parameter behind any of this any more
        either (item 13): the registry no longer carries one at all, so
        `len(state.keyframes)` is the only count there ever is, and Add
        appends through `KEYFRAME_SET` (`_set_keyframe`) exactly like every
        other keyframe edit, the same address `_keyframe_row` itself uses
        for Snap, a kind change, or a seed edit. There is no separate
        resize address left to disagree with the tuple, which is what made
        the count and the tuple able to drift apart in the first place (a
        preset could set one to 3 while the tuple stayed at 6, a bug the
        final review of the previous pass reproduced): with one write path
        into the list, that class of bug cannot happen again.

        Add sits at the bottom, after the rows, the usual place for the one
        control that appends to a list rather than at its head where a
        header used to be.
        """
        imgui.separator_text("Keyframes")
        count = len(state.keyframes)
        for index, keyframe in enumerate(state.keyframes):
            self._keyframe_row(index, keyframe, state, count)
        if self._vector_error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ERROR_COLOR))
            imgui.text_wrapped(self._vector_error)
            imgui.pop_style_color()
        if imgui.button("Add keyframe"):
            self._set_keyframe(count, default_keyframe(count))

    def _keyframe_row(
        self, index: int, keyframe: Keyframe, state: ControlState, count: int
    ) -> None:
        """One keyframe: one line where it fits, two where it does not.

        A two line layout shipped once already, to fit a 280px panel (task 9
        review, finding 1), the same bar the perform panel's rows are held
        to. That bar fit those rows because each one carries a single
        control; a keyframe entry carries up to ten, and squeezing a
        compound row into a floor calibrated for a simple one is what made
        the seed fields unreadably narrow and forced the split that time.
        This entry sets its own floor instead (`_KEYFRAME_ROW_COMBOS`),
        measured against the row it is about to draw rather than a fitted
        guess, so it does not overstate the requirement
        (`keyframe_row_fits_one_line`, `keyframe_row_required_width`).

        Below that floor the row still has to go somewhere: rather than
        silently overflowing the panel the way it did before task 9's
        review caught it, it degrades to the same two line shape that
        shipped once already, cramped rather than broken. This is the one
        thing allowed to change the row's height: the kind switch below is
        still never allowed to, on either layout.

        The seed fields and the vector controls (Load, Randomize) are both
        always drawn, one pair greyed, the same rule the perform panel's
        latent section already follows for `vector_mode`: a kind switch
        changes which controls the hand can reach, never how many lines the
        row takes.
        """
        imgui.push_id(index)
        available = imgui.get_content_region_avail().x
        if keyframe_row_fits_one_line(index, available):
            self._keyframe_row_one_line(index, keyframe, state, count)
        else:
            self._keyframe_row_two_line(index, keyframe, state, count)
        imgui.pop_id()

    def _keyframe_row_one_line(
        self, index: int, keyframe: Keyframe, state: ControlState, count: int
    ) -> None:
        is_vector = self._keyframe_identity(index, keyframe)
        imgui.same_line()
        self._keyframe_seed_fields(index, keyframe, is_vector)
        imgui.same_line()
        self._keyframe_vector_controls(index, keyframe, is_vector)
        imgui.same_line()
        self._keyframe_actions(index, keyframe, state, count)

    def _keyframe_row_two_line(
        self, index: int, keyframe: Keyframe, state: ControlState, count: int
    ) -> None:
        """The narrow fallback: identity and the seed fields, then, indented,
        the vector controls and the destructive actions. Split there rather
        than at the old kind/content boundary, because Load and Randomize
        are what item 8 restored and are the reason the row got wide enough
        to need this fallback at all: keeping them with Snap and Remove
        puts every button on its own line and every value field on the
        other, which is what actually buys back the most width per line
        (measured).
        """
        is_vector = self._keyframe_identity(index, keyframe)
        imgui.same_line()
        self._keyframe_seed_fields(index, keyframe, is_vector)
        # No `imgui.new_line()` here: the seed fields end the line without a
        # trailing `same_line()`, which is already enough for the next item
        # to start a fresh one. An explicit `new_line()` on top of that
        # closes the current line a second time, inserting a whole blank
        # line's height between the two (confirmed by measurement: the
        # entry was rendering three line-heights tall for two lines of
        # content, the maintainer's screenshot of a reflowed row with far
        # more space below the first line than a single row's height).
        imgui.indent()
        self._keyframe_vector_controls(index, keyframe, is_vector)
        imgui.same_line()
        self._keyframe_actions(index, keyframe, state, count)
        imgui.unindent()

    def _keyframe_identity(self, index: int, keyframe: Keyframe) -> bool:
        """The row's index, its kind, and Project. Returns whether it is vector kind.

        Kind is a radio pair, Seed or Vector, full words: a checkbox read as
        "add some vector-ness" the same way `vector_mode` did in the perform
        panel (item 1), and once the row had the width for whole words
        there was no reason left to abbreviate the one control that picks
        the mode every other control on the row keys off.

        Project is greyed on a seed row. `generator.py`'s `_keyframe_to_w`
        reads `keyframe.project` only for a vector keyframe
        (`_vec_to_w`); a seed keyframe goes through `_blended_w`, which
        never looks at it. Live on a seed row it looked like the other half
        of the mode switch and did nothing, which is what the maintainer
        actually caught. Greyed rather than hidden, the same stable
        footprint rule as the seed fields and the vector controls, so the
        row's width does not change out from under the kind switch either.
        Every other per-row control was checked against the same question:
        Remove acts on whichever kind the row already is, Snap replaces the
        kind outright with the current navigation mode (`captured_keyframe`),
        and the seed fields and Load/Randomize were already greyed by kind,
        so Project was the only one left drawn live with no effect.
        """
        imgui.text(str(index))
        imgui.same_line()
        is_vector = keyframe.kind == "vec"
        changed_kind = False
        new_kind = keyframe.kind
        if imgui.radio_button("Seed", not is_vector):
            changed_kind, new_kind = True, "seed"
        imgui.same_line()
        if imgui.radio_button("Vector", is_vector):
            changed_kind, new_kind = True, "vec"
        if changed_kind:
            self._set_keyframe(index, dataclasses.replace(keyframe, kind=new_kind))
            is_vector = new_kind == "vec"
        imgui.same_line()
        if not is_vector:
            imgui.begin_disabled()
        changed_project, project = imgui.checkbox("Project", keyframe.project)
        if changed_project:
            self._set_keyframe(index, dataclasses.replace(keyframe, project=project))
        if not is_vector:
            imgui.end_disabled()
        return is_vector

    def _keyframe_seed_fields(
        self, index: int, keyframe: Keyframe, is_vector: bool
    ) -> None:
        width = imgui.get_font_size() * _SEED_WIDTH_EMS
        if is_vector:
            imgui.begin_disabled()
        imgui.set_next_item_width(width)
        changed_x, seed_x = imgui.input_float("##seed_x", keyframe.seed_x)
        imgui.same_line()
        imgui.set_next_item_width(width)
        changed_y, seed_y = imgui.input_float("##seed_y", keyframe.seed_y)
        if is_vector:
            imgui.end_disabled()
        if not is_vector and (changed_x or changed_y):
            self._set_keyframe(
                index, dataclasses.replace(keyframe, seed_x=seed_x, seed_y=seed_y)
            )

    def _keyframe_vector_controls(
        self, index: int, keyframe: Keyframe, is_vector: bool
    ) -> None:
        """Load and Randomize, the two per-keyframe vector controls the old
        app had (`widgets/looping_widget.py` `vec_viz`) that the new one
        dropped. Load opens a picker rather than the old app's separate
        path field and Browse button, matching `perform.py`'s own vector
        row rather than inventing a second pattern for the same action.

        Both are greyed for a seed keyframe, the counterpart to the seed
        fields greying for a vector one. Randomize is greyed further still
        without a loaded model: it needs `z_dim` to build a vector the
        right length, and doing something else instead, or nothing
        explicable, would be worse than a greyed button.
        """
        model_info = self._runtime.model_info_store.snapshot()
        if not is_vector:
            imgui.begin_disabled()
        if imgui.button("Load"):
            self._vector_dialog = (
                index,
                pfd.open_file("Load a keyframe vector", "", _VECTOR_FILTER),
            )
        if not is_vector:
            imgui.end_disabled()
        imgui.same_line()
        can_randomize = is_vector and model_info is not None
        if not can_randomize:
            imgui.begin_disabled()
        if imgui.button("Randomize"):
            vec = tuple(random.gauss(0.0, 1.0) for _ in range(model_info.z_dim))
            self._set_keyframe(index, dataclasses.replace(keyframe, vec=vec))
        if not can_randomize:
            imgui.end_disabled()

    def _keyframe_actions(
        self, index: int, keyframe: Keyframe, state: ControlState, count: int
    ) -> None:
        if imgui.button("Snap"):
            self._set_keyframe(index, captured_keyframe(keyframe, state))
        imgui.same_line()
        if count <= 1:
            imgui.begin_disabled()
        if imgui.button("Remove"):
            self._emit(KEYFRAME_REMOVE, RemoveKeyframe(index))
        if count <= 1:
            imgui.end_disabled()

    def _take_vector_load(self, state: ControlState) -> None:
        """Apply a keyframe Load dialog's result, once it has one.

        Drained once per `gui()` pass rather than per row, matching
        `perform.py`'s own vector dialog, and called from `gui()` itself
        rather than from `_keyframe_rows`: it is opened from inside a
        possibly-disabled row, in a section that may not even be drawn by
        the time the dialog resolves (`gui`'s own comment on why), so it has
        to run regardless of which section is showing. The dialog resolves
        on whatever later frame the performer picks a file on, by which time
        the row it was opened for may have moved or been removed, so the
        index is re-checked against the keyframes actually on hand right now
        rather than trusted blind.
        """
        if self._vector_dialog is None:
            return
        index, dialog = self._vector_dialog
        if not dialog.ready():
            return
        self._vector_dialog = None
        result = dialog.result()
        if not result:
            return
        if not (0 <= index < len(state.keyframes)):
            return
        self._vector_error = None
        try:
            values = load_vector_file(result[0])
        except Exception as exc:
            logger.exception("Could not load keyframe vector %s", result[0])
            self._vector_error = f"Could not load the vector. {describe(exc)}"
            return
        keyframe = state.keyframes[index]
        self._set_keyframe(index, dataclasses.replace(keyframe, vec=tuple(values)))

    def _set_keyframe(self, index: int, keyframe: Keyframe) -> None:
        self._emit(KEYFRAME_SET, SetKeyframe(index, keyframe))

    def _scrub_rows(self, state: ControlState, noise_mode: bool) -> None:
        """Alpha and index, both ordinary bound controls.

        Touch grace already makes scrubbing and integration coexist for
        every other motion-owned parameter (`motion.py`), and the loop's
        phase is no different, so nothing extra is needed here to make
        dragging this slider hold the loop still for the length of the drag.

        Index is ranged to the current keyframe count and shown one-based,
        through `drag_int_mapped`, while `ControlState` and OSC keep the
        zero-based, registry-bounded value: the old app did the same
        translation at its own edge (`widgets/looping_widget.py`,
        `self.params.index + 1` in and `(idx - 1) % num_keyframes` out).
        Bounding the drag to the count rather than the registry's 2**31 - 1
        is what stops it from scrubbing to a keyframe that does not exist;
        it stays a drag, not a typed field, because unlike the count this is
        a position the performer scrubs through and watches take effect.

        `noise_mode` greys Index rather than hiding it, unlike the section
        swap in `gui`: it is a control, not a section, and the same rule
        every other conditional control in this panel already follows. The
        noise loop is a single segment (`core/control.py`'s
        `_noise_latent_vector`), so `_loop_w`, the only reader of
        `RenderParams.loop_index` (`generator.py`), never runs while it plays
        (`derive_mode` returns `"vec"`, not `"loop"`), which makes Index live
        but inert exactly the way Project once was on a seed keyframe row.
        Alpha stays live in both modes: it is what the noise table is sampled
        at (`control.py`'s `_noise_latent_vector`), the loop's phase either
        way.
        """
        imgui.separator_text("Scrub")
        self._binder.slider_float("loop_alpha", "Alpha")
        count = max(1, len(state.keyframes))
        self._binder.drag_int_mapped(
            "loop_index",
            "Index",
            minimum=1,
            maximum=count,
            to_display=lambda stored: stored + 1,
            to_stored=lambda shown: shown - 1,
            enabled=not noise_mode,
        )

    def _noise_rows(self, state: ControlState) -> None:
        """Seed and Radius, and the rebuild note. Drawn only in Noise loop mode.

        The mode checkbox that used to live here moved to Control's radio
        (`_control_rows`), so both fields are unconditionally live now: this
        section only ever appears while `noise_loop` is selected, and neither
        field has a further reason to grey once it does.
        """
        imgui.separator_text("Noise loop")
        # Typed fields, not drags: a seed's neighbours are unrelated noise
        # patterns, and every distinct radius costs a table build (up to
        # ~4s at the top of its range now, see `params.py`), so nothing
        # about either is a position worth sweeping through mid-drag.
        self._binder.input_int("noise_loop_seed", "Seed")
        self._binder.input_float("noise_radius", "Radius", format="%.2f")
        # Gated on `loop_active`, not shown whenever this section is: only
        # while playing does a noise loop put the generator in `"vec"` mode
        # and ungrey Project in Perform, which is what this note describes
        # (`_PROJECT_NOTE`'s own comment). Showing it with Play off would
        # describe a checkbox the UI is simultaneously greying.
        if state.loop_active:
            self._note(_PROJECT_NOTE)
        self._noise_pending_row(state)

    def _noise_pending_row(self, state: ControlState) -> None:
        # Matches the one condition under which the control loop ever
        # requests a build (control.py `tick`): showing this outside it would
        # claim a rebuild is under way when none was ever asked for.
        if not (state.loop_active and state.noise_loop):
            return
        model_info = self._runtime.model_info_store.snapshot()
        desired = desired_noise_key(state, model_info)
        built = self._runtime.control_loop.noise_table_key
        if noise_table_pending(desired, built):
            imgui.text_disabled(_pending_note(state.noise_radius))

    def _pulse_rows(self) -> None:
        # Natural widths, not the panel's: an address, an IP and a port are
        # all fixed-format values (`ControlBinder.input_text`'s own
        # docstring), so none of them reads any better in the hundreds of
        # spare pixels a docked panel leaves lying around. 24 ems comfortably
        # holds an OSC path like "/pulse/aftertouch"; 16 an IPv4 address,
        # written out in full; 7 a port, five digits at most.
        imgui.separator_text("Pulse")
        self._binder.input_text(
            "pulse_address", "Address", hint="/pulse", natural_ems=24.0
        )
        self._binder.input_text("pulse_ip", "IP", hint="127.0.0.1", natural_ems=16.0)
        self._note(_IP_NOTE)
        # A typed field: a port has no meaningful neighbour to scrub through
        # on the way to the one a performer actually wants.
        self._binder.input_int("pulse_port", "Port", natural_ems=7.0)

    def _note(self, text: str) -> None:
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(text)
        imgui.pop_style_color()
