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
    REGISTRY,
    ControlState,
    Keyframe,
    RemoveKeyframe,
    SetKeyframe,
)
from autolume.live.errors import describe
from autolume.live.ui.controls import ControlBinder
from autolume.live.ui.panels.perform import load_vector_file
from autolume.live.ui.theme import ERROR_COLOR

logger = logging.getLogger(__name__)

# Wide enough to read a typical seed value, e.g. "123456.78", comfortably:
# the field was 1.4 down to fit a two row entry into a 280px panel (task 9
# review), which is the mistake item 8 undoes. See `_KEYFRAME_ROW_COMBOS` in
# the test suite for the width this and the rest of the row were measured
# against.
_SEED_WIDTH_EMS = 5.5
_VECTOR_FILTER = ["Vector files", "*.npy *.pt"]
_PROJECT_NOTE = (
    "Project in Perform also applies to this loop. "
    "Off, each step is read as a raw W row instead of a latent."
)
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
    """`keyframe` with its content replaced by the current snapshot's latent state.

    The kind is not touched: a seed keyframe stays a seed keyframe, filled
    from `latent_x`/`latent_y`, and a vector one stays a vector keyframe,
    filled from `latent_vec`. This is the new architecture's replacement for
    the old render-side Snap, one field object in rather than a mutation the
    render thread had to reach back for.
    """
    if keyframe.kind == "vec":
        return dataclasses.replace(keyframe, vec=state.latent_vec)
    return dataclasses.replace(keyframe, seed_x=state.latent_x, seed_y=state.latent_y)


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
        self._transport_rows()
        self._keyframe_rows(state)
        self._scrub_rows(state)
        self._noise_rows(state)
        self._pulse_rows()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _transport_rows(self) -> None:
        """Play, the time or speed switch, and Stop at cycle end.

        Both the seconds and the speed control are always drawn, one of them
        greyed, the same no-reflow rule the latent section follows: a mode
        switch never changes which rows are on screen, only whether the hand
        can reach them. Seconds is a typed field rather than a slider: a
        loop's length is a quantity a performer sets, not a position worth
        sweeping through and watching change frame by frame the way alpha is.
        """
        imgui.separator_text("Transport")
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

        `keyframe_count` itself is untouched: still a registry parameter,
        still bindable and OSC-addressable at `/loop/keyframes`, still kept
        in sync with the tuple length by `mapping.py`. Only its UI control
        is gone. Add still writes it, the same as every other resize.

        Add sits at the bottom, after the rows, the usual place for the one
        control that appends to a list rather than at its head where a
        header used to be.
        """
        imgui.separator_text("Keyframes")
        count = len(state.keyframes)
        for index, keyframe in enumerate(state.keyframes):
            self._keyframe_row(index, keyframe, state, count)
        self._take_vector_load(state)
        if self._vector_error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ERROR_COLOR))
            imgui.text_wrapped(self._vector_error)
            imgui.pop_style_color()
        if imgui.button("Add keyframe"):
            self._emit(REGISTRY["keyframe_count"].address, count + 1)

    def _keyframe_row(
        self, index: int, keyframe: Keyframe, state: ControlState, count: int
    ) -> None:
        """One keyframe, on one line: identity, content, Snap, Remove.

        A two line layout shipped for one review cycle to fit a 280px panel
        (task 9 review, finding 1), the same bar the perform panel's rows
        are held to. That bar fit those rows because each one carries a
        single control; a keyframe entry carries seven, and squeezing a
        compound row into a floor calibrated for a simple one is what made
        the seed fields unreadably narrow and forced the split in the first
        place. This entry now sets its own floor instead
        (`_KEYFRAME_ROW_COMBOS`), and one line is what that floor buys back.

        The seed fields and the vector controls (Load, Randomize) are both
        always drawn, one pair greyed, the same rule the perform panel's
        latent section already follows for `vector_mode`: a kind switch
        changes which controls the hand can reach, never how many lines the
        row takes.
        """
        imgui.push_id(index)
        is_vector = self._keyframe_identity(index, keyframe)
        imgui.same_line()
        self._keyframe_seed_fields(index, keyframe, is_vector)
        imgui.same_line()
        self._keyframe_vector_controls(index, keyframe, is_vector)
        imgui.same_line()
        self._keyframe_actions(index, keyframe, state, count)
        imgui.pop_id()

    def _keyframe_identity(self, index: int, keyframe: Keyframe) -> bool:
        imgui.text(str(index))
        imgui.same_line()
        is_vector = keyframe.kind == "vec"
        # "Vec" and "Proj" rather than "Vector" and "Project": short enough
        # that they never became part of what this row had to buy width
        # back from.
        changed_kind, checked = imgui.checkbox("Vec", is_vector)
        if changed_kind:
            new_kind = "vec" if checked else "seed"
            self._set_keyframe(index, dataclasses.replace(keyframe, kind=new_kind))
            is_vector = checked
        imgui.same_line()
        changed_project, project = imgui.checkbox("Proj", keyframe.project)
        if changed_project:
            self._set_keyframe(index, dataclasses.replace(keyframe, project=project))
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
        `perform.py`'s own vector dialog: it is opened from inside a
        possibly-disabled row, but resolves on whatever later frame the
        performer picks a file on, by which time the row it was opened for
        may have moved or been removed, so the index is re-checked against
        the keyframes actually on hand right now rather than trusted blind.
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

    def _scrub_rows(self, state: ControlState) -> None:
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
        )

    def _noise_rows(self, state: ControlState) -> None:
        imgui.separator_text("Noise loop")
        self._binder.checkbox("noise_loop", "Noise loop")
        live = bool(self._binder.value("noise_loop"))
        # Typed fields, not drags: a seed's neighbours are unrelated noise
        # patterns, and every distinct radius costs a table build (up to
        # ~4s at the top of its range now, see `params.py`), so nothing
        # about either is a position worth sweeping through mid-drag.
        self._binder.input_int("noise_loop_seed", "Seed", enabled=live)
        self._binder.input_float(
            "noise_radius", "Radius", enabled=live, format="%.2f"
        )
        if live:
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
        imgui.separator_text("Pulse")
        self._binder.input_text("pulse_address", "Address", hint="/pulse")
        self._binder.input_text("pulse_ip", "IP", hint="127.0.0.1")
        self._note(_IP_NOTE)
        # A typed field: a port has no meaningful neighbour to scrub through
        # on the way to the one a performer actually wants.
        self._binder.input_int("pulse_port", "Port")

    def _note(self, text: str) -> None:
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(text)
        imgui.pop_style_color()
