"""Loop panel: keyframe and noise loop playback, and the outbound sync pulse.

Everything here is either a bindable registry parameter, drawn through
`ControlBinder` like any other performable row, or a keyframe edit, which is
structured state with no registry address of its own (design.md). Keyframe
rows are plain widgets for that reason: there is no binding to mark and no
mapping row to open, the same argument that leaves the vector row in
`perform.py` without a driver marker.

The noise loop's radius and seed rebuild a background table that can take
tens of seconds at the far end of the radius range (`core/noiseloop.py`).
Motion keeps running on the previous table while that happens, which is
correct and would otherwise look broken: nothing on screen says a rebuild is
under way. `_noise_pending_row` is the whole of what says so, read off the
same published snapshot the control loop itself reads rather than a new
channel of its own.
"""

import dataclasses
from typing import Callable

from imgui_bundle import imgui

from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import ModelInfo
from autolume.live.core.params import (
    KEYFRAME_REMOVE,
    KEYFRAME_SET,
    REGISTRY,
    ControlState,
    Keyframe,
    RemoveKeyframe,
    SetKeyframe,
)
from autolume.live.ui.controls import ControlBinder

_SEED_WIDTH_EMS = 1.4
_PENDING_NOTE = (
    "Noise table is rebuilding. "
    "Motion stays on the previous table until it is ready."
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

    def gui(self) -> None:
        state = self._binder.state()
        self._transport_rows()
        self._keyframe_rows(state)
        self._scrub_rows()
        self._noise_rows(state)
        self._pulse_rows()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _transport_rows(self) -> None:
        """Play, the time or speed switch, and Perfect loop.

        Both the seconds and the speed control are always drawn, one of them
        greyed, the same no-reflow rule the latent section follows: a mode
        switch never changes which rows are on screen, only whether the hand
        can reach them.
        """
        imgui.separator_text("Transport")
        self._binder.checkbox("loop_active", "Play")
        self._binder.checkbox("loop_uses_time", "Use time")
        uses_time = bool(self._binder.value("loop_uses_time"))
        self._binder.slider_float("loop_time", "Seconds", enabled=uses_time)
        self._binder.slider_float("loop_speed", "Speed", enabled=not uses_time)
        self._binder.checkbox("perfect_loop", "Perfect loop")

    def _keyframe_rows(self, state: ControlState) -> None:
        imgui.separator_text("Keyframes")
        self._binder.slider_int("keyframe_count", "Keyframes")
        imgui.same_line()
        if imgui.button("Add"):
            self._emit(REGISTRY["keyframe_count"].address, len(state.keyframes) + 1)
        count = len(state.keyframes)
        for index, keyframe in enumerate(state.keyframes):
            self._keyframe_row(index, keyframe, state, count)

    def _keyframe_row(
        self, index: int, keyframe: Keyframe, state: ControlState, count: int
    ) -> None:
        """One keyframe, on two lines: kind and Project, then its content.

        Two lines rather than one because one line does not fit the panel at
        any docked width this suite checks (task 9 review, finding 1): the
        old app had the same seven widgets on one line and simply had no
        test to notice it did not fit either, which is parity with a broken
        layout rather than a reason to keep it. The second line is indented
        to read as a continuation of the first rather than a row of its own.

        The content half is two seed fields or nothing, never both, so a
        vector keyframe's row is not left with two live number fields for a
        value it cannot hold. Snap is offered either way: it is the only way
        to put content into a vector keyframe at all, and a shortcut for a
        seed one.
        """
        imgui.push_id(index)
        is_vector = self._keyframe_kind_row(index, keyframe)
        self._keyframe_content_row(index, keyframe, state, is_vector, count)
        imgui.pop_id()

    def _keyframe_kind_row(self, index: int, keyframe: Keyframe) -> bool:
        imgui.text(str(index))
        imgui.same_line()
        is_vector = keyframe.kind == "vec"
        # "Vec" and "Proj" rather than "Vector" and "Project": at the
        # narrowest docked width and the largest font scale this suite
        # checks, the full words alone overflow the panel before the
        # content row even starts (task 9 review, finding 1, round 2).
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

    def _keyframe_content_row(
        self,
        index: int,
        keyframe: Keyframe,
        state: ControlState,
        is_vector: bool,
        count: int,
    ) -> None:
        imgui.indent()
        self._keyframe_seed_fields(index, keyframe, is_vector)
        imgui.same_line()
        if imgui.button("Snap"):
            self._set_keyframe(index, captured_keyframe(keyframe, state))
        imgui.same_line()
        if count <= 1:
            imgui.begin_disabled()
        if imgui.button("Del"):
            self._emit(KEYFRAME_REMOVE, RemoveKeyframe(index))
        if count <= 1:
            imgui.end_disabled()
        imgui.unindent()

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

    def _set_keyframe(self, index: int, keyframe: Keyframe) -> None:
        self._emit(KEYFRAME_SET, SetKeyframe(index, keyframe))

    def _scrub_rows(self) -> None:
        """Alpha and index, both ordinary bound controls.

        Touch grace already makes scrubbing and integration coexist for
        every other motion-owned parameter (`motion.py`), and the loop's
        phase is no different, so nothing extra is needed here to make
        dragging this slider hold the loop still for the length of the drag.
        """
        imgui.separator_text("Scrub")
        self._binder.slider_float("loop_alpha", "Alpha")
        self._binder.drag_int("loop_index", "Index")

    def _noise_rows(self, state: ControlState) -> None:
        imgui.separator_text("Noise loop")
        self._binder.checkbox("noise_loop", "Noise loop")
        live = bool(self._binder.value("noise_loop"))
        self._binder.drag_int("noise_loop_seed", "Seed", enabled=live)
        # Commits only on release: a radius rebuilds a table that can take
        # tens of seconds at the top of its range (core/noiseloop.py), and a
        # dragged slider must not queue one rebuild request per tick.
        self._binder.slider_float(
            "noise_radius", "Radius", enabled=live, commit_on_release=True
        )
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
            imgui.text_disabled(_PENDING_NOTE)

    def _pulse_rows(self) -> None:
        imgui.separator_text("Pulse")
        self._binder.input_text("pulse_address", "Address", hint="/pulse")
        self._binder.input_text("pulse_ip", "IP", hint="127.0.0.1")
        self._note(_IP_NOTE)
        self._binder.drag_int("pulse_port", "Port")

    def _note(self, text: str) -> None:
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(text)
        imgui.pop_style_color()
