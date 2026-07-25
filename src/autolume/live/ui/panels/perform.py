"""Perform panel: the parameters played during a show.

Every row is the same row: a chip saying what drives the parameter, then the
control. The model is one of them. It used to be a button and a label, drawn by
hand, which was already the odd one out and became a wrong one once a mapping
row could switch models from a controller: the parameter could be driven and
the row was the only one with nowhere to say so.

What the model is doing goes in the preview rather than here. A load takes
seconds and can fail, and the surface the performer is looking at while it
happens is the one showing the frames it is about to change.
"""

from typing import Callable

from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.ui.controls import ControlBinder

_BROWSE = "Browse"
_NO_MODEL = "No model loaded"


def button_width(label: str) -> float:
    """How wide a button reading `label` will be drawn.

    Measured rather than fixed, so the row keeps its proportions at every
    display scale and in whatever the theme sets for padding.
    """
    return imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 2.0


class PerformPanel:
    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        self._open_dialog: pfd.open_file | None = None

    def gui(self) -> None:
        self._model_row()
        self._latent_rows()
        self._noise_rows()
        self._render_rows()
        self._status_row()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _model_row(self) -> None:
        """The model, as an ordinary row: chip, path, Browse.

        The field holds the path itself, so a path can be pasted or typed as
        well as picked, and it commits on Enter or on losing focus.

        Browse is drawn disabled exactly when the field is, which the binder
        reports: a source driving the parameter would overwrite whatever the
        dialog picked. The chip is drawn before either of them and stays
        clickable, so it is still the way back.
        """
        imgui.separator_text("Model")
        reserve = button_width(_BROWSE) + imgui.get_style().item_spacing.x
        live = self._binder.input_text(
            "pkl_path", "##model", hint=_NO_MODEL, reserve=reserve
        )
        imgui.same_line()
        if not live:
            imgui.begin_disabled()
        if imgui.button(_BROWSE):
            self._open_dialog = pfd.open_file(
                "Choose a model pickle", "", ["Model files", "*.pkl"]
            )
        if not live:
            imgui.end_disabled()
        # Drained outside the disabled block, so a dialog opened before a
        # source took the row over still delivers what it was opened for.
        self._take_dialog_result()

    def _take_dialog_result(self) -> None:
        if self._open_dialog is None or not self._open_dialog.ready():
            return
        result = self._open_dialog.result()
        self._open_dialog = None
        if result:
            self._emit("/model/path", str(result[0]))

    def _latent_rows(self) -> None:
        imgui.separator_text("Latent")
        self._binder.drag_float("latent_x", "Latent x")
        self._binder.drag_float("latent_y", "Latent y")
        self._binder.checkbox("anim_playing", "Animate")
        self._binder.slider_float("anim_speed_x", "Speed x")
        self._binder.slider_float("anim_speed_y", "Speed y")
        self._binder.slider_float("truncation_psi", "Truncation")

    def _noise_rows(self) -> None:
        imgui.separator_text("Noise")
        self._binder.checkbox("noise_enabled", "Noise on")
        # What the box shows, not what the store holds, so the rows grey in the
        # same frame as the click that greys them.
        live = bool(self._binder.value("noise_enabled"))
        self._binder.slider_float("global_noise", "Amount", enabled=live)
        self._binder.drag_int("noise_seed", "Seed", enabled=live)
        self._binder.checkbox("noise_anim", "Animate noise", enabled=live)

    def _render_rows(self) -> None:
        imgui.separator_text("Render")
        self._binder.slider_int("fps_cap", "Frame limit")

    def _status_row(self) -> None:
        imgui.separator()
        imgui.text(f"Render {self._runtime.render_loop.fps():.1f} fps")
        if self._runtime.osc.port is not None:
            imgui.text(f"OSC port {self._runtime.osc.port}")
