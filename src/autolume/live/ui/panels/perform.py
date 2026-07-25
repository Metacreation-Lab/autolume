"""Perform panel: the parameters played during a show."""

from typing import Callable

from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.ui.controls import ERROR_COLOR, ControlBinder


class PerformPanel:
    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        self._open_dialog: pfd.open_file | None = None

    def gui(self) -> None:
        # The binder's snapshot rather than one of our own, so the model row
        # and the controls beside it always describe the same frame.
        self._model_row(self._binder.state())
        self._latent_rows()
        self._noise_rows()
        self._render_rows()
        self._status_row()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _model_row(self, state) -> None:
        if imgui.button("Open model"):
            self._open_dialog = pfd.open_file(
                "Choose a model pickle", "", ["Model files", "*.pkl"]
            )
        if self._open_dialog is not None and self._open_dialog.ready():
            result = self._open_dialog.result()
            if result:
                self._emit("/model/path", str(result[0]))
            self._open_dialog = None
        imgui.same_line()
        if self._runtime.model_host.loading():
            imgui.text_disabled("Loading")
        elif state.pkl_path:
            imgui.text(state.pkl_path)
        else:
            imgui.text_disabled("No model loaded")
        error = self._runtime.model_host.error()
        if error:
            imgui.text_colored(imgui.ImVec4(*ERROR_COLOR), error)

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
