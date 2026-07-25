"""Perform panel: the parameters played during a show."""

from typing import Callable

from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.ui.controls import ControlBinder


class PerformPanel:
    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        self._open_dialog: pfd.open_file | None = None

    def gui(self) -> None:
        state = self._runtime.control_store.snapshot()
        self._model_row(state)
        self._latent_rows(state)
        self._noise_rows(state)
        self._render_rows(state)
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
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), error)

    def _latent_rows(self, state) -> None:
        imgui.separator_text("Latent")
        self._binder.drag_float("latent_x", "Latent x", state.latent_x)
        self._binder.drag_float("latent_y", "Latent y", state.latent_y)
        self._binder.checkbox("anim_playing", "Animate", state.anim_playing)
        self._binder.slider_float("anim_speed_x", "Speed x", state.anim_speed_x)
        self._binder.slider_float("anim_speed_y", "Speed y", state.anim_speed_y)
        self._binder.slider_float(
            "truncation_psi", "Truncation", state.truncation_psi
        )

    def _noise_rows(self, state) -> None:
        imgui.separator_text("Noise")
        self._binder.checkbox("noise_enabled", "Noise on", state.noise_enabled)
        live = state.noise_enabled
        self._binder.slider_float(
            "global_noise", "Amount", state.global_noise, enabled=live
        )
        self._binder.drag_int(
            "noise_seed", "Seed", state.noise_seed, enabled=live
        )
        self._binder.checkbox(
            "noise_anim", "Animate noise", state.noise_anim, enabled=live
        )

    def _render_rows(self, state) -> None:
        imgui.separator_text("Render")
        self._binder.slider_int("fps_cap", "Frame limit", state.fps_cap)

    def _status_row(self) -> None:
        imgui.separator()
        imgui.text(f"Render {self._runtime.render_loop.fps():.1f} fps")
        if self._runtime.osc.port is not None:
            imgui.text(f"OSC port {self._runtime.osc.port}")
