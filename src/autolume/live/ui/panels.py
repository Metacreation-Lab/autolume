"""Control panels: pure views that read snapshots and emit control events."""

from imgui_bundle import imgui, immvision, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.core.params import REGISTRY


def _spec(name):
    return REGISTRY[name]


class PerformPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._open_dialog: pfd.open_file | None = None

    def gui(self) -> None:
        state = self._runtime.control_store.snapshot()
        self._model_row(state)
        self._latent_rows(state)
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
        changed, value = imgui.drag_float("Latent x", state.latent_x, 0.01)
        if changed:
            self._emit("/latent/x", value)
        changed, value = imgui.drag_float("Latent y", state.latent_y, 0.01)
        if changed:
            self._emit("/latent/y", value)
        changed, value = imgui.checkbox("Animate", state.anim_playing)
        if changed:
            self._emit("/anim/playing", value)
        spec = _spec("anim_speed_x")
        changed, value = imgui.slider_float(
            "Speed x", state.anim_speed_x, spec.minimum, spec.maximum
        )
        if changed:
            self._emit("/anim/speed/x", value)
        spec = _spec("anim_speed_y")
        changed, value = imgui.slider_float(
            "Speed y", state.anim_speed_y, spec.minimum, spec.maximum
        )
        if changed:
            self._emit("/anim/speed/y", value)

    def _render_rows(self, state) -> None:
        spec = _spec("truncation_psi")
        changed, value = imgui.slider_float(
            "Truncation", state.truncation_psi, spec.minimum, spec.maximum
        )
        if changed:
            self._emit("/trunc/psi", value)
        spec = _spec("fps_cap")
        changed, value = imgui.slider_int(
            "Frame limit", state.fps_cap, int(spec.minimum), int(spec.maximum)
        )
        if changed:
            self._emit("/render/fps", value)

    def _status_row(self) -> None:
        imgui.separator()
        imgui.text(f"Render {self._runtime.render_loop.fps():.1f} fps")
        if self._runtime.osc.port is not None:
            imgui.text(f"OSC port {self._runtime.osc.port}")


class PreviewPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._last_seq = -1

    def gui(self) -> None:
        seq, frame = self._runtime.preview.latest()
        if frame is None:
            imgui.text_disabled("Waiting for frames")
            return
        refresh = seq != self._last_seq
        self._last_seq = seq
        immvision.image_display_resizable(
            "##preview", frame, refresh_image=refresh
        )
