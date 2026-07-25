"""Preview panel: displays the newest rendered frame."""

from imgui_bundle import imgui, immvision


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
