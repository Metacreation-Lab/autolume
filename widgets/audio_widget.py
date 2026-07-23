import imgui
import numpy as np

from assets import ACTIVE_RED
from audio.engine import AudioEngine
from audio.features import FEATURE_NAMES
from audio.publisher import OSC_PREFIX
from utils.gui_utils import imgui_utils


class AudioWidget:
    """Thin imgui view over a headless AudioEngine."""

    def __init__(self, viz):
        self.viz = viz
        self.engine = AudioEngine(viz.osc_dispatcher)
        self._scale_max = 0.0
        self._onset_glow = 0.0

    def close(self):
        self.engine.close()

    def _dot(self, color):
        draw_list = imgui.get_window_draw_list()
        size = imgui.get_text_line_height()
        pos = imgui.get_cursor_screen_pos()
        draw_list.add_circle_filled(pos.x + size * 0.5, pos.y + size * 0.5, size * 0.28,
                                    imgui.get_color_u32_rgba(*color))
        imgui.dummy(size, size)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        self.engine.update()
        if not show:
            return
        viz = self.viz
        engine = self.engine

        imgui.text("Input")
        imgui.same_line()
        labels = [label for _, label in engine.devices] or ["No input devices"]
        device_pos = min(engine.device_pos, len(labels) - 1)
        with imgui_utils.item_width(viz.app.font_size * 14), \
                imgui_utils.grayed_out(engine.enabled):
            changed, selected = imgui.combo("##audio_device", device_pos, labels)
            if changed and not engine.enabled:
                engine.select_device(selected)
        imgui.same_line()
        if imgui_utils.button("Refresh##audio", enabled=not engine.enabled):
            engine.refresh()
        imgui.same_line()
        toggle_label = "Disable##audio" if engine.enabled else "Enable##audio"
        if imgui_utils.button(toggle_label, enabled=bool(engine.devices) or engine.enabled):
            if engine.enabled:
                engine.disable()
            else:
                engine.enable()

        if engine.error:
            imgui.text_colored(engine.error, 1.0, 0.4, 0.4, 1.0)

        if engine.enabled:
            self._dot(ACTIVE_RED)
            imgui.same_line()
            imgui.text(f"Running at {engine.sample_rate / 1000:g} kHz")
        else:
            self._dot((0.5, 0.5, 0.5, 1.0))
            imgui.same_line()
            imgui.text("Not running")

        if engine.spectrum is not None:
            values = engine.spectrum.astype(np.float32)
            self._scale_max = 0.2 * float(values.max()) + 0.8 * self._scale_max
            imgui.plot_histogram("##audio_spectrum", values,
                                 graph_size=(viz.app.font_size * 24,
                                             imgui.get_text_line_height_with_spacing() * 2),
                                 scale_min=0.0, scale_max=max(self._scale_max, 1e-6))
        else:
            imgui.plot_histogram("##audio_spectrum", np.zeros(64, dtype=np.float32),
                                 graph_size=(viz.app.font_size * 24,
                                             imgui.get_text_line_height_with_spacing() * 2),
                                 scale_min=0.0, scale_max=1.0)

        imgui.separator()
        meter_x = viz.app.font_size * 5
        meter_w = viz.app.font_size * 10
        with imgui_utils.grayed_out(not engine.enabled):
            for name in FEATURE_NAMES:
                value = engine.features[name]
                imgui.text(name.capitalize())
                imgui.same_line(meter_x)
                if name == "onset":
                    self._onset_glow = max(value, self._onset_glow * 0.85)
                    color = ACTIVE_RED if engine.enabled and self._onset_glow > 0.05 else (0.5, 0.5, 0.5, 1.0)
                    self._dot(color)
                else:
                    imgui.progress_bar(value, (meter_w, 0), "")
                imgui.same_line(meter_x + meter_w + viz.app.font_size)
                imgui.text_disabled(OSC_PREFIX + name)
