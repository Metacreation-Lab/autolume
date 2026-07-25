"""Audio panel: pick an input, watch what it hears, read what went wrong.

The panel is a view onto one `AudioStatus` snapshot per frame. It never
touches the engine, which lives on the audio thread. Every control turns
into a queued command instead, so choosing a device cannot block a frame.
"""

import math

import numpy as np
from imgui_bundle import imgui

from autolume.audio.features import FEATURE_NAMES

_ERROR_COLOR = (1.0, 0.3, 0.3, 1.0)
_NO_DEVICES = "No input devices found"
# Sized in multiples of the font size so the panel holds its proportions on
# every display scale.
_SPECTRUM_EMS = 4.0
_BAR_EMS = 12.0
# Keeps a silent input from being auto scaled up into a wall of noise.
_SPECTRUM_FLOOR = 0.02
_IDLE_SPECTRUM = np.zeros(64, dtype=np.float32)


def device_labels(devices: tuple[tuple[int, str], ...]) -> list[str]:
    """Combo entries for the device list, never empty so the combo stays drawn."""
    if not devices:
        return [_NO_DEVICES]
    return [label for _, label in devices]


def device_index(devices: tuple[tuple[int, str], ...], device_pos: int) -> int:
    """The combo selection, kept inside the list the panel is about to draw."""
    if not devices:
        return 0
    return min(max(device_pos, 0), len(devices) - 1)


def bar_value(value: object) -> float:
    """A feature as a number a bar can be drawn from.

    A NaN would propagate through the clamp and reach `progress_bar` still
    claiming to be in range, so it is shown as silence instead.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def spectrum_values(spectrum: np.ndarray | None) -> np.ndarray:
    """The spectrum as something `plot_histogram` accepts, idle bars if absent."""
    if spectrum is None or spectrum.size == 0:
        return _IDLE_SPECTRUM
    return np.ascontiguousarray(spectrum, dtype=np.float32)


def spectrum_ceiling(values: np.ndarray) -> float:
    """The top of the plot, floored so silence does not get scaled up to noise."""
    peak = float(values.max())
    if not math.isfinite(peak):
        return _SPECTRUM_FLOOR
    return max(peak, _SPECTRUM_FLOOR)


class AudioPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        # Held while the slider is dragged. The status only catches up after
        # the audio thread applies the command, and drawing the stale value in
        # between makes the handle stutter under the mouse.
        self._sensitivity: float | None = None

    def gui(self) -> None:
        status = self._runtime.audio.status()
        self._device_row(status)
        self._sensitivity_row(status)
        self._feature_rows(status)
        self._spectrum_row(status)
        self._error_row(status)

    def _device_row(self, status) -> None:
        imgui.separator_text("Input")
        labels = device_labels(status.devices)
        selected = device_index(status.devices, status.device_pos)
        # The engine refuses a device change while it is running, so the combo
        # says so by going grey rather than by accepting a click that does
        # nothing.
        locked = status.enabled or not status.devices
        if locked:
            imgui.begin_disabled()
        imgui.set_next_item_width(-1)
        changed, chosen = imgui.combo("##device", selected, labels)
        if locked:
            imgui.end_disabled()
        if changed and status.devices:
            self._runtime.audio.select_device(chosen)
        self._enable_button(status)
        imgui.same_line()
        # The only way out of an empty device list, which is what a performer
        # sees after plugging an interface in with Autolume already running.
        if imgui.button("Rescan"):
            self._runtime.audio.refresh()
        imgui.same_line()
        if status.enabled:
            imgui.text(f"{status.sample_rate} Hz")
        else:
            imgui.text_disabled("Audio is off")

    def _enable_button(self, status) -> None:
        # Sized for the wider of the two labels so the row does not resize
        # under the cursor when the button flips.
        width = imgui.get_style().frame_padding.x * 2.0 + max(
            imgui.calc_text_size("Enable").x, imgui.calc_text_size("Disable").x
        )
        label = "Disable" if status.enabled else "Enable"
        if imgui.button(label, imgui.ImVec2(width, 0.0)):
            if status.enabled:
                self._runtime.audio.disable()
            else:
                self._runtime.audio.enable()

    def _sensitivity_row(self, status) -> None:
        shown = (
            status.onset_sensitivity
            if self._sensitivity is None
            else self._sensitivity
        )
        changed, value = imgui.slider_float("Onset sensitivity", shown, 0.0, 1.0)
        if changed:
            self._sensitivity = value
            self._runtime.audio.set_onset_sensitivity(value)
        if imgui.is_item_deactivated():
            self._sensitivity = None

    def _feature_rows(self, status) -> None:
        imgui.separator_text("Features")
        # Driven by FEATURE_NAMES rather than by the snapshot, so the rows keep
        # their order and their count whatever the engine last reported.
        for name in FEATURE_NAMES:
            value = bar_value(status.features.get(name, 0.0))
            fraction = min(max(value, 0.0), 1.0)
            imgui.progress_bar(
                fraction,
                imgui.ImVec2(imgui.get_font_size() * _BAR_EMS, 0.0),
                f"{name} {value:.2f}",
            )

    def _spectrum_row(self, status) -> None:
        imgui.separator_text("Spectrum")
        values = spectrum_values(status.spectrum)
        imgui.plot_histogram(
            "##spectrum",
            values,
            scale_min=0.0,
            scale_max=spectrum_ceiling(values),
            graph_size=imgui.ImVec2(-1.0, imgui.get_font_size() * _SPECTRUM_EMS),
        )

    def _error_row(self, status) -> None:
        if not status.error:
            return
        imgui.separator()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_ERROR_COLOR))
        imgui.text_wrapped(status.error)
        imgui.pop_style_color()
