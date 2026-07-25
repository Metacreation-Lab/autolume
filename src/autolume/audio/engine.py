"""Headless audio engine: capture -> features -> publish. No imgui, no app UI."""

import logging

from autolume.audio.capture import (AudioStream, AudioStreamError,
                                    list_input_devices, refresh_devices)
from autolume.audio.features import (FEATURE_NAMES, ONSET_SENSITIVITY_DEFAULT,
                                     FeatureExtractor)

logger = logging.getLogger(__name__)


class AudioEngine:
    """Owns device selection, the capture stream, feature extraction and publishing.

    `publish` is called with the feature mapping after every successful update.
    """

    def __init__(self, publish):
        self._publish = publish
        self.devices = list_input_devices()
        self.device_pos = 0  # position in self.devices, not a device index
        self.stream = None
        self.extractor = None
        self.error = None
        self.features = {name: 0.0 for name in FEATURE_NAMES}
        self.spectrum = None
        self.onset_sensitivity = ONSET_SENSITIVITY_DEFAULT
        self._compute_warned = False

    @property
    def enabled(self):
        return self.stream is not None

    @property
    def sample_rate(self):
        return self.stream.sample_rate if self.stream is not None else 0

    def select_device(self, pos):
        if not self.enabled:
            self.device_pos = pos

    def set_onset_sensitivity(self, value):
        self.onset_sensitivity = min(1.0, max(0.0, value))
        if self.extractor is not None:
            self.extractor.onset_sensitivity = self.onset_sensitivity

    def enable(self):
        self.error = None
        self._compute_warned = False
        if not self.devices:
            self.error = "No audio input devices found"
            return
        index, label = self.devices[self.device_pos]
        try:
            self.stream = AudioStream(index)
            self.extractor = FeatureExtractor(self.stream.sample_rate,
                                              onset_sensitivity=self.onset_sensitivity)
        except AudioStreamError as exc:
            self.error = str(exc)
            self.stream = None

    def disable(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.extractor = None
        self.spectrum = None
        self.features = {name: 0.0 for name in FEATURE_NAMES}

    def close(self):
        self.disable()

    def refresh(self):
        if self.enabled:
            return
        selected = (self.devices[self.device_pos][1]
                    if 0 <= self.device_pos < len(self.devices) else None)
        try:
            refresh_devices()
        except Exception:
            logger.exception("Audio device refresh failed")
        self.devices = list_input_devices()
        # Keep the prior selection if that device is still present (indices can
        # shift after re-enumeration, so match by name).
        self.device_pos = next((pos for pos, (_, label) in enumerate(self.devices)
                                if label == selected), 0)

    def update(self):
        """Read the stream, compute features, publish them. Call once per frame."""
        if not self.enabled:
            return
        if not self.stream.active:
            self.error = f"{self.stream.device_name} stopped. Check the device and enable again."
            self.disable()
            return
        try:
            self.features, self.spectrum = self.extractor.compute(self.stream.read())
            self._compute_warned = False
        except Exception:
            if not self._compute_warned:
                logger.exception("Audio feature computation failed")
                self._compute_warned = True
            return
        self._publish(self.features)
