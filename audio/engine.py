"""Headless audio engine: capture -> features -> publish. No imgui, no app UI."""

import logging

from audio.capture import (AudioStream, AudioStreamError, list_input_devices,
                           refresh_devices)
from audio.features import FEATURE_NAMES, FeatureExtractor
from audio.publisher import FeaturePublisher

logger = logging.getLogger(__name__)


class AudioEngine:
    """Owns device selection, the capture stream, feature extraction and publishing."""

    def __init__(self, dispatcher):
        self.publisher = FeaturePublisher(dispatcher)
        self.devices = list_input_devices()
        self.device_pos = 0  # position in self.devices, not a device index
        self.stream = None
        self.extractor = None
        self.error = None
        self.features = {name: 0.0 for name in FEATURE_NAMES}
        self.spectrum = None
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

    def enable(self):
        self.error = None
        self._compute_warned = False
        if not self.devices:
            self.error = "No audio input devices found"
            return
        index, label = self.devices[self.device_pos]
        try:
            self.stream = AudioStream(index)
            self.extractor = FeatureExtractor(self.stream.sample_rate)
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
        try:
            refresh_devices()
        except Exception:
            logger.exception("Audio device refresh failed")
        self.devices = list_input_devices()
        self.device_pos = 0

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
        self.publisher.publish(self.features)
