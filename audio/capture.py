"""Mono audio input via sounddevice. Device listing is metadata only."""

import logging
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioStreamError(Exception):
    """Raised when an audio input stream cannot be opened."""


def list_input_devices():
    """Return (index, label) for every input device, system default first."""
    hostapis = sd.query_hostapis()
    devices = [(index, info) for index, info in enumerate(sd.query_devices())
               if info["max_input_channels"] > 0]
    names = [info["name"] for _, info in devices]
    try:
        default = sd.default.device[0]
    except Exception:
        default = -1

    result = []
    for index, info in devices:
        label = info["name"]
        if names.count(label) > 1:
            label = f"{label} ({hostapis[info['hostapi']]['name']})"
        result.append((index, label))
    result.sort(key=lambda item: item[0] != default)
    return result


def refresh_devices():
    """Re-snapshot the device list. PortAudio caches it at initialization."""
    sd._terminate()
    sd._initialize()


class AudioStream:
    """Input stream keeping the most recent window of mono samples."""

    def __init__(self, device, window_size=2048):
        self.device_name = f"device {device}"
        self.sample_rate = 0
        self._buffer = np.zeros(window_size, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None
        try:
            info = sd.query_devices(device)
            self.device_name = info["name"]
            self.sample_rate = int(info["default_samplerate"])
            self._stream = sd.InputStream(device=device, channels=1,
                                          samplerate=self.sample_rate,
                                          dtype="float32",
                                          callback=self._callback)
            self._stream.start()
        except Exception as exc:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
            raise AudioStreamError(
                f"Could not open {self.device_name}. {exc}") from exc
        logger.info("Audio stream started on %s at %d Hz",
                    self.device_name, self.sample_rate)

    def _callback(self, indata, frames, time_info, status):
        # Runs on the PortAudio thread. Copy only, never raise.
        try:
            if status:
                logger.debug("Audio stream status: %s", status)
            samples = indata[:, 0]
            with self._lock:
                n = min(len(samples), len(self._buffer))
                self._buffer = np.roll(self._buffer, -n)
                self._buffer[-n:] = samples[-n:]
        except Exception:
            logger.exception("Audio callback failed")

    def read(self):
        with self._lock:
            return self._buffer.copy()

    @property
    def active(self):
        try:
            return bool(self._stream.active)
        except Exception:
            return False

    def close(self):
        logger.info("Stopping audio stream")
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            logger.exception("Error closing audio stream")
