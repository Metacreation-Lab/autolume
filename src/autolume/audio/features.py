"""Scalar audio features for OSC mapping. Pure numpy, no audio I/O."""

from collections import deque

import numpy as np

FEATURE_NAMES = ["level", "bass", "mid", "high", "onset"]

# Band edges in Hz. high runs to Nyquist.
BANDS = {"bass": (20, 150), "mid": (150, 2000), "high": (2000, None)}

# Onset never fires below this raw RMS level, so pure silence cannot trigger it.
# This is a fixed floor, not a user control.
ONSET_SILENCE_FLOOR = 0.01

# Onset sensitivity runs 0 (off) to 1 (many hits). Above 0 it maps to the
# spectral-flux threshold as mean + K*std, with K interpolated MAX -> MIN.
# Exactly 0 disables onset entirely. Default keeps K near 3.4.
ONSET_SENSITIVITY_DEFAULT = 0.65
ONSET_K_MIN = 1.0
ONSET_K_MAX = 8.0


class AutoGain:
    """Normalize a nonnegative value to 0-1 against a decaying running max."""

    def __init__(self, decay=0.995, gate=1e-4):
        self.decay = decay
        self.gate = gate
        self.running_max = 0.0

    def __call__(self, value):
        self.running_max = max(value, self.running_max * self.decay)
        if self.running_max <= self.gate:
            return 0.0
        return min(value / self.running_max, 1.0)


class Smoother:
    """Exponential smoothing with fast attack and slow release."""

    def __init__(self, attack=0.6, release=0.15):
        self.attack = attack
        self.release = release
        self.value = 0.0

    def __call__(self, target):
        coeff = self.attack if target > self.value else self.release
        self.value += coeff * (target - self.value)
        return self.value


class FeatureExtractor:
    """Compute normalized scalar features from the latest mono sample window."""

    def __init__(self, sample_rate, n_fft=2048, onset_sensitivity=ONSET_SENSITIVITY_DEFAULT):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.onset_sensitivity = onset_sensitivity
        self._window = np.hanning(n_fft).astype(np.float32)
        self._freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self._band_masks = {}
        for name, (lo, hi) in BANDS.items():
            mask = self._freqs >= lo
            if hi is not None:
                mask &= self._freqs < hi
            self._band_masks[name] = mask
        self._gains = {name: AutoGain() for name in ["level", *BANDS]}
        self._smoothers = {name: Smoother() for name in ["level", *BANDS]}
        self._prev_mag = None
        self._flux_history = deque(maxlen=43)
        self._refractory = 0

    def compute(self, samples):
        x = np.asarray(samples, dtype=np.float32)[-self.n_fft:]
        if x.shape[0] < self.n_fft:
            x = np.pad(x, (self.n_fft - x.shape[0], 0))
        mag = np.abs(np.fft.rfft(x * self._window)) * (2.0 / self.n_fft)

        raw = {"level": float(np.sqrt(np.mean(np.square(x))))}
        for name, mask in self._band_masks.items():
            raw[name] = float(mag[mask].mean()) if mask.any() else 0.0

        features = {name: self._smoothers[name](self._gains[name](value))
                    for name, value in raw.items()}
        features["onset"] = self._detect_onset(mag, raw["level"])
        return features, mag

    def _detect_onset(self, mag, level):
        if self._prev_mag is None:
            self._prev_mag = mag
            return 0.0
        flux = float(np.maximum(mag - self._prev_mag, 0.0).sum())
        self._prev_mag = mag

        onset = 0.0
        if self._refractory > 0:
            self._refractory -= 1
        elif (self.onset_sensitivity > 0.0 and len(self._flux_history) >= 10
              and level >= ONSET_SILENCE_FLOOR):
            k = ONSET_K_MAX - self.onset_sensitivity * (ONSET_K_MAX - ONSET_K_MIN)
            history = np.array(self._flux_history)
            threshold = history.mean() + k * history.std()
            if flux > threshold and flux > 1e-4:
                onset = 1.0
                self._refractory = 5
        self._flux_history.append(flux)
        return onset
