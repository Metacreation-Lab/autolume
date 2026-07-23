import numpy as np

from audio.features import FeatureExtractor, FEATURE_NAMES

SR = 44100
N_FFT = 2048


def sine(freq, n=N_FFT, amplitude=0.5):
    t = np.arange(n) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_names_and_bounds_on_noise():
    rng = np.random.default_rng(0)
    extractor = FeatureExtractor(SR)
    for _ in range(50):
        features, spectrum = extractor.compute(rng.uniform(-0.5, 0.5, N_FFT).astype(np.float32))
        assert sorted(features) == sorted(FEATURE_NAMES)
        assert all(0.0 <= v <= 1.0 for v in features.values())
        assert spectrum.shape == (N_FFT // 2 + 1,)


def test_silence_produces_zeros():
    extractor = FeatureExtractor(SR)
    for _ in range(10):
        features, _ = extractor.compute(np.zeros(N_FFT, dtype=np.float32))
    assert features["level"] == 0.0
    assert features["bass"] == 0.0
    assert features["onset"] == 0.0


def test_bass_sine_drives_bass_not_high():
    extractor = FeatureExtractor(SR)
    for _ in range(30):
        features, _ = extractor.compute(sine(80))
    assert features["bass"] > 0.5
    assert features["high"] < 0.1


def test_high_sine_drives_high_not_bass():
    extractor = FeatureExtractor(SR)
    for _ in range(30):
        features, _ = extractor.compute(sine(8000))
    assert features["high"] > 0.5
    assert features["bass"] < 0.1


def test_onset_fires_on_burst_after_silence():
    extractor = FeatureExtractor(SR)
    for _ in range(15):
        features, _ = extractor.compute(np.zeros(N_FFT, dtype=np.float32))
    features, _ = extractor.compute(sine(200, amplitude=0.9))
    assert features["onset"] == 1.0
    features, _ = extractor.compute(sine(200, amplitude=0.9))
    assert features["onset"] == 0.0  # refractory
