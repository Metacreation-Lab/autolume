import math

from autolume.live.core.noiseloop import NoiseLoop


def test_vector_length_matches_dim():
    loop = NoiseLoop(seed=0, radius=1.0, dim=8)
    assert len(loop.vector(0.3)) == 8


def test_vector_is_periodic_across_a_full_cycle():
    loop = NoiseLoop(seed=1, radius=2.0, dim=16)
    start = loop.vector(0.0)
    end = loop.vector(1.0)
    assert len(start) == len(end)
    for a, b in zip(start, end):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_same_seed_and_radius_are_deterministic():
    first = NoiseLoop(seed=42, radius=1.5, dim=32).vector(0.37)
    second = NoiseLoop(seed=42, radius=1.5, dim=32).vector(0.37)
    assert first == second


def test_different_seed_gives_a_different_vector():
    a = NoiseLoop(seed=1, radius=1.0, dim=32).vector(0.37)
    b = NoiseLoop(seed=2, radius=1.0, dim=32).vector(0.37)
    assert a != b


def test_small_radius_stays_finite():
    loop = NoiseLoop(seed=0, radius=0.01, dim=32)
    vec = loop.vector(0.5)
    assert all(math.isfinite(v) for v in vec)


def test_large_radius_stays_finite():
    loop = NoiseLoop(seed=0, radius=100.0, dim=32)
    vec = loop.vector(0.5)
    assert all(math.isfinite(v) for v in vec)
