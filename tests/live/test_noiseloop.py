import math
import threading
import time

import numpy as np

from autolume.live.core.noiseloop import NoiseLoop, NoiseLoopTable, NoiseLoopTableBuilder


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


def test_sample_matches_vector_per_dimension():
    """`NoiseLoopTable`'s builder samples one dimension at a time to have
    somewhere to yield the GIL; it must see the same values `vector()`
    reports for the whole row."""
    loop = NoiseLoop(seed=5, radius=1.5, dim=6)
    vector = loop.vector(0.61)
    assert tuple(loop.sample(i, 0.61) for i in range(6)) == vector


# --- NoiseLoopTable: interpolation and periodicity --------------------------


def _table_of(seed, radius, dim, steps):
    loop = NoiseLoop(seed, radius, dim)
    values = np.array(
        [loop.vector(step / steps) for step in range(steps)], dtype=np.float32
    )
    return NoiseLoopTable(key=(seed, radius, dim), values=values)


def test_table_vector_matches_a_sample_exactly_at_a_grid_point():
    table = _table_of(seed=1, radius=1.0, dim=4, steps=8)
    for step in range(8):
        assert tuple(table.values[step].tolist()) == table.vector(step / 8)


def test_table_vector_interpolates_between_grid_points():
    values = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    table = NoiseLoopTable(key=(0, 1.0, 2), values=values)
    got = table.vector(0.125)  # halfway between step 0 and step 1 of 4
    assert math.isclose(got[0], 0.5, abs_tol=1e-6)
    assert math.isclose(got[1], 1.0, abs_tol=1e-6)


def test_table_vector_wraps_the_last_step_back_to_the_first():
    values = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    table = NoiseLoopTable(key=(0, 1.0, 1), values=values)
    # Halfway between the last step (index 3) and the first (index 0, the wrap).
    got = table.vector(7 / 8)
    assert math.isclose(got[0], 2.5, abs_tol=1e-6)


def test_table_vector_is_exactly_periodic():
    table = _table_of(seed=2, radius=1.0, dim=4, steps=16)
    assert table.vector(0.0) == table.vector(1.0)


# --- NoiseLoopTableBuilder: background build, coalescing, immutable publish -


def _fake_table(key, steps):
    dim = key[2]
    return NoiseLoopTable(key=key, values=np.zeros((steps, dim), dtype=np.float32))


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def test_table_builder_publishes_nothing_before_a_request():
    builder = NoiseLoopTableBuilder(build=_fake_table)
    assert builder.store.snapshot() is None
    builder.stop()


def test_table_builder_builds_in_the_background():
    builder = NoiseLoopTableBuilder(build=_fake_table)
    builder.request_build((1, 1.0, 4))
    assert _wait_for(lambda: builder.store.snapshot() is not None)
    assert builder.store.snapshot().key == (1, 1.0, 4)
    builder.stop()


def test_table_builder_coalesces_to_the_newest_key():
    """Rapid requests (a dragged radius slider) must not queue every one."""
    release = threading.Event()
    built = []

    def slow_build(key, steps):
        release.wait(timeout=2.0)
        built.append(key)
        return _fake_table(key, steps)

    builder = NoiseLoopTableBuilder(build=slow_build)
    builder.request_build((1, 1.0, 4))
    builder.request_build((2, 1.0, 4))
    builder.request_build((3, 1.0, 4))
    release.set()
    assert _wait_for(
        lambda: builder.store.snapshot() is not None
        and builder.store.snapshot().key == (3, 1.0, 4)
    )
    assert (2, 1.0, 4) not in built
    builder.stop()


def test_table_builder_swallows_a_failing_build_without_raising():
    def failing(key, steps):
        raise RuntimeError("boom")

    builder = NoiseLoopTableBuilder(build=failing)
    builder.request_build((1, 1.0, 4))
    time.sleep(0.05)
    assert builder.store.snapshot() is None

    # Swap before requesting, so there is no race with the worker thread
    # picking the new key up under the old build function.
    builder._build = _fake_table
    builder.request_build((2, 1.0, 4))  # the thread must still be alive
    assert _wait_for(lambda: builder.store.snapshot() is not None)
    builder.stop()


def test_table_builder_end_to_end_matches_direct_sampling_closely():
    """The real build path (real yielding, real NoiseLoop), just over a
    short enough cycle that waiting for it costs nothing in the suite."""
    builder = NoiseLoopTableBuilder(steps=64)
    builder.request_build((9, 1.0, 5))
    assert _wait_for(lambda: builder.store.snapshot() is not None, timeout=5.0)
    table = builder.store.snapshot()
    direct = NoiseLoop(9, 1.0, 5)
    worst = max(
        abs(a - b)
        for alpha in (i / 200 for i in range(200))
        for a, b in zip(table.vector(alpha), direct.vector(alpha))
    )
    assert worst < 0.05
    builder.stop()


# --- fidelity: table-plus-interpolation vs direct sampling -------------------
#
# Required measurement (task-5-report.md has the full numbers): the table
# must not visibly degrade the loop. The large radius extreme is the
# expensive, meaningful case (the circle covers much more of the noise field
# per step there), so it alone is checked at the real production step count;
# the other two are already accurate at far fewer steps and are kept cheap.


def _max_and_mean_error(seed, radius, dim, steps, samples=300):
    table = _table_of(seed, radius, dim, steps)
    direct = NoiseLoop(seed, radius, dim)
    diffs = [
        abs(a - b)
        for alpha in (i / samples for i in range(samples))
        for a, b in zip(table.vector(alpha), direct.vector(alpha))
    ]
    return max(diffs), sum(diffs) / len(diffs)


def test_fidelity_at_the_default_radius_is_negligible():
    worst, mean = _max_and_mean_error(seed=3, radius=1.0, dim=6, steps=512)
    assert worst < 0.001
    assert mean < 0.0005


def test_fidelity_at_the_small_radius_extreme_is_negligible():
    worst, mean = _max_and_mean_error(seed=3, radius=0.01, dim=6, steps=512)
    assert worst < 0.0001
    assert mean < 0.00005


def test_fidelity_at_the_large_radius_extreme_at_the_production_step_count():
    worst, mean = _max_and_mean_error(seed=3, radius=100.0, dim=4, steps=4096)
    assert worst < 0.01
    assert mean < 0.005
