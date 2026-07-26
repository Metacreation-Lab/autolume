"""The simplex noise loop: a periodic latent vector driven by loop alpha.

`NoiseLoop` is ported from `widgets/looping_widget.py`'s `OSN` (project
authored, no NVIDIA header, direct port allowed per design.md's clean-room
rule). One `OpenSimplex` sampler per output dimension, each walked around a
circle of `radius` diameter; sampling all dimensions at the same angle keeps
every component periodic in lockstep, so `vector(0.0) == vector(1.0)`.

Direct sampling is correct but not cheap enough to call every control tick:
opensimplex 0.4.5 has no numba installed, so it is pure Python, measured at
~4.6 ms per `vector()` call at dim 512, which is over half of the 8 ms
budget at 125 Hz. Because the loop is periodic and deterministic, a full
cycle can be precomputed once into a table and looked up with linear
interpolation at a few microseconds per tick; `NoiseLoopTable` and
`NoiseLoopTableBuilder` below do that, off the control thread. See
`task-5-report.md` for the measurements this design was chosen from.

The step count is not fixed: `sample()` walks a circle of radius `radius /
2` (see below), so the circle's circumference scales linearly with
`radius`, and a bigger radius crosses proportionally more of the noise field
per cycle and needs proportionally more samples to resolve it. A fixed step
count sized for the largest allowed radius massively oversamples the common,
small-radius case, which is what made the first version of this table take
~35 s to build even at the default radius. `table_steps()` scales the count
to the radius instead, see its docstring and `task-5-report.md`.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from opensimplex import OpenSimplex

from autolume.live.core.store import LatestValueStore

logger = logging.getLogger(__name__)

# Samples per unit of radius. Fit from the measured minimal step count that
# holds max interpolation error at or under ~0.0064 (see task-5-report.md):
# that minimum scales linearly with radius at ~40 samples/unit once radius is
# a few units or more (e.g. minimal N was 120 at radius 3, 407 at radius 10,
# 4070 at radius 100 — all ~40x their radius); 42 keeps a small margin above
# that observed minimum.
_STEPS_PER_RADIUS_UNIT = 42
# Floor: below here the numeric error is already negligible regardless (a
# small radius barely moves through the noise field at all), but a table
# needs enough temporal samples to read as a smoothly varying loop rather
# than a handful of straight-line segments.
_TABLE_STEPS_MIN = 128
# Ceiling: the largest allowed radius (100.0, the top of `noise_radius`'s
# registry bounds) lands here; this is the previously validated worst case
# step count, error ~0.006 (mean ~0.001) out of a [-1, 1] range, which does
# not read as a visible degradation of the loop.
_TABLE_STEPS_MAX = 4096


def table_steps(radius: float) -> int:
    """How many samples one full cycle needs at `radius`, clamped to a sane range.

    Public (not `_`-prefixed) because Task 9's radius slider UI may want to
    know the step count a given radius will build, e.g. to estimate wait
    time; `ControlLoop` and `_build_table` are not the only reasonable
    callers.
    """
    return min(
        _TABLE_STEPS_MAX,
        max(_TABLE_STEPS_MIN, math.ceil(_STEPS_PER_RADIUS_UNIT * radius)),
    )


# A build samples up to `table_steps(radius) * dim` points with a pure
# Python library, so it must give the GIL back regularly or it starves the
# control thread for its whole duration. A real `time.sleep()` between small
# chunks measured far better than `sleep(0)` here: `sleep(0)` is a
# scheduling hint with no guaranteed OS timeslice, and calling it at high
# frequency under contention measured *worse* than not yielding at all
# (observed: p95 tick lateness in the hundreds of ms). An actual sleep
# forces this thread off the CPU for a real interval, which is what gives
# the control thread's tick a reliable window to run in. Tuned on the dev
# machine at dim 512 (the worst case, z_dim=512, at the top of the allowed
# radius range where the step count is largest): with a build in flight,
# the 125 Hz tick's p95 lateness measured ~5.8-6.3 ms against a ~2.7/3.6 ms
# idle median/p95, comfortably inside the 8 ms tick budget; a finer chunk
# (more, shorter sleeps) pushed the same build from ~35 s to ~57 s wall
# clock for a p95 that was already met, so it bought nothing. See
# task-5-report.md for the full measurements.
_YIELD_CHUNK_DIMS = 256
_YIELD_SLEEP_SECONDS = 0.001


def _valmap(value: float, istart: float, istop: float, ostart: float, ostop: float) -> float:
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


class NoiseLoop:
    def __init__(self, seed: int, radius: float, dim: int) -> None:
        self._radius = radius
        self._samplers = [OpenSimplex(seed=seed + i) for i in range(dim)]

    @property
    def dim(self) -> int:
        return len(self._samplers)

    def sample(self, index: int, alpha: float) -> float:
        """The value of dimension `index` alone, at `alpha`.

        `vector()` composes this over every dimension. `NoiseLoopTable`'s
        builder calls it one dimension (or a small chunk of dimensions) at a
        time instead, so it has somewhere to yield the GIL between them.
        """
        angle = 2.0 * math.pi * alpha
        x = _valmap(math.cos(angle), -1.0, 1.0, 0.0, self._radius)
        y = _valmap(math.sin(angle), -1.0, 1.0, 0.0, self._radius)
        return self._samplers[index].noise2(x, y)

    def vector(self, alpha: float) -> tuple[float, ...]:
        return tuple(self.sample(index, alpha) for index in range(self.dim))


@dataclass(frozen=True, eq=False)
class NoiseLoopTable:
    """One precomputed cycle of a `NoiseLoop`, looked up by linear interpolation.

    `values` holds `steps` samples of the full cycle, `values[i]` at
    `alpha = i / steps`. Interpolating and wrapping the last step back to the
    first keeps `vector(alpha)` periodic exactly, not just approximately:
    `alpha=1.0` maps back onto `values[0]`, the same sample `vector(0.0)`
    reads, bit for bit.
    """

    key: tuple[int, float, int]
    values: np.ndarray  # shape (steps, dim), float32

    def vector(self, alpha: float) -> tuple[float, ...]:
        steps = self.values.shape[0]
        position = (alpha % 1.0) * steps
        lower = int(position) % steps
        upper = (lower + 1) % steps
        frac = position - math.floor(position)
        row = self.values[lower] * (1.0 - frac) + self.values[upper] * frac
        return tuple(row.tolist())


def _build_table(key: tuple[int, float, int]) -> NoiseLoopTable:
    """Sample a full cycle, timesharing the GIL as it goes (see module docstring)."""
    seed, radius, dim = key
    steps = table_steps(radius)
    loop = NoiseLoop(seed, radius, dim)
    values = np.empty((steps, dim), dtype=np.float32)
    for step in range(steps):
        alpha = step / steps
        for start in range(0, dim, _YIELD_CHUNK_DIMS):
            end = min(start + _YIELD_CHUNK_DIMS, dim)
            for index in range(start, end):
                values[step, index] = loop.sample(index, alpha)
            time.sleep(_YIELD_SLEEP_SECONDS)
    return NoiseLoopTable(key=key, values=values)


class NoiseLoopTableBuilder:
    """Builds `NoiseLoopTable`s off the control thread, one at a time.

    Mirrors `ModelHost`'s loader thread (generator.py): `request_build` never
    blocks the caller, concurrent requests coalesce to the newest key rather
    than queuing every intermediate one (what makes a dragged radius slider
    safe), and a finished table is published with one `LatestValueStore.set`
    so a reader never observes a half built table. The worker thread starts
    lazily on the first request, so constructing a builder that never builds
    anything never spins up a thread.
    """

    def __init__(
        self,
        build: Callable[[tuple[int, float, int]], NoiseLoopTable] = _build_table,
    ) -> None:
        self._build = build
        self._lock = threading.Lock()
        self._pending_key: tuple[int, float, int] | None = None
        self.store: LatestValueStore[NoiseLoopTable | None] = LatestValueStore(None)
        self._wakeup = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def request_build(self, key: tuple[int, float, int]) -> None:
        with self._lock:
            self._pending_key = key
            if self._thread is None:
                self._running = True
                self._thread = threading.Thread(
                    target=self._run, name="noise-loop-table", daemon=True
                )
                self._thread.start()
        self._wakeup.set()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        self._running = False
        self._wakeup.set()
        thread.join(timeout=2.0)

    def _run(self) -> None:
        while self._running:
            self._wakeup.wait()
            self._wakeup.clear()
            if not self._running:
                return
            with self._lock:
                key = self._pending_key
            if key is None:
                continue
            try:
                table = self._build(key)
            except Exception:
                logger.exception("Noise loop table build failed for %r", key)
                with self._lock:
                    if self._pending_key == key:
                        self._pending_key = None
                continue
            with self._lock:
                won = self._pending_key == key
                if won:
                    self._pending_key = None
            if won:
                self.store.set(table)
            if self._pending_key is not None:
                self._wakeup.set()
