"""The simplex noise loop: a periodic latent vector driven by loop alpha.

Ported from `widgets/looping_widget.py`'s `OSN` (project authored, no NVIDIA
header, direct port allowed per design.md's clean-room rule). One
`OpenSimplex` sampler per output dimension, each walked around a circle of
`radius` diameter; sampling all dimensions at the same angle keeps every
component periodic in lockstep, so `vector(0.0) == vector(1.0)`.
"""

import math

from opensimplex import OpenSimplex


def _valmap(value: float, istart: float, istop: float, ostart: float, ostop: float) -> float:
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


class NoiseLoop:
    def __init__(self, seed: int, radius: float, dim: int) -> None:
        self._radius = radius
        self._samplers = [OpenSimplex(seed=seed + i) for i in range(dim)]

    def vector(self, alpha: float) -> tuple[float, ...]:
        angle = 2.0 * math.pi * alpha
        x = _valmap(math.cos(angle), -1.0, 1.0, 0.0, self._radius)
        y = _valmap(math.sin(angle), -1.0, 1.0, 0.0, self._radius)
        return tuple(sampler.noise2(x, y) for sampler in self._samplers)
