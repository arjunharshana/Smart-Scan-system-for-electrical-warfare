from __future__ import annotations

import math

import numpy as np

from rf_environment.domain.signal import IdealSignal


class AdditiveNoise:
    def __init__(self, noise_floor_dbm: float = -100.0, seed: int | None = None) -> None:
        self.noise_floor_dbm = float(noise_floor_dbm)
        self.rng = np.random.default_rng(seed)

    def sample_dbm(self) -> float:
        # Small AWGN fluctuation around the configured floor.
        return float(self.noise_floor_dbm + self.rng.normal(0.0, 0.5))


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


def mw_to_dbm(mw: float) -> float:
    if mw <= 0:
        return -200.0
    return 10.0 * math.log10(mw)
