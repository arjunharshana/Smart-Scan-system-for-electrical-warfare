from __future__ import annotations

import numpy as np

from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class RandomScheduler(ScanScheduler):
    name = "random"

    def __init__(self, bands_hz: list[float], seed: int | None = None) -> None:
        super().__init__(bands_hz)
        self.rng = np.random.default_rng(seed)

    def select_frequency(self, observation: Observation | None) -> float:
        idx = int(self.rng.integers(0, len(self.bands_hz)))
        self.last_selected = self.bands_hz[idx]
        return self.last_selected

    def update(self, observation: Observation, reward: float) -> None:
        return None
