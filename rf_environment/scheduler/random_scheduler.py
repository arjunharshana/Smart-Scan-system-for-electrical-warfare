from __future__ import annotations

import numpy as np

from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class RandomScheduler(ScanScheduler):
    name = "random"
    category = "baseline"

    def __init__(self, bands_hz: list[float], seed: int | None = None) -> None:
        super().__init__(bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_frequency(self, observation: Observation | None) -> float:
        idx = int(self.rng.integers(0, len(self.bands_hz)))
        self.last_selected = self.bands_hz[idx]
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Uniform random selection across {len(self.bands_hz)} available channels (prob={1/len(self.bands_hz):.3f})",
            "rule": "uniform_random",
            "estimated_value": 0.0,
            "exploration_bonus": 0.0,
        }
        return self.last_selected

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        return None

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)
