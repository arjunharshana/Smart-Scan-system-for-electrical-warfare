from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler


class RandomScheduler(BaseScheduler):
    name = "random"
    category = "baseline"

    def __init__(self, bands_hz: list[float], seed: int | None = None) -> None:
        super().__init__(bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
        idx = int(self.rng.integers(0, len(self.bands_hz)))
        self.last_selected = self.bands_hz[idx]
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Uniform random selection across {len(self.bands_hz)} available channels (prob={1/len(self.bands_hz):.3f})",
            "rule": "uniform_random",
            "estimated_value": 0.0,
            "exploration_bonus": 0.0,
        }
        return idx

    def select_frequency(self, observation: Any = None) -> float:
        idx = self.select_bin(observation)
        return self.bands_hz[idx]

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)
