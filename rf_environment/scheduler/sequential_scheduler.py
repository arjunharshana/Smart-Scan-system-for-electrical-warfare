from __future__ import annotations

from typing import Any
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler


class SequentialScheduler(BaseScheduler):
    name = "sequential"
    category = "baseline"

    def __init__(self, bands_hz: list[float]) -> None:
        super().__init__(bands_hz)
        self._index = -1

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
        self._index = (self._index + 1) % len(self.bands_hz)
        self.last_selected = self.bands_hz[self._index]
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Systematic round-robin scan: Band index {self._index + 1}/{len(self.bands_hz)}",
            "rule": "deterministic_cycle",
            "estimated_value": 0.0,
            "exploration_bonus": 0.0,
        }
        return self._index

    def select_frequency(self, observation: Any = None) -> float:
        idx = self.select_bin(observation)
        return self.bands_hz[idx]

    def reset(self) -> None:
        super().reset()
        self._index = -1
