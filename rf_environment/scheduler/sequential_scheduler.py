from __future__ import annotations

from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class SequentialScheduler(ScanScheduler):
    name = "sequential"
    category = "baseline"

    def __init__(self, bands_hz: list[float]) -> None:
        super().__init__(bands_hz)
        self._index = -1

    def select_frequency(self, observation: Observation | None) -> float:
        self._index = (self._index + 1) % len(self.bands_hz)
        self.last_selected = self.bands_hz[self._index]
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Systematic round-robin scan: Band index {self._index + 1}/{len(self.bands_hz)}",
            "rule": "deterministic_cycle",
            "estimated_value": 0.0,
            "exploration_bonus": 0.0,
        }
        return self.last_selected

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        return None

    def reset(self) -> None:
        super().reset()
        self._index = -1
