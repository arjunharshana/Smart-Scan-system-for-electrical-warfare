from __future__ import annotations

from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class SequentialScheduler(ScanScheduler):
    name = "sequential"

    def __init__(self, bands_hz: list[float]) -> None:
        super().__init__(bands_hz)
        self._index = -1

    def select_frequency(self, observation: Observation | None) -> float:
        self._index = (self._index + 1) % len(self.bands_hz)
        self.last_selected = self.bands_hz[self._index]
        return self.last_selected

    def update(self, observation: Observation, reward: float) -> None:
        return None
