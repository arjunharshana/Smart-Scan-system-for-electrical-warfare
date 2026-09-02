from __future__ import annotations

from abc import ABC, abstractmethod

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation


class ScanScheduler(ABC):
    """Selects the next receiver center frequency from observations only."""

    name: str

    def __init__(self, bands_hz: list[float]) -> None:
        if not bands_hz:
            raise ValueError("Scheduler requires at least one scan band")
        self.bands_hz = [float(b) for b in bands_hz]
        self.last_selected: float | None = None

    @abstractmethod
    def select_frequency(self, observation: Observation | None) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, observation: Observation, reward: float) -> None:
        raise NotImplementedError

    def get_state(self) -> SchedulerState:
        return SchedulerState(name=self.name, selected_frequency_hz=self.last_selected, arm_stats=[])
