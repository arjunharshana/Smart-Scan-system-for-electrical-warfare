from __future__ import annotations

from abc import ABC, abstractmethod

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation


class ScanScheduler(ABC):
    """Selects the next receiver center frequency from observations and temporal context only."""

    name: str = "base"
    category: str = "baseline"  # baseline, non_stationary, contextual

    def __init__(self, bands_hz: list[float]) -> None:
        if not bands_hz:
            raise ValueError("Scheduler requires at least one scan band")
        self.bands_hz = [float(b) for b in bands_hz]
        self.last_selected: float | None = None
        self.last_explanation: dict = {}

    def select_action(self, context: Observation | None) -> float:
        """Selects the next center frequency (Hz) using observation context."""
        return self.select_frequency(context)

    @abstractmethod
    def select_frequency(self, observation: Observation | None) -> float:
        """Core frequency selection method. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        """Updates internal statistics from scan feedback."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets all internal scheduler learning state."""
        self.last_selected = None
        self.last_explanation = {}

    def get_explanation(self) -> dict:
        """Returns authentic rationale for the most recent frequency choice."""
        return dict(self.last_explanation)

    def get_state(self) -> SchedulerState:
        return SchedulerState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=[],
            explanation=self.get_explanation(),
        )
