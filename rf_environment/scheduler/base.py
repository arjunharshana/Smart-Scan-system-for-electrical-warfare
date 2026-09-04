from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rf_environment.domain.action import ScanAction
from rf_environment.domain.metrics import SchedulerTelemetryState
from rf_environment.domain.state import SchedulerObservation


class BaseScheduler(ABC):
    """Canonical interface for all scan schedulers.

    Separation of concerns:
    - Consumes canonical SchedulerObservation (zero ground truth).
    - Emits canonical ScanAction(frequency_bin).
    - Receives transitions via observe(obs, action, reward, next_obs, done).
    """

    name: str = "base"
    category: str = "baseline"

    def __init__(self, bands_hz: list[float]) -> None:
        if not bands_hz:
            raise ValueError("Scheduler requires at least one scan band")
        self.bands_hz = [float(b) for b in bands_hz]
        self.last_selected_bin: int = 0
        self.last_selected: float | None = None
        self.last_action: ScanAction | None = None
        self.last_explanation: dict[str, Any] = {}
        self.train_mode: bool = True

    def train(self) -> None:
        self.train_mode = True

    def eval(self) -> None:
        self.train_mode = False

    def select_action(
        self,
        observation: SchedulerObservation | Any = None,
    ) -> ScanAction | float:
        """Canonical action selection returning ScanAction(frequency_bin)."""
        if isinstance(observation, SchedulerObservation):
            bin_idx = self.select_bin(observation)
            action = ScanAction(frequency_bin=bin_idx)
            self.last_selected_bin = bin_idx
            self.last_selected = self.bands_hz[bin_idx]
            self.last_action = action
            return action
        # Legacy fallback if called with None or legacy Observation
        bin_idx = self.select_bin_legacy(observation)
        self.last_selected_bin = bin_idx
        self.last_selected = self.bands_hz[bin_idx]
        self.last_action = ScanAction(frequency_bin=bin_idx)
        return self.last_selected

    @abstractmethod
    def select_bin(self, observation: SchedulerObservation) -> int:
        """Core bin selection logic implemented by subclasses."""
        raise NotImplementedError

    def select_bin_legacy(self, observation: Any = None) -> int:
        dummy = SchedulerObservation(
            timestamp=0.0,
            current_frequency_bin=self.last_selected_bin,
            last_detection=False,
            last_detection_bin=None,
            last_detection_strength=None,
            recent_detection_history=(),
            recent_frequency_history=(),
            scan_count_by_bin=tuple([0] * len(self.bands_hz)),
            time_since_scan_by_bin=tuple([0.0] * len(self.bands_hz)),
            time_since_last_detection=0.0,
        )
        return self.select_bin(dummy)

    def select_frequency(self, observation: Any = None) -> float:
        if isinstance(observation, SchedulerObservation):
            bin_idx = self.select_bin(observation)
        else:
            bin_idx = self.select_bin_legacy(observation)
        self.last_selected_bin = bin_idx
        self.last_selected = self.bands_hz[bin_idx]
        return self.last_selected

    def observe(
        self,
        observation: SchedulerObservation | Any,
        action: ScanAction | None = None,
        reward: float = 0.0,
        next_observation: SchedulerObservation | None = None,
        done: bool = False,
    ) -> None:
        if not self.train_mode:
            return
        if hasattr(observation, "observation") and hasattr(observation, "action") and hasattr(observation, "reward"):
            trans = observation
            self.update_policy(trans.observation, trans.action, trans.reward, trans.next_observation, trans.done)
            return
        self.update_policy(observation, action, reward, next_observation, done)

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
    ) -> None:
        pass

    def update(
        self,
        observation: Any,
        reward: float,
        action: float | int | None = None,
    ) -> None:
        """Legacy update hook for backward compatibility."""
        pass

    def reset(self) -> None:
        self.last_selected = None
        self.last_selected_bin = 0
        self.last_action = None
        self.last_explanation = {}

    def get_explanation(self) -> dict[str, Any]:
        return dict(self.last_explanation)

    def get_state(self) -> SchedulerTelemetryState:
        return SchedulerTelemetryState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=[],
            explanation=self.get_explanation(),
        )


ScanScheduler = BaseScheduler
