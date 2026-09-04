from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler


class RLScheduler(BaseScheduler):
    """Architectural placeholder for future Reinforcement Learning scan schedulers.

    Consumes canonical SchedulerObservation and emits canonical ScanAction(frequency_bin).
    Plugs into the common BaseScheduler interface without requiring changes to
    the RF simulator, receiver, detector, or evaluator.
    """

    name = "rl"
    category = "rl"

    def __init__(
        self,
        bands_hz: list[float],
        allow_fallback_policy: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(bands_hz)
        self.allow_fallback_policy = allow_fallback_policy
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
        if not self.allow_fallback_policy:
            raise NotImplementedError(
                "RL policy not implemented yet. Set allow_fallback_policy=True for architecture integration testing."
            )
        idx = int(self.rng.integers(0, len(self.bands_hz)))
        self.last_selected_bin = idx
        self.last_selected = self.bands_hz[idx]
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"RL Placeholder Fallback: Random bin {idx}/{len(self.bands_hz)} selection",
            "rule": "rl_fallback",
        }
        return idx

    def select_action(
        self,
        observation: SchedulerObservation | Any = None,
    ) -> ScanAction:
        bin_idx = self.select_bin(observation)
        action = ScanAction(frequency_bin=bin_idx)
        self.last_action = action
        return action

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
    ) -> None:
        """Placeholder for future experience collection / replay buffer updates."""
        pass

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)
