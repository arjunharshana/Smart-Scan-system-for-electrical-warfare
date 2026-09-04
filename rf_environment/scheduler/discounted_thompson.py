from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.metrics import SchedulerTelemetryState
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler


class DiscountedThompsonSamplingScheduler(BaseScheduler):
    """Discounted Thompson Sampling (D-TS) algorithm for non-stationary environments."""

    name = "discounted_thompson"
    category = "non_stationary"

    def __init__(
        self,
        bands_hz: list[float],
        gamma: float = 0.95,
        seed: int | None = None,
    ) -> None:
        super().__init__(bands_hz)
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self.seed = seed
        n = len(self.bands_hz)
        self.alpha = [1.0] * n
        self.beta = [1.0] * n
        self.rng = np.random.default_rng(seed)
        self._last_arm: int | None = None

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
        samples = [float(self.rng.beta(a, b)) for a, b in zip(self.alpha, self.beta)]
        self._last_arm = max(range(len(samples)), key=lambda i: samples[i])
        self.last_selected = self.bands_hz[self._last_arm]

        chosen_i = self._last_arm
        a, b = self.alpha[chosen_i], self.beta[chosen_i]
        mean_p = a / (a + b)
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Sampled highest discounted probability ({samples[chosen_i]:.3f}) from Beta(α={a:.2f}, β={b:.2f}, γ={self.gamma})",
            "rule": "discounted_posterior_sampling",
            "estimated_value": mean_p,
            "exploration_bonus": float(samples[chosen_i] - mean_p),
            "alpha": a,
            "beta": b,
            "gamma": self.gamma,
        }
        return self._last_arm

    def select_frequency(self, observation: Any = None) -> float:
        arm = self.select_bin(observation)
        return self.bands_hz[arm]

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
    ) -> None:
        n = len(self.bands_hz)
        for i in range(n):
            self.alpha[i] = 1.0 + self.gamma * (self.alpha[i] - 1.0)
            self.beta[i] = 1.0 + self.gamma * (self.beta[i] - 1.0)

        chosen_i = action.frequency_bin
        if 0 <= chosen_i < n:
            if next_observation.last_detection or reward > 0:
                self.alpha[chosen_i] += 1.0
            else:
                self.beta[chosen_i] += 1.0

    def update(self, observation: Any, reward: float, action: float | int | None = None) -> None:
        if action is not None:
            if isinstance(action, int):
                chosen_i = action
            else:
                chosen_i = min(range(len(self.bands_hz)), key=lambda idx: abs(self.bands_hz[idx] - float(action)))
        elif self._last_arm is not None:
            chosen_i = self._last_arm
        else:
            return

        n = len(self.bands_hz)
        for i in range(n):
            self.alpha[i] = 1.0 + self.gamma * (self.alpha[i] - 1.0)
            self.beta[i] = 1.0 + self.gamma * (self.beta[i] - 1.0)

        if 0 <= chosen_i < n:
            if reward > 0:
                self.alpha[chosen_i] += 1.0
            else:
                self.beta[chosen_i] += 1.0

    def reset(self) -> None:
        super().reset()
        n = len(self.bands_hz)
        self.alpha = [1.0] * n
        self.beta = [1.0] * n
        self.rng = np.random.default_rng(self.seed)
        self._last_arm = None

    def get_state(self) -> SchedulerTelemetryState:
        totals = [a + b for a, b in zip(self.alpha, self.beta)]
        stats = [
            {
                "frequency_hz": f,
                "alpha": a,
                "beta": b,
                "effective_count": (a + b - 2.0),
                "value": a / t if t else 0.0,
                "probability": a / t if t else 0.0,
                "gamma": self.gamma,
            }
            for f, a, b, t in zip(self.bands_hz, self.alpha, self.beta, totals)
        ]
        return SchedulerTelemetryState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=stats,
            explanation=self.get_explanation(),
        )
