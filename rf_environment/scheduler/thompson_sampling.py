from __future__ import annotations

import numpy as np

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class ThompsonSamplingScheduler(ScanScheduler):
    name = "thompson_sampling"
    category = "baseline"

    def __init__(self, bands_hz: list[float], seed: int | None = None) -> None:
        super().__init__(bands_hz)
        self.seed = seed
        n = len(self.bands_hz)
        self.alpha = [1.0] * n
        self.beta = [1.0] * n
        self.rng = np.random.default_rng(seed)
        self._last_arm: int | None = None

    def select_frequency(self, observation: Observation | None) -> float:
        samples = [float(self.rng.beta(a, b)) for a, b in zip(self.alpha, self.beta)]
        self._last_arm = max(range(len(samples)), key=lambda i: samples[i])
        self.last_selected = self.bands_hz[self._last_arm]
        chosen_i = self._last_arm
        a, b = self.alpha[chosen_i], self.beta[chosen_i]
        mean_p = a / (a + b)
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Sampled highest success probability ({samples[chosen_i]:.3f}) from Beta(α={a:.1f}, β={b:.1f}, mean={mean_p:.3f})",
            "rule": "posterior_sampling",
            "estimated_value": mean_p,
            "exploration_bonus": float(samples[chosen_i] - mean_p),
            "alpha": a,
            "beta": b,
        }
        return self.last_selected

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        if self._last_arm is None:
            return
        i = self._last_arm
        if reward > 0:
            self.alpha[i] += 1.0
        else:
            self.beta[i] += 1.0

    def reset(self) -> None:
        super().reset()
        n = len(self.bands_hz)
        self.alpha = [1.0] * n
        self.beta = [1.0] * n
        self.rng = np.random.default_rng(self.seed)
        self._last_arm = None

    def get_state(self) -> SchedulerState:
        totals = [a + b for a, b in zip(self.alpha, self.beta)]
        stats = [
            {
                "frequency_hz": f,
                "alpha": a,
                "beta": b,
                "count": int(a + b - 2.0),
                "value": a / t if t else 0.0,
                "probability": a / t if t else 0.0,
            }
            for f, a, b, t in zip(self.bands_hz, self.alpha, self.beta, totals)
        ]
        return SchedulerState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=stats,
            explanation=self.get_explanation(),
        )
