from __future__ import annotations

import numpy as np

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class ThompsonSamplingScheduler(ScanScheduler):
    name = "thompson_sampling"

    def __init__(self, bands_hz: list[float], seed: int | None = None) -> None:
        super().__init__(bands_hz)
        n = len(self.bands_hz)
        self.alpha = [1.0] * n
        self.beta = [1.0] * n
        self.rng = np.random.default_rng(seed)
        self._last_arm: int | None = None

    def select_frequency(self, observation: Observation | None) -> float:
        samples = [float(self.rng.beta(a, b)) for a, b in zip(self.alpha, self.beta)]
        self._last_arm = max(range(len(samples)), key=lambda i: samples[i])
        self.last_selected = self.bands_hz[self._last_arm]
        return self.last_selected

    def update(self, observation: Observation, reward: float) -> None:
        if self._last_arm is None:
            return
        i = self._last_arm
        if reward > 0:
            self.alpha[i] += 1.0
        else:
            self.beta[i] += 1.0

    def get_state(self) -> SchedulerState:
        totals = [a + b for a, b in zip(self.alpha, self.beta)]
        stats = [
            {
                "frequency_hz": f,
                "alpha": a,
                "beta": b,
                "probability": a / t if t else 0.0,
            }
            for f, a, b, t in zip(self.bands_hz, self.alpha, self.beta, totals)
        ]
        return SchedulerState(name=self.name, selected_frequency_hz=self.last_selected, arm_stats=stats)
