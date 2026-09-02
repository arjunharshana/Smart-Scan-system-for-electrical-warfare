from __future__ import annotations

import math

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class UCB1Scheduler(ScanScheduler):
    name = "ucb1"

    def __init__(self, bands_hz: list[float], exploration: float = 1.0) -> None:
        super().__init__(bands_hz)
        self.exploration = float(exploration)
        n = len(self.bands_hz)
        self.counts = [0] * n
        self.values = [0.0] * n
        self.t = 0
        self._last_arm: int | None = None

    def select_frequency(self, observation: Observation | None) -> float:
        self.t += 1
        for i, count in enumerate(self.counts):
            if count == 0:
                self._last_arm = i
                self.last_selected = self.bands_hz[i]
                return self.last_selected
        scores = []
        for i, (count, value) in enumerate(zip(self.counts, self.values)):
            bonus = self.exploration * math.sqrt(math.log(self.t) / count)
            scores.append(value + bonus)
        self._last_arm = max(range(len(scores)), key=lambda i: scores[i])
        self.last_selected = self.bands_hz[self._last_arm]
        return self.last_selected

    def update(self, observation: Observation, reward: float) -> None:
        if self._last_arm is None:
            return
        i = self._last_arm
        self.counts[i] += 1
        n = self.counts[i]
        self.values[i] += (reward - self.values[i]) / n

    def get_state(self) -> SchedulerState:
        stats = [
            {"frequency_hz": f, "count": c, "value": v, "probability": max(v, 0.0)}
            for f, c, v in zip(self.bands_hz, self.counts, self.values)
        ]
        return SchedulerState(name=self.name, selected_frequency_hz=self.last_selected, arm_stats=stats)
