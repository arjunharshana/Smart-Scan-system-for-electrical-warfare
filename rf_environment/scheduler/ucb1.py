from __future__ import annotations

import math

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class UCB1Scheduler(ScanScheduler):
    name = "ucb1"
    category = "baseline"

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
                self.last_explanation = {
                    "action_mhz": self.last_selected / 1e6,
                    "reason": f"Initial arm exploration: Band {i+1}/{len(self.bands_hz)} never probed",
                    "rule": "initial_exploration",
                    "estimated_value": 0.0,
                    "exploration_bonus": float("inf"),
                    "arm_pulls": 0,
                }
                return self.last_selected

        scores = []
        bonuses = []
        for i, (count, value) in enumerate(zip(self.counts, self.values)):
            bonus = self.exploration * math.sqrt(math.log(self.t) / count)
            bonuses.append(bonus)
            scores.append(value + bonus)

        self._last_arm = max(range(len(scores)), key=lambda i: scores[i])
        self.last_selected = self.bands_hz[self._last_arm]
        chosen_i = self._last_arm
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Highest UCB score ({scores[chosen_i]:.3f}) = Mean Reward ({self.values[chosen_i]:.3f}) + Bonus ({bonuses[chosen_i]:.3f})",
            "rule": "ucb_max",
            "estimated_value": self.values[chosen_i],
            "exploration_bonus": bonuses[chosen_i],
            "arm_pulls": self.counts[chosen_i],
        }
        return self.last_selected

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        if self._last_arm is None:
            return
        i = self._last_arm
        self.counts[i] += 1
        n = self.counts[i]
        self.values[i] += (reward - self.values[i]) / n

    def reset(self) -> None:
        super().reset()
        n = len(self.bands_hz)
        self.counts = [0] * n
        self.values = [0.0] * n
        self.t = 0
        self._last_arm = None

    def get_state(self) -> SchedulerState:
        stats = []
        for i, (f, c, v) in enumerate(zip(self.bands_hz, self.counts, self.values)):
            bonus = (
                self.exploration * math.sqrt(math.log(max(self.t, 1)) / c)
                if c > 0
                else float("inf")
            )
            stats.append({
                "frequency_hz": f,
                "count": c,
                "value": v,
                "probability": max(v, 0.0),
                "bonus": bonus if c > 0 else 0.0,
                "upper_bound": v + bonus if c > 0 else 0.0,
            })
        return SchedulerState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=stats,
            explanation=self.get_explanation(),
        )
