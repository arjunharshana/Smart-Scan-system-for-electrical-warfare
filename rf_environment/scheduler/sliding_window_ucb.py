from __future__ import annotations

from collections import deque
import math
from typing import Any

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class SlidingWindowUCBScheduler(ScanScheduler):
    """Sliding-Window Upper Confidence Bound (SW-UCB) algorithm.

    Maintains a FIFO window of the most recent N observations.
    Adapts quickly when the RF environment reward distribution changes
    because stale observations naturally roll out of the window.
    """

    name = "sw_ucb"
    category = "non_stationary"

    def __init__(
        self,
        bands_hz: list[float],
        window_size: int = 50,
        exploration: float = 1.0,
    ) -> None:
        super().__init__(bands_hz)
        self.window_size = int(window_size)
        self.exploration = float(exploration)
        self.t = 0

        # FIFO window of (arm_index, reward, timestamp)
        self.window: deque[tuple[int, float, int]] = deque(maxlen=self.window_size)
        self._last_arm: int | None = None

    def select_frequency(self, observation: Observation | None) -> float:
        self.t += 1
        n = len(self.bands_hz)

        # Count pulls and sum rewards within current window
        counts = [0] * n
        reward_sums = [0.0] * n
        for arm, r, _ in self.window:
            counts[arm] += 1
            reward_sums[arm] += r

        # Check for unprobed arms in current window
        for i, count in enumerate(counts):
            if count == 0:
                self._last_arm = i
                self.last_selected = self.bands_hz[i]
                self.last_explanation = {
                    "action_mhz": self.last_selected / 1e6,
                    "reason": f"Sliding window exploration: Band {i+1}/{n} has 0 scans in last {self.window_size} steps",
                    "rule": "window_exploration",
                    "estimated_value": 0.0,
                    "exploration_bonus": float("inf"),
                    "window_pulls": 0,
                }
                return self.last_selected

        effective_t = min(self.t, self.window_size)
        scores = []
        bonuses = []
        means = []
        for i in range(n):
            mean = reward_sums[i] / counts[i]
            bonus = self.exploration * math.sqrt((2.0 * math.log(max(effective_t, 2))) / counts[i])
            means.append(mean)
            bonuses.append(bonus)
            scores.append(mean + bonus)

        self._last_arm = max(range(n), key=lambda i: scores[i])
        self.last_selected = self.bands_hz[self._last_arm]
        chosen = self._last_arm
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": f"Highest SW-UCB score ({scores[chosen]:.3f}) = Window Mean ({means[chosen]:.3f}) + Bonus ({bonuses[chosen]:.3f})",
            "rule": "sw_ucb_max",
            "estimated_value": means[chosen],
            "exploration_bonus": bonuses[chosen],
            "window_pulls": counts[chosen],
        }
        return self.last_selected

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        if self._last_arm is None:
            return
        self.window.append((self._last_arm, float(reward), self.t))

    def reset(self) -> None:
        super().reset()
        self.t = 0
        self.window.clear()
        self._last_arm = None

    def get_state(self) -> SchedulerState:
        n = len(self.bands_hz)
        counts = [0] * n
        reward_sums = [0.0] * n
        for arm, r, _ in self.window:
            counts[arm] += 1
            reward_sums[arm] += r

        effective_t = min(max(self.t, 2), self.window_size)
        stats = []
        for i, f in enumerate(self.bands_hz):
            c = counts[i]
            mean = reward_sums[i] / c if c > 0 else 0.0
            bonus = (
                self.exploration * math.sqrt((2.0 * math.log(effective_t)) / c)
                if c > 0
                else float("inf")
            )
            stats.append({
                "frequency_hz": f,
                "count": c,
                "value": mean,
                "probability": max(mean, 0.0),
                "bonus": bonus if c > 0 else 0.0,
                "upper_bound": mean + bonus if c > 0 else 0.0,
                "window_size": self.window_size,
            })

        return SchedulerState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=stats,
            explanation=self.get_explanation(),
        )
