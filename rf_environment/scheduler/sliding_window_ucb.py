from __future__ import annotations

from collections import deque
import math
from typing import Any

from rf_environment.domain.action import ScanAction
from rf_environment.domain.metrics import SchedulerTelemetryState
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler


class SlidingWindowUCBScheduler(BaseScheduler):
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

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
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
                return i

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
        self.window.append((action.frequency_bin, float(reward), self.t))

    def update(self, observation: Any, reward: float, action: float | int | None = None) -> None:
        if action is not None:
            if isinstance(action, int):
                arm = action
            else:
                arm = min(range(len(self.bands_hz)), key=lambda idx: abs(self.bands_hz[idx] - float(action)))
        elif self._last_arm is not None:
            arm = self._last_arm
        else:
            return
        self.window.append((arm, float(reward), self.t))

    def reset(self) -> None:
        super().reset()
        self.t = 0
        self.window.clear()
        self._last_arm = None

    def get_state(self) -> SchedulerTelemetryState:
        n = len(self.bands_hz)
        counts = [0] * n
        reward_sums = [0.0] * n
        for arm, r, _ in self.window:
            counts[arm] += 1
            reward_sums[arm] += r

        effective_t = min(max(self.t, 2), self.window_size)
        stats = []
        for i in range(n):
            mean = reward_sums[i] / counts[i] if counts[i] > 0 else 0.0
            bonus = (
                self.exploration * math.sqrt((2.0 * math.log(effective_t)) / counts[i])
                if counts[i] > 0
                else float("inf")
            )
            stats.append({
                "frequency_hz": self.bands_hz[i],
                "count": counts[i],
                "value": mean,
                "probability": max(mean, 0.0),
                "bonus": bonus if counts[i] > 0 else 0.0,
                "upper_bound": mean + bonus if counts[i] > 0 else 0.0,
            })
        return SchedulerTelemetryState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=stats,
            explanation=self.get_explanation(),
        )
