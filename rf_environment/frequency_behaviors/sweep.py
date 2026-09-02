from __future__ import annotations

from rf_environment.domain.enums import FrequencyBehaviorType
from rf_environment.frequency_behaviors.base import FrequencyBehavior


class FrequencySweep(FrequencyBehavior):
    behavior_type = FrequencyBehaviorType.SWEEP

    def __init__(
        self,
        start_hz: float,
        end_hz: float,
        step_hz: float,
        direction: str = "up",
        wrap_around: bool = True,
        dwell_steps: int = 1,
    ) -> None:
        if step_hz <= 0:
            raise ValueError("step_hz must be positive")
        self.start_hz = float(start_hz)
        self.end_hz = float(end_hz)
        self.step_hz = float(step_hz)
        self.direction = direction
        self.wrap_around = wrap_around
        self.dwell_steps = max(1, int(dwell_steps))
        span = abs(self.end_hz - self.start_hz)
        n = int(round(span / self.step_hz)) + 1
        lo, hi = (self.start_hz, self.end_hz) if self.start_hz <= self.end_hz else (self.end_hz, self.start_hz)
        self.grid = [lo + i * self.step_hz for i in range(n)]
        if self.grid and abs(self.grid[-1] - hi) > 1e-6:
            self.grid.append(hi)
        if direction == "down":
            self.grid = list(reversed(self.grid))
        elif direction == "bidirectional":
            up = list(self.grid)
            down = list(reversed(self.grid[1:-1])) if len(self.grid) > 2 else []
            self.grid = up + down

    def get_frequency(self, time_step: int, state: dict | None = None) -> float:
        idx = time_step // self.dwell_steps
        if not self.grid:
            return self.start_hz
        if self.wrap_around:
            return self.grid[idx % len(self.grid)]
        return self.grid[min(idx, len(self.grid) - 1)]
