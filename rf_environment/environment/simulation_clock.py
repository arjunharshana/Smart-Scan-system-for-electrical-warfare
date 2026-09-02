from __future__ import annotations


class SimulationClock:
    def __init__(self, time_step_ms: float = 10.0, total_time_steps: int | None = None) -> None:
        self.time_step_ms = float(time_step_ms)
        self.total_time_steps = total_time_steps
        self.time_step = -1

    def reset(self) -> None:
        self.time_step = -1

    def advance(self) -> int:
        self.time_step += 1
        return self.time_step

    @property
    def time_ms(self) -> float:
        return max(self.time_step, 0) * self.time_step_ms

    def finished(self) -> bool:
        if self.total_time_steps is None:
            return False
        return self.time_step + 1 >= self.total_time_steps
