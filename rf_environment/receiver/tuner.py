from __future__ import annotations

from rf_environment.domain.receiver import ReceiverState
from rf_environment.domain.signal import IdealSignal


def bands_overlap(f1: float, bw1: float, f2: float, bw2: float) -> bool:
    a0, a1 = f1 - bw1 / 2.0, f1 + bw1 / 2.0
    b0, b1 = f2 - bw2 / 2.0, f2 + bw2 / 2.0
    return a0 < b1 and b0 < a1


class Tuner:
    def __init__(self, time_step_ms: float, tuning_time_ms: float = 0.0) -> None:
        self.time_step_ms = max(time_step_ms, 1e-9)
        self.tuning_time_ms = float(tuning_time_ms)
        self.remaining_steps = 0
        self.center_frequency_hz: float | None = None

    def request_tune(self, frequency_hz: float) -> None:
        if self.center_frequency_hz is None or abs(frequency_hz - self.center_frequency_hz) > 1e-6:
            steps = int((self.tuning_time_ms + self.time_step_ms - 1e-9) // self.time_step_ms)
            self.remaining_steps = max(0, steps)
        self.center_frequency_hz = frequency_hz

    def tick(self) -> bool:
        """Return True if still tuning this step."""
        if self.remaining_steps > 0:
            self.remaining_steps -= 1
            return True
        return False
