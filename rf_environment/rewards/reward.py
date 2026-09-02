from __future__ import annotations


class RewardCalculator:
    """Separate from bandit algorithms so cost functions can change later."""

    def __init__(
        self,
        hit: float = 1.0,
        miss: float = -0.1,
        false_alarm: float = -0.5,
        correct_rejection: float = 0.0,
        scan_cost: float = 0.01,
    ) -> None:
        self.hit = hit
        self.miss = miss
        self.false_alarm = false_alarm
        self.correct_rejection = correct_rejection
        self.scan_cost = scan_cost

    def compute(self, outcome: str) -> float:
        mapping = {
            "HIT": self.hit,
            "MISS": self.miss,
            "FALSE_ALARM": self.false_alarm,
            "CORRECT_REJECTION": self.correct_rejection,
        }
        return mapping.get(outcome, 0.0) - self.scan_cost
