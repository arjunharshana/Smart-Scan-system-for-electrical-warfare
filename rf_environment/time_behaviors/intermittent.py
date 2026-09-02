from __future__ import annotations

import numpy as np

from rf_environment.domain.enums import TimeBehaviorType
from rf_environment.time_behaviors.base import TimeBehavior


class Intermittent(TimeBehavior):
    behavior_type = TimeBehaviorType.INTERMITTENT

    def __init__(self, p_transmit: float = 0.3, seed: int | None = None) -> None:
        if not 0.0 <= p_transmit <= 1.0:
            raise ValueError("p_transmit must be in [0, 1]")
        self.p_transmit = float(p_transmit)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, bool] = {}

    def is_transmitting(self, time_step: int) -> bool:
        if time_step in self._cache:
            return self._cache[time_step]
        value = bool(self.rng.random() < self.p_transmit)
        self._cache[time_step] = value
        return value
