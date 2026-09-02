from __future__ import annotations

from enum import Enum

import numpy as np

from rf_environment.domain.enums import FrequencyBehaviorType
from rf_environment.frequency_behaviors.base import FrequencyBehavior


class HopMode(str, Enum):
    SEQUENTIAL = "sequential"
    PSEUDO_RANDOM = "pseudo_random"
    RANDOM = "random"


class FrequencyHopping(FrequencyBehavior):
    behavior_type = FrequencyBehaviorType.HOPPING

    def __init__(
        self,
        frequencies_hz: list[float],
        mode: str = HopMode.SEQUENTIAL.value,
        dwell_steps: int = 1,
        seed: int | None = None,
        sequence: list[int] | None = None,
    ) -> None:
        if not frequencies_hz:
            raise ValueError("FrequencyHopping requires at least one frequency")
        self.frequencies_hz = [float(f) for f in frequencies_hz]
        self.mode = HopMode(mode)
        self.dwell_steps = max(1, int(dwell_steps))
        self.rng = np.random.default_rng(seed)
        if sequence is not None:
            self.sequence = [int(i) % len(self.frequencies_hz) for i in sequence]
        elif self.mode == HopMode.PSEUDO_RANDOM:
            order = np.arange(len(self.frequencies_hz))
            self.rng.shuffle(order)
            self.sequence = order.tolist()
        else:
            self.sequence = list(range(len(self.frequencies_hz)))
        self._cache: dict[int, float] = {}

    def get_frequency(self, time_step: int, state: dict | None = None) -> float:
        hop_index = time_step // self.dwell_steps
        if hop_index in self._cache:
            return self._cache[hop_index]
        if self.mode == HopMode.RANDOM:
            idx = int(self.rng.integers(0, len(self.frequencies_hz)))
        else:
            idx = self.sequence[hop_index % len(self.sequence)]
        value = self.frequencies_hz[idx]
        self._cache[hop_index] = value
        return value
