from __future__ import annotations

import numpy as np

from rf_environment.domain.enums import FrequencyBehaviorType
from rf_environment.frequency_behaviors.base import FrequencyBehavior


class RandomFrequency(FrequencyBehavior):
    behavior_type = FrequencyBehaviorType.RANDOM

    def __init__(
        self,
        frequencies_hz: list[float] | None = None,
        min_frequency_hz: float | None = None,
        max_frequency_hz: float | None = None,
        quantization_hz: float = 1_000_000.0,
        seed: int | None = None,
        dwell_steps: int = 1,
    ) -> None:
        self.frequencies_hz = [float(f) for f in frequencies_hz] if frequencies_hz else None
        self.min_frequency_hz = min_frequency_hz
        self.max_frequency_hz = max_frequency_hz
        self.quantization_hz = float(quantization_hz)
        self.dwell_steps = max(1, int(dwell_steps))
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, float] = {}

    def reseed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._cache.clear()

    def reset(self) -> None:
        self._cache.clear()
        if not self.frequencies_hz and (min_frequency_hz is None or max_frequency_hz is None):
            raise ValueError("RandomFrequency needs a frequency set or a min/max range")

    def get_frequency(self, time_step: int, state: dict | None = None) -> float:
        slot = time_step // self.dwell_steps
        if slot in self._cache:
            return self._cache[slot]
        if self.frequencies_hz:
            value = float(self.rng.choice(self.frequencies_hz))
        else:
            lo = float(self.min_frequency_hz)
            hi = float(self.max_frequency_hz)
            raw = float(self.rng.uniform(lo, hi))
            q = self.quantization_hz
            value = round(raw / q) * q
            value = min(max(value, lo), hi)
        self._cache[slot] = value
        return value
