from __future__ import annotations

from rf_environment.scheduler.base import ScanScheduler
from rf_environment.scheduler.random_scheduler import RandomScheduler
from rf_environment.scheduler.sequential_scheduler import SequentialScheduler
from rf_environment.scheduler.thompson_sampling import ThompsonSamplingScheduler
from rf_environment.scheduler.ucb1 import UCB1Scheduler


def default_scan_bands(min_hz: float, max_hz: float, bandwidth_hz: float) -> list[float]:
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth must be positive")
    bands = []
    freq = min_hz + bandwidth_hz / 2.0
    while freq <= max_hz - bandwidth_hz / 2.0 + 1e-6:
        bands.append(freq)
        freq += bandwidth_hz
    return bands or [min_hz]


def create_scheduler(name: str, bands_hz: list[float], seed: int = 0) -> ScanScheduler:
    key = name.lower().replace("-", "_")
    if key == "random":
        return RandomScheduler(bands_hz, seed=seed)
    if key == "sequential":
        return SequentialScheduler(bands_hz)
    if key == "ucb1":
        return UCB1Scheduler(bands_hz)
    if key in {"thompson", "thompson_sampling"}:
        return ThompsonSamplingScheduler(bands_hz, seed=seed)
    raise ValueError(f"Unknown scheduler: {name}")
