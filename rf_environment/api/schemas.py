from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimulationControl(BaseModel):
    steps: int | None = None


class EmitterCreate(BaseModel):
    id: str
    type: str
    frequency_behavior: dict[str, Any]
    time_behavior: dict[str, Any]
    power_dbm: float = -20
    bandwidth_hz: float = 1_000_000
    modulation: str = "UNKNOWN"
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class SchedulerSelect(BaseModel):
    type: str
