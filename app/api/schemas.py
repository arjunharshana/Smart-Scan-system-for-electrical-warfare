from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    scheduler: str = "V4.1"
    version: str = "4.1"
    device: str = "cpu"
    timestamp: float


class SimulationStatusResponse(BaseModel):
    state: str = Field(description="IDLE, RUNNING, or PAUSED")
    time_step: int
    simulation_time_s: float
    scenario_name: str
    scheduler_name: str
    scheduler_type: str
    speed: str
    seed: int
    version: str = "4.1"


class SimulationControlRequest(BaseModel):
    steps: int | None = Field(default=None, ge=1, description="Number of steps to advance, or None for continuous")


class SimulationResetRequest(BaseModel):
    seed: int | None = Field(default=None, description="Optional deterministic random seed")
    scenario_name: str | None = Field(default=None, description="Optional scenario filename to load")
    scheduler_name: str | None = Field(default=None, description="Optional scheduler algorithm name")


class SimulationSpeedRequest(BaseModel):
    speed: str = Field(default="1x", description="Speed multiplier: 0.5x, 1x, 2x, 5x, or max")


class ScenarioItem(BaseModel):
    id: str
    name: str
    filename: str
    description: str
    total_steps: int | None = None
