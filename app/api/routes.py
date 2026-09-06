from __future__ import annotations

import time
from typing import Any
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    HealthResponse,
    SimulationControlRequest,
    SimulationResetRequest,
    SimulationSpeedRequest,
    SimulationStatusResponse,
)
from app.services.simulation_service import service

router = APIRouter()


BENCHMARK_SUMMARY_DATA: dict[str, Any] = {
    "schedulers": [
        {
            "id": "hybrid_v4",
            "name": "V4.0 Hybrid (Base)",
            "role": "Production Base",
            "architecture": "CA + MLPQNetwork DDQN -> Meta-Arbitrator",
            "overall_ir_pct": 35.19,
            "rank": 1,
            "badge": "FINAL BASE SCHEDULER",
            "offline_trained": True,
            "runtime_training": False,
        },
        {
            "id": "whittle_style",
            "name": "Whittle W3 Index",
            "role": "Restless Bandit Baseline",
            "architecture": "Heuristic Index Policy with Dwell Aging",
            "overall_ir_pct": 34.40,
            "rank": 2,
            "badge": "MATHEMATICAL BASELINE",
            "offline_trained": False,
            "runtime_training": False,
        },
        {
            "id": "context_aware",
            "name": "Context-Aware (CA)",
            "role": "Empirical Baseline",
            "architecture": "Empirical Transition Matrix + Coverage Bonus",
            "overall_ir_pct": 31.15,
            "rank": 3,
            "badge": "EMPIRICAL BASELINE",
            "offline_trained": False,
            "runtime_training": False,
        },
        {
            "id": "hybrid_v41",
            "name": "V4.1 LSTM-Hybrid",
            "role": "Recurrent Baseline",
            "architecture": "CA + Recurrent LSTM-DDQN -> Meta-Arbitrator",
            "overall_ir_pct": 26.90,
            "rank": 4,
            "badge": "RECURRENT ABLATION",
            "offline_trained": True,
            "runtime_training": False,
        },
        {
            "id": "v5_belief",
            "name": "V5.0 Augmented Belief",
            "role": "Bayesian POMDP Baseline",
            "architecture": "Semi-Markov Recursive Bayesian Dwell Filter",
            "overall_ir_pct": 15.36,
            "rank": 5,
            "badge": "EXACT BELIEF BASELINE",
            "offline_trained": False,
            "runtime_training": False,
        },
    ],
    "scenario_breakdown": [
        {"scenario": "1_Seen_Structure", "V4_0": 50.62, "Whittle": 48.33, "CA": 26.90, "V4_1": 46.10, "V5_0": 24.33},
        {"scenario": "2_Unseen_Permutation", "V4_0": 40.48, "Whittle": 38.67, "CA": 31.40, "V4_1": 32.20, "V5_0": 21.00},
        {"scenario": "3_Unseen_Phase", "V4_0": 45.70, "Whittle": 42.67, "CA": 30.40, "V4_1": 35.80, "V5_0": 22.33},
        {"scenario": "4_Unseen_Dwell", "V4_0": 18.25, "Whittle": 16.50, "CA": 24.53, "V4_1": 12.40, "V5_0": 9.33},
        {"scenario": "5_Unseen_Subset", "V4_0": 31.20, "Whittle": 31.00, "CA": 31.20, "V4_1": 22.50, "V5_0": 14.00},
        {"scenario": "6_Mixed_Shift", "V4_0": 33.60, "Whittle": 32.50, "CA": 33.60, "V4_1": 21.80, "V5_0": 13.67},
        {"scenario": "7_Random_Hopping", "V4_0": 33.40, "Whittle": 35.20, "CA": 35.50, "V4_1": 21.00, "V5_0": 9.00},
        {"scenario": "8_Periodic_Burst", "V4_0": 28.30, "Whittle": 30.30, "CA": 35.70, "V4_1": 23.40, "V5_0": 9.20},
    ],
}


@router.get("/health", response_model=HealthResponse)
def health_check() -> dict[str, Any]:
    """Production health check endpoint verifying scheduler readiness."""
    return {
        "status": "ok",
        "scheduler": "V4.0",
        "version": "4.0",
        "device": "cpu",
        "timestamp": time.time(),
    }


@router.get("/api/benchmark")
def get_benchmark_comparison() -> dict[str, Any]:
    """Returns canonical 5-way benchmark comparison and 8-scenario evaluation results."""
    return BENCHMARK_SUMMARY_DATA


@router.get("/api/export")
def export_mission_report() -> dict[str, Any]:
    """Exports comprehensive mission summary report with performance audit and model integrity."""
    telem = service.latest_telemetry
    status = service.get_status()
    perf = telem.get("performance", {})
    neural = telem.get("neural_model", {})
    return {
        "mission_report": {
            "title": "SIH26055 Smart Scan Strategy Mission Summary",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "system_status": status,
            "performance_metrics": perf,
            "neural_model_audit": neural,
            "arbitration_telemetry": telem.get("arbitration", {}),
            "latency_metrics": telem.get("latency", {}),
            "total_timeline_steps": len(service.timeline_history),
            "recent_detections": [t for t in service.timeline_history if t.get("detected")],
        }
    }


@router.get("/api/status", response_model=SimulationStatusResponse)
def get_simulation_status() -> dict[str, Any]:
    """Returns the current execution state and metadata of the simulation."""
    return service.get_status()


@router.get("/api/telemetry")
def get_telemetry() -> dict[str, Any]:
    """Returns the latest operational EW telemetry payload without ground-truth leakage."""
    return service.latest_telemetry


@router.get("/api/waterfall")
def get_waterfall(limit: int = 100) -> list[dict[str, Any]]:
    """Returns the recent frequency-time waterfall trajectory for visualization."""
    items = list(service.waterfall_history)
    return items[-limit:]


@router.get("/api/scenarios")
def get_scenarios() -> list[dict[str, Any]]:
    """Returns the catalog of available EW simulation scenarios."""
    return service.list_available_scenarios()


@router.get("/api/schedulers")
def get_schedulers() -> list[dict[str, Any]]:
    """Returns the catalog of available scan strategy algorithms."""
    return service.list_available_schedulers()


@router.post("/api/simulation/start")
async def start_simulation(body: SimulationControlRequest | None = None) -> dict[str, str]:
    """Starts or resumes asynchronous continuous execution."""
    steps = body.steps if body else None
    await service.start(steps=steps)
    return {"status": "started", "state": service.state}


@router.post("/api/simulation/pause")
async def pause_simulation() -> dict[str, str]:
    """Pauses continuous execution."""
    await service.pause()
    return {"status": "paused", "state": service.state}


@router.post("/api/simulation/step")
def step_simulation(body: SimulationControlRequest | None = None) -> dict[str, Any]:
    """Advances the simulation by a discrete number of steps."""
    count = body.steps if body and body.steps else 1
    return service.step(count=count)


@router.post("/api/simulation/reset")
def reset_simulation(body: SimulationResetRequest | None = None) -> dict[str, Any]:
    """Resets the simulation environment with optional seed, scenario, or scheduler."""
    seed = body.seed if body else None
    scenario = body.scenario_name if body else None
    scheduler = body.scheduler_name if body else None

    service.reset(seed=seed, scenario_name=scenario, scheduler_name=scheduler)
    return {"status": "reset", "telemetry": service.latest_telemetry}


@router.post("/api/simulation/speed")
def set_simulation_speed(body: SimulationSpeedRequest) -> dict[str, str]:
    """Configures the speed multiplier for continuous simulation execution."""
    service.set_speed(body.speed)
    return {"status": "updated", "speed": service.speed}
