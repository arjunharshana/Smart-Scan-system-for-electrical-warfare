from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from rf_environment.api.schemas import (
    BenchmarkRequest,
    EmitterCreate,
    SchedulerSelect,
    SimulationControl,
)
from rf_environment.api.service import SimulationService

router = APIRouter()
service = SimulationService()


def get_service() -> SimulationService:
    return service


@router.get("/simulation/state")
def simulation_state():
    try:
        return service.require_env().public_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/simulation/start")
async def simulation_start(body: SimulationControl | None = None):
    try:
        await service.start(steps=body.steps if body else None)
        return {"status": "started"}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/simulation/pause")
async def simulation_pause():
    await service.pause()
    return {"status": "paused"}


@router.post("/simulation/stop")
async def simulation_stop():
    await service.stop()
    return {"status": "stopped"}


@router.post("/simulation/reset")
def simulation_reset():
    service.reset()
    return {"status": "reset"}


@router.post("/simulation/step")
def simulation_step(body: SimulationControl | None = None):
    env = service.require_env()
    count = body.steps if body and body.steps else 1
    last = None
    for _ in range(count):
        last = env.step()
        if last.get("finished"):
            break
    return last


@router.get("/emitters")
def list_emitters():
    env = service.require_env()
    result = []
    for emitter in env.emitters.values():
        if emitter._state:
            result.append(emitter.get_state().to_dict())
        else:
            result.append({"emitter_id": emitter.emitter_id, "emitter_type": emitter.emitter_type.value})
    return result


@router.get("/emitters/{emitter_id}")
def get_emitter(emitter_id: str):
    env = service.require_env()
    emitter = env.emitters.get(emitter_id)
    if not emitter:
        raise HTTPException(status_code=404, detail="Emitter not found")
    if emitter._state:
        return emitter.get_state().to_dict()
    return {"emitter_id": emitter.emitter_id, "emitter_type": emitter.emitter_type.value}


@router.post("/emitters")
def create_emitter(body: EmitterCreate):
    service.add_emitter(body.model_dump())
    return {"status": "created", "id": body.id}


@router.put("/emitters/{emitter_id}")
def update_emitter(emitter_id: str, body: EmitterCreate):
    service.update_emitter(emitter_id, body.model_dump())
    return {"status": "updated", "id": emitter_id}


@router.delete("/emitters/{emitter_id}")
def delete_emitter(emitter_id: str):
    service.delete_emitter(emitter_id)
    return {"status": "deleted", "id": emitter_id}


@router.get("/spectrum")
def spectrum():
    env = service.require_env()
    latest = env.ground_truth.latest
    return {
        "spectrum": env.spectrum,
        "receiver": env.receiver.get_state().to_dict(),
        "signals": [
            e.to_dict()
            for e in (latest.emitters if latest else [])
            if e.transmitting
        ],
    }


@router.get("/ground-truth")
def ground_truth():
    env = service.require_env()
    if env.ground_truth.latest is None:
        return {"timestamp": None, "emitters": []}
    return env.ground_truth.latest.to_dict()


@router.get("/receiver")
@router.get("/receiver/state")
def receiver():
    return service.require_env().receiver.get_state().to_dict()


@router.get("/scheduler")
@router.get("/scheduler/state")
def scheduler():
    return service.require_env().scheduler.get_state().to_dict()


@router.post("/scheduler")
def set_scheduler(body: SchedulerSelect):
    service.set_scheduler(body.type)
    return service.require_env().scheduler.get_state().to_dict()


@router.get("/metrics")
def metrics():
    env = service.require_env()
    return env.metrics.snapshot(max(env.clock.time_step, 0)).to_dict()


@router.get("/events")
@router.get("/simulation/events")
def events(limit: int = 100):
    env = service.require_env()
    return [e.to_dict() for e in env.events.recent(limit)]


@router.get("/observations")
def observations():
    env = service.require_env()
    t = env.clock.time_step
    hist = env.temporal_context.build_history(t).to_dict()
    return {
        "time_step": t,
        "receiver_frequency_hz": env.receiver.center_frequency_hz,
        "bandwidth_hz": env.receiver.instantaneous_bandwidth_hz,
        "temporal_history": hist,
    }


@router.get("/opportunities")
def opportunities():
    env = service.require_env()
    return env.get_opportunities_summary()


@router.post("/benchmark")
def benchmark(body: BenchmarkRequest | None = None):
    from rf_environment.experiments.runner import BenchmarkRunner
    from rf_environment.environment.scenario import load_scenario
    from rf_environment.app.config import DEFAULT_SCENARIO

    req = body or BenchmarkRequest()
    scenario_path = req.scenario_path or str(DEFAULT_SCENARIO)
    scenario_dict = load_scenario(scenario_path)

    runner = BenchmarkRunner()
    results = runner.run_benchmark(
        scenario=scenario_dict,
        algorithms=req.algorithms,
        seeds=req.seeds,
        steps=req.steps,
        scenario_name=str(scenario_path),
    )
    import numpy as np

    def _clean_val(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, dict):
            return {k: _clean_val(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_clean_val(val) for val in v]
        return v

    summary_records = [_clean_val(r) for r in results["summary_df"].to_dict(orient="records")]
    return {
        "scenario": results["scenario"],
        "algorithms": results["algorithms"],
        "seeds": results["seeds"],
        "steps": results["steps"],
        "summary": summary_records,
        "runs_count": len(results["runs"]),
    }


@router.get("/waterfall")
def waterfall(limit: int = 200):
    env = service.require_env()
    items = list(env.waterfall)
    return items[-limit:]


@router.websocket("/ws/simulation")
async def websocket_simulation(ws: WebSocket):
    await ws.accept()
    service.clients.append(ws)
    try:
        if service.env:
            await ws.send_json({"event_type": "SNAPSHOT", "payload": service.env.public_state()})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in service.clients:
            service.clients.remove(ws)
