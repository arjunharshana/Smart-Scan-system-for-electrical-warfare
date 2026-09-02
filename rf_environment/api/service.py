from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket

from rf_environment.domain.events import SimulationEvent
from rf_environment.environment.builder import build_environment
from rf_environment.environment.rf_environment import RFEnvironment
from rf_environment.emitters.factory import create_emitter
from rf_environment.scheduler.factory import create_scheduler, default_scan_bands


class SimulationService:
    def __init__(self) -> None:
        self.scenario: dict[str, Any] = {}
        self.env: RFEnvironment | None = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.clients: list[WebSocket] = []
        self.realtime_delay_s = 0.0

    def load_scenario(self, scenario: dict[str, Any], scheduler_name: str | None = None) -> None:
        self.scenario = scenario
        self.env = build_environment(scenario, scheduler_name=scheduler_name)
        self.env.events.subscribe(self._on_event)

    def require_env(self) -> RFEnvironment:
        if self.env is None:
            raise RuntimeError("Simulation is not loaded")
        return self.env

    def _on_event(self, event: SimulationEvent) -> None:
        if not self.clients:
            return
        payload = event.to_dict()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for ws in list(self.clients):
            loop.create_task(ws.send_json(payload))

    async def start(self, steps: int | None = None) -> None:
        env = self.require_env()
        env.running = True
        env.paused = False
        from rf_environment.domain.enums import EventType

        env.events.publish(max(env.clock.time_step, 0), EventType.SIMULATION_STARTED)
        if self._task and not self._task.done():
            return

        async def _run() -> None:
            remaining = steps
            while env.running and not env.paused and not env.clock.finished():
                if remaining is not None and remaining <= 0:
                    break
                env.step()
                if remaining is not None:
                    remaining -= 1
                if self.realtime_delay_s:
                    await asyncio.sleep(self.realtime_delay_s)
                else:
                    await asyncio.sleep(0)
            env.running = False

        self._task = asyncio.create_task(_run())

    async def pause(self) -> None:
        env = self.require_env()
        env.paused = True
        env.running = False
        from rf_environment.domain.enums import EventType

        env.events.publish(max(env.clock.time_step, 0), EventType.SIMULATION_PAUSED)

    async def stop(self) -> None:
        env = self.require_env()
        env.running = False
        env.paused = False
        from rf_environment.domain.enums import EventType

        env.events.publish(max(env.clock.time_step, 0), EventType.SIMULATION_STOPPED)
        if self._task:
            self._task.cancel()
            self._task = None

    def reset(self, scheduler_name: str | None = None) -> None:
        name = scheduler_name or (self.env.scheduler.name if self.env else None)
        self.load_scenario(self.scenario, scheduler_name=name)
        from rf_environment.domain.enums import EventType

        self.env.events.publish(0, EventType.SIMULATION_RESET)

    def add_emitter(self, config: dict[str, Any]) -> None:
        env = self.require_env()
        seed = int(self.scenario.get("simulation", {}).get("seed", 0))
        emitter = create_emitter(config, seed=seed)
        env.emitters[emitter.emitter_id] = emitter
        emitters = list(self.scenario.get("emitters", []))
        emitters.append(config)
        self.scenario["emitters"] = emitters

    def update_emitter(self, emitter_id: str, config: dict[str, Any]) -> None:
        config = {**config, "id": emitter_id}
        env = self.require_env()
        seed = int(self.scenario.get("simulation", {}).get("seed", 0))
        env.emitters[emitter_id] = create_emitter(config, seed=seed)
        updated = []
        for item in self.scenario.get("emitters", []):
            if item.get("id") == emitter_id:
                updated.append(config)
            else:
                updated.append(item)
        self.scenario["emitters"] = updated

    def delete_emitter(self, emitter_id: str) -> None:
        env = self.require_env()
        env.emitters.pop(emitter_id, None)
        self.scenario["emitters"] = [
            e for e in self.scenario.get("emitters", []) if e.get("id") != emitter_id
        ]

    def set_scheduler(self, name: str) -> None:
        env = self.require_env()
        spectrum = env.spectrum
        bw = env.receiver.instantaneous_bandwidth_hz
        bands = default_scan_bands(spectrum["min_frequency_hz"], spectrum["max_frequency_hz"], bw)
        env.scheduler = create_scheduler(name, bands)
        env._initialized = False
