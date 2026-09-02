from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario


@dataclass
class ExperimentResult:
    scenario_name: str
    scheduler: str
    seed: int
    metrics: dict[str, Any]


class ExperimentRunner:
    def run(
        self,
        scenario: dict[str, Any],
        scheduler_name: str,
        seed: int | None = None,
        steps: int | None = None,
        scenario_name: str = "scenario",
    ) -> ExperimentResult:
        scenario = dict(scenario)
        sim = dict(scenario.get("simulation", {}))
        if seed is not None:
            sim["seed"] = seed
        scenario["simulation"] = sim
        env = build_environment(scenario, scheduler_name=scheduler_name)
        env.run(steps=steps)
        snapshot = env.metrics.snapshot(max(env.clock.time_step, 0))
        return ExperimentResult(
            scenario_name=scenario_name,
            scheduler=scheduler_name,
            seed=int(sim.get("seed", 0)),
            metrics=snapshot.to_dict(),
        )

    def compare(
        self,
        scenario: dict[str, Any],
        schedulers: list[str],
        seed: int = 42,
        steps: int | None = None,
        scenario_name: str = "scenario",
    ) -> list[ExperimentResult]:
        return [
            self.run(scenario, name, seed=seed, steps=steps, scenario_name=scenario_name)
            for name in schedulers
        ]

    def run_path(self, path: str, scheduler_name: str, **kwargs) -> ExperimentResult:
        return self.run(load_scenario(path), scheduler_name, **kwargs)
