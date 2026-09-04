from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from rf_environment.environment.scenario import load_scenario
from rf_environment.experiments.runner import BenchmarkRunner


def run_all_benchmarks():
    scenarios = [
        ("Stationary Fixed (Single Continuous Emitter)", "rf_environment/scenarios/stationary_fixed.yaml", 300),
        ("Periodic Time (Time-Slotted Periodic Emitter)", "rf_environment/scenarios/periodic_time.yaml", 300),
        ("Deterministic Hopping (Cyclic 3-Band Hopper)", "rf_environment/scenarios/deterministic_hopping.yaml", 300),
        ("Random Hopping (5-Band Agile Hopper)", "rf_environment/scenarios/random_hopping.yaml", 300),
        ("Mixed Environment (Multi-Emitter Complex RF)", "rf_environment/scenarios/mixed_environment.yaml", 400),
    ]

    algorithms = [
        "sequential",
        "random",
        "ucb1",
        "thompson",
        "sw_ucb",
        "discounted_thompson",
        "context_aware",
    ]

    seeds = [42, 101, 202, 303, 404]
    runner = BenchmarkRunner()

    all_scenario_results = {}

    for name, path, steps in scenarios:
        print(f"Running benchmark on {name} ({len(algorithms)} algos x {len(seeds)} seeds = {len(algorithms)*len(seeds)} runs)...")
        sc_dict = load_scenario(path)
        bench = runner.run_benchmark(
            scenario=sc_dict,
            algorithms=algorithms,
            seeds=seeds,
            steps=steps,
            scenario_name=name,
        )

        summary_df = bench["summary_df"]
        runs_df = bench["runs_df"]

        all_scenario_results[name] = {
            "scenario_path": path,
            "steps": steps,
            "seeds": seeds,
            "summary": summary_df.to_dict(orient="records"),
            "raw_runs": runs_df.to_dict(orient="records"),
        }

    out_file = Path("scratch/benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_scenario_results, f, indent=2)

    print(f"\nAll benchmarks completed successfully! Exported to {out_file}")


if __name__ == "__main__":
    run_all_benchmarks()
