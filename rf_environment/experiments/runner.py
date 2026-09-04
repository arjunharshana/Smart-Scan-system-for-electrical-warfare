from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario


class BenchmarkRunner:
    """Multi-seed experiment and benchmark execution engine for scan schedulers.

    Ensures that for each seed, all algorithms face the exact same frozen RF environment
    realization (identical emitter sequences, channel noise, and detector conditions).
    """

    def __init__(self) -> None:
        pass

    def run_single(
        self,
        scenario: dict[str, Any],
        scheduler_name: str,
        seed: int = 42,
        steps: int | None = None,
        scenario_name: str = "scenario",
    ) -> dict[str, Any]:
        """Runs a single simulation run for one scheduler under a specified seed."""
        scenario_copy = copy.deepcopy(scenario)
        if "simulation" not in scenario_copy:
            scenario_copy["simulation"] = {}
        scenario_copy["simulation"]["seed"] = seed
        if steps is not None:
            scenario_copy["simulation"]["total_time_steps"] = steps

        env = build_environment(scenario_copy, scheduler_name=scheduler_name)
        results = env.run(steps=steps)
        snapshot = env.metrics.snapshot(max(env.clock.time_step, 0))
        snap_dict = snapshot.to_dict()

        return {
            "scenario": scenario_name,
            "seed": seed,
            "algorithm": scheduler_name,
            "steps": env.clock.time_step,
            "metric_scope": snap_dict.get("metric_scope", "HOP"),
            "Pd": snap_dict.get("probability_of_detection", 0.0),
            "Pfa": snap_dict.get("probability_of_false_alarm", 0.0),
            "opportunity_coverage": snap_dict.get("opportunity_coverage", 0.0),
            "detection_given_coverage": snap_dict.get("detection_given_coverage", 0.0),
            "interception_ratio": snap_dict.get("interception_ratio", 0.0),
            "hop_coverage_ratio": snap_dict.get("hop_coverage_ratio", 0.0),
            "hop_detection_ratio": snap_dict.get("hop_detection_ratio", 0.0),
            "hop_interception_ratio": snap_dict.get("hop_interception_ratio", 0.0),
            "total_hops": snap_dict.get("total_hops", 0),
            "covered_hops": snap_dict.get("covered_hops", 0),
            "intercepted_hops": snap_dict.get("intercepted_hops", 0),
            "episode_coverage_ratio": snap_dict.get("episode_coverage_ratio", 0.0),
            "episode_interception_ratio": snap_dict.get("episode_interception_ratio", 0.0),
            "avg_intercept_time": snap_dict.get("average_intercept_time"),
            "median_intercept_time": snap_dict.get("median_intercept_time"),
            "prediction_accuracy": snap_dict.get("prediction_accuracy", 0.0),
            "avg_reward": snap_dict.get("average_reward", 0.0),
            "cumulative_reward": snap_dict.get("cumulative_reward", 0.0),
            "hits": snap_dict.get("hits", 0),
            "step_coverage_ratio": snap_dict.get("step_coverage_ratio", 0.0),
            "unique_emitters_detected": snap_dict.get("unique_emitters_detected", 0),
            "opportunities_total": snap_dict.get("opportunities_total", 0),
            "opportunities_intercepted": snap_dict.get("opportunities_intercepted", 0),
            "identity_verified": snap_dict.get("identity_verified", True),
        }

    def run_benchmark(
        self,
        scenario: dict[str, Any],
        algorithms: list[str],
        seeds: list[int] | None = None,
        steps: int | None = None,
        scenario_name: str = "scenario",
    ) -> dict[str, Any]:
        """Executes a multi-seed benchmark comparing all algorithms on identical RF realizations."""
        if seeds is None:
            seeds = [1, 2, 3, 4, 5]

        run_records = []
        for seed in seeds:
            for alg in algorithms:
                res = self.run_single(
                    scenario=scenario,
                    scheduler_name=alg,
                    seed=seed,
                    steps=steps,
                    scenario_name=scenario_name,
                )
                run_records.append(res)

        df_runs = pd.DataFrame(run_records)

        # Compute statistical aggregates across seeds per algorithm
        summary_rows = []
        metric_cols = [
            "Pd",
            "Pfa",
            "hop_coverage_ratio",
            "hop_interception_ratio",
            "detection_given_coverage",
            "opportunity_coverage",
            "interception_ratio",
            "step_coverage_ratio",
            "avg_intercept_time",
            "median_intercept_time",
            "prediction_accuracy",
            "avg_reward",
            "cumulative_reward",
            "hits",
            "unique_emitters_detected",
        ]

        for alg in algorithms:
            sub = df_runs[df_runs["algorithm"] == alg]
            row: dict[str, Any] = {"Algorithm": alg}
            for col in metric_cols:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    mean_v = float(vals.mean())
                    std_v = float(vals.std()) if len(vals) > 1 else 0.0
                    median_v = float(vals.median())
                    ci95 = 1.96 * (std_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                    row[f"{col}_mean"] = mean_v
                    row[f"{col}_std"] = std_v
                    row[f"{col}_median"] = median_v
                    row[f"{col}_ci95"] = ci95
                    row[col] = f"{mean_v:.3f} ± {std_v:.3f}"
                else:
                    row[f"{col}_mean"] = None
                    row[f"{col}_std"] = None
                    row[f"{col}_median"] = None
                    row[f"{col}_ci95"] = None
                    row[col] = "N/A"
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows).replace({np.nan: None})
        # Clean any remaining NaNs in numeric columns
        for col in df_summary.columns:
            df_summary[col] = df_summary[col].apply(lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

        return {
            "runs": run_records,
            "runs_df": df_runs,
            "summary_df": df_summary,
            "algorithms": algorithms,
            "seeds": seeds,
            "steps": steps,
            "scenario": scenario_name,
        }

    def export_results(
        self,
        benchmark_results: dict[str, Any],
        output_dir: str | Path,
        prefix: str = "benchmark",
    ) -> tuple[Path, Path]:
        """Exports benchmark runs and summary to CSV and JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        csv_file = out_path / f"{prefix}_runs.csv"
        json_file = out_path / f"{prefix}_results.json"

        benchmark_results["runs_df"].to_csv(csv_file, index=False)

        exportable = {
            "scenario": benchmark_results["scenario"],
            "algorithms": benchmark_results["algorithms"],
            "seeds": benchmark_results["seeds"],
            "steps": benchmark_results["steps"],
            "summary": benchmark_results["summary_df"].to_dict(orient="records"),
            "runs": benchmark_results["runs"],
        }
        with open(json_file, "w") as f:
            json.dump(exportable, f, indent=2)

        return csv_file, json_file

    def run(
        self,
        scenario: dict[str, Any],
        scheduler_name: str,
        seed: int | None = None,
        steps: int | None = None,
        scenario_name: str = "scenario",
    ) -> ExperimentResult:
        s_seed = seed if seed is not None else 42
        single = self.run_single(scenario, scheduler_name, seed=s_seed, steps=steps, scenario_name=scenario_name)
        scenario_copy = copy.deepcopy(scenario)
        if seed is not None:
            if "simulation" not in scenario_copy:
                scenario_copy["simulation"] = {}
            scenario_copy["simulation"]["seed"] = seed
        env = build_environment(scenario_copy, scheduler_name=scheduler_name)
        env.run(steps=steps)
        snapshot = env.metrics.snapshot(max(env.clock.time_step, 0))
        return ExperimentResult(
            scenario_name=scenario_name,
            scheduler=scheduler_name,
            seed=s_seed,
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


from dataclasses import dataclass


@dataclass
class ExperimentResult:
    scenario_name: str
    scheduler: str
    seed: int
    metrics: dict[str, Any]


# Backward compatibility alias
ExperimentRunner = BenchmarkRunner
