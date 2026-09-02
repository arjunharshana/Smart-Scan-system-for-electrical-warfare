from __future__ import annotations

import copy
from typing import Any
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.domain.metrics import MetricSnapshot
from dashboard.utils import (
    build_time_series_dataframe,
    build_waterfall_dataframe,
    build_arm_stats_dataframe,
)


AVAILABLE_ALGORITHMS = ["sequential", "random", "ucb1", "thompson"]

ALGORITHM_DESCRIPTIONS = {
    "sequential": "Systematically scans frequency bands in fixed sequential order from lowest to highest frequency.",
    "random": "Randomly selects a frequency band at each time step uniformly without learning.",
    "ucb1": "Upper Confidence Bound (UCB1) multi-armed bandit algorithm. Balances exploiting high-reward bands with exploring uncertain bands using statistical confidence bounds.",
    "thompson": "Thompson Sampling Bayesian bandit. Maintains Beta posterior distributions per band and samples success probabilities to efficiently detect active frequency channels.",
}


def run_single_simulation(
    scenario: dict[str, Any],
    scheduler_name: str,
    steps: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """Runs a single simulation run and packages the full diagnostic payload."""
    scenario_copy = copy.deepcopy(scenario)
    if seed is not None:
        if "simulation" not in scenario_copy:
            scenario_copy["simulation"] = {}
        scenario_copy["simulation"]["seed"] = seed

    env = build_environment(scenario_copy, scheduler_name=scheduler_name)
    results = env.run(steps=steps)
    snapshot: MetricSnapshot = env.metrics.snapshot(env.clock.time_step)

    time_df = build_time_series_dataframe(results)
    waterfall_df = build_waterfall_dataframe(list(env.waterfall))
    arm_df = build_arm_stats_dataframe(env.scheduler.get_state().to_dict())

    # Compile Emitter Discovery & Stats
    emitter_stats = []
    first_intercepts = snapshot.time_to_first_intercept or {}
    for eid, emitter in env.emitters.items():
        state = emitter.get_state().to_dict() if emitter._state else {}
        first_t = first_intercepts.get(eid)
        emitter_stats.append({
            "Emitter ID": eid,
            "Type": state.get("emitter_type", emitter.emitter_type.value),
            "Freq (MHz)": state.get("frequency_hz", 0.0) / 1e6,
            "Bandwidth (MHz)": state.get("bandwidth_hz", 0.0) / 1e6,
            "Power (dBm)": state.get("power_dbm", -30.0),
            "Modulation": state.get("modulation", "UNKNOWN"),
            "First Intercept (Step)": f"Step {first_t}" if first_t is not None else "Not Detected",
            "Detected": "✅ Yes" if first_t is not None else "❌ No",
        })

    return {
        "env": env,
        "results": results,
        "snapshot": snapshot,
        "time_df": time_df,
        "waterfall_df": waterfall_df,
        "arm_df": arm_df,
        "emitter_stats_df": pd.DataFrame(emitter_stats),
        "scheduler_name": scheduler_name,
        "steps": steps,
        "scenario": scenario_copy,
    }


def run_benchmark_comparison(
    scenario: dict[str, Any],
    algorithms: list[str] | None = None,
    steps: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs a side-by-side benchmark comparing multiple algorithms on the same scenario."""
    if algorithms is None:
        algorithms = AVAILABLE_ALGORITHMS

    runs = {}
    summary_rows = []
    combined_rewards = []

    for alg in algorithms:
        sim_data = run_single_simulation(scenario, scheduler_name=alg, steps=steps, seed=seed)
        runs[alg] = sim_data
        snap = sim_data["snapshot"]

        summary_rows.append({
            "Algorithm": alg.upper(),
            "Total Scans": snap.total_scans,
            "Hits": snap.hits,
            "Misses": snap.misses,
            "False Alarms": snap.false_alarms,
            "P(Detection)": snap.probability_of_detection,
            "Interception Ratio": snap.interception_ratio,
            "Avg Reward": snap.average_reward,
            "Total Cumulative Reward": sim_data["time_df"]["Cumulative Reward"].iloc[-1] if not sim_data["time_df"].empty else 0.0,
            "Unique Detected": snap.unique_emitters_detected,
            "Avg Intercept Latency": f"{snap.average_intercept_time:.1f} steps" if snap.average_intercept_time is not None else "N/A",
        })

        # Gather reward trajectories
        time_df = sim_data["time_df"]
        for _, row in time_df.iterrows():
            combined_rewards.append({
                "Time Step": row["Time Step"],
                "Cumulative Reward": row["Cumulative Reward"],
                "Rolling Hit Rate": row["Rolling Hit Rate"],
                "Algorithm": alg.upper(),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="Total Cumulative Reward", ascending=False)
    combined_trajectory_df = pd.DataFrame(combined_rewards)

    return {
        "runs": runs,
        "summary_df": summary_df,
        "trajectory_df": combined_trajectory_df,
        "algorithms": algorithms,
        "steps": steps,
        "seed": seed,
    }
