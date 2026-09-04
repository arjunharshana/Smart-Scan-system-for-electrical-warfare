from __future__ import annotations

import copy
from typing import Any
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.domain.metrics import MetricSnapshot
from rf_environment.experiments.runner import BenchmarkRunner
from dashboard.utils import (
    build_time_series_dataframe,
    build_waterfall_dataframe,
    build_arm_stats_dataframe,
)


AVAILABLE_ALGORITHMS = [
    "sequential",
    "random",
    "ucb1",
    "thompson",
    "sw_ucb",
    "discounted_thompson",
    "context_aware",
]

ALGORITHM_DESCRIPTIONS = {
    "sequential": "Baseline: Cycles systematically through scan bands in deterministic ascending frequency order.",
    "random": "Baseline: Uniformly random selection across all available channels without learning.",
    "ucb1": "Stationary Bandit: Upper Confidence Bound (UCB1). Balances empirical mean reward with statistical confidence exploration.",
    "thompson": "Stationary Bandit: Thompson Sampling with Beta posteriors. Samples success probabilities per channel.",
    "sw_ucb": "Non-Stationary Bandit: Sliding Window UCB. Retains only the most recent N observations to rapidly adapt to hopping.",
    "discounted_thompson": "Non-Stationary Bandit: Discounted Thompson Sampling. Applies exponential decay (gamma < 1.0) so recent observations dominate.",
    "context_aware": "Contextual / Temporal: Learns observed transition matrix P(f_{next} | f_{prev}) from detections, fused with recency and coverage bonuses.",
}


def run_single_simulation(
    scenario: dict[str, Any],
    scheduler_name: str,
    steps: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """Runs a single simulation and packages the comprehensive telemetry and diagnostic payload."""
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

    # Compile Emitter Discovery & Status Table (with clear ground-truth vs observation labels)
    emitter_stats = []
    first_intercepts = snapshot.time_to_first_intercept or {}
    for eid, emitter in env.emitters.items():
        state = emitter.get_state().to_dict() if emitter._state else {}
        first_t = first_intercepts.get(eid)
        emitter_stats.append({
            "Emitter ID [Truth]": eid,
            "Type [Truth]": state.get("emitter_type", emitter.emitter_type.value),
            "Current Freq (MHz) [Truth]": f"{state.get('frequency_hz', 0.0) / 1e6:.1f}",
            "Bandwidth (MHz) [Truth]": f"{state.get('bandwidth_hz', 0.0) / 1e6:.1f}",
            "Power (dBm) [Truth]": f"{state.get('power_dbm', -30.0):.1f}",
            "First Intercept [Observed]": f"Step {first_t}" if first_t is not None else "Not Detected",
            "Detection Status [Evaluation]": "🎯 Intercepted" if first_t is not None else "❌ Undetected",
        })

    # Transmission Opportunities Summary
    opps_summary = env.get_opportunities_summary()
    opportunities_df = pd.DataFrame(opps_summary) if opps_summary else pd.DataFrame()

    # Observed Transition Matrix
    transition_matrix_df = env.transition_tracker.get_transition_matrix()

    # Prediction Log
    predictions_records = []
    for p in env.transition_tracker.predictions:
        predictions_records.append({
            "Step": p.step,
            "Previous Band (MHz)": f"{p.previous_frequency_hz / 1e6:.0f}",
            "Predicted Band (MHz)": f"{p.predicted_frequency_hz / 1e6:.0f}",
            "Observed Band (MHz)": f"{p.actual_observed_frequency_hz / 1e6:.0f}" if p.actual_observed_frequency_hz else "—",
            "Confidence": f"{p.confidence:.2%}",
            "Outcome": "✅ Correct" if p.correct else "❌ Incorrect",
        })
    predictions_df = pd.DataFrame(predictions_records)

    # "Why Did We Miss?" Diagnostic Records
    why_missed_records = []
    for r in results:
        outcome = r.get("outcome", {}).get("outcome", "")
        t = r.get("timestamp", 0)
        rx_f = r.get("receiver", {}).get("center_frequency_hz", 0.0) / 1e6
        diag = r.get("diagnostic_reason", "")
        # Highlight misses, false alarms, and off-frequency steps
        why_missed_records.append({
            "Step": t,
            "Receiver Freq (MHz)": rx_f,
            "Scan Outcome": outcome,
            "Diagnostic Rationale": diag,
        })
    why_missed_df = pd.DataFrame(why_missed_records)

    return {
        "env": env,
        "results": results,
        "snapshot": snapshot,
        "time_df": time_df,
        "waterfall_df": waterfall_df,
        "arm_df": arm_df,
        "emitter_stats_df": pd.DataFrame(emitter_stats),
        "opportunities_df": opportunities_df,
        "transition_matrix_df": transition_matrix_df,
        "predictions_df": predictions_df,
        "why_missed_df": why_missed_df,
        "scheduler_name": scheduler_name,
        "steps": steps,
        "scenario": scenario_copy,
    }


def run_benchmark_comparison(
    scenario: dict[str, Any],
    algorithms: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs a multi-seed benchmark comparing all algorithms on identical frozen RF realizations."""
    if algorithms is None:
        algorithms = AVAILABLE_ALGORITHMS
    if seeds is None:
        seeds = [seed] if seed is not None else [1, 2, 3]

    runner = BenchmarkRunner()
    results = runner.run_benchmark(
        scenario=scenario,
        algorithms=algorithms,
        seeds=seeds,
        steps=steps,
    )
    return results
