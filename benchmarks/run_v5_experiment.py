"""benchmarks/run_v5_experiment.py
Evaluation and ablation benchmark for V5.0 Augmented Belief-State Bayesian Scheduler.

Ablations:
- B0: Frequency belief only (D=1)
- B1: Augmented belief (F, tau, D) with uniform hop transitions
- B2: Augmented belief (F, tau, D) with online learned hop transition matrix

Canonical 8 Scenarios:
Evaluated across 5 random seeds [10, 20, 30, 40, 50] (300 steps per episode).
"""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.scenarios_v4_1 import get_canonical_test_scenarios
from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.belief.config import BeliefSchedulerConfig
from rf_environment.scheduler.belief.scheduler import V5BeliefScheduler


def evaluate_v5_on_scenario(
    scenario_dict: dict[str, Any],
    ablation_mode: str = "B1",
    seeds: list[int] | None = None,
    steps: int = 300,
    exploration_mode: str = "uncertainty",
    uncertainty_weight: float = 0.08,
) -> dict[str, Any]:
    """Evaluates V5BeliefScheduler on a given scenario dictionary over specified seeds."""
    if seeds is None:
        seeds = [10, 20, 30, 40, 50]

    cfg = BeliefSchedulerConfig(
        ablation_mode=ablation_mode,
        exploration_mode=exploration_mode,
        uncertainty_weight=uncertainty_weight,
    )

    irs: list[float] = []
    det_rates: list[float] = []
    scan_effs: list[float] = []
    rewards: list[float] = []
    step_times_us: list[float] = []

    for s in seeds:
        sc_copy = copy.deepcopy(scenario_dict)
        sc_copy["simulation"]["seed"] = s
        sc_copy["simulation"]["total_time_steps"] = steps

        env = build_environment(sc_copy)
        sched = V5BeliefScheduler(env.bands_hz, config=cfg, seed=s)
        env.scheduler = sched
        sched.reset()

        # Run episode and measure latency
        t_start = time.perf_counter()
        step_results = env.run(steps=steps)
        t_elapsed = time.perf_counter() - t_start

        avg_us = (t_elapsed / steps) * 1e6
        step_times_us.append(avg_us)

        snap = env.metrics.snapshot(env.clock.time_step).to_dict()
        ir = float(snap.get("interception_ratio", 0.0))
        det_rate = float(snap.get("probability_of_detection", 0.0))
        hits = sum(1 for r in step_results if r.observation.last_detection)
        eff = float(hits) / float(max(len(step_results), 1))
        ep_reward = float(sum(r.reward for r in step_results))

        irs.append(ir)
        det_rates.append(det_rate)
        scan_effs.append(eff)
        rewards.append(ep_reward)

    return {
        "ablation_mode": ablation_mode,
        "ir_mean": float(np.mean(irs)),
        "ir_std": float(np.std(irs)),
        "det_rate_mean": float(np.mean(det_rates)),
        "scan_eff_mean": float(np.mean(scan_effs)),
        "reward_mean": float(np.mean(rewards)),
        "latency_us_mean": float(np.mean(step_times_us)),
        "per_seed_ir": irs,
    }


def run_benchmark() -> dict[str, Any]:
    print("=" * 70)
    print("🎯 V5.0 Augmented Belief-State Bayesian Scheduler Evaluation")
    print("=" * 70)

    test_scenarios = get_canonical_test_scenarios()
    seeds = [10, 20, 30, 40, 50]
    ablations = ["B0", "B1", "B2"]

    results: dict[str, dict[str, Any]] = {mode: {} for mode in ablations}

    for mode in ablations:
        print(f"\n--- Running Ablation {mode} across 8 Canonical Scenarios (5 seeds each) ---")
        mode_irs = []
        for sc_name, sc_dict in test_scenarios.items():
            res = evaluate_v5_on_scenario(sc_dict, ablation_mode=mode, seeds=seeds)
            results[mode][sc_name] = res
            mode_irs.append(res["ir_mean"])
            print(f"  [{mode}] {sc_name:<25}: IR = {res['ir_mean']*100:.2f}% ± {res['ir_std']*100:.2f}% | Latency: {res['latency_us_mean']:.1f} µs")

        mean_ir = float(np.mean(mode_irs))
        results[mode]["OVERALL_MEAN"] = mean_ir
        print(f"  >>> [{mode}] OVERALL MEAN IR: {mean_ir*100:.2f}%")

    # Baseline comparison references from verified historical runs
    baselines = {
        "CA": {
            "1_Seen_Structure": 0.2690,
            "2_Unseen_Permutation": 0.3140,
            "3_Unseen_Phase": 0.3040,
            "4_Unseen_Dwell": 0.2453,
            "5_Unseen_Subset": 0.2740,
            "6_Mixed_Shift": 0.3067,
            "7_Random_Hopping": 0.2637,
            "8_Periodic_Burst": 0.5150,
            "OVERALL_MEAN": 0.3115,
            "latency_us": 53.3,
            "training": "No",
            "gt_access": "No",
        },
        "V4.0": {
            "1_Seen_Structure": 0.5062,
            "2_Unseen_Permutation": 0.4048,
            "3_Unseen_Phase": 0.4570,
            "4_Unseen_Dwell": 0.1957,
            "5_Unseen_Subset": 0.2142,
            "6_Mixed_Shift": 0.1471,
            "7_Random_Hopping": 0.1418,
            "8_Periodic_Burst": 0.7483,
            "OVERALL_MEAN": 0.3519,
            "latency_us": 85.0,
            "training": "Yes",
            "gt_access": "No",
        },
        "V4.1_H64": {
            "1_Seen_Structure": 0.3480,
            "2_Unseen_Permutation": 0.2870,
            "3_Unseen_Phase": 0.3260,
            "4_Unseen_Dwell": 0.1830,
            "5_Unseen_Subset": 0.2210,
            "6_Mixed_Shift": 0.2200,
            "7_Random_Hopping": 0.1520,
            "8_Periodic_Burst": 0.4150,
            "OVERALL_MEAN": 0.2690,
            "latency_us": 110.8,
            "training": "Yes",
            "gt_access": "No",
        },
        "Whittle_W3": {
            "1_Seen_Structure": 0.2990,
            "2_Unseen_Permutation": 0.4680,
            "3_Unseen_Phase": 0.3850,
            "4_Unseen_Dwell": 0.0843,
            "5_Unseen_Subset": 0.2920,
            "6_Mixed_Shift": 0.0917,
            "7_Random_Hopping": 0.2617,
            "8_Periodic_Burst": 0.8700,
            "OVERALL_MEAN": 0.3440,
            "latency_us": 162.9,
            "training": "No",
            "gt_access": "No",
        },
    }

    full_output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seeds": seeds,
        "ablations": results,
        "baselines": baselines,
    }

    # Save results to json
    out_dir = PROJECT_ROOT / "models" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v5_belief_results.json", "w") as f:
        json.dump(full_output, f, indent=2)
    with open(PROJECT_ROOT / "v5_belief_results.json", "w") as f:
        json.dump(full_output, f, indent=2)

    print("\n✅ Results successfully saved to v5_belief_results.json")
    return full_output


if __name__ == "__main__":
    run_benchmark()
