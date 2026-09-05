from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.base import ScanScheduler
from rf_environment.domain.observation import Observation
from rf_environment.receiver.tuner import bands_overlap


class DeterministicOracle(ScanScheduler):
    """ORACLE — EVALUATION UPPER BOUND for deterministic cyclic hopper (200 -> 400 -> 600)."""
    def __init__(self, bands_hz: list[float], sequence: list[float]):
        super().__init__(bands_hz=bands_hz)
        self.name = "oracle"
        self.category = "oracle"
        self.sequence = sequence

    def select_frequency(self, observation: Observation | None) -> float:
        next_t = 0 if observation is None else (observation.timestamp + 1)
        target = self.sequence[next_t % len(self.sequence)]
        self.last_selected = target
        return float(target)

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        pass


def create_deterministic_hopper_scenario(seed: int = 42, steps: int = 1000, scheduler_name: str = "context_aware") -> dict[str, Any]:
    return {
        "simulation": {
            "total_time_steps": steps,
            "time_step_ms": 10,
            "seed": seed,
        },
        "spectrum": {
            "min_frequency_hz": 100_000_000,
            "max_frequency_hz": 700_000_000,
        },
        "receiver": {
            "instantaneous_bandwidth_hz": 20_000_000,
            "sensitivity_dbm": -90,
            "noise_floor_dbm": -150,
            "detection_threshold_db": 6,
            "tuning_time_ms": 0,
        },
        "detector": {
            "p_detection": 1.0,
            "p_false_alarm": 0.0,
        },
        "scheduler": {
            "type": scheduler_name,
            "bands_hz": [200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000],
        },
        "channel": {
            "noise_floor_dbm": -150,
            "path_loss": False,
        },
        "emitters": [
            {
                "id": "E_DET_HOP",
                "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [200_000_000, 400_000_000, 600_000_000],
                    "mode": "sequential",
                    "dwell_steps": 1,
                },
                "time_behavior": {
                    "type": "continuous",
                },
                "power_dbm": -20,
                "bandwidth_hz": 2_000_000,
            }
        ],
    }


def run_test_4_and_5():
    print("=" * 70)
    print("RUNNING TEST 4: PERFECT DETERMINISTIC HOPPER (1000 steps, 200->400->600)")
    print("RUNNING TEST 5: TRANSITION LEARNING WARM-UP CURVE")
    print("=" * 70)

    schedulers = [
        "oracle",
        "sequential",
        "random",
        "ucb1",
        "thompson",
        "sw_ucb",
        "discounted_thompson",
        "context_aware",
    ]

    seeds = [1, 2, 3, 4, 5, 42, 101, 202, 303, 404]
    summary_results = []
    context_aware_warmup_windows = []

    for sched_name in schedulers:
        actual_overlaps = []
        actual_detections = []
        prediction_accuracies = []
        intercept_delays = []
        total_hits_list = []

        for seed in seeds:
            sc = create_deterministic_hopper_scenario(seed=seed, steps=1000, scheduler_name="random" if sched_name == "oracle" else sched_name)
            env = build_environment(sc, scheduler_name="random" if sched_name == "oracle" else sched_name)

            if sched_name == "oracle":
                oracle = DeterministicOracle(
                    bands_hz=[200e6, 300e6, 400e6, 500e6, 600e6],
                    sequence=[200e6, 400e6, 600e6],
                )
                env.scheduler = oracle

            # Record timestep by timestep
            hits = 0
            first_hit = None
            step_evals = []

            for t in range(1000):
                res = env.step()
                gt = res["ground_truth"]["emitters"][0]
                rx_f = res["observation"]["receiver_frequency_hz"]
                det = res["detection"]["detected"]
                overlap = (rx_f == gt["frequency_hz"])

                if overlap and det:
                    hits += 1
                    if first_hit is None:
                        first_hit = t

                # Track window prediction records if context_aware on seed 42
                step_evals.append({
                    "t": t,
                    "overlap": overlap,
                    "hit": bool(overlap and det),
                })

            actual_overlap_rate = hits / 1000.0
            actual_overlaps.append(actual_overlap_rate)
            intercept_delays.append(first_hit if first_hit is not None else 1000)
            total_hits_list.append(hits)

            pred_acc = env.metrics.snapshot(1000).prediction_accuracy
            prediction_accuracies.append(pred_acc)

            # Test 5: For Context-Aware on seed 42, calculate window-by-window stats (50-step windows)
            if sched_name == "context_aware" and seed == 42:
                # Group into 20 windows of 50 steps
                all_preds = list(env.transition_tracker.predictions)
                for w_idx in range(20):
                    w_start = w_idx * 50
                    w_end = (w_idx + 1) * 50
                    w_steps = step_evals[w_start:w_end]
                    w_hits = sum(s["hit"] for s in w_steps)
                    w_step_coverage = w_hits / 50.0

                    # Predictions in this window
                    w_preds = [p for p in all_preds if w_start <= p.step < w_end]
                    w_pred_correct = sum(1 for p in w_preds if p.correct)
                    w_pred_acc = (w_pred_correct / len(w_preds)) if w_preds else 0.0

                    # Cumulative predictions up to w_end
                    cum_preds = [p for p in all_preds if p.step < w_end]
                    cum_pred_acc = (sum(1 for p in cum_preds if p.correct) / len(cum_preds)) if cum_preds else 0.0

                    context_aware_warmup_windows.append({
                        "window": f"{w_start}-{w_end}",
                        "w_start": w_start,
                        "w_end": w_end,
                        "step_coverage": w_step_coverage,
                        "window_hits": w_hits,
                        "window_prediction_accuracy": w_pred_acc,
                        "cumulative_prediction_accuracy": cum_pred_acc,
                        "predictions_count": len(w_preds),
                    })

        summary_results.append({
            "scheduler": sched_name.upper() if sched_name != "oracle" else "ORACLE — EVALUATION UPPER BOUND",
            "actual_overlap_rate_mean": float(np.mean(actual_overlaps)),
            "actual_overlap_rate_std": float(np.std(actual_overlaps)),
            "step_coverage_mean": float(np.mean(actual_overlaps)),
            "step_coverage_std": float(np.std(actual_overlaps)),
            "prediction_accuracy_mean": float(np.mean(prediction_accuracies)),
            "prediction_accuracy_std": float(np.std(prediction_accuracies)),
            "first_intercept_delay_mean": float(np.mean(intercept_delays)),
            "first_intercept_delay_std": float(np.std(intercept_delays)),
            "total_hits_mean": float(np.mean(total_hits_list)),
            "total_hits_std": float(np.std(total_hits_list)),
        })

    df_summary = pd.DataFrame(summary_results)
    print("\nTEST 4 RESULTS: DETERMINISTIC HOPPER (Mean ± Std across 10 seeds):")
    print(df_summary[["scheduler", "step_coverage_mean", "prediction_accuracy_mean", "first_intercept_delay_mean", "total_hits_mean"]].to_string())

    print("\nTEST 5 RESULTS: CONTEXT-AWARE WARM-UP CURVE (Seed 42, 50-step windows):")
    df_warmup = pd.DataFrame(context_aware_warmup_windows)
    print(df_warmup[["window", "step_coverage", "window_prediction_accuracy", "cumulative_prediction_accuracy", "predictions_count"]].to_string())

    out_file = Path("scratch/validation/test_4_5_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "summary": summary_results,
            "warmup_windows": context_aware_warmup_windows,
        }, f, indent=2)


if __name__ == "__main__":
    run_test_4_and_5()
