from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.sliding_window_ucb import SlidingWindowUCBScheduler
from rf_environment.scheduler.discounted_thompson import DiscountedThompsonSamplingScheduler


def create_agility_scenario(dwell: int, steps: int = 600, seed: int = 42) -> dict[str, Any]:
    return {
        "simulation": {"total_time_steps": steps, "time_step_ms": 10, "seed": seed},
        "spectrum": {"min_frequency_hz": 100_000_000, "max_frequency_hz": 700_000_000},
        "receiver": {
            "instantaneous_bandwidth_hz": 20_000_000,
            "sensitivity_dbm": -90,
            "noise_floor_dbm": -150,
            "detection_threshold_db": 6,
            "tuning_time_ms": 0,
        },
        "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
        "scheduler": {
            "type": "sequential",
            "bands_hz": [200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000],
        },
        "channel": {"noise_floor_dbm": -150, "path_loss": False},
        "emitters": [
            {
                "id": "E_AGILE_HOP",
                "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [200_000_000, 400_000_000, 600_000_000],
                    "mode": "sequential",
                    "dwell_steps": dwell,
                },
                "time_behavior": {"type": "continuous"},
                "power_dbm": -20,
                "bandwidth_hz": 2_000_000,
            }
        ],
    }


def run_test_9():
    print("=" * 70)
    print("RUNNING TEST 9: FREQUENCY AGILITY SPEED TEST")
    print("=" * 70)

    dwells = [20, 10, 5, 3, 1]
    schedulers = ["ucb1", "thompson", "sw_ucb", "discounted_thompson", "context_aware"]
    seeds = [42, 101, 202]

    # Store: dwell -> scheduler -> metrics
    records = []

    for dwell in dwells:
        for sched in schedulers:
            covs = []
            hits_list = []
            preds = []
            delays = []

            for seed in seeds:
                sc = create_agility_scenario(dwell=dwell, steps=600, seed=seed)
                env = build_environment(sc, scheduler_name=sched)

                hits = 0
                first_hit = None
                for t in range(600):
                    res = env.step()
                    gt = res["ground_truth"]["emitters"][0]
                    rx_f = res["observation"]["receiver_frequency_hz"]
                    det = res["detection"]["detected"]
                    if rx_f == gt["frequency_hz"] and det:
                        hits += 1
                        if first_hit is None:
                            first_hit = t

                snap = env.metrics.snapshot(600)
                covs.append(hits / 600.0)
                hits_list.append(hits)
                preds.append(snap.prediction_accuracy)
                delays.append(first_hit if first_hit is not None else 600)

            records.append({
                "dwell": dwell,
                "scheduler": sched.upper(),
                "step_coverage_mean": float(np.mean(covs)),
                "step_coverage_std": float(np.std(covs)),
                "hits_mean": float(np.mean(hits_list)),
                "prediction_accuracy_mean": float(np.mean(preds)),
                "intercept_delay_mean": float(np.mean(delays)),
            })

    df_res = pd.DataFrame(records)
    print("\nTEST 9 RESULTS: STEP COVERAGE BY DWELL & SCHEDULER:")
    pivot_cov = df_res.pivot(index="dwell", columns="scheduler", values="step_coverage_mean")
    print(pivot_cov.to_string())

    print("\nTEST 9 RESULTS: PREDICTION ACCURACY BY DWELL & SCHEDULER:")
    pivot_pred = df_res.pivot(index="dwell", columns="scheduler", values="prediction_accuracy_mean")
    print(pivot_pred.to_string())

    out_file = Path("scratch/validation/test_9_results.json")
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    return df_res


def run_test_10():
    print("\n" + "=" * 70)
    print("RUNNING TEST 10: MEMORY / WINDOW SENSITIVITY TEST (Dwell=3, 600 steps)")
    print("=" * 70)

    # SW-UCB window sizes: 10, 25, 50, 100, 200
    sw_windows = [10, 25, 50, 100, 200]
    sw_records = []

    for w in sw_windows:
        covs = []
        hits_list = []
        for seed in [42, 101, 202]:
            sc = create_agility_scenario(dwell=3, steps=600, seed=seed)
            env = build_environment(sc)
            env.scheduler = SlidingWindowUCBScheduler(bands_hz=env.scheduler.bands_hz, window_size=w)

            hits = 0
            for t in range(600):
                res = env.step()
                gt = res["ground_truth"]["emitters"][0]
                rx_f = res["observation"]["receiver_frequency_hz"]
                det = res["detection"]["detected"]
                if rx_f == gt["frequency_hz"] and det:
                    hits += 1

            covs.append(hits / 600.0)
            hits_list.append(hits)

        sw_records.append({
            "window_size": w,
            "step_coverage_mean": float(np.mean(covs)),
            "step_coverage_std": float(np.std(covs)),
            "hits_mean": float(np.mean(hits_list)),
        })

    df_sw = pd.DataFrame(sw_records)
    print("\nSW-UCB Window Size Sensitivity:")
    print(df_sw.to_string())

    # Discounted Thompson gamma: 0.80, 0.90, 0.95, 0.98, 0.995
    gammas = [0.80, 0.90, 0.95, 0.98, 0.995]
    dt_records = []

    for g in gammas:
        covs = []
        hits_list = []
        for seed in [42, 101, 202]:
            sc = create_agility_scenario(dwell=3, steps=600, seed=seed)
            env = build_environment(sc)
            env.scheduler = DiscountedThompsonSamplingScheduler(bands_hz=env.scheduler.bands_hz, gamma=g, seed=seed)

            hits = 0
            for t in range(600):
                res = env.step()
                gt = res["ground_truth"]["emitters"][0]
                rx_f = res["observation"]["receiver_frequency_hz"]
                det = res["detection"]["detected"]
                if rx_f == gt["frequency_hz"] and det:
                    hits += 1

            covs.append(hits / 600.0)
            hits_list.append(hits)

        dt_records.append({
            "gamma": g,
            "step_coverage_mean": float(np.mean(covs)),
            "step_coverage_std": float(np.std(covs)),
            "hits_mean": float(np.mean(hits_list)),
        })

    df_dt = pd.DataFrame(dt_records)
    print("\nDiscounted Thompson Gamma Sensitivity:")
    print(df_dt.to_string())

    out_file = Path("scratch/validation/test_10_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "sw_ucb": sw_records,
            "discounted_thompson": dt_records,
        }, f, indent=2)


if __name__ == "__main__":
    run_test_9()
    run_test_10()
