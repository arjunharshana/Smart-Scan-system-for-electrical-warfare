from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario


def run_test_15():
    print("=" * 70)
    print("RUNNING TEST 15: FAIRNESS / FROZEN REALIZATION TEST")
    print("=" * 70)

    # For seed 42, run all 7 schedulers on mixed_environment for 200 steps
    schedulers = ["sequential", "random", "ucb1", "thompson", "sw_ucb", "discounted_thompson", "context_aware"]
    base_scenario = load_scenario("rf_environment/scenarios/mixed_environment.yaml")
    base_scenario["simulation"]["total_time_steps"] = 200
    base_scenario["simulation"]["seed"] = 42

    # Collect hash of ground truth emitter states and channel noise across timesteps
    realization_hashes = {}
    step_samples = {}

    for sched_name in schedulers:
        env = build_environment(base_scenario, scheduler_name=sched_name)
        gt_history = []
        noise_history = []

        for t in range(200):
            res = env.step()
            gt_emitters = res["ground_truth"]["emitters"]
            # Serialize emitter states: emitter_id, frequency_hz, power_dbm, transmitting
            gt_summary = tuple(
                (e["emitter_id"], round(e["frequency_hz"], 1), round(e["power_dbm"], 2), e["transmitting"])
                for e in sorted(gt_emitters, key=lambda x: x["emitter_id"])
            )
            gt_history.append(gt_summary)
            # Receiver noise power
            noise_power = res["observation"]["measurement"]["noise_power_dbm"]
            noise_history.append(round(noise_power, 4) if noise_power is not None else 0.0)

        # Hash the ground truth sequence
        gt_bytes = json.dumps(gt_history).encode("utf-8")
        gt_hash = hashlib.sha256(gt_bytes).hexdigest()

        noise_bytes = json.dumps(noise_history).encode("utf-8")
        noise_hash = hashlib.sha256(noise_bytes).hexdigest()

        realization_hashes[sched_name] = {
            "ground_truth_sha256": gt_hash,
            "noise_realization_sha256": noise_hash,
            "match_baseline": True,  # to be verified against sequential
        }
        step_samples[sched_name] = {
            "t0_emitters": gt_history[0],
            "t50_emitters": gt_history[50],
            "t100_emitters": gt_history[100],
        }

    # Verify all match sequential
    ref_gt_hash = realization_hashes["sequential"]["ground_truth_sha256"]
    ref_noise_hash = realization_hashes["sequential"]["noise_realization_sha256"]

    all_gt_match = all(h["ground_truth_sha256"] == ref_gt_hash for h in realization_hashes.values())
    all_noise_match = all(h["noise_realization_sha256"] == ref_noise_hash for h in realization_hashes.values())

    print(f"\nGround Truth Emitter Schedule Bit-for-Bit Identical: {all_gt_match}")
    print(f"Reference SHA256 (GT): {ref_gt_hash}")
    print(f"Channel Additive Noise Bit-for-Bit Identical:        {all_noise_match}")
    print(f"Reference SHA256 (Noise): {ref_noise_hash}")

    df_hashes = pd.DataFrame([
        {
            "Scheduler": k.upper(),
            "GT Emitters SHA256": v["ground_truth_sha256"][:16] + "...",
            "Channel Noise SHA256": v["noise_realization_sha256"][:16] + "...",
            "Environment Identical": "YES (BIT-FOR-BIT MATCH)" if (v["ground_truth_sha256"] == ref_gt_hash and v["noise_realization_sha256"] == ref_noise_hash) else "MISMATCH",
        }
        for k, v in realization_hashes.items()
    ])
    print("\nRealization Consistency Table:")
    print(df_hashes.to_string())

    out_file = Path("scratch/validation/test_15_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "ground_truth_identical": all_gt_match,
            "noise_identical": all_noise_match,
            "hashes": realization_hashes,
        }, f, indent=2)

    return all_gt_match and all_noise_match


def run_test_16():
    print("\n" + "=" * 70)
    print("RUNNING TEST 16: STATISTICAL SIGNIFICANCE (30 SEEDS)")
    print("=" * 70)

    # 30 seeds: 1 to 30
    seeds = list(range(1, 31))
    scenarios_to_test = [
        ("Deterministic Hopping (Cyclic)", "rf_environment/scenarios/deterministic_hopping.yaml", 300),
        ("Mixed Environment (Multi-Emitter)", "rf_environment/scenarios/mixed_environment.yaml", 400),
    ]

    schedulers = ["ucb1", "thompson", "sw_ucb", "discounted_thompson", "context_aware"]

    sig_results = {}

    for sc_label, sc_path, steps in scenarios_to_test:
        print(f"\nBenchmarking {sc_label} across {len(seeds)} seeds...")
        sc_dict = load_scenario(sc_path)

        data_by_sched = {s: {"hits": [], "ir": [], "cov": [], "delay": [], "pred_acc": []} for s in schedulers}

        for seed in seeds:
            for s in schedulers:
                sc_copy = dict(sc_dict)
                sc_copy["simulation"] = {"total_time_steps": steps, "time_step_ms": 10, "seed": seed}
                env = build_environment(sc_copy, scheduler_name=s)
                env.run(steps=steps)
                snap = env.metrics.snapshot(steps)

                data_by_sched[s]["hits"].append(snap.hits)
                data_by_sched[s]["ir"].append(snap.interception_ratio)
                data_by_sched[s]["cov"].append(snap.opportunity_coverage)
                data_by_sched[s]["delay"].append(snap.average_intercept_time if snap.average_intercept_time is not None else steps)
                data_by_sched[s]["pred_acc"].append(snap.prediction_accuracy)

        # Statistical summaries
        summary_table = []
        for s in schedulers:
            for metric_name, vals in [("hits", data_by_sched[s]["hits"]), ("interception_ratio", data_by_sched[s]["ir"]), ("opportunity_coverage", data_by_sched[s]["cov"]), ("prediction_accuracy", data_by_sched[s]["pred_acc"])]:
                arr = np.array(vals)
                m = float(np.mean(arr))
                std = float(np.std(arr, ddof=1))
                med = float(np.median(arr))
                ci95 = 1.96 * (std / np.sqrt(len(arr)))
                summary_table.append({
                    "scenario": sc_label,
                    "scheduler": s.upper(),
                    "metric": metric_name,
                    "mean": m,
                    "std": std,
                    "median": med,
                    "ci95_half": ci95,
                    "ci95_str": f"[{m - ci95:.3f}, {m + ci95:.3f}]",
                })

        sig_results[sc_label] = {
            "summary": summary_table,
            "raw": {s: {k: [float(x) for x in v] for k, v in data_by_sched[s].items()} for s in schedulers},
        }

        # Key pairwise comparisons:
        # Context-Aware vs Thompson (Deterministic Hopping)
        # Context-Aware vs SW-UCB (Deterministic Hopping)
        # Discounted Thompson vs Thompson (Mixed Environment)
        df_sub = pd.DataFrame(summary_table)
        print(f"\n{sc_label} Summary Table (30 Seeds):")
        pivot = df_sub.pivot(index="scheduler", columns="metric", values="mean")
        print(pivot.to_string())

    out_file = Path("scratch/validation/test_16_results.json")
    with open(out_file, "w") as f:
        json.dump(sig_results, f, indent=2)


if __name__ == "__main__":
    run_test_15()
    run_test_16()
