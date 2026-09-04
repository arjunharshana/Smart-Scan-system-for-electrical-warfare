from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.base import ScanScheduler
from rf_environment.domain.observation import Observation
from rf_environment.receiver.tuner import bands_overlap


class OracleScheduler(ScanScheduler):
    """ORACLE — EVALUATION UPPER BOUND.
    
    Allowed to see ground truth. At every timestep t, chooses next_frequency = emitter_frequency(t+1)
    so that at timestep t+1 the receiver is exactly tuned to the active emitter frequency.
    """
    def __init__(self, bands_hz: list[float], emitter_ref: Any = None):
        super().__init__(bands_hz=bands_hz)
        self.name = "oracle"
        self.category = "oracle"
        self.emitter_ref = emitter_ref
        self.current_t = 0

    def select_frequency(self, observation: Observation | None) -> float:
        if self.emitter_ref is not None:
            # Predict ground truth for next timestep t+1
            next_t = self.current_t + 1
            f = self.emitter_ref.get_frequency(next_t)
            self.last_selected = float(f)
            return float(f)
        return self.bands_hz[0]

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        if observation is not None:
            self.current_t = observation.timestamp

    def reset(self) -> None:
        super().reset()
        self.current_t = 0


def create_random_hopper_scenario(seed: int, steps: int, scheduler_name: str = "random") -> dict[str, Any]:
    """Creates the exact minimal scenario described in Test 1."""
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
            "instantaneous_bandwidth_hz": 20_000_000,  # Exactly covers 1 channel (e.g. 200 MHz +/- 10 MHz)
            "sensitivity_dbm": -90,
            "noise_floor_dbm": -150,  # No noise
            "detection_threshold_db": 6,
            "tuning_time_ms": 0,
        },
        "detector": {
            "p_detection": 1.0,  # Pd = 1 when receiver overlaps emitter
            "p_false_alarm": 0.0,  # No false alarms
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
                "id": "E_RAND_HOP",
                "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000],
                    "mode": "random",
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


def run_test_1_and_2():
    print("=" * 70)
    print("RUNNING TEST 1: ONE-STEP RANDOM HOPPER SANITY TEST (1000 steps, 10 seeds)")
    print("RUNNING TEST 2: PERFECT ORACLE UPPER BOUND")
    print("=" * 70)

    seeds = [1, 2, 3, 4, 5, 42, 101, 202, 303, 404]
    schedulers = [
        "sequential",
        "random",
        "ucb1",
        "thompson",
        "sw_ucb",
        "discounted_thompson",
        "context_aware",
        "oracle",
    ]

    records_summary = []
    sample_timesteps = None

    for sched_name in schedulers:
        overlap_rates = []
        detection_rates = []
        opp_coverages = []
        interception_ratios = []
        rewards = []

        for seed in seeds:
            sc = create_random_hopper_scenario(seed=seed, steps=1000, scheduler_name="random" if sched_name == "oracle" else sched_name)
            env = build_environment(sc, scheduler_name="random" if sched_name == "oracle" else sched_name)

            if sched_name == "oracle":
                # Plug in Oracle scheduler
                emitter = env.emitters["E_RAND_HOP"]
                oracle = OracleScheduler(
                    bands_hz=[200e6, 300e6, 400e6, 500e6, 600e6],
                    emitter_ref=emitter.frequency_behavior,
                )
                # Align initial receiver frequency to emitter's step 0 frequency
                f0 = emitter.frequency_behavior.get_frequency(0)
                env.receiver.tune(f0)
                env.receiver.tuner.center_frequency_hz = f0
                env.scheduler = oracle

            # Manual timestep-by-timestep tracking
            step_records = []
            total_transmitting_steps = 0
            total_overlapping_steps = 0
            total_detections = 0
            total_interceptions = 0

            for t in range(1000):
                res = env.step()
                gt = res["ground_truth"]["emitters"][0]
                e_freq = gt["frequency_hz"]
                e_transmitting = gt["transmitting"]
                rx_freq = res["observation"]["receiver_frequency_hz"]
                rx_bw = res["observation"]["receiver_bandwidth_hz"]
                rx_min = rx_freq - rx_bw / 2.0
                rx_max = rx_freq + rx_bw / 2.0

                overlap = bands_overlap(rx_freq, rx_bw, e_freq, gt["bandwidth_hz"])
                detected = res["detection"]["detected"]
                intercepted = bool(overlap and detected)
                opp_id = env.metrics.opportunity_tracker.completed_opportunities[0].opportunity_id if env.metrics.opportunity_tracker.completed_opportunities else (
                    list(env.metrics.opportunity_tracker.active_opportunities.values())[0].opportunity_id if env.metrics.opportunity_tracker.active_opportunities else "NONE"
                )

                if e_transmitting:
                    total_transmitting_steps += 1
                if overlap:
                    total_overlapping_steps += 1
                if detected and overlap:
                    total_detections += 1
                if intercepted:
                    total_interceptions += 1

                step_records.append({
                    "t": t,
                    "emitter_frequency": e_freq,
                    "receiver_frequency": rx_freq,
                    "receiver_window": f"[{rx_min/1e6:.1f}, {rx_max/1e6:.1f}]",
                    "overlap": overlap,
                    "detection": detected,
                    "interception": intercepted,
                    "opportunity_id": opp_id,
                })

            # Save sample timesteps for report
            if sched_name == "random" and seed == 42 and sample_timesteps is None:
                sample_timesteps = pd.DataFrame(step_records[:20])

            actual_overlap_rate = total_overlapping_steps / total_transmitting_steps if total_transmitting_steps > 0 else 0.0
            actual_detection_rate = total_detections / total_overlapping_steps if total_overlapping_steps > 0 else 0.0

            snapshot = env.metrics.snapshot(env.clock.time_step)
            opp_coverage = snapshot.opportunity_coverage
            interception_ratio = snapshot.interception_ratio

            overlap_rates.append(actual_overlap_rate)
            detection_rates.append(actual_detection_rate)
            opp_coverages.append(opp_coverage)
            interception_ratios.append(interception_ratio)
            rewards.append(snapshot.cumulative_reward)

        records_summary.append({
            "scheduler": sched_name.upper() if sched_name != "oracle" else "ORACLE — EVALUATION UPPER BOUND",
            "actual_overlap_rate_mean": float(np.mean(overlap_rates)),
            "actual_overlap_rate_std": float(np.std(overlap_rates)),
            "actual_detection_rate_mean": float(np.mean(detection_rates)),
            "actual_detection_rate_std": float(np.std(detection_rates)),
            "opportunity_coverage_mean": float(np.mean(opp_coverages)),
            "opportunity_coverage_std": float(np.std(opp_coverages)),
            "interception_ratio_mean": float(np.mean(interception_ratios)),
            "interception_ratio_std": float(np.std(interception_ratios)),
            "cumulative_reward_mean": float(np.mean(rewards)),
            "cumulative_reward_std": float(np.std(rewards)),
        })

    df_res = pd.DataFrame(records_summary)
    print("\nTEST 1 & TEST 2 RESULTS (1000 steps x 10 seeds):")
    print(df_res[["scheduler", "actual_overlap_rate_mean", "actual_detection_rate_mean", "opportunity_coverage_mean", "interception_ratio_mean", "cumulative_reward_mean"]].to_string())

    out_file = Path("scratch/validation/test_1_2_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "summary": records_summary,
            "sample_timesteps": sample_timesteps.to_dict(orient="records") if sample_timesteps is not None else [],
        }, f, indent=2)

    return df_res


def run_test_3():
    print("\n" + "=" * 70)
    print("RUNNING TEST 3: RANDOM LOWER-BOUND TEST (10,000 steps, 10 seeds)")
    print("=" * 70)

    seeds = [1, 2, 3, 4, 5, 42, 101, 202, 303, 404]
    overlap_rates = []
    interception_rates = []

    for seed in seeds:
        sc = create_random_hopper_scenario(seed=seed, steps=10000, scheduler_name="random")
        env = build_environment(sc, scheduler_name="random")

        total_transmitting = 0
        total_overlap = 0
        total_intercept = 0

        for t in range(10000):
            res = env.step()
            gt = res["ground_truth"]["emitters"][0]
            rx_cf = res["observation"]["receiver_frequency_hz"]
            rx_bw = res["observation"]["receiver_bandwidth_hz"]
            overlap = bands_overlap(rx_cf, rx_bw, gt["frequency_hz"], gt["bandwidth_hz"])
            detected = res["detection"]["detected"]

            if gt["transmitting"]:
                total_transmitting += 1
            if overlap:
                total_overlap += 1
            if overlap and detected:
                total_intercept += 1

        ov_rate = total_overlap / total_transmitting
        int_rate = total_intercept / total_transmitting
        overlap_rates.append(ov_rate)
        interception_rates.append(int_rate)
        print(f"  Seed {seed:4d}: empirical overlap = {ov_rate:.4f} ({ov_rate*100:.2f}%), interception = {int_rate:.4f} ({int_rate*100:.2f}%)")

    mean_ov = float(np.mean(overlap_rates))
    std_ov = float(np.std(overlap_rates))
    ci95_ov = 1.96 * (std_ov / np.sqrt(len(seeds)))

    mean_int = float(np.mean(interception_rates))
    std_int = float(np.std(interception_rates))
    ci95_int = 1.96 * (std_int / np.sqrt(len(seeds)))

    print("\nTEST 3 STATISTICAL COMPARISON:")
    print(f"Theoretical Expectation:        0.2000 (20.00%)")
    print(f"Empirical Overlap Rate:         {mean_ov:.4f} ± {std_ov:.4f} (95% CI: [{mean_ov - ci95_ov:.4f}, {mean_ov + ci95_ov:.4f}])")
    print(f"Empirical Interception Rate:    {mean_int:.4f} ± {std_int:.4f} (95% CI: [{mean_int - ci95_int:.4f}, {mean_int + ci95_int:.4f}])")

    out_file = Path("scratch/validation/test_3_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "theoretical": 0.2000,
            "overlap_mean": mean_ov,
            "overlap_std": std_ov,
            "overlap_ci95": ci95_ov,
            "interception_mean": mean_int,
            "interception_std": std_int,
            "interception_ci95": ci95_int,
            "per_seed_overlap": overlap_rates,
            "per_seed_interception": interception_rates,
        }, f, indent=2)


if __name__ == "__main__":
    run_test_1_and_2()
    run_test_3()
