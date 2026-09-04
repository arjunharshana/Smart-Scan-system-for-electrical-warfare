from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.rewards.reward import RewardCalculator
from rf_environment.domain.enums import OutcomeType


class CustomRewardCalculator(RewardCalculator):
    def __init__(self, hit: float = 1.0, miss: float = 0.0, false_alarm: float = 0.0, cr: float = 0.0, latency_penalty: float = 0.0):
        self.hit = hit
        self.miss = miss
        self.false_alarm = false_alarm
        self.cr = cr
        self.latency_penalty = latency_penalty

    def compute(self, outcome: str, delay: int = 0) -> float:
        r = 0.0
        if outcome == OutcomeType.HIT.value:
            r = self.hit - self.latency_penalty * delay
        elif outcome == OutcomeType.MISS.value:
            r = self.miss
        elif outcome == OutcomeType.FALSE_ALARM.value:
            r = self.false_alarm
        elif outcome == OutcomeType.CORRECT_REJECTION.value:
            r = self.cr
        return r


def create_two_emitter_scenario(seed: int = 42, steps: int = 400) -> dict[str, Any]:
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
            "type": "thompson",
            "bands_hz": [200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000],
        },
        "channel": {"noise_floor_dbm": -150, "path_loss": False},
        "emitters": [
            {
                "id": "E1_CONT",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": 300_000_000},
                "time_behavior": {"type": "continuous"},
                "power_dbm": -20,
                "bandwidth_hz": 2_000_000,
            },
            {
                "id": "E2_BURST",
                "type": "communication",
                "frequency_behavior": {"type": "fixed", "frequency_hz": 600_000_000},
                "time_behavior": {
                    "type": "periodic",
                    "on_duration": 5,
                    "off_duration": 15,
                    "phase": 2,
                },
                "power_dbm": -22,
                "bandwidth_hz": 2_000_000,
            },
        ],
    }


def run_test_13():
    print("=" * 70)
    print("RUNNING TEST 13: MULTI-EMITTER PRIORITY TEST")
    print("=" * 70)

    # Reward Config A: strongly favors immediate detections (+1 hit, 0 miss)
    # Reward Config B: includes exploration / missed-opportunity cost (+1 hit, -1 miss)
    schedulers = ["thompson", "discounted_thompson", "context_aware"]
    reward_configs = [
        ("Config A: +1 Hit, 0 Miss", 1.0, 0.0),
        ("Config B: +1 Hit, -1 Miss", 1.0, -1.0),
    ]

    records = []

    for cfg_label, r_hit, r_miss in reward_configs:
        for sched_name in schedulers:
            e1_hits_list = []
            e2_hits_list = []
            total_rewards = []
            opp_covs = []

            for seed in [1, 2, 3, 4, 5]:
                sc = create_two_emitter_scenario(seed=seed, steps=400)
                env = build_environment(sc, scheduler_name=sched_name)
                env.reward_calculator = CustomRewardCalculator(hit=r_hit, miss=r_miss, false_alarm=-0.1, cr=0.0)

                e1_hits = 0
                e2_hits = 0
                total_rew = 0.0

                for t in range(400):
                    res = env.step()
                    gt_states = res["ground_truth"]["emitters"]
                    rx_f = res["observation"]["receiver_frequency_hz"]
                    det = res["detection"]["detected"]

                    if det:
                        for e in gt_states:
                            if e["frequency_hz"] == rx_f and e["transmitting"]:
                                if e["emitter_id"] == "E1_CONT":
                                    e1_hits += 1
                                elif e["emitter_id"] == "E2_BURST":
                                    e2_hits += 1
                    total_rew += res["outcome"]["reward"]

                snap = env.metrics.snapshot(400)
                e1_hits_list.append(e1_hits)
                e2_hits_list.append(e2_hits)
                total_rewards.append(total_rew)
                opp_covs.append(snap.opportunity_coverage)

            records.append({
                "reward_config": cfg_label,
                "scheduler": sched_name.upper(),
                "e1_continuous_hits_mean": float(np.mean(e1_hits_list)),
                "e2_burst_hits_mean": float(np.mean(e2_hits_list)),
                "e1_e2_ratio": float(np.mean(e1_hits_list) / max(1e-3, np.mean(e2_hits_list))),
                "opportunity_coverage_mean": float(np.mean(opp_covs)),
                "total_reward_mean": float(np.mean(total_rewards)),
            })

    df_res = pd.DataFrame(records)
    print("\nTEST 13 RESULTS: MULTI-EMITTER DWELL CONCENTRATION:")
    print(df_res[["reward_config", "scheduler", "e1_continuous_hits_mean", "e2_burst_hits_mean", "e1_e2_ratio", "opportunity_coverage_mean"]].to_string())

    out_file = Path("scratch/validation/test_13_results.json")
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    return df_res


def run_test_14():
    print("\n" + "=" * 70)
    print("RUNNING TEST 14: REWARD ABLATION TEST")
    print("=" * 70)

    # A: reward = +1 for hit
    # B: reward = +1 hit, -1 miss
    # C: reward = +1 hit, -0.2 miss, -0.1 false alarm
    # D: reward includes intercept-time penalty (0.01 per step delay)

    ablation_cases = [
        ("A: Hit +1, Miss 0", 1.0, 0.0, 0.0, 0.0),
        ("B: Hit +1, Miss -1", 1.0, -1.0, 0.0, 0.0),
        ("C: Hit +1, Miss -0.2, FA -0.1", 1.0, -0.2, -0.1, 0.0),
        ("D: Delay Penalty (0.02/step)", 1.0, -0.2, -0.1, 0.02),
    ]

    schedulers = ["ucb1", "thompson", "context_aware"]
    records = []

    for label, r_hit, r_miss, r_fa, r_lat in ablation_cases:
        for sched_name in schedulers:
            hits_list = []
            covs = []
            delays = []

            for seed in [1, 2, 3]:
                sc = create_two_emitter_scenario(seed=seed, steps=400)
                env = build_environment(sc, scheduler_name=sched_name)
                env.reward_calculator = CustomRewardCalculator(hit=r_hit, miss=r_miss, false_alarm=r_fa, cr=0.0, latency_penalty=r_lat)

                hits = 0
                for t in range(400):
                    res = env.step()
                    if res["detection"]["detected"]:
                        hits += 1

                snap = env.metrics.snapshot(400)
                hits_list.append(hits)
                covs.append(snap.opportunity_coverage)
                delays.append(snap.average_intercept_time if snap.average_intercept_time is not None else 400)

            records.append({
                "ablation_case": label,
                "scheduler": sched_name.upper(),
                "hits_mean": float(np.mean(hits_list)),
                "opportunity_coverage_mean": float(np.mean(covs)),
                "avg_intercept_delay_mean": float(np.mean(delays)),
            })

    df_ablation = pd.DataFrame(records)
    print("\nTEST 14 RESULTS: REWARD FUNCTION ABLATION:")
    print(df_ablation[["ablation_case", "scheduler", "hits_mean", "opportunity_coverage_mean", "avg_intercept_delay_mean"]].to_string())

    out_file = Path("scratch/validation/test_14_results.json")
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    return df_ablation


if __name__ == "__main__":
    run_test_13()
    run_test_14()
