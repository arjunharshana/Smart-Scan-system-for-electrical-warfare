from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.receiver.tuner import bands_overlap
from rf_environment.scheduler.base import ScanScheduler
from rf_environment.domain.observation import Observation


class FixedChannelScheduler(ScanScheduler):
    """Freezes receiver tuned permanently to target frequency (e.g. 300 MHz)."""
    def __init__(self, target_hz: float, bands_hz: list[float]):
        super().__init__(bands_hz=bands_hz)
        self.target_hz = target_hz
        self.name = "fixed_frozen"
        self.category = "frozen"

    def select_frequency(self, observation: Observation | None) -> float:
        self.last_selected = self.target_hz
        return self.target_hz

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        pass


def run_test_7():
    print("=" * 70)
    print("RUNNING TEST 7: DETECTOR VS SCHEDULER SEPARATION")
    print("=" * 70)

    # 300 MHz fixed emitter, continuous, Pd=100%, Pfa=0%, no noise
    schedulers = ["sequential", "random", "ucb1", "thompson", "sw_ucb", "discounted_thompson", "context_aware"]
    results = []

    for name in schedulers:
        scenario = {
            "simulation": {"total_time_steps": 500, "time_step_ms": 10, "seed": 42},
            "spectrum": {"min_frequency_hz": 100_000_000, "max_frequency_hz": 500_000_000},
            "receiver": {
                "instantaneous_bandwidth_hz": 20_000_000,
                "sensitivity_dbm": -90,
                "noise_floor_dbm": -150,
                "detection_threshold_db": 6,
                "tuning_time_ms": 0,
            },
            "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
            "scheduler": {"type": name},
            "channel": {"noise_floor_dbm": -150, "path_loss": False},
            "emitters": [
                {
                    "id": "E_FIXED",
                    "type": "radar",
                    "frequency_behavior": {"type": "fixed", "frequency_hz": 300_000_000},
                    "time_behavior": {"type": "continuous"},
                    "power_dbm": -20,
                    "bandwidth_hz": 2_000_000,
                }
            ],
        }

        env = build_environment(scenario, scheduler_name=name)
        env.run(steps=500)

        snapshot = env.metrics.snapshot(500)
        cov = snapshot.opportunity_coverage
        dgc = snapshot.detection_given_coverage
        ir = snapshot.interception_ratio

        # Identity check: IR = Coverage * DGC
        product = cov * dgc
        diff = abs(ir - product)

        # Step-level coverage (fraction of 500 steps spent on 300 MHz)
        step_cov = snapshot.step_coverage_ratio

        results.append({
            "scheduler": name.upper(),
            "opportunity_coverage": cov,
            "detection_given_coverage": dgc,
            "interception_ratio": ir,
            "coverage_x_dgc_product": product,
            "mathematical_identity_holds": diff < 1e-5,
            "step_coverage_ratio": step_cov,
            "Pd": snapshot.probability_of_detection,
            "Pfa": snapshot.probability_of_false_alarm,
            "total_hits": snapshot.hits,
        })

    df_res = pd.DataFrame(results)
    print("\nTEST 7 RESULTS (Perfect Detector Pd=1.0, Pfa=0.0):")
    print(df_res[["scheduler", "opportunity_coverage", "detection_given_coverage", "interception_ratio", "mathematical_identity_holds", "step_coverage_ratio", "total_hits"]].to_string())

    out_file = Path("scratch/validation/test_7_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    return df_res


def run_test_8():
    print("\n" + "=" * 70)
    print("RUNNING TEST 8: DETECTOR-ONLY STRESS TEST (Scheduler Frozen on 300 MHz)")
    print("=" * 70)

    # 4 SNR cases:
    # Case A: High SNR (Signal -20 dBm, Noise -100 dBm, Sensitivity -90 dBm, Threshold 6 dB, SNR = +80 dB)
    # Case B: Medium SNR (Signal -80 dBm, Noise -100 dBm, SNR = +20 dB)
    # Case C: Low SNR (Signal -93 dBm, Noise -100 dBm, SNR = +7 dB, near threshold 6 dB)
    # Case D: Below Sensitivity (Signal -98 dBm, Sensitivity -90 dBm, SNR = +2 dB < threshold 6 dB)

    cases = [
        ("Case A: High SNR (+80 dB)", -20.0, -100.0, -90.0, 6.0),
        ("Case B: Medium SNR (+20 dB)", -80.0, -100.0, -90.0, 6.0),
        ("Case C: Low SNR (+7 dB)", -93.0, -100.0, -90.0, 6.0),
        ("Case D: Below Sensitivity (-98 dBm < -90 dBm)", -98.0, -100.0, -90.0, 6.0),
    ]

    results = []

    for label, sig_pwr, noise_pwr, sens, thresh in cases:
        scenario = {
            "simulation": {"total_time_steps": 500, "time_step_ms": 10, "seed": 42},
            "spectrum": {"min_frequency_hz": 100_000_000, "max_frequency_hz": 500_000_000},
            "receiver": {
                "instantaneous_bandwidth_hz": 20_000_000,
                "sensitivity_dbm": sens,
                "noise_floor_dbm": noise_pwr,
                "detection_threshold_db": thresh,
                "tuning_time_ms": 0,
            },
            "detector": {
                "detection_threshold_db": thresh,
                "p_detection": 0.95,
                "p_false_alarm": 0.05,
            },
            "scheduler": {"type": "sequential"},
            "channel": {"noise_floor_dbm": noise_pwr, "path_loss": False},
            "emitters": [
                {
                    "id": "E_FIXED",
                    "type": "radar",
                    "frequency_behavior": {"type": "fixed", "frequency_hz": 300_000_000},
                    "time_behavior": {"type": "continuous"},
                    "power_dbm": sig_pwr,
                    "bandwidth_hz": 2_000_000,
                }
            ],
        }

        env = build_environment(scenario)
        # Freeze scheduler permanently on 300 MHz
        env.scheduler = FixedChannelScheduler(target_hz=300_000_000, bands_hz=env.scheduler.bands_hz)
        env.receiver.tune(300_000_000)
        env.receiver.tuner.center_frequency_hz = 300_000_000

        env.run(steps=500)
        snap = env.metrics.snapshot(500)

        results.append({
            "case": label,
            "signal_power_dbm": sig_pwr,
            "noise_floor_dbm": noise_pwr,
            "sensitivity_dbm": sens,
            "threshold_db": thresh,
            "expected_snr_db": sig_pwr - noise_pwr,
            "empirical_Pd": snap.probability_of_detection,
            "empirical_Pfa": snap.probability_of_false_alarm,
            "miss_rate": snap.miss_rate,
            "total_scans": snap.total_scans,
            "hits": snap.hits,
            "misses": snap.misses,
            "in_band_trials": snap.hits + snap.misses,
        })

    df_cases = pd.DataFrame(results)
    print("\nTEST 8 RESULTS: DETECTOR-ONLY STRESS TEST:")
    print(df_cases[["case", "expected_snr_db", "empirical_Pd", "empirical_Pfa", "miss_rate", "hits", "misses"]].to_string())

    out_file = Path("scratch/validation/test_8_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    return df_cases


if __name__ == "__main__":
    run_test_7()
    run_test_8()
