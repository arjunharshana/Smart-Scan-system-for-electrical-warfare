from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.receiver.tuner import bands_overlap


def run_test_11():
    print("=" * 70)
    print("RUNNING TEST 11: PHASE / ALIASING TEST (Stroboscopic Blindness)")
    print("=" * 70)

    # 1. Sweep period = 20 bands, test phases 0 to 19
    # Emitter: 5 ON, 15 OFF (period = 20 steps). Fixed at 400 MHz.
    # Spectrum: 100-500 MHz, 20 bands of 20 MHz each. 400 MHz is band index 14.
    phase_records = []

    for phase in range(20):
        scenario = {
            "simulation": {"total_time_steps": 300, "time_step_ms": 10, "seed": 42},
            "spectrum": {"min_frequency_hz": 100_000_000, "max_frequency_hz": 500_000_000},
            "receiver": {
                "instantaneous_bandwidth_hz": 20_000_000,
                "sensitivity_dbm": -90,
                "noise_floor_dbm": -150,
                "detection_threshold_db": 6,
                "tuning_time_ms": 0,
            },
            "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
            "scheduler": {"type": "sequential"},
            "channel": {"noise_floor_dbm": -150, "path_loss": False},
            "emitters": [
                {
                    "id": "E_PERIODIC",
                    "type": "communication",
                    "frequency_behavior": {"type": "fixed", "frequency_hz": 400_000_000},
                    "time_behavior": {
                        "type": "periodic",
                        "on_duration": 5,
                        "off_duration": 15,
                        "phase": phase,
                    },
                    "power_dbm": -20,
                    "bandwidth_hz": 2_000_000,
                }
            ],
        }

        env = build_environment(scenario, scheduler_name="sequential")
        env.run(steps=300)
        snap = env.metrics.snapshot(300)

        phase_records.append({
            "phase": phase,
            "sweep_period": 20,
            "emitter_period": 20,
            "opportunity_coverage": snap.opportunity_coverage,
            "interception_ratio": snap.interception_ratio,
            "total_hits": snap.hits,
            "first_intercept_delay": snap.average_intercept_time,
            "outcome_verdict": "BLIND (0% Intercept)" if snap.hits == 0 else "INTERCEPTED",
        })

    df_phase = pd.DataFrame(phase_records)
    print("\nPhase Sweep (Phases 0-19 with Scan Period 20):")
    print(df_phase[["phase", "sweep_period", "emitter_period", "opportunity_coverage", "interception_ratio", "total_hits", "outcome_verdict"]].to_string())

    # 2. Break harmonic aliasing: change receiver sweep period to 19 bands and 21 bands
    print("\nNon-Harmonic Scan Period Tests (Sweep Period 19 & 21 vs Emitter Period 20):")
    non_harmonic_records = []

    for n_bands, label in [(19, "Period 19 (Prime / Coprime)"), (21, "Period 21 (Coprime)")]:
        # Construct explicit bands list with n_bands
        bws = 400_000_000 / n_bands
        bands = [100_000_000 + (i + 0.5) * bws for i in range(n_bands)]

        # Run across all 20 phases to see if blindness disappears!
        hits_across_phases = []
        cov_across_phases = []

        for phase in range(20):
            scenario = {
                "simulation": {"total_time_steps": 300, "time_step_ms": 10, "seed": 42},
                "spectrum": {"min_frequency_hz": 100_000_000, "max_frequency_hz": 500_000_000},
                "receiver": {
                    "instantaneous_bandwidth_hz": bws,
                    "sensitivity_dbm": -90,
                    "noise_floor_dbm": -150,
                    "detection_threshold_db": 6,
                    "tuning_time_ms": 0,
                },
                "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
                "scheduler": {"type": "sequential", "bands_hz": bands},
                "channel": {"noise_floor_dbm": -150, "path_loss": False},
                "emitters": [
                    {
                        "id": "E_PERIODIC",
                        "type": "communication",
                        "frequency_behavior": {"type": "fixed", "frequency_hz": 400_000_000},
                        "time_behavior": {
                            "type": "periodic",
                            "on_duration": 5,
                            "off_duration": 15,
                            "phase": phase,
                        },
                        "power_dbm": -20,
                        "bandwidth_hz": 2_000_000,
                    }
                ],
            }

            env = build_environment(scenario, scheduler_name="sequential")
            env.run(steps=300)
            snap = env.metrics.snapshot(300)
            hits_across_phases.append(snap.hits)
            cov_across_phases.append(snap.opportunity_coverage)

        blind_count = sum(1 for h in hits_across_phases if h == 0)
        non_harmonic_records.append({
            "configuration": label,
            "scan_cycle_length": n_bands,
            "emitter_period": 20,
            "mean_opportunity_coverage": float(np.mean(cov_across_phases)),
            "mean_hits": float(np.mean(hits_across_phases)),
            "blind_phases_count": blind_count,
            "blind_phases_fraction": f"{blind_count}/20 ({blind_count/20*100:.1f}%)",
        })

    df_non_harm = pd.DataFrame(non_harmonic_records)
    print(df_non_harm.to_string())

    out_file = Path("scratch/validation/test_11_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "phase_sweep_period_20": phase_records,
            "non_harmonic_comparison": non_harmonic_records,
        }, f, indent=2)

    return df_phase, df_non_harm


def run_test_12():
    print("\n" + "=" * 70)
    print("RUNNING TEST 12: RECEIVER BANDWIDTH TEST")
    print("=" * 70)

    # 1, 2, 3, 5 channels bandwidth
    # Spectrum: 100-700 MHz (total span 600 MHz).
    # Active hopping frequencies: 200, 300, 400, 500, 600 MHz (1 channel = 20 MHz)
    # Bandwidths:
    # 1 channel: 20 MHz (covers 1 channel)
    # 2 channels: 40 MHz (covers 2 adjacent channels)
    # 3 channels: 60 MHz (covers 3 channels)
    # 5 channels: 100 MHz (covers 5 channels)

    bw_configs = [
        ("1 Channel (20 MHz)", 20_000_000),
        ("2 Channels (40 MHz)", 40_000_000),
        ("3 Channels (60 MHz)", 60_000_000),
        ("5 Channels (100 MHz)", 100_000_000),
    ]

    bw_records = []

    for label, bw_hz in bw_configs:
        # Evaluate Random Hopper (dwell=1) and Deterministic Hopper (dwell=1)
        for hop_mode in ["random", "sequential"]:
            hits_list = []
            overlap_rates = []

            for seed in [1, 2, 3, 4, 5]:
                # Construct candidate bands spaced by bw_hz
                bands = []
                f = 100_000_000 + bw_hz / 2.0
                while f <= 700_000_000 - bw_hz / 2.0 + 1e-6:
                    bands.append(f)
                    f += bw_hz

                scenario = {
                    "simulation": {"total_time_steps": 1000, "time_step_ms": 10, "seed": seed},
                    "spectrum": {"min_frequency_hz": 100_000_000, "max_frequency_hz": 700_000_000},
                    "receiver": {
                        "instantaneous_bandwidth_hz": bw_hz,
                        "sensitivity_dbm": -90,
                        "noise_floor_dbm": -150,
                        "detection_threshold_db": 6,
                        "tuning_time_ms": 0,
                    },
                    "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
                    "scheduler": {"type": "random", "bands_hz": bands},
                    "channel": {"noise_floor_dbm": -150, "path_loss": False},
                    "emitters": [
                        {
                            "id": "E_HOP",
                            "type": "radar",
                            "frequency_behavior": {
                                "type": "hopping",
                                "frequencies_hz": [200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000],
                                "mode": hop_mode,
                                "dwell_steps": 1,
                            },
                            "time_behavior": {"type": "continuous"},
                            "power_dbm": -20,
                            "bandwidth_hz": 2_000_000,
                        }
                    ],
                }

                env = build_environment(scenario, scheduler_name="random")

                hits = 0
                for t in range(1000):
                    res = env.step()
                    gt = res["ground_truth"]["emitters"][0]
                    rx_cf = res["observation"]["receiver_frequency_hz"]
                    rx_bw = res["observation"]["receiver_bandwidth_hz"]
                    overlap = bands_overlap(rx_cf, rx_bw, gt["frequency_hz"], gt["bandwidth_hz"])
                    if overlap:
                        hits += 1

                hits_list.append(hits)
                overlap_rates.append(hits / 1000.0)

            bw_records.append({
                "bandwidth_label": label,
                "bandwidth_mhz": bw_hz / 1e6,
                "hopper_type": hop_mode.upper(),
                "empirical_overlap_rate_mean": float(np.mean(overlap_rates)),
                "empirical_overlap_rate_std": float(np.std(overlap_rates)),
                "total_hits_mean": float(np.mean(hits_list)),
            })

    df_bw = pd.DataFrame(bw_records)
    print("\nTEST 12 RESULTS: RECEIVER BANDWIDTH SCALING:")
    print(df_bw.to_string())

    out_file = Path("scratch/validation/test_12_results.json")
    with open(out_file, "w") as f:
        json.dump(bw_records, f, indent=2)

    return df_bw


if __name__ == "__main__":
    run_test_11()
    run_test_12()
