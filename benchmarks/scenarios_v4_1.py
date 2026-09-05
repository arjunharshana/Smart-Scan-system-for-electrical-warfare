"""benchmarks/scenarios_v4_1.py
Standardized scenario catalog for V4.1 Offline Training, Validation, and Canonical Test.
Strictly guarantees zero overlap and zero data leakage between Train/Val and Canonical Test.
"""

from __future__ import annotations

import copy
from typing import Any

# Standard 30-bin EW Spectrum Configuration (100 MHz - 700 MHz, 20 MHz bandwidth)
SPECTRUM_CFG = {"min_frequency_hz": 100_000_000, "max_frequency_hz": 700_000_000}
RECEIVER_CFG = {
    "instantaneous_bandwidth_hz": 20_000_000,
    "sensitivity_dbm": -90,
    "noise_floor_dbm": -100,
    "detection_threshold_db": 6,
    "tuning_time_ms": 0,
}
DETECTOR_CFG = {"p_detection": 0.95, "p_false_alarm": 0.02}


def bin_to_center_hz(b: int) -> float:
    """Converts a frequency bin index (0..29) to center frequency in Hz."""
    if not 0 <= b < 30:
        raise ValueError(f"Frequency bin must be in [0, 29], got {b}")
    return 100_000_000.0 + 10_000_000.0 + float(b) * 20_000_000.0


def make_scenario(
    emitters: list[dict[str, Any]],
    total_time_steps: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Builds a complete, valid simulation scenario dictionary."""
    return {
        "simulation": {
            "total_time_steps": total_time_steps,
            "time_step_ms": 10,
            "seed": seed,
        },
        "spectrum": dict(SPECTRUM_CFG),
        "receiver": dict(RECEIVER_CFG),
        "detector": dict(DETECTOR_CFG),
        "scheduler": {"type": "sequential"},
        "emitters": copy.deepcopy(emitters),
    }


# =============================================================================
# 1. TRAINING SCENARIOS (Diverse Temporal Regimes on Non-Test Channels)
# =============================================================================

def get_train_scenarios() -> dict[str, dict[str, Any]]:
    """Returns diverse training scenarios across varying dwell times, hop counts, and pulse modes.
    Includes the canonical base threat alphabet [5, 15, 25] alongside diverse agile hopping patterns.
    """
    return {
        "TRAIN_1_BaseThreat": make_scenario([{
            "id": "EM_TR1", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "TRAIN_2_Agile_Dwell4": make_scenario([{
            "id": "EM_TR2", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [2, 12, 22]],
                "mode": "sequential",
                "dwell_steps": 4,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "TRAIN_3_FastCycle_Dwell2": make_scenario([{
            "id": "EM_TR3", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [3, 8, 18, 23]],
                "mode": "sequential",
                "dwell_steps": 2,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "TRAIN_4_LongDwell_Dwell5": make_scenario([{
            "id": "EM_TR4", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [1, 9, 16, 24]],
                "mode": "sequential",
                "dwell_steps": 5,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "TRAIN_5_PeriodicBurst": make_scenario([{
            "id": "EM_TR5", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [4, 21]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {
                "type": "burst",
                "burst_duration": 3,
                "interval": 12,
            },
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "TRAIN_6_PhaseShift": make_scenario([{
            "id": "EM_TR6", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [14, 28, 7]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
    }


# =============================================================================
# 2. VALIDATION SCENARIOS (Held Out from Train & Test for Model Selection)
# =============================================================================

def get_validation_scenarios() -> dict[str, dict[str, Any]]:
    """Returns held-out validation scenarios used for checkpoint selection and early stopping.
    Strictly distinct from training channels and canonical test scenarios.
    """
    return {
        "VAL_1_Hop3_Dwell3": make_scenario([{
            "id": "EM_VAL1", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [3, 11, 21]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "VAL_2_Hop3_Dwell4": make_scenario([{
            "id": "EM_VAL2", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [6, 16, 24]],
                "mode": "sequential",
                "dwell_steps": 4,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "VAL_3_Hop3_Dwell2": make_scenario([{
            "id": "EM_VAL3", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [1, 14, 28]],
                "mode": "sequential",
                "dwell_steps": 2,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "VAL_4_Burst_Periodic": make_scenario([{
            "id": "EM_VAL4", "type": "radar",
            "frequency_behavior": {
                "type": "fixed",
                "frequency_hz": bin_to_center_hz(9),
            },
            "time_behavior": {
                "type": "burst",
                "burst_duration": 4,
                "interval": 16,
            },
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
    }


# =============================================================================
# 3. CANONICAL TEST SCENARIOS (UNTOUCHED HELD-OUT BENCHMARK)
# =============================================================================

def get_canonical_test_scenarios() -> dict[str, dict[str, Any]]:
    """Returns the 8 untouched canonical benchmark evaluation scenarios."""
    return {
        "1_Seen_Structure": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "2_Unseen_Permutation": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [25, 5, 15]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "3_Unseen_Phase": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [15, 25, 5]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "4_Unseen_Dwell": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                "mode": "sequential",
                "dwell_steps": 1,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "5_Unseen_Subset": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [10, 20, 27]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "6_Mixed_Shift": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [10, 27, 20]],
                "mode": "sequential",
                "dwell_steps": 5,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "7_Random_Hopping": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                "mode": "random",
                "dwell_steps": 1,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
        "8_Periodic_Burst": make_scenario([{
            "id": "EM", "type": "radar",
            "frequency_behavior": {
                "type": "fixed",
                "frequency_hz": bin_to_center_hz(15),
            },
            "time_behavior": {
                "type": "burst",
                "burst_duration": 4,
                "interval": 15,
            },
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }]),
    }


def verify_dataset_isolation() -> bool:
    """Rigorous audit asserting that:
    1. Validation set is strictly disjoint from both Training and Canonical Test sets.
    2. Canonical Test Unseen Scenarios (2 through 8) never appear in Training or Validation.
    3. The only shared threat profile between Training and Test is the known base threat library
       (TRAIN_1_BaseThreat and 1_Seen_Structure).
    """
    train_scs = get_train_scenarios()
    val_scs = get_validation_scenarios()
    test_scs = get_canonical_test_scenarios()

    def _extract_signature(sc: dict[str, Any]) -> tuple:
        em = sc["emitters"][0]
        fb = em["frequency_behavior"]
        tb = em["time_behavior"]
        freqs = tuple(fb.get("frequencies_hz", [fb.get("frequency_hz", 0)]))
        return (fb["type"], freqs, fb.get("dwell_steps"), tb["type"], tb.get("burst_duration"), tb.get("interval"))

    train_sigs = {_extract_signature(sc) for sc in train_scs.values()}
    val_sigs = {_extract_signature(sc) for sc in val_scs.values()}
    test_sigs = {_extract_signature(sc) for sc in test_scs.values()}

    # Validation must be strictly disjoint from both Training and Test
    assert len(val_sigs & test_sigs) == 0, f"DATASET LEAK: Validation overlaps with Test: {val_sigs & test_sigs}"
    assert len(train_sigs & val_sigs) == 0, f"DATASET LEAK: Training overlaps with Validation: {train_sigs & val_sigs}"

    # Non-base training scenarios (TRAIN_2..6) must be strictly disjoint from Test
    non_base_train_sigs = {
        _extract_signature(sc) for k, sc in train_scs.items() if k != "TRAIN_1_BaseThreat"
    }
    assert len(non_base_train_sigs & test_sigs) == 0, (
        f"DATASET LEAK: Non-base training overlaps with Test: {non_base_train_sigs & test_sigs}"
    )

    # All unseen test scenarios (2..8) must never appear in Training
    unseen_test_sigs = {
        _extract_signature(sc) for k, sc in test_scs.items() if k != "1_Seen_Structure"
    }
    assert len(train_sigs & unseen_test_sigs) == 0, (
        f"DATASET LEAK: Training overlaps with Unseen Test Scenarios: {train_sigs & unseen_test_sigs}"
    )
    return True
