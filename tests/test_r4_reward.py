from __future__ import annotations

import ast
import inspect
from dataclasses import fields

from rf_environment.domain.action import ScanAction
from rf_environment.domain.enums import EmitterType, FrequencyBehaviorType, TimeBehaviorType
from rf_environment.domain.state import SchedulerObservation
from rf_environment.domain.transition import Transition
from rf_environment.emitters.base import EmitterState
from rf_environment.rewards.r4_reward import R4RewardCalculator


def _make_observation(
    last_detection: bool,
    last_detection_strength: float | None = None,
    current_bin: int = 2,
    timestamp: float = 1.0,
) -> SchedulerObservation:
    return SchedulerObservation(
        timestamp=timestamp,
        current_frequency_bin=current_bin,
        last_detection=last_detection,
        last_detection_bin=current_bin if last_detection else None,
        last_detection_strength=last_detection_strength,
        recent_detection_history=(last_detection,),
        recent_frequency_history=(current_bin,),
        scan_count_by_bin=(0, 0, 1, 0),
        time_since_scan_by_bin=(999.0, 999.0, 0.0, 999.0),
        time_since_last_detection=0.0 if last_detection else 999.0,
    )


# =====================================================================
# Unit Tests (Tests 1 - 8 as specified in V3 R4 Specification)
# =====================================================================


def test_1_no_detection():
    """Test 1: No detection (last_detection=False, strength=None) -> R = -0.05."""
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=False, last_detection_strength=None)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - (-0.05)) < 1e-9, f"Expected -0.05, got {reward}"


def test_2_weak_detected_signal():
    """Test 2: Weak detected signal at sensitivity (last_detection=True, strength=-90 dBm) -> R = 1.00."""
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=True, last_detection_strength=-90.0)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - 1.00) < 1e-9, f"Expected 1.00, got {reward}"


def test_3_medium_detected_signal():
    """Test 3: Medium detected signal (last_detection=True, strength=-70 dBm).

    q = (-70 + 90) / 40 = 0.5
    r_strength = 0.15 * 0.5 = 0.075
    R = 1.00 + 0.00 + 0.075 = 1.075
    """
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=True, last_detection_strength=-70.0)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - 1.075) < 1e-9, f"Expected 1.075, got {reward}"


def test_4_strong_detected_signal():
    """Test 4: Strong detected signal (last_detection=True, strength=-50 dBm).

    q = (-50 + 90) / 40 = 1.0
    r_strength = 0.15 * 1.0 = 0.15
    R = 1.00 + 0.00 + 0.15 = 1.15
    """
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=True, last_detection_strength=-50.0)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - 1.15) < 1e-9, f"Expected 1.15, got {reward}"


def test_5_very_strong_signal_clipping():
    """Test 5: Very strong signal clipping (last_detection=True, strength=-30 dBm).

    q = clip((-30 + 90) / 40 = 1.5, 0.0, 1.0) = 1.0
    r_strength = 0.15 * 1.0 = 0.15
    R = 1.15
    """
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=True, last_detection_strength=-30.0)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - 1.15) < 1e-9, f"Expected 1.15, got {reward}"

    # Also test extremely high power (+10 dBm)
    obs_extreme = _make_observation(last_detection=True, last_detection_strength=10.0)
    assert abs(r4.compute(obs_extreme, action) - 1.15) < 1e-9


def test_6_below_sensitivity_no_detection_retained_historical_strength():
    """Test 6: Below sensitivity / no detection with retained historical strength.

    last_detection = False
    strength = retained historical value (-60 dBm)
    Expected: R = -0.05
    The historical strength MUST NOT be credited.
    """
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=False, last_detection_strength=-60.0)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - (-0.05)) < 1e-9, f"Expected -0.05, got {reward}"


def test_7_false_alarm():
    """Test 7: False alarm where detector triggers on noise with no signal power.

    last_detection = True
    last_detection_strength = None
    Expected: R = 1.00
    The reward calculator itself must not inspect false_alarm.
    """
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=True, last_detection_strength=None)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - 1.00) < 1e-9, f"Expected 1.00, got {reward}"


def test_8_genuine_signal():
    """Test 8: Genuine signal (last_detection=True, last_detection_strength=-50 dBm).

    Expected: R = 1.15
    """
    r4 = R4RewardCalculator()
    obs = _make_observation(last_detection=True, last_detection_strength=-50.0)
    action = ScanAction(frequency_bin=2)

    reward = r4.compute(obs, action)
    assert abs(reward - 1.15) < 1e-9, f"Expected 1.15, got {reward}"


# =====================================================================
# Ground-Truth Firewall Tests
# =====================================================================


def test_ground_truth_firewall_invariance():
    """Firewall Test: Reward is completely invariant to hidden simulator & ground truth state.

    Identical SchedulerObservation and ScanAction are presented while hidden simulator
    states (EmitterState, true frequencies, emitter identities, transmission status,
    opportunity state, channel noise/loss) are altered.
    Expected: Reward A == Reward B for all test configurations.
    """
    r4 = R4RewardCalculator()
    action = ScanAction(frequency_bin=1)

    test_cases = [
        # (det, strength)
        (False, None),
        (False, -65.0),
        (True, -90.0),
        (True, -70.0),
        (True, -50.0),
        (True, -30.0),
        (True, None),
    ]

    for det, strength in test_cases:
        obs = _make_observation(last_detection=det, last_detection_strength=strength)

        # Simulator State Variant A: Emitter active at 300 MHz, power -20 dBm
        state_a = EmitterState(
            emitter_id="RADAR_ALPHA",
            emitter_type=EmitterType.RADAR,
            timestamp=1,
            frequency_hz=300e6,
            power_dbm=-20.0,
            bandwidth_hz=2e6,
            transmitting=True,
            frequency_behavior=FrequencyBehaviorType.FIXED,
            time_behavior=TimeBehaviorType.CONTINUOUS,
        )
        reward_a = r4.compute(obs, action)

        # Simulator State Variant B: Emitter silent or hopping at 850 MHz, different ID
        state_b = EmitterState(
            emitter_id="COMM_BETA",
            emitter_type=EmitterType.COMMUNICATION,
            timestamp=1,
            frequency_hz=850e6,
            power_dbm=-80.0,
            bandwidth_hz=10e6,
            transmitting=False,
            frequency_behavior=FrequencyBehaviorType.HOPPING,
            time_behavior=TimeBehaviorType.INTERMITTENT,
        )
        reward_b = r4.compute(obs, action)

        # Simulator State Variant C: Multiple active emitters, jamming, channel variations
        state_c = EmitterState(
            emitter_id="JAMMER_GAMMA",
            emitter_type=EmitterType.RADAR,
            timestamp=1,
            frequency_hz=150e6,
            power_dbm=10.0,
            bandwidth_hz=20e6,
            transmitting=True,
            frequency_behavior=FrequencyBehaviorType.SWEEP,
            time_behavior=TimeBehaviorType.PERIODIC,
        )
        reward_c = r4.compute(obs, action)

        assert reward_a == reward_b == reward_c, (
            f"Firewall violation! Hidden simulator state leaked into reward: "
            f"A={reward_a}, B={reward_b}, C={reward_c}"
        )


def test_ground_truth_static_dependency_audit():
    """Static AST Audit: R4RewardCalculator module imports zero evaluator/ground-truth modules."""
    import rf_environment.rewards.r4_reward as r4_mod

    source = inspect.getsource(r4_mod)
    tree = ast.parse(source)

    banned_terms = {
        "metrics",
        "opportunity_tracker",
        "opportunity",
        "ground_truth",
        "emitters",
        "channel",
        "simulation_clock",
        "evaluator",
        "oracle",
        "outcome",
    }

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module.lower())
            for alias in node.names:
                imported_names.add(alias.name.lower())

    for name in imported_names:
        for banned in banned_terms:
            assert banned not in name, (
                f"Forbidden dependency '{banned}' detected in r4_reward imports: '{name}'"
            )


# =====================================================================
# Canonical Schema Integrity Tests
# =====================================================================


def test_scheduler_observation_schema_invariants():
    """Verify SchedulerObservation has exactly 10 fields and does not embed reward."""
    field_names = [f.name for f in fields(SchedulerObservation)]
    assert len(field_names) == 10, f"Expected exactly 10 fields, got {len(field_names)}: {field_names}"
    assert "reward" not in field_names, "SchedulerObservation must NOT embed reward"

    expected_fields = {
        "timestamp",
        "current_frequency_bin",
        "last_detection",
        "last_detection_bin",
        "last_detection_strength",
        "recent_detection_history",
        "recent_frequency_history",
        "scan_count_by_bin",
        "time_since_scan_by_bin",
        "time_since_last_detection",
    }
    assert set(field_names) == expected_fields


def test_scan_action_schema_invariants():
    """Verify ScanAction has exactly frequency_bin field."""
    field_names = [f.name for f in fields(ScanAction)]
    assert field_names == ["frequency_bin"]


def test_transition_schema_invariants():
    """Verify Transition has canonical fields."""
    field_names = [f.name for f in fields(Transition)]
    expected = ["observation", "action", "reward", "next_observation", "done", "info"]
    assert field_names == expected
