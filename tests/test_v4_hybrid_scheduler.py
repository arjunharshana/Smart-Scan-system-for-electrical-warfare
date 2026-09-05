from __future__ import annotations

import inspect
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.hybrid.arbitrator import ArbitrationMode, HybridMetaArbitrator
from rf_environment.scheduler.hybrid.hybrid_scheduler import HybridScheduler
from rf_environment.util.seeding import derive_seed


def _make_obs(
    timestamp: float = 1.0,
    current_bin: int = 0,
    last_detection: bool = False,
    last_detection_bin: int | None = None,
    last_detection_strength: float | None = None,
    num_bins: int = 4,
    det_hist: tuple[bool, ...] = (False, False, False),
    freq_hist: tuple[int, ...] = (0, 1, 2),
) -> SchedulerObservation:
    return SchedulerObservation(
        timestamp=timestamp,
        current_frequency_bin=current_bin,
        last_detection=last_detection,
        last_detection_bin=last_detection_bin,
        last_detection_strength=last_detection_strength,
        recent_detection_history=det_hist,
        recent_frequency_history=freq_hist,
        scan_count_by_bin=tuple([1] * num_bins),
        time_since_scan_by_bin=tuple([1.0] * num_bins),
        time_since_last_detection=None,
    )


# =====================================================================
# Test 1: End-to-End Pipeline Integration
# =====================================================================


def test_1_end_to_end_pipeline_integration():
    bands = [1e8, 2e8, 3e8, 4e8]
    scheduler = HybridScheduler(bands, seed=42)
    obs = _make_obs(num_bins=4)

    # 1. Action selection
    bin_idx = scheduler.select_bin(obs)
    assert isinstance(bin_idx, int)
    assert 0 <= bin_idx < 4

    action = scheduler.select_action(obs)
    assert isinstance(action, ScanAction)
    assert action.frequency_bin == bin_idx

    freq = scheduler.select_frequency(obs)
    assert freq == bands[bin_idx]

    # 2. Policy update
    next_obs = _make_obs(
        timestamp=2.0,
        current_bin=bin_idx,
        last_detection=True,
        last_detection_bin=bin_idx,
        last_detection_strength=-70.0,
        num_bins=4,
    )
    scheduler.update_policy(obs, action, reward=1.0, next_observation=next_obs, done=False)

    # 3. Diagnostics
    diag = scheduler.get_diagnostics()
    assert diag["total_decisions"] >= 1
    assert "avg_ddqn_weight" in diag
    assert "avg_ca_weight" in diag
    assert 0.0 <= diag["avg_ddqn_weight"] <= 1.0
    assert 0.0 <= diag["avg_ca_weight"] <= 1.0


# =====================================================================
# Test 2: Common Action Space Integrity
# =====================================================================


def test_2_common_action_space_integrity():
    for n in [2, 5, 16, 32]:
        bands = [1e8 + i * 1e7 for i in range(n)]
        sched = HybridScheduler(bands, seed=123)
        obs = _make_obs(num_bins=n, freq_hist=tuple(range(min(n, 3))))

        action = sched.select_action(obs)
        assert isinstance(action, ScanAction)
        assert hasattr(action, "frequency_bin")
        assert 0 <= action.frequency_bin < n

        # Verify ScanAction only has frequency_bin
        fields = [f for f in dir(action) if not f.startswith("_")]
        assert "frequency_bin" in fields


# =====================================================================
# Test 3: DDQN Exploitation Under Predictable Cyclic Inputs
# =====================================================================


def test_3_ddqn_exploitation_mode():
    arb = HybridMetaArbitrator(num_bins=4)
    obs = _make_obs(num_bins=4, freq_hist=(0, 1, 2, 3), det_hist=(True, True, True))

    # DDQN has strong confidence on bin 2
    q_values = np.array([0.5, 0.6, 5.0, 0.4], dtype=np.float32)
    ca_scores = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    ca_details = {
        "transition_probs": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        "activity_scores": np.zeros(4, dtype=np.float32),
    }

    chosen_bin, telemetry = arb.arbitrate(obs, q_values, ca_scores, ca_details)

    assert chosen_bin == 2
    assert telemetry["arbitration_mode"] == ArbitrationMode.DDQN_EXPLOIT.value
    assert telemetry["DDQN_weight"] >= 0.85
    assert telemetry["CA_weight"] <= 0.15
    assert telemetry["DDQN_confidence"] >= arb.ddqn_conf_threshold


# =====================================================================
# Test 4: CA Adaptation Under Surprise
# =====================================================================


def test_4_ca_adaptation_mode_under_surprise():
    arb = HybridMetaArbitrator(num_bins=4)
    obs = _make_obs(num_bins=4)

    # Simulate prediction and repeated misses on expected bin
    arb.last_predicted_bin = 2
    next_obs_miss = _make_obs(timestamp=2.0, num_bins=4, last_detection=False)
    for _ in range(6):
        arb.update_feedback(obs, selected_bin=2, next_observation=next_obs_miss)

    assert arb.surprise > arb.surprise_threshold

    # CA has adapted to bin 1, while DDQN is still stuck on bin 2
    q_values = np.array([0.5, 0.6, 5.0, 0.4], dtype=np.float32)
    ca_scores = np.array([0.1, 0.8, 0.05, 0.05], dtype=np.float32)
    ca_details = {
        "transition_probs": np.array([0.1, 0.8, 0.05, 0.05], dtype=np.float32),
        "activity_scores": np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
    }

    chosen_bin, telemetry = arb.arbitrate(obs, q_values, ca_scores, ca_details)

    assert telemetry["arbitration_mode"] == ArbitrationMode.CA_ADAPT.value
    assert telemetry["CA_weight"] >= 0.80
    assert telemetry["DDQN_weight"] <= 0.20
    assert chosen_bin == 1  # CA preference wins under surprise


# =====================================================================
# Test 5: Controlled Exploration on Low Confidence
# =====================================================================


def test_5_controlled_exploration_on_low_confidence():
    arb = HybridMetaArbitrator(num_bins=4)
    obs = _make_obs(num_bins=4, det_hist=(False, False, False, False), freq_hist=())

    # Completely flat Q-values and uniform CA scores
    q_values = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    ca_scores = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    ca_details = {
        "transition_probs": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        "activity_scores": np.zeros(4, dtype=np.float32),
        "exploration_scores": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }

    chosen_bin, telemetry = arb.arbitrate(obs, q_values, ca_scores, ca_details)

    assert telemetry["arbitration_mode"] == ArbitrationMode.EXPLORE_DISCOVERY.value
    assert telemetry["CA_weight"] >= 0.70
    assert chosen_bin == 2  # Exploration score should guide discovery


# =====================================================================
# Test 6: Ground Truth Firewall
# =====================================================================


def test_6_ground_truth_firewall_static_and_dynamic():
    import rf_environment.scheduler.hybrid.arbitrator as arb_module
    import rf_environment.scheduler.hybrid.hybrid_scheduler as sched_module

    # Static firewall check: inspect source code of both modules
    for mod in [arb_module, sched_module]:
        source = inspect.getsource(mod)
        assert "EmitterState" not in source, f"Ground-truth leak in {mod}"
        assert "realization" not in source, f"Ground-truth leak in {mod}"
        assert "active_emitters" not in source, f"Ground-truth leak in {mod}"

    # Dynamic firewall check: Scheduler decisions are purely a function of observation
    bands = [1e8, 2e8, 3e8, 4e8]
    sched1 = HybridScheduler(bands, seed=99)
    sched2 = HybridScheduler(bands, seed=99)

    obs = _make_obs(num_bins=4, current_bin=1, last_detection=True, last_detection_bin=1)

    act1 = sched1.select_action(obs)
    act2 = sched2.select_action(obs)
    assert act1.frequency_bin == act2.frequency_bin


# =====================================================================
# Test 7: Determinism With Fixed Seed
# =====================================================================


def test_7_determinism_with_fixed_seed():
    bands = [1e8, 2e8, 3e8, 4e8, 5e8]
    sched_a = HybridScheduler(bands, seed=777)
    sched_b = HybridScheduler(bands, seed=777)

    for step in range(10):
        obs = _make_obs(
            timestamp=float(step),
            current_bin=step % 5,
            last_detection=(step % 2 == 0),
            last_detection_bin=(step % 5) if (step % 2 == 0) else None,
            num_bins=5,
            det_hist=((step % 2 == 0),),
            freq_hist=((step % 5),),
        )

        act_a = sched_a.select_action(obs)
        act_b = sched_b.select_action(obs)
        assert act_a.frequency_bin == act_b.frequency_bin

        next_obs = _make_obs(
            timestamp=float(step + 1),
            current_bin=act_a.frequency_bin,
            num_bins=5,
        )
        sched_a.update_policy(obs, act_a, reward=0.5, next_observation=next_obs, done=False)
        sched_b.update_policy(obs, act_b, reward=0.5, next_observation=next_obs, done=False)

    diag_a = sched_a.get_diagnostics()
    diag_b = sched_b.get_diagnostics()
    assert diag_a == diag_b
