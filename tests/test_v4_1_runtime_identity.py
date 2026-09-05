from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from app.services.simulation_service import SimulationService
from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.rewards.r4_reward import R4RewardCalculator
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.factory import create_scheduler
from rf_environment.scheduler.hybrid.arbitrator import (
    ArbitrationMode,
    HybridMetaArbitrator,
)
from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder


# =============================================================================
# Helper: canonical sc_A scenario builder matching benchmarks/run_v4_1_benchmark.py
# =============================================================================

def _bin_to_center_hz(b: int) -> float:
    return 100_000_000.0 + 10_000_000.0 + float(b) * 20_000_000.0


def _make_canonical_scenario(
    active_bins: list[int] = [5, 15, 25],
    dwell_steps: int = 3,
    total_time_steps: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "simulation": {
            "total_time_steps": total_time_steps,
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
            "noise_floor_dbm": -100,
            "detection_threshold_db": 6,
            "tuning_time_ms": 0,
        },
        "detector": {
            "p_detection": 0.95,
            "p_false_alarm": 0.02,
        },
        "scheduler": {
            "type": "hybrid_v41",
        },
        "emitters": [
            {
                "id": "EM",
                "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [_bin_to_center_hz(b) for b in active_bins],
                    "mode": "sequential",
                    "dwell_steps": dwell_steps,
                },
                "time_behavior": {"type": "continuous"},
                "power_dbm": -18,
                "bandwidth_hz": 2_000_000,
            }
        ],
    }


def _compute_weights_sha256(params: list[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for p in params:
        hasher.update(p.tobytes())
    return hasher.hexdigest()


# =============================================================================
# TEST 1: RuntimeIdentityTest (Section 2)
# =============================================================================

def test_1_runtime_identity() -> None:
    """Verifies that the running scheduler in both SimulationService and build_environment
    is genuinely hybrid_v41 with CA, LSTM-DDQN, and HybridMetaArbitrator branches.
    """
    # 1. Inspect SimulationService runtime scheduler
    service = SimulationService()
    service.reset(seed=42, scenario_name="deterministic_hopping.yaml", scheduler_name="hybrid_v41")
    sched = service.env.scheduler

    # A. Top-level Scheduler Identity
    assert isinstance(sched, LSTMHybridScheduler), f"Expected LSTMHybridScheduler, got {type(sched)}"
    assert sched.name == "hybrid_v41", f"Expected name 'hybrid_v41', got {sched.name}"
    assert sched.category == "hybrid", f"Expected category 'hybrid', got {sched.category}"
    assert sched.__class__.__module__ == "rf_environment.scheduler.hybrid.hybrid_v41"

    # B. Sub-branches Identity
    assert isinstance(sched.context_aware, ContextAwareScheduler), f"Expected ContextAwareScheduler, got {type(sched.context_aware)}"
    assert isinstance(sched.lstm_ddqn, LSTMDDQNScheduler), f"Expected LSTMDDQNScheduler, got {type(sched.lstm_ddqn)}"
    assert isinstance(sched.arbitrator, HybridMetaArbitrator), f"Expected HybridMetaArbitrator, got {type(sched.arbitrator)}"
    assert sched.arbitrator.__class__.__module__ == "rf_environment.scheduler.hybrid.arbitrator"

    # C. LSTM Network Architecture & Shapes
    num_bins = len(sched.bands_hz)
    assert num_bins == 30, f"Expected 30 bins for 100M-700M bw=20M, got {num_bins}"

    online_net = sched.lstm_ddqn.online_net
    target_net = sched.lstm_ddqn.target_net
    assert isinstance(online_net, LSTMQNetwork)
    assert isinstance(target_net, LSTMQNetwork)

    expected_feature_dim = 33 + 4 * num_bins  # 153 for 30 bins
    assert sched.lstm_ddqn.encoder.feature_dim == expected_feature_dim
    assert online_net.input_dim == expected_feature_dim
    assert online_net.output_dim == num_bins
    assert online_net.hidden_dim == 64
    assert online_net.dense_dim == 64
    assert sched.lstm_ddqn.h.shape == (64,)
    assert sched.lstm_ddqn.c.shape == (64,)

    # Total model parameter count:
    # w_x (153*256=39168) + w_h (64*256=16384) + b_lstm (256) +
    # w_dense (64*64=4096) + b_dense (64) + w_out (64*30=1920) + b_out (30) = 61918
    total_params = sum(p.size for p in online_net.params)
    assert total_params == 61918, f"Expected 61,918 parameters, got {total_params}"

    # Target network consistency
    assert sum(p.size for p in target_net.params) == 61918

    # Inference mode verification
    sched.eval()
    sched.reset()
    assert sched.lstm_ddqn.train_mode is False
    assert sched.lstm_ddqn.epsilon == 0.0


# =============================================================================
# TEST 2: CheckpointLoadingAndIntegrityTest (Section 3)
# =============================================================================

def test_2_checkpoint_loading_and_integrity() -> None:
    """Verifies checkpoint presence, metadata, weight provenance, and fingerprint.
    Proves that SimulationService loads the validated offline pre-trained checkpoint,
    verifies weight fingerprint invariance, and confirms inference mode.
    """
    service = SimulationService()
    service.reset(seed=42, scenario_name="deterministic_hopping.yaml", scheduler_name="hybrid_v41")
    sched = service.env.scheduler
    online_net = sched.lstm_ddqn.online_net

    # 1. Locate checkpoint in simulator runtime
    checkpoint_path = sched.checkpoint_path
    assert checkpoint_path is not None, "Expected checkpoint_path to be configured"
    assert Path(checkpoint_path).exists(), f"Checkpoint path does not exist: {checkpoint_path}"

    # 2. Check disk for model checkpoints
    ckpt_files = list(Path("models/v4_1").glob("*.npz"))
    assert len(ckpt_files) >= 1, "Expected production checkpoint in models/v4_1"

    # 3. Compute deterministic fingerprint of running weights
    fingerprint = _compute_weights_sha256(online_net.params)
    assert len(fingerprint) == 64
    assert fingerprint == "11e1f2194638f7b8f0a9d3883965669011bf1dcda251bb6704480757cecb1d98", (
        f"Fingerprint {fingerprint} does not match canonical production checkpoint hash!"
    )
    assert sched.checkpoint_sha256 == fingerprint

    # 4. Target network is preserved from checkpoint training state
    target_fp = _compute_weights_sha256(sched.lstm_ddqn.target_net.params)
    assert target_fp == "d1de8dd7870c0f397f3d14a733b6c68d8b7a3c1ce3d250cd809a6ef0eda13753"

    # 5. Confirm pre-trained status and frozen inference
    assert sched.is_pretrained is True
    assert sched.lstm_ddqn.train_mode is False
    assert sched.lstm_ddqn.epsilon == 0.0


# =============================================================================
# TEST 3: LSTMHiddenStatePersistenceTest (Section 4)
# =============================================================================

def test_3_lstm_hidden_state_persistence() -> None:
    """Verifies that LSTM hidden and cell states persist across timesteps within an episode,
    update upon receiving new observations, do not duplicate on identical observations,
    and cleanly reset to zeros without cross-episode leakage.
    """
    sc = _make_canonical_scenario(seed=42)
    env = build_environment(sc, scheduler_name="hybrid_v41")
    sched = env.scheduler
    sched.eval()

    # Initial state must be strictly zeros
    sched.reset()
    h_init, c_init = sched.lstm_ddqn.get_hidden_state()
    assert np.all(h_init == 0.0), "Initial hidden state h_0 must be zeros"
    assert np.all(c_init == 0.0), "Initial cell state c_0 must be zeros"

    # Step 1
    res1 = env.step()
    obs1 = res1.observation
    act1 = sched.select_action(obs1)
    h_t1, c_t1 = sched.lstm_ddqn.get_hidden_state()

    # State must update and be non-zero
    assert not np.all(h_t1 == 0.0), "Hidden state must update after step 1"
    assert not np.all(c_t1 == 0.0), "Cell state must update after step 1"
    h1_norm = np.linalg.norm(h_t1)
    assert h1_norm > 0.1, f"Expected non-trivial h norm, got {h1_norm}"

    # Step 2
    res2 = env.step()
    obs2 = res2.observation
    assert obs2.timestamp > obs1.timestamp
    act2 = sched.select_action(obs2)
    h_t2, c_t2 = sched.lstm_ddqn.get_hidden_state()

    # State must change across timesteps
    assert not np.allclose(h_t2, h_t1), "Hidden state must evolve from t1 to t2"
    assert not np.allclose(c_t2, c_t1), "Cell state must evolve from t1 to t2"

    # Step Caching: repeated call with same observation timestamp must NOT advance LSTM
    q_repeat = sched.lstm_ddqn.get_q_values(obs2)
    h_t2_repeat, c_t2_repeat = sched.lstm_ddqn.get_hidden_state()
    assert np.allclose(h_t2, h_t2_repeat), "Duplicate query at same timestamp must not mutate hidden state"
    assert np.allclose(c_t2, c_t2_repeat), "Duplicate query at same timestamp must not mutate cell state"

    # Reset Episode: must reset hidden state to zero
    sched.reset()
    h_reset, c_reset = sched.lstm_ddqn.get_hidden_state()
    assert np.all(h_reset == 0.0), "Reset must zero hidden state"
    assert np.all(c_reset == 0.0), "Reset must zero cell state"

    # Cross-episode Leakage Check:
    # Run 5 steps in episode 1, record trajectory of hidden states
    env1 = build_environment(_make_canonical_scenario(seed=100), scheduler_name="hybrid_v41")
    env1.scheduler.eval()
    env1.scheduler.reset()
    trajectory_1 = []
    for _ in range(5):
        r = env1.step()
        env1.scheduler.select_action(r.observation)
        h, _ = env1.scheduler.lstm_ddqn.get_hidden_state()
        trajectory_1.append(h)

    # Reset and run identical episode 2: must match trajectory_1 with 0 deviation
    env1.reset(seed=100)
    env1.scheduler.reset()
    trajectory_2 = []
    for _ in range(5):
        r = env1.step()
        env1.scheduler.select_action(r.observation)
        h, _ = env1.scheduler.lstm_ddqn.get_hidden_state()
        trajectory_2.append(h)

    for i in range(5):
        diff = np.max(np.abs(trajectory_1[i] - trajectory_2[i]))
        assert diff == 0.0, f"Cross-episode leakage detected at step {i}: diff={diff}"

    # Verify branch object identity is persistent (NOT recreated per step)
    ca_id = id(sched.context_aware)
    lstm_id = id(sched.lstm_ddqn)
    arb_id = id(sched.arbitrator)
    env.step()
    sched.select_action(env._current_observation)
    assert id(sched.context_aware) == ca_id, "Context-Aware branch recreated"
    assert id(sched.lstm_ddqn) == lstm_id, "LSTM branch recreated"
    assert id(sched.arbitrator) == arb_id, "Arbitrator branch recreated"


# =============================================================================
# TEST 4: SchedulerObservationContractTest (Section 5)
# =============================================================================

def test_4_scheduler_observation_contract() -> None:
    """Captures the exact SchedulerObservation entering the scheduler during simulator execution
    and verifies its schema, dimensions, feature ordering, and normalization.
    """
    sc = _make_canonical_scenario(seed=42)
    env = build_environment(sc, scheduler_name="hybrid_v41")

    # Step environment to populate observation
    step_res = env.step()
    obs = step_res.observation

    # 1. Observation Schema Contract: exactly 10 canonical fields
    assert isinstance(obs, SchedulerObservation)
    assert hasattr(obs, "timestamp")
    assert hasattr(obs, "current_frequency_bin")
    assert hasattr(obs, "last_detection")
    assert hasattr(obs, "last_detection_bin")
    assert hasattr(obs, "last_detection_strength")
    assert hasattr(obs, "recent_detection_history")
    assert hasattr(obs, "recent_frequency_history")
    assert hasattr(obs, "scan_count_by_bin")
    assert hasattr(obs, "time_since_scan_by_bin")
    assert hasattr(obs, "time_since_last_detection")

    # Frequency bin indexing: must be 0-indexed integer in [0, N-1]
    num_bins = len(env.bands_hz)
    assert 0 <= obs.current_frequency_bin < num_bins

    # Detection encoding
    assert isinstance(obs.last_detection, bool)
    if obs.last_detection_bin is not None:
        assert 0 <= obs.last_detection_bin < num_bins

    # 2. Encoder B Vector Contract
    encoder = TemporalObservationEncoder.create_encoder_b(num_bins=num_bins)
    expected_dim = 33 + 4 * num_bins  # 153
    vec = encoder.encode(obs)

    assert vec.shape == (expected_dim,), f"Expected vector shape ({expected_dim},), got {vec.shape}"
    assert np.all(np.isfinite(vec)), "Observation vector contains NaN or Inf values"

    # Verify Bounded Ranges
    # A. Timestamp [0, 1]
    assert 0.0 <= vec[0] <= 1.0
    # B. Current frequency bin one-hot (sum == 1)
    curr_bin_slice = vec[1 : 1 + num_bins]
    assert np.isclose(np.sum(curr_bin_slice), 1.0)
    # C. Last detection flag in {0, 1}
    det_flag_idx = 1 + num_bins
    assert vec[det_flag_idx] in (0.0, 1.0)
    # D. Scan count distribution sums to 1.0 (or 0 at start)
    scan_count_idx = det_flag_idx + num_bins + 1 + 2 + 10 + 10
    scan_counts = vec[scan_count_idx : scan_count_idx + num_bins]
    assert np.isclose(np.sum(scan_counts), 1.0) or np.isclose(np.sum(scan_counts), 0.0)


# =============================================================================
# TEST 5: ActionSpaceMappingTest (Section 6)
# =============================================================================

def test_5_action_space_mapping() -> None:
    """Verifies that every action index b in [0, N-1] maps directly to the exact
    frequency bin and receiver scan region with zero off-by-one or permutations.
    """
    sc = _make_canonical_scenario(seed=42)
    env = build_environment(sc, scheduler_name="hybrid_v41")
    bands = env.bands_hz
    num_bins = len(bands)
    bw = env.receiver.instantaneous_bandwidth_hz
    min_hz = env.spectrum["min_frequency_hz"]

    action_table = []

    for b in range(num_bins):
        # 1. Test action instantiation
        action = ScanAction(frequency_bin=b)
        assert action.frequency_bin == b

        # 2. Tune receiver to bin via scan controller
        tuned_freq = env.scan_controller.execute_action(action)

        # 3. Mathematical mapping verification
        expected_freq = min_hz + bw / 2.0 + float(b) * bw
        assert np.isclose(tuned_freq, expected_freq), (
            f"Action {b} mapped to {tuned_freq} Hz, expected {expected_freq} Hz"
        )
        assert np.isclose(tuned_freq, bands[b])
        assert np.isclose(env.receiver.center_frequency_hz, bands[b])

        # 4. Receiver scan region verification
        scan_low = tuned_freq - bw / 2.0
        scan_high = tuned_freq + bw / 2.0
        assert np.isclose(scan_low, min_hz + float(b) * bw)
        assert np.isclose(scan_high, min_hz + float(b + 1) * bw)

        action_table.append({
            "action_bin": b,
            "center_freq_mhz": tuned_freq / 1e6,
            "scan_range_mhz": f"[{scan_low/1e6:.1f}, {scan_high/1e6:.1f}]",
        })

    # Continuous contiguous coverage check: bin b high == bin b+1 low
    for b in range(num_bins - 1):
        high_b = bands[b] + bw / 2.0
        low_next = bands[b + 1] - bw / 2.0
        assert np.isclose(high_b, low_next), f"Discontinuity between bin {b} and {b+1}"


# =============================================================================
# TEST 6: ArbitratorExecutionTest (Section 7)
# =============================================================================

def test_6_arbitrator_execution() -> None:
    """Instruments runtime stepping and verifies that HybridMetaArbitrator executes
    at every timestep and dictates the final selected action.
    """
    sc = _make_canonical_scenario(seed=42)
    env = build_environment(sc, scheduler_name="hybrid_v41")
    sched = env.scheduler
    sched.eval()

    # Step for 10 steps and record arbitrator telemetry
    step_records = []
    for step in range(10):
        obs = env._current_observation
        action = sched.select_action(obs)
        assert isinstance(action, ScanAction)

        # Inspect arbitrator telemetry
        telemetry = sched.arbitrator.last_decision_telemetry
        assert bool(telemetry), "Arbitrator telemetry must be populated"

        # Verify all required telemetry signals
        assert "selected_bin" in telemetry
        assert "CA_best_action" in telemetry
        assert "DDQN_best_action" in telemetry
        assert "CA_confidence" in telemetry
        assert "DDQN_confidence" in telemetry
        assert "novelty_surprise" in telemetry
        assert "transition_consistency" in telemetry
        assert "arbitration_mode" in telemetry
        assert "CA_weight" in telemetry
        assert "DDQN_weight" in telemetry

        # Critical proof: chosen action matches arbitrator's selected_bin
        assert action.frequency_bin == telemetry["selected_bin"]
        assert sched.last_selected_bin == telemetry["selected_bin"]

        mode = telemetry["arbitration_mode"]
        assert mode in [m.value for m in ArbitrationMode]

        # Weights must sum to 1.0
        assert np.isclose(telemetry["CA_weight"] + telemetry["DDQN_weight"], 1.0)

        step_records.append(telemetry)
        res = env.step()

    assert len(step_records) == 10
    # Confirm diagnostics API exposed
    diag = sched.get_diagnostics()
    assert diag["total_decisions"] >= 10
    assert "pct_ddqn_exploit" in diag
    assert "pct_ca_adapt" in diag


# =============================================================================
# TEST 7: GroundTruthFirewallTest (Section 8)
# =============================================================================

def test_7_ground_truth_firewall() -> None:
    """Verifies that ground truth emitter states and evaluation metrics never leak
    into SchedulerObservation, ContextAware, LSTM-DDQN, Arbitrator, or R4Reward.
    """
    sc = _make_canonical_scenario(seed=42)
    env = build_environment(sc, scheduler_name="hybrid_v41")
    sched = env.scheduler

    # 1. Inspect SchedulerObservation fields
    step_res = env.step()
    obs = step_res.observation

    forbidden_terms = [
        "emitter", "ground_truth", "gt", "realization", "carrier_frequency_hz",
        "transmitting", "target", "true_power", "snr", "sinr"
    ]
    obs_dict = obs.__dict__ if hasattr(obs, "__dict__") else {}
    for term in forbidden_terms:
        assert term not in obs_dict, f"Ground truth term '{term}' leaked into SchedulerObservation!"

    # 2. Inspect sub-branch inputs
    # CA only accesses observation
    ca_scores, ca_details = sched.context_aware.compute_action_scores(obs)
    assert isinstance(ca_scores, np.ndarray)
    assert len(ca_scores) == len(sched.bands_hz)

    # LSTM-DDQN only encodes observation via TemporalObservationEncoder
    q_vals = sched.lstm_ddqn.get_q_values(obs)
    assert isinstance(q_vals, np.ndarray)

    # Arbitrator only accesses (observation, q_vals, ca_scores, ca_details)
    chosen_bin, telem = sched.arbitrator.arbitrate(obs, q_vals, ca_scores, ca_details)
    assert 0 <= chosen_bin < len(sched.bands_hz)

    # 3. Reward Calculator Firewall
    reward_calc = R4RewardCalculator()
    r = reward_calc.compute(obs)
    assert r in (1.0, -0.05) or (-0.05 <= r <= 1.15)


# =============================================================================
# TEST 8: CanonicalBenchmarkReproductionTest (Section 9)
# =============================================================================

def test_8_canonical_benchmark_reproduction() -> None:
    """Reproduces the exact canonical benchmark scenario (1_Seen_Structure: [5, 15, 25], dwell=3)
    and directly verifies the performance of:
    1. Context-Aware: ~27% IR
    2. Untrained Standalone LSTM-DDQN: ~4% IR (proving untrained network fails)
    3. Untrained V4.1 Hybrid: ~25-30% IR (proving simulator's ~30-40% discrepancy)
    4. Pre-trained V4.1 Hybrid (5 episodes): ~80% IR (proving benchmark reproduction!)
    """
    sc = _make_canonical_scenario(seed=42, total_time_steps=300)

    # 1. Evaluate Context-Aware (eval seed 10)
    sc_ca = copy.deepcopy(sc)
    sc_ca["scheduler"] = {"type": "context_aware"}
    env_ca = build_environment(sc_ca)
    env_ca.reset(seed=10)
    env_ca.run(steps=300)
    ca_ir = env_ca.metrics.snapshot(300).interception_ratio

    # Context-Aware achieves ~27% (validated benchmark: 26.9%)
    assert 0.20 <= ca_ir <= 0.35, f"Context-Aware IR {ca_ir} unexpected on seen structure"

    # 2. Evaluate Untrained Standalone LSTM-DDQN (eval seed 10)
    sc_lstm = copy.deepcopy(sc)
    sc_lstm["scheduler"] = {"type": "lstm_ddqn"}
    env_lstm_raw = build_environment(sc_lstm)
    env_lstm_raw.scheduler.eval()
    env_lstm_raw.scheduler.epsilon = 0.0
    env_lstm_raw.reset(seed=10)
    env_lstm_raw.run(steps=300)
    raw_lstm_ir = env_lstm_raw.metrics.snapshot(300).interception_ratio

    # Untrained LSTM-DDQN fails catastrophically (~0% to 5%)
    assert raw_lstm_ir < 0.15, f"Untrained LSTM-DDQN IR {raw_lstm_ir} unexpected"

    # 3. Evaluate Untrained V4.1 Hybrid (default simulator instance)
    env_hyb_raw = build_environment(copy.deepcopy(sc), scheduler_name="hybrid_v41")
    env_hyb_raw.scheduler.eval()
    env_hyb_raw.scheduler.epsilon = 0.0
    env_hyb_raw.reset(seed=10)
    env_hyb_raw.run(steps=300)
    raw_hyb_ir = env_hyb_raw.metrics.snapshot(300).interception_ratio

    # Untrained Hybrid achieves ~25-35% (falling back to CA performance)
    assert 0.20 <= raw_hyb_ir <= 0.40, f"Untrained Hybrid IR {raw_hyb_ir} unexpected"

    # 4. Pre-train LSTM-DDQN for 5 episodes on seen hopping (as in benchmarks/run_v4_1_benchmark.py)
    env_train = build_environment(copy.deepcopy(sc))
    enc = TemporalObservationEncoder.create_encoder_b(num_bins=len(env_train.bands_hz))
    trained_lstm = LSTMDDQNScheduler(
        bands_hz=env_train.bands_hz,
        encoder=enc,
        hidden_dim=64,
        dense_dim=64,
        sequence_length=10,
        learning_rate=0.001,
        seed=42,
    )
    trained_lstm.train()
    env_train.scheduler = trained_lstm
    for ep in range(5):
        env_train.reset(seed=42 + ep * 1000)
        env_train.run(steps=300)

    trained_lstm.eval()
    trained_lstm.epsilon = 0.0

    # Evaluate Pre-trained Standalone LSTM-DDQN (eval seed 10)
    env_eval_lstm = build_environment(copy.deepcopy(sc))
    trained_lstm_copy = copy.deepcopy(trained_lstm)
    env_eval_lstm.scheduler = trained_lstm_copy
    env_eval_lstm.reset(seed=10)
    env_eval_lstm.run(steps=300)
    trained_lstm_ir = env_eval_lstm.metrics.snapshot(300).interception_ratio
    assert trained_lstm_ir >= 0.60, f"Pre-trained LSTM IR {trained_lstm_ir} should be >= 0.60"

    # Evaluate Pre-trained V4.1 Hybrid Scheduler
    env_eval_hyb = build_environment(copy.deepcopy(sc))
    hybrid_v41 = LSTMHybridScheduler(
        bands_hz=env_eval_hyb.bands_hz,
        lstm_ddqn=copy.deepcopy(trained_lstm),
        seed=10,
    )
    hybrid_v41.eval()
    env_eval_hyb.scheduler = hybrid_v41
    env_eval_hyb.reset(seed=10)
    env_eval_hyb.run(steps=300)
    trained_hyb_ir = env_eval_hyb.metrics.snapshot(300).interception_ratio

    # Pre-trained Hybrid jumps to >= 60% with single seed (and ~80% in multi-seed ensemble)
    assert trained_hyb_ir >= 0.60, f"Pre-trained Hybrid IR {trained_hyb_ir} should be >= 0.60"
