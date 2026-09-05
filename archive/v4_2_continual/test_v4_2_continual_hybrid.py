from __future__ import annotations

import copy
import inspect
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.factory import create_scheduler
from rf_environment.scheduler.hybrid.arbitrator import ArbitrationMode, HybridMetaArbitrator
from rf_environment.scheduler.hybrid.continual.checkpoint_manager import ModelCheckpointManager
from rf_environment.scheduler.hybrid.continual.config import V42ContinualConfig
from rf_environment.scheduler.hybrid.continual.online_replay import OnlineReplayBuffer
from rf_environment.scheduler.hybrid.continual.online_trainer import OnlineTrainer
from rf_environment.scheduler.hybrid.continual.regime_detector import RegimeChangeDetector, RegimeState
from rf_environment.scheduler.hybrid.continual.reward import ObservableOnlineReward
from rf_environment.scheduler.hybrid.continual.validation_gate import ValidationGate
from rf_environment.scheduler.hybrid.hybrid_v42 import V42ContinualAdaptiveHybrid
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork


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
# Test 1: Construction Direct and Factory
# =====================================================================


def test_1_construction_direct_and_factory():
    bands = [1e8, 2e8, 3e8, 4e8]

    # Direct construction
    sched_direct = V42ContinualAdaptiveHybrid(bands, seed=42)
    assert sched_direct.name == "hybrid_v42"
    assert sched_direct.category == "hybrid"
    assert isinstance(sched_direct.regime_detector, RegimeChangeDetector)
    assert isinstance(sched_direct.online_replay, OnlineReplayBuffer)
    assert isinstance(sched_direct.validation_gate, ValidationGate)
    assert isinstance(sched_direct.checkpoint_manager, ModelCheckpointManager)

    # Factory construction
    sched_factory = create_scheduler("hybrid_v42", bands, seed=42)
    assert isinstance(sched_factory, V42ContinualAdaptiveHybrid)
    assert sched_factory.num_bins == 4


# =====================================================================
# Test 2: Observable Reward Isolation and Determinism
# =====================================================================


def test_2_observable_reward_isolation_and_determinism():
    reward_calc = ObservableOnlineReward(num_bins=4)

    # 1. Detection with power
    obs_det = _make_obs(last_detection=True, last_detection_bin=1, last_detection_strength=-70.0)
    r1 = reward_calc.compute_reward(obs_det, action=1)
    assert r1 > 1.0  # 1.0 + strength bonus

    # 2. Consecutive detection on same bin yields persistence bonus
    obs_persist = _make_obs(last_detection=True, last_detection_bin=1, last_detection_strength=-70.0)
    r2 = reward_calc.compute_reward(obs_persist, action=1)
    assert r2 > r1  # includes persistence bonus

    # 3. Miss incurs scan cost
    obs_miss = _make_obs(last_detection=False)
    r_miss = reward_calc.compute_reward(obs_miss, action=2)
    assert r_miss < 0.0

    # 4. Determinism: identical observations yield identical rewards
    reward_calc_a = ObservableOnlineReward(num_bins=4)
    reward_calc_b = ObservableOnlineReward(num_bins=4)
    ra = reward_calc_a.compute_reward(obs_det, action=1)
    rb = reward_calc_b.compute_reward(obs_det, action=1)
    assert ra == rb


# =====================================================================
# Test 3: Regime Detector Stable to Suspected Transition
# =====================================================================


def test_3_regime_detector_stable_to_suspected_transition():
    cfg = V42ContinualConfig(min_shift_evidence_steps=5, regime_window=10)
    det = RegimeChangeDetector(config=cfg)
    assert det.state == RegimeState.STABLE

    # 1. Steady good detections keep state STABLE
    for step in range(10):
        obs = _make_obs(timestamp=float(step + 1), last_detection=True, last_detection_bin=1)
        det.update(
            observation=obs,
            ddqn_confidence=0.8,
            ca_confidence=0.7,
            surprise=0.05,
            online_reward=1.0,
            transition_consistency=0.9,
            predicted_bin=1,
            executed_bin=1,
        )
    assert det.state == RegimeState.STABLE

    # 2. Single isolated noisy miss does NOT immediately trigger shift
    obs_noise = _make_obs(timestamp=11.0, last_detection=False)
    det.update(
        observation=obs_noise,
        ddqn_confidence=0.1,
        ca_confidence=0.2,
        surprise=0.6,
        online_reward=-0.05,
        transition_consistency=0.1,
        predicted_bin=1,
        executed_bin=1,
    )
    assert det.state == RegimeState.STABLE

    # 3. Sustained degradation triggers SUSPECTED_SHIFT
    for step in range(12, 30):
        obs_miss = _make_obs(timestamp=float(step), last_detection=False)
        state = det.update(
            observation=obs_miss,
            ddqn_confidence=0.05,
            ca_confidence=0.2,
            surprise=0.8,
            online_reward=-0.15,
            transition_consistency=0.0,
            predicted_bin=1,
            executed_bin=1,
        )
    assert det.state == RegimeState.SUSPECTED_SHIFT


# =====================================================================
# Test 4: Regime Detector Reset
# =====================================================================


def test_4_regime_detector_reset():
    cfg = V42ContinualConfig(min_shift_evidence_steps=3)
    det = RegimeChangeDetector(config=cfg)

    # Force degradation
    for step in range(10):
        obs = _make_obs(timestamp=float(step + 1), last_detection=False)
        det.update(obs, 0.0, 0.0, 0.9, -0.2, 0.0, 1, 1)

    assert det.state == RegimeState.SUSPECTED_SHIFT

    # Reset
    det.reset()
    assert det.state == RegimeState.STABLE
    assert det.consecutive_shift_evidence == 0
    assert len(det.q_confidences) == 0


# =====================================================================
# Test 5: Mixed Online Replay Proportions and Shapes
# =====================================================================


def test_5_mixed_online_replay_proportions_and_shapes():
    cfg = V42ContinualConfig(sequence_length=4, recent_replay_ratio=0.75)
    replay = OnlineReplayBuffer(config=cfg, seed=42)

    # Populate recent buffer with 10 steps (state = 100 + step)
    for i in range(10):
        s = np.full(8, 100 + i, dtype=np.float32)
        s_next = np.full(8, 100 + i + 1, dtype=np.float32)
        replay.push(s, action=i % 4, reward=1.0, next_state=s_next, done=(i == 9))

    # Populate historical buffer with 10 steps (state = 200 + step)
    for i in range(10):
        s = np.full(8, 200 + i, dtype=np.float32)
        s_next = np.full(8, 200 + i + 1, dtype=np.float32)
        replay.push_historical(s, action=i % 4, reward=1.0, next_state=s_next, done=(i == 9))

    # Sample mixed batch of size 8
    states, actions, rewards, next_states, dones = replay.sample_mixed(batch_size=8)
    assert states.shape == (8, 4, 8)
    assert actions.shape == (8, 4)
    assert rewards.shape == (8, 4)
    assert next_states.shape == (8, 4, 8)
    assert dones.shape == (8, 4)

    # Check that both recent (>= 100) and historical (>= 200) sequences are present
    first_vals = states[:, 0, 0]
    has_recent = any(100 <= v < 200 for v in first_vals)
    has_historical = any(200 <= v < 300 for v in first_vals)
    assert has_recent and has_historical


# =====================================================================
# Test 6: Replay Episode Boundary Invariants
# =====================================================================


def test_6_replay_episode_boundary_invariants():
    cfg = V42ContinualConfig(sequence_length=5)
    replay = OnlineReplayBuffer(config=cfg, seed=123)

    # Episode 1: 6 steps
    for i in range(6):
        s = np.full(4, 10 + i, dtype=np.float32)
        replay.push(s, 0, 0.0, s, done=(i == 5))

    # Episode 2: 6 steps
    for i in range(6):
        s = np.full(4, 20 + i, dtype=np.float32)
        replay.push(s, 0, 0.0, s, done=(i == 5))

    # Sample batches and ensure no sequence crosses episode boundary
    for _ in range(5):
        states, _, _, _, _ = replay.sample_mixed(batch_size=4)
        for b in range(4):
            ep_series = states[b, :, 0] // 10
            assert np.all(ep_series == ep_series[0]), f"Cross-episode contamination: {states[b, :, 0]}"


# =====================================================================
# Test 7: Online Trainer Candidate Isolation
# =====================================================================


def test_7_online_trainer_candidate_isolation():
    cfg = V42ContinualConfig(adaptation_updates=5, sequence_length=4, adaptation_batch_size=4)
    trainer = OnlineTrainer(config=cfg)

    prod_net = LSTMQNetwork(input_dim=8, output_dim=4, hidden_dim=16, dense_dim=16, seed=42)
    target_net = prod_net.clone()

    # Save original weights of production net
    prod_w_x_orig = np.copy(prod_net.w_x)

    replay = OnlineReplayBuffer(config=cfg, seed=42)
    for i in range(12):
        s = np.random.randn(8).astype(np.float32)
        s_next = np.random.randn(8).astype(np.float32)
        replay.push(s, i % 4, 1.0, s_next, done=False)

    # Train candidate
    cand_net, stats = trainer.train_candidate(prod_net, target_net, replay)

    # Candidate weights must have changed
    assert not np.array_equal(cand_net.w_x, prod_w_x_orig)
    # Production network weights must remain 100% UNTOUCHED
    np.testing.assert_array_equal(prod_net.w_x, prod_w_x_orig)
    assert stats["gradient_steps"] == 5


# =====================================================================
# Test 8: Validation Gate Accept and Reject
# =====================================================================


def test_8_validation_gate_accept_and_reject():
    cfg = V42ContinualConfig(max_historical_loss_degradation=1.20)
    gate = ValidationGate(config=cfg)

    net_prod = LSTMQNetwork(input_dim=8, output_dim=4, hidden_dim=16, dense_dim=16, seed=42)
    net_tgt = net_prod.clone()

    # Validation batch
    val_batch = (
        np.random.randn(4, 5, 8).astype(np.float32),
        np.random.randint(0, 4, size=(4, 5)),
        np.ones((4, 5), dtype=np.float32),
        np.random.randn(4, 5, 8).astype(np.float32),
        np.zeros((4, 5), dtype=np.float32),
    )

    # Identical candidate should pass
    cand_identical = net_prod.clone()
    acc_same, telem_same = gate.evaluate(cand_identical, net_prod, net_tgt, val_batch, val_batch)
    assert acc_same is True
    assert telem_same["decision"] == "ACCEPT"

    # Degraded candidate (corrupted weights) should be rejected on historical validation
    cand_bad = net_prod.clone()
    cand_bad.b_out += 50.0
    acc_bad, telem_bad = gate.evaluate(cand_bad, net_prod, net_tgt, val_batch, val_batch)
    assert acc_bad is False
    assert telem_bad["decision"] == "REJECT"


# =====================================================================
# Test 9: Checkpoint Manager Rollback Fidelity
# =====================================================================


def test_9_checkpoint_manager_rollback_fidelity():
    net = LSTMQNetwork(input_dim=8, output_dim=4, hidden_dim=16, dense_dim=16, seed=42)
    tgt = net.clone()
    mgr = ModelCheckpointManager(net, tgt)

    orig_w_x = np.copy(net.w_x)
    orig_b_out = np.copy(net.b_out)

    # Corrupt production network
    net.w_x += 10.0
    net.b_out += 5.0
    assert not np.array_equal(net.w_x, orig_w_x)

    # Rollback
    mgr.rollback()
    np.testing.assert_array_equal(net.w_x, orig_w_x)
    np.testing.assert_array_equal(net.b_out, orig_b_out)
    assert mgr.rollbacks_performed == 1


# =====================================================================
# Test 10: Arbitrator Regime Modulation
# =====================================================================


def test_10_arbitrator_regime_modulation():
    bands = [1e8, 2e8, 3e8, 4e8]
    sched = V42ContinualAdaptiveHybrid(bands, seed=42)

    obs = _make_obs(num_bins=4, last_detection=True, last_detection_bin=1)

    # 1. In STABLE state, DDQN can guide exploit
    sched.regime_detector.state = RegimeState.STABLE
    b_stable = sched.select_bin(obs)
    assert 0 <= b_stable < 4

    # 2. In SUSPECTED_SHIFT state, CA is enforced
    sched.regime_detector.state = RegimeState.SUSPECTED_SHIFT
    b_shift = sched.select_bin(obs)
    # The mode recorded in telemetry must be CA_ADAPT
    assert sched.mode_counts[ArbitrationMode.CA_ADAPT.value] >= 1


# =====================================================================
# Test 11: Ground Truth Firewall Static and Dynamic
# =====================================================================


def test_11_ground_truth_firewall_static_and_dynamic():
    import rf_environment.scheduler.hybrid.continual.checkpoint_manager as cm_mod
    import rf_environment.scheduler.hybrid.continual.config as cfg_mod
    import rf_environment.scheduler.hybrid.continual.online_replay as or_mod
    import rf_environment.scheduler.hybrid.continual.online_trainer as ot_mod
    import rf_environment.scheduler.hybrid.continual.regime_detector as rd_mod
    import rf_environment.scheduler.hybrid.continual.reward as rw_mod
    import rf_environment.scheduler.hybrid.continual.validation_gate as vg_mod
    import rf_environment.scheduler.hybrid.hybrid_v42 as h42_mod

    modules_to_audit = [cm_mod, cfg_mod, or_mod, ot_mod, rd_mod, rw_mod, vg_mod, h42_mod]
    forbidden_terms = [
        "EmitterState",
        "realization",
        "active_emitters",
        "is_transmitting",
        "emitter.frequency",
    ]

    for mod in modules_to_audit:
        source = inspect.getsource(mod)
        for term in forbidden_terms:
            assert term not in source, f"Forbidden term '{term}' found in {mod.__name__}"

    # Dynamic firewall test
    bands = [1e8, 2e8, 3e8, 4e8]
    sched1 = V42ContinualAdaptiveHybrid(bands, seed=99)
    sched2 = V42ContinualAdaptiveHybrid(bands, seed=99)

    obs = _make_obs(num_bins=4, current_bin=1, last_detection=True, last_detection_bin=1)
    act1 = sched1.select_action(obs)
    act2 = sched2.select_action(obs)
    assert act1.frequency_bin == act2.frequency_bin


# =====================================================================
# Test 12: Determinism With Fixed Seed
# =====================================================================


def test_12_determinism_with_fixed_seed():
    bands = [1e8, 2e8, 3e8, 4e8]
    sched_a = V42ContinualAdaptiveHybrid(bands, seed=888)
    sched_b = V42ContinualAdaptiveHybrid(bands, seed=888)

    for step in range(15):
        obs = _make_obs(
            timestamp=float(step + 1),
            current_bin=step % 4,
            last_detection=(step % 3 == 0),
            last_detection_bin=(step % 4) if (step % 3 == 0) else None,
            num_bins=4,
        )

        act_a = sched_a.select_action(obs)
        act_b = sched_b.select_action(obs)
        assert act_a.frequency_bin == act_b.frequency_bin

        next_obs = _make_obs(timestamp=float(step + 2), current_bin=act_a.frequency_bin, num_bins=4)
        sched_a.observe(obs, act_a, 0.5, next_obs, done=False)
        sched_b.observe(obs, act_b, 0.5, next_obs, done=False)

    diag_a = sched_a.get_diagnostics()
    diag_b = sched_b.get_diagnostics()
    assert diag_a["total_decisions"] == diag_b["total_decisions"]
    assert diag_a["regime_state"] == diag_b["regime_state"]
    assert diag_a["mode_counts"] == diag_b["mode_counts"]
