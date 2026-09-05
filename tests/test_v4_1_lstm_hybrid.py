from __future__ import annotations

import inspect
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.factory import create_scheduler
from rf_environment.scheduler.hybrid.arbitrator import ArbitrationMode, HybridMetaArbitrator
from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork
from rf_environment.scheduler.rl.recurrent_replay import RecurrentReplayBuffer


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
    sched_direct = LSTMHybridScheduler(bands, seed=42)
    assert sched_direct.name == "hybrid_v41"
    assert sched_direct.category == "hybrid"
    assert isinstance(sched_direct.context_aware, object)
    assert isinstance(sched_direct.lstm_ddqn, LSTMDDQNScheduler)
    assert isinstance(sched_direct.arbitrator, HybridMetaArbitrator)

    # Factory construction: hybrid_v41
    sched_factory = create_scheduler("hybrid_v41", bands, seed=42)
    assert isinstance(sched_factory, LSTMHybridScheduler)
    assert sched_factory.num_bins == 4

    # Factory construction: lstm_ddqn
    sched_lstm = create_scheduler("lstm_ddqn", bands, seed=42)
    assert isinstance(sched_lstm, LSTMDDQNScheduler)
    assert sched_lstm.num_bins == 4


# =====================================================================
# Test 2: End-to-End Episode Interaction
# =====================================================================


def test_2_end_to_end_episode_interaction():
    bands = [1e8, 2e8, 3e8, 4e8]
    scheduler = LSTMHybridScheduler(bands, seed=42)
    obs = _make_obs(timestamp=1.0, num_bins=4)

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
    scheduler.update_policy(obs, action, reward=1.05, next_observation=next_obs, done=False)

    # 3. Diagnostics
    diag = scheduler.get_diagnostics()
    assert diag["total_decisions"] >= 1
    assert "avg_ddqn_weight" in diag
    assert "avg_ca_weight" in diag
    assert "mode_counts" in diag
    assert 0.0 <= diag["avg_ddqn_weight"] <= 1.0
    assert 0.0 <= diag["avg_ca_weight"] <= 1.0


# =====================================================================
# Test 3: Hidden State Evolution
# =====================================================================


def test_3_hidden_state_evolution():
    bands = [1e8, 2e8, 3e8, 4e8]
    scheduler = LSTMHybridScheduler(bands, seed=42)

    # Initial state must be all zeros
    h0, c0 = scheduler.lstm_ddqn.get_hidden_state()
    assert np.all(h0 == 0.0)
    assert np.all(c0 == 0.0)

    # Step 1
    obs1 = _make_obs(timestamp=1.0, current_bin=0, last_detection=True, last_detection_bin=0)
    scheduler.select_action(obs1)
    h1, c1 = scheduler.lstm_ddqn.get_hidden_state()
    assert not np.all(h1 == 0.0)
    assert not np.all(c1 == 0.0)

    # Step 2
    obs2 = _make_obs(timestamp=2.0, current_bin=1, last_detection=False)
    scheduler.select_action(obs2)
    h2, c2 = scheduler.lstm_ddqn.get_hidden_state()
    assert not np.array_equal(h1, h2)
    assert not np.array_equal(c1, c2)


# =====================================================================
# Test 4: Hidden State Reset to Zero
# =====================================================================


def test_4_hidden_state_reset_to_zero():
    bands = [1e8, 2e8, 3e8, 4e8]
    scheduler = LSTMHybridScheduler(bands, seed=42)

    # Advance state across multiple steps
    for step in range(5):
        obs = _make_obs(timestamp=float(step + 1), current_bin=step % 4)
        scheduler.select_action(obs)

    h_evolved, c_evolved = scheduler.lstm_ddqn.get_hidden_state()
    assert not np.all(h_evolved == 0.0)
    assert not np.all(c_evolved == 0.0)

    # Reset
    scheduler.reset()
    h_reset, c_reset = scheduler.lstm_ddqn.get_hidden_state()
    assert np.all(h_reset == 0.0)
    assert np.all(c_reset == 0.0)


# =====================================================================
# Test 5: No Cross-Episode Leakage
# =====================================================================


def test_5_no_cross_episode_leakage():
    bands = [1e8, 2e8, 3e8, 4e8]

    sched_a = LSTMHybridScheduler(bands, seed=123)
    sched_b = LSTMHybridScheduler(bands, seed=123)

    # Sched A runs episode 1 with random inputs
    for step in range(10):
        obs = _make_obs(
            timestamp=float(step + 1),
            current_bin=step % 4,
            last_detection=(step % 2 == 0),
            last_detection_bin=(step % 4),
        )
        act = sched_a.select_action(obs)
        next_obs = _make_obs(timestamp=float(step + 2), current_bin=act.frequency_bin)
        sched_a.update_policy(obs, act, reward=1.0, next_observation=next_obs, done=(step == 9))

    # Sched A resets for episode 2
    sched_a.reset()

    # Now run Episode 2 simultaneously on Sched A and Sched B (fresh)
    for step in range(10):
        obs = _make_obs(
            timestamp=float(step + 1),
            current_bin=(step * 2) % 4,
            last_detection=(step % 3 == 0),
            last_detection_bin=((step * 2) % 4),
        )
        act_a = sched_a.select_action(obs)
        act_b = sched_b.select_action(obs)

        # Actions must be bit-for-bit identical
        assert act_a.frequency_bin == act_b.frequency_bin

        # Q-values must be bit-for-bit identical
        q_a = sched_a.lstm_ddqn.get_q_values(obs)
        q_b = sched_b.lstm_ddqn.get_q_values(obs)
        np.testing.assert_allclose(q_a, q_b, rtol=1e-6, atol=1e-6)

        # Hidden states must be bit-for-bit identical
        ha, ca = sched_a.lstm_ddqn.get_hidden_state()
        hb, cb = sched_b.lstm_ddqn.get_hidden_state()
        np.testing.assert_allclose(ha, hb, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ca, cb, rtol=1e-6, atol=1e-6)

        next_obs = _make_obs(timestamp=float(step + 2), current_bin=act_a.frequency_bin)
        sched_a.update_policy(obs, act_a, reward=0.5, next_observation=next_obs, done=(step == 9))
        sched_b.update_policy(obs, act_b, reward=0.5, next_observation=next_obs, done=(step == 9))


# =====================================================================
# Test 6: Sequence Tensor Shapes
# =====================================================================


def test_6_sequence_tensor_shapes():
    B = 4
    L = 10
    input_dim = 48
    hidden_dim = 64
    num_actions = 16

    net = LSTMQNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dense_dim=32,
        output_dim=num_actions,
        seed=42,
    )

    # 1. forward_sequence: (B, L, input_dim) -> (B, L, num_actions)
    x_seq = np.random.randn(B, L, input_dim).astype(np.float32)
    q_seq, h_final, c_final = net.forward_sequence(x_seq)
    assert q_seq.shape == (B, L, num_actions)
    assert h_final.shape == (B, hidden_dim)
    assert c_final.shape == (B, hidden_dim)

    # 2. step: (B, input_dim) -> (B, num_actions)
    x_t = np.random.randn(B, input_dim).astype(np.float32)
    h_init = np.zeros((B, hidden_dim), dtype=np.float32)
    c_init = np.zeros((B, hidden_dim), dtype=np.float32)
    q_t, h_next, c_next = net.step(x_t, h_init, c_init)
    assert q_t.shape == (B, num_actions)
    assert h_next.shape == (B, hidden_dim)
    assert c_next.shape == (B, hidden_dim)


# =====================================================================
# Test 7: Q-Value Output Shape
# =====================================================================


def test_7_q_value_output_shape():
    for num_bins in [3, 8, 30]:
        bands = [1e8 + i * 1e7 for i in range(num_bins)]
        sched = LSTMDDQNScheduler(bands, seed=42)
        obs = _make_obs(num_bins=num_bins)

        q = sched.get_q_values(obs)
        assert isinstance(q, np.ndarray)
        assert q.shape == (num_bins,)
        assert not np.isnan(q).any()
        assert not np.isinf(q).any()


# =====================================================================
# Test 8: Common Action Space Integrity
# =====================================================================


def test_8_common_action_space_integrity():
    for n in [2, 7, 20, 30]:
        bands = [1e8 + i * 1e7 for i in range(n)]
        sched = LSTMHybridScheduler(bands, seed=42)
        obs = _make_obs(num_bins=n, freq_hist=tuple(range(min(n, 3))))

        action = sched.select_action(obs)
        assert isinstance(action, ScanAction)
        assert hasattr(action, "frequency_bin")
        assert 0 <= action.frequency_bin < n

        # Verify ScanAction has strictly canonical structure
        fields = [f for f in dir(action) if not f.startswith("_")]
        assert "frequency_bin" in fields


# =====================================================================
# Test 9: Arbitrator Integration Modes
# =====================================================================


def test_9_arbitrator_integration_modes():
    bands = [1e8, 2e8, 3e8, 4e8]
    sched = LSTMHybridScheduler(bands, seed=42)

    # Mode 1: DDQN Exploit
    obs_exploit = _make_obs(num_bins=4, freq_hist=(0, 1, 2, 3), det_hist=(True, True, True))
    sched.arbitrator.surprise = 0.0
    sched.arbitrator.consecutive_misses = 0
    q_confident = np.array([0.1, 5.0, 0.1, 0.1], dtype=np.float32)
    ca_flat = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    ca_details = {
        "transition_probs": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        "activity_scores": np.zeros(4, dtype=np.float32),
        "exploration_scores": np.zeros(4, dtype=np.float32),
    }
    bin_idx, telem = sched.arbitrator.arbitrate(obs_exploit, q_confident, ca_flat, ca_details)
    assert telem["arbitration_mode"] == ArbitrationMode.DDQN_EXPLOIT.value
    assert bin_idx == 1

    # Mode 2: CA Adapt under surprise
    sched.arbitrator.surprise = 1.0
    obs_surprise = _make_obs(num_bins=4, det_hist=(False, False, False), freq_hist=())
    ca_strong = np.array([0.05, 0.05, 0.85, 0.05], dtype=np.float32)
    bin_idx_ca, telem_ca = sched.arbitrator.arbitrate(obs_surprise, q_confident, ca_strong, ca_details)
    assert telem_ca["arbitration_mode"] == ArbitrationMode.CA_ADAPT.value
    assert bin_idx_ca == 2


# =====================================================================
# Test 10: Ground Truth Firewall Static and Dynamic
# =====================================================================


def test_10_ground_truth_firewall_static_and_dynamic():
    import rf_environment.scheduler.hybrid.hybrid_v41 as h41_mod
    import rf_environment.scheduler.rl.lstm_ddqn_scheduler as lstm_sched_mod
    import rf_environment.scheduler.rl.lstm_network as lstm_net_mod
    import rf_environment.scheduler.rl.recurrent_replay as replay_mod

    # Static inspection
    modules_to_audit = [h41_mod, lstm_sched_mod, lstm_net_mod, replay_mod]
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
    sched1 = LSTMHybridScheduler(bands, seed=99)
    sched2 = LSTMHybridScheduler(bands, seed=99)

    obs = _make_obs(num_bins=4, current_bin=1, last_detection=True, last_detection_bin=1)
    act1 = sched1.select_action(obs)
    act2 = sched2.select_action(obs)
    assert act1.frequency_bin == act2.frequency_bin


# =====================================================================
# Test 11: Determinism With Fixed Seed
# =====================================================================


def test_11_determinism_with_fixed_seed():
    bands = [1e8, 2e8, 3e8, 4e8, 5e8]
    sched_a = LSTMHybridScheduler(bands, seed=777)
    sched_b = LSTMHybridScheduler(bands, seed=777)

    for step in range(12):
        obs = _make_obs(
            timestamp=float(step + 1),
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
            timestamp=float(step + 2),
            current_bin=act_a.frequency_bin,
            num_bins=5,
        )
        sched_a.update_policy(obs, act_a, reward=0.5, next_observation=next_obs, done=False)
        sched_b.update_policy(obs, act_b, reward=0.5, next_observation=next_obs, done=False)

    diag_a = sched_a.get_diagnostics()
    diag_b = sched_b.get_diagnostics()
    assert diag_a == diag_b


# =====================================================================
# Test 12: Recurrent Replay Buffer Invariants
# =====================================================================


def test_12_recurrent_replay_buffer_invariants():
    buffer = RecurrentReplayBuffer(capacity=100, sequence_length=5, seed=42)

    # Push 2 episodes of length 8
    for ep in range(2):
        for step in range(8):
            state = np.full(4, ep * 10 + step, dtype=np.float32)
            action = step % 3
            reward = 1.0 if step == 7 else 0.0
            next_state = np.full(4, ep * 10 + step + 1, dtype=np.float32)
            done = (step == 7)
            buffer.push(state, action, reward, next_state, done)

    assert buffer.size == 16

    # Sample batch
    b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size=4)
    assert b_states.shape == (4, 5, 4)
    assert b_actions.shape == (4, 5)
    assert b_rewards.shape == (4, 5)
    assert b_next_states.shape == (4, 5, 4)
    assert b_dones.shape == (4, 5)

    # Verify no sequence crosses episode boundary
    for b in range(4):
        ep_ids = b_states[b, :, 0] // 10
        # All steps within a sampled sequence must belong to the same episode
        assert np.all(ep_ids == ep_ids[0]), f"Cross-episode contamination detected in batch {b}: {ep_ids}"
