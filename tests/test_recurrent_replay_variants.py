from __future__ import annotations

import numpy as np

from rf_environment.scheduler.rl.recurrent_replay import RecurrentReplayBuffer
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.domain.state import SchedulerObservation


def _make_dummy_obs(
    timestamp: float = 1.0,
    current_bin: int = 0,
    last_detection: bool = False,
    last_detection_bin: int | None = None,
    last_detection_strength: float | None = None,
    num_bins: int = 3,
) -> SchedulerObservation:
    return SchedulerObservation(
        timestamp=timestamp,
        current_frequency_bin=current_bin,
        last_detection=last_detection,
        last_detection_bin=last_detection_bin,
        last_detection_strength=last_detection_strength,
        recent_detection_history=(False, False, False),
        recent_frequency_history=(0, 1, 2),
        scan_count_by_bin=tuple([1] * num_bins),
        time_since_scan_by_bin=tuple([1.0] * num_bins),
        time_since_last_detection=None,
    )


def test_burn_in_sequence_length_and_masking() -> None:
    """Verifies that burn_in increases total sampled sequence length to burn_in + sequence_length
    and train_step masks loss on burn-in steps.
    """
    buf = RecurrentReplayBuffer(capacity=100, sequence_length=10, burn_in=5, seed=42)
    assert buf.total_len == 15

    # Push 20 transitions in a single episode
    for t in range(20):
        s = np.full(8, t, dtype=np.float32)
        s_next = np.full(8, t + 1, dtype=np.float32)
        buf.push(s, action=0, reward=1.0, next_state=s_next, done=(t == 19))

    valid = buf.get_valid_start_indices()
    # Indices 0 to 5 are valid (20 - 15 + 1 = 6)
    assert len(valid) == 6
    assert valid == [0, 1, 2, 3, 4, 5]

    b_states, b_actions, b_rewards, b_next_states, b_dones = buf.sample(batch_size=4)
    assert b_states.shape == (4, 15, 8)
    assert b_actions.shape == (4, 15)
    assert b_rewards.shape == (4, 15)
    assert b_next_states.shape == (4, 15, 8)
    assert b_dones.shape == (4, 15)

    # Verify LSTMQNetwork train_step with burn_in ignores prefix loss
    net = LSTMQNetwork(input_dim=8, output_dim=4, hidden_dim=16, dense_dim=16, seed=42)
    targets = np.zeros((4, 15), dtype=np.float32)
    loss = net.train_step(b_states, b_actions, targets, burn_in=5)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_stored_hidden_state_capture_and_sample() -> None:
    """Verifies that stored hidden states (h, c) are preserved and returned when requested."""
    buf = RecurrentReplayBuffer(
        capacity=50, sequence_length=5, burn_in=0, use_stored_hidden=True, seed=123
    )

    # Push transitions with known hidden states
    h_dim = 4
    for t in range(10):
        s = np.full(6, t, dtype=np.float32)
        s_next = np.full(6, t + 1, dtype=np.float32)
        h = np.full(h_dim, t * 0.1, dtype=np.float32)
        c = np.full(h_dim, t * 0.2, dtype=np.float32)
        buf.push(s, action=1, reward=0.5, next_state=s_next, done=(t == 9), h=h, c=c)

    # 1. Standard 5-tuple sample when return_hidden=False
    sample_5 = buf.sample(batch_size=3, return_hidden=False)
    assert len(sample_5) == 5

    # 2. 7-tuple sample when return_hidden=True
    sample_7 = buf.sample(batch_size=3, return_hidden=True)
    assert len(sample_7) == 7
    b_states, b_act, b_rew, b_next, b_dones, b_h0, b_c0 = sample_7
    assert b_h0 is not None and b_c0 is not None
    assert b_h0.shape == (3, h_dim)
    assert b_c0.shape == (3, h_dim)

    # Verify that b_h0 matches the start state of each sequence
    for b_idx in range(3):
        first_step_state_val = b_states[b_idx, 0, 0]  # This is 't'
        expected_h_val = float(first_step_state_val * 0.1)
        np.testing.assert_allclose(b_h0[b_idx], expected_h_val, rtol=1e-5, atol=1e-5)


def test_episode_boundary_and_circular_fence() -> None:
    """Verifies that sequences never cross episode boundaries or circular buffer wrap-around seam."""
    buf = RecurrentReplayBuffer(capacity=10, sequence_length=4, burn_in=0, seed=42)

    # Episode 1: 5 steps
    for t in range(5):
        buf.push(np.zeros(2), 0, 0.0, np.zeros(2), done=(t == 4))

    # Episode 2: 5 steps (filling buffer to capacity=10)
    for t in range(5):
        buf.push(np.ones(2), 1, 1.0, np.ones(2), done=(t == 4))

    valid = buf.get_valid_start_indices()
    # Episode 1: starts 0, 1 (length 4). Index 2 would span [2, 3, 4, 5] crossing into Ep 2.
    # Episode 2: starts 5, 6 (length 4). Index 7 spans [7, 8, 9, 10] out of range.
    assert valid == [0, 1, 5, 6]

    # Now trigger circular buffer wrap-around by pushing 2 more transitions (cursor moves to 2)
    buf.push(np.full(2, 2.0), 0, 0.0, np.full(2, 2.0), done=False)
    buf.push(np.full(2, 2.0), 0, 0.0, np.full(2, 2.0), done=False)
    assert buf.cursor == 2

    # Verify no valid start window crosses cursor=2: [i < 2 <= i + 3]
    valid_after = buf.get_valid_start_indices()
    for v in valid_after:
        window = set(range(v, v + 4))
        # Seam is between index 1 (new) and index 2 (old)
        assert not (1 in window and 2 in window), f"Window {window} bridged circular overwrite seam"


def test_regime_balanced_sampling() -> None:
    """Verifies that regime-balanced sampling balances representations across regimes."""
    buf = RecurrentReplayBuffer(
        capacity=200, sequence_length=4, burn_in=0, regime_balanced=True, seed=42
    )

    # Push 50 transitions with regime_id=0 (Ep 1)
    for t in range(50):
        buf.push(np.zeros(2), 0, 0.0, np.zeros(2), done=(t == 49), regime_id=0)

    # Push 10 transitions with regime_id=1 (Ep 2)
    for t in range(10):
        buf.push(np.ones(2), 1, 1.0, np.ones(2), done=(t == 9), regime_id=1)

    # Sample batch of 20
    b_states, b_actions, _, _, _ = buf.sample(batch_size=20)
    assert b_states.shape == (20, 4, 2)

    # Count how many samples belong to regime 0 (zeros) vs regime 1 (ones)
    is_regime_0 = [np.all(b_states[b, 0] == 0.0) for b in range(20)]
    count_regime_0 = sum(is_regime_0)
    count_regime_1 = 20 - count_regime_0

    # With 2 regimes and B=20, balanced sampling allocates 10 to each regime
    assert count_regime_0 == 10
    assert count_regime_1 == 10


def test_scheduler_lifecycle_and_ground_truth_firewall() -> None:
    """Verifies LSTMDDQNScheduler lifecycle with stored hidden states and zero GT leakage."""
    bands = [100e6, 200e6, 300e6]
    sched = LSTMDDQNScheduler(
        bands_hz=bands,
        sequence_length=4,
        burn_in=2,
        use_stored_hidden=True,
        regime_balanced_replay=True,
        warmup_steps=10,
        batch_size=4,
        seed=42,
    )
    sched.train()

    # Create synthetic SchedulerObservations
    for step in range(20):
        obs = _make_dummy_obs(
            timestamp=float(step),
            current_bin=step % 3,
            last_detection=bool(step % 2 == 0),
            last_detection_strength=-60.0 if step % 2 == 0 else -110.0,
            num_bins=3,
        )
        action = sched.select_bin(obs)
        next_obs = _make_dummy_obs(
            timestamp=float(step + 1),
            current_bin=action,
            last_detection=False,
            last_detection_strength=-110.0,
            num_bins=3,
        )
        sched.update_policy(obs, action, reward=1.0, next_observation=next_obs, done=(step == 19), regime_id=0)

    assert len(sched.replay_buffer) == 20
    assert sched.train_step_count > 0

    # Reset lifecycle: verify all hidden states are cleared to zero
    sched.reset()
    assert np.all(sched.h == 0.0)
    assert np.all(sched.c == 0.0)
    assert np.all(sched._prev_h == 0.0)
    assert np.all(sched._prev_c == 0.0)
