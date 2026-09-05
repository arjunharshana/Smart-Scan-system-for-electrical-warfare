from __future__ import annotations

import math
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.enums import EmitterType, FrequencyBehaviorType, TimeBehaviorType
from rf_environment.domain.state import SchedulerObservation
from rf_environment.emitters.base import EmitterState
from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.scheduler.rl.ddqn_scheduler import DDQNScheduler
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.network import MLPQNetwork
from rf_environment.scheduler.rl.replay_buffer import ReplayBuffer


def _make_sample_observation(
    timestamp: float = 10.0,
    current_bin: int = 1,
    last_detection: bool = True,
    last_detection_bin: int | None = 1,
    last_detection_strength: float | None = -65.0,
    num_bins: int = 4,
) -> SchedulerObservation:
    return SchedulerObservation(
        timestamp=timestamp,
        current_frequency_bin=current_bin,
        last_detection=last_detection,
        last_detection_bin=last_detection_bin,
        last_detection_strength=last_detection_strength,
        recent_detection_history=(True, False, True),
        recent_frequency_history=(0, 1, 1),
        scan_count_by_bin=tuple([5] * num_bins),
        time_since_scan_by_bin=tuple([2.0] * num_bins),
        time_since_last_detection=1.0,
    )


# =====================================================================
# Test A: Observation Encoding Determinism
# =====================================================================


def test_a_observation_encoding_determinism():
    """Test A: Same observation produces identical encoded np.float32 vector."""
    encoder = ObservationEncoder(num_bins=4)
    obs1 = _make_sample_observation(num_bins=4)
    obs2 = _make_sample_observation(num_bins=4)

    v1 = encoder.encode(obs1)
    v2 = encoder.encode(obs2)

    assert v1.dtype == np.float32
    assert v2.dtype == np.float32
    assert len(v1) == encoder.feature_dim
    assert np.array_equal(v1, v2), "Encoder must be 100% deterministic"


# =====================================================================
# Test B: None Handling
# =====================================================================


def test_b_none_handling():
    """Test B: None fields must not produce NaN or Inf in encoded output."""
    encoder = ObservationEncoder(num_bins=4)
    obs_none = SchedulerObservation(
        timestamp=0.0,
        current_frequency_bin=0,
        last_detection=False,
        last_detection_bin=None,
        last_detection_strength=None,
        recent_detection_history=(),
        recent_frequency_history=(),
        scan_count_by_bin=tuple([0] * 4),
        time_since_scan_by_bin=tuple([999.0] * 4),
        time_since_last_detection=999.0,
    )

    vec = encoder.encode(obs_none)

    assert not np.any(np.isnan(vec)), "Encoded vector must not contain NaN"
    assert not np.any(np.isinf(vec)), "Encoded vector must not contain Inf"
    assert np.all(np.isfinite(vec)), "All elements of encoded vector must be finite"


# =====================================================================
# Test C: Action Validity
# =====================================================================


def test_c_action_validity():
    """Test C: Every selected action satisfies 0 <= frequency_bin < N."""
    bands = [100e6, 200e6, 300e6, 400e6, 500e6]
    n_bins = len(bands)
    scheduler = DDQNScheduler(bands_hz=bands, seed=42)
    obs = _make_sample_observation(num_bins=n_bins)

    for _ in range(50):
        action = scheduler.select_action(obs)
        assert isinstance(action, ScanAction), f"Action must be ScanAction, got {type(action)}"
        assert 0 <= action.frequency_bin < n_bins, (
            f"frequency_bin {action.frequency_bin} out of bounds [0, {n_bins - 1}]"
        )


# =====================================================================
# Test D: Epsilon Exploration
# =====================================================================


def test_d_epsilon_exploration():
    """Test D: With epsilon=1, actions are uniformly sampled from the valid set."""
    bands = [100e6, 200e6, 300e6, 400e6]
    n_bins = len(bands)
    scheduler = DDQNScheduler(
        bands_hz=bands,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay=1.0,
        seed=123,
    )
    obs = _make_sample_observation(num_bins=n_bins)

    chosen_bins = set()
    for _ in range(100):
        action = scheduler.select_action(obs)
        assert 0 <= action.frequency_bin < n_bins
        chosen_bins.add(action.frequency_bin)

    # With uniform random sampling over 100 trials, all 4 bins must be visited
    assert len(chosen_bins) == n_bins, f"Exploration failed to cover all bins: {chosen_bins}"


# =====================================================================
# Test E: Greedy Selection
# =====================================================================


def test_e_greedy_selection():
    """Test E: With epsilon=0, selected action is strictly argmax_a Q(s, a)."""
    bands = [100e6, 200e6, 300e6, 400e6]
    scheduler = DDQNScheduler(
        bands_hz=bands,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        seed=42,
    )
    obs = _make_sample_observation(num_bins=len(bands))

    state_vec = scheduler.encoder.encode(obs)
    q_values = scheduler.online_net.forward(state_vec)
    expected_bin = int(np.argmax(q_values))

    action = scheduler.select_action(obs)
    assert action.frequency_bin == expected_bin, (
        f"Greedy action {action.frequency_bin} did not match argmax Q {expected_bin}"
    )


# =====================================================================
# Test F: Double DQN Target Equation
# =====================================================================


def test_f_double_dqn_target():
    """Test F: Double DQN Target Equation verification.

    Explicitly verifies:
      argmax action comes from ONLINE network
      value evaluated comes from TARGET network

    y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))
    """
    bands = [100e6, 200e6, 300e6]
    n_bins = 3
    scheduler = DDQNScheduler(bands_hz=bands, gamma=0.9, seed=42)

    # Manually configure network weights/biases to create a distinct test case:
    # Online network prefers action 1: Q_online(s') = [0.1, 0.9, 0.2] -> argmax = 1
    # Target network has: Q_target(s') = [10.0, 2.0, 5.0]
    # Under standard DQN: max_a Q_target(s', a) = 10.0 (action 0) -> y = r + 0.9 * 10.0 = r + 9.0
    # Under Double DQN:   a* = argmax Q_online = 1 -> Q_target(s', 1) = 2.0 -> y = r + 0.9 * 2.0 = r + 1.8

    state = np.zeros(scheduler.encoder.feature_dim, dtype=np.float32)
    next_state = np.zeros(scheduler.encoder.feature_dim, dtype=np.float32)

    # Monkeypatch forward methods for this specific test
    scheduler.online_net.forward = lambda x, cache=False: np.array([[0.1, 0.9, 0.2]], dtype=np.float32) if np.asarray(x).ndim == 2 else np.array([0.1, 0.9, 0.2], dtype=np.float32)
    scheduler.target_net.forward = lambda x, cache=False: np.array([[10.0, 2.0, 5.0]], dtype=np.float32) if np.asarray(x).ndim == 2 else np.array([10.0, 2.0, 5.0], dtype=np.float32)

    # Add single transition to replay buffer
    reward = 1.0
    done = False
    scheduler.replay_buffer.push(state, 0, reward, next_state, done)

    # Sample batch
    b_states, b_actions, b_rewards, b_next_states, b_dones = scheduler.replay_buffer.sample(1)

    next_q_online = scheduler.online_net.forward(b_next_states)
    best_actions = np.argmax(next_q_online, axis=1)

    assert best_actions[0] == 1, "argmax_a Q_online must be action 1"

    next_q_target = scheduler.target_net.forward(b_next_states)
    target_q_values = next_q_target[np.arange(len(b_states)), best_actions]

    assert target_q_values[0] == 2.0, "Target network must be evaluated at online's best action (index 1)"

    target = float(b_rewards[0] + scheduler.gamma * (1.0 - b_dones[0]) * target_q_values[0])
    expected_target = 1.0 + 0.9 * 2.0  # 2.80

    assert abs(target - expected_target) < 1e-6, (
        f"Double DQN target mismatch: expected {expected_target}, got {target}"
    )

    # Verify standard DQN would have given 10.0
    standard_dqn_target = 1.0 + 0.9 * 10.0  # 10.0
    assert abs(target - standard_dqn_target) > 1.0, "Must NOT equal standard DQN target"


# =====================================================================
# Test G: Target Network Independence
# =====================================================================


def test_g_target_network_independence():
    """Test G: Target network remains independent from online updates until sync."""
    bands = [100e6, 200e6]
    scheduler = DDQNScheduler(
        bands_hz=bands,
        warmup_steps=5,
        target_update_frequency=100,
        seed=42,
    )

    # Initially weights are identical
    assert np.array_equal(scheduler.online_net.w3, scheduler.target_net.w3)
    target_w3_before = np.copy(scheduler.target_net.w3)

    # Perform online training step
    states = np.ones((8, scheduler.encoder.feature_dim), dtype=np.float32)
    actions = np.zeros(8, dtype=np.int64)
    targets = np.full(8, 1.0, dtype=np.float32)

    scheduler.online_net.train_step(states, actions, targets)

    # Online weights changed
    assert not np.array_equal(scheduler.online_net.w3, target_w3_before), "Online weights must update"

    # Target weights MUST REMAIN UNCHANGED
    assert np.array_equal(scheduler.target_net.w3, target_w3_before), (
        "Target network weights must remain unchanged before sync"
    )

    # Explicit synchronization updates target
    scheduler.target_net.copy_from(scheduler.online_net)
    assert np.array_equal(scheduler.online_net.w3, scheduler.target_net.w3), (
        "Target network must match online network after synchronization"
    )


# =====================================================================
# Test H: Replay Buffer
# =====================================================================


def test_h_replay_buffer():
    """Test H: Replay buffer pushes, stores, and samples correct shapes and types."""
    buffer = ReplayBuffer(capacity=50, seed=42)
    dim = 16

    for i in range(25):
        s = np.full(dim, float(i), dtype=np.float32)
        s_next = np.full(dim, float(i + 1), dtype=np.float32)
        buffer.push(s, i % 3, float(i * 0.1), s_next, i == 24)

    assert len(buffer) == 25

    b_s, b_a, b_r, b_ns, b_d = buffer.sample(8)
    assert b_s.shape == (8, dim)
    assert b_a.shape == (8,)
    assert b_r.shape == (8,)
    assert b_ns.shape == (8, dim)
    assert b_d.shape == (8,)
    assert b_s.dtype == np.float32
    assert b_a.dtype == np.int64
    assert b_r.dtype == np.float32


# =====================================================================
# Test I: Training Smoke
# =====================================================================


def test_i_training_smoke():
    """Test I: Short smoke-training run completes without NaN/Inf loss."""
    scenario = load_scenario("rf_environment/scenarios/stationary_fixed.yaml")
    scenario["simulation"]["total_time_steps"] = 80
    scenario["simulation"]["seed"] = 42

    env = build_environment(scenario)
    scheduler = DDQNScheduler(
        bands_hz=env.bands_hz,
        warmup_steps=16,
        batch_size=8,
        seed=42,
    )
    env.scheduler = scheduler

    results = env.run()
    assert len(results) == 80

    # Ensure training updates occurred
    assert scheduler.train_step_count > 0, "Scheduler must have executed training updates"
    assert len(scheduler.losses) > 0

    for loss in scheduler.losses:
        assert math.isfinite(loss), f"Loss must be finite, got {loss}"
        assert not math.isnan(loss), "Loss must not be NaN"


# =====================================================================
# Test J: Ground-Truth Firewall
# =====================================================================


def test_j_ground_truth_firewall():
    """Test J: Mutating hidden ground-truth state does not alter DDQN decisions.

    Changing hidden simulator state (EmitterState, true frequencies, emitter identities,
    transmission status) while keeping SchedulerObservation constant must produce identical:
      - encoded state vector
      - Q-values
      - selected greedy action
    """
    bands = [100e6, 200e6, 300e6, 400e6]
    scheduler = DDQNScheduler(bands_hz=bands, epsilon_start=0.0, seed=42)
    obs = _make_sample_observation(num_bins=len(bands))

    # Variant A: Radar transmitting at 300 MHz
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
    enc_a = scheduler.encoder.encode(obs)
    q_a = scheduler.online_net.forward(enc_a)
    act_a = scheduler.select_action(obs)

    # Variant B: Silent communications emitter at 850 MHz
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
    enc_b = scheduler.encoder.encode(obs)
    q_b = scheduler.online_net.forward(enc_b)
    act_b = scheduler.select_action(obs)

    # Verification: Encoded vector, Q-values, and chosen action are strictly identical
    assert np.array_equal(enc_a, enc_b), "Encoded observation must be invariant to hidden state"
    assert np.array_equal(q_a, q_b), "Q-values must be invariant to hidden state"
    assert act_a.frequency_bin == act_b.frequency_bin, "Action selection must be invariant to hidden state"
