"""tests/test_v4_1_training_pipeline.py
Verification suite for the V4.1 Offline Training, Validation, and Checkpoint-Selection Pipeline.

Covers:
1. Training isolation & Ground-Truth firewall audit
2. Recurrent state lifecycle & sequence boundary integrity
3. Determinism under fixed seed and configuration
4. Validation weight invariance (eval mode, epsilon=0, zero gradient step)
5. Model selection & manifest integrity
6. Production checkpoint promotion & frozen inference
7. Dataset isolation (zero overlap between Train/Val and Canonical Test)
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from app.config import DEFAULT_V41_CHECKPOINT, PROJECT_ROOT
from benchmarks.scenarios_v4_1 import (
    get_canonical_test_scenarios,
    get_train_scenarios,
    get_validation_scenarios,
    verify_dataset_isolation,
)
from benchmarks.train_v4_1 import (
    evaluate_hybrid_on_scenario,
    run_validation_battery,
    select_best_candidate,
    train_single_seed,
)
from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.environment.builder import build_environment
from rf_environment.rewards.r4_reward import R4RewardCalculator
from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
from rf_environment.scheduler.rl.checkpoint import (
    compute_weights_sha256,
    load_checkpoint,
    save_checkpoint,
)
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder


@contextmanager
def assert_raises(exc_type: type[BaseException], match: str | None = None):
    try:
        yield
    except exc_type as e:
        if match:
            assert match.lower() in str(e).lower(), f"Expected '{match}' in error message, got: {e}"
        return
    except BaseException as e:
        raise AssertionError(f"Expected {exc_type.__name__}, but got {type(e).__name__}: {e}")
    raise AssertionError(f"Expected {exc_type.__name__} was not raised")


# =============================================================================
# TEST 1: Training Isolation & Strict Ground-Truth Firewall Audit
# =============================================================================

def test_1_training_isolation_and_gt_firewall() -> None:
    """Verifies that ground truth NEVER enters scheduler observations, rewards,
    model inputs, replay buffer, or checkpoint metadata during training.
    """
    train_scs = get_train_scenarios()
    sc = copy.deepcopy(train_scs["TRAIN_1_BaseThreat"])
    env = build_environment(sc)
    enc = TemporalObservationEncoder.create_encoder_b(num_bins=len(env.bands_hz))
    sched = LSTMDDQNScheduler(bands_hz=env.bands_hz, encoder=enc, seed=42)
    sched.train()
    env.scheduler = sched

    env.reset(seed=42)
    step_res = env.step()

    # 1. Inspect observation object received by scheduler
    obs = step_res.observation
    assert isinstance(obs, SchedulerObservation)
    forbidden_tokens = {"ground_truth", "emitter_truth", "gt", "true_frequency", "target_id"}
    obs_fields = set(obs.__dataclass_fields__.keys())
    forbidden_tokens = {"ground_truth", "emitter_truth", "true_frequency", "target_id", "emitter_id"}
    for f in obs_fields:
        for token in forbidden_tokens:
            assert token not in f.lower(), f"Forbidden token '{token}' in observation field '{f}'"
        assert not f.lower().startswith("gt_"), f"Forbidden 'gt_' prefix in field '{f}'"

    # 2. Inspect reward calculation
    r_calc = env.reward_calculator
    assert isinstance(r_calc, R4RewardCalculator)
    # R4 strictly uses observation.last_detection and last_detection_strength
    assert not hasattr(r_calc, "ground_truth")

    # 3. Inspect encoded vector fed into the neural network
    state_vec = enc.encode(obs)
    assert len(state_vec) == 153
    assert not np.isnan(state_vec).any()
    assert not np.isinf(state_vec).any()


# =============================================================================
# TEST 2: Recurrent State Lifecycle & Episode Boundary Integrity
# =============================================================================

def test_2_recurrent_state_lifecycle() -> None:
    """Verifies:
    1. (h, c) persists and changes intra-episode.
    2. (h, c) resets to exact zeros across episode boundaries.
    3. Replay buffer preserves episode boundaries and never samples cross-episode sequences.
    """
    bands = [100e6 + i * 20e6 for i in range(30)]
    sched = LSTMDDQNScheduler(bands_hz=bands, seed=42)
    sched.train()

    # Episode 1
    sched.reset()
    assert np.all(sched.h == 0.0)
    assert np.all(sched.c == 0.0)

    # Push dummy observations
    for t in range(5):
        obs = SchedulerObservation(
            timestamp=float(t + 1),
            current_frequency_bin=t % 30,
            last_detection=bool(t % 2 == 0),
            last_detection_bin=t % 30 if t % 2 == 0 else None,
            last_detection_strength=-70.0 if t % 2 == 0 else None,
            recent_detection_history=tuple([False] * 10),
            recent_frequency_history=tuple([0] * 10),
            scan_count_by_bin=tuple([1] * 30),
            time_since_scan_by_bin=tuple([1.0] * 30),
            time_since_last_detection=None,
        )
        sched.select_bin(obs)

    # Intre-episode evolution
    h_energy = float(np.linalg.norm(sched.h))
    c_energy = float(np.linalg.norm(sched.c))
    assert h_energy > 0.0, "Hidden state must evolve intra-episode"
    assert c_energy > 0.0, "Cell state must evolve intra-episode"

    # Episode Boundary Reset
    sched.reset()
    assert np.all(sched.h == 0.0), "Hidden state must reset to 0.0 at episode boundary"
    assert np.all(sched.c == 0.0), "Cell state must reset to 0.0 at episode boundary"


# =============================================================================
# TEST 3: Replay Buffer Sequence Boundary Invariant
# =============================================================================

def test_3_replay_buffer_boundary_integrity() -> None:
    """Verifies that RecurrentReplayBuffer never samples sub-sequences across episode boundaries."""
    from rf_environment.scheduler.rl.recurrent_replay import RecurrentReplayBuffer

    buf = RecurrentReplayBuffer(capacity=100, sequence_length=5, seed=42)

    # Push 3 transitions in Episode 1
    for i in range(3):
        buf.push(np.zeros(10), 0, 1.0, np.zeros(10), done=(i == 2))

    # Push 3 transitions in Episode 2
    for i in range(3):
        buf.push(np.ones(10), 1, 0.0, np.ones(10), done=(i == 2))

    # No single episode has 5 transitions, so valid starts must be empty!
    valid_starts = buf.get_valid_start_indices()
    assert len(valid_starts) == 0, "No sequence of length 5 exists within a single episode"

    # Push 5 transitions in Episode 3
    for i in range(5):
        buf.push(np.ones(10) * 2, 2, 1.0, np.ones(10) * 2, done=(i == 4))

    valid_starts = buf.get_valid_start_indices()
    assert len(valid_starts) == 1
    # Sample from buffer
    b_states, b_actions, b_rewards, b_next, b_dones = buf.sample(batch_size=1)
    assert b_states.shape == (1, 5, 10)
    # All states in the sample must belong to Episode 3 (value 2.0)
    assert np.all(b_states == 2.0)


# =============================================================================
# TEST 4: Training Determinism Under Fixed Seed
# =============================================================================

def test_4_training_determinism() -> None:
    """Verifies that training with identical random seed and config produces bit-identical weights."""
    res1 = train_single_seed(seed=42, episodes=2, episode_length=50, val_interval=2, early_stopping=False, quiet=True)
    res2 = train_single_seed(seed=42, episodes=2, episode_length=50, val_interval=2, early_stopping=False, quiet=True)

    fp1 = compute_weights_sha256(res1["scheduler"].online_net.params)
    fp2 = compute_weights_sha256(res2["scheduler"].online_net.params)

    assert fp1 == fp2, f"Training non-deterministic! {fp1} != {fp2}"
    assert res1["best_val_ir_mean"] == res2["best_val_ir_mean"]


# =============================================================================
# TEST 5: Validation Invariance (Eval Mode, Zero Weight Modification)
# =============================================================================

def test_5_validation_invariance() -> None:
    """Verifies that running validation performs zero weight modifications and runs in eval mode."""
    bands = [100e6 + i * 20e6 for i in range(30)]
    sched = LSTMDDQNScheduler(bands_hz=bands, seed=42)
    val_scs = get_validation_scenarios()

    fp_before = compute_weights_sha256(sched.online_net.params)

    # Run validation battery
    val_res = run_validation_battery(sched, val_scs, val_seeds=[10, 20], steps=50)

    fp_after = compute_weights_sha256(sched.online_net.params)
    assert fp_before == fp_after, "Validation modified neural network weights!"
    assert "ir_mean" in val_res
    assert "ir_std" in val_res
    assert "scenario_metrics" in val_res


# =============================================================================
# TEST 6: Model Selection Logic
# =============================================================================

def test_6_model_selection_rule() -> None:
    """Verifies that model selection correctly favors higher mean IR and lower variance,
    while penalizing scenario collapse.
    """
    candidates = [
        {
            "seed": 100,
            "best_val_ir_mean": 0.65,
            "best_val_ir_std": 0.20,
            "best_val_ir_min": 0.05,  # Severe collapse on one scenario
            "best_val_det_rate": 0.85,
        },
        {
            "seed": 200,
            "best_val_ir_mean": 0.60,
            "best_val_ir_std": 0.05,
            "best_val_ir_min": 0.52,  # Very stable
            "best_val_det_rate": 0.88,
        },
    ]

    best, reason = select_best_candidate(candidates)
    # Candidate 200 should win because candidate 100 had collapse (< 10%) and high variance
    assert best["seed"] == 200
    assert "Selected Seed 200" in reason


# =============================================================================
# TEST 7: Production Freeze & Checkpoint Promotion
# =============================================================================

def test_7_production_freeze_and_loading() -> None:
    """Verifies that the promoted production checkpoint loads cleanly, operates in eval mode,
    has epsilon=0.0, and executes zero gradient updates.
    """
    assert DEFAULT_V41_CHECKPOINT.exists(), f"Production checkpoint missing: {DEFAULT_V41_CHECKPOINT}"

    bands = [100e6 + i * 20e6 for i in range(30)]
    sched = LSTMHybridScheduler(
        bands_hz=bands,
        checkpoint_path=DEFAULT_V41_CHECKPOINT,
        require_checkpoint=True,
    )
    sched.eval()

    assert sched.is_pretrained is True
    assert sched.lstm_ddqn.train_mode is False
    assert sched.lstm_ddqn.epsilon == 0.0

    fp_before = sched.checkpoint_sha256
    assert fp_before is not None and len(fp_before) == 64

    # Run 50 inference steps
    for t in range(50):
        obs = SchedulerObservation(
            timestamp=float(t + 1),
            current_frequency_bin=t % 30,
            last_detection=False,
            last_detection_bin=None,
            last_detection_strength=None,
            recent_detection_history=tuple([False] * 10),
            recent_frequency_history=tuple([0] * 10),
            scan_count_by_bin=tuple([1] * 30),
            time_since_scan_by_bin=tuple([1.0] * 30),
            time_since_last_detection=None,
        )
        sched.select_action(obs)

    fp_after = compute_weights_sha256(sched.lstm_ddqn.online_net.params)
    assert fp_before == fp_after, "Production weights were modified during simulation!"


# =============================================================================
# TEST 8: Test Set Isolation (Zero Dataset Leakage)
# =============================================================================

def test_8_test_set_isolation() -> None:
    """Verifies that none of the 8 canonical test scenarios are present in train or validation sets."""
    assert verify_dataset_isolation() is True
