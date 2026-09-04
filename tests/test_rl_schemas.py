from __future__ import annotations

import dataclasses
from dataclasses import fields, FrozenInstanceError
import sys

from rf_environment.domain.action import ScanAction, validate_action
from rf_environment.domain.state import SchedulerObservation
from rf_environment.domain.transition import Episode, StepResult, Transition
from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.rl_scheduler import RLScheduler
import rf_environment.domain as domain_mod
import rf_environment.domain.metrics as metrics_mod
import rf_environment.domain.state as state_mod


class AssertRaises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} was not raised")
        return issubclass(exc_type, self.exc_type)


def _create_test_env():
    scenario = {
        "simulation": {"total_time_steps": 20, "time_step_ms": 10, "seed": 42},
        "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 500e6},
        "receiver": {"instantaneous_bandwidth_hz": 20e6, "tuning_time_ms": 0},
        "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
        "scheduler": {"type": "sequential"},
        "emitters": [
            {
                "id": "E1",
                "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [200e6, 300e6, 400e6],
                    "mode": "sequential",
                    "dwell_steps": 1,
                },
                "time_behavior": {"type": "continuous"},
                "power_dbm": -20,
                "bandwidth_hz": 1e6,
            }
        ],
    }
    return build_environment(scenario)


# ======================================================================
# Section 18 Mandatory Architecture & Schema Tests
# ======================================================================

def test_scan_action_has_only_frequency_bin():
    """Test that canonical ScanAction contains strictly one field: frequency_bin: int."""
    action_fields = fields(ScanAction)
    assert len(action_fields) == 1, f"ScanAction must have exactly 1 field, got {len(action_fields)}"
    assert action_fields[0].name == "frequency_bin"
    assert action_fields[0].type in (int, "int")

    action = ScanAction(frequency_bin=3)
    assert action.frequency_bin == 3

    # Immutable / frozen
    with AssertRaises((FrozenInstanceError, AttributeError)):
        action.frequency_bin = 5

    # Reject non-canonical extra fields
    with AssertRaises(TypeError):
        ScanAction(frequency_bin=0, dwell_steps=1)
    with AssertRaises(TypeError):
        ScanAction(frequency_bin=0, frequency_hz=200e6)


def test_scan_action_rejects_invalid_frequency_bin():
    """Test that invalid frequency_bin values and non-ScanAction types are rejected."""
    env = _create_test_env()
    num_bins = len(env.bands_hz)

    # validate_action helper
    validate_action(ScanAction(frequency_bin=0), num_bins)
    validate_action(ScanAction(frequency_bin=num_bins - 1), num_bins)

    with AssertRaises(ValueError):
        validate_action(ScanAction(frequency_bin=-1), num_bins)
    with AssertRaises(ValueError):
        validate_action(ScanAction(frequency_bin=num_bins), num_bins)
    with AssertRaises(TypeError):
        validate_action("not_an_action", num_bins)

    # env.step validation
    env.reset(seed=42)
    with AssertRaises(ValueError):
        env.step(ScanAction(frequency_bin=-1))
    with AssertRaises(ValueError):
        env.step(ScanAction(frequency_bin=num_bins + 10))


def test_scheduler_observation_has_exact_canonical_fields():
    """Test that SchedulerObservation contains exactly the 10 canonical fields in order."""
    canonical_fields = [
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
    ]
    actual_fields = [f.name for f in fields(SchedulerObservation)]
    assert actual_fields == canonical_fields, (
        f"SchedulerObservation fields mismatch: expected {canonical_fields}, got {actual_fields}"
    )
    assert len(actual_fields) == 10

    # Verify non-canonical fields do not exist
    forbidden_fields = [
        "current_time",
        "current_frequency",
        "current_bandwidth",
        "detector",
        "detected_frequency",
        "signal_strength",
        "snr",
        "false_alarm",
        "recently_scanned_bins",
        "scan_counts",
        "time_since_current_frequency_scan",
        "evaluator_metrics",
        "emitter_state",
        "opportunity_labels",
        "reward",
    ]
    for field_name in forbidden_fields:
        assert not hasattr(SchedulerObservation, field_name)


def test_scheduler_observation_is_frozen():
    """Test that SchedulerObservation is strictly immutable (frozen=True)."""
    assert SchedulerObservation.__dataclass_params__.frozen is True

    env = _create_test_env()
    obs = env.reset(seed=42)

    with AssertRaises((FrozenInstanceError, AttributeError)):
        obs.timestamp = 10.0
    with AssertRaises((FrozenInstanceError, AttributeError)):
        obs.current_frequency_bin = 2
    with AssertRaises((FrozenInstanceError, AttributeError)):
        obs.last_detection = True


def test_scheduler_observation_contains_no_ground_truth():
    """Test that scheduler observations contain zero ground truth or evaluator state."""
    env = _create_test_env()
    obs = env.reset(seed=42)

    forbidden_gt_attrs = [
        "emitter_id",
        "emitters",
        "ground_truth",
        "transmitting_ids",
        "true_frequency",
        "power_dbm",
        "opportunity_id",
        "scope",
        "future_hops",
        "next_transmission_time",
    ]
    for forbidden in forbidden_gt_attrs:
        assert not hasattr(obs, forbidden)

    # Run several steps and verify every observation
    for i in range(5):
        obs, _, _, _ = env.step(ScanAction(frequency_bin=i % len(env.bands_hz)))
        for forbidden in forbidden_gt_attrs:
            assert not hasattr(obs, forbidden)


def test_reward_not_in_observation():
    """Test that reward is returned separately and is NOT embedded in SchedulerObservation."""
    assert "reward" not in [f.name for f in fields(SchedulerObservation)]
    assert not hasattr(SchedulerObservation, "reward")

    env = _create_test_env()
    env.reset(seed=42)
    next_obs, reward, done, info = env.step(ScanAction(frequency_bin=0))

    assert isinstance(reward, (float, int))
    assert not hasattr(next_obs, "reward")


def test_scheduler_state_alias_does_not_exist():
    """Test that SchedulerState alias is removed and does not exist in domain exports."""
    assert not hasattr(state_mod, "SchedulerState")
    assert not hasattr(domain_mod, "SchedulerState")
    assert not hasattr(metrics_mod, "SchedulerState")
    assert "SchedulerState" not in dir(domain_mod)


def test_transition_has_canonical_fields():
    """Test that Transition is frozen and contains canonical fields."""
    assert Transition.__dataclass_params__.frozen is True

    env = _create_test_env()
    obs0 = env.reset(seed=42)
    action0 = ScanAction(frequency_bin=1)
    obs1, r0, d0, info0 = env.step(action0)

    trans = Transition(
        observation=obs0,
        action=action0,
        reward=r0,
        next_observation=obs1,
        done=d0,
        info=info0,
    )

    assert trans.observation == obs0
    assert trans.action == action0
    assert trans.reward == r0
    assert trans.next_observation == obs1
    assert trans.done == d0
    assert trans.info == info0

    # Frozen
    with AssertRaises((FrozenInstanceError, AttributeError)):
        trans.reward = 999.0


def test_reset_reproducibility():
    """Test that env.reset(seed=42) produces bit-for-bit identical trajectory sequences."""
    env1 = _create_test_env()
    env2 = _create_test_env()

    obs1 = env1.reset(seed=123)
    obs2 = env2.reset(seed=123)

    assert obs1 == obs2

    for i in range(15):
        action = ScanAction(frequency_bin=i % len(env1.bands_hz))
        next_obs1, r1, d1, info1 = env1.step(action)
        next_obs2, r2, d2, info2 = env2.step(action)

        assert next_obs1 == next_obs2
        assert r1 == r2
        assert d1 == d2


def test_episode_state_resets():
    """Test that episode reset clears all episode-dependent counters and history."""
    env = _create_test_env()
    env.reset(seed=42)

    # Advance several steps
    for i in range(10):
        env.step(ScanAction(frequency_bin=i % len(env.bands_hz)))

    # Reset
    obs = env.reset(seed=42)

    assert obs.timestamp == 0.0
    assert obs.last_detection is False
    assert obs.last_detection_bin is None
    assert obs.last_detection_strength is None
    assert obs.recent_detection_history == ()
    assert obs.recent_frequency_history == ()
    assert obs.scan_count_by_bin == tuple([0] * len(env.bands_hz))
    assert env.metrics.total_scans == 0
    assert env.metrics.hits == 0
    assert len(env.metrics.opportunity_tracker.active_opportunities) == 0


def test_scheduler_environment_boundary():
    """Test strict isolation between scheduler and environment components."""
    env = _create_test_env()
    obs = env.reset(seed=42)

    scheduler = env.scheduler
    assert scheduler is not None

    # Scheduler has no access to hidden RF state
    assert not hasattr(scheduler, "emitters")
    assert not hasattr(scheduler, "ground_truth")
    assert not hasattr(scheduler, "channel")
    assert not hasattr(scheduler, "opportunity_tracker")

    # observe() accepts both canonical Transition and argument list
    action = scheduler.select_action(obs)
    assert isinstance(action, ScanAction)
    next_obs, r, d, info = env.step(action)

    trans = Transition(obs, action, r, next_obs, d)
    scheduler.observe(trans)  # Transition object
    scheduler.observe(obs, action, r, next_obs, d)  # Arg list


def test_ground_truth_firewall():
    """Test that mutating hidden emitter metadata cannot alter scheduler observation."""
    env = _create_test_env()
    obs_0 = env.reset(seed=42)

    # Mutate hidden ground-truth metadata on emitter (e.g. rename ID or inject internal label)
    env.emitters["E1"].id = "MUTATED_SECRET_ID"

    # Execute step with off-frequency scan
    off_bin = 0  # 110 MHz, far from 200-400 MHz
    next_obs, r, d, info = env.step(ScanAction(frequency_bin=off_bin))

    # Scheduler observation is strictly isolated
    assert not hasattr(next_obs, "emitter_id")
    assert not hasattr(next_obs, "emitters")
    assert "MUTATED_SECRET_ID" not in str(next_obs)

    # StepResult isolates observation from ground_truth
    step_res = env.step(ScanAction(frequency_bin=1))
    assert isinstance(step_res, StepResult)
    assert isinstance(step_res.observation, SchedulerObservation)
    assert "MUTATED_SECRET_ID" not in str(step_res.observation)
    assert "ground_truth" in step_res
    assert step_res["ground_truth"] is not None


def test_rl_scheduler_placeholder_loop():
    """Test RLScheduler placeholder operates cleanly in standard loop."""
    env = _create_test_env()
    scheduler = RLScheduler(env.bands_hz, allow_fallback_policy=True, seed=42)
    episode = Episode(episode_id=1)

    obs = env.reset(seed=42)
    scheduler.reset()
    done = False
    step_count = 0

    while not done and step_count < 10:
        action = scheduler.select_action(obs)
        assert isinstance(action, ScanAction)
        next_obs, reward, done, info = env.step(action)
        assert isinstance(next_obs, SchedulerObservation)
        scheduler.observe(Transition(obs, action, reward, next_obs, done))

        trans = Transition(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )
        episode.append(trans)
        obs = next_obs
        step_count += 1

    assert episode.total_steps == 10
    assert len(episode.transitions) == 10


def test_temporal_ordering():
    """Test strict causal ordering: Obs_t -> Action_t -> Step -> Obs_{t+1}."""
    env = _create_test_env()
    obs_0 = env.reset(seed=42)
    assert obs_0.timestamp == 0.0

    action_0 = ScanAction(frequency_bin=0)
    obs_1, r_0, done_0, info_0 = env.step(action_0)

    assert obs_1.current_frequency_bin == 0
    assert obs_1.recent_frequency_history[-1] == 0

    action_1 = ScanAction(frequency_bin=2)
    obs_2, r_1, done_1, info_1 = env.step(action_1)

    assert obs_2.current_frequency_bin == 2
    assert obs_2.recent_frequency_history[-1] == 2
    assert len(obs_2.recent_frequency_history) == 2
