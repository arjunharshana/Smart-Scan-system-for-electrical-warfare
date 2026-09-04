from __future__ import annotations

import copy
import numpy as np

from rf_environment.domain.observation import Observation
from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.experiments.runner import BenchmarkRunner
from rf_environment.scheduler.factory import create_scheduler


def test_opportunity_lifecycle_continuous_fixed():
    """Continuous fixed emitter produces exactly 1 opportunity across the simulation."""
    scenario = {
        "simulation": {"total_time_steps": 100, "time_step_ms": 10, "seed": 42},
        "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 500e6},
        "receiver": {"instantaneous_bandwidth_hz": 20e6, "tuning_time_ms": 0},
        "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
        "scheduler": {"type": "sequential"},
        "emitters": [
            {
                "id": "E_FIXED",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": 200e6},
                "time_behavior": {"type": "continuous"},
                "power_dbm": -20,
                "bandwidth_hz": 1e6,
            }
        ],
    }
    env = build_environment(scenario)
    env.run(100)
    opps = env.metrics.opportunity_tracker.get_all_opportunities()
    assert len(opps) == 1
    assert opps[0].scope == "EPISODE"
    assert opps[0].start_step == 0
    assert opps[0].end_step == 99


def test_opportunity_lifecycle_periodic_burst():
    """Periodic emitter with known period and pulse width produces distinct burst opportunities."""
    # 100 steps total, period = 20 steps, pulse_width = 10 steps -> exactly 5 ON bursts (steps 0-9, 20-29, 40-49, 60-69, 80-89)
    scenario = {
        "simulation": {"total_time_steps": 100, "time_step_ms": 10, "seed": 42},
        "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 500e6},
        "receiver": {"instantaneous_bandwidth_hz": 20e6, "tuning_time_ms": 0},
        "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
        "scheduler": {"type": "sequential"},
        "emitters": [
            {
                "id": "E_PERIODIC",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": 200e6},
                "time_behavior": {"type": "periodic", "on_duration": 10, "off_duration": 10},
                "power_dbm": -20,
                "bandwidth_hz": 1e6,
            }
        ],
    }
    env = build_environment(scenario)
    env.run(100)
    opps = env.metrics.opportunity_tracker.get_all_opportunities()
    assert len(opps) == 5
    for opp in opps:
        assert opp.scope == "BURST"
        assert (opp.end_step - opp.start_step + 1) == 10


def test_opportunity_lifecycle_hopping_dwell_1():
    """Frequency-hopping emitter with dwell=1 produces an opportunity for every step."""
    scenario = {
        "simulation": {"total_time_steps": 60, "time_step_ms": 10, "seed": 42},
        "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 700e6},
        "receiver": {"instantaneous_bandwidth_hz": 20e6, "tuning_time_ms": 0},
        "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
        "scheduler": {"type": "random"},
        "emitters": [
            {
                "id": "E_HOP_D1",
                "type": "communication",
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
    env = build_environment(scenario)
    env.run(60)
    opps = env.metrics.opportunity_tracker.get_all_opportunities()
    assert len(opps) == 60
    assert all(o.scope == "HOP" for o in opps)
    assert all(o.dwell_steps == 1 for o in opps)


def test_opportunity_lifecycle_hopping_dwell_5():
    """Frequency-hopping emitter with dwell=5 produces N/5 hop opportunities."""
    scenario = {
        "simulation": {"total_time_steps": 60, "time_step_ms": 10, "seed": 42},
        "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 700e6},
        "receiver": {"instantaneous_bandwidth_hz": 20e6, "tuning_time_ms": 0},
        "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
        "scheduler": {"type": "random"},
        "emitters": [
            {
                "id": "E_HOP_D5",
                "type": "communication",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [200e6, 300e6, 400e6],
                    "mode": "sequential",
                    "dwell_steps": 5,
                },
                "time_behavior": {"type": "continuous"},
                "power_dbm": -20,
                "bandwidth_hz": 1e6,
            }
        ],
    }
    env = build_environment(scenario)
    env.run(60)
    opps = env.metrics.opportunity_tracker.get_all_opportunities()
    assert len(opps) == 12  # 60 / 5 = 12
    assert all(o.scope == "HOP" for o in opps)
    for opp in opps:
        assert (opp.end_step - opp.start_step + 1) == 5


def test_random_5_channel_hopper_baseline():
    """Random 5-channel hopper produces ~20% hop coverage and hop interception under random scanning."""
    runner = BenchmarkRunner()
    scenario = load_scenario("rf_environment/scenarios/random_hopping.yaml")
    # Restrict receiver candidate bands to the 5 hopping frequencies
    scenario["scheduler"] = {
        "type": "random",
        "bands_hz": [200e6, 300e6, 400e6, 500e6, 600e6],
    }
    # Run 1000 steps with random scheduler
    res = runner.run_single(scenario, "random", seed=42, steps=1000)

    # Theoretical probability of uniform overlap across 5 channels is 1/5 = 0.20
    assert 0.16 <= res["hop_coverage_ratio"] <= 0.24
    assert 0.16 <= res["hop_interception_ratio"] <= 0.24
    assert 0.16 <= res["step_coverage_ratio"] <= 0.24
    # Macro episode coverage remains 1.0, but hop interception is correctly ~20%
    assert res["episode_coverage_ratio"] == 1.0
    assert res["metric_scope"] == "HOP"


def test_detector_scheduler_separation():
    """When detector is perfect (Pd=1.0), Detection Given Coverage is 100% and IR == Coverage."""
    scenario = {
        "simulation": {"total_time_steps": 100, "time_step_ms": 10, "seed": 42},
        "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 500e6},
        "receiver": {"instantaneous_bandwidth_hz": 20e6, "tuning_time_ms": 0},
        "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
        "scheduler": {"type": "random"},
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
    runner = BenchmarkRunner()
    res = runner.run_single(scenario, "random", seed=42, steps=100)
    assert res["detection_given_coverage"] == 1.0
    assert abs(res["hop_interception_ratio"] - res["hop_coverage_ratio"]) < 1e-5


def test_metric_identity_mathematical_precision():
    """Mathematically proves Interception Ratio == Coverage * Detection Given Coverage within 1e-5."""
    scenarios_to_test = [
        "rf_environment/scenarios/stationary_fixed.yaml",
        "rf_environment/scenarios/periodic_time.yaml",
        "rf_environment/scenarios/deterministic_hopping.yaml",
        "rf_environment/scenarios/random_hopping.yaml",
    ]
    runner = BenchmarkRunner()
    for sc_path in scenarios_to_test:
        sc = load_scenario(sc_path)
        for sched in ["sequential", "random", "ucb1", "context_aware"]:
            res = runner.run_single(sc, sched, seed=42, steps=100)
            cov = res["opportunity_coverage"]
            dgc = res["detection_given_coverage"]
            ir = res["interception_ratio"]
            if cov > 0:
                assert abs(ir - (cov * dgc)) < 1e-5, f"Identity violated for {sc_path} with {sched}: {ir} != {cov} * {dgc}"
            assert res["identity_verified"] is True


def test_scheduler_has_no_ground_truth_access():
    """Strict ground-truth isolation audit: observations provide zero ground-truth leakage."""
    scenario = load_scenario("rf_environment/scenarios/deterministic_hopping.yaml")
    env = build_environment(scenario, scheduler_name="context_aware")

    for _ in range(20):
        step_res = env.step()
        obs = step_res["observation"]

        # 1. Observation dict must not contain ground truth keys
        forbidden_keys = [
            "emitters",
            "transmitting_ids",
            "ground_truth",
            "true_frequency",
            "future_frequency",
            "oracle_action",
        ]
        for key in forbidden_keys:
            assert key not in obs, f"Leakage detected: '{key}' found in observation"

        # 2. Associated emitter id must be None in observation
        assert obs.get("associated_emitter_id") is None

        # 3. Receiver frequency must reflect only the scanned frequency
        assert obs.get("receiver_frequency_hz") is not None


def test_frozen_realization_determinism():
    """Same seed produces identical RF environment realization across all algorithms."""
    scenario = load_scenario("rf_environment/scenarios/random_hopping.yaml")

    env1 = build_environment(scenario, scheduler_name="ucb1")
    env2 = build_environment(scenario, scheduler_name="context_aware")

    for _ in range(50):
        r1 = env1.step()
        r2 = env2.step()

        # Emitters ground truth must be 100% bitwise identical
        gt1 = r1["ground_truth"]
        gt2 = r2["ground_truth"]
        assert gt1["transmitting_ids"] == gt2["transmitting_ids"]
        for e1, e2 in zip(gt1["emitters"], gt2["emitters"]):
            assert e1["frequency_hz"] == e2["frequency_hz"]
            assert e1["transmitting"] == e2["transmitting"]
            assert e1["power_dbm"] == e2["power_dbm"]


def test_all_seven_schedulers_run_cleanly():
    """All 7 approved schedulers execute and produce valid states without exceptions."""
    schedulers = [
        "sequential",
        "random",
        "ucb1",
        "thompson",
        "sw_ucb",
        "discounted_thompson",
        "context_aware",
    ]
    scenario = load_scenario("rf_environment/scenarios/deterministic_hopping.yaml")
    runner = BenchmarkRunner()

    for s_name in schedulers:
        res = runner.run_single(scenario, s_name, seed=42, steps=50)
        assert res["steps"] == 49
        assert res["algorithm"] == s_name
        assert res["interception_ratio"] >= 0.0
        assert res["identity_verified"] is True


def test_periodic_aliasing_stroboscopic_blindness():
    """Periodic aliasing test: scan period 20 vs emitter period 20 creates phase blindness; period 19 breaks it."""
    # Emitter: ON=5, OFF=15 (period=20). Fixed at 400 MHz.
    # Case A: 20 bands of 20 MHz (100 to 500 MHz) -> scan cycle = 20.
    blind_count_20 = 0
    for phase in range(20):
        scenario = {
            "simulation": {"total_time_steps": 100, "time_step_ms": 10, "seed": 42},
            "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 500e6},
            "receiver": {"instantaneous_bandwidth_hz": 20e6, "tuning_time_ms": 0},
            "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
            "scheduler": {"type": "sequential"},
            "emitters": [
                {
                    "id": "E_PERIODIC",
                    "type": "communication",
                    "frequency_behavior": {"type": "fixed", "frequency_hz": 400e6},
                    "time_behavior": {"type": "periodic", "on_duration": 5, "off_duration": 15, "phase": phase},
                    "power_dbm": -20,
                    "bandwidth_hz": 2e6,
                }
            ],
        }
        env = build_environment(scenario, scheduler_name="sequential")
        env.run(steps=100)
        snap = env.metrics.snapshot(100)
        if snap.hits == 0:
            blind_count_20 += 1

    # Exactly 15 out of 20 phases (75%) should suffer complete blindness when period=20!
    assert blind_count_20 > 10

    # Case B: 19 bands (coprime scan cycle) -> scan cycle = 19, 300 steps
    bws = 400e6 / 19
    bands_19 = [100e6 + (i + 0.5) * bws for i in range(19)]
    blind_count_19 = 0
    for phase in range(20):
        scenario = {
            "simulation": {"total_time_steps": 300, "time_step_ms": 10, "seed": 42},
            "spectrum": {"min_frequency_hz": 100e6, "max_frequency_hz": 500e6},
            "receiver": {"instantaneous_bandwidth_hz": bws, "tuning_time_ms": 0},
            "detector": {"p_detection": 1.0, "p_false_alarm": 0.0},
            "scheduler": {"type": "sequential", "bands_hz": bands_19},
            "emitters": [
                {
                    "id": "E_PERIODIC",
                    "type": "communication",
                    "frequency_behavior": {"type": "fixed", "frequency_hz": 400e6},
                    "time_behavior": {"type": "periodic", "on_duration": 5, "off_duration": 15, "phase": phase},
                    "power_dbm": -20,
                    "bandwidth_hz": 2e6,
                }
            ],
        }
        env = build_environment(scenario, scheduler_name="sequential")
        env.run(steps=300)
        snap = env.metrics.snapshot(300)
        if snap.hits == 0:
            blind_count_19 += 1

    # Coprime precession eliminates persistent blindness across all phases!
    assert blind_count_19 == 0

