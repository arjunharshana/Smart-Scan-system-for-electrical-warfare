from pathlib import Path

from rf_environment.environment.builder import build_from_path
from rf_environment.experiments.runner import ExperimentRunner
from rf_environment.environment.scenario import load_scenario

SCENARIO = Path(__file__).resolve().parents[1] / "rf_environment" / "scenarios" / "basic.yaml"


def test_same_seed_is_reproducible():
    a = build_from_path(SCENARIO, scheduler_name="sequential")
    b = build_from_path(SCENARIO, scheduler_name="sequential")
    ra = a.run()
    rb = b.run()
    fa = [step["ground_truth"]["emitters"] for step in ra]
    fb = [step["ground_truth"]["emitters"] for step in rb]
    assert fa == fb


def test_experiment_compares_schedulers():
    scenario = load_scenario(SCENARIO)
    scenario["simulation"]["total_time_steps"] = 40
    results = ExperimentRunner().compare(
        scenario,
        ["sequential", "random", "ucb1", "thompson_sampling"],
        seed=42,
        scenario_name="basic",
    )
    assert len(results) == 4
    assert all(r.metrics["total_scans"] == 40 for r in results)
