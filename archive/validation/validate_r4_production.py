from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import numpy as np

from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.rewards.r4_reward import R4RewardCalculator
from rf_environment.rewards.reward import RewardCalculator

SCENARIOS = [
    ("stationary_fixed", "rf_environment/scenarios/stationary_fixed.yaml"),
    ("deterministic_hopping", "rf_environment/scenarios/deterministic_hopping.yaml"),
    ("random_hopping", "rf_environment/scenarios/random_hopping.yaml"),
    ("periodic_time", "rf_environment/scenarios/periodic_time.yaml"),
]

SEEDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]


def run_episode(scenario_cfg: dict, seed: int, reward_type: str = "r4") -> dict:
    scen = copy.deepcopy(scenario_cfg)
    scen.setdefault("simulation", {})["seed"] = seed

    if reward_type == "r4":
        reward_calc = R4RewardCalculator()
    elif reward_type == "old":
        reward_calc = RewardCalculator()
    else:
        raise ValueError(f"Unknown reward_type {reward_type}")

    env = build_environment(scen)
    env.reward_calculator = reward_calc

    step_rewards = []
    results = env.run()
    for res in results:
        step_rewards.append(res.reward)

    snapshot = env.metrics.snapshot(max(env.clock.time_step, 0)).to_dict()

    return {
        "step_rewards": step_rewards,
        "mean_reward": float(np.mean(step_rewards)),
        "std_reward": float(np.std(step_rewards)),
        "min_reward": float(np.min(step_rewards)),
        "max_reward": float(np.max(step_rewards)),
        "detection_rate": float(snapshot.get("hits", 0)) / float(max(env.clock.time_step, 1)),
        "interception_ratio": float(snapshot.get("interception_ratio", 0.0)),
        "opportunity_coverage": float(snapshot.get("opportunity_coverage", 0.0)),
        "all_finite": all(math.isfinite(r) for r in step_rewards),
        "in_bounds": all(-0.05 - 1e-9 <= r <= 1.15 + 1e-9 for r in step_rewards) if reward_type == "r4" else True,
    }


def main():
    print("=" * 80)
    print("🚀 V3 REWARD R4 PRODUCTION VALIDATION & COMPARISON PIPELINE")
    print("=" * 80)

    integration_results = {}
    comparison_results = {}

    all_integration_pass = True

    for scen_name, scen_path in SCENARIOS:
        print(f"\nEvaluating Scenario: {scen_name} ({scen_path})")
        base_cfg = load_scenario(scen_path)

        integration_results[scen_name] = []
        comparison_results[scen_name] = {"r4": [], "old": []}

        for seed in SEEDS:
            # 1. R4 Run
            r4_res = run_episode(base_cfg, seed, reward_type="r4")
            # 2. Old Reward Run
            old_res = run_episode(base_cfg, seed, reward_type="old")

            # Check bounds and finiteness for R4
            passed = r4_res["all_finite"] and r4_res["in_bounds"]
            if not passed:
                all_integration_pass = False

            integration_results[scen_name].append({
                "seed": seed,
                "passed": passed,
                "all_finite": r4_res["all_finite"],
                "in_bounds": r4_res["in_bounds"],
                "min_reward": r4_res["min_reward"],
                "max_reward": r4_res["max_reward"],
                "mean_reward": r4_res["mean_reward"],
            })

            comparison_results[scen_name]["r4"].append(r4_res)
            comparison_results[scen_name]["old"].append(old_res)

            print(
                f"   Seed {seed:2d}: "
                f"R4=[{r4_res['min_reward']:.2f}, {r4_res['max_reward']:.2f}], mean={r4_res['mean_reward']:.4f} "
                f"| Old=[{old_res['min_reward']:.2f}, {old_res['max_reward']:.2f}], mean={old_res['mean_reward']:.4f} "
                f"| Status={'PASS' if passed else 'FAIL'}"
            )

    print("\n" + "=" * 80)
    print("📊 AGGREGATED COMPARISON TABLE (R4 vs OLD REWARD)")
    print("=" * 80)
    print(f"{'Scenario':<25} | {'Reward':<5} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'Det Rate':<8} | {'Intercept':<9} | {'Opp Cov':<8}")
    print("-" * 105)

    summary_table = []

    for scen_name, _ in SCENARIOS:
        for rtype in ["r4", "old"]:
            runs = comparison_results[scen_name][rtype]
            mean_r = float(np.mean([r["mean_reward"] for r in runs]))
            std_r = float(np.mean([r["std_reward"] for r in runs]))
            min_r = float(np.min([r["min_reward"] for r in runs]))
            max_r = float(np.max([r["max_reward"] for r in runs]))
            det_rate = float(np.mean([r["detection_rate"] for r in runs]))
            interception = float(np.mean([r["interception_ratio"] for r in runs]))
            opp_cov = float(np.mean([r["opportunity_coverage"] for r in runs]))

            label = "R4 (V3)" if rtype == "r4" else "Old"
            print(
                f"{scen_name:<25} | {label:<5} | {mean_r:8.4f} | {std_r:8.4f} | {min_r:8.4f} | {max_r:8.4f} | {det_rate:8.4f} | {interception:9.4f} | {opp_cov:8.4f}"
            )

            summary_table.append({
                "scenario": scen_name,
                "reward_type": label,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "min_reward": min_r,
                "max_reward": max_r,
                "detection_rate": det_rate,
                "interception_ratio": interception,
                "opportunity_coverage": opp_cov,
            })

    output_payload = {
        "integration_passed": all_integration_pass,
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "integration_results": integration_results,
        "summary_table": summary_table,
    }

    out_file = Path("scratch/validation/validate_r4_results.json")
    out_file.write_text(json.dumps(output_payload, indent=2))
    print(f"\nSaved full validation payload to {out_file}")

    if all_integration_pass:
        print("\n🎉 ALL INTEGRATION AND BOUNDS CHECKS PASSED!")
        return 0
    else:
        print("\n❌ SOME INTEGRATION CHECKS FAILED!")
        return 1


if __name__ == "__main__":
    main()
