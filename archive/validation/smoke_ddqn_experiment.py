from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.scheduler.rl.ddqn_scheduler import DDQNScheduler


def run_smoke_training(
    scenario_path: str = "rf_environment/scenarios/deterministic_hopping.yaml",
    seed: int = 42,
    num_episodes: int = 1,
    steps_per_episode: int = 400,
) -> dict:
    scenario_cfg = load_scenario(scenario_path)
    scenario_cfg.setdefault("simulation", {})["seed"] = seed
    scenario_cfg["simulation"]["total_time_steps"] = steps_per_episode

    env = build_environment(scenario_cfg)

    # Initialize DDQN Scheduler with reasonable baseline hyperparameters
    scheduler = DDQNScheduler(
        bands_hz=env.bands_hz,
        gamma=0.95,
        learning_rate=0.001,
        replay_capacity=10000,
        batch_size=32,
        warmup_steps=64,
        target_update_frequency=50,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        hidden_dimension=64,
        seed=seed,
    )
    env.scheduler = scheduler

    total_steps = 0
    all_rewards = []
    all_q_values = []
    nan_or_inf_detected = False

    print("=" * 70)
    print("🚀 Running DDQN Short Smoke Training Experiment")
    print(f"Scenario: deterministic_hopping | Seed: {seed} | Steps: {steps_per_episode}")
    print("=" * 70)

    for ep in range(num_episodes):
        results = env.run(steps=steps_per_episode)
        for res in results:
            total_steps += 1
            r = res.reward
            all_rewards.append(r)
            if not math.isfinite(r):
                nan_or_inf_detected = True

            # Track Q-values from explanation if available
            exp = scheduler.last_explanation
            if "q_values" in exp:
                q_vals = exp["q_values"]
                all_q_values.extend(q_vals)
                if any(not math.isfinite(q) for q in q_vals):
                    nan_or_inf_detected = True

    # Check network weights for NaN/Inf
    for p in [scheduler.online_net.w1, scheduler.online_net.b1, scheduler.online_net.w2, scheduler.online_net.b2, scheduler.online_net.w3, scheduler.online_net.b3]:
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            nan_or_inf_detected = True

    losses = scheduler.losses
    if losses:
        for l in losses:
            if not math.isfinite(l):
                nan_or_inf_detected = True
        initial_loss = float(np.mean(losses[: min(10, len(losses))]))
        final_loss = float(np.mean(losses[-min(10, len(losses)) :]))
    else:
        initial_loss = 0.0
        final_loss = 0.0

    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    min_q = float(np.min(all_q_values)) if all_q_values else 0.0
    max_q = float(np.max(all_q_values)) if all_q_values else 0.0

    report = {
        "scenario": "deterministic_hopping",
        "seed": seed,
        "environment_steps": total_steps,
        "training_updates": scheduler.train_step_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "mean_reward": mean_reward,
        "final_epsilon": float(scheduler.epsilon),
        "q_min": min_q,
        "q_max": max_q,
        "nan_inf_occurrence": nan_or_inf_detected,
    }

    print("\n📊 Smoke Training Results:")
    print(f"   Environment Steps:      {report['environment_steps']}")
    print(f"   Training Updates:       {report['training_updates']}")
    print(f"   Initial Loss (1st 10):  {report['initial_loss']:.6f}")
    print(f"   Final Loss (last 10):   {report['final_loss']:.6f}")
    print(f"   Mean Step Reward:       {report['mean_reward']:.4f}")
    print(f"   Final Epsilon:          {report['final_epsilon']:.4f}")
    print(f"   Q-value Range:          [{report['q_min']:.4f}, {report['q_max']:.4f}]")
    print(f"   NaN/Inf Occurrence:     {'YES (FAILED)' if nan_or_inf_detected else 'NONE (PASSED)'}")
    print("=" * 70)

    out_file = Path("scratch/validation/smoke_ddqn_results.json")
    out_file.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out_file}\n")
    return report


if __name__ == "__main__":
    run_smoke_training()
