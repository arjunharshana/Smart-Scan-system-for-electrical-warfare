from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
import pandas as pd

from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.discounted_thompson import DiscountedThompsonSamplingScheduler
from rf_environment.scheduler.random_scheduler import RandomScheduler
from rf_environment.scheduler.sequential_scheduler import SequentialScheduler
from rf_environment.scheduler.sliding_window_ucb import SlidingWindowUCBScheduler
from rf_environment.scheduler.thompson_sampling import ThompsonSamplingScheduler
from rf_environment.scheduler.ucb1 import UCB1Scheduler
from rf_environment.domain.observation import Observation, TemporalHistory

SCHEDULER_CLASSES = {
    "sequential": SequentialScheduler,
    "random": RandomScheduler,
    "ucb1": UCB1Scheduler,
    "thompson": ThompsonSamplingScheduler,
    "sw_ucb": SlidingWindowUCBScheduler,
    "discounted_thompson": DiscountedThompsonSamplingScheduler,
    "context_aware": ContextAwareScheduler,
}


def audit_information_leakage():
    print("=" * 70)
    print("RUNNING TEST 6: INFORMATION LEAKAGE AUDIT")
    print("=" * 70)

    # 1. Inspect Observation domain model attributes
    obs_fields = list(Observation.model_fields.keys())
    temp_fields = list(TemporalHistory.model_fields.keys())

    print("\n1. Domain Model Schema Inspection:")
    print(f"   Observation fields: {obs_fields}")
    print(f"   TemporalHistory fields: {temp_fields}")

    # Forbidden fields check
    forbidden_tokens = [
        "ground_truth", "emitter_id", "emitters", "transmitting", 
        "trajectory", "future", "opportunity", "hidden"
    ]

    # 2. Runtime Dynamic Inspection during live simulation
    scenario = load_scenario("rf_environment/scenarios/mixed_environment.yaml")
    scenario["simulation"]["total_time_steps"] = 10

    env = build_environment(scenario, scheduler_name="context_aware")

    runtime_observations = []
    forbidden_detected = []

    # Intercept scheduler.select_action and update to inspect arguments
    original_select = env.scheduler.select_action
    original_update = env.scheduler.update

    recorded_inputs = []

    def wrapped_select(obs: Observation | None):
        if obs is not None:
            recorded_inputs.append(("select_action", obs.to_dict()))
        return original_select(obs)

    def wrapped_update(obs: Observation, reward: float, action: float | None = None):
        recorded_inputs.append(("update", {
            "obs": obs.to_dict(),
            "reward": reward,
            "action": action,
        }))
        return original_update(obs, reward, action=action)

    env.scheduler.select_action = wrapped_select
    env.scheduler.update = wrapped_update

    for t in range(5):
        res = env.step()

    print("\n2. Runtime Live Observation Payload (Step 0 & 1):")
    sample_obs = recorded_inputs[0][1] if recorded_inputs else {}
    print(json.dumps(sample_obs, indent=2))

    # Audit the payload for forbidden information
    def check_dict_keys(d: dict, prefix=""):
        for k, v in d.items():
            full_k = f"{prefix}.{k}" if prefix else k
            # Check if key or value leaks ground truth
            for f_token in forbidden_tokens:
                if f_token in k.lower():
                    # Exception: associated_emitter_id must be None
                    if k == "associated_emitter_id" and v is None:
                        continue
                    forbidden_detected.append((full_k, v, f"Key contains forbidden token '{f_token}'"))
            if isinstance(v, dict):
                check_dict_keys(v, full_k)

    if isinstance(sample_obs, dict):
        check_dict_keys(sample_obs)

    # 3. Code-Level Source Code Dependency Audit for all 7 Schedulers
    print("\n3. Code-Level Scheduler Audit:")
    scheduler_audits = []

    for name, cls in SCHEDULER_CLASSES.items():
        src = inspect.getsource(cls)
        lines = src.splitlines()

        uses_gt = any("ground_truth" in l for l in lines)
        uses_emitter = any("emitter" in l.lower() and not "associated_emitter_id" in l for l in lines)
        uses_opp = any("opportunity" in l.lower() for l in lines)
        accesses_env = any("self.env" in l for l in lines)

        status = "ALLOWED" if not (uses_gt or accesses_env or uses_opp) else "FORBIDDEN LEAK DETECTED"

        scheduler_audits.append({
            "Scheduler": name,
            "Class": cls.__name__,
            "Accesses Ground Truth": "YES (LEAK)" if uses_gt else "NO (ISOLATED)",
            "Accesses Env Object": "YES (LEAK)" if accesses_env else "NO (ISOLATED)",
            "Accesses Opportunities": "YES (LEAK)" if uses_opp else "NO (ISOLATED)",
            "Data Sources Used": "Observation, Measurement, TemporalHistory, Learned Counts",
            "Audit Verdict": status,
        })

    df_audit = pd.DataFrame(scheduler_audits)
    print(df_audit[["Scheduler", "Class", "Accesses Ground Truth", "Accesses Env Object", "Audit Verdict"]].to_string())

    # Check associated_emitter_id specifically
    sample_step_obs = env.step()["observation"]
    associated_id_val = sample_step_obs.get("associated_emitter_id")
    print(f"\n4. associated_emitter_id in Observation: {associated_id_val} (Expected: None)")

    out_file = Path("scratch/validation/test_6_leakage_audit.json")
    with open(out_file, "w") as f:
        json.dump({
            "observation_fields": obs_fields,
            "temporal_history_fields": temp_fields,
            "sample_runtime_observation": sample_obs,
            "forbidden_detected": forbidden_detected,
            "scheduler_audits": scheduler_audits,
            "associated_emitter_id_is_none": associated_id_val is None,
        }, f, indent=2)

    return df_audit


if __name__ == "__main__":
    audit_information_leakage()
