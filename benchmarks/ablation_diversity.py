"""benchmarks/ablation_diversity.py
Controlled Research Ablation: Measuring Training Diversity vs. Temporal Specialization in V4.1.

Conditions:
- Condition A (1 Regime): Base threat [5, 15, 25], dwell=3
- Condition B (2 Regimes): Base threat + Agile Dwell 4 [2, 12, 22]
- Condition C (3 Regimes): Base threat + Agile Dwell 4 + Fast Cycle Dwell 2 [3, 8, 18, 23]
- Condition D (6 Regimes): Full 6-regime training distribution (Production configuration)

All conditions use the exact same architecture, seeds [42, 123, 456, 789], 30 episodes,
300 steps, hyperparams, validation protocol on held-out VAL_1..4, and evaluation on the
untouched 8 canonical test scenarios across 10 eval seeds.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.scenarios_v4_1 import (
    get_canonical_test_scenarios,
    get_train_scenarios,
    get_validation_scenarios,
    verify_dataset_isolation,
)
from benchmarks.train_v4_1 import (
    evaluate_canonical_test_suite,
    select_best_candidate,
    train_single_seed,
)


def _worker_train_seed(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Helper worker to execute train_single_seed in a separate process."""
    return train_single_seed(**kwargs)


def run_condition(
    condition_id: str,
    condition_name: str,
    train_scenarios: list[tuple[str, dict[str, Any]]],
    output_dir: Path,
    seeds: list[int] = [42, 123, 456, 789],
    episodes: int = 30,
    episode_length: int = 300,
    val_interval: int = 5,
    eval_seeds: list[int] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    parallel: bool = True,
) -> dict[str, Any]:
    """Executes offline training, validation, candidate selection, and canonical test evaluation
    for one experimental condition.
    """
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🔬 RUNNING ABLATION: {condition_id.upper()} — {condition_name}")
    print(f"Training Regimes ({len(train_scenarios)}): {[s[0] for s in train_scenarios]}")
    print(f"Seeds: {seeds} | Episodes: {episodes} | Episode Length: {episode_length}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    train_kwargs_list = [
        {
            "seed": s,
            "episodes": episodes,
            "episode_length": episode_length,
            "val_interval": val_interval,
            "val_seeds": [10, 20, 30],
            "early_stopping": True,
            "patience": 4,
            "learning_rate": 0.001,
            "batch_size": 32,
            "sequence_length": 10,
            "warmup_steps": 64,
            "target_update_freq": 100,
            "epsilon_decay": 0.9992,
            "quiet": True,
            "train_scenarios": train_scenarios,
        }
        for s in seeds
    ]

    candidate_records: list[dict[str, Any]] = []

    if parallel and len(seeds) > 1:
        print(f"⚡ Spawning {len(seeds)} parallel training workers...")
        with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
            futures = [executor.submit(_worker_train_seed, kw) for kw in train_kwargs_list]
            for s, f in zip(seeds, futures):
                res = f.result()
                sched = res["scheduler"]
                cand_path = output_dir / f"lstm_ddqn_seed{s}.npz"
                sched.save_checkpoint(
                    cand_path,
                    metadata={
                        "condition": condition_id,
                        "training_seed": s,
                        "episodes_trained": res["episodes_trained"],
                        "best_val_ir_mean": res["best_val_ir_mean"],
                        "best_val_ir_std": res["best_val_ir_std"],
                        "role": "candidate",
                    },
                )
                cand_hash = sched.checkpoint_sha256
                res["checkpoint_file"] = str(cand_path)
                res["sha256"] = cand_hash
                candidate_records.append(res)
                print(
                    f"   ✅ [Candidate Ready] Seed {s:3d} -> Val IR: {res['best_val_ir_mean']*100:5.1f}% "
                    f"(±{res['best_val_ir_std']*100:4.1f}%), Min: {res['best_val_ir_min']*100:4.1f}% | "
                    f"SHA256: {cand_hash[:16]}..."
                )
    else:
        for kw in train_kwargs_list:
            s = kw["seed"]
            res = train_single_seed(**kw)
            sched = res["scheduler"]
            cand_path = output_dir / f"lstm_ddqn_seed{s}.npz"
            sched.save_checkpoint(
                cand_path,
                metadata={
                    "condition": condition_id,
                    "training_seed": s,
                    "episodes_trained": res["episodes_trained"],
                    "best_val_ir_mean": res["best_val_ir_mean"],
                    "best_val_ir_std": res["best_val_ir_std"],
                    "role": "candidate",
                },
            )
            cand_hash = sched.checkpoint_sha256
            res["checkpoint_file"] = str(cand_path)
            res["sha256"] = cand_hash
            candidate_records.append(res)
            print(
                f"   ✅ [Candidate Ready] Seed {s:3d} -> Val IR: {res['best_val_ir_mean']*100:5.1f}% "
                f"(±{res['best_val_ir_std']*100:4.1f}%), Min: {res['best_val_ir_min']*100:4.1f}% | "
                f"SHA256: {cand_hash[:16]}..."
            )

    # Model Selection
    best_cand, reason = select_best_candidate(candidate_records)
    print(f"\n🏆 Selected Checkpoint: Seed {best_cand['seed']} ({reason})")

    # Promote to production checkpoint within condition folder
    promoted_ckpt = output_dir / "production_checkpoint.npz"
    shutil.copyfile(best_cand["checkpoint_file"], promoted_ckpt)

    # Evaluate on untouched Canonical Test Suite
    test_results = evaluate_canonical_test_suite(
        production_checkpoint_path=promoted_ckpt,
        eval_seeds=eval_seeds,
        steps=episode_length,
    )

    # Compute overall summary metrics
    scenario_irs = {k: v["v41_hybrid_ir_mean"] for k, v in test_results.items()}
    overall_mean_ir = float(np.mean(list(scenario_irs.values())))

    # Aggregate overall mode distributions
    total_modes: dict[str, int] = {}
    for sc_res in test_results.values():
        for m_k, m_v in sc_res["mode_distribution"].items():
            total_modes[m_k] = total_modes.get(m_k, 0) + m_v

    total_decisions = max(sum(total_modes.values()), 1)
    exploit_share = total_modes.get("DDQN_EXPLOIT", 0) / total_decisions * 100.0
    adapt_share = total_modes.get("CA_ADAPT", 0) / total_decisions * 100.0
    blend_share = total_modes.get("BLENDED", 0) / total_decisions * 100.0

    mean_detection_rate = float(np.mean([v["v41_det_rate_mean"] for v in test_results.values()]))
    elapsed = time.time() - t_start

    print(f"Condition {condition_id} Completed in {elapsed:.1f}s | Overall Test IR: {overall_mean_ir*100:.2f}%")

    return {
        "condition_id": condition_id,
        "condition_name": condition_name,
        "train_scenarios": [s[0] for s in train_scenarios],
        "training_seeds": seeds,
        "selected_seed": best_cand["seed"],
        "checkpoint_file": str(promoted_ckpt),
        "checkpoint_sha256": best_cand["sha256"],
        "selection_reason": reason,
        "validation_ir_mean": best_cand["best_val_ir_mean"],
        "validation_ir_std": best_cand["best_val_ir_std"],
        "validation_ir_min": best_cand["best_val_ir_min"],
        "validation_det_rate": best_cand["best_val_det_rate"],
        "canonical_test_scenarios": test_results,
        "overall_test_ir_mean": overall_mean_ir,
        "overall_detection_rate": mean_detection_rate,
        "overall_mode_distribution": {
            "exploit_pct": exploit_share,
            "adapt_pct": adapt_share,
            "blend_pct": blend_share,
            "raw_counts": total_modes,
        },
        "candidate_summary": [
            {
                "seed": c["seed"],
                "val_ir_mean": c["best_val_ir_mean"],
                "val_ir_std": c["best_val_ir_std"],
                "val_ir_min": c["best_val_ir_min"],
                "sha256": c["sha256"],
            }
            for c in candidate_records
        ],
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    # Ensure dataset isolation
    verify_dataset_isolation()

    all_train_scs = get_train_scenarios()

    cond_a_scs = [("TRAIN_1_BaseThreat", all_train_scs["TRAIN_1_BaseThreat"])]
    cond_b_scs = [
        ("TRAIN_1_BaseThreat", all_train_scs["TRAIN_1_BaseThreat"]),
        ("TRAIN_2_Agile_Dwell4", all_train_scs["TRAIN_2_Agile_Dwell4"]),
    ]
    cond_c_scs = [
        ("TRAIN_1_BaseThreat", all_train_scs["TRAIN_1_BaseThreat"]),
        ("TRAIN_2_Agile_Dwell4", all_train_scs["TRAIN_2_Agile_Dwell4"]),
        ("TRAIN_3_FastCycle_Dwell2", all_train_scs["TRAIN_3_FastCycle_Dwell2"]),
    ]
    cond_d_scs = list(all_train_scs.items())

    base_output = Path("models/ablation")
    base_output.mkdir(parents=True, exist_ok=True)

    conditions_data = [
        ("condition_a", "1 Regime (Base Threat)", cond_a_scs, base_output / "condition_a_1regime"),
        ("condition_b", "2 Regimes (Base + Agile d=4)", cond_b_scs, base_output / "condition_b_2regimes"),
        ("condition_c", "3 Regimes (Base + Agile d=4 + Fast d=2)", cond_c_scs, base_output / "condition_c_3regimes"),
        ("condition_d", "6 Regimes (Full Diverse Gym)", cond_d_scs, base_output / "condition_d_6regimes"),
    ]

    results: dict[str, Any] = {}

    for cond_id, cond_name, tr_scs, out_d in conditions_data:
        res = run_condition(
            condition_id=cond_id,
            condition_name=cond_name,
            train_scenarios=tr_scs,
            output_dir=out_d,
            seeds=[42, 123, 456, 789],
            episodes=30,
            episode_length=300,
            val_interval=5,
            eval_seeds=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            parallel=True,
        )
        results[cond_id] = res

    # Save full ablation output
    results_path = base_output / "diversity_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 All Ablation Results Saved to: {results_path}")

    # Print Comparison Table
    print("\n" + "=" * 105)
    print("📊 FINAL COMPARISON TABLE: TRAINING REGIME DIVERSITY vs CANONICAL TEST PERFORMANCE")
    print("=" * 105)
    header = f"{'Training Diversity':<20} | {'Seen':<7} | {'Permut':<7} | {'Phase':<7} | {'Dwell':<7} | {'Subset':<7} | {'Mixed':<7} | {'Random':<7} | {'Burst':<7} | {'Overall':<7}"
    print(header)
    print("-" * 105)

    order = [
        ("condition_a", "1 regime (Base)"),
        ("condition_b", "2 regimes"),
        ("condition_c", "3 regimes"),
        ("condition_d", "6 regimes"),
    ]

    sc_keys = [
        "1_Seen_Structure",
        "2_Unseen_Permutation",
        "3_Unseen_Phase",
        "4_Unseen_Dwell",
        "5_Unseen_Subset",
        "6_Mixed_Shift",
        "7_Random_Hopping",
        "8_Periodic_Burst",
    ]

    for c_id, label in order:
        r = results[c_id]
        sc_irs = [r["canonical_test_scenarios"][k]["v41_hybrid_ir_mean"] * 100.0 for k in sc_keys]
        ov_ir = r["overall_test_ir_mean"] * 100.0
        row_str = f"{label:<20} | " + " | ".join(f"{ir:6.1f}%" for ir in sc_irs) + f" | {ov_ir:6.1f}%"
        print(row_str)

    print("-" * 105)


if __name__ == "__main__":
    main()
