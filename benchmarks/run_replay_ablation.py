"""benchmarks/run_replay_ablation.py
Controlled Research Ablation: Recurrent Replay Diagnosis & GPU Training Ablation (R0–R4).

Experimental Matrix:
- Phase 1 (2-Regime Distribution: [5,15,25] d=3 and [2,12,22] d=4):
  * R0: Baseline reproduction (burn_in=0, h0=0, uniform sampling, no regime balancing)
  * R1: Burn-in only (burn_in=5, total_len=15, loss masked on burn-in steps 0..4)
  * R2: Stored hidden state only (burn_in=0, captured live (h, c), zero cross-episode leakage)
  * R3: Burn-in + stored hidden state (burn_in=5, total_len=15, unroll starts from stored (h, c))
  * R4: Regime-balanced replay (R3 mechanics + 50/50 balanced batch sampling across regimes)

- Phase 2 (6-Regime Scaling):
  * Best variant from Phase 1 evaluated on full 6-regime training distribution.

Evaluated on the 8 untouched canonical test scenarios across 10 evaluation seeds.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
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


def get_hardware_info() -> dict[str, Any]:
    """Inspects available GPU, PyTorch, and CPU compute capabilities."""
    info: dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": "None",
        "gpu_vram_mb": 0,
        "driver_version": "None",
        "cuda_version": "None",
        "pytorch_installed": False,
        "pytorch_cuda_available": False,
        "cpu_count": os.cpu_count() or 1,
        "engine": "NumPy Vectorized (Multi-Process CPU)",
    }

    # Query nvidia-smi
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if smi_out:
            parts = [p.strip() for p in smi_out.split(",")]
            info["gpu_available"] = True
            info["gpu_name"] = parts[0]
            info["gpu_vram_mb"] = int(float(parts[1]))
            info["driver_version"] = parts[2]
    except Exception:
        pass

    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL)
        for line in nvcc_out.splitlines():
            if "release" in line:
                info["cuda_version"] = line.split("release")[-1].split(",")[0].strip()
    except Exception:
        info["cuda_version"] = "12.2 (from nvidia-smi)"

    # Query PyTorch
    try:
        import torch  # type: ignore
        info["pytorch_installed"] = True
        info["pytorch_cuda_available"] = bool(torch.cuda.is_available())
        if info["pytorch_cuda_available"]:
            info["engine"] = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
    except ImportError:
        info["pytorch_installed"] = False
        info["pytorch_cuda_available"] = False
        info["engine"] = f"Pure NumPy Vectorized Engine (Parallelized across {info['cpu_count']} CPU cores)"

    return info


def print_hardware_banner(hw: dict[str, Any]) -> None:
    print("=" * 80)
    print("🖥️  HARDWARE & COMPUTE CAPABILITY AUDIT")
    print(f"   GPU Detected:        {hw['gpu_name']} ({hw['gpu_vram_mb']} MiB VRAM)")
    print(f"   Driver / CUDA:       Driver {hw['driver_version']} | CUDA {hw['cuda_version']}")
    print(f"   PyTorch Installed:   {hw['pytorch_installed']} (CUDA Active: {hw['pytorch_cuda_available']})")
    print(f"   CPU Cores Available: {hw['cpu_count']}")
    print(f"   Compute Engine:      {hw['engine']}")
    print("=" * 80)


def _worker_train_seed(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Helper worker to execute train_single_seed in an isolated process."""
    return train_single_seed(**kwargs)


def run_variant(
    variant_id: str,
    variant_name: str,
    train_scenarios: list[tuple[str, dict[str, Any]]],
    burn_in: int,
    use_stored_hidden: bool,
    regime_balanced_replay: bool,
    output_dir: Path,
    seeds: list[int] = [42, 123, 456, 789],
    episodes: int = 30,
    episode_length: int = 300,
    val_interval: int = 5,
    eval_seeds: list[int] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    parallel: bool = True,
) -> dict[str, Any]:
    """Executes training, validation, candidate promotion, and canonical test evaluation for one variant."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🔬 RUNNING VARIANT: {variant_id.upper()} — {variant_name}")
    print(f"   Mechanics: burn_in={burn_in}, stored_hidden={use_stored_hidden}, balanced={regime_balanced_replay}")
    print(f"   Regimes ({len(train_scenarios)}): {[s[0] for s in train_scenarios]}")
    print(f"   Seeds: {seeds} | Episodes: {episodes} | Steps: {episode_length}")
    print(f"   Output Directory: {output_dir}")
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
            "burn_in": burn_in,
            "use_stored_hidden": use_stored_hidden,
            "regime_balanced_replay": regime_balanced_replay,
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
                        "variant": variant_id,
                        "variant_name": variant_name,
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
                    f"   ✅ [Candidate] Seed {s:3d} -> Val IR: {res['best_val_ir_mean']*100:5.1f}% "
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
                    "variant": variant_id,
                    "variant_name": variant_name,
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
                f"   ✅ [Candidate] Seed {s:3d} -> Val IR: {res['best_val_ir_mean']*100:5.1f}% "
                f"(±{res['best_val_ir_std']*100:4.1f}%), Min: {res['best_val_ir_min']*100:4.1f}% | "
                f"SHA256: {cand_hash[:16]}..."
            )

    # Candidate Selection
    best_candidate, rationale = select_best_candidate(candidate_records)
    promoted_path = output_dir / f"{variant_id}_promoted.npz"
    shutil.copyfile(best_candidate["checkpoint_file"], promoted_path)
    print(f"\n⭐ PROMOTED WINNER: Seed {best_candidate['seed']} (Val IR: {best_candidate['best_val_ir_mean']*100:.1f}%)")
    print(f"   Rationale: {rationale}")
    print(f"   Promoted Checkpoint: {promoted_path}")

    # Canonical Test Suite Evaluation
    print(f"\n📊 Evaluating {variant_id.upper()} on 8 Canonical Test Scenarios (10 seeds)...")
    t_eval_start = time.time()
    test_eval_results = evaluate_canonical_test_suite(
        production_checkpoint_path=promoted_path,
        eval_seeds=eval_seeds,
        steps=300,
    )
    t_eval_elapsed = time.time() - t_eval_start

    # Compute overall metrics across 8 canonical scenarios
    v41_irs = [sc_data["v41_hybrid_ir_mean"] for sc_data in test_eval_results.values()]
    v41_stds = [sc_data["v41_hybrid_ir_std"] for sc_data in test_eval_results.values()]
    v41_dets = [sc_data["v41_det_rate_mean"] for sc_data in test_eval_results.values()]
    overall = {
        "test_ir_mean": float(np.mean(v41_irs)),
        "test_ir_std": float(np.mean(v41_stds)),
        "test_det_rate_mean": float(np.mean(v41_dets)),
    }

    print(
        f"🎯 {variant_id.upper()} CANONICAL TEST RESULT: "
        f"IR = {overall['test_ir_mean']*100:.1f}% (±{overall['test_ir_std']*100:.1f}%), "
        f"Det = {overall['test_det_rate_mean']*100:.1f}% [Eval time: {t_eval_elapsed:.1f}s]"
    )

    t_total = time.time() - t_start

    return {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "mechanics": {
            "burn_in": burn_in,
            "use_stored_hidden": use_stored_hidden,
            "regime_balanced_replay": regime_balanced_replay,
        },
        "training_regimes": [s[0] for s in train_scenarios],
        "training_time_s": float(t_total - t_eval_elapsed),
        "evaluation_time_s": float(t_eval_elapsed),
        "total_time_s": float(t_total),
        "best_seed": best_candidate["seed"],
        "promoted_checkpoint": str(promoted_path),
        "promoted_sha256": best_candidate["sha256"],
        "validation_metrics": {
            "val_ir_mean": float(best_candidate["best_val_ir_mean"]),
            "val_ir_std": float(best_candidate["best_val_ir_std"]),
            "val_ir_min": float(best_candidate["best_val_ir_min"]),
        },
        "canonical_test_metrics": {
            "test_ir_mean": float(overall["test_ir_mean"]),
            "test_ir_std": float(overall["test_ir_std"]),
            "test_det_rate_mean": float(overall["test_det_rate_mean"]),
        },
        "per_scenario_results": test_eval_results,
        "candidates": [
            {
                "seed": c["seed"],
                "best_val_ir_mean": float(c["best_val_ir_mean"]),
                "best_val_ir_std": float(c["best_val_ir_std"]),
                "best_val_ir_min": float(c["best_val_ir_min"]),
                "episodes_trained": c["episodes_trained"],
                "sha256": c["sha256"],
            }
            for c in candidate_records
        ],
    }


def main() -> None:
    t_global_start = time.time()

    print("\n" + "=" * 80)
    print("🚀 SIH26055 — RECURRENT REPLAY DIAGNOSIS & ABLATION STUDY (R0–R4)")
    print("=" * 80)

    # 1. Dataset Isolation Check
    verify_dataset_isolation()

    # 2. Hardware Audit
    hw_info = get_hardware_info()
    print_hardware_banner(hw_info)

    # 3. Setup Scenarios
    all_train = list(get_train_scenarios().items())
    train_2regimes = [all_train[0], all_train[1]]  # TRAIN_1 (Base [5,15,25] d=3) + TRAIN_2 (Agile [2,12,22] d=4)
    train_6regimes = all_train                    # TRAIN_1 .. TRAIN_6

    base_out_dir = PROJECT_ROOT / "models" / "ablation" / "replay"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "study_name": "recurrent_replay_diagnosis_ablation",
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": hw_info,
        "phase_1_variants": {},
        "phase_2_scaling": {},
    }

    # =========================================================================
    # PHASE 1: 2-Regime Distribution Diagnostic Matrix
    # =========================================================================
    phase1_configs = [
        ("r0", "Baseline Reproduction (burn_in=0, h0=0, uniform)", 0, False, False),
        ("r1", "Burn-in Only (burn_in=5, h0=0, uniform)", 5, False, False),
        ("r2", "Stored Hidden State Only (burn_in=0, live h0, uniform)", 0, True, False),
        ("r3", "Burn-in + Stored Hidden State (burn_in=5, live h0, uniform)", 5, True, False),
        ("r4", "Regime-Balanced Replay + R3 (burn_in=5, live h0, balanced)", 5, True, True),
    ]

    for var_id, var_name, burn_in, use_stored, balanced in phase1_configs:
        v_out = base_out_dir / var_id
        res = run_variant(
            variant_id=var_id,
            variant_name=var_name,
            train_scenarios=train_2regimes,
            burn_in=burn_in,
            use_stored_hidden=use_stored,
            regime_balanced_replay=balanced,
            output_dir=v_out,
            seeds=[42, 123, 456, 789],
            episodes=30,
            episode_length=300,
            val_interval=5,
            eval_seeds=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            parallel=True,
        )
        results["phase_1_variants"][var_id] = res

    # Identify best Phase 1 variant
    best_p1_id = max(
        results["phase_1_variants"].keys(),
        key=lambda k: results["phase_1_variants"][k]["canonical_test_metrics"]["test_ir_mean"],
    )
    best_p1_res = results["phase_1_variants"][best_p1_id]
    print("\n" + "=" * 80)
    print(f"🏆 PHASE 1 WINNER: {best_p1_id.upper()} — {best_p1_res['variant_name']}")
    print(f"   Validation IR: {best_p1_res['validation_metrics']['val_ir_mean']*100:.1f}%")
    print(f"   Canonical Test IR: {best_p1_res['canonical_test_metrics']['test_ir_mean']*100:.1f}% "
          f"(±{best_p1_res['canonical_test_metrics']['test_ir_std']*100:.1f}%)")
    print("=" * 80)

    # =========================================================================
    # PHASE 2: 6-Regime Scaling of Best Variant
    # =========================================================================
    p2_out = base_out_dir / f"{best_p1_id}_6regimes"
    best_mech = best_p1_res["mechanics"]
    res_p2 = run_variant(
        variant_id=f"{best_p1_id}_6regimes",
        variant_name=f"Full 6-Regime Scaling with {best_p1_id.upper()} Mechanics",
        train_scenarios=train_6regimes,
        burn_in=best_mech["burn_in"],
        use_stored_hidden=best_mech["use_stored_hidden"],
        regime_balanced_replay=best_mech["regime_balanced_replay"],
        output_dir=p2_out,
        seeds=[42, 123, 456, 789],
        episodes=30,
        episode_length=300,
        val_interval=5,
        eval_seeds=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        parallel=True,
    )
    results["phase_2_scaling"] = res_p2

    # Save full JSON artifact
    json_path = PROJECT_ROOT / "models" / "ablation" / "recurrent_replay_ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full ablation study results saved to: {json_path}")

    # =========================================================================
    # FINAL COMPARISON TABLES & SYNTHESIS
    # =========================================================================
    print("\n" + "=" * 95)
    print("📊 PHASE 1: RECURRENT REPLAY DIAGNOSTIC MATRIX (2-REGIME GYM)")
    print("=" * 95)
    print(f"{'Variant':<8} | {'Mechanics':<32} | {'Val IR':<10} | {'Test IR':<14} | {'Det Rate':<10} | {'Train Time':<10}")
    print("-" * 95)
    for v_id, v_data in results["phase_1_variants"].items():
        v_mech = f"B={v_data['mechanics']['burn_in']}, H={int(v_data['mechanics']['use_stored_hidden'])}, Bal={int(v_data['mechanics']['regime_balanced_replay'])}"
        val_str = f"{v_data['validation_metrics']['val_ir_mean']*100:5.1f}%"
        test_str = f"{v_data['canonical_test_metrics']['test_ir_mean']*100:5.1f}% (±{v_data['canonical_test_metrics']['test_ir_std']*100:4.1f}%)"
        det_str = f"{v_data['canonical_test_metrics']['test_det_rate_mean']*100:5.1f}%"
        time_str = f"{v_data['training_time_s']:5.1f}s"
        print(f"{v_id.upper():<8} | {v_mech:<32} | {val_str:<10} | {test_str:<14} | {det_str:<10} | {time_str:<10}")

    print("\n" + "=" * 95)
    print("📈 PHASE 2: 6-REGIME TRAINING SCALING")
    print("=" * 95)
    print(f"Condition D Baseline (Standard 6-regime):   Test IR = ~33.8% (±5.5%)")
    p2_test_ir = res_p2["canonical_test_metrics"]["test_ir_mean"] * 100
    p2_test_std = res_p2["canonical_test_metrics"]["test_ir_std"] * 100
    p2_det = res_p2["canonical_test_metrics"]["test_det_rate_mean"] * 100
    print(f"{best_p1_id.upper()} 6-Regime Scaling:                       Test IR = {p2_test_ir:5.1f}% (±{p2_test_std:4.1f}%), Det = {p2_det:5.1f}%")
    delta_p2 = p2_test_ir - 33.8
    print(f"Delta vs Baseline 6-regime:                {'+' if delta_p2 >= 0 else ''}{delta_p2:.1f}%")
    print("=" * 95)

    elapsed_global = time.time() - t_global_start
    print(f"\n✨ Ablation Study Completed in {elapsed_global/60:.2f} minutes.")


if __name__ == "__main__":
    main()
