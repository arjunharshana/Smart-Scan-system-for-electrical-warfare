from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.rl.checkpoint_v40 import load_v40_checkpoint, save_v40_checkpoint
from rf_environment.scheduler.rl.ddqn_scheduler import DDQNScheduler
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder

SPECTRUM_CFG = {"min_frequency_hz": 100_000_000, "max_frequency_hz": 700_000_000}
RECEIVER_CFG = {
    "instantaneous_bandwidth_hz": 20_000_000,
    "sensitivity_dbm": -90,
    "noise_floor_dbm": -100,
    "detection_threshold_db": 6,
    "tuning_time_ms": 0,
}
DETECTOR_CFG = {"p_detection": 0.95, "p_false_alarm": 0.02}


def bin_to_center_hz(b: int) -> float:
    return 100_000_000.0 + 10_000_000.0 + float(b) * 20_000_000.0


def make_train_scenario(seed: int = 42) -> dict[str, Any]:
    return {
        "simulation": {
            "total_time_steps": 300,
            "time_step_ms": 10,
            "seed": seed,
        },
        "spectrum": SPECTRUM_CFG,
        "receiver": RECEIVER_CFG,
        "detector": DETECTOR_CFG,
        "scheduler": {"type": "sequential"},
        "emitters": [{
            "id": "EM",
            "type": "radar",
            "frequency_behavior": {
                "type": "hopping",
                "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                "mode": "sequential",
                "dwell_steps": 3,
            },
            "time_behavior": {"type": "continuous"},
            "power_dbm": -18,
            "bandwidth_hz": 2_000_000,
        }],
    }


def train_and_save_v4_0(
    output_path: str | Path = "models/v4_0/production_checkpoint.npz",
    seed: int = 42,
    num_episodes: int = 5,
    steps_per_episode: int = 300,
) -> Path:
    out_file = Path(output_path).resolve()
    print("=== OFFLINE V4.0 DDQN TRAINING ===")
    print(f"Target Checkpoint: {out_file}")
    print(f"Base Seed: {seed} | Episodes: {num_episodes} | Steps/Ep: {steps_per_episode}")

    base_sc = make_train_scenario(seed=seed)
    env = build_environment(copy.deepcopy(base_sc))
    bands = env.bands_hz
    num_bins = len(bands)

    encoder = TemporalObservationEncoder.create_encoder_b(num_bins=num_bins)
    scheduler = DDQNScheduler(
        bands_hz=bands,
        gamma=0.95,
        learning_rate=0.001,
        replay_capacity=10000,
        batch_size=32,
        warmup_steps=64,
        target_update_frequency=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        hidden_dimension=64,
        encoder=encoder,
        seed=seed,
    )
    scheduler.train()
    env.scheduler = scheduler

    t0 = time.time()
    for ep in range(num_episodes):
        ep_seed = seed + ep * 1000
        cur_eps = scheduler.epsilon
        env.reset(seed=ep_seed)
        scheduler.epsilon = cur_eps
        results = env.run(steps=steps_per_episode)
        hits = sum(1 for r in results if r.observation.last_detection)
        print(
            f"  Episode {ep+1}/{num_episodes} (seed={ep_seed}): "
            f"Hits={hits}/{len(results)} ({hits/len(results)*100:.1f}%), "
            f"Loss={scheduler.last_loss:.4f}, Epsilon={scheduler.epsilon:.3f}"
        )

    t_train = time.time() - t0
    print(f"Training completed in {t_train:.2f}s")

    # Freeze scheduler
    scheduler.eval()
    scheduler.epsilon = 0.0

    metadata = {
        "description": "SIH26055 V4.0 Feed-Forward DDQN Production Checkpoint",
        "training_scenario": "1_Seen_Structure",
        "training_regime": "[5, 15, 25], dwell=3",
        "episodes_trained": num_episodes,
        "seed": seed,
        "evaluation_role": "V4.0 Hybrid Primary Scheduler (Benchmark Winner: 35.19% IR)",
    }

    saved_path = scheduler.save_checkpoint(out_file, metadata=metadata)
    print(f"✅ Checkpoint saved successfully: {saved_path}")
    print(f"   SHA256 Fingerprint: {scheduler.checkpoint_sha256}")

    # Verify loading
    loaded = load_v40_checkpoint(saved_path)
    print(f"✅ Cryptographic verification passed. Format version: {loaded['metadata']['format_version']}")
    return saved_path


if __name__ == "__main__":
    train_and_save_v4_0()
