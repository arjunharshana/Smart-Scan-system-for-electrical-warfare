"""rf_environment/scheduler/rl/checkpoint_v40.py
Production serialization, loading, and integrity verification for V4.0 Feed-Forward DDQN.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_FORMAT_VERSION = "4.0"
REQUIRED_WEIGHT_KEYS = [
    "w1",
    "b1",
    "w2",
    "b2",
    "w3",
    "b3",
]

FORBIDDEN_GROUND_TRUTH_KEYS = {
    "emitter",
    "emitters",
    "ground_truth",
    "gt",
    "realization",
    "carrier_frequency_hz",
    "transmitting",
    "true_power",
    "snr",
    "sinr",
}


def compute_v40_weights_sha256(weights: dict[str, np.ndarray]) -> str:
    """Computes a deterministic SHA256 hexadecimal digest of MLPQNetwork parameters."""
    hasher = hashlib.sha256()
    for key in REQUIRED_WEIGHT_KEYS:
        if key in weights:
            hasher.update(np.ascontiguousarray(weights[key]).tobytes())
    return hasher.hexdigest()


def _audit_metadata_for_ground_truth(meta: Any, prefix: str = "") -> None:
    """Ensures zero ground-truth or evaluator state is stored in checkpoint metadata."""
    if isinstance(meta, dict):
        for k, v in meta.items():
            k_lower = str(k).lower()
            tokens = set(k_lower.replace("-", "_").split("_"))
            if tokens & FORBIDDEN_GROUND_TRUTH_KEYS or k_lower in FORBIDDEN_GROUND_TRUTH_KEYS:
                raise ValueError(
                    f"Ground truth firewall violation: checkpoint metadata contains forbidden key '{prefix}{k}'"
                )
            _audit_metadata_for_ground_truth(v, prefix=f"{prefix}{k}.")
    elif isinstance(meta, (list, tuple)):
        for i, item in enumerate(meta):
            _audit_metadata_for_ground_truth(item, prefix=f"{prefix}[{i}].")


def save_v40_checkpoint(
    scheduler: Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serializes a trained DDQNScheduler into a versioned, verifiable .npz checkpoint.

    Strict Invariants:
    1. Zero ground truth or evaluator state.
    2. Complete parameter persistence for both online and target networks.
    3. Cryptographic SHA256 integrity verification.
    """
    save_path = Path(path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    online_net = getattr(scheduler, "online_net", None)
    target_net = getattr(scheduler, "target_net", None)
    if online_net is None or target_net is None:
        raise ValueError("Scheduler must have 'online_net' and 'target_net' attributes to serialize.")

    # 1. Extract weights
    online_weights = {
        "w1": online_net.w1,
        "b1": online_net.b1,
        "w2": online_net.w2,
        "b2": online_net.b2,
        "w3": online_net.w3,
        "b3": online_net.b3,
    }
    target_weights = {
        "w1": target_net.w1,
        "b1": target_net.b1,
        "w2": target_net.w2,
        "b2": target_net.b2,
        "w3": target_net.w3,
        "b3": target_net.b3,
    }

    # Compute SHA256 fingerprint
    weights_sha256 = compute_v40_weights_sha256(online_weights)
    if hasattr(scheduler, "checkpoint_sha256"):
        scheduler.checkpoint_sha256 = weights_sha256
    scheduler.checkpoint_path = str(save_path)
    scheduler.is_pretrained = True

    # 2. Build metadata
    num_bins = len(getattr(scheduler, "bands_hz", [])) or online_net.output_dim
    bands_hz = [float(b) for b in getattr(scheduler, "bands_hz", [])]

    meta_payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "scheduler_type": getattr(scheduler, "name", "ddqn"),
        "model_architecture": "MLPQNetwork",
        "architecture": {
            "input_dim": int(online_net.input_dim),
            "output_dim": int(online_net.output_dim),
            "hidden_dim": int(online_net.hidden_dim),
            "num_bins": num_bins,
        },
        "encoder": {
            "type": "encoder_b" if hasattr(scheduler.encoder, "history_length") else "encoder_a",
            "feature_dim": int(getattr(scheduler.encoder, "feature_dim", online_net.input_dim)),
            "history_length": int(getattr(scheduler.encoder, "history_length", 10)),
        },
        "bands_hz": bands_hz,
        "weights_sha256": weights_sha256,
        "total_parameters": int(
            online_net.w1.size + online_net.b1.size +
            online_net.w2.size + online_net.b2.size +
            online_net.w3.size + online_net.b3.size
        ),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "custom_metadata": metadata or {},
    }

    # Audit ground truth firewall
    _audit_metadata_for_ground_truth(meta_payload)

    # 3. Assemble numpy array bundle
    arrays_to_save: dict[str, np.ndarray] = {
        "__metadata__": np.array(json.dumps(meta_payload), dtype=object),
    }
    for k, v in online_weights.items():
        arrays_to_save[f"online_{k}"] = np.ascontiguousarray(v)
    for k, v in target_weights.items():
        arrays_to_save[f"target_{k}"] = np.ascontiguousarray(v)

    # 4. Atomic write
    tmp_path = save_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **arrays_to_save)
    tmp_path.replace(save_path)

    return save_path


def load_v40_checkpoint(
    path: str | Path,
    expected_input_dim: int | None = None,
    expected_output_dim: int | None = None,
    expected_bands_hz: list[float] | None = None,
) -> dict[str, Any]:
    """Loads and strictly validates a V4.0 Feed-Forward DDQN checkpoint.

    Fails loudly if the file is missing, corrupt, dimensionally incompatible,
    or has an invalid cryptographic hash.
    """
    ckpt_path = Path(path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"V4.0 CHECKPOINT NOT FOUND — INFERENCE UNAVAILABLE\n"
            f"Expected checkpoint at: {ckpt_path}\n"
            f"Refusing to start with random/untrained weights."
        )

    try:
        data = np.load(ckpt_path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"Failed to open checkpoint file '{ckpt_path}': {exc}") from exc

    if "__metadata__" not in data:
        raise ValueError(f"Invalid checkpoint '{ckpt_path}': missing '__metadata__' entry.")

    raw_meta = str(data["__metadata__"])
    try:
        meta = json.loads(raw_meta)
    except Exception as exc:
        raise ValueError(f"Corrupt checkpoint metadata in '{ckpt_path}': {exc}") from exc

    # 1. Format Version Validation
    fmt_ver = meta.get("format_version")
    if fmt_ver != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format version mismatch: expected '{CHECKPOINT_FORMAT_VERSION}', got '{fmt_ver}'"
        )

    # 2. Architecture Validation
    arch = meta.get("architecture", {})
    in_dim = arch.get("input_dim")
    out_dim = arch.get("output_dim")

    if expected_input_dim is not None and in_dim != expected_input_dim:
        raise ValueError(
            f"Checkpoint input dimension mismatch: model expects {expected_input_dim}, checkpoint has {in_dim}"
        )
    if expected_output_dim is not None and out_dim != expected_output_dim:
        raise ValueError(
            f"Checkpoint output dimension mismatch: model expects {expected_output_dim}, checkpoint has {out_dim}"
        )

    # 3. Extract and Verify Weights
    online_weights = {}
    target_weights = {}
    for key in REQUIRED_WEIGHT_KEYS:
        on_k = f"online_{key}"
        tg_k = f"target_{key}"
        if on_k not in data:
            raise ValueError(f"Checkpoint missing required weight tensor '{on_k}'")
        if tg_k not in data:
            raise ValueError(f"Checkpoint missing required weight tensor '{tg_k}'")
        online_weights[key] = np.array(data[on_k], copy=True)
        target_weights[key] = np.array(data[tg_k], copy=True)

    # 4. Cryptographic Hash Verification
    saved_sha = meta.get("weights_sha256")
    actual_sha = compute_v40_weights_sha256(online_weights)
    if saved_sha and actual_sha != saved_sha:
        raise ValueError(
            f"Cryptographic integrity verification failed for '{ckpt_path}':\n"
            f"  Recorded SHA256: {saved_sha}\n"
            f"  Computed SHA256: {actual_sha}\n"
            f"The checkpoint weights have been modified or corrupted."
        )

    return {
        "metadata": meta,
        "online_weights": online_weights,
        "target_weights": target_weights,
        "weights_sha256": actual_sha,
        "path": str(ckpt_path),
    }
