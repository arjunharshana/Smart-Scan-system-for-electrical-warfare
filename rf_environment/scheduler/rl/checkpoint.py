from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_FORMAT_VERSION = "4.1"
REQUIRED_WEIGHT_KEYS = [
    "w_x",
    "w_h",
    "b_lstm",
    "w_dense",
    "b_dense",
    "w_out",
    "b_out",
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


def compute_weights_sha256(weights: dict[str, np.ndarray] | list[np.ndarray]) -> str:
    """Computes a deterministic SHA256 hexadecimal digest of neural network parameters."""
    hasher = hashlib.sha256()
    if isinstance(weights, dict):
        for key in REQUIRED_WEIGHT_KEYS:
            if key in weights:
                hasher.update(np.ascontiguousarray(weights[key]).tobytes())
    else:
        for p in weights:
            hasher.update(np.ascontiguousarray(p).tobytes())
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


def save_checkpoint(
    scheduler: Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serializes a trained LSTMDDQNScheduler into a versioned, verifiable .npz checkpoint.

    Strict Invariants:
    1. Zero ground truth or evaluator state.
    2. Zero runtime episode memory (h, c are omitted; reset to zero on load).
    3. Complete parameter persistence for both online and target networks.
    4. Cryptographic SHA256 integrity verification.
    """
    save_path = Path(path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    online_net = getattr(scheduler, "online_net", None)
    target_net = getattr(scheduler, "target_net", None)
    if online_net is None or target_net is None:
        raise ValueError("Scheduler must have 'online_net' and 'target_net' attributes to serialize.")

    # 1. Extract weights
    online_weights = {
        "w_x": online_net.w_x,
        "w_h": online_net.w_h,
        "b_lstm": online_net.b_lstm,
        "w_dense": online_net.w_dense,
        "b_dense": online_net.b_dense,
        "w_out": online_net.w_out,
        "b_out": online_net.b_out,
    }
    target_weights = {
        "w_x": target_net.w_x,
        "w_h": target_net.w_h,
        "b_lstm": target_net.b_lstm,
        "w_dense": target_net.w_dense,
        "b_dense": target_net.b_dense,
        "w_out": target_net.w_out,
        "b_out": target_net.b_out,
    }

    # Compute SHA256 fingerprint
    weights_sha256 = compute_weights_sha256(online_weights)
    if hasattr(scheduler, "checkpoint_sha256"):
        scheduler.checkpoint_sha256 = weights_sha256
    scheduler.checkpoint_path = str(save_path)
    scheduler.is_pretrained = True

    # 2. Build metadata
    num_bins = len(getattr(scheduler, "bands_hz", [])) or online_net.output_dim
    bands_hz = [float(b) for b in getattr(scheduler, "bands_hz", [])]

    meta_payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "scheduler_type": getattr(scheduler, "name", "lstm_ddqn"),
        "model_architecture": "LSTMQNetwork",
        "architecture": {
            "input_dim": int(online_net.input_dim),
            "output_dim": int(online_net.output_dim),
            "hidden_dim": int(online_net.hidden_dim),
            "dense_dim": int(online_net.dense_dim),
            "sequence_length": int(getattr(scheduler, "sequence_length", 10)),
            "burn_in": int(getattr(scheduler, "burn_in", 0)),
            "num_bins": num_bins,
        },
        "encoder": {
            "type": "encoder_b",
            "feature_dim": int(getattr(scheduler.encoder, "feature_dim", online_net.input_dim)),
            "history_length": int(getattr(scheduler.encoder, "history_length", 10)),
        },
        "bands_hz": bands_hz,
        "weights_sha256": weights_sha256,
        "total_parameters": int(sum(p.size for p in online_net.params)),
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


def load_checkpoint(
    path: str | Path,
    expected_input_dim: int | None = None,
    expected_output_dim: int | None = None,
    expected_bands_hz: list[float] | None = None,
) -> dict[str, Any]:
    """Loads and strictly validates a V4.1 LSTM-DDQN checkpoint.

    Fails loudly if the file is missing, corrupt, dimensionally incompatible,
    or has an invalid cryptographic hash.
    """
    ckpt_path = Path(path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"V4.1 production checkpoint not found at: {ckpt_path}\n"
            f"Refusing to start with random/untrained LSTM weights."
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

    # 2. Architecture Dimensionality Validation
    arch = meta.get("architecture", {})
    in_dim = arch.get("input_dim")
    out_dim = arch.get("output_dim")

    if expected_input_dim is not None and in_dim != expected_input_dim:
        raise ValueError(
            f"Architecture mismatch in '{ckpt_path}': checkpoint input_dim={in_dim} "
            f"!= expected input_dim={expected_input_dim}"
        )

    if expected_output_dim is not None and out_dim != expected_output_dim:
        raise ValueError(
            f"Action space mismatch in '{ckpt_path}': checkpoint output_dim={out_dim} "
            f"!= expected output_dim={expected_output_dim}"
        )

    if expected_bands_hz is not None and len(expected_bands_hz) != out_dim:
        raise ValueError(
            f"Frequency bands mismatch in '{ckpt_path}': len(bands_hz)={len(expected_bands_hz)} "
            f"!= checkpoint action dimension={out_dim}"
        )

    # 3. Extract and Validate Weight Arrays
    online_weights: dict[str, np.ndarray] = {}
    target_weights: dict[str, np.ndarray] = {}

    for k in REQUIRED_WEIGHT_KEYS:
        on_key = f"online_{k}"
        tgt_key = f"target_{k}"
        if on_key not in data or tgt_key not in data:
            raise ValueError(f"Incomplete checkpoint '{ckpt_path}': missing array '{on_key}' or '{tgt_key}'.")
        online_weights[k] = np.asarray(data[on_key], dtype=np.float32)
        target_weights[k] = np.asarray(data[tgt_key], dtype=np.float32)

    # 4. Cryptographic Hash Validation
    expected_hash = meta.get("weights_sha256")
    actual_hash = compute_weights_sha256(online_weights)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Checkpoint corruption detected in '{ckpt_path}':\n"
            f"Metadata SHA256: {expected_hash}\n"
            f"Computed SHA256: {actual_hash}"
        )

    return {
        "path": str(ckpt_path),
        "metadata": meta,
        "online_weights": online_weights,
        "target_weights": target_weights,
        "sha256": actual_hash,
    }
