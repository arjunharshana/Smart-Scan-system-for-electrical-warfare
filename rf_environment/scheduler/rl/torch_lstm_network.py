from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn


class TorchLSTMQNetwork(nn.Module):
    """GPU-accelerated PyTorch Recurrent Q-Network (DRQN) with bidirectional NumPy weight mapping.

    Implements:
        x_t -> cuDNN LSTM -> ReLU(Dense(h_t)) -> Q(s_t, :)
    Matches the exact mathematical specification of NumPy LSTMQNetwork with 1-to-1 weight equivalence.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        dense_dim: int = 64,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dense_dim = int(dense_dim)

        if seed is not None:
            torch.manual_seed(seed)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=True,
        )

        # Standard forget gate bias initialization to 1.0 (matching NumPy LSTM)
        with torch.no_grad():
            self.lstm.bias_ih_l0[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)
            self.lstm.bias_hh_l0.zero_()

        self.dense = nn.Linear(self.hidden_dim, self.dense_dim, bias=True)
        self.out = nn.Linear(self.dense_dim, self.output_dim, bias=True)

        if device is not None:
            self.to(device)

    def init_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns zeros initial hidden and cell states [B, H]."""
        dev = device or next(self.parameters()).device
        h_0 = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=dev)
        c_0 = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=dev)
        return h_0, c_0

    def step(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward inference.
        x: [B, D], h_prev: [B, H], c_prev: [B, H] -> q: [B, N], h_next: [B, H], c_next: [B, H]
        """
        x_in = x.unsqueeze(1)  # [B, 1, D]
        hx = (h_prev.unsqueeze(0), c_prev.unsqueeze(0))
        lstm_out, (h_n, c_n) = self.lstm(x_in, hx)
        a = torch.relu(self.dense(lstm_out.squeeze(1)))
        q = self.out(a)
        return q, h_n.squeeze(0), c_n.squeeze(0)

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        h_0: torch.Tensor | None = None,
        c_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence forward pass.
        x_seq: [B, L, D] -> q_seq: [B, L, N], h_final: [B, H], c_final: [B, H]
        """
        if h_0 is not None and c_0 is not None:
            hx = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        else:
            hx = None
        lstm_out, (h_n, c_n) = self.lstm(x_seq, hx)
        a = torch.relu(self.dense(lstm_out))
        q_seq = self.out(a)
        return q_seq, h_n.squeeze(0), c_n.squeeze(0)

    def to_numpy_weights(self) -> dict[str, np.ndarray]:
        """Extracts and converts PyTorch weights into NumPy-format dictionary.
        Handles transposition and gate order mapping:
            PyTorch: [i, f, g, o] -> NumPy: [i, f, o, g]
        """
        w_ih = self.lstm.weight_ih_l0.detach().cpu().numpy()
        w_hh = self.lstm.weight_hh_l0.detach().cpu().numpy()
        b_ih = self.lstm.bias_ih_l0.detach().cpu().numpy()
        b_hh = self.lstm.bias_hh_l0.detach().cpu().numpy()
        b_tot = b_ih + b_hh

        D, H = self.input_dim, self.hidden_dim
        w_x = np.zeros((D, 4 * H), dtype=np.float32)
        w_h = np.zeros((H, 4 * H), dtype=np.float32)
        b_lstm = np.zeros(4 * H, dtype=np.float32)

        # i (0..H)
        w_x[:, 0:H] = w_ih[0:H, :].T
        w_h[:, 0:H] = w_hh[0:H, :].T
        b_lstm[0:H] = b_tot[0:H]

        # f (H..2H)
        w_x[:, H : 2 * H] = w_ih[H : 2 * H, :].T
        w_h[:, H : 2 * H] = w_hh[H : 2 * H, :].T
        b_lstm[H : 2 * H] = b_tot[H : 2 * H]

        # o (2H..3H) <- PyTorch 3H..4H
        w_x[:, 2 * H : 3 * H] = w_ih[3 * H : 4 * H, :].T
        w_h[:, 2 * H : 3 * H] = w_hh[3 * H : 4 * H, :].T
        b_lstm[2 * H : 3 * H] = b_tot[3 * H : 4 * H]

        # g (3H..4H) <- PyTorch 2H..3H
        w_x[:, 3 * H : 4 * H] = w_ih[2 * H : 3 * H, :].T
        w_h[:, 3 * H : 4 * H] = w_hh[2 * H : 3 * H, :].T
        b_lstm[3 * H : 4 * H] = b_tot[2 * H : 3 * H]

        # Dense head
        w_dense = self.dense.weight.detach().cpu().numpy().T
        b_dense = self.dense.bias.detach().cpu().numpy()

        # Out layer
        w_out = self.out.weight.detach().cpu().numpy().T
        b_out = self.out.bias.detach().cpu().numpy()

        return {
            "w_x": w_x,
            "w_h": w_h,
            "b_lstm": b_lstm,
            "w_dense": w_dense,
            "b_dense": b_dense,
            "w_out": w_out,
            "b_out": b_out,
        }

    def from_numpy_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Loads NumPy-format dictionary into PyTorch model parameters."""
        D, H = self.input_dim, self.hidden_dim
        w_x = np.asarray(weights["w_x"], dtype=np.float32)
        w_h = np.asarray(weights["w_h"], dtype=np.float32)
        b_lstm = np.asarray(weights["b_lstm"], dtype=np.float32)

        w_ih = np.zeros((4 * H, D), dtype=np.float32)
        w_hh = np.zeros((4 * H, H), dtype=np.float32)
        b_ih = np.zeros(4 * H, dtype=np.float32)

        # i (0..H)
        w_ih[0:H, :] = w_x[:, 0:H].T
        w_hh[0:H, :] = w_h[:, 0:H].T
        b_ih[0:H] = b_lstm[0:H]

        # f (H..2H)
        w_ih[H : 2 * H, :] = w_x[:, H : 2 * H].T
        w_hh[H : 2 * H, :] = w_h[:, H : 2 * H].T
        b_ih[H : 2 * H] = b_lstm[H : 2 * H]

        # g (2H..3H) <- NumPy 3H..4H
        w_ih[2 * H : 3 * H, :] = w_x[:, 3 * H : 4 * H].T
        w_hh[2 * H : 3 * H, :] = w_h[:, 3 * H : 4 * H].T
        b_ih[2 * H : 3 * H] = b_lstm[3 * H : 4 * H]

        # o (3H..4H) <- NumPy 2H..3H
        w_ih[3 * H : 4 * H, :] = w_x[:, 2 * H : 3 * H].T
        w_hh[3 * H : 4 * H, :] = w_h[:, 2 * H : 3 * H].T
        b_ih[3 * H : 4 * H] = b_lstm[2 * H : 3 * H]

        dev = next(self.parameters()).device
        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(torch.from_numpy(w_ih).to(dev))
            self.lstm.weight_hh_l0.copy_(torch.from_numpy(w_hh).to(dev))
            self.lstm.bias_ih_l0.copy_(torch.from_numpy(b_ih).to(dev))
            self.lstm.bias_hh_l0.zero_()

            self.dense.weight.copy_(torch.from_numpy(weights["w_dense"].T).to(dev))
            self.dense.bias.copy_(torch.from_numpy(weights["b_dense"]).to(dev))

            self.out.weight.copy_(torch.from_numpy(weights["w_out"].T).to(dev))
            self.out.bias.copy_(torch.from_numpy(weights["b_out"]).to(dev))

    def clone(self) -> TorchLSTMQNetwork:
        """Creates an identical clone with detached weights on the same device."""
        dev = next(self.parameters()).device
        cloned = TorchLSTMQNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            dense_dim=self.dense_dim,
            device=dev,
        )
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned

    def copy_from(self, source: TorchLSTMQNetwork) -> None:
        """Copies weights from source network directly."""
        with torch.no_grad():
            for p_dest, p_src in zip(self.parameters(), source.parameters()):
                p_dest.copy_(p_src)

    def save_torch_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Saves PyTorch state dict and architecture metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "architecture": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "dense_dim": self.dense_dim,
            },
            "metadata": metadata or {},
        }
        torch.save(payload, p)
        return p

    def save_as_numpy_checkpoint(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        target_net: TorchLSTMQNetwork | None = None,
        bands_hz: list[float] | None = None,
    ) -> Path:
        """Exports weights into standard V4.1 NumPy .npz checkpoint format.
        Guarantees 100% plug-and-play compatibility with production runtime and load_checkpoint().
        """
        import datetime
        from rf_environment.scheduler.rl.checkpoint import (
            CHECKPOINT_FORMAT_VERSION,
            REQUIRED_WEIGHT_KEYS,
            compute_weights_sha256,
            _audit_metadata_for_ground_truth,
        )

        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        online_np = self.to_numpy_weights()
        target_np = target_net.to_numpy_weights() if target_net is not None else copy.deepcopy(online_np)

        weights_sha256 = compute_weights_sha256(online_np)
        bands = [float(b) for b in bands_hz] if bands_hz is not None else [float(i) for i in range(self.output_dim)]

        meta_payload: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "scheduler_type": "lstm_ddqn",
            "model_architecture": "LSTMQNetwork",
            "architecture": {
                "input_dim": int(self.input_dim),
                "output_dim": int(self.output_dim),
                "hidden_dim": int(self.hidden_dim),
                "dense_dim": int(self.dense_dim),
                "sequence_length": 10,
                "burn_in": 0,
                "num_bins": len(bands),
            },
            "encoder": {
                "type": "encoder_b",
                "feature_dim": int(self.input_dim),
                "history_length": 10,
            },
            "bands_hz": bands,
            "weights_sha256": weights_sha256,
            "total_parameters": int(sum(arr.size for arr in online_np.values())),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "custom_metadata": metadata or {},
        }

        _audit_metadata_for_ground_truth(meta_payload)

        arrays_to_save: dict[str, np.ndarray] = {
            "__metadata__": np.array(json.dumps(meta_payload), dtype=object),
        }
        for k in REQUIRED_WEIGHT_KEYS:
            arrays_to_save[f"online_{k}"] = np.ascontiguousarray(online_np[k])
            arrays_to_save[f"target_{k}"] = np.ascontiguousarray(target_np[k])

        tmp_path = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **arrays_to_save)
        tmp_path.replace(p)
        return p

