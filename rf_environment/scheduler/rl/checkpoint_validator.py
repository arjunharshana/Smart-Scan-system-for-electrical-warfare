from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch

from rf_environment.scheduler.rl.checkpoint import load_checkpoint
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork
from rf_environment.scheduler.rl.torch_lstm_network import TorchLSTMQNetwork


def verify_checkpoint_equivalence(
    torch_net: TorchLSTMQNetwork,
    numpy_net: LSTMQNetwork,
    num_samples: int = 100,
    seq_length: int = 10,
    tolerance: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Tests numerical Q-value equivalence between PyTorch and NumPy networks across random inputs.

    Generates random test sequences, evaluates both networks, and computes discrepancy metrics.
    Raises AssertionError if max_abs_diff exceeds tolerance.
    """
    rng = np.random.default_rng(seed)
    input_dim = torch_net.input_dim
    batch_size = num_samples

    # Generate synthetic input sequences: [B, L, D]
    x_test = rng.standard_normal((batch_size, seq_length, input_dim)).astype(np.float32)

    # 1. First verify single-step step() numerical equivalence (strict tolerance 1e-4)
    torch_net.eval()
    dev = next(torch_net.parameters()).device
    with torch.no_grad():
        x_single = torch.from_numpy(x_test[:, 0, :]).to(dev)
        h0_single = torch.zeros((batch_size, torch_net.hidden_dim), dtype=torch.float32, device=dev)
        c0_single = torch.zeros((batch_size, torch_net.hidden_dim), dtype=torch.float32, device=dev)
        q_torch_1, _, _ = torch_net.step(x_single, h0_single, c0_single)
        q_torch_1_np = q_torch_1.detach().cpu().numpy()

    q_numpy_1_np = np.zeros_like(q_torch_1_np)
    for b in range(batch_size):
        h = np.zeros(numpy_net.hidden_dim, dtype=np.float32)
        c = np.zeros(numpy_net.hidden_dim, dtype=np.float32)
        q_step, _, _ = numpy_net.step(x_test[b, 0], h, c)
        q_numpy_1_np[b] = q_step

    single_step_diff = float(np.max(np.abs(q_torch_1_np - q_numpy_1_np)))
    if single_step_diff > tolerance:
        raise AssertionError(
            f"Single-step Q-value equivalence check FAILED: max_abs_diff={single_step_diff:.6e} > tolerance={tolerance:.6e}"
        )


    # 2. PyTorch forward sequence pass
    with torch.no_grad():
        x_torch = torch.from_numpy(x_test).to(dev)
        q_torch_seq, h_torch_final, c_torch_final = torch_net.forward_sequence(x_torch)
        q_torch_np = q_torch_seq.detach().cpu().numpy()
        h_torch_np = h_torch_final.detach().cpu().numpy()
        c_torch_np = c_torch_final.detach().cpu().numpy()

    # 3. NumPy unroll step-by-step
    q_numpy_seq = np.zeros_like(q_torch_np)
    h_numpy_final = np.zeros((batch_size, numpy_net.hidden_dim), dtype=np.float32)
    c_numpy_final = np.zeros((batch_size, numpy_net.hidden_dim), dtype=np.float32)

    for b in range(batch_size):
        h = np.zeros(numpy_net.hidden_dim, dtype=np.float32)
        c = np.zeros(numpy_net.hidden_dim, dtype=np.float32)
        for t in range(seq_length):
            x_step = x_test[b, t]
            q_step, h, c = numpy_net.step(x_step, h, c)
            q_numpy_seq[b, t] = q_step
        h_numpy_final[b] = h
        c_numpy_final[b] = c

    # 4. Discrepancy analysis
    abs_diff = np.abs(q_torch_np - q_numpy_seq)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    norm_target = np.maximum(np.abs(q_torch_np), np.abs(q_numpy_seq))
    relative_err = float(np.max(abs_diff / (norm_target + 1e-7)))

    h_abs_diff = float(np.max(np.abs(h_torch_np - h_numpy_final)))
    c_abs_diff = float(np.max(np.abs(c_torch_np - c_numpy_final)))

    is_equivalent = bool(max_abs_diff < tolerance)

    results = {
        "is_equivalent": is_equivalent,
        "single_step_max_abs_diff": single_step_diff,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_err": relative_err,
        "h_max_abs_diff": h_abs_diff,
        "c_max_abs_diff": c_abs_diff,
        "num_samples": num_samples,
        "seq_length": seq_length,
        "tolerance": tolerance,
    }

    if not is_equivalent:
        raise AssertionError(
            f"Checkpoint numerical equivalence check FAILED: max_abs_diff={max_abs_diff:.6e} > tolerance={tolerance:.6e}"
        )

    return results



def verify_checkpoint_file_equivalence(
    checkpoint_path: str | Path,
    torch_net: TorchLSTMQNetwork,
    num_samples: int = 100,
    seq_length: int = 10,
    tolerance: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Validates that a saved .npz checkpoint loads cleanly into NumPy LSTMQNetwork
    and matches the originating PyTorch network's Q-values within tolerance.
    """
    ckpt_data = load_checkpoint(
        path=checkpoint_path,
        expected_input_dim=torch_net.input_dim,
        expected_output_dim=torch_net.output_dim,
    )

    numpy_net = LSTMQNetwork(
        input_dim=torch_net.input_dim,
        output_dim=torch_net.output_dim,
        hidden_dim=torch_net.hidden_dim,
        dense_dim=torch_net.dense_dim,
    )
    numpy_net.load_weights_dict(ckpt_data["online_weights"])

    results = verify_checkpoint_equivalence(
        torch_net=torch_net,
        numpy_net=numpy_net,
        num_samples=num_samples,
        seq_length=seq_length,
        tolerance=tolerance,
        seed=seed,
    )
    results["checkpoint_path"] = str(checkpoint_path)
    results["sha256"] = ckpt_data["sha256"]
    return results
