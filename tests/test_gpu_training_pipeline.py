from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch

from rf_environment.domain.state import SchedulerObservation
from rf_environment.domain.action import ScanAction
from rf_environment.scheduler.rl.checkpoint import load_checkpoint
from rf_environment.scheduler.rl.checkpoint_validator import (
    verify_checkpoint_equivalence,
    verify_checkpoint_file_equivalence,
)
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork
from rf_environment.scheduler.rl.torch_lstm_network import TorchLSTMQNetwork
from rf_environment.scheduler.rl.torch_lstm_ddqn_scheduler import TorchLSTMDDQNScheduler


def test_cuda_hardware_sanity() -> None:
    """Verifies that CUDA is available and PyTorch can allocate and compute on GPU."""
    assert torch.cuda.is_available(), "CUDA should be available on this system."
    device = torch.device("cuda:0")
    a = torch.randn(10, 10, device=device)
    b = torch.randn(10, 10, device=device)
    c = torch.matmul(a, b)
    assert c.is_cuda
    assert c.shape == (10, 10)


def _check_weight_conversion_for_hidden(hidden_dim: int) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = 16
    output_dim = 10
    dense_dim = 64

    torch_net = TorchLSTMQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dense_dim=dense_dim,
        device=device,
        seed=123 + hidden_dim,
    )

    numpy_weights = torch_net.to_numpy_weights()

    numpy_net = LSTMQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dense_dim=dense_dim,
    )
    numpy_net.load_weights_dict(numpy_weights)

    # Test numerical equivalence
    metrics = verify_checkpoint_equivalence(
        torch_net=torch_net,
        numpy_net=numpy_net,
        num_samples=50,
        seq_length=10,
        tolerance=1e-4,
        seed=42,
    )
    assert metrics["is_equivalent"]
    assert metrics["max_abs_diff"] < 1e-4


def test_bidirectional_weight_conversion_h64() -> None:
    _check_weight_conversion_for_hidden(64)


def test_bidirectional_weight_conversion_h128() -> None:
    _check_weight_conversion_for_hidden(128)


def test_bidirectional_weight_conversion_h256() -> None:
    _check_weight_conversion_for_hidden(256)


def test_checkpoint_export_import_roundtrip() -> None:
    """Verifies that saving a PyTorch network as a NumPy .npz checkpoint
    and loading via canonical load_checkpoint succeeds and verifies equivalence.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_net = TorchLSTMQNetwork(
        input_dim=16,
        output_dim=10,
        hidden_dim=128,
        dense_dim=64,
        device=device,
        seed=456,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_checkpoint.npz"
        torch_net.save_as_numpy_checkpoint(
            path=ckpt_path,
            metadata={"test_run": "sanity"},
            bands_hz=[1e6 * i for i in range(10)],
        )

        assert ckpt_path.exists()

        # Load using canonical load_checkpoint
        loaded_data = load_checkpoint(ckpt_path)
        assert loaded_data["metadata"]["architecture"]["hidden_dim"] == 128
        assert loaded_data["sha256"] is not None

        # Verify Q-value equivalence
        res = verify_checkpoint_file_equivalence(ckpt_path, torch_net, num_samples=30)
        assert res["is_equivalent"]
        assert res["max_abs_diff"] < 1e-4


def test_torch_lstm_ddqn_scheduler_training_step() -> None:
    """Verifies that TorchLSTMDDQNScheduler trains cleanly on GPU with valid telemetry."""
    bands = [1e6 * i for i in range(10)]
    sched = TorchLSTMDDQNScheduler(
        bands_hz=bands,
        hidden_dim=64,
        dense_dim=64,
        sequence_length=5,
        burn_in=0,
        use_stored_hidden=True,
        regime_balanced_replay=False,
        warmup_steps=10,
        batch_size=8,
        seed=789,
    )

    sched.train()
    rng = np.random.default_rng(789)

    def make_obs(step: int) -> SchedulerObservation:
        return SchedulerObservation(
            timestamp=float(step) * 0.001,
            current_frequency_bin=step % 10,
            last_detection=(step % 3 == 0),
            last_detection_bin=step % 10 if (step % 3 == 0) else None,
            last_detection_strength=-65.0 if (step % 3 == 0) else None,
            recent_detection_history=tuple([(i % 3 == 0) for i in range(10)]),
            recent_frequency_history=tuple([i % 10 for i in range(10)]),
            scan_count_by_bin=tuple([step + 1] * 10),
            time_since_scan_by_bin=tuple([float(step) * 0.1] * 10),
            time_since_last_detection=0.1,
        )

    # Push enough transitions to trigger training
    for step in range(30):
        obs = make_obs(step)
        action = ScanAction(frequency_bin=int(rng.integers(0, 10)))
        reward = float(rng.uniform(-1.0, 1.0))
        next_obs = make_obs(step + 1)
        done = (step % 15 == 14)

        sched.update_policy(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )

    assert sched.train_step_count > 0
    assert len(sched.losses) > 0
    assert len(sched.grad_norms) > 0
    assert not np.isnan(sched.last_loss)

    # Export to NumPy scheduler and verify inference
    np_sched = sched.to_numpy_scheduler()
    eval_obs = make_obs(100)
    np_sched.eval()
    selected_bin = np_sched.select_bin(eval_obs)
    assert 0 <= selected_bin < 10



def test_ground_truth_firewall() -> None:
    """Verifies that attempting to save ground-truth metadata is intercepted and blocked."""
    torch_net = TorchLSTMQNetwork(input_dim=16, output_dim=10, hidden_dim=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "gt_violation.npz"
        forbidden_metadatas = [
            {"emitter": "radar_1"},
            {"gt_frequencies": [1e6]},
            {"true_power": -20.0},
        ]
        for bad_meta in forbidden_metadatas:
            try:
                torch_net.save_as_numpy_checkpoint(ckpt_path, metadata=bad_meta)
                failed = False
            except ValueError as e:
                failed = True
                assert "Ground truth firewall violation" in str(e)
            assert failed, f"Failed to block forbidden metadata: {bad_meta}"
