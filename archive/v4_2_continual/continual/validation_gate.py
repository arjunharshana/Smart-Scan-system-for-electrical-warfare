from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.scheduler.hybrid.continual.config import V42ContinualConfig
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork


class ValidationGate:
    """Candidate-model validation gate protecting production weights from degradation.

    Evaluates candidate recurrent model against the current production baseline on
    independent validation batches sampled from the online replay buffer.

    Acceptance Rules:
    1. Recent Regime: Candidate must not degrade recent Bellman loss relative to baseline
       (loss_candidate <= loss_baseline * (1.0 + margin) or loss_candidate is absolute low).
    2. Historical Retention: Candidate historical Bellman loss must NOT exceed baseline
       historical loss by more than max_historical_loss_degradation (e.g. 1.25x).

    Invariants:
    1. Strictly zero ground-truth access.
    2. Read-only evaluation: leaves candidate and production networks unmodified.
    """

    def __init__(self, config: V42ContinualConfig | None = None) -> None:
        self.config = config or V42ContinualConfig()

    def evaluate(
        self,
        candidate_net: LSTMQNetwork,
        production_net: LSTMQNetwork,
        target_net: LSTMQNetwork,
        recent_val_batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        historical_val_batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        gamma: float = 0.95,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluates candidate model against production baseline."""
        # 1. Evaluate Recent Regime Loss
        loss_cand_recent = self._compute_bellman_loss(candidate_net, target_net, recent_val_batch, gamma)
        loss_prod_recent = self._compute_bellman_loss(production_net, target_net, recent_val_batch, gamma)

        # 2. Evaluate Historical Retention Loss (if historical data exists)
        has_historical = historical_val_batch is not None
        if has_historical:
            loss_cand_hist = self._compute_bellman_loss(candidate_net, target_net, historical_val_batch, gamma)
            loss_prod_hist = self._compute_bellman_loss(production_net, target_net, historical_val_batch, gamma)
            hist_ratio = float(loss_cand_hist) / float(max(loss_prod_hist, 1e-4))
        else:
            loss_cand_hist = 0.0
            loss_prod_hist = 0.0
            hist_ratio = 1.0

        # 3. Acceptance Logic
        # A. Check historical retention: must not severely degrade on old regime
        historical_pass = True
        if has_historical and hist_ratio > self.config.max_historical_loss_degradation:
            historical_pass = False

        # B. Check recent improvement/parity: candidate should adapt well to new regime
        recent_pass = True
        if loss_cand_recent > (loss_prod_recent * (1.0 + self.config.recent_loss_improvement_margin) + 0.05):
            recent_pass = False

        accepted = historical_pass and recent_pass

        if not historical_pass:
            decision_reason = f"REJECT: Catastrophic forgetting on historical data (ratio {hist_ratio:.2f} > {self.config.max_historical_loss_degradation:.2f})"
        elif not recent_pass:
            decision_reason = f"REJECT: Candidate failed to improve new regime (loss {loss_cand_recent:.4f} > {loss_prod_recent:.4f})"
        else:
            decision_reason = f"ACCEPT: Candidate passed (recent loss: {loss_cand_recent:.4f} vs {loss_prod_recent:.4f}, hist ratio: {hist_ratio:.2f})"

        telemetry = {
            "decision": "ACCEPT" if accepted else "REJECT",
            "reason": decision_reason,
            "loss_cand_recent": float(loss_cand_recent),
            "loss_prod_recent": float(loss_prod_recent),
            "loss_cand_hist": float(loss_cand_hist),
            "loss_prod_hist": float(loss_prod_hist),
            "hist_ratio": float(hist_ratio),
            "historical_pass": bool(historical_pass),
            "recent_pass": bool(recent_pass),
        }

        return accepted, telemetry

    def _compute_bellman_loss(
        self,
        online_net: LSTMQNetwork,
        target_net: LSTMQNetwork,
        batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        gamma: float,
    ) -> float:
        b_states, b_actions, b_rewards, b_next_states, b_dones = batch

        # Forward online sequence
        q_online, _, _ = online_net.forward_sequence(b_states)
        pred_q = np.take_along_axis(q_online, b_actions[:, :, np.newaxis], axis=2).squeeze(2)

        # Target sequence calculation via Double DQN
        next_q_online, _, _ = online_net.forward_sequence(b_next_states)
        best_next_actions = np.argmax(next_q_online, axis=2)

        next_q_target, _, _ = target_net.forward_sequence(b_next_states)
        target_next_q = np.take_along_axis(
            next_q_target, best_next_actions[:, :, np.newaxis], axis=2
        ).squeeze(2)

        targets = b_rewards + gamma * (1.0 - b_dones) * target_next_q
        td_errors = pred_q - targets
        return float(np.mean(td_errors ** 2))
