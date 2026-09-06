from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.hybrid.arbitrator import ArbitrationMode, HybridMetaArbitrator
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.util.seeding import derive_seed


class LSTMHybridScheduler(BaseScheduler):
    """V4.1 Hybrid Context-Aware + LSTM-DDQN Scheduler with Rule-Based Meta-Arbitration.

    Integrates genuine temporal working memory via an LSTM Recurrent Q-Network inside the
    predictive branch, while retaining Context-Aware's empirical online counts and the
    transparent rule-based Meta-Arbitrator.

    Architectural Invariants:
    1. Consumes strictly canonical SchedulerObservation (zero ground truth).
    2. Emits strictly canonical ScanAction(frequency_bin) over common action space {0..N-1}.
    3. Maintains hidden-state lifecycle: reset() clears (h, c) to zero with no cross-episode leakage.
    4. Offline training only: during evaluation, LSTM-DDQN weights remain frozen while CA and
       arbitrator adapt online to detector observables.
    """

    name = "hybrid_v41"
    category = "hybrid"

    def __init__(
        self,
        bands_hz: list[float],
        context_aware: ContextAwareScheduler | None = None,
        lstm_ddqn: LSTMDDQNScheduler | None = None,
        arbitrator: HybridMetaArbitrator | None = None,
        ca_config: dict[str, Any] | None = None,
        lstm_ddqn_config: dict[str, Any] | None = None,
        arbitrator_config: dict[str, Any] | None = None,
        seed: int | None = None,
        checkpoint_path: str | Path | None = None,
        require_checkpoint: bool = False,
    ) -> None:
        super().__init__(bands_hz)
        self.num_bins = len(self.bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 1. Context-Aware Adaptive Branch
        ca_cfg = ca_config or {}
        ca_seed = derive_seed(seed, "context_aware") if seed is not None else None
        self.context_aware = context_aware or ContextAwareScheduler(
            bands_hz=self.bands_hz,
            w_transition=float(ca_cfg.get("w_transition", 0.55)),
            w_activity=float(ca_cfg.get("w_activity", 0.25)),
            w_exploration=float(ca_cfg.get("w_exploration", 0.20)),
            smoothing=float(ca_cfg.get("smoothing", 0.05)),
            stale_threshold=int(ca_cfg.get("stale_threshold", 15)),
            seed=ca_seed,
        )

        # 2. Recurrent LSTM-DDQN Predictive Branch
        lstm_cfg = lstm_ddqn_config or {}
        lstm_seed = derive_seed(seed, "lstm_ddqn") if seed is not None else None
        ckpt_path = checkpoint_path or lstm_cfg.get("checkpoint_path")
        req_ckpt = require_checkpoint or bool(lstm_cfg.get("require_checkpoint", False))

        if lstm_ddqn is not None:
            self.lstm_ddqn = lstm_ddqn
            if ckpt_path is not None:
                self.lstm_ddqn.load_checkpoint(ckpt_path)
        else:
            self.lstm_ddqn = LSTMDDQNScheduler(
                bands_hz=self.bands_hz,
                hidden_dim=int(lstm_cfg.get("hidden_dim", 64)),
                dense_dim=int(lstm_cfg.get("dense_dim", 64)),
                sequence_length=int(lstm_cfg.get("sequence_length", 10)),
                burn_in=int(lstm_cfg.get("burn_in", 0)),
                learning_rate=float(lstm_cfg.get("learning_rate", 0.001)),
                gamma=float(lstm_cfg.get("gamma", 0.95)),
                replay_capacity=int(lstm_cfg.get("replay_capacity", 10000)),
                batch_size=int(lstm_cfg.get("batch_size", 32)),
                warmup_steps=int(lstm_cfg.get("warmup_steps", 64)),
                target_update_frequency=int(lstm_cfg.get("target_update_frequency", 100)),
                epsilon_start=float(lstm_cfg.get("epsilon_start", 1.0)),
                epsilon_end=float(lstm_cfg.get("epsilon_end", 0.05)),
                epsilon_decay=float(lstm_cfg.get("epsilon_decay", 0.995)),
                seed=lstm_seed,
                checkpoint_path=ckpt_path,
                require_checkpoint=req_ckpt,
            )

        # 3. Rule-Based Meta-Arbitrator
        arb_cfg = arbitrator_config or {}
        self.arbitrator = arbitrator or HybridMetaArbitrator(
            num_bins=self.num_bins,
            ddqn_conf_threshold=float(arb_cfg.get("ddqn_conf_threshold", 0.25)),
            ca_conf_threshold=float(arb_cfg.get("ca_conf_threshold", 0.30)),
            surprise_threshold=float(arb_cfg.get("surprise_threshold", 0.40)),
            consistency_threshold=float(arb_cfg.get("consistency_threshold", 0.40)),
            surprise_decay=float(arb_cfg.get("surprise_decay", 0.85)),
            default_ddqn_weight=float(arb_cfg.get("default_ddqn_weight", 0.70)),
            default_ca_weight=float(arb_cfg.get("default_ca_weight", 0.30)),
        )

        # Diagnostics & Telemetry
        self.decision_history: list[dict[str, Any]] = []
        self.mode_counts: dict[str, int] = {m.value: 0 for m in ArbitrationMode}
        self.total_decisions: int = 0
        self.sum_ddqn_weight: float = 0.0
        self.sum_ca_weight: float = 0.0

    def select_bin(self, observation: SchedulerObservation) -> int:
        """Evaluates CA and LSTM-DDQN branches and arbitrates the next frequency bin."""
        # 1. Evaluate Context-Aware Branch
        ca_scores, ca_details = self.context_aware.compute_action_scores(observation)

        # 2. Evaluate LSTM-DDQN Branch (updating recurrent temporal memory)
        q_values = self.lstm_ddqn.get_q_values(observation)

        # 3. Meta-Arbitration
        chosen_bin, telemetry = self.arbitrator.arbitrate(
            observation=observation,
            q_values=q_values,
            ca_scores=ca_scores,
            ca_details=ca_details,
        )

        # 4. Telemetry Logging
        self.total_decisions += 1
        mode_val = telemetry["arbitration_mode"]
        self.mode_counts[mode_val] = self.mode_counts.get(mode_val, 0) + 1
        self.sum_ddqn_weight += telemetry["DDQN_weight"]
        self.sum_ca_weight += telemetry["CA_weight"]

        self.last_selected_bin = chosen_bin
        self.last_selected = self.bands_hz[chosen_bin]
        self.decision_history.append(telemetry)

        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "rule": "lstm_hybrid_meta_arbitration",
            "mode": mode_val,
            "ddqn_weight": telemetry["DDQN_weight"],
            "ca_weight": telemetry["CA_weight"],
            "ddqn_conf": telemetry["DDQN_confidence"],
            "ca_conf": telemetry["CA_confidence"],
            "surprise": telemetry["novelty_surprise"],
            "consistency": telemetry["transition_consistency"],
            "lstm_ddqn_best_bin": telemetry["DDQN_best_action"],
            "ca_best_bin": telemetry["CA_best_action"],
            "reason": (
                f"Mode={mode_val} (w_LSTM={telemetry['DDQN_weight']:.2f}, w_CA={telemetry['CA_weight']:.2f}) "
                f"-> Picked Bin {chosen_bin} ({self.last_selected/1e6:.1f} MHz)"
            ),
        }
        return chosen_bin

    def select_action(self, observation: SchedulerObservation) -> ScanAction:
        bin_idx = self.select_bin(observation)
        return ScanAction(frequency_bin=bin_idx)

    def select_frequency(self, observation: Any = None) -> float:
        bin_idx = self.select_bin(observation)
        return self.bands_hz[bin_idx]

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction | int,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
        regime_id: int | None = None,
    ) -> None:
        """Updates internal state across all branches and arbitrator feedback."""
        act = action if isinstance(action, ScanAction) else ScanAction(frequency_bin=int(action))

        # 1. Update Context-Aware Empirical Matrix
        self.context_aware.update_policy(observation, act, reward, next_observation, done)

        # 2. Update LSTM-DDQN Replay Buffer & Gradient Step (self-gates on train_mode)
        self.lstm_ddqn.update_policy(observation, act, reward, next_observation, done, regime_id=regime_id)

        # 3. Update Arbitrator Feedback
        self.arbitrator.update_feedback(observation, act.frequency_bin, next_observation)

    def observe(
        self,
        observation: SchedulerObservation | Any,
        action: ScanAction | None = None,
        reward: float = 0.0,
        next_observation: SchedulerObservation | None = None,
        done: bool = False,
    ) -> None:
        """Ensures online tracking and arbitration feedback run even during evaluation."""
        if hasattr(observation, "observation") and hasattr(observation, "action") and hasattr(observation, "reward"):
            trans = observation
            self.update_policy(trans.observation, trans.action, trans.reward, trans.next_observation, trans.done)
            return
        self.update_policy(observation, action, reward, next_observation, done)

    def reset(self) -> None:
        """Resets all branches, clearing LSTM hidden state (zero cross-episode leakage)."""
        super().reset()
        self.context_aware.reset()
        self.lstm_ddqn.reset()
        self.arbitrator.reset()
        self.decision_history.clear()
        self.mode_counts = {m.value: 0 for m in ArbitrationMode}
        self.total_decisions = 0
        self.sum_ddqn_weight = 0.0
        self.sum_ca_weight = 0.0

    def reseed(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        ca_seed = derive_seed(seed, "context_aware") if seed is not None else None
        lstm_seed = derive_seed(seed, "lstm_ddqn") if seed is not None else None
        if hasattr(self.context_aware, "reseed"):
            self.context_aware.reseed(ca_seed)
        elif hasattr(self.context_aware, "rng"):
            self.context_aware.rng = np.random.default_rng(ca_seed)
        if hasattr(self.lstm_ddqn, "reseed"):
            self.lstm_ddqn.reseed(lstm_seed)
        elif hasattr(self.lstm_ddqn, "rng"):
            self.lstm_ddqn.rng = np.random.default_rng(lstm_seed)

    def train(self) -> None:
        super().train()
        self.lstm_ddqn.train()

    def eval(self) -> None:
        super().eval()
        self.lstm_ddqn.eval()

    def get_diagnostics(self) -> dict[str, Any]:
        """Returns aggregate diagnostic summary of arbitration and recurrent state."""
        n_dec = max(self.total_decisions, 1)
        return {
            "total_decisions": self.total_decisions,
            "mode_counts": dict(self.mode_counts),
            "pct_ddqn_exploit": float(self.mode_counts.get(ArbitrationMode.DDQN_EXPLOIT.value, 0)) / float(n_dec) * 100.0,
            "pct_ca_adapt": float(self.mode_counts.get(ArbitrationMode.CA_ADAPT.value, 0)) / float(n_dec) * 100.0,
            "pct_blended": float(self.mode_counts.get(ArbitrationMode.BLENDED.value, 0)) / float(n_dec) * 100.0,
            "pct_explore": float(self.mode_counts.get(ArbitrationMode.EXPLORE_DISCOVERY.value, 0)) / float(n_dec) * 100.0,
            "avg_ddqn_weight": float(self.sum_ddqn_weight) / float(n_dec),
            "avg_ca_weight": float(self.sum_ca_weight) / float(n_dec),
            "lstm_hidden_dim": self.lstm_ddqn.hidden_dim,
            "lstm_sequence_length": self.lstm_ddqn.sequence_length,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "is_pretrained": self.is_pretrained,
        }

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Saves underlying LSTM-DDQN model parameters to checkpoint."""
        return self.lstm_ddqn.save_checkpoint(path, metadata=metadata)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads validated checkpoint into underlying LSTM-DDQN and freezes inference."""
        return self.lstm_ddqn.load_checkpoint(path)

    @property
    def checkpoint_path(self) -> str | None:
        return self.lstm_ddqn.checkpoint_path

    @property
    def checkpoint_sha256(self) -> str | None:
        return self.lstm_ddqn.checkpoint_sha256

    @property
    def is_pretrained(self) -> bool:
        return getattr(self.lstm_ddqn, "is_pretrained", False)
