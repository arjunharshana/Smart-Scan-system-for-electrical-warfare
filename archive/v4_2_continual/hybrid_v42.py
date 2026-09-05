from __future__ import annotations

import time
from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.hybrid.arbitrator import ArbitrationMode, HybridMetaArbitrator
from rf_environment.scheduler.hybrid.continual.checkpoint_manager import ModelCheckpointManager
from rf_environment.scheduler.hybrid.continual.config import V42ContinualConfig
from rf_environment.scheduler.hybrid.continual.online_replay import OnlineReplayBuffer
from rf_environment.scheduler.hybrid.continual.online_trainer import OnlineTrainer
from rf_environment.scheduler.hybrid.continual.regime_detector import RegimeChangeDetector, RegimeState
from rf_environment.scheduler.hybrid.continual.reward import ObservableOnlineReward
from rf_environment.scheduler.hybrid.continual.validation_gate import ValidationGate
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.util.seeding import derive_seed


class V42ContinualAdaptiveHybrid(BaseScheduler):
    """V4.2 Continual Adaptive Hybrid Scheduler with Controlled Online Learning.

    Extends the V4.1 Dual-Branch Hybrid with an autonomous continual learning engine:
    1. Observable-Only Online Reward: Computes learning signals strictly from detector feedback.
    2. Regime Change Detector: 5-state machine (STABLE, SUSPECTED_SHIFT, ADAPTING, VALIDATING, ROLLBACK).
    3. Mixed Online Replay: Dual-reservoir (70% recent / 30% historical) to prevent catastrophic forgetting.
    4. Controlled Online Trainer: Bounded BPTT updates on candidate network copy.
    5. Validation Gate: Approves candidate only if new regime adapts without degrading historical retention.
    6. Checkpoint Manager: Immediate bit-for-bit rollback on validation rejection.

    Invariants:
    1. Strictly zero ground-truth access (enforced by construction and verified by test suite).
    2. Production weights remain untouched during training until validated.
    3. Common action space {0..N-1} emitted as ScanAction(frequency_bin).
    4. Deterministic and reproducible across all seeds.
    """

    name = "hybrid_v42"
    category = "hybrid"

    def __init__(
        self,
        bands_hz: list[float],
        context_aware: ContextAwareScheduler | None = None,
        lstm_ddqn: LSTMDDQNScheduler | None = None,
        arbitrator: HybridMetaArbitrator | None = None,
        config: V42ContinualConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(bands_hz)
        self.num_bins = len(self.bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.config = config or V42ContinualConfig()

        # 1. Dual-Branch Schedulers
        ca_seed = derive_seed(seed, "ca") if seed is not None else None
        self.context_aware = context_aware or ContextAwareScheduler(
            bands_hz=self.bands_hz,
            seed=ca_seed,
        )

        lstm_seed = derive_seed(seed, "lstm_ddqn") if seed is not None else None
        self.lstm_ddqn = lstm_ddqn or LSTMDDQNScheduler(
            bands_hz=self.bands_hz,
            seed=lstm_seed,
        )

        # 2. Transparent Meta-Arbitrator
        self.arbitrator = arbitrator or HybridMetaArbitrator(
            num_bins=self.num_bins,
            surprise_threshold=self.config.surprise_threshold,
        )

        # 3. Continual Learning Engine
        replay_seed = derive_seed(seed, "replay") if seed is not None else None
        self.online_replay = OnlineReplayBuffer(
            config=self.config,
            seed=replay_seed,
        )
        self.reward_calculator = ObservableOnlineReward(num_bins=self.num_bins)
        self.regime_detector = RegimeChangeDetector(config=self.config)
        self.validation_gate = ValidationGate(config=self.config)
        self.checkpoint_manager = ModelCheckpointManager(
            production_net=self.lstm_ddqn.online_net,
            target_net=self.lstm_ddqn.target_net,
        )
        self.online_trainer = OnlineTrainer(config=self.config)

        # Step Decision Caching
        self.last_selected_bin: int = 0
        self.last_predicted_bin: int = 0
        self.last_arbitration_telemetry: dict[str, Any] = {}
        self.last_validation_telemetry: dict[str, Any] = {}

        # Telemetry & Diagnostics
        self.total_decisions: int = 0
        self.mode_counts: dict[str, int] = {m.value: 0 for m in ArbitrationMode}
        self.sum_ddqn_weight: float = 0.0
        self.sum_ca_weight: float = 0.0
        self.regime_state_counts: dict[str, int] = {s.value: 0 for s in RegimeState}

    def select_bin(self, observation: SchedulerObservation) -> int:
        """Computes dual-branch recommendations, applies regime modulation, and selects frequency bin."""
        # 1. Context-Aware Recommendation
        ca_scores, ca_details = self.context_aware.compute_action_scores(observation)
        best_ca_bin = int(np.argmax(ca_scores))

        # 2. LSTM-DDQN Recommendation
        q_values = self.lstm_ddqn.get_q_values(observation)
        best_ddqn_bin = int(np.argmax(q_values))
        self.last_predicted_bin = best_ddqn_bin

        # 3. Base Meta-Arbitration
        chosen_bin, telem = self.arbitrator.arbitrate(observation, q_values, ca_scores, ca_details)

        # 4. Regime-State Arbitration Modulation
        current_state = self.regime_detector.state
        w_ddqn = float(telem.get("DDQN_weight", 0.5))
        w_ca = float(telem.get("CA_weight", 0.5))
        final_mode = telem.get("arbitration_mode", ArbitrationMode.BLENDED.value)

        if current_state in {RegimeState.SUSPECTED_SHIFT, RegimeState.ADAPTING}:
            # Elevated CA responsibility during detected disruption
            w_ca = max(w_ca, self.config.ca_weight_floor_suspected)
            w_ddqn = 1.0 - w_ca
            chosen_bin = best_ca_bin
            final_mode = ArbitrationMode.CA_ADAPT.value

        elif current_state == RegimeState.ROLLBACK:
            # Conservative arbitration after rollback
            w_ca = max(w_ca, self.config.ca_weight_floor_rollback)
            w_ddqn = 1.0 - w_ca
            chosen_bin = best_ca_bin
            final_mode = ArbitrationMode.CA_ADAPT.value

        self.last_selected_bin = chosen_bin
        self.last_selected = self.bands_hz[chosen_bin]
        self.last_arbitration_telemetry = telem

        # Update telemetry
        self.total_decisions += 1
        self.mode_counts[final_mode] = self.mode_counts.get(final_mode, 0) + 1
        self.regime_state_counts[current_state.value] = self.regime_state_counts.get(current_state.value, 0) + 1
        self.sum_ddqn_weight += w_ddqn
        self.sum_ca_weight += w_ca

        return chosen_bin

    def select_action(self, observation: SchedulerObservation) -> ScanAction:
        bin_idx = self.select_bin(observation)
        return ScanAction(frequency_bin=bin_idx)

    def observe(
        self,
        observation: SchedulerObservation | Any,
        action: ScanAction | int | None = None,
        reward: float = 0.0,
        next_observation: SchedulerObservation | None = None,
        done: bool = False,
    ) -> None:
        """Ingests detector feedback, computes observable reward, updates detector, and adapts."""
        # Normalize arguments
        if hasattr(observation, "observation") and hasattr(observation, "action"):
            obs = observation.observation
            act = observation.action
            next_obs = observation.next_observation
            d = bool(observation.done)
        else:
            obs = observation
            act = action
            next_obs = next_observation
            d = bool(done)

        act_obj = act if isinstance(act, ScanAction) else ScanAction(frequency_bin=int(act or 0))
        act_bin = act_obj.frequency_bin

        # 1. Update Context-Aware empirical transition model
        self.context_aware.update_policy(obs, act_obj, reward, next_obs, d)

        # 2. Compute Observable-Only Online Reward
        online_reward = self.reward_calculator.compute_reward(next_obs, act_bin)

        # 3. Push to Online Replay Buffer
        state_vec = self.lstm_ddqn.encoder.encode(obs)
        next_state_vec = self.lstm_ddqn.encoder.encode(next_obs)
        self.online_replay.push(state_vec, act_bin, online_reward, next_state_vec, d)

        # 4. Update Arbitrator Feedback
        self.arbitrator.update_feedback(obs, act_bin, next_obs)

        # 5. Advance Regime Change Detector
        ddqn_conf = float(self.last_arbitration_telemetry.get("DDQN_confidence", 0.5))
        ca_conf = float(self.last_arbitration_telemetry.get("CA_confidence", 0.5))
        surprise = float(self.arbitrator.surprise)
        trans_cons = float(self.last_arbitration_telemetry.get("transition_consistency", 0.5))

        self.regime_detector.update(
            observation=next_obs,
            ddqn_confidence=ddqn_conf,
            ca_confidence=ca_conf,
            surprise=surprise,
            online_reward=online_reward,
            transition_consistency=trans_cons,
            predicted_bin=self.last_predicted_bin,
            executed_bin=act_bin,
        )

        # 6. Controlled Online Adaptation Check
        if self.regime_detector.can_adapt(self.online_replay.has_min_sequences()):
            self._execute_adaptation_cycle()

    def _execute_adaptation_cycle(self) -> None:
        """Executes a controlled candidate training and validation cycle."""
        # 1. Transition to ADAPTING
        self.regime_detector.trigger_adapting()

        # 2. Train Candidate Network on Mixed Replay
        candidate_net, train_stats = self.online_trainer.train_candidate(
            production_net=self.lstm_ddqn.online_net,
            target_net=self.lstm_ddqn.target_net,
            replay_buffer=self.online_replay,
        )
        self.checkpoint_manager.record_attempt(
            train_stats["gradient_steps"],
            train_stats["duration_ms"],
        )

        # 3. Transition to VALIDATING
        self.regime_detector.trigger_validating()

        # Sample validation batches
        batch_size = self.config.validation_batch_size
        recent_val = self.online_replay.sample_recent_validation(batch_size)
        hist_val = self.online_replay.sample_historical_validation(batch_size)

        accepted, val_telem = self.validation_gate.evaluate(
            candidate_net=candidate_net,
            production_net=self.lstm_ddqn.online_net,
            target_net=self.lstm_ddqn.target_net,
            recent_val_batch=recent_val,
            historical_val_batch=hist_val,
            gamma=self.config.gamma,
        )
        self.last_validation_telemetry = val_telem

        # 4. Deploy or Rollback
        if accepted:
            self.checkpoint_manager.deploy_candidate(candidate_net)
            self.regime_detector.on_validation_accepted()
        else:
            self.checkpoint_manager.rollback()
            self.checkpoint_manager.record_rejection()
            self.regime_detector.on_validation_rejected()

    def populate_historical_experience(self, source_buffer: Any) -> None:
        """Loads baseline transitions into the protected historical reservoir."""
        if hasattr(self.online_replay, "populate_historical_from_buffer"):
            self.online_replay.populate_historical_from_buffer(source_buffer)

    def reset(self) -> None:
        """Resets all branches and trackers, clearing LSTM hidden states and recent buffer."""
        super().reset()
        self.context_aware.reset()
        self.lstm_ddqn.reset()
        self.arbitrator.reset()
        self.reward_calculator.reset()
        self.regime_detector.reset()
        self.online_replay.reset()

        self.last_selected_bin = 0
        self.last_predicted_bin = 0
        self.last_arbitration_telemetry.clear()
        self.last_validation_telemetry.clear()

        self.total_decisions = 0
        self.mode_counts = {m.value: 0 for m in ArbitrationMode}
        self.regime_state_counts = {s.value: 0 for s in RegimeState}
        self.sum_ddqn_weight = 0.0
        self.sum_ca_weight = 0.0

    def get_diagnostics(self) -> dict[str, Any]:
        """Returns comprehensive diagnostic telemetry for continual learning and arbitration."""
        n = max(self.total_decisions, 1)
        checkpoint_telem = self.checkpoint_manager.get_telemetry()
        regime_telem = self.regime_detector.get_telemetry()

        return {
            "total_decisions": self.total_decisions,
            "mode_counts": dict(self.mode_counts),
            "regime_state_counts": dict(self.regime_state_counts),
            "pct_ddqn_exploit": float(self.mode_counts.get(ArbitrationMode.DDQN_EXPLOIT.value, 0)) / float(n) * 100.0,
            "pct_ca_adapt": float(self.mode_counts.get(ArbitrationMode.CA_ADAPT.value, 0)) / float(n) * 100.0,
            "pct_blended": float(self.mode_counts.get(ArbitrationMode.BLENDED.value, 0)) / float(n) * 100.0,
            "pct_explore": float(self.mode_counts.get(ArbitrationMode.EXPLORE_DISCOVERY.value, 0)) / float(n) * 100.0,
            "avg_ddqn_weight": float(self.sum_ddqn_weight) / float(n),
            "avg_ca_weight": float(self.sum_ca_weight) / float(n),
            "regime_state": self.regime_detector.state.value,
            "regime_shifts_detected": regime_telem.get("total_detected_shifts", 0),
            "adaptation_attempts": checkpoint_telem.get("adaptation_attempts", 0),
            "adaptations_accepted": checkpoint_telem.get("adaptations_accepted", 0),
            "adaptations_rejected": checkpoint_telem.get("adaptations_rejected", 0),
            "rollbacks_performed": checkpoint_telem.get("rollbacks_performed", 0),
            "total_gradient_steps": checkpoint_telem.get("total_gradient_steps", 0),
            "time_spent_adapting_ms": checkpoint_telem.get("time_spent_adapting_ms", 0.0),
            "last_validation": dict(self.last_validation_telemetry),
        }
