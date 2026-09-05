from __future__ import annotations

from collections import deque
from typing import Any
import numpy as np

from rf_environment.channel.channel import ChannelModel
from rf_environment.domain.action import ScanAction
from rf_environment.domain.enums import EventType, OutcomeType
from rf_environment.domain.ground_truth import ScanOutcome
from rf_environment.domain.observation import Observation
from rf_environment.domain.state import SchedulerObservation
from rf_environment.domain.transition import StepResult
from rf_environment.emitters.base import BaseEmitter
from rf_environment.environment.ground_truth import GroundTruthStore
from rf_environment.environment.observation_builder import ObservationBuilder
from rf_environment.environment.simulation_clock import SimulationClock
from rf_environment.metrics.metrics_engine import MetricsEngine
from rf_environment.receiver.detector import Detector
from rf_environment.receiver.receiver import Receiver
from rf_environment.receiver.scan_controller import ScanController
from rf_environment.receiver.tuner import bands_overlap
from rf_environment.rewards.r4_reward import R4RewardCalculator
from rf_environment.rewards.reward import RewardCalculator
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.signal.base import SignalSource
from rf_environment.signal.python_source import PythonSignalSource
from rf_environment.util.seeding import derive_seed
from rf_environment.visualization.event_stream import EventStream


class RFEnvironment:
    """RL-Ready RF Environment Simulating EW Radar/Communication Interception.

    Architectural separation:
    1. Simulation Clock & Physics: Emitters, propagation channel, noise.
    2. Receiver & Scan Control: Receiver hardware, tuner, scan controller.
    3. Detector & Canonical Observation Builder: Imperfect detection and SchedulerObservation.
    4. Evaluator: Ground-truth OpportunityTracker and MetricsEngine (completely hidden from scheduler).
    5. Scheduler Interface: Decoupled decision maker consuming observations and emitting actions.
    """

    def __init__(
        self,
        emitters: list[BaseEmitter],
        receiver: Receiver,
        detector: Detector,
        scheduler: BaseScheduler | None = None,
        clock: SimulationClock | None = None,
        channel: ChannelModel | None = None,
        signal_source: SignalSource | None = None,
        events: EventStream | None = None,
        metrics: MetricsEngine | None = None,
        reward_calculator: RewardCalculator | R4RewardCalculator | None = None,
        spectrum: dict[str, float] | None = None,
        waterfall_limit: int = 2000,
        bands_hz: list[float] | None = None,
        history_length: int = 10,
    ) -> None:
        self.emitters = {e.emitter_id: e for e in emitters}
        self.receiver = receiver
        self.detector = detector
        self.scheduler = scheduler
        self.clock = clock or SimulationClock()
        self.channel = channel or ChannelModel()
        self.signal_source = signal_source or PythonSignalSource()
        self.events = events or EventStream()
        self.metrics = metrics or MetricsEngine()
        self.reward_calculator = (
            reward_calculator if reward_calculator is not None else R4RewardCalculator()
        )
        self.ground_truth = GroundTruthStore()
        self.spectrum = spectrum or {"min_frequency_hz": 100e6, "max_frequency_hz": 1e9}
        self.waterfall: deque[dict[str, Any]] = deque(maxlen=waterfall_limit)
        self.paused = False
        self.running = False
        self._initialized = False
        self.train_mode = True

        # Scan bands
        if bands_hz is not None:
            self.bands_hz = [float(b) for b in bands_hz]
        elif self.scheduler is not None and hasattr(self.scheduler, "bands_hz"):
            self.bands_hz = list(self.scheduler.bands_hz)
        else:
            self.bands_hz = [self.receiver.center_frequency_hz]

        # Scan Controller & Canonical Observation Builder
        self.scan_controller = ScanController(
            receiver=self.receiver,
            bands_hz=self.bands_hz,
            min_freq_hz=self.spectrum.get("min_frequency_hz"),
            max_freq_hz=self.spectrum.get("max_frequency_hz"),
        )
        self.observation_builder = ObservationBuilder(
            num_bins=len(self.bands_hz),
            history_length=history_length,
        )

        # Context manager and observed transition tracker for legacy & contextual components
        from rf_environment.environment.context_manager import TemporalContextManager
        from rf_environment.metrics.transition_tracker import ObservedTransitionTracker

        self.temporal_context = TemporalContextManager(bands_hz=self.bands_hz)
        self.transition_tracker = ObservedTransitionTracker(bands_hz=self.bands_hz)

        # Current decision state
        self._current_action: ScanAction | None = None
        self._current_observation: SchedulerObservation = (
            self.observation_builder.build_initial_observation(
                current_frequency_bin=self.scan_controller.current_frequency_bin,
            )
        )

    def train(self) -> None:
        """Sets environment and scheduler to training mode."""
        self.train_mode = True
        if self.scheduler and hasattr(self.scheduler, "train"):
            self.scheduler.train()

    def eval(self) -> None:
        """Sets environment and scheduler to evaluation mode (frozen learning)."""
        self.train_mode = False
        if self.scheduler and hasattr(self.scheduler, "eval"):
            self.scheduler.eval()

    def reseed(self, seed: int) -> None:
        """Deterministically reseeds all stochastic components in the simulation."""
        self.detector.reseed(derive_seed(seed, "detector"))
        if hasattr(self.channel.noise, "reseed"):
            self.channel.noise.reseed(derive_seed(seed, "channel"))
        for eid, emitter in self.emitters.items():
            emitter.reseed(derive_seed(seed, "emitter", eid))
        if self.scheduler and hasattr(self.scheduler, "reseed"):
            self.scheduler.reseed(derive_seed(seed, "scheduler"))

    def reset(self, seed: int | None = None) -> SchedulerObservation:
        """Resets the simulation episode and returns the initial observation.

        Conceptually:
            observation = env.reset(seed=42)
        """
        if seed is not None:
            self.reseed(seed)
        else:
            for emitter in self.emitters.values():
                emitter.reset()

        self.clock.reset()
        self.ground_truth.reset()
        self.metrics.reset()
        self.temporal_context.reset()
        self.transition_tracker.reset()
        self.waterfall.clear()
        self.running = True
        self.paused = False

        if self.scheduler:
            self.scheduler.reset()

        self.scan_controller.reset(initial_bin=0)
        self.observation_builder.reset()

        self._current_observation = self.observation_builder.build_initial_observation(
            current_frequency_bin=0,
        )
        self._current_action = None
        self._initialized = False
        return self._current_observation

    def _ensure_initial_tune(self) -> None:
        if self._initialized:
            return
        if self.scheduler:
            action = self.scheduler.select_action(self._current_observation)
            if not isinstance(action, ScanAction):
                if isinstance(action, (int, np.integer)):
                    action = ScanAction(frequency_bin=int(action))
                else:
                    bin_idx = min(range(len(self.bands_hz)), key=lambda i: abs(self.bands_hz[i] - float(action)))
                    action = ScanAction(frequency_bin=bin_idx)
            self._current_action = action
            tuned_freq = self.scan_controller.execute_action(action)
            self.events.publish(
                0,
                EventType.SCHEDULER_DECISION,
                frequency_hz=tuned_freq,
                scheduler=self.scheduler.name,
            )
        self._initialized = True

    def step(
        self,
        action: ScanAction | float | int | None = None,
    ) -> StepResult:
        """Advances simulation by one time step with the commanded scan action.

        Usage:
            # RL episode loop:
            next_obs, reward, done, info = env.step(action)

            # Legacy parameterless step (uses internal scheduler):
            result = env.step()
            gt = result["ground_truth"]
        """
        self._ensure_initial_tune()

        # 1. Resolve action
        if action is None:
            if self._current_action is not None:
                resolved_action = self._current_action
                self._current_action = None
            elif self.scheduler is not None:
                raw_action = self.scheduler.select_action(self._current_observation)
                if isinstance(raw_action, ScanAction):
                    resolved_action = raw_action
                elif isinstance(raw_action, (int, np.integer)):
                    resolved_action = ScanAction(frequency_bin=int(raw_action))
                else:
                    bin_idx = min(range(len(self.bands_hz)), key=lambda i: abs(self.bands_hz[i] - float(raw_action)))
                    resolved_action = ScanAction(frequency_bin=bin_idx)
            else:
                raise ValueError("step() called without action and no internal scheduler configured.")
        elif isinstance(action, (int, np.integer)):
            resolved_action = ScanAction(frequency_bin=int(action))
        elif isinstance(action, (float, np.floating)):
            bin_idx = min(range(len(self.bands_hz)), key=lambda i: abs(self.bands_hz[i] - float(action)))
            resolved_action = ScanAction(frequency_bin=bin_idx)
        elif isinstance(action, ScanAction):
            resolved_action = action
        else:
            raise TypeError(f"Invalid action type: {type(action).__name__}. Must be ScanAction, int, float, or None.")

        # 2. Execute action via ScanController
        tuned_freq = self.scan_controller.execute_action(resolved_action)

        # 3. Advance simulation time
        t = self.clock.advance()
        self.events.publish(t, EventType.SIMULATION_STEP, time_step=t)

        # 4. Evolve emitters and ground truth
        states = [emitter.step(t) for emitter in self.emitters.values()]
        gt = self.ground_truth.record(t, states)
        for state in states:
            self.events.publish(t, EventType.EMITTER_STATE_CHANGED, **state.to_dict())

        # 5. Channel propagation and noise
        ideal = [
            self.signal_source.generate(state)
            for state in states
            if state.transmitting
        ]
        channel_out = self.channel.apply(ideal, states)

        # 6. Receiver measurement and detection
        rx_state = self.receiver.get_state()
        self.events.publish(
            t,
            EventType.SCAN_STARTED,
            frequency_hz=rx_state.center_frequency_hz,
            bandwidth_hz=rx_state.instantaneous_bandwidth_hz,
        )
        measurement = self.receiver.observe(t, channel_out)
        detection = self.detector.detect(measurement)

        rx_cf = measurement.center_frequency_hz
        rx_bw = measurement.bandwidth_hz
        in_band_tx = [
            s.emitter_id
            for s in states
            if s.transmitting and bands_overlap(rx_cf, rx_bw, s.frequency_hz, s.bandwidth_hz)
        ]
        intercepted = in_band_tx if detection.detected and not detection.false_alarm else []

        active_tx_states = [s for s in states if s.transmitting]

        # 7. Diagnostic outcome classification
        if detection.detected and in_band_tx:
            outcome = OutcomeType.HIT.value
            diagnostic_reason = f"Target Intercepted: Receiver scanned {rx_cf/1e6:.1f} MHz, covering active emitter(s) {in_band_tx}."
        elif detection.detected and not in_band_tx:
            outcome = OutcomeType.FALSE_ALARM.value
            diagnostic_reason = f"False Alarm: Detector triggered on noise at {rx_cf/1e6:.1f} MHz in an unoccupied band."
        elif (not detection.detected) and in_band_tx:
            outcome = OutcomeType.MISS.value
            diagnostic_reason = f"Detector Miss: Emitter(s) {in_band_tx} were in band {rx_cf/1e6:.1f} MHz, but detector failed to trigger."
        else:
            outcome = OutcomeType.CORRECT_REJECTION.value
            if active_tx_states:
                first_active = active_tx_states[0]
                diagnostic_reason = f"Receiver Off-Frequency: Receiver scanned {rx_cf/1e6:.1f} MHz while {first_active.emitter_id} transmitted at {first_active.frequency_hz/1e6:.1f} MHz."
            else:
                diagnostic_reason = f"No Transmissions: Receiver scanned {rx_cf/1e6:.1f} MHz and no emitter was active in the spectrum."

        associated = in_band_tx[0] if intercepted else None
        if detection.detected and associated:
            detection.emitter_id = associated
            detection.associated = True

        # 8. Evaluation & Ground-Truth Opportunity Tracking (Strict Evaluator Isolation)
        self.metrics.opportunity_tracker.step(
            timestamp=t,
            emitter_states=states,
            rx_center_freq_hz=rx_cf,
            rx_bandwidth_hz=rx_bw,
            detector_detected=detection.detected,
            is_false_alarm=detection.false_alarm,
        )

        # 9. Observed Transition Learning (based strictly on receiver observation)
        pred_record = self.transition_tracker.step_observation(
            timestamp=t,
            scanned_freq_hz=rx_cf,
            detected=detection.detected,
        )
        if pred_record is not None:
            self.metrics.record_prediction(pred_record.correct)

        # 10. Build Canonical SchedulerObservation (strictly NO ground truth)
        next_observation = self.observation_builder.step(
            timestamp=float(t),
            scanned_bin=resolved_action.frequency_bin,
            detected=detection.detected,
            signal_strength=measurement.signal_power_dbm,
        )

        # 11. Compute production V3 R4 reward or fallback legacy reward
        if isinstance(self.reward_calculator, R4RewardCalculator):
            reward = self.reward_calculator.compute(
                observation=next_observation,
                action=resolved_action,
            )
        else:
            reward = self.reward_calculator.compute(outcome)

        scan_outcome = ScanOutcome(
            timestamp=t,
            outcome=outcome,
            detected=detection.detected,
            in_band_transmitting_ids=in_band_tx,
            intercepted_ids=intercepted,
            associated_emitter_id=associated,
            reward=reward,
        )

        # 11. Legacy observation for backward compatibility
        legacy_obs = Observation(
            timestamp=t,
            receiver_frequency_hz=rx_cf,
            receiver_bandwidth_hz=rx_bw,
            detected=detection.detected,
            snr_db=detection.snr_db,
            associated_emitter_id=None,  # Scheduler never sees ground-truth emitter identity
            reward=reward,
            measurement=measurement,
        )
        self.temporal_context.update(t, rx_cf, detection.detected, reward)
        self.temporal_context.enrich_observation(legacy_obs)

        # 12. Notify internal scheduler if attached
        done = self.clock.finished()
        if self.scheduler:
            self.scheduler.observe(
                self._current_observation,
                resolved_action,
                reward,
                next_observation,
                done,
            )

        self._current_observation = next_observation

        snapshot = self.metrics.record(
            timestamp=t,
            outcome=outcome,
            reward=reward,
            transmitting_ids=gt.transmitting_ids,
            intercepted_ids=intercepted,
            detected=detection.detected,
            associated_emitter_id=associated,
        )

        self.events.publish(
            t,
            EventType.SCAN_RESULT,
            receiver_frequency_hz=rx_cf,
            detected=detection.detected,
            snr_db=detection.snr_db,
            emitter_id=associated,
            outcome=outcome,
        )
        if outcome == OutcomeType.HIT.value:
            self.events.publish(t, EventType.HIT, emitter_id=associated, frequency_hz=rx_cf)
        elif outcome == OutcomeType.MISS.value:
            self.events.publish(t, EventType.MISS, in_band=in_band_tx, frequency_hz=rx_cf)
        elif outcome == OutcomeType.FALSE_ALARM.value:
            self.events.publish(t, EventType.FALSE_ALARM, frequency_hz=rx_cf)

        self.events.publish(t, EventType.METRIC_UPDATE, **snapshot.to_dict())

        self.waterfall.append(
            {
                "timestamp": t,
                "ground_truth": [
                    {
                        "emitter_id": s.emitter_id,
                        "frequency_hz": s.frequency_hz,
                        "power_dbm": s.power_dbm,
                        "transmitting": s.transmitting,
                    }
                    for s in states
                ],
                "receiver_frequency_hz": rx_cf,
                "receiver_bandwidth_hz": rx_bw,
                "detected": detection.detected,
                "snr_db": detection.snr_db,
                "outcome": outcome,
                "diagnostic_reason": diagnostic_reason,
            }
        )

        if done:
            self.running = False
            self.metrics.opportunity_tracker.finalize(t)
            self.events.publish(t, EventType.SIMULATION_COMPLETED, **snapshot.to_dict())

        scheduler_dict = (
            self.scheduler.get_state().to_dict()
            if self.scheduler
            else {}
        )

        info: dict[str, Any] = {
            "timestamp": t,
            "ground_truth": gt.to_dict(),
            "observation": legacy_obs.to_dict(),
            "scheduler_observation": {
                "timestamp": next_observation.timestamp,
                "current_frequency_bin": next_observation.current_frequency_bin,
                "last_detection": next_observation.last_detection,
                "last_detection_bin": next_observation.last_detection_bin,
                "last_detection_strength": next_observation.last_detection_strength,
                "recent_detection_history": list(next_observation.recent_detection_history),
                "recent_frequency_history": list(next_observation.recent_frequency_history),
                "scan_count_by_bin": list(next_observation.scan_count_by_bin),
                "time_since_scan_by_bin": list(next_observation.time_since_scan_by_bin),
                "time_since_last_detection": next_observation.time_since_last_detection,
            },
            "detection": detection.to_dict(),
            "outcome": scan_outcome.to_dict(),
            "metrics": snapshot.to_dict(),
            "receiver": self.receiver.get_state().to_dict(),
            "scheduler": scheduler_dict,
            "diagnostic_reason": diagnostic_reason,
            "finished": done,
            "tuned_frequency_hz": tuned_freq,
        }

        return StepResult(
            observation=next_observation,
            reward=reward,
            done=done,
            info=info,
        )

    def run(self, steps: int | None = None) -> list[StepResult]:
        """Runs the simulation for a number of steps or until completion."""
        self.running = True
        self.paused = False
        results = []
        remaining = steps
        while not self.clock.finished():
            if remaining is not None and remaining <= 0:
                break
            results.append(self.step())
            if remaining is not None:
                remaining -= 1
        return results

    def public_state(self) -> dict[str, Any]:
        latest_gt = self.ground_truth.latest.to_dict() if self.ground_truth.latest else None
        sched_state = self.scheduler.get_state().to_dict() if self.scheduler else {}
        return {
            "time_step": self.clock.time_step,
            "running": self.running,
            "paused": self.paused,
            "emitters": [e.get_state().to_dict() for e in self.emitters.values() if e._state],
            "receiver": self.receiver.get_state().to_dict(),
            "scheduler": sched_state,
            "metrics": self.metrics.snapshot(max(self.clock.time_step, 0)).to_dict(),
            "ground_truth": latest_gt,
            "spectrum": self.spectrum,
            "transition_stats": self.transition_tracker.get_stats().to_dict(),
        }

    def get_opportunities_summary(self) -> list[dict[str, Any]]:
        return [o.to_summary_dict() for o in self.metrics.opportunity_tracker.get_all_opportunities()]
