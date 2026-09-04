from __future__ import annotations

from collections import deque
from typing import Any

from rf_environment.channel.channel import ChannelModel
from rf_environment.channel.noise import AdditiveNoise
from rf_environment.channel.propagation import FreeSpacePathLoss
from rf_environment.domain.enums import EventType, OutcomeType
from rf_environment.domain.ground_truth import ScanOutcome
from rf_environment.domain.observation import Observation
from rf_environment.emitters.base import BaseEmitter
from rf_environment.environment.ground_truth import GroundTruthStore
from rf_environment.environment.simulation_clock import SimulationClock
from rf_environment.metrics.metrics_engine import MetricsEngine
from rf_environment.receiver.detector import Detector
from rf_environment.receiver.receiver import Receiver
from rf_environment.receiver.tuner import bands_overlap
from rf_environment.rewards.reward import RewardCalculator
from rf_environment.scheduler.base import ScanScheduler
from rf_environment.signal.base import SignalSource
from rf_environment.signal.python_source import PythonSignalSource
from rf_environment.visualization.event_stream import EventStream


class RFEnvironment:
    def __init__(
        self,
        emitters: list[BaseEmitter],
        receiver: Receiver,
        detector: Detector,
        scheduler: ScanScheduler,
        clock: SimulationClock,
        channel: ChannelModel,
        signal_source: SignalSource | None = None,
        events: EventStream | None = None,
        metrics: MetricsEngine | None = None,
        reward_calculator: RewardCalculator | None = None,
        spectrum: dict[str, float] | None = None,
        waterfall_limit: int = 2000,
    ) -> None:
        self.emitters = {e.emitter_id: e for e in emitters}
        self.receiver = receiver
        self.detector = detector
        self.scheduler = scheduler
        self.clock = clock
        self.channel = channel
        self.signal_source = signal_source or PythonSignalSource()
        self.events = events or EventStream()
        self.metrics = metrics or MetricsEngine()
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.ground_truth = GroundTruthStore()
        self.spectrum = spectrum or {"min_frequency_hz": 100e6, "max_frequency_hz": 1e9}
        self.waterfall: deque[dict[str, Any]] = deque(maxlen=waterfall_limit)
        self.paused = False
        self.running = False
        self._initialized = False

        # V2: Temporal context and observed transition learning
        from rf_environment.environment.context_manager import TemporalContextManager
        from rf_environment.metrics.transition_tracker import ObservedTransitionTracker

        self.temporal_context = TemporalContextManager(bands_hz=self.scheduler.bands_hz)
        self.transition_tracker = ObservedTransitionTracker(bands_hz=self.scheduler.bands_hz)

    def _ensure_initial_tune(self) -> None:
        if self._initialized:
            return
        freq = self.scheduler.select_action(None)
        self.receiver.tune(freq)
        self.events.publish(
            0,
            EventType.SCHEDULER_DECISION,
            frequency_hz=freq,
            scheduler=self.scheduler.name,
        )
        self._initialized = True

    def step(self) -> dict[str, Any]:
        self._ensure_initial_tune()
        t = self.clock.advance()
        self.events.publish(t, EventType.SIMULATION_STEP, time_step=t)

        states = [emitter.step(t) for emitter in self.emitters.values()]
        gt = self.ground_truth.record(t, states)
        for state in states:
            self.events.publish(t, EventType.EMITTER_STATE_CHANGED, **state.to_dict())

        ideal = [
            self.signal_source.generate(state)
            for state in states
            if state.transmitting
        ]
        channel_out = self.channel.apply(ideal, states)

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

        reward = self.reward_calculator.compute(outcome)
        associated = in_band_tx[0] if intercepted else None
        if detection.detected and associated:
            detection.emitter_id = associated
            detection.associated = True

        # V2 Opportunity Tracking (strict ground truth separation: only used by evaluator)
        self.metrics.opportunity_tracker.step(
            timestamp=t,
            emitter_states=states,
            rx_center_freq_hz=rx_cf,
            rx_bandwidth_hz=rx_bw,
            detector_detected=detection.detected,
            is_false_alarm=detection.false_alarm,
        )

        # V2 Observed Transition Learning (based strictly on receiver observation)
        pred_record = self.transition_tracker.step_observation(
            timestamp=t,
            scanned_freq_hz=rx_cf,
            detected=detection.detected,
        )
        if pred_record is not None:
            self.metrics.record_prediction(pred_record.correct)

        scan_outcome = ScanOutcome(
            timestamp=t,
            outcome=outcome,
            detected=detection.detected,
            in_band_transmitting_ids=in_band_tx,
            intercepted_ids=intercepted,
            associated_emitter_id=associated,
            reward=reward,
        )

        # Build clean observation for scheduler (NO ground truth leak)
        observation = Observation(
            timestamp=t,
            receiver_frequency_hz=rx_cf,
            receiver_bandwidth_hz=rx_bw,
            detected=detection.detected,
            snr_db=detection.snr_db,
            associated_emitter_id=None,  # Scheduler never sees ground-truth emitter identity
            reward=reward,
            measurement=measurement,
        )

        # Update and enrich with temporal context
        self.temporal_context.update(t, rx_cf, detection.detected, reward)
        self.temporal_context.enrich_observation(observation)

        self.scheduler.update(observation, reward, action=rx_cf)
        next_freq = self.scheduler.select_action(observation)
        self.receiver.tune(next_freq)

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

        self.events.publish(
            t,
            EventType.SCHEDULER_DECISION,
            frequency_hz=next_freq,
            scheduler=self.scheduler.name,
        )
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

        finished = self.clock.finished()
        if finished:
            self.running = False
            self.metrics.opportunity_tracker.finalize(t)
            self.events.publish(t, EventType.SIMULATION_COMPLETED, **snapshot.to_dict())

        return {
            "timestamp": t,
            "ground_truth": gt.to_dict(),
            "observation": observation.to_dict(),
            "detection": detection.to_dict(),
            "outcome": scan_outcome.to_dict(),
            "metrics": snapshot.to_dict(),
            "receiver": self.receiver.get_state().to_dict(),
            "scheduler": self.scheduler.get_state().to_dict(),
            "diagnostic_reason": diagnostic_reason,
            "finished": finished,
        }

    def run(self, steps: int | None = None) -> list[dict[str, Any]]:
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
        return {
            "time_step": self.clock.time_step,
            "running": self.running,
            "paused": self.paused,
            "emitters": [e.get_state().to_dict() for e in self.emitters.values() if e._state],
            "receiver": self.receiver.get_state().to_dict(),
            "scheduler": self.scheduler.get_state().to_dict(),
            "metrics": self.metrics.snapshot(max(self.clock.time_step, 0)).to_dict(),
            "ground_truth": latest_gt,
            "spectrum": self.spectrum,
            "transition_stats": self.transition_tracker.get_stats().to_dict(),
        }

    def get_opportunities_summary(self) -> list[dict[str, Any]]:
        return [o.to_summary_dict() for o in self.metrics.opportunity_tracker.get_all_opportunities()]
