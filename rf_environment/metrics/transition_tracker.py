from __future__ import annotations

from typing import Any
import pandas as pd

from rf_environment.domain.transition import ObservedTransitionStats, PredictionRecord


class ObservedTransitionTracker:
    """Learns empirical frequency transitions solely from receiver observations.
    
    Zero ground truth is used: transitions are recorded exclusively between
    successive receiver detections.
    """

    def __init__(self, bands_hz: list[float], smoothing: float = 0.05) -> None:
        self.bands_hz = sorted([float(b) for b in bands_hz])
        self.smoothing = smoothing

        # Counts: source_band -> target_band -> count
        self.counts: dict[float, dict[float, int]] = {
            src: {dst: 0 for dst in self.bands_hz} for src in self.bands_hz
        }
        self.last_detected_freq: float | None = None
        self.last_detected_time: int | None = None
        self.pending_prediction: tuple[float, float] | None = None  # (predicted_freq, confidence)

        self.predictions: list[PredictionRecord] = []
        self.total_transitions = 0
        self.total_predictions = 0
        self.correct_predictions = 0

    def step_observation(
        self,
        timestamp: int,
        scanned_freq_hz: float,
        detected: bool,
    ) -> PredictionRecord | None:
        """Processes a receiver scan result and records transitions and prediction outcomes."""
        record: PredictionRecord | None = None

        # 1. Evaluate pending prediction if the receiver probed the predicted frequency
        if self.pending_prediction is not None and self.last_detected_freq is not None:
            pred_freq, conf = self.pending_prediction
            if abs(pred_freq - scanned_freq_hz) < 1e5:
                # Receiver tested the prediction!
                is_correct = bool(detected)
                self.total_predictions += 1
                if is_correct:
                    self.correct_predictions += 1

                record = PredictionRecord(
                    step=timestamp,
                    previous_frequency_hz=self.last_detected_freq,
                    predicted_frequency_hz=pred_freq,
                    actual_observed_frequency_hz=scanned_freq_hz if detected else None,
                    correct=is_correct,
                    confidence=conf,
                )
                self.predictions.append(record)
                self.pending_prediction = None

        # 2. If a signal was detected, record empirical transition and issue new prediction
        if detected:
            if self.last_detected_freq is not None:
                src = self._closest_band(self.last_detected_freq)
                dst = self._closest_band(scanned_freq_hz)
                self.counts[src][dst] += 1
                self.total_transitions += 1

            self.last_detected_freq = scanned_freq_hz
            self.last_detected_time = timestamp

            # Issue next prediction from this observed detection
            self.pending_prediction = self.predict_next(scanned_freq_hz)

        return record

    def _closest_band(self, freq_hz: float) -> float:
        return min(self.bands_hz, key=lambda b: abs(b - freq_hz))

    def predict_next(self, current_freq_hz: float) -> tuple[float, float]:
        """Predicts the next active band from current observed frequency.
        
        Returns (predicted_frequency_hz, confidence).
        """
        src = self._closest_band(current_freq_hz)
        row = self.counts[src]
        total_pulls = sum(row.values())

        if total_pulls == 0:
            # Uniform prior
            return current_freq_hz, 1.0 / len(self.bands_hz)

        best_dst = max(self.bands_hz, key=lambda b: row[b])
        prob = (row[best_dst] + self.smoothing) / (total_pulls + self.smoothing * len(self.bands_hz))
        return best_dst, float(prob)

    def get_transition_matrix(self) -> pd.DataFrame:
        """Returns the empirical transition probability matrix as a DataFrame."""
        labels = [f"{b / 1e6:.0f}" for b in self.bands_hz]
        matrix = []
        for src in self.bands_hz:
            row_counts = self.counts[src]
            row_total = sum(row_counts.values())
            if row_total == 0:
                # Uniform distribution when unobserved
                matrix.append([1.0 / len(self.bands_hz)] * len(self.bands_hz))
            else:
                matrix.append([
                    (row_counts[dst] + self.smoothing) / (row_total + self.smoothing * len(self.bands_hz))
                    for dst in self.bands_hz
                ])
        return pd.DataFrame(matrix, index=labels, columns=labels)

    def get_stats(self) -> ObservedTransitionStats:
        matrix_df = self.get_transition_matrix()
        counts_dict = {
            f"{src/1e6:.0f}": {f"{dst/1e6:.0f}": c for dst, c in self.counts[src].items()}
            for src in self.bands_hz
        }
        probs_dict = {idx: row.to_dict() for idx, row in matrix_df.iterrows()}

        acc = (
            self.correct_predictions / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        return ObservedTransitionStats(
            frequencies_hz=self.bands_hz,
            counts=counts_dict,
            probabilities=probs_dict,
            total_transitions=self.total_transitions,
            total_predictions=self.total_predictions,
            correct_predictions=self.correct_predictions,
            prediction_accuracy=acc,
        )

    def reset(self) -> None:
        self.counts = {src: {dst: 0 for dst in self.bands_hz} for src in self.bands_hz}
        self.last_detected_freq = None
        self.last_detected_time = None
        self.pending_prediction = None
        self.predictions.clear()
        self.total_transitions = 0
        self.total_predictions = 0
        self.correct_predictions = 0
