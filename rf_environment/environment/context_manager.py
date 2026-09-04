from __future__ import annotations

from collections import deque
from typing import Any

from rf_environment.domain.observation import Observation, TemporalHistory


class TemporalContextManager:
    """Maintains temporal context and historical observation features.

    Strictly uses receiver observations and actions. Never accesses ground truth.
    """

    def __init__(self, bands_hz: list[float], window_size: int = 20) -> None:
        self.bands_hz = sorted([float(b) for b in bands_hz])
        self.window_size = window_size

        self.recent_frequencies: deque[float] = deque(maxlen=window_size)
        self.recent_detections: deque[bool] = deque(maxlen=window_size)
        self.recent_rewards: deque[float] = deque(maxlen=window_size)

        self.last_scanned_time: dict[float, int] = {}
        self.last_detected_freq: float | None = None
        self.last_detected_time: int | None = None

        self.recent_transitions: deque[list[float]] = deque(maxlen=10)
        self.last_action: float | None = None

        # Per-band recent window detection tracker
        self._band_window: dict[float, deque[int]] = {
            b: deque(maxlen=window_size) for b in self.bands_hz
        }

    def update(
        self,
        timestamp: int,
        scanned_freq_hz: float,
        detected: bool,
        reward: float,
    ) -> None:
        """Updates internal history from a single receiver scan observation."""
        band = self._closest_band(scanned_freq_hz)
        self.recent_frequencies.append(scanned_freq_hz)
        self.recent_detections.append(detected)
        self.recent_rewards.append(reward)
        self.last_scanned_time[band] = timestamp
        self.last_action = scanned_freq_hz

        # Update per-band window
        self._band_window[band].append(1 if detected else 0)

        if detected:
            if self.last_detected_freq is not None:
                self.recent_transitions.append([self.last_detected_freq, scanned_freq_hz])
            self.last_detected_freq = scanned_freq_hz
            self.last_detected_time = timestamp

    def _closest_band(self, freq_hz: float) -> float:
        return min(self.bands_hz, key=lambda b: abs(b - freq_hz))

    def build_history(self, current_time: int) -> TemporalHistory:
        """Constructs a clean TemporalHistory domain object."""
        time_since_detect = (
            current_time - self.last_detected_time
            if self.last_detected_time is not None
            else None
        )

        activity_levels = {}
        time_since_scanned = {}
        for b in self.bands_hz:
            key = f"{b/1e6:.1f}"
            win = self._band_window[b]
            activity_levels[key] = sum(win) / len(win) if win else 0.0
            last_t = self.last_scanned_time.get(b)
            time_since_scanned[key] = (current_time - last_t) if last_t is not None else 999

        return TemporalHistory(
            previous_frequencies=list(self.recent_frequencies),
            previous_detections=list(self.recent_detections),
            previous_rewards=list(self.recent_rewards),
            last_detected_frequency_hz=self.last_detected_freq,
            time_since_last_detection=time_since_detect,
            band_activity_levels=activity_levels,
            time_since_band_last_scanned=time_since_scanned,
            recent_transitions=list(self.recent_transitions),
        )

    def enrich_observation(self, observation: Observation) -> Observation:
        """Attaches temporal history and previous action to the observation."""
        observation.previous_action = self.last_action
        observation.recent_history = self.build_history(observation.timestamp)
        return observation

    def reset(self) -> None:
        self.recent_frequencies.clear()
        self.recent_detections.clear()
        self.recent_rewards.clear()
        self.last_scanned_time.clear()
        self.last_detected_freq = None
        self.last_detected_time = None
        self.recent_transitions.clear()
        self.last_action = None
        for b in self.bands_hz:
            self._band_window[b].clear()
