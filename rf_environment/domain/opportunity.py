from __future__ import annotations

from typing import Any
from rf_environment.domain.enums import Serializable


class TransmissionOpportunity(Serializable):
    """Represents a transmission opportunity (hop, burst, or episode) by an emitter.
    
    Supports discrete frequency-agile hops/dwells as well as continuous/burst episodes.
    """

    opportunity_id: str
    emitter_id: str
    start_step: int
    end_step: int | None = None
    # Map step index -> frequency in Hz active at that step
    trajectory: dict[int, float] = {}
    frequency_hz: float | None = None  # Primary center frequency for this opportunity
    bandwidth_hz: float = 0.0
    power_dbm: float = 0.0
    hop_index: int | None = None  # Dwell/hop counter for agile emitters
    dwell_steps: int = 1  # Duration of dwell in simulation steps
    scope: str = "HOP"  # HOP, BURST, EPISODE
    receiver_entered: bool = False  # Did receiver window cover active frequency at any step?
    detected: bool = False  # Was a detection reported during an overlap?
    first_intercept_time: int | None = None  # Step index of first successful intercept
    time_to_intercept: int | None = None  # first_intercept_time - start_step
    status: str = "ACTIVE"  # ACTIVE, INTERCEPTED, MISSED, EXPIRED

    @property
    def intercepted(self) -> bool:
        """True if successfully intercepted (requires both receiver_entered and detected)."""
        return self.receiver_entered and self.detected

    def to_summary_dict(self) -> dict[str, Any]:
        """Human-readable dictionary for dashboard and telemetry reporting."""
        if self.frequency_hz is not None:
            freq_str = f"{self.frequency_hz / 1e6:.1f} MHz"
        else:
            freq_values = list(self.trajectory.values())
            if not freq_values:
                freq_str = "Unknown"
            elif len(set(freq_values)) == 1:
                freq_str = f"{freq_values[0] / 1e6:.1f} MHz"
            else:
                freq_str = f"{min(freq_values)/1e6:.1f}–{max(freq_values)/1e6:.1f} MHz ({len(freq_values)} hops)"

        return {
            "Opportunity ID": self.opportunity_id,
            "Emitter ID": self.emitter_id,
            "Scope": self.scope,
            "Hop Index": self.hop_index if self.hop_index is not None else "—",
            "Start Step": self.start_step,
            "End Step": self.end_step if self.end_step is not None else "Active",
            "Duration": (self.end_step - self.start_step + 1) if self.end_step is not None else "Ongoing",
            "Frequency": freq_str,
            "Receiver Entered": "✅ Yes" if self.receiver_entered else "❌ No",
            "Detected": "🎯 Intercepted" if self.intercepted else ("⚠️ Covered Only" if self.receiver_entered else "❌ Missed"),
            "First Intercept": f"Step {self.first_intercept_time}" if self.first_intercept_time is not None else "—",
            "Delay (Steps)": self.time_to_intercept if self.time_to_intercept is not None else "—",
            "Status": self.status,
        }
