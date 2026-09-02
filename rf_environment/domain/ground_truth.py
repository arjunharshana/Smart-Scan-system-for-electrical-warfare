from __future__ import annotations

from rf_environment.domain.emitter import EmitterState
from rf_environment.domain.enums import Serializable


class GroundTruthFrame(Serializable):
    timestamp: int
    emitters: list[EmitterState]
    transmitting_ids: list[str]


class ScanOutcome(Serializable):
    timestamp: int
    outcome: str
    detected: bool
    in_band_transmitting_ids: list[str]
    intercepted_ids: list[str]
    associated_emitter_id: str | None = None
    reward: float = 0.0
