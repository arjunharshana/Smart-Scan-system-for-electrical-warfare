from __future__ import annotations

from collections import deque

from rf_environment.domain.emitter import EmitterState
from rf_environment.domain.ground_truth import GroundTruthFrame


class GroundTruthStore:
    def __init__(self, history_limit: int = 5000) -> None:
        self.history: deque[GroundTruthFrame] = deque(maxlen=history_limit)
        self.latest: GroundTruthFrame | None = None

    def record(self, timestamp: int, emitters: list[EmitterState]) -> GroundTruthFrame:
        frame = GroundTruthFrame(
            timestamp=timestamp,
            emitters=emitters,
            transmitting_ids=[e.emitter_id for e in emitters if e.transmitting],
        )
        self.latest = frame
        self.history.append(frame)
        return frame

    def reset(self) -> None:
        self.history.clear()
        self.latest = None
