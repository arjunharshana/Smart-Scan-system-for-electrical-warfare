from __future__ import annotations

from collections import deque
from typing import Callable

from rf_environment.domain.enums import EventType
from rf_environment.domain.events import SimulationEvent


EventHandler = Callable[[SimulationEvent], None]


class EventStream:
    def __init__(self, history_limit: int = 2000) -> None:
        self.history: deque[SimulationEvent] = deque(maxlen=history_limit)
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def publish(self, timestamp: int, event_type: EventType, /, **payload) -> SimulationEvent:
        event = SimulationEvent(timestamp=timestamp, event_type=event_type, payload=payload)
        self.history.append(event)
        for handler in list(self._handlers):
            handler(event)
        return event

    def recent(self, limit: int = 100) -> list[SimulationEvent]:
        items = list(self.history)
        return items[-limit:]

    def reset(self) -> None:
        self.history.clear()
