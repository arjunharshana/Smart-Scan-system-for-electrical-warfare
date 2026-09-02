from __future__ import annotations

from rf_environment.domain.enums import EmitterType
from rf_environment.emitters.base import BaseEmitter


class CommunicationEmitter(BaseEmitter):
    emitter_type = EmitterType.COMMUNICATION
