from __future__ import annotations

from rf_environment.domain.enums import FrequencyBehaviorType, Serializable


class FrequencySnapshot(Serializable):
    behavior: FrequencyBehaviorType
    frequency_hz: float
