from __future__ import annotations

import math

from rf_environment.domain.emitter import EmitterState


class FreeSpacePathLoss:
    """Optional spatial abstraction. Disabled when reference_distance_m is None."""

    def __init__(self, enabled: bool = False, reference_distance_m: float = 1.0) -> None:
        self.enabled = enabled
        self.reference_distance_m = reference_distance_m

    def attenuation_db(self, emitter: EmitterState, rx_position: tuple[float, float, float]) -> float:
        if not self.enabled:
            return 0.0
        dx = emitter.position_x - rx_position[0]
        dy = emitter.position_y - rx_position[1]
        dz = emitter.position_z - rx_position[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        distance = max(distance, self.reference_distance_m)
        c = 299_792_458.0
        wavelength = c / max(emitter.frequency_hz, 1.0)
        fspl = 20.0 * math.log10(4.0 * math.pi * distance / wavelength)
        return fspl
