from __future__ import annotations

from rf_environment.domain.signal import IdealSignal
from rf_environment.channel.noise import dbm_to_mw, mw_to_dbm


class Interference:
    """Sums overlapping in-band interferers as extra noise-like power."""

    def extra_power_dbm(self, in_band: list[IdealSignal], primary: IdealSignal | None) -> float:
        others = [s for s in in_band if primary is None or s.emitter_id != primary.emitter_id]
        if not others:
            return -200.0
        return mw_to_dbm(sum(dbm_to_mw(s.power_dbm) for s in others))
