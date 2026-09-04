from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np

from rf_environment.domain.emitter import EmitterState
from rf_environment.domain.enums import FrequencyBehaviorType, TimeBehaviorType
from rf_environment.domain.opportunity import TransmissionOpportunity
from rf_environment.receiver.tuner import bands_overlap

logger = logging.getLogger(__name__)


class OpportunityTracker:
    """Tracks transmission opportunities and evaluates interception performance.

    Supports three opportunity lifecycles:
    1. Frequency-Hopping / Agile Emitter:
       - Opportunity scope: "HOP"
       - Each discrete dwell / hop is an individual opportunity.
       - A new opportunity starts on each hop index transition or frequency change.
    2. Periodic / Bursty Emitter:
       - Opportunity scope: "BURST"
       - Each active ON burst is an individual opportunity.
    3. Fixed Continuous Emitter:
       - Opportunity scope: "EPISODE"
       - One opportunity spans the continuous transmission duration.

    Strictly separates:
    - receiver_entered: Did receiver tune to an overlapping band during the opportunity?
    - detected: Was an actual detection reported during overlap?
    - intercepted: receiver_entered AND detected.
    - time_to_intercept: first_intercept_time - start_step.
    """

    def __init__(self) -> None:
        # Active discrete opportunities keyed by emitter_id
        self.active_opportunities: dict[str, TransmissionOpportunity] = {}
        self.completed_opportunities: list[TransmissionOpportunity] = []
        self._opportunity_counters: dict[str, int] = {}

        # Macro episode tracking (for dual-scope reporting)
        self.active_episodes: dict[str, TransmissionOpportunity] = {}
        self.completed_episodes: list[TransmissionOpportunity] = []
        self._episode_counters: dict[str, int] = {}

    def _is_agile(self, state: EmitterState) -> bool:
        return state.frequency_behavior in (
            FrequencyBehaviorType.HOPPING,
            FrequencyBehaviorType.RANDOM,
            FrequencyBehaviorType.SWEEP,
        )

    def _opportunity_scope(self, state: EmitterState) -> str:
        if self._is_agile(state):
            return "HOP"
        if state.time_behavior in (
            TimeBehaviorType.PERIODIC,
            TimeBehaviorType.BURST,
            TimeBehaviorType.INTERMITTENT,
        ):
            return "BURST"
        return "EPISODE"

    def step(
        self,
        timestamp: int,
        emitter_states: list[EmitterState],
        rx_center_freq_hz: float,
        rx_bandwidth_hz: float,
        detector_detected: bool,
        is_false_alarm: bool = False,
    ) -> list[TransmissionOpportunity]:
        """Updates opportunity tracking for one simulation step."""
        transmitting_emitter_ids = set()

        for state in emitter_states:
            eid = state.emitter_id
            is_agile = self._is_agile(state)
            scope = self._opportunity_scope(state)

            if state.transmitting:
                transmitting_emitter_ids.add(eid)

                # -------------------------------------------------------------
                # 1. Macro Episode Tracking (Continuous span of transmission)
                # -------------------------------------------------------------
                if eid not in self.active_episodes:
                    self._episode_counters[eid] = self._episode_counters.get(eid, 0) + 1
                    ep = TransmissionOpportunity(
                        opportunity_id=f"{eid}_EP_{self._episode_counters[eid]:04d}",
                        emitter_id=eid,
                        start_step=timestamp,
                        trajectory={timestamp: state.frequency_hz},
                        bandwidth_hz=state.bandwidth_hz,
                        power_dbm=state.power_dbm,
                        scope="EPISODE",
                    )
                    self.active_episodes[eid] = ep
                else:
                    self.active_episodes[eid].trajectory[timestamp] = state.frequency_hz

                # Check macro episode overlap
                ep = self.active_episodes[eid]
                ep_in_band = bands_overlap(
                    rx_center_freq_hz,
                    rx_bandwidth_hz,
                    state.frequency_hz,
                    state.bandwidth_hz,
                )
                if ep_in_band:
                    ep.receiver_entered = True
                    if detector_detected and not is_false_alarm:
                        if not ep.detected:
                            ep.detected = True
                            ep.first_intercept_time = timestamp
                            ep.time_to_intercept = max(0, timestamp - ep.start_step)

                # -------------------------------------------------------------
                # 2. Discrete Opportunity Tracking (Hop, Burst, or Fixed Episode)
                # -------------------------------------------------------------
                active_opp = self.active_opportunities.get(eid)
                need_new_opportunity = False

                if active_opp is None:
                    need_new_opportunity = True
                elif is_agile:
                    # For agile emitters, determine if hop transitioned
                    if state.hop_index is not None and active_opp.hop_index is not None:
                        hop_changed = state.hop_index != active_opp.hop_index
                    else:
                        hop_changed = abs(state.frequency_hz - (active_opp.frequency_hz or 0.0)) > 1e-6

                    if hop_changed:
                        # Finalize previous hop
                        active_opp.end_step = timestamp - 1
                        active_opp.status = "INTERCEPTED" if active_opp.detected else "MISSED"
                        self.completed_opportunities.append(active_opp)
                        need_new_opportunity = True

                if need_new_opportunity:
                    self._opportunity_counters[eid] = self._opportunity_counters.get(eid, 0) + 1
                    opp_id = f"{eid}_{scope}_{self._opportunity_counters[eid]:04d}"
                    opp = TransmissionOpportunity(
                        opportunity_id=opp_id,
                        emitter_id=eid,
                        start_step=timestamp,
                        trajectory={timestamp: state.frequency_hz},
                        frequency_hz=state.frequency_hz,
                        bandwidth_hz=state.bandwidth_hz,
                        power_dbm=state.power_dbm,
                        hop_index=state.hop_index,
                        dwell_steps=state.dwell_steps,
                        scope=scope,
                    )
                    self.active_opportunities[eid] = opp
                else:
                    opp = self.active_opportunities[eid]
                    opp.trajectory[timestamp] = state.frequency_hz

                # Check discrete opportunity overlap
                in_band = bands_overlap(
                    rx_center_freq_hz,
                    rx_bandwidth_hz,
                    state.frequency_hz,
                    state.bandwidth_hz,
                )
                if in_band:
                    opp.receiver_entered = True
                    if detector_detected and not is_false_alarm:
                        if not opp.detected:
                            opp.detected = True
                            opp.first_intercept_time = timestamp
                            opp.time_to_intercept = max(0, timestamp - opp.start_step)

        # Close any opportunities/episodes for emitters that stopped transmitting
        closed_eids = [eid for eid in self.active_opportunities if eid not in transmitting_emitter_ids]
        for eid in closed_eids:
            opp = self.active_opportunities.pop(eid)
            opp.end_step = timestamp - 1
            opp.status = "INTERCEPTED" if opp.detected else "MISSED"
            self.completed_opportunities.append(opp)

        closed_ep_eids = [eid for eid in self.active_episodes if eid not in transmitting_emitter_ids]
        for eid in closed_ep_eids:
            ep = self.active_episodes.pop(eid)
            ep.end_step = timestamp - 1
            ep.status = "INTERCEPTED" if ep.detected else "MISSED"
            self.completed_episodes.append(ep)

        return self.get_all_opportunities()

    def finalize(self, last_timestamp: int) -> None:
        """Closes any remaining active opportunities at simulation completion."""
        for opp in list(self.active_opportunities.values()):
            opp.end_step = last_timestamp
            opp.status = "INTERCEPTED" if opp.detected else "MISSED"
            self.completed_opportunities.append(opp)
        self.active_opportunities.clear()

        for ep in list(self.active_episodes.values()):
            ep.end_step = last_timestamp
            ep.status = "INTERCEPTED" if ep.detected else "MISSED"
            self.completed_episodes.append(ep)
        self.active_episodes.clear()

    def get_all_opportunities(self) -> list[TransmissionOpportunity]:
        """Returns all completed and currently active opportunities."""
        active = list(self.active_opportunities.values())
        for opp in active:
            opp.status = "INTERCEPTED" if opp.detected else "ACTIVE"
        return self.completed_opportunities + active

    def get_all_episodes(self) -> list[TransmissionOpportunity]:
        """Returns all completed and active macro episodes."""
        active = list(self.active_episodes.values())
        for ep in active:
            ep.status = "INTERCEPTED" if ep.detected else "ACTIVE"
        return self.completed_episodes + active

    def compute_summary(self) -> dict[str, Any]:
        """Computes formal opportunity-based interception metrics across scopes."""
        all_opps = self.get_all_opportunities()
        total = len(all_opps)

        # Determine evaluation scope: HOP > BURST > EPISODE
        hop_opps = [o for o in all_opps if o.scope == "HOP"]
        burst_opps = [o for o in all_opps if o.scope == "BURST"]
        if hop_opps:
            metric_scope = "HOP"
        elif burst_opps:
            metric_scope = "BURST"
        else:
            metric_scope = "EPISODE"

        if total == 0:
            return {
                "metric_scope": metric_scope,
                "opportunities_total": 0,
                "opportunities_covered": 0,
                "opportunities_intercepted": 0,
                "opportunity_coverage": 0.0,
                "detection_given_coverage": 0.0,
                "interception_ratio": 0.0,
                "total_hops": 0,
                "covered_hops": 0,
                "intercepted_hops": 0,
                "hop_coverage_ratio": 0.0,
                "hop_detection_ratio": 0.0,
                "hop_interception_ratio": 0.0,
                "total_episodes": 0,
                "covered_episodes": 0,
                "intercepted_episodes": 0,
                "episode_coverage_ratio": 0.0,
                "episode_interception_ratio": 0.0,
                "episode_detection_given_coverage": 0.0,
                "average_intercept_time": None,
                "median_intercept_time": None,
                "intercept_time_std": None,
                "identity_verified": True,
            }

        covered = sum(1 for o in all_opps if o.receiver_entered)
        intercepted = sum(1 for o in all_opps if o.intercepted)

        coverage = covered / total if total > 0 else 0.0
        det_given_cov = intercepted / covered if covered > 0 else 0.0
        interception_ratio = intercepted / total if total > 0 else 0.0

        # Mathematical Identity Verification: IR == Coverage * DGC
        identity_verified = True
        if covered > 0:
            lhs = interception_ratio
            rhs = coverage * det_given_cov
            if abs(lhs - rhs) > 1e-5:
                identity_verified = False
                logger.warning(
                    f"Metric identity check failure: IR ({lhs:.6f}) != Coverage ({coverage:.6f}) * DGC ({det_given_cov:.6f})"
                )

        # Explicit Hop-Level Metrics
        total_hops = len(hop_opps)
        covered_hops = sum(1 for o in hop_opps if o.receiver_entered)
        intercepted_hops = sum(1 for o in hop_opps if o.intercepted)
        hop_cov = covered_hops / total_hops if total_hops > 0 else 0.0
        hop_dgc = intercepted_hops / covered_hops if covered_hops > 0 else 0.0
        hop_ir = intercepted_hops / total_hops if total_hops > 0 else 0.0

        # Macro Episode Metrics
        all_episodes = self.get_all_episodes()
        total_ep = len(all_episodes)
        covered_ep = sum(1 for e in all_episodes if e.receiver_entered)
        intercepted_ep = sum(1 for e in all_episodes if e.intercepted)
        ep_cov = covered_ep / total_ep if total_ep > 0 else 0.0
        ep_dgc = intercepted_ep / covered_ep if covered_ep > 0 else 0.0
        ep_ir = intercepted_ep / total_ep if total_ep > 0 else 0.0

        delays = [o.time_to_intercept for o in all_opps if o.time_to_intercept is not None]
        avg_delay = float(np.mean(delays)) if delays else None
        median_delay = float(np.median(delays)) if delays else None
        std_delay = float(np.std(delays)) if len(delays) > 1 else (0.0 if delays else None)

        return {
            "metric_scope": metric_scope,
            # Primary Opportunity Metrics (scoped to current scenario type)
            "opportunities_total": total,
            "opportunities_covered": covered,
            "opportunities_intercepted": intercepted,
            "opportunity_coverage": coverage,
            "detection_given_coverage": det_given_cov,
            "interception_ratio": interception_ratio,
            # Explicit Hop Metrics
            "total_hops": total_hops,
            "covered_hops": covered_hops,
            "intercepted_hops": intercepted_hops,
            "hop_coverage_ratio": hop_cov,
            "hop_detection_ratio": hop_dgc,
            "hop_interception_ratio": hop_ir,
            # Macro Episode Metrics
            "total_episodes": total_ep,
            "covered_episodes": covered_ep,
            "intercepted_episodes": intercepted_ep,
            "episode_coverage_ratio": ep_cov,
            "episode_detection_given_coverage": ep_dgc,
            "episode_interception_ratio": ep_ir,
            # Delay Metrics
            "average_intercept_time": avg_delay,
            "median_intercept_time": median_delay,
            "intercept_time_std": std_delay,
            "identity_verified": identity_verified,
        }

    def reset(self) -> None:
        self.active_opportunities.clear()
        self.completed_opportunities.clear()
        self._opportunity_counters.clear()
        self.active_episodes.clear()
        self.completed_episodes.clear()
        self._episode_counters.clear()
