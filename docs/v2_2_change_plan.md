# V2.2 Technical Change Plan — Metric-Correct Smart Scan & Explainable Dashboard
**Project:** SIH26055 – Smart Scan Strategy for Electronic Warfare  
**Version:** 2.2  
**Date:** 2026-09-04  
**Author:** AI Engineering Team  

---

## 1. Current Implementation Inspection

### 1.1 Architecture & Opportunity Tracking
In V2.1:
- `TransmissionOpportunity` (`rf_environment/domain/opportunity.py`):
  - Tracks a continuous transmission episode per emitter ID (`eid`).
  - Contains fields: `opportunity_id`, `emitter_id`, `start_step`, `end_step`, `trajectory` (dict mapping step to frequency), `receiver_entered`, `detected`, `first_intercept_time`, `time_to_intercept`, `status`.
- `OpportunityTracker` (`rf_environment/metrics/opportunity_tracker.py`):
  - Stores active opportunities in `self.active_opportunities: dict[str, TransmissionOpportunity]` keyed strictly by `eid`.
  - When `state.transmitting` is True:
    - If `eid not in self.active_opportunities`: creates a new `TransmissionOpportunity`.
    - If `eid in self.active_opportunities`: appends to `opp.trajectory[timestamp] = state.frequency_hz`.
  - Only when `state.transmitting` becomes False (or at simulation finalize) is the opportunity closed.
- Schedulers (`rf_environment/scheduler/`):
  - 7 schedulers supported: `SequentialScheduler`, `RandomScheduler`, `UCB1Scheduler`, `ThompsonSamplingScheduler`, `SlidingWindowUCBScheduler`, `DiscountedThompsonSamplingScheduler`, `ContextAwareScheduler`.
- Observation & Isolation (`rf_environment/environment/rf_environment.py`, `context_manager.py`):
  - Observation passes strictly receiver measurements (`timestamp`, `receiver_frequency_hz`, `receiver_bandwidth_hz`, `detected`, `snr_db`, `reward`).
  - `associated_emitter_id` is hidden from the scheduler (`None`).
  - Context manager maintains historical detections and band activities derived solely from receiver scans.
- Metrics Engine (`rf_environment/metrics/metrics_engine.py`):
  - Computes `opportunity_coverage`, `detection_given_coverage`, `interception_ratio`, `step_coverage_ratio`, `prediction_accuracy`.
- Dashboard (`dashboard/`):
  - Streamlit application displaying Live RF waterfall, performance metrics, comparison table, and intelligence view.

---

## 2. Identified Problems

1. **Continuous Hopping Emitter Over-Counting (Primary V2.1 Anomaly)**:
   - For continuous frequency-hopping emitters (e.g. `deterministic_hopping.yaml`, `random_hopping.yaml`), `state.transmitting` is True across all 1,000 steps.
   - Consequently, `OpportunityTracker` records exactly **1 opportunity** for the entire 1,000-step simulation.
   - The first time the receiver intersects the hopper (e.g. at step 3 or 4), `receiver_entered` becomes `True` and `detected` becomes `True`.
   - At simulation end: `total_opps = 1`, `intercepted = 1` $\rightarrow$ `interception_ratio = 100%`!
   - However, the empirical step-level overlap for a 5-channel random hopper is only ~20.0%, matching the theoretical $1/5$ probability.
   - Displaying `Interception = 100%` without distinguishing macro episode from micro hop is scientifically misleading.

2. **Absence of Hop / Dwell Granularity**:
   - In Electronic Warfare, frequency agility operates on discrete dwells / hops.
   - Each dwell represents a distinct transmission opportunity that can be intercepted or missed.
   - The current architecture lacks explicit `hop_index`, per-hop start/end steps, and per-hop detection flags.

3. **Ambiguity in Metric Scopes**:
   - Metrics do not declare whether they are evaluated at the `HOP`, `BURST`, `EPISODE`, or `STEP` level.
   - This prevents judges and operators from understanding what 100% vs 20% signifies.

4. **Missing "Why Did We Miss?" Diagnostic Aggregation**:
   - While `rf_environment.py` generates string reasons for each step, there is no aggregated statistical classification showing the breakdown between Off-Frequency (scheduling miss), Detector Miss (in-band SNR failure), or Inactive periods.

5. **Dashboard Explainability & Interactivity Gaps**:
   - Lacks step-by-step frozen realization Replay Mode with Play/Pause/Speed controls.
   - Lacks an Opportunity Inspector to inspect why a specific hop was intercepted or missed.

---

## 3. Required Changes

### 3.1 Domain & Opportunity Segmentation
1. **Extend `TransmissionOpportunity` (`rf_environment/domain/opportunity.py`)**:
   - Add fields:
     - `hop_index: int | None = None`
     - `frequency_hz: float | None = None` (center frequency of this hop/dwell)
     - `scope: str = "HOP"` (`"HOP"`, `"BURST"`, `"EPISODE"`)
     - `dwell_steps: int = 1`
   - Preserve `trajectory: dict[int, float]` for full backward compatibility.
   - Ensure explicit event separation:
     - `receiver_entered: bool`: receiver bandwidth overlapped emitter frequency during this opportunity.
     - `detected: bool`: detector successfully triggered during an in-band overlap.
     - `intercepted: bool`: property returning `self.receiver_entered and self.detected`.
2. **Expose Dwell & Hop State in `EmitterState` (`rf_environment/domain/emitter.py`, `emitters/base.py`)**:
   - Expose `dwell_steps: int = 1` and `hop_index: int | None = None` in `EmitterState` generated by `BaseEmitter.step()`.
   - For `FrequencyHopping` and `RandomFrequency`, `hop_index = time_step // dwell_steps`.
   - For `FixedFrequency`, `hop_index = None`, `dwell_steps = 1`.

### 3.2 Opportunity Tracker Refactoring (`rf_environment/metrics/opportunity_tracker.py`)
1. **Per-Emitter, Per-Hop/Burst Opportunity Lifecycle**:
   - Support distinct emitter semantics:
     - **Fixed Continuous Emitter**: 1 opportunity across the continuous transmission episode.
     - **Periodic / Bursty Emitter**: 1 opportunity per ON burst (opened on rising edge of `transmitting`, closed on falling edge).
     - **Frequency-Hopping / Agile Emitter**: 1 opportunity per discrete hop/dwell!
       - Transition detected if `state.hop_index != active_opp.hop_index` or (`state.frequency_hz != active_opp.frequency_hz` when `dwell_steps == 1`).
       - On hop transition: finalize previous hop opportunity (`end_step = timestamp - 1`, `status = "INTERCEPTED" if detected else "MISSED"`), and initialize new hop opportunity with `opportunity_id = f"{eid}_HOP_{hop_counter:04d}"`.
       - When emitter stops transmitting, finalize active hop.
2. **Dual-Scope Accounting (Hop-Level & Episode-Level)**:
   - Compute both:
     - **Hop-Level**: `hop_coverage_ratio`, `hop_interception_ratio`, `hop_detection_ratio` (Detection Given Coverage).
     - **Episode / Burst-Level**: `opportunity_coverage`, `interception_ratio`, `detection_given_coverage`.
   - For hopping scenarios, hop-level is the primary agile metric; for periodic scenarios, burst-level is primary.

### 3.3 Metrics Engine Updates (`rf_environment/metrics/metrics_engine.py`, `domain/metrics.py`)
1. **Add Explicit Fields to `MetricSnapshot`**:
   - `hop_coverage_ratio: float = 0.0`
   - `hop_interception_ratio: float = 0.0`
   - `hop_detection_ratio: float = 0.0`
   - `total_hops: int = 0`
   - `covered_hops: int = 0`
   - `intercepted_hops: int = 0`
   - `metric_scope: str = "HOP"`
2. **Strict Identity Enforcement**:
   - Verify mathematically:
     $$\text{Interception Ratio} \approx \text{Coverage} \times \text{Detection Given Coverage}$$
     $$\text{Hop Interception Ratio} \approx \text{Hop Coverage} \times \text{Detection Given Coverage}$$
     $$| \text{LHS} - \text{RHS} | < 10^{-5}$$
   - Raise diagnostic warning if violated.

### 3.4 Preservation of Ground-Truth Isolation
- Ensure `Observation` never exposes `EmitterState`, `ground_truth`, `transmitting_ids`, or true frequency.
- Context-Aware scheduler continues to learn $P(f_{next} | f_{prev})$ strictly from observed detector firings.
- Add regression test `test_scheduler_has_no_ground_truth_access()`.

### 3.5 Benchmark Runner (`rf_environment/experiments/runner.py`)
- Update `run_single` and `run_benchmark` to log `hop_coverage_ratio`, `hop_interception_ratio`, `hop_detection_ratio`, `step_coverage_ratio`, `opportunity_coverage`, `interception_ratio`.
- Preserve frozen realization across seeds.

### 3.6 Explainable 4-Tab Dashboard Redesign (`dashboard/`)
- **Tab 1: Live RF / Smart Scan**:
  - Top KPI cards: Current Rx Freq, Rx Bandwidth, Current Detection, SNR, Active Scheduler, Reward.
  - Interactive Altair waterfall with moving receiver bandwidth window $[f_c - BW/2, f_c + BW/2]$.
  - Visual legend: Emitter active, Hopping trajectory, Receiver window, Scan Hit, Scan Miss, False Alarm, Off-Frequency.
  - "Current Decision" panel: "WHY THIS FREQUENCY?" detailing actual scheduler parameters (e.g. Context-Aware transition prob + activity + exploration; UCB1 mean + bonus; TS sample; etc.).
  - Replay Mode: Play, Pause, Restart, Speed controls (1x, 5x, 10x), step scrubber.
  - Opportunity Inspector: Click or select individual hops/bursts from a table or slider to inspect start, end, frequency, coverage, detection, intercept latency.
- **Tab 2: Performance**:
  - Clear separation of Hop-Level vs Episode-Level KPIs.
  - Algorithm comparison table across all 7 schedulers with highlighted top performers.
- **Tab 3: Why Did We Miss?**:
  - Classifies every step and opportunity into:
    1. Off-Frequency (Receiver in wrong band)
    2. Correct Frequency but Detector Miss ($SNR < \gamma$ or detection failure)
    3. Emitter Not Transmitting (Receiver scanned empty channel while target idle)
    4. No Valid Opportunity
  - Visual breakdown and percentage breakdown.
- **Tab 4: Algorithm Intelligence**:
  - Observed Transition Probability Matrix heatmap ($P(f_{next} | f_{prev})$).
  - Current prediction vs actual receiver action.
  - Detailed algorithm-specific score breakdown.
  - Periodic aliasing diagnostic ($T_{scan}$ vs $T_{emitter}$, phase offset, synchronization warning).

---

## 4. Files Affected

| File | Change Description |
|------|--------------------|
| `rf_environment/domain/opportunity.py` | Add `hop_index`, `frequency_hz`, `scope`, `dwell_steps` to `TransmissionOpportunity` |
| `rf_environment/domain/emitter.py` | Add `dwell_steps` and `hop_index` to `EmitterState` |
| `rf_environment/emitters/base.py` | Populate `dwell_steps` and `hop_index` during `step()` |
| `rf_environment/metrics/opportunity_tracker.py` | Refactor opportunity lifecycle for per-hop and per-burst segmentation |
| `rf_environment/domain/metrics.py` | Add hop-level metric fields and `metric_scope` to `MetricSnapshot` |
| `rf_environment/metrics/metrics_engine.py` | Integrate hop-level computation and mathematical identity verification |
| `rf_environment/environment/rf_environment.py` | Ensure event logging and waterfall reflect per-hop opportunities |
| `rf_environment/experiments/runner.py` | Include hop metrics and scope in benchmark outputs and exports |
| `rf_environment/api/routes.py` | Ensure `/metrics` and `/opportunities` endpoints expose hop metrics |
| `dashboard/app.py` | Redesign layout to 4 main tabs, add Replay controls and Current Decision panel |
| `dashboard/simulation_runner.py` | Collect opportunity diagnostic records and miss classifications |
| `dashboard/utils.py` | Add helpers for miss classification, opportunity inspection, and decision explanations |
| `dashboard/components/spectral_view.py` | Add Current Decision panel and Replay mode integration |
| `dashboard/components/metrics_view.py` | Update with hop-level KPIs and scoped cards |
| `dashboard/components/comparison_view.py` | Update scoreboard with Hop Coverage, Hop Interception, DGC |
| `dashboard/components/intelligence_view.py` | Enhance with transition heatmap and periodic aliasing diagnostic |
| `dashboard/components/why_missed_view.py` | New component for Tab 3 "Why Did We Miss?" |
| `tests/test_v2_2_opportunities.py` | Comprehensive regression suite for V2.2 requirements |

---

## 5. Backward Compatibility Considerations

1. **API & Snapshot Compatibility**:
   - `interception_ratio`, `opportunity_coverage`, `detection_given_coverage`, `step_coverage_ratio` remain intact with their existing data types.
   - For continuous hopping scenarios, `opportunity_coverage` and `interception_ratio` will now reflect hop-level segmentation (or both episode and hop fields will be provided).
   - Any external client consuming V2.0/V2.1 JSON outputs will continue to function without schema errors.
2. **Scheduler Interface**:
   - The 7 schedulers retain identical signatures (`select_action(obs)`, `update(obs, reward, action)`).
   - No scheduler logic is changed merely to inflate scores.
3. **No New Algorithms**:
   - No Deep RL, neural networks, or unapproved heuristics are introduced.

---

## 6. Test & Verification Plan

1. **Automated Regression Suite (`tests/test_v2_2_opportunities.py`)**:
   - `test_opportunity_lifecycle_continuous_fixed`: Continuous fixed emitter yields exactly 1 opportunity.
   - `test_opportunity_lifecycle_periodic_burst`: Periodic emitter with $K$ ON bursts yields exactly $K$ burst opportunities.
   - `test_opportunity_lifecycle_hopping_dwell_1`: Hopping emitter with dwell=1 yields $N$ hop opportunities for $N$ steps.
   - `test_opportunity_lifecycle_hopping_dwell_5`: Hopping emitter with dwell=5 yields $N/5$ hop opportunities.
   - `test_random_hopper_baseline_20_percent`: 5-channel uniform random hopper scanned by random receiver yields $\approx 20\%$ hop coverage and $\approx 20\%$ hop interception.
   - `test_metric_identity_mathematical_precision`: Verifies $|\text{IR} - (\text{Cov} \times \text{DGC})| < 10^{-5}$ across all scenarios.
   - `test_scheduler_has_no_ground_truth_access`: Asserts observations contain no ground-truth leakages.
   - `test_frozen_realization_determinism`: Identical seeds yield identical trajectories across all 7 schedulers.
   - `test_periodic_aliasing_diagnostic`: Period 20 vs 20 exhibits aliasing; Period 19 vs 20 breaks aliasing.
   - `test_detector_scheduler_separation`: Perfect detector yields 100% DGC, demonstrating that interception limits arise from scheduling.
2. **Existing Test Suite**:
   - Run `PYTHONPATH=. .venv/bin/python run_tests.py` and verify all 26 existing tests pass (with updated assertions where per-hop semantics apply).
3. **Verification Experiments**:
   - Execute Test A (Random Hopper), Test B (Deterministic Hopper), Test C (Dwell Sensitivity), Test D (Periodic Aliasing), Test E (Multi-Emitter).
   - Generate `docs/v2_2_validation_report.md`.
