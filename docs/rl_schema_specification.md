# RL-Ready Observation & Action Schema Specification
## SIH26055 – Smart Scan Strategy for Electronic Warfare

This document defines the canonical observation, action, and transition schemas implemented for the reinforcement learning foundation of the Smart Scan EW system.

---

## 1. System Architecture & Information Barrier

```text
                         ┌─────────────────────┐
                         │   RF Environment    │
                         │                     │
                         │   Hidden RF State   │
                         │   Emitters          │
                         │   Propagation       │
                         │   Additive Noise    │
                         └──────────┬──────────┘
                                    │
                              Receiver/Detector
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Observation Builder │
                         │                     │
                         │ 10 Frozen Fields    │
                         │ Zero Future Lookahead│
                         │ Zero Ground Truth   │
                         └──────────┬──────────┘
                                    │
                            SchedulerObservation
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    BaseScheduler    │
                         │                     │
                         │ Sequential          │
                         │ Random              │
                         │ UCB1                │
                         │ Thompson Sampling   │
                         │ SW-UCB              │
                         │ Discounted TS       │
                         │ Context-Aware       │
                         │ RLScheduler         │
                         └──────────┬──────────┘
                                    │
                                ScanAction(frequency_bin)
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Scan Controller   │
                         │                     │
                         │ Action Validation   │
                         │ Frequency Bin Map   │
                         │ Receiver Tuning     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              Receiver / RF Env
```

Separately and independently:

```text
Hidden RF State ──→ Ground Truth ──→ Evaluator (OpportunityTracker & MetricsEngine)
                                            │
                                            ├── STEP Metrics
                                            ├── HOP Metrics
                                            ├── BURST Metrics
                                            ├── EPISODE Metrics
                                            └── Coverage / Interception Precision
```

---

## 2. Canonical Observation Schema (`SchedulerObservation`)

### 2.1 Dataclass Definition

The observation schema is frozen and immutable:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SchedulerObservation:
    timestamp: float
    current_frequency_bin: int
    last_detection: bool
    last_detection_bin: int | None
    last_detection_strength: float | None
    recent_detection_history: tuple[bool, ...]
    recent_frequency_history: tuple[int, ...]
    scan_count_by_bin: tuple[int, ...]
    time_since_scan_by_bin: tuple[float, ...]
    time_since_last_detection: float
```

### 2.2 Field Provenance & Ground-Truth Isolation Table

| # | Field Name | Data Type | Source Component | Available at Decision Time? | Ground Truth? | Description |
|---|---|---|---|---|---|---|
| 1 | `timestamp` | `float` | Simulation Clock | **Yes** | **No** | Elapsed simulation time exposed to receiver |
| 2 | `current_frequency_bin` | `int` | Scan Controller | **Yes** | **No** | Discrete bin index of currently scanned frequency band (bash \le 	ext{bin} < N$) |
| 3 | `last_detection` | `bool` | Energy Detector | **Yes** | **No** | Empirical detection boolean from the most recent scan |
| 4 | `last_detection_bin` | `int \| None` | Receiver / Detector | **Yes** | **No** | Most recent frequency bin where a detection was observed, or `None` |
| 5 | `last_detection_strength` | `float \| None` | Receiver Front-End | **Yes** | **No** | Measured in-band signal power in dBm of the last detection |
| 6 | `recent_detection_history` | `tuple[bool, ...]` | Observation Builder Memory | **Yes** | **No** | Bounded tuple (=10$) of recent detection outcomes |
| 7 | `recent_frequency_history` | `tuple[int, ...]` | Observation Builder Memory | **Yes** | **No** | Bounded tuple (=10$) of recent scanned frequency bin indices |
| 8 | `scan_count_by_bin` | `tuple[int, ...]` | Observation Builder Memory | **Yes** | **No** | Total scan count accumulated per frequency bin across the episode |
| 9 | `time_since_scan_by_bin` | `tuple[float, ...]` | Observation Builder Memory | **Yes** | **No** | Steps elapsed since each bin was last scanned (99.0$ for unscanned bins) |
| 10 | `time_since_last_detection` | `float` | Derived Observable | **Yes** | **No** | Steps elapsed since receiver reported any detection (99.0$ if none) |

### 2.3 Forbidden Ground-Truth Elements (Strictly Isolated)

The following ground-truth information is strictly isolated in the simulator/evaluator and is **never** present in `SchedulerObservation`:
- `emitters`: Hidden real emitter configurations, true frequencies, powers, and trajectories.
- `in_band_emitter_ids` / `transmitting_ids`: Ground-truth emitter identities.
- `future_hops` / `next_transmission_time`: Future RF emission events.
- `opportunity_id` / `scope`: Evaluator-only opportunity tracking labels.
- `reward`: Rewards are returned separately in `StepResult` / `Transition` tuples and never embedded inside `SchedulerObservation`.

---

## 3. Canonical Action Schema (`ScanAction`)

### 3.1 Dataclass Definition

The canonical action schema represents a pure frequency bin choice:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ScanAction:
    frequency_bin: int
```

### 3.2 Action Control Scope

- **What the scheduler controls**:
  - `frequency_bin`: Which channel to scan: $\mathcal{A} = \{0, 1, \dots, N-1\}$.
- **What the scheduler does NOT control**:
  - Physical frequency in Hz (mapped by `ScanController`).
  - Receiver bandwidth (fixed hardware parameter).
  - Dwell duration (fixed step size in simulation).
  - RF propagation, noise levels, or channel physics.

### 3.3 Action Validation Invariants

```python
validate_action(action: ScanAction, num_bins: int) -> None
```
- Ensures `isinstance(action.frequency_bin, int)`.
- Ensures bash \le 	ext{action.frequency\_bin} < N$.
- Rejects invalid bin indices or non-integer types with descriptive `ValueError` / `TypeError`.

---

## 4. Canonical Transition Schema (`Transition` & `StepResult`)

### 4.1 Transition Dataclass

```python
@dataclass(frozen=True)
class Transition:
    observation: SchedulerObservation
    action: ScanAction
    reward: float
    next_observation: SchedulerObservation
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
```

### 4.2 StepResult Dual-Interface

`env.step(action)` returns `StepResult`, which simultaneously supports:
1. **Gym / RL Unpacking**:
   ```python
   next_state, reward, done, info = env.step(action)
   ```
2. **Legacy Dictionary Access**:
   ```python
   result = env.step()
   gt = result["ground_truth"]
   obs = result["observation"]
   metrics = result["metrics"]
   ```

---

## 5. Temporal Ordering Contract

At decision step $:

29	ext{Observation}_t \longrightarrow 	ext{Action}_t \longrightarrow 	ext{Receiver Tuning} \longrightarrow \Delta t 	ext{ passes} \longrightarrow 	ext{Physics Evolve} \longrightarrow 	ext{Detector} \longrightarrow 	ext{Observation}_{t+1}, 	ext{Reward}_t, 	ext{Done}_t29

- $	ext{Action}_t$ is selected strictly from $	ext{Observation}_t$.
- $	ext{Observation}_{t+1}$ contains the detector measurement produced by executing $	ext{Action}_t$.
- The scheduler's `observe(state, action, reward, next_state, done)` hook receives the completed transition for learning.
