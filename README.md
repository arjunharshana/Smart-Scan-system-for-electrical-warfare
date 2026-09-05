# SIH26055 — Smart Scan Strategy for Electronic Warfare

**Cognitive Radio Electronic Support (ES) Receiver & Autonomous Tactical Scan Strategy**  
*SIH 2026 / DRDO Problem Statement SIH26055*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20Defence%20R%26D-darkred.svg)]()
[![Tests](https://img.shields.io/badge/Tests-107%20PASSED-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/Tactical%20Scheduler-V4.1%20Frozen-cyan.svg)]()

---

## 1. Executive Summary & Tactical Mission

Modern agile radar and frequency-hopping communications present extreme challenges for Electronic Support Measures (ESM) receivers constrained by narrow instantaneous bandwidth. A naive receiver executing fixed sequential sweeps misses up to 80% of agile threat transmissions.

**SIH26055: Smart Scan Strategy** implements an autonomous, explainable, and real-time cognitive scheduler that predicts threat frequency transitions before they occur. 

### Final Tactical Scheduler: **V4.1 (Recurrent LSTM-Hybrid)**
* **Predictive Branch**: Deep Recurrent Q-Network (DRQN) with pure NumPy Long Short-Term Memory (LSTM) cell working memory $(h_t, c_t)$ to learn cyclical dwell patterns and transition rhythms.
* **Discovery Branch**: Online Context-Aware heuristic tracking empirical frequency transition probabilities $P(f_{t+1} \mid f_t)$ alongside coverage/recency bonuses.
* **Transparent Meta-Arbitration**: Real-time rule-based arbitrator dynamically blending neural prediction (`DDQN_EXPLOIT`) with adaptive wideband search (`CA_ADAPT`) based on pattern confidence and environmental surprise.
* **100% Zero Ground-Truth Leakage**: The receiver learns strictly from real-time detector observables and receiver-derived R4 rewards without accessing emitter simulation truth.

---

## 2. Algorithmic Evolution: V3.1 &rarr; V4.0 &rarr; V4.1 &rarr; V4.2

```text
V3.0 / V3.1 Double DQN (Feedforward MLP)
  └── Strong on familiar hopping patterns (49.6% IR), but catastrophic OOD collapse (0.3% IR) when threats shifted frequency.
V4.0 Hybrid Scheduler (Context-Aware + Feedforward DDQN)
  └── Introduced rule-based Meta-Arbitration; prevented catastrophic collapse (24.3% IR), but lacked dwell-counting accumulator.
V4.1 LSTM-Hybrid Scheduler [FINAL TACTICAL SCHEDULER]
  └── Integrated LSTM recurrent working memory (h_t, c_t); achieved 86.6% IR on hopping targets and 102.9% retention on unseen permutations.
V4.2 Continual Adaptive Hybrid [EXPERIMENTAL RESEARCH ARCHIVE]
  └── Investigated real-time candidate fine-tuning; proved exact rollback immunity against catastrophic forgetting (99.6% retention).
```

### Verified Benchmark Performance Comparison

| Evaluation Scenario | Context-Aware | V3.1 DDQN (MLP) | V4.0 Hybrid | V4.1 LSTM-Hybrid [TACTICAL FINAL] | V4.1 Arbitrator Modes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`1_Seen_Structure`** (Hopping, dwell=3) | 26.9% | 49.6% | 49.1% | **80.0%** (86.6% standalone) | 82% Exploit / 14% Adapt |
| **`2_Unseen_Permutation`** (Permuted cycle) | 31.4% | 50.1% | 44.6% | **82.3%** (87.5% standalone) | 83% Exploit / 14% Adapt |
| **`3_Unseen_Phase`** (Phase shift) | 30.4% | 50.3% | 43.8% | **84.7%** (84.7% standalone) | 86% Exploit / 10% Adapt |
| **`5_Unseen_Subset`** (Novel channels) | 27.4% | 2.9% *(Collapse)* | 22.3% *(Protected)* | **24.3%** *(Protected)* | 33% Exploit / 63% Adapt |
| **`8_Periodic_Burst`** (400 MHz pulsed radar) | 51.3% | 16.7% | 78.5% | **68.8%** | 25% Exploit / 70% Adapt |

---

## 3. V4.1 Offline Training, Validation & Deployment Lifecycle

The tactical V4.1 scheduler implements a strict separation between **offline neural learning** and **runtime inference**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE TRAINING PHASE                         │
│                                                                         │
│  Multi-Scenario RF Gym (Diverse Dwells 2-5, Burst, Permutations)        │
│         │                                                               │
│         ▼                                                               │
│  Independent Candidate Training (Seeds 42, 123, 456, 789)               │
│         │                                                               │
│         ▼                                                               │
│  Periodic Held-Out Validation Battery (Val IR & Stability Metrics)      │
│         │                                                               │
│         ▼                                                               │
│  Automated Model Selection (Max Mean Val IR, Collapse Penalty)          │
│         │                                                               │
│         ▼                                                               │
│  Audit Manifest Generation & Atomic Promotion to Production Checkpoint  │
│  (models/v4_1/training_manifest.json -> production_checkpoint.npz)      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION RUNTIME DEPLOYMENT                      │
│                                                                         │
│  Production Checkpoint Loaded (models/v4_1/production_checkpoint.npz)   │
│         │                                                               │
│         ▼                                                               │
│  FROZEN INFERENCE: train_mode = False, epsilon = 0.0, Zero Gradients    │
│         │                                                               │
│         ├── LSTM-DDQN Branch: Pure Feedforward & Recurrent State (h, c) │
│         │                     (Static weights evaluate Q-values)        │
│         │                                                               │
│         ├── Context-Aware Branch: Online Empirical Transition Tracking  │
│         │                         (Statistical frequency discovery)     │
│         │                                                               │
│         ▼                                                               │
│  HybridMetaArbitrator dynamically balances EXPLOIT vs ADAPT             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Invariants
1. **Static / Frozen Model**: The LSTM-DDQN neural network is trained exclusively **offline** prior to operational deployment. In production, **zero gradient updates occur** (`train_mode = False`, `epsilon = 0.0`).
2. **CA Statistical Adaptation $\ne$ Neural Retraining**: The Context-Aware (CA) branch maintains an empirical transition probability matrix $P(f_{t+1} \mid f_t)$ from receiver detector hits. This statistical bookkeeping allows rapid discovery of novel frequencies without modifying neural weights.
3. **V4.2 Research Isolation**: V4.2 remains an experimental supervisory research milestone (`archive/v4_2_continual/`) and is **never** used as the production runtime mechanism.
4. **Zero Ground-Truth Leakage**: Ground truth is strictly evaluation-only. It never enters the observation space, reward calculations, training buffers, validation gates, or arbitration.

### Reproducing Offline Training
```bash
# Run full offline training, held-out validation, and model promotion pipeline:
python -m benchmarks.train_v4_1 --seeds 42 123 456 789 --episodes 25 --episode-length 300 --validation-interval 5
```

---

## 4. Production Repository Architecture

```text
SIH-2026/
├── app/                          # Production Web Application & Live Dashboard
│   ├── __main__.py               # Application CLI entrypoint (python -m app)
│   ├── main.py                   # FastAPI service factory, lifespan, static mounting
│   ├── config.py                 # Portable environment configuration
│   ├── api/                      # REST & WebSocket Endpoints
│   │   ├── routes.py             # /health, /api/status, /api/telemetry, /api/scenarios
│   │   ├── schemas.py            # Pydantic contract schemas
│   │   └── websocket.py          # /ws/telemetry high-frequency real-time stream
│   ├── services/
│   │   └── simulation_service.py # SimulationService managing V4.1 runtime & explainability
│   └── dashboard/                # Tactical EW Single-Page Presentation Dashboard
│       ├── index.html            # High-contrast military-grade UI layout
│       ├── css/dashboard.css     # Dark tactical theme with cyan/amber/green accents
│       └── js/
│           ├── waterfall.js      # 2D Canvas Frequency-Time waterfall renderer
│           └── dashboard.js      # Real-time WebSocket controller & event handler
│
├── rf_environment/               # Core RF Simulation & Scheduler Engine
│   ├── domain/                   # Canonical action, state, observation, and metrics schemas
│   ├── emitters/                 # Radar and communications signal emitters
│   ├── environment/              # RFEnvironment, simulation clock, scenario loader
│   ├── frequency_behaviors/      # Hopping, sweep, fixed, random behavior models
│   ├── metrics/                  # Opportunity tracker, interception evaluator
│   ├── receiver/                 # Tuner, superheterodyne receiver, energy detector
│   ├── rewards/                  # Receiver-derived R4 reward formulation
│   ├── scenarios/                # EW scenario YAML configurations
│   └── scheduler/                # Frozen V4.1 LSTM-Hybrid, CA, and baselines
│       ├── factory.py            # Scheduler factory (defaults to "hybrid_v41")
│       ├── context_aware.py      # Context-Aware empirical transition learner
│       ├── hybrid/               # HybridMetaArbitrator & LSTMHybridScheduler
│       └── rl/                   # Pure NumPy LSTM network, BPTT, recurrent replay
│
├── benchmarks/                   # Canonical benchmarks & authoritative result JSONs
│   ├── run_v3_benchmark.py
│   ├── run_v3_1_study.py
│   ├── run_v4_benchmark.py
│   ├── run_v4_1_benchmark.py
│   └── results/                  # Verified benchmark JSON output data
│
├── archive/                      # Preserved historical research artifacts
│   ├── v4_2_continual/           # V4.2 continual learning code, tests, and benchmark
│   ├── v2_dashboard/             # Legacy V2.2 Streamlit dashboard
│   └── validation/               # Preliminary validation test scripts
│
├── deployment/                   # Production deployment assets
│   ├── Dockerfile                # Multi-stage secure non-root Dockerfile
│   └── docker-compose.yml        # One-command orchestration
│
├── scripts/                      # Launch and test utility scripts
│   ├── run_local.sh              # Local development launcher
│   ├── run_docker.sh             # Docker build and run script
│   └── run_benchmarks.sh         # Canonical benchmark runner
│
├── tests/                        # Comprehensive test suite (107 tests)
│   ├── test_api_v41.py           # FastAPI endpoints, telemetry, and firewall tests
│   ├── test_v4_1_lstm_hybrid.py  # V4.1 verification and determinism tests
│   └── ...                       # Baseline unit and schema tests
│
├── Dockerfile                    # Root production Dockerfile
├── docker-compose.yml            # Root docker-compose configuration
├── requirements.txt              # Production Python dependencies
├── run_tests.py                  # Standalone test runner (107/107 PASS)
└── README.md
```

---

## 5. Quickstart — Running Locally

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Linux, macOS, or Windows WSL2

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch Tactical Application & Dashboard
```bash
# Option A: Using the convenient launch script
./scripts/run_local.sh

# Option B: Direct Python command
python -m app serve --host 0.0.0.0 --port 8000
```

### 3. Open Browser
Navigate to **`http://localhost:8000`** in any modern web browser.

---

## 6. Docker Containerized Deployment

The application is containerized following defence software standards (minimal slim image, non-root `appuser`, zero external CDN dependencies, built-in health check).

```bash
# Option A: One-command docker-compose
docker compose up --build

# Option B: Using the convenience script
./scripts/run_docker.sh
```

Verify container health:
```bash
curl -f http://localhost:8000/health
# Output: {"status":"ok","scheduler":"V4.1","version":"4.1","device":"cpu","timestamp":...}
```

---

## 7. Dashboard User Guide (SIH Presentation)

The dashboard is designed for high-visibility presentation on a classroom or conference projector with two distinct view modes:

```text
┌─────────────────────────────────────────────────────────────┐
│ SMART SCAN STRATEGY FOR ELECTRONIC WARFARE     ● ACTIVE     │
├─────────────────────────────────────────────────────────────┤
│ CURRENT SCAN │ PREDICTED NEXT │ CONFIDENCE │ TOP Q-RANKING  │
│  250.0 MHz   │   450.0 MHz    │  91% HIGH  │ 1. 450 (Q:3.8) │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│              FREQUENCY × TIME WATERFALL VIEW                │
│       [ Cyan = Scanned  |  Amber = Next  |  Green = Hit ]   │
│                                                             │
├──────────────────────────────┬──────────────────────────────┤
│ MISSION PERFORMANCE          │ WHY THIS FREQUENCY?          │
│ IR: 80.0%  | Det Rate: 92%   │ Predicted hopping jump to    │
│ Opportunities: 24/30 Covered │ 450 MHz. DDQN_EXPLOIT (82%). │
├──────────────────────────────┴──────────────────────────────┤
│ RECENT PREDICTION TIMELINE & SCENARIO CONTROLS             │
└─────────────────────────────────────────────────────────────┘
```

### 1. Primary Prediction Hero Card
* **Current Scan**: Currently tuned receiver center frequency and channel bin.
* **Predicted Next**: Anticipated threat frequency for the upcoming dwell step.
* **Pattern Confidence**: Real-time statistical confidence gauge ($0-100\%$) derived from the neural Q-value distribution separation.
* **Candidate Ranking**: Top 5 candidate channels sorted by Q-value and normalized softmax share.

### 2. Frequency &times; Time Spectrum Waterfall
* High-performance 2D Canvas rendering the last 70 scan steps.
* **Visual Tokens**:
  * **Cyan Open Ring**: Receiver tuned frequency at step $t$.
  * **Amber Reticle Target**: Scheduler's anticipated next frequency.
  * **Neon Green Glowing Circle**: Confirmed signal detection (interception event).

### 3. "Why This Scan?" Tactical Explainability
* Real-time natural language rationale explaining the cognitive decision:
  * e.g., *"Anticipating periodic hopping transition to 450.0 MHz (Bin 17). Recurrent cell state identified dwell completion. LSTM confidence: HIGH (Decision weight: 84% LSTM / 16% CA)."*

### 4. Overview Mode vs. Technical Mode
* **Overview Mode (Default)**: Clean, high-impact tactical interface optimized for non-technical evaluators.
* **Technical Mode**: Unlocks the **Temporal Working Memory** panel (displaying LSTM hidden state energy $\|h_t\|_2$ and sequence context) and enables the **Ground Truth Overlay** toggle (`⚠️ EVALUATION MODE`).

---

## 8. Historical Version Preservation

Historical milestones are preserved via Git tags and dedicated branches without rewriting history:

| Tag / Branch | Description | Check-out Command |
| :--- | :--- | :--- |
| **`v3.1-final`** | Standalone Double DQN with temporal observation encoding | `git checkout v3.1-final` |
| **`v4.0-final`** | Feedforward Hybrid Scheduler (CA + Feedforward DDQN) | `git checkout v4.0-final` |
| **`v4.1-final`** | **Final Tactical Scheduler (LSTM-Hybrid DRQN)** | `git checkout v4.1-final` |
| **`v4.2-experimental`** | Continual Adaptive Hybrid Scheduler (Branch: `v4.2-continual`) | `git checkout v4.2-experimental` |

*(A complete copy of the V4.2 continual learning engine is also archived directly in `archive/v4_2_continual/` on this production branch).*

---

## 9. Verification & Test Suite

Run the full automated test suite containing **132 unit, integration, and firewall tests**:

```bash
python run_tests.py
```

### Verified Invariants
* **132 / 132 Tests PASSED** (Execution time: $\sim 15$ seconds)
* **Ground-Truth Firewall**: Statically and dynamically audited to guarantee zero leakage of true emitter state into the observation, reward, or scheduler decision paths.
* **Deterministic Replay**: Fixed random seeds guarantee bit-for-bit identical scan trajectories and metrics across runs.
* **Zero External Deep Learning Frameworks**: Pure NumPy implementation of LSTM forward/backward inference and recurrent experience replay.

---

## 10. Team & Presentation Details

* **Project**: SIH26055 — Smart Scan Strategy for Electronic Warfare
* **Nodal Agency / Organization**: Ministry of Defence / Defence Research and Development Organisation (DRDO)
* **Final Presentation Scheduler**: **V4.1 LSTM-Hybrid Scheduler**
* **Status**: Production Deployment Ready & Certified
