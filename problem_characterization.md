# Mathematical Problem Characterization: Smart Scan Strategy for Electronic Warfare

**Project**: SIH26055 / DRDO Smart Scan Strategy  
**Study Scope**: Rigorous Codebase Audit, Mathematical Formulation, and Generative Model Identification  
**Target Repository**: `Rishant-gupta/Smart-Scan-system-for-electrical-warfare`  
**Date**: September 2026  

---

## Executive Summary: What Is Our Problem Actually?

From a rigorous mathematical and software audit of the simulator codebase, our problem is **NOT** an uncoupled Restless Multi-Armed Bandit (RMAB), nor is it a classical Markov Decision Process (MDP).

> **Core Mathematical Definition**:  
> The smart scan surveillance problem is a **Partially Observable Discrete-Time Semi-Markov Process (PODTSMP)** with **severe action-dependent observation censoring** and **strong cross-channel informational coupling induced by a single agile hidden emitter**.

### Key Architectural & Mathematical Realities

1. **State Is Hidden & Multi-Dimensional**:  
   The true physical state $S_t = (F_t, \tau_t, \text{tx}_t, \text{Regime}_t)$ comprises the active emitter frequency $F_t$, the elapsed dwell counter $\tau_t$, the burst gating state $\text{tx}_t$, and the temporal regime.
2. **Observations Are Censored & Partial**:  
   The receiver operates a single instantaneous $20\text{ MHz}$ tuner across a $600\text{ MHz}$ spectrum ($N=30$ discrete bins). At step $t$, the receiver senses exactly **1 bin** ($3.3\%$ of the spectrum) and receives **zero direct information** from the other 29 bins ($96.7\%$ censoring).
3. **Dynamics Are Semi-Markovian, Not Markovian**:  
   Because dwell time $D \ge 1$ holds the frequency constant for $D$ consecutive steps, the transition probability $P(F_{t+1} \mid F_t)$ is non-stationary and non-Markovian in frequency alone. Hopping requires knowing the elapsed dwell time $\tau_t \in \{0, \dots, D-1\}$.
4. **Frequency Bins Are Strictly Coupled**:  
   There is only one emitter active at any time. The activity of bin $j$ and bin $k$ are mutually exclusive ($\sum_i \mathbf{1}_{\{F_t = f_i\}} \le 1$). Detecting an emitter in bin $j$ immediately collapses the probability of all other 29 bins to zero (up to detector false alarm rate $P_{FA}$). Channels do **not** evolve as independent arms.
5. **Current Canonical Benchmark Measures Memorization Over Generalization**:  
   Canonical scenarios use static 3-frequency cyclic loops with fixed dwell times and zero within-episode regime shifts. Schedulers that memorize deterministic sequence indices appear artificially superior to general adaptive policies.

---

## 1. Full Codebase Audit & Implementation Mapping

This audit traces the actual Python code paths, verifying exact mechanisms and file locations.

| Component / Subsystem | Implementation File | Key Class / Function | Underlying Mathematical / Operational Mechanism |
| :--- | :--- | :--- | :--- |
| **Frequency Hopping Generator** | [`rf_environment/frequency_behaviors/hopping.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py#L17-L55) | `FrequencyHopping.get_frequency` | `hop_index = time_step // self.dwell_steps`. Computes sequential index `sequence[hop_index % len(sequence)]` or random draw `rng.integers(0, len(frequencies_hz))` cached per hop. |
| **Dwell-Time Engine** | [`rf_environment/frequency_behaviors/hopping.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py#L45-L46) | Integer division `// dwell_steps` | **Strictly constant integer dwell**. Dwell is fixed for an emitter (`dwell_steps` $\in \{1, 2, 3, 4, 5\}$). Elapsed dwell step is implicitly `time_step % dwell_steps`. |
| **Periodic Burst Generator** | [`rf_environment/time_behaviors/burst.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/time_behaviors/burst.py#L7-L24) | `Burst.is_transmitting` | Deterministic modulo gating: `offset = (time_step - start_time) % interval`. Transmitting if `offset < burst_duration`. |
| **Continuous Emitter Generator** | [`rf_environment/time_behaviors/continuous.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/time_behaviors/continuous.py#L7-L12) | `Continuous.is_transmitting` | Trivially returns `True` for all $t$. |
| **Radar Emitter Base** | [`rf_environment/emitters/base.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/emitters/base.py#L33-L55) | `BaseEmitter.step` | Evaluates transmission gating and frequency; constructs ground-truth `EmitterState(frequency_hz, transmitting, dwell_steps, hop_index)`. |
| **Receiver Front-End** | [`rf_environment/receiver/receiver.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/receiver/receiver.py#L47-L85) | `Receiver.observe` | Checks `bands_overlap(center, bw, signal.frequency_hz, signal.bandwidth_hz)`. Filters by sensitivity ($-90\text{ dBm}$) and computes received signal power and SNR. |
| **Imperfect Energy Detector** | [`rf_environment/receiver/detector.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/receiver/detector.py#L27-L56) | `Detector.detect` | If signal present in band: detection drawn with $P_D = 0.95$. If no signal in band: false alarm drawn with $P_{FA} = 0.02$. |
| **Observation Builder** | [`rf_environment/environment/observation_builder.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/environment/observation_builder.py#L62-L108) | `ObservationBuilder.step` | Constructs immutable, frozen `SchedulerObservation` containing only legitimate receiver observables (scanned bin, detection bool, signal strength, scan counts, timers). |
| **Reward Engine (R4)** | [`rf_environment/rewards/r4_reward.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/rewards/r4_reward.py#L10-L75) | `R4RewardCalculator.compute` | $R_4 = r_{\text{det}} (+1.0) + r_{\text{step}} (-0.05) + r_{\text{strength}} (\in [0, 0.15])$. Depends strictly on receiver-visible detections. |
| **Context-Aware Scheduler** | [`rf_environment/scheduler/context_aware.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/scheduler/context_aware.py#L69-L140) | `ContextAwareScheduler.compute_action_scores` | Combines 1st-order empirical transition probability $P(j \mid \text{last\_det})$, recent activity rate, and coverage uncertainty $\min(1, \tau_j / K_{\text{stale}})$. |
| **V4.0 Feed-Forward DDQN** | [`rf_environment/scheduler/rl/ddqn_scheduler.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/scheduler/rl/ddqn_scheduler.py#L14-L120) | `DDQNScheduler` + `MLPQNetwork` | Feed-forward MLP consuming encoded observation vector. No internal recurrent hidden state; relies on sliding history window. |
| **V4.1 LSTM-DDQN** | [`rf_environment/scheduler/rl/lstm_ddqn_scheduler.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/scheduler/rl/lstm_ddqn_scheduler.py#L15-L160) | `LSTMDDQNScheduler` | Pure-NumPy recurrent working memory maintaining hidden state $(h_t, c_t)$ across steps. |
| **Whittle-Style Scheduler** | [`rf_environment/scheduler/whittle/scheduler.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/scheduler/whittle/scheduler.py#L15-L95) | `WhittleScheduler.select_bin` | Computes per-arm heuristic index balancing Bayesian belief, empirical transition likelihood, uncertainty aging, dwell persistence, and periodicity. |
| **Ground-Truth Storage** | [`rf_environment/domain/ground_truth.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/domain/ground_truth.py) | `GroundTruthStore` | Completely isolated from scheduler runtime; used exclusively by `OpportunityTracker` for post-hoc metric scoring. |
| **Opportunity Tracker & IR** | [`rf_environment/metrics/opportunity_tracker.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/metrics/opportunity_tracker.py#L25-L90) | `OpportunityTracker.step` | Measures true transmission opportunities and true detections. Interception ratio $\text{IR} = \frac{\text{hits}}{\text{opportunities}}$. |
| **Scenario Catalog** | [`benchmarks/scenarios_v4_1.py`](file:///home/rishant-gupta/Projects/SIH-2026/benchmarks/scenarios_v4_1.py#L54-L305) | `get_canonical_test_scenarios`, etc. | Defines 8 canonical test scenarios, 6 training scenarios, and 4 held-out validation scenarios. |

---

## 2. Mathematical Reconstruction of the Environment

We formalize the execution flow of the simulator at discrete time steps $t \in \{0, 1, 2, \dots, T-1\}$:

```
[Hidden Environment State S_t]
  ├── Emitter Carrier Frequency F_t ∈ {f_0, ..., f_{N-1}}
  ├── Elapsed Dwell Step τ_t ∈ {0, ..., D-1}
  ├── Transmission Gate tx_t ∈ {0, 1}
  └── Threat Regime / Hopping Law θ_t
           │
           ▼
[State Transition P(S_{t+1} | S_t)]
  ├── If τ_t < D - 1: τ_{t+1} = τ_t + 1, F_{t+1} = F_t
  └── If τ_t = D - 1: τ_{t+1} = 0, F_{t+1} ~ T(F_t → · | θ_t)
           │
           ▼ (Concurrent Interaction)
[Receiver Action A_t ∈ {0, ..., N-1}]
           │
           ▼
[Imperfect Observation O_t]
  ├── In-Band Indicator: I_t = 1_{F_t ∈ Bin(A_t)} ∧ tx_t
  └── Detection Event Y_t ~ Bernoulli(P_D · I_t + P_{FA} · (1 - I_t))
           │
           ▼
[Scheduler Internal State / History H_t = (A_0, Y_0, ..., A_t, Y_t)]
           │
           ▼
[Scheduler Policy π(A_{t+1} | H_t)]
```

### Formal Mathematical Specification

1. **Hidden State ($S_t \in \mathcal{S}$)**:
   $$S_t = \left( F_t, \tau_t, \text{tx}_t, \theta_t \right)$$
   - $F_t \in \{f_0, \dots, f_{N-1}\}$: True emitter frequency center.
   - $\tau_t \in \{0, 1, \dots, D - 1\}$: Elapsed dwell step on current channel.
   - $\text{tx}_t \in \{0, 1\}$: Emitter transmission gating state.
   - $\theta_t = (\mathcal{F}_{\text{hop}}, D, \text{Mode}, I_{\text{burst}}, B_{\text{burst}})$: Temporal regime parameter vector.

2. **State Transition Kernel ($P(S_{t+1} \mid S_t)$)**:
   The transition does **not** depend on receiver action $A_t$ (passive scanning assumption):
   $$P(S_{t+1} \mid S_t, A_t) = P(S_{t+1} \mid S_t)$$
   Assuming a stationary within-episode regime $\theta_t = \theta$:
   - **Dwell Counter Transition**:
     $$\tau_{t+1} = (\tau_t + 1) \pmod D$$
   - **Frequency Transition**:
     $$F_{t+1} = \begin{cases}
     F_t & \text{if } \tau_t < D - 1 \\
     \text{NextHop}(F_t, \theta) & \text{if } \tau_t = D - 1
     \end{cases}$$
   - **Transmission Gating Transition** (for Periodic Burst):
     $$\text{tx}_{t+1} = \mathbf{1}_{\{(t+1 - t_{\text{start}}) \pmod I < B\}}$$

3. **Receiver Action Space ($\mathcal{A}$)**:
   $$\mathcal{A} = \{0, 1, \dots, N-1\}, \quad N = 30$$
   The action selects the instantaneous receiver center frequency: $f_{\text{rx}}(t) = 100\text{ MHz} + 10\text{ MHz} + A_t \times 20\text{ MHz}$.

4. **Observation Emission Probability ($P(O_t \mid S_t, A_t)$)**:
   Let $I_t = \mathbf{1}_{\{F_t \in \text{bin}(A_t)\}} \cdot \text{tx}_t$.
   $$P(Y_t = 1 \mid S_t, A_t) = P_D \cdot I_t + P_{FA} \cdot (1 - I_t)$$
   In our environment, $P_D = 0.95$ and $P_{FA} = 0.02$.
   - **Hit (Interception)**: $I_t = 1$ and $Y_t = 1$ (prob $0.95$).
   - **Miss**: $I_t = 1$ and $Y_t = 0$ (prob $0.05$).
   - **False Alarm**: $I_t = 0$ and $Y_t = 1$ (prob $0.02$).
   - **Correct Rejection**: $I_t = 0$ and $Y_t = 0$ (prob $0.98$).

---

## 3. Rigorous Answers to the 16 System Identification Questions

Derived directly from the codebase:

### Q1: What is the hidden state?
The hidden state is the vector $S_t = (F_t, \tau_t, \text{tx}_t, \theta)$ containing:
1. True carrier frequency $F_t \in \mathbb{R}^+$.
2. Elapsed dwell counter $\tau_t \in \{0, \dots, D-1\}$.
3. Transmission gating status $\text{tx}_t \in \{0, 1\}$.
4. Hopping sequence parameters $\theta = (\mathcal{F}_{\text{sequence}}, D, \text{mode})$.

### Q2: What is observable?
The scheduler strictly observes:
- Current tuned bin $A_t$.
- Binary detection outcome $Y_t \in \{0, 1\}$.
- Received signal power $\text{RSSI}_t$ (when $Y_t = 1$).
- Timestamp $t$.
- Derived receiver timers: $\tau_i^{\text{scan}} = t - t_{\text{last\_scan}}(i)$, $\tau^{\text{detect}} = t - t_{\text{last\_detection}}$, and scan counters.

### Q3: What is unobservable?
1. True emitter carrier frequency $F_t$ when $A_t \ne \text{bin}(F_t)$.
2. True emitter carrier frequency when $A_t = \text{bin}(F_t)$ but detector misses ($Y_t = 0$).
3. Elapsed dwell duration $\tau_t$.
4. True dwell limit $D$.
5. Transmission state $\text{tx}_t$ when off-channel.
6. The identity and hopping pattern of the scenario.

### Q4: Is the process deterministic?
**Mixed (piecewise deterministic with stochastic observation noise):**
- In 7 of the 8 canonical scenarios, the emitter's physical trajectory $F_t$ is **strictly deterministic** (a periodic cycle or burst).
- In Scenario 7 (`7_Random_Hopping`), the hop sequence is **pseudo-random**.
- In **ALL** scenarios, the observation process is **stochastic** due to detector noise ($P_D = 0.95, P_{FA} = 0.02$).

### Q5: Is it stochastic?
Yes. Even when the emitter trajectory is deterministic, partial observability and detector flips ($P_D=0.95, P_{FA}=0.02$) induce a stochastic POMDP observation space.

### Q6: Is it Markov?
**NO.** The frequency trajectory $F_{t+1}$ is **NOT Markovian in frequency alone**.
$$P(F_{t+1} \mid F_t) \ne P(F_{t+1} \mid F_t, F_{t-1}, \dots)$$
Because dwell duration $D \ge 2$ holds $F_{t+1} = F_t$ until elapsed dwell reaches $D-1$, transition probabilities change as a function of elapsed time on the channel.

### Q7: If not Markov, what history length is required?
To make the system Markovian without a continuous latent belief estimator:
- The scheduler must know the elapsed dwell counter $\tau_t$.
- In the worst-case canonical scenario (`6_Mixed_Shift`, $D=5$), at least **$5$ steps of contiguous channel observation** are required to identify dwell phase.
- For periodic bursts (`8_Periodic_Burst`, $I=15$), a history window of at least **$15$ steps** is required to observe full cycle periodicity.

### Q8: Is it periodic?
**YES.** All non-random canonical scenarios exhibit exact periodicity:
- `1_Seen_Structure`: Period $T = 3 \text{ hops} \times 3 \text{ dwell} = 9\text{ steps}$.
- `2_Unseen_Permutation`: Period $T = 3 \times 3 = 9\text{ steps}$.
- `3_Unseen_Phase`: Period $T = 3 \times 3 = 9\text{ steps}$.
- `4_Unseen_Dwell`: Period $T = 3 \times 1 = 3\text{ steps}$.
- `5_Unseen_Subset`: Period $T = 3 \times 3 = 9\text{ steps}$.
- `6_Mixed_Shift`: Period $T = 3 \times 5 = 15\text{ steps}$.
- `8_Periodic_Burst`: Period $T = 15\text{ steps}$.

### Q9: Is it semi-Markov?
**YES.** By standard stochastic process definitions, the emitter state transitions to a new frequency only after a sojourn/dwell time $D$. It is a discrete-time semi-Markov chain (or Markov renewal process) with deterministic dwell distributions.

### Q10: Does dwell time form part of the state?
**YES, fundamentally.** Any state representation omitting elapsed dwell $\tau_t$ cannot distinguish whether the emitter is about to hop or will remain on the channel for additional steps.

### Q11: Do different frequency bins evolve independently?
**NO.** Frequency bins do **NOT** evolve independently.

### Q12: Are frequency bins coupled through a common hidden emitter state?
**YES.** A single physical emitter generates the entire RF environment. The channel occupancy vector $X_t \in \{0, 1\}^N$ satisfies the conservation constraint:
$$\sum_{i=0}^{N-1} X_t^{(i)} \le 1$$
Occupancy of bin $j$ implies non-occupancy of all $k \ne j$.

### Q13: Does observing one frequency provide information about another?
**YES, profoundly.**
- Positive information: Detecting frequency $f_A$ informs the receiver that frequencies $f_B, f_C$ are currently inactive, and (via empirical transition matrix $\hat{T}$) predicts that $f_B$ is likely to follow.
- Negative information: Scanning frequency $f_A$ and observing a miss increases the posterior probability that the emitter resides in one of the other candidate bins.

### Q14: Does the process have regime changes?
**Across scenarios: YES. Within an episode: NO.**
- Across training and validation runs, the simulator switches between 6 different regimes (dwell $2, 3, 4, 5$, bursts, and agile hops).
- Within each 300-step evaluation episode, the regime $\theta$ remains completely stationary.

### Q15: Are regimes stationary within an episode?
**YES.** In the current canonical benchmark, the emitter parameters ($\mathcal{F}, D, \text{mode}$) never mutate midway through an episode.

### Q16: Can the scheduler infer the regime from observations?
**YES, with sufficient probing.**
- By observing 1–2 complete hopping cycles ($10-25$ steps), an adaptive estimator can identify:
  1. The active frequency subset $\mathcal{F}_{\text{active}}$.
  2. The dwell duration $D$ (from consecutive detection streaks).
  3. The transition matrix $\hat{T}(j \to i)$.

---

## 4. True Generative Model Classification

Evaluating the mathematical candidates:

| Candidate Model | Mathematical Definition | Matches Simulator? | Justification |
| :--- | :--- | :---: | :--- |
| **A. Deterministic Sequence** | $F_{t+1} = g(F_t, t)$ | **Partially** | 7 of 8 scenarios follow deterministic hopping, but observation noise ($P_D, P_{FA}$) prevents deterministic inversion. |
| **B. 1st-Order Markov Chain** | $P(F_{t+1} \mid F_t)$ | **NO** | Fails because dwell $D > 1$ creates duration memory. $F_{t+1}$ depends on elapsed dwell $\tau_t$. |
| **C. Higher-Order Markov** | $P(F_{t+1} \mid F_t, F_{t-1}, \dots)$ | **Approximation** | An order-$D$ Markov chain can represent fixed dwell, but requires exponential state expansion as $D$ grows. |
| **D. Semi-Markov Process** | $(F_k, D_k)$ joint kernel | **YES (Exact)** | States transition according to $T(j \to i)$ after a dwell duration $D_j$. Matches the codebase implementation perfectly. |
| **E. Periodic Process** | $F_{t+T} = F_t$ | **YES (Subset)** | Exact for scenarios 1, 2, 3, 4, 5, 6, and 8; fails for Scenario 7 (Random). |
| **F. Switching / Regime Process** | $S_t \sim P(S_t \mid \theta_t)$ | **YES (Multi-Scenario)** | Describes the cross-scenario diversity across training and test suites. |
| **G. Partially Observable (POMDP)** | Observation censoring via $A_t$ | **YES (Fundamental)** | The receiver only samples 1 of 30 channels, inducing an information state / belief state over unobserved arms. |

### Final Classification
> **The true generative model of the simulator is:**  
> **A Partially Observable Discrete-Time Hidden Semi-Markov Process (PO-HSMM) with Stationary Within-Episode Dynamics and Action-Dependent Observation Censoring.**

---

## 5. Dwell-Time Model Analysis

In [`rf_environment/frequency_behaviors/hopping.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py#L45):
```python
hop_index = time_step // self.dwell_steps
```

### Analytical Consequences
1. **Dwell is Strictly Deterministic and Clock-Synchronous**:  
   The emitter remains on its assigned channel for exactly `dwell_steps` clock cycles. It does not jitter, drift, or follow a geometric or Poisson distribution.
2. **Transition Probability as a Function of Elapsed Dwell**:
   $$P(F_{t+1} = F_t \mid F_t, \tau_t) = \begin{cases} 1.0 & \text{if } \tau_t < D - 1 \\ 0.0 & \text{if } \tau_t = D - 1 \end{cases}$$
   $$P(F_{t+1} \ne F_t \mid F_t, \tau_t) = \begin{cases} 0.0 & \text{if } \tau_t < D - 1 \\ 1.0 & \text{if } \tau_t = D - 1 \end{cases}$$
3. **Why 1st-Order Markov Models (like Context-Aware) Fail on Dwell**:  
   A 1st-order Markov model only estimates $P(\text{next} \mid \text{current})$. It calculates a static self-transition probability:
   $$P(\text{stay}) = \frac{D - 1}{D}, \quad P(\text{hop}) = \frac{1}{D}$$
   On every step, it applies this same static probability, exhibiting memoryless geometric decay. It cannot know whether the emitter has stayed for 1 step or is on its final step before a hop.
4. **Why Whittle W3 Degraded on Dwell Changes**:  
   In Whittle W3, the dwell persistence term was parameterized with default dwell $\hat{d} = 3$:
   $$V_{\text{dwell}}(i) = \max\left(0, 1.0 - \frac{k_i}{3}\right)$$
   When evaluated on Scenario 4 ($D=1$), the scheduler stayed on the channel expecting 2 more hits, achieving only **8.43% IR**. When evaluated on Scenario 6 ($D=5$), the persistence reward expired after 3 hits, causing premature abandonment.

---

## 6. Frequency Coupling & The Whittle/RMAB Assumption Check

### The Arm Independence Assumption of RMAB
In Whittle's index theory (Whittle 1988) and dynamic spectrum access restless bandit models (Liu & Zhao 2010):
$$\mathcal{P}(S_{t+1} \mid S_t) = \prod_{i=1}^N P_i(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)})$$
This requires:
1. Each channel $i$ evolves according to its own independent Markov chain (e.g. independent primary users).
2. The state of channel $i$ conveys zero information about the state of channel $j$.

### Simulator Reality: Total Informational Coupling
In our environment:
1. **Physical Conservation**: Only 1 emitter exists. If bin $j$ is transmitting, bins $k \ne j$ are idle.
2. **Predictive Coupling**: The emitter hops between channels according to a transition sequence $f_{(1)} \to f_{(2)} \to f_{(3)}$. Bin $k$'s future state is directly determined by bin $j$'s current state.
3. **Observation Coupling**: Scanning bin $j$ provides information about **all** bins. A hit in $j$ proves non-occupancy in all $k \ne j$. A miss in $j$ redistributes probability mass across the remaining channels.

### Conclusion on Whittle Formulations
> **LITERATURE FACT**: Restless bandit indexability theorems (Whittle 1988, Weber & Weiss 1990, Liu & Zhao 2010) require arm independence.  
> **OUR SIMULATOR FACT**: Frequency channels in our simulator are coupled through a single underlying hopping emitter.  
> **INFERENCE**: The mathematical preconditions for Whittle indexability are violated.  
> **RESEARCH RECOMMENDATION**: Our scheduler must be designated strictly as a **"Whittle-Style Heuristic Index Scheduler"**. Any claim of theoretical indexability or asymptotic optimality is mathematically false for this problem formulation.

---

## 7. Minimal Sufficient State Representation

A state representation $Z_t$ is **sufficient** if it forms a Markov state for decision-making:
$$P(S_{t+1}, R_{t+1} \mid Z_t, A_t, H_t) = P(S_{t+1}, R_{t+1} \mid Z_t, A_t)$$

### Analytical Derivation of Minimal Sufficient Statistic
For our PO-HSMM environment, the minimal sufficient statistic $Z_t$ consists of:

$$Z_t = \left( \mathbf{b}_t, \hat{\tau}_t, \hat{D}, \mathbf{\hat{T}}, \hat{\Delta}_{\text{burst}} \right)$$

Where:
1. **Belief Vector $\mathbf{b}_t \in \Delta^{N-1}$**:  
   Posterior distribution over current emitter location: $b_i(t) = P(F_t \in \text{bin } i \mid H_t)$.
2. **Elapsed Dwell Estimate $\hat{\tau}_t \in \mathbb{N}$**:  
   Consecutive steps the emitter is believed to have occupied the currently active channel.
3. **Estimated Dwell Duration $\hat{D} \in \mathbb{N}$**:  
   Estimated dwell capacity of the active regime.
4. **Transition Kernel Matrix $\mathbf{\hat{T}} \in \mathbb{R}^{N \times N}$**:  
   Directed hopping transition probabilities between frequency bins.
5. **Periodic Burst Interval $\hat{\Delta}_{\text{burst}} \in \mathbb{R}^+$**:  
   Recurrence interval for gated pulse emitters.

### Dimension Comparison

| Representation | Dimensionality ($N=30$) | Information Capacity | Captures Dwell? | Captures Periodicity? |
| :--- | :---: | :---: | :---: | :---: |
| **Context-Aware State** | $30 \times 30 + 30$ counts | Empirical 1st-order Markov | ✗ | ✗ |
| **V4.0 Encoded Observation** | 146 scalar features | Fixed 10-step history window | Partial | Only fixed Fourier periods |
| **V4.1 LSTM Hidden State** | $h_t \in \mathbb{R}^{64}, c_t \in \mathbb{R}^{64}$ | Unconstrained continuous latent memory | Capable, but prone to interference | Capable, but prone to phase lock |
| **Minimal Sufficient State $Z_t$** | $30 + 1 + 1 + 30 \times 30 + 1 = 933$ parameters | Compact analytical sufficient statistic | **✓ (Exact)** | **✓ (Exact)** |

---

## 8. Benchmark Difficulty & Simplification Audit

An audit of the 8 canonical scenarios reveals substantial simplifications that make the current benchmark unrepresentative of real-world EW environments:

| Real-World EW Dimension | Simulator Canonical Benchmark | Simplification Severity | Risk to Algorithmic Validity |
| :--- | :--- | :---: | :--- |
| **Hopping Alphabet Size** | Exactly 3 bins active out of 30 ($10\%$ spectrum usage). | **High** | Schedulers can easily ignore $90\%$ of the spectrum once the 3 channels are found. |
| **Hopping Law** | Deterministic circular permutation in 7 of 8 scenarios ($5 \to 15 \to 25$). | **Extreme** | Rewards rigid sequence memorization over dynamic search. |
| **Dwell Duration** | Fixed integer ($D \in \{1, 3, 5\}$). Zero jitter or variance. | **High** | Eliminates stochastic dwell timing; models never encounter timing jitter. |
| **Number of Emitters** | Exactly 1 emitter in all canonical scenarios. | **Extreme** | Eliminates pulse interleaving, co-channel interference, and multi-emitter deinterleaving. |
| **SNR & Path Loss** | Fixed power ($-18\text{ dBm}$), SNR $> 80\text{ dB}$. Free space, no fading. | **High** | Detector almost never drops below threshold due to propagation loss. |
| **Regime Non-Stationarity** | Completely stationary within an episode (300 steps). | **High** | Evaluates only zero-shot transfer; never tests online regime tracking or adaptation. |
| **False Alarm / Miss Rates** | Fixed $P_D = 0.95, P_{FA} = 0.02$. Zero noise bursts. | **Moderate** | Benign error rates allow simple thresholding to succeed. |

### Concluding Assessment
> **The current canonical benchmark primarily measures sequence memorization of clean, deterministic, single-emitter 3-hop cycles.**  
> Algorithms that achieve high interception ratios on this benchmark (such as V4.0's 50.6% on Scenario 1 or Whittle's 87.0% on Scenario 8) are exploiting fixed structural regularities rather than demonstrating general cognitive scanning under uncertainty.
