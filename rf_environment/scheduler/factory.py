from __future__ import annotations

from typing import Any

from rf_environment.scheduler.base import ScanScheduler
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.discounted_thompson import DiscountedThompsonSamplingScheduler
from rf_environment.scheduler.hybrid.hybrid_scheduler import HybridScheduler
from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
from rf_environment.scheduler.random_scheduler import RandomScheduler
from rf_environment.scheduler.rl.ddqn_scheduler import DDQNScheduler
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.scheduler.rl_scheduler import RLScheduler
from rf_environment.scheduler.sequential_scheduler import SequentialScheduler
from rf_environment.scheduler.sliding_window_ucb import SlidingWindowUCBScheduler
from rf_environment.scheduler.thompson_sampling import ThompsonSamplingScheduler
from rf_environment.scheduler.ucb1 import UCB1Scheduler


SCHEDULER_METADATA = {
    "sequential": {
        "name": "Sequential",
        "category": "baseline",
        "description": "Systematically scans frequency bands in fixed sequential order.",
    },
    "random": {
        "name": "Random",
        "category": "baseline",
        "description": "Selects frequency bands uniformly at random without learning.",
    },
    "ucb1": {
        "name": "UCB1",
        "category": "baseline",
        "description": "Stationary Upper Confidence Bound bandit balancing mean reward with exploration bonus.",
    },
    "thompson": {
        "name": "Thompson Sampling",
        "category": "baseline",
        "description": "Stationary Bayesian bandit sampling Beta posterior distributions per frequency band.",
    },
    "sw_ucb": {
        "name": "Sliding Window UCB",
        "category": "non_stationary",
        "description": "Non-stationary UCB using only the last N observations to adapt to frequency changes.",
    },
    "discounted_thompson": {
        "name": "Discounted Thompson",
        "category": "non_stationary",
        "description": "Non-stationary Bayesian bandit with exponential memory discounting (gamma < 1.0).",
    },
    "context_aware": {
        "name": "Context-Aware Transition",
        "category": "contextual",
        "description": "Learns empirical frequency transition patterns from detections combined with recency and coverage bonuses.",
    },
    "rl": {
        "name": "RL Scheduler (Placeholder)",
        "category": "rl",
        "description": "Reinforcement learning scan scheduler interface placeholder.",
    },
    "ddqn": {
        "name": "Double DQN",
        "category": "rl",
        "description": "Double Deep Q-Network baseline scanning scheduler learning Q-values with target network separation.",
    },
    "hybrid_v4": {
        "name": "Hybrid CA + DDQN",
        "category": "hybrid",
        "description": "V4.0 Hybrid scheduler arbitrating between Context-Aware online adaptation and DDQN pattern prediction.",
    },
    "lstm_ddqn": {
        "name": "LSTM-DDQN (DRQN)",
        "category": "rl",
        "description": "V4.1 Recurrent Double DQN with LSTM temporal working memory (h_t, c_t).",
    },
    "hybrid_v41": {
        "name": "Hybrid CA + LSTM-DDQN",
        "category": "hybrid",
        "description": "V4.1 Hybrid scheduler arbitrating between Context-Aware online adaptation and LSTM-DDQN temporal memory.",
    },
}


def default_scan_bands(min_hz: float, max_hz: float, bandwidth_hz: float) -> list[float]:
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth must be positive")
    bands = []
    freq = min_hz + bandwidth_hz / 2.0
    while freq <= max_hz - bandwidth_hz / 2.0 + 1e-6:
        bands.append(freq)
        freq += bandwidth_hz
    return bands or [min_hz]


def create_scheduler(
    name: str,
    bands_hz: list[float],
    seed: int = 0,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ScanScheduler:
    key = name.lower().replace("-", "_")
    if key in {"random"}:
        return RandomScheduler(bands_hz, seed=seed)
    if key in {"sequential"}:
        return SequentialScheduler(bands_hz)
    if key in {"ucb1", "ucb"}:
        return UCB1Scheduler(bands_hz)
    if key in {"thompson", "thompson_sampling", "ts"}:
        return ThompsonSamplingScheduler(bands_hz, seed=seed)
    if key in {"sw_ucb", "sliding_window_ucb", "sliding_window"}:
        return SlidingWindowUCBScheduler(bands_hz, window_size=50)
    if key in {"discounted_thompson", "discounted_ts", "d_ts"}:
        return DiscountedThompsonSamplingScheduler(bands_hz, gamma=0.95, seed=seed)
    if key in {"context_aware", "contextual", "transition", "context"}:
        return ContextAwareScheduler(bands_hz, seed=seed)
    if key in {"rl", "rl_scheduler"}:
        return RLScheduler(bands_hz, allow_fallback_policy=True, seed=seed)
    if key in {"ddqn", "double_dqn"}:
        return DDQNScheduler(bands_hz, seed=seed)
    if key in {"hybrid", "hybrid_v4", "hybrid_meta"}:
        cfg = config or kwargs
        return HybridScheduler(
            bands_hz,
            ca_config=cfg.get("context_aware"),
            ddqn_config=cfg.get("ddqn"),
            arbitrator_config=cfg.get("arbitrator"),
            seed=seed,
        )
    if key in {"lstm_ddqn", "drqn", "lstm_dqn"}:
        cfg = config or kwargs
        return LSTMDDQNScheduler(bands_hz, seed=seed, **cfg)
    if key in {"hybrid_v41", "hybrid_lstm", "lstm_hybrid"}:
        cfg = config or kwargs
        return LSTMHybridScheduler(
            bands_hz,
            ca_config=cfg.get("context_aware"),
            lstm_ddqn_config=cfg.get("lstm_ddqn"),
            arbitrator_config=cfg.get("arbitrator"),
            seed=seed,
        )
    raise ValueError(f"Unknown scheduler: {name}. Available: {list(SCHEDULER_METADATA.keys())}")
