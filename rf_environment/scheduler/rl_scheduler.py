"""Placeholder for a future RL / contextual-bandit scan scheduler."""

from rf_environment.scheduler.base import ScanScheduler


class RLScheduler(ScanScheduler):
    name = "rl"

    def select_frequency(self, observation):
        raise NotImplementedError("RL scheduler is reserved for a later milestone")

    def update(self, observation, reward):
        raise NotImplementedError("RL scheduler is reserved for a later milestone")
