"""Opponent pool for self-play.

Stores snapshots of policy parameters and provides sampling utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class OpponentPool:
    """Lightweight storage for previous policy snapshots."""

    max_size: int = 8
    rng_seed: int | None = None
    _snapshots: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def add(self, policy: BaseAlgorithm) -> None:
        """Save a copy of the policy parameters for later sampling."""

        state = policy.policy.state_dict()
        self._snapshots.append({k: v.detach().cpu().clone() for k, v in state.items()})
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)

    def sample(self) -> dict | None:
        """Return a random snapshot, or None if the pool is empty."""

        if not self._snapshots:
            return None
        idx = self._rng.integers(0, len(self._snapshots))
        return self._snapshots[int(idx)]

    def load_into(self, policy: BaseAlgorithm) -> bool:
        """Load a randomly sampled snapshot into the provided policy.

        Returns True if a snapshot was loaded.
        """

        snapshot = self.sample()
        if snapshot is None:
            return False
        policy.policy.load_state_dict(snapshot)
        return True
