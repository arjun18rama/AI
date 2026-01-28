"""Opponent pool utilities for self-play training."""
from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Optional
import random

import numpy as np
import torch


class OpponentPool:
    """Stores snapshots of policies for self-play.

    The pool keeps state_dict copies of policies. It can emit a callable that
    runs inference using a cloned policy. This keeps the environment decoupled
    from training while still enabling self-play with historical opponents.
    """

    def __init__(self, max_size: int = 8) -> None:
        self._snapshots: Deque[dict] = deque(maxlen=max_size)

    def add_policy(self, policy: torch.nn.Module) -> None:
        """Snapshot a policy by storing a CPU copy of its state_dict."""
        state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        self._snapshots.append(state)

    def sample_policy(self, policy_factory: Callable[[], torch.nn.Module]) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Return a callable opponent policy sampled from the pool.

        The returned callable expects a numpy observation and returns a numpy
        action. If the pool is empty, returns None.
        """
        if not self._snapshots:
            return None

        snapshot = random.choice(list(self._snapshots))
        policy = policy_factory()
        policy.load_state_dict(snapshot)
        policy.eval()

        def _policy_fn(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_tensor, _, _ = policy(obs_tensor)
                return action_tensor.squeeze(0).cpu().numpy()

        return _policy_fn
