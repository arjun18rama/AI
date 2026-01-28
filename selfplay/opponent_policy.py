"""Helpers for mixing policy actions during self-play."""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


def combine_actions(
    main_action: np.ndarray,
    opponent_action: np.ndarray,
    split_index: int,
) -> np.ndarray:
    """Combine agent and opponent actions into the ArenaEnv action vector."""

    if split_index <= 0:
        raise ValueError("split_index must be positive.")
    return np.concatenate([main_action, opponent_action], axis=-1)


class OpponentActionWrapper(gym.Wrapper):
    """Wrap an ArenaEnv to inject opponent actions."""

    def __init__(
        self,
        env: gym.Env,
        opponent_model: Optional[BaseAlgorithm] = None,
        split_index: int | None = None,
    ) -> None:
        super().__init__(env)
        action_dim = int(env.action_space.shape[0])
        if split_index is None:
            split_index = action_dim // 2
        if split_index <= 0 or split_index >= action_dim:
            raise ValueError("split_index must be within the action dimension.")
        self._split_index = split_index
        self._opponent_model = opponent_model
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._split_index,),
            dtype=np.float32,
        )
        self._last_obs: np.ndarray | None = None

    def set_opponent_model(self, opponent_model: BaseAlgorithm) -> None:
        """Attach an opponent model used to generate opponent actions."""

        self._opponent_model = opponent_model

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray):
        if self._last_obs is None:
            raise RuntimeError("OpponentActionWrapper.step called before reset.")
        if self._opponent_model is None:
            raise RuntimeError("OpponentActionWrapper missing opponent model.")
        opponent_action = self._predict_opponent_action(self._last_obs)
        combined_action = combine_actions(action, opponent_action, self._split_index)
        obs, reward, terminated, truncated, info = self.env.step(combined_action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def _predict_opponent_action(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._opponent_model.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.float32)
