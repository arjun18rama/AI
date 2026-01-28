"""Training entry point for parallel self-play PPO."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.arena_env import ArenaConfig, ArenaEnv
from selfplay.opponent_pool import OpponentPool


class SelfPlayCallback(BaseCallback):
    """Periodically snapshots the policy and updates opponents."""

    def __init__(self, pool: OpponentPool, update_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.pool = pool
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq == 0:
            self.pool.add_policy(self.model.policy)
            self._update_opponents()
        return True

    def _update_opponents(self) -> None:
        def policy_factory():
            policy = copy.deepcopy(self.model.policy)
            policy.to("cpu")
            return policy

        opponent_policy = self.pool.sample_policy(policy_factory)
        self.training_env.env_method("set_opponent_policy", opponent_policy)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_env(config: dict) -> gym.Env:
    arena_config = ArenaConfig(
        episode_length=config["env"]["episode_length"],
        control_timestep=config["env"]["control_timestep"],
        physics_timestep=config["env"]["physics_timestep"],
        opponent_distance=config["env"]["opponent_distance"],
        fall_height=config["env"]["fall_height"],
        action_scale=config["env"]["action_scale"],
        healthy_height=config["env"]["healthy_height"],
    )
    return ArenaEnv(config=arena_config)


def build_vec_env(config: dict):
    env_count = config["training"]["num_envs"]
    env_fns = [lambda: make_env(config) for _ in range(env_count)]
    if config["training"]["vec_env"] == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)

    vec_env = build_vec_env(config)
    pool = OpponentPool(max_size=config["selfplay"]["pool_size"])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config["training"]["learning_rate"],
        n_steps=config["training"]["n_steps"],
        batch_size=config["training"]["batch_size"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
        n_epochs=config["training"]["n_epochs"],
        clip_range=config["training"]["clip_range"],
        verbose=1,
        device="auto",
    )

    pool.add_policy(model.policy)
    vec_env.env_method("set_opponent_policy", pool.sample_policy(lambda: copy.deepcopy(model.policy)))

    callback = SelfPlayCallback(pool=pool, update_freq=config["selfplay"]["snapshot_freq"])
    model.learn(total_timesteps=config["training"]["total_timesteps"], callback=callback)

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "ppo_selfplay")


if __name__ == "__main__":
    torch.set_num_threads(1)
    np.random.seed(0)
    main()
