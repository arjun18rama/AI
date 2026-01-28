"""Training entry point for self-play PPO."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.arena_env import ArenaEnv
from selfplay.opponent_pool import OpponentPool


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_env(seed: int, frame_skip: int, reward_mode: str) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = ArenaEnv(frame_skip=frame_skip, seed=seed, reward_mode=reward_mode)
        return env

    return _init


def build_vec_env(config: dict) -> gym.Env:
    num_envs = int(config["training"]["num_envs"])
    frame_skip = int(config["training"]["frame_skip"])
    use_subproc = bool(config["training"].get("use_subproc", False))
    reward_mode = str(config["training"].get("reward_mode", "cooperative"))

    env_fns = [make_env(seed, frame_skip, reward_mode) for seed in range(num_envs)]
    if use_subproc and num_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    log_dir = Path(config["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["training"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    vec_env = build_vec_env(config)

    ppo_config = config["ppo"]
    device = config["training"]["device"]
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        ent_coef=ppo_config["ent_coef"],
        vf_coef=ppo_config["vf_coef"],
        max_grad_norm=ppo_config["max_grad_norm"],
        learning_rate=ppo_config["learning_rate"],
        tensorboard_log=str(log_dir),
        device=device,
        verbose=1,
    )

    opponent_pool = OpponentPool(max_size=config["selfplay"]["pool_size"])
    total_timesteps = int(config["training"]["total_timesteps"])
    save_every = int(config["selfplay"]["snapshot_every"])

    timesteps = 0
    while timesteps < total_timesteps:
        rollout = min(save_every, total_timesteps - timesteps)
        model.learn(total_timesteps=rollout, reset_num_timesteps=False)
        timesteps += rollout
        opponent_pool.add(model)

    model.save(str(checkpoint_path))


if __name__ == "__main__":
    main()
