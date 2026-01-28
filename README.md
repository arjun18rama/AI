# AI Self-Play Humanoid PPO

## Purpose
This project trains a self-play Proximal Policy Optimization (PPO) agent to control two MuJoCo humanoids in a shared arena. The goal is to learn stable locomotion and balance while inducing imbalance in an opponent, using a single shared policy that acts on both agents. The main entrypoint for training is `train.py`.【F:train.py†L1-L80】

## High-level architecture
- **Training loop** (`train.py`): Loads YAML config, builds a vectorized environment, trains PPO in rollouts, and periodically snapshots opponents into a pool for self-play. It saves the final checkpoint to the configured path.【F:train.py†L9-L79】
- **Environments** (`envs/`):
  - `envs/arena_env.py` defines the two-agent arena, observation composition, rewards, and termination conditions.【F:envs/arena_env.py†L1-L186】
  - `envs/humanoid_env.py` defines the humanoid body spec and a single-agent environment used for debugging or experiments.【F:envs/humanoid_env.py†L1-L201】
- **Self-play utilities** (`selfplay/opponent_pool.py`): Stores and samples prior policy snapshots for opponent diversity during training.【F:selfplay/opponent_pool.py†L1-L51】
- **Configuration** (`configs/default.yaml`): Centralizes training hyperparameters, vectorization settings, and snapshot cadence.【F:configs/default.yaml†L1-L20】

## Setup prerequisites
- **Python** 3.10+ recommended.
- **System dependencies**: MuJoCo (and its license/installation), plus any OS packages required by `mujoco`, `gymnasium`, and `stable-baselines3`.
- **Python packages** (typical): `mujoco`, `gymnasium`, `stable-baselines3`, `numpy`, `PyYAML`.

### Local development setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mujoco gymnasium stable-baselines3 numpy PyYAML
```

## Usage examples
### Train with the default config
```bash
python train.py --config configs/default.yaml
```

### Train with a custom config
```bash
python train.py --config /path/to/your_config.yaml
```

### Configure device selection
To use Apple Metal (MPS) with a PyTorch build that supports it, set `training.device: mps` in your YAML config (for example, in `configs/default.yaml`). Otherwise, keep the default `training.device: cpu` or set it to `cuda` for NVIDIA GPUs.

### Use the environments directly (API example)
```python
from envs.arena_env import ArenaEnv

env = ArenaEnv(frame_skip=10, seed=0)
obs, info = env.reset()
# Random action example
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Contribution guidelines
- **Branching**: Create feature branches off `main` (e.g., `feature/your-change`) and keep changes focused.
- **Pull requests**: Provide a clear summary and include any relevant configuration updates. If you change training behavior, note the expected impact on checkpoints/logging.
- **Testing expectations**:
  - Run at least one training sanity check (short run) when modifying the training loop or environment logic.
  - If you change configuration defaults, verify that `python train.py --config configs/default.yaml` still starts.
- **Code navigation**:
  - Training entrypoint: `train.py`.
  - Environment definitions: `envs/arena_env.py`, `envs/humanoid_env.py`.
  - Self-play logic: `selfplay/opponent_pool.py`.
  - Default hyperparameters: `configs/default.yaml`.
