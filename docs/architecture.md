# Architecture Overview

## Major components and responsibilities

- **Training entry point (`train.py`)**
  - Parses CLI arguments, loads configuration, builds the vectorized environment, and runs PPO training loops.
  - Orchestrates self-play by periodically snapshotting the current policy into an opponent pool and saving checkpoints.
- **Environment definitions (`envs/`)**
  - `envs/arena_env.py`: Two-humanoid self-play arena; constructs the MuJoCo model, builds observations/actions, and computes rewards.
  - `envs/humanoid_env.py`: Single-humanoid diagnostic environment; defines shared humanoid body specification and XML builder used by the arena.
- **Self-play utilities (`selfplay/`)**
  - `selfplay/opponent_pool.py`: Stores historical policy snapshots and provides sampling/loading helpers for opponent selection.
- **Configuration (`configs/`)**
  - `configs/default.yaml`: Centralized hyperparameters and training/self-play settings (env count, timesteps, PPO args, snapshot cadence).

## Data flow and integration points

1. **Configuration load**: `train.py` reads `configs/default.yaml` to collect training, PPO, and self-play parameters.
2. **Environment construction**: `train.py` builds a vectorized environment using `envs/arena_env.py` (via `build_vec_env` and `make_env`). The arena environment imports `HUMANOID_SPEC` and `build_humanoid_xml` from `envs/humanoid_env.py` to ensure consistent joint/actuator ordering across agents.
3. **Training loop**: `train.py` instantiates a Stable Baselines3 PPO model and runs incremental `learn()` calls.
4. **Self-play snapshots**: After each rollout chunk, `train.py` pushes the current PPO policy into `selfplay/opponent_pool.py`, which clones the policy state dict for future opponent sampling.
5. **Checkpoint output**: At the end of training, `train.py` saves model parameters to the configured checkpoint path.

Integration points to external libraries:
- **MuJoCo** (`mujoco`): Creates and steps the physics model in `envs/arena_env.py` and `envs/humanoid_env.py`.
- **Gymnasium** (`gymnasium`): Provides the Env API, spaces, and seeding utilities in the environment modules.
- **Stable Baselines3** (`stable_baselines3`): Implements PPO and vectorized env wrappers used in `train.py` and `selfplay/opponent_pool.py`.
- **YAML** (`yaml`): Loads configuration in `train.py`.

## Key module boundaries and extension points

- **Environment boundary (`envs/`)**
  - *Boundary*: All simulation, reward shaping, and observation design are isolated in `envs/arena_env.py` and `envs/humanoid_env.py`.
  - *Extension points*:
    - Add new reward terms or opponent interactions in `envs/arena_env.py`.
    - Adjust humanoid morphology or observation layout in `envs/humanoid_env.py`, keeping `HUMANOID_SPEC` consistent.
    - Introduce additional environment variants as new modules under `envs/` and switch the factory in `train.py`.

- **Training boundary (`train.py`)**
  - *Boundary*: Orchestrates configuration parsing, vectorized env selection, PPO construction, and training scheduling.
  - *Extension points*:
    - Swap PPO for another algorithm by changing the model instantiation in `train.py`.
    - Add evaluation loops, logging hooks, or custom callbacks in the training loop.
    - Extend CLI/config options to expose new hyperparameters.

- **Self-play boundary (`selfplay/`)**
  - *Boundary*: `selfplay/opponent_pool.py` is the sole owner of opponent snapshot storage and sampling.
  - *Extension points*:
    - Implement alternative sampling strategies (e.g., weighted, curriculum-based) in `OpponentPool.sample()`.
    - Add metadata (e.g., Elo, timestamps) to snapshots and modify loading logic in `OpponentPool.load_into()`.

- **Configuration boundary (`configs/`)**
  - *Boundary*: YAML files define defaults for training, PPO, and self-play cadence.
  - *Extension points*:
    - Add new config sections (e.g., evaluation, logging targets) and read them in `train.py`.
    - Provide environment-specific configs as additional YAML files in `configs/`.
