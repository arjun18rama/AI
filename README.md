# parallel-3d-selfplay-rl

Train a single shared reinforcement learning policy across many parallel 3D physics arenas with self-play between two simplified humanoid agents. The system is optimized for fast, headless MuJoCo simulation and Gymnasium-compliant environments.

## Goals
- **Single shared policy** trained with PPO across many environments.
- **Self-play** between two simplified humanoids in the same arena.
- **Fast simulation** on Apple Silicon (M2 CPU) with headless MuJoCo.
- **Emergent behavior** driven by physics and reward shaping (no scripted fighting).

## Architecture overview
```
configs/
  default.yaml           # hyperparameters
envs/
  humanoid_env.py        # simplified humanoid XML builder
  arena_env.py           # two-humanoid self-play arena
selfplay/
  opponent_pool.py       # historical policy pool for self-play
train.py                 # PPO training loop
```

## Environment design
### Simplified bodies
- Capsule torso + two legs per agent.
- Four hinge joints per humanoid (two hips, two knees).
- Free root joint for 6-DOF movement.

### Rewards (per step)
- **Upright**: encourage torso height above the fall threshold.
- **Balance**: reward alignment of torso up vector with world up.
- **Opponent off-balance**: reward lowering the opponent’s torso height.
- **Control penalty**: discourage excessive torques.

### Termination
- Episode ends when either agent falls below a configurable height threshold.

## Training
### Requirements
- Python 3.10+
- `mujoco`
- `gymnasium`
- `stable-baselines3`
- `torch`

Install dependencies:
```bash
pip install mujoco gymnasium stable-baselines3 torch pyyaml
```

### Run training
```bash
python train.py --config configs/default.yaml
```

### Vectorization note
The default config uses `DummyVecEnv` so the self-play opponent policy can be passed by reference to each environment. If you switch to `SubprocVecEnv`, you may need to disable self-play snapshots or adjust the opponent policy to be picklable.

## Design decisions
- **Headless MuJoCo**: no rendering during training for speed and determinism.
- **Minimal morphology**: simple joint structure for faster learning and stability.
- **Reward shaping**: balance and stability terms dominate; opponent interaction emerges from physics without hardcoded combat techniques.

## Next steps
- Tune reward weights for more aggressive interaction.
- Expand the humanoid morphology while keeping capsules and joints.
- Add evaluation scripts for policy comparison across opponent snapshots.
