import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from envs.arena_env import ArenaEnv


def test_arena_env_reset_smoke() -> None:
    env = ArenaEnv()
    obs, info = env.reset()

    assert obs is not None
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
