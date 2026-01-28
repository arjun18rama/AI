import pytest

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")

from selfplay.opponent_pool import OpponentPool


class _DummyPolicy:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(1, 1)


def test_opponent_pool_add_and_sample() -> None:
    pool = OpponentPool(max_size=2, rng_seed=0)
    pool.add(_DummyPolicy())

    snapshot = pool.sample()
    assert snapshot is not None
    assert len(snapshot) > 0


def test_opponent_pool_max_size_trims_oldest() -> None:
    pool = OpponentPool(max_size=1, rng_seed=0)
    pool.add(_DummyPolicy())
    pool.add(_DummyPolicy())

    assert len(pool._snapshots) == 1
