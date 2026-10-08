import contextlib
import io
import warnings

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from conftest import TECH
from rl.env.env_portfolio import StockPortfolioEnv
from rl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from rl.env.env_stocktrading_stoploss import StockTradingEnvStopLoss
from rl.preprocessing.data import data_split
from rl.preprocessing.preprocessors import add_covariance_matrix


def run_episode(env, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = scale * rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        _, _, terminated, truncated, info = env.step(action)
    return info


def test_portfolio_env(processed):
    data = data_split(add_covariance_matrix(processed, lookback=60), "2016-01-01", "2018-07-01")
    env = StockPortfolioEnv(data, 3, 100, 1e6, 0.001, 1e-4, 3, 3, TECH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 2-D observations are flagged as unusual
        check_env(env)
    info = run_episode(env)
    stats = info["episode_stats"]
    assert stats["total_cost"] > 0
    values = env.save_asset_memory()
    assert len(values) == data.index.nunique()
    weights = env.save_action_memory()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    # One row per decision, dated by the day it was made, as in StockTradingEnv.
    assert list(weights.index) == list(values.date.iloc[:-1])


def test_portfolio_weights_are_dated_when_chosen(processed):
    data = data_split(add_covariance_matrix(processed, lookback=60), "2017-01-01", "2017-01-10")
    env = StockPortfolioEnv(data, 3, 100, 1e6, 0.0, 1e-4, 3, 3, TECH)
    env.reset()
    first_day = env.date_memory[0]
    env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    weights = env.save_action_memory()
    assert list(weights.index) == [first_day]
    assert weights.iloc[0].to_numpy() == pytest.approx(env.softmax_normalization(np.array([1.0, 0, 0])))


def test_portfolio_costs_reduce_value(processed):
    data = data_split(add_covariance_matrix(processed, lookback=60), "2017-01-01", "2018-01-01")
    free = StockPortfolioEnv(data, 3, 100, 1e6, 0.0, 1e-4, 3, 3, TECH)
    costly = StockPortfolioEnv(data, 3, 100, 1e6, 0.01, 1e-4, 3, 3, TECH)
    run_episode(free)
    run_episode(costly)
    assert costly.portfolio_value < free.portfolio_value


def test_portfolio_first_allocation_pays_costs(processed):
    data = data_split(add_covariance_matrix(processed, lookback=60), "2017-01-01", "2017-03-01")
    env = StockPortfolioEnv(data, 3, 100, 1e6, 0.01, 1e-4, 3, 3, TECH)
    env.reset()
    # Equal weights from cash: the whole account is invested and pays 1%.
    env.step(np.zeros(3, dtype=np.float32))
    assert env.cost == pytest.approx(0.01 * 1e6)
    relative = env.data.close.to_numpy() / data.loc[0].close.to_numpy()
    assert env.portfolio_value == pytest.approx(0.99e6 * relative.mean())


def test_portfolio_log_return_reward(processed):
    data = data_split(add_covariance_matrix(processed, lookback=60), "2017-01-01", "2017-03-01")
    env = StockPortfolioEnv(data, 3, 100, 1e6, 0.001, 1.0, 3, 3, TECH, reward_type="log_return")
    env.reset()
    begin = env.portfolio_value
    _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert reward == pytest.approx(np.log(env.portfolio_value / begin))
    with pytest.raises(ValueError):
        StockPortfolioEnv(data, 3, 100, 1e6, 0.0, 1.0, 3, 3, TECH, reward_type="sharpe")


@pytest.mark.parametrize("env_class", [StockTradingEnvCashpenalty, StockTradingEnvStopLoss])
def test_penalty_envs(processed, env_class):
    env = env_class(
        processed,
        daily_information_cols=["open", "close", "high", "low", "volume"] + TECH,
        print_verbosity=10**9,
        patient=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        check_env(env)
        env.reset(seed=3)
        first_start = env.starting_point
        env.reset(seed=3)
        assert env.starting_point == first_start  # seeded random start
        info = run_episode(env, scale=0.1, seed=3)
    assert "total_assets" in info["episode_stats"]
    values = env.save_asset_memory()
    # Records start on the episode's first date.
    assert values.date.iloc[0] == env.dates[env.starting_point]
    assert len(values) == env.current_step
