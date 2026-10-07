import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from conftest import TECH, make_panel
from rl.env.env_stocktrading import StockTradingEnv
from rl.preprocessing.data import data_split


def make_env(df, tech=(), hmax=10, initial_amount=1000.0, cost=0.0, **kwargs):
    n = df["tic"].nunique()
    tech = list(tech)
    return StockTradingEnv(
        df,
        stock_dim=n,
        hmax=hmax,
        initial_amount=initial_amount,
        buy_cost_pct=kwargs.pop("buy_cost_pct", cost),
        sell_cost_pct=kwargs.pop("sell_cost_pct", cost),
        reward_scaling=kwargs.pop("reward_scaling", 1.0),
        state_space=1 + 2 * n + len(tech) * n,
        action_space=n,
        tech_indicator_list=tech,
        print_verbosity=0,
        **kwargs,
    )


def test_passes_stable_baselines_env_checker(processed):
    env = make_env(data_split(processed, "2017-01-01", "2018-01-01"), TECH, hmax=100, initial_amount=1e6)
    check_env(env)


def test_episode_length_accounting_and_memories(processed):
    data = data_split(processed, "2017-01-01", "2018-01-01")
    env = make_env(data, TECH, hmax=100, initial_amount=1e6, cost=0.001)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    n_days = data.index.nunique()
    rng = np.random.default_rng(0)
    steps, terminated = 0, False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, env.stock_dim))
        steps += 1
        assert not truncated
        assert env.cash >= -1e-6
        assert (env.holdings >= 0).all()
        assert np.allclose(env.holdings, np.round(env.holdings))
        prices = data.loc[env.day].sort_values("tic")["close"].to_numpy()
        assert env.asset_memory[-1] == pytest.approx(env.cash + env.holdings @ prices)
    assert steps == n_days - 1
    assert len(env.asset_memory) == len(env.date_memory) == n_days
    assert env.date_memory == sorted(data["date"].unique())
    stats = info["episode_stats"]
    assert stats["end_total_asset"] == pytest.approx(env.asset_memory[-1])
    actions = env.save_action_memory()
    assert actions.shape == (n_days - 1, env.stock_dim)
    assert list(actions.columns) == sorted(data["tic"].unique())
    with pytest.raises(RuntimeError):
        env.step(np.zeros(env.stock_dim))


def test_buy_and_sell_charge_costs():
    df = make_panel([[10.0], [10.0], [12.0]])
    env = make_env(df, hmax=10, initial_amount=1000.0, cost=0.01)
    env.reset()
    env.step(np.array([1.0]))  # buy 10 shares at 10 with 1% cost
    assert env.holdings[0] == 10
    assert env.cash == pytest.approx(1000 - 10 * 10 * 1.01)
    env.step(np.array([-0.5]))  # sell 5 shares at 10 with 1% cost
    assert env.holdings[0] == 5
    assert env.cash == pytest.approx(899 + 5 * 10 * 0.99)
    assert env.cost == pytest.approx(1.0 + 0.5)
    assert env.trades == 2
    # Final value uses the last day's price.
    assert env.asset_memory[-1] == pytest.approx(env.cash + 5 * 12)


def test_buys_are_limited_by_cash_including_costs():
    df = make_panel([[100.0], [100.0]])
    env = make_env(df, hmax=100, initial_amount=1000.0, cost=0.01)
    env.reset()
    env.step(np.array([1.0]))
    # 10 shares would cost 1010 with fees; only 9 are affordable.
    assert env.holdings[0] == 9
    assert env.cash == pytest.approx(1000 - 9 * 101)
    assert env.cash >= 0


def test_cannot_sell_more_than_held_and_small_actions_round_to_zero():
    df = make_panel([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    env = make_env(df, hmax=10, initial_amount=1000.0)
    env.reset()
    env.step(np.array([0.3, -1.0]))  # buy 3 of T0, nothing to sell of T1
    assert list(env.holdings) == [3, 0]
    env.step(np.array([-1.0, 0.05]))  # sell all 3 (not 10); 0.5 share rounds to 0
    assert list(env.holdings) == [0, 0]
    assert list(env.actions_memory[-1]) == [-3, 0]


def test_larger_buy_orders_fill_first_when_cash_is_short():
    df = make_panel([[10.0, 10.0], [10.0, 10.0]])
    env = make_env(df, hmax=10, initial_amount=120.0)
    env.reset()
    env.step(np.array([0.5, 1.0]))
    assert list(env.holdings) == [2, 10]


def test_turbulence_liquidates_with_costs_and_blocks_buys():
    df = make_panel([[10.0], [10.0], [10.0], [10.0]], turbulence=[0, 0, 99, 99])
    env = make_env(df, hmax=10, initial_amount=1000.0, cost=0.01, turbulence_threshold=50)
    env.reset()
    env.step(np.array([1.0]))
    assert env.holdings[0] == 10
    cost_before = env.cost
    env.step(np.array([1.0]))  # day 1 is calm: buys allowed
    assert env.holdings[0] == 20
    env.step(np.array([1.0]))  # day 2 is turbulent: sell everything
    assert env.holdings[0] == 0
    # The old environment recorded zero cost for forced liquidations.
    assert env.cost == pytest.approx(cost_before + 1.0 + 20 * 10 * 0.01)


def test_continues_from_previous_state():
    df = make_panel([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
    previous = [500.0, 9.0, 9.0, 3.0, 4.0]
    env = make_env(df, hmax=10, initial_amount=1000.0, initial=False, previous_state=previous)
    env.reset()
    assert env.cash == 500.0
    assert list(env.holdings) == [3.0, 4.0]
    # Previous holdings are valued at the new window's prices.
    assert env.asset_memory[0] == pytest.approx(500 + 3 * 10 + 4 * 20)


def test_log_return_reward():
    df = make_panel([[10.0], [12.0]])
    env = make_env(df, hmax=10, initial_amount=100.0, reward_type="log_return", reward_scaling=2.0)
    env.reset()
    _, reward, terminated, _, _ = env.step(np.array([1.0]))
    assert terminated
    assert reward == pytest.approx(2.0 * np.log(120.0 / 100.0))


def test_reward_is_scaled_asset_change():
    df = make_panel([[10.0], [12.0]])
    env = make_env(df, hmax=10, initial_amount=100.0, reward_scaling=0.5)
    env.reset()
    _, reward, _, _, _ = env.step(np.array([1.0]))
    assert reward == pytest.approx(0.5 * (120.0 - 100.0))


def test_rejects_unbalanced_data_and_bad_dimensions():
    df = make_panel([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    unbalanced = df.iloc[1:]  # day 0 loses a ticker
    with pytest.raises(ValueError, match="same tickers"):
        make_env(unbalanced)
    with pytest.raises(ValueError, match="state_space"):
        StockTradingEnv(df, 2, 10, 1000, 0, 0, 1, 99, 2, [])
    with pytest.raises(ValueError, match="reward_type"):
        make_env(df, reward_type="sharpe")
