import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from rl.trade import metrics
from rl.trade.backtest import backtest_stats, compare_stats
from rl.trade.baselines import buy_and_hold, equal_weight_rebalanced

PYFOLIO_STATS = [
    "Annual return",
    "Cumulative returns",
    "Annual volatility",
    "Sharpe ratio",
    "Calmar ratio",
    "Stability",
    "Max drawdown",
    "Omega ratio",
    "Sortino ratio",
    "Skew",
    "Kurtosis",
    "Tail ratio",
    "Daily value at risk",
]


def test_return_and_risk_formulas():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    assert metrics.cumulative_returns(r) == pytest.approx(1.01 * 0.98 * 1.03 - 1)
    years = 4 / 252
    assert metrics.annual_return(r) == pytest.approx((1.01 * 0.98 * 1.03) ** (1 / years) - 1)
    assert metrics.annual_volatility(r) == pytest.approx(r.std() * np.sqrt(252))
    assert metrics.sharpe_ratio(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    assert metrics.sortino_ratio(r) == pytest.approx(r.mean() * 252 / downside)
    assert metrics.omega_ratio(r) == pytest.approx(0.04 / 0.02)


def test_max_drawdown_counts_the_starting_capital():
    values = pd.Series([100.0, 120.0, 90.0, 100.0])
    assert metrics.max_drawdown(values.pct_change()) == pytest.approx(90 / 120 - 1)
    # A loss on the very first day is a drawdown from the initial capital.
    assert metrics.max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)
    # No drawdown, so the Calmar ratio is undefined.
    assert np.isnan(metrics.calmar_ratio(pd.Series([0.01] * 10)))


def test_degenerate_inputs_return_nan():
    flat = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(metrics.sharpe_ratio(flat))
    assert np.isnan(metrics.sortino_ratio(flat))
    assert np.isnan(metrics.annual_return(pd.Series([], dtype=float)))


def test_backtest_stats_has_pyfolio_fields(capsys):
    account = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=60).strftime("%Y-%m-%d"),
            "account_value": 1e6 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 60)),
        }
    )
    stats = backtest_stats(account)
    assert list(stats.index) == PYFOLIO_STATS
    assert capsys.readouterr().out == ""  # logged, not printed
    table = compare_stats({"A": account, "B": account.rename(columns={"account_value": "close"})})
    assert list(table.columns) == ["A", "B"]
    assert table.loc["Final value", "A"] == pytest.approx(account.account_value.iloc[-1])
    assert table.loc["Sharpe ratio", "A"] == pytest.approx(table.loc["Sharpe ratio", "B"])


def test_buy_and_hold_tracks_equal_initial_weights():
    df = make_panel([[10.0, 20.0], [20.0, 20.0], [20.0, 10.0]])
    values = buy_and_hold(df, initial_amount=100.0).account_value.to_numpy()
    # 5 shares of T0 and 2.5 of T1.
    np.testing.assert_allclose(values, [100.0, 150.0, 125.0])


def test_rebalanced_portfolio_and_costs():
    df = make_panel([[10.0, 20.0], [20.0, 20.0], [20.0, 10.0]])
    values = equal_weight_rebalanced(df, 100.0, rebalance_every=1).account_value.to_numpy()
    # Daily rebalancing earns the average of the stocks' returns each day.
    np.testing.assert_allclose(values, [100.0, 150.0, 150.0 * 0.75])
    with_costs = equal_weight_rebalanced(df, 100.0, transaction_cost_pct=0.01, rebalance_every=1)
    assert (with_costs.account_value.to_numpy()[1:] < values[1:]).all()
    with pytest.raises(ValueError):
        equal_weight_rebalanced(df, rebalance_every=0)
