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
    assert np.isnan(metrics.sharpe_ratio(pd.Series([0.01] * 30)))  # std is float residue, not 0
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
    df = make_panel(
        [[10.0, 20.0], [20.0, 20.0], [20.0, 10.0]],
        open=[[8.0, 25.0], [10.0, 20.0], [40.0, 5.0]],
    )
    values = buy_and_hold(df, initial_amount=100.0).account_value.to_numpy()
    # First date is cash; the next open buys 5 T0 and 2.5 T1, held through gaps.
    np.testing.assert_allclose(values, [100.0, 150.0, 125.0])
    costly = buy_and_hold(df, initial_amount=100.0, buy_cost_pct=0.1).account_value
    np.testing.assert_allclose(costly, [100, 150 / 1.1, 125 / 1.1])


def test_rebalanced_portfolio_and_costs():
    df = make_panel(
        [[10.0, 20.0], [20.0, 20.0], [20.0, 10.0]],
        open=[[10.0, 20.0], [10.0, 20.0], [20.0, 20.0]],
    )
    values = equal_weight_rebalanced(df, 100.0, rebalance_every=1).account_value.to_numpy()
    # Signals rebalance at the next open, then earn the open-to-close return.
    np.testing.assert_allclose(values, [100.0, 150.0, 150.0 * 0.75])
    with_costs = equal_weight_rebalanced(df, 100.0, transaction_cost_pct=0.01, rebalance_every=1)
    assert (with_costs.account_value.to_numpy()[1:] < values[1:]).all()
    # Second-fill turnover is 50/1.01, not a percentage of pre-trade targets.
    np.testing.assert_allclose(
        with_costs.account_value, [100, 150 / 1.01, 0.75 * 149.5 / 1.01]
    )
    with pytest.raises(ValueError):
        equal_weight_rebalanced(df, rebalance_every=0)


def test_rebalance_cadence_counts_signal_dates_and_retains_gap_pnl():
    df = make_panel(
        [[10, 20], [20, 20], [30, 20], [40, 40], [20, 40]],
        open=[[10, 20], [10, 20], [20, 10], [40, 10], [20, 20]],
    )
    values = equal_weight_rebalanced(df, 100, rebalance_every=2).account_value
    # Fill on dates 1 and 3. Five T0 and 2.5 T1 shares are held until date 3,
    # when open equity 225 buys 2.8125 T0 and 11.25 T1 shares.
    np.testing.assert_allclose(values, [100, 150, 200, 562.5, 506.25])


@pytest.mark.parametrize("baseline", [buy_and_hold, equal_weight_rebalanced])
@pytest.mark.parametrize("column", ["open", "close"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 0.0, -1.0])
def test_baselines_require_finite_positive_execution_prices(baseline, column, value):
    df = make_panel([[10], [12]])
    df.loc[1, column] = value
    with pytest.raises(ValueError):
        baseline(df)


@pytest.mark.parametrize("baseline", [buy_and_hold, equal_weight_rebalanced])
def test_baselines_require_raw_open_and_start_single_date_in_cash(baseline):
    df = make_panel([[10]])
    np.testing.assert_allclose(baseline(df, 100, 0.01).account_value, [100])
    with pytest.raises(ValueError):
        baseline(df.drop(columns="open"))
