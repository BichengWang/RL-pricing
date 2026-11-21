"""The benchmark's market data and execution model."""

import numpy as np
import pandas as pd
import pytest

from rl.bench.engine import Account, PolicyError, simulate
from rl.bench.market import FIELDS, MarketData
from rl.trade.baselines import target_weights_strategy


def panel(closes, opens=None, start="2020-01-01"):
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    n_dates, n_tickers = closes.shape
    return MarketData(
        dates=pd.bdate_range(start, periods=n_dates).to_numpy(dtype="datetime64[D]"),
        tickers=tuple("T{}".format(i) for i in range(n_tickers)),
        open=opens,
        high=np.fmax(opens, closes),
        low=np.fmin(opens, closes),
        close=closes,
        volume=np.where(np.isfinite(closes), 1.0, np.nan),
    )


@pytest.fixture
def walk():
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 3)), axis=0))
    opens = closes * np.exp(rng.normal(0, 0.003, closes.shape))
    return panel(closes, opens)


def test_matches_the_established_weights_backtest(walk):
    target = np.array([0.5, 0.3, 0.1])
    result = simulate(walk, lambda view: target, start=1, transaction_cost=0.002, initial=1000)
    frame = walk.to_frame()
    weights = pd.DataFrame([target] * (len(walk) - 1), index=frame.date.unique()[:-1], columns=walk.tickers)
    reference = target_weights_strategy(frame, weights, 1000, 0.002)
    np.testing.assert_allclose(result.equity, reference.account_value, rtol=1e-12)


def test_policy_sees_only_history_through_the_decision_close(walk):
    seen = []

    def act(view):
        seen.append((view.date, len(view.history), view.history.close[-1].copy()))
        return None

    simulate(walk, act, start=10, transaction_cost=0.0, stop=20)
    assert [date for date, _, _ in seen] == list(walk.dates[9:19])
    for date, length, last_close in seen:
        assert length == walk.index_of(date) + 1
        np.testing.assert_array_equal(last_close, walk.close[length - 1])


def test_first_fill_is_next_open_and_equity_starts_at_initial_capital():
    data = panel([[10.0], [12.0], [12.0]], opens=[[10.0], [11.0], [12.0]])
    result = simulate(data, lambda view: np.ones(1), start=1, transaction_cost=0.0, initial=110)
    # Buy 10 shares at the 11 open, mark at the 12 close.
    np.testing.assert_allclose(result.equity, [110, 120, 120])
    assert result.dates[0] == data.dates[0]


def test_holding_trades_nothing_and_drifts_with_prices(walk):
    def buy_once(view):
        return None if view.weights.sum() > 0 else np.full(3, 1 / 3)

    result = simulate(walk, buy_once, start=1, transaction_cost=0.01)
    assert result.turnover[0] > 0.98
    assert (result.turnover[1:] == 0).all() and (result.costs[1:] == 0).all()
    assert not np.allclose(result.weights[-1], result.weights[0])


def test_delisted_holding_is_sold_at_its_last_close_and_never_rebought():
    closes = np.array([[10.0, 10.0]] * 6)
    closes[3:, 1] = np.nan
    closes[2, 1] = 8.0
    data = panel(closes)
    result = simulate(data, lambda view: np.array([0.5, 0.5]), start=1, transaction_cost=0.01, initial=100)
    # Session 3 sells the T1 holding at its last close (8), paying 1% on it.
    t1_value = result.weights[1][1] * result.equity[2]
    assert result.costs[2] >= 0.01 * t1_value
    assert result.costs[2] == pytest.approx(0.01 * t1_value, rel=0.02)
    assert (result.weights[2:, 1] == 0).all()
    assert np.isfinite(result.equity).all()
    # Half the account stays in cash once T1 cannot be bought.
    np.testing.assert_allclose(result.weights[3:, 0], 0.5, rtol=1e-3)


def test_forced_exit_pays_the_fee_on_last_close_value():
    data = panel([[10.0], [8.0], [np.nan], [np.nan]])
    account = Account(data, transaction_cost=0.01, initial=100)
    account.trade(1, np.ones(1))
    held = account.equity(1)
    turnover, cost = account.trade(2, None)
    assert cost == pytest.approx(0.01 * held)
    assert account.cash == pytest.approx(0.99 * held)
    assert turnover == pytest.approx(1.0)


@pytest.mark.parametrize(
    "target", [np.array([0.7, 0.7, 0.0]), np.array([-0.1, 0.5, 0.5]), np.array([np.nan, 0, 0]), np.ones(2)]
)
def test_untradeable_targets_are_rejected(walk, target):
    with pytest.raises(PolicyError):
        simulate(walk, lambda view: target, start=1, transaction_cost=0.0)


def test_policies_cannot_modify_the_data(walk):
    def vandal(view):
        view.history.close[-1, 0] = 1.0

    with pytest.raises(ValueError):
        simulate(walk, vandal, start=1, transaction_cost=0.0)


def test_frame_round_trip_keeps_unlisted_sessions_missing():
    closes = np.array([[10.0, np.nan], [11.0, 20.0], [12.0, 21.0]])
    data = panel(closes)
    again = MarketData.from_frame(data.to_frame())
    assert again.sha256() == data.sha256()
    assert np.isnan(again.close[0, 1])
    assert again.between(data.dates[1], data.dates[2]).dates.tolist() == [data.dates[1]]


@pytest.mark.parametrize("field", ["open", "close"])
def test_open_and_close_must_be_missing_together(field):
    arrays = {name: np.full((2, 1), 10.0) for name in FIELDS}
    arrays[field][1, 0] = np.nan
    with pytest.raises(ValueError, match="same sessions"):
        MarketData(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"), ("A",), **arrays)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -0.001, 1.0, 1.5])
def test_account_rejects_a_transaction_cost_outside_zero_to_one(walk, cost):
    with pytest.raises(ValueError, match=r"transaction_cost must be in \[0, 1\)"):
        Account(walk, cost)


@pytest.mark.parametrize("initial", [0.0, -1.0, float("nan"), float("inf")])
def test_account_rejects_non_positive_or_non_finite_initial_capital(walk, initial):
    with pytest.raises(ValueError, match="initial capital must be positive"):
        Account(walk, 0.0, initial)


@pytest.mark.parametrize(
    "start, stop",
    [
        (0, None),  # the first decision would be at session -1, which wraps to the last close: look-ahead
        (-1, None),
        (60, None),  # nothing left to trade
        (61, None),
        (30, 30),
        (30, 61),  # past the data
    ],
)
def test_simulate_needs_a_session_before_start_and_a_window_inside_the_data(walk, start, stop):
    def act(view):
        raise AssertionError("no decision may be made before the window is validated")

    expected = "Need 1 <= start < stop <= 60, got {} and {}".format(start, 60 if stop is None else stop)
    with pytest.raises(ValueError, match=expected):
        simulate(walk, act, start, 0.0, stop=stop)
