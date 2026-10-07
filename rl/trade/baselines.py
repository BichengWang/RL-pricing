"""Rule-based portfolios to benchmark trained agents against.

Each function takes a long (date, tic) frame with a ``close`` column and
returns a frame with ``date`` and ``account_value`` columns, the same format
as :meth:`rl.env.env_stocktrading.StockTradingEnv.save_asset_memory`.
"""

import numpy as np
import pandas as pd


def _close_matrix(df):
    prices = df.pivot_table(index="date", columns="tic", values="close").sort_index()
    if prices.isna().any().any():
        raise ValueError("Every date must have a close price for every ticker")
    return prices


def buy_and_hold(df, initial_amount=1_000_000, buy_cost_pct=0.0):
    """Split cash equally across all tickers on the first day and hold."""
    prices = _close_matrix(df)
    first = prices.iloc[0].to_numpy()
    budget = initial_amount / len(first)
    shares = budget / (first * (1 + buy_cost_pct))
    values = prices.to_numpy() @ shares
    return pd.DataFrame({"date": prices.index, "account_value": values})


def equal_weight_rebalanced(
    df, initial_amount=1_000_000, transaction_cost_pct=0.0, rebalance_every=21
):
    """Equal-weight portfolio rebalanced every ``rebalance_every`` days.

    Transaction costs are charged on the value traded at each rebalance.
    """
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1")
    frame = _close_matrix(df)
    prices = frame.to_numpy()
    n_days, n_tickers = prices.shape
    target = np.full(n_tickers, 1.0 / n_tickers)
    positions = initial_amount * target / (1 + transaction_cost_pct)
    values = np.empty(n_days)
    values[0] = positions.sum()
    for t in range(1, n_days):
        positions = positions * prices[t] / prices[t - 1]
        total = positions.sum()
        if t % rebalance_every == 0:
            # Charge costs on the value traded to get back to equal weights.
            turnover = np.abs(total * target - positions).sum()
            total -= transaction_cost_pct * turnover
            positions = total * target
        values[t] = total
    return pd.DataFrame({"date": frame.index, "account_value": values})


def target_weights_strategy(df, weights, initial_amount=1_000_000, transaction_cost_pct=0.0):
    """Account values of a portfolio rebalanced daily to target weights.

    ``weights`` is indexed by date with one column per ticker: the fraction
    of the account to hold in each stock from that day's close to the next.
    Whatever is not invested is cash; weights summing to more than one
    borrow cash at no interest. Transaction costs are charged on the value
    traded at each rebalance.
    """
    frame = _close_matrix(df)
    weights = weights.reindex(index=frame.index, columns=frame.columns)
    if weights.isna().any().any():
        raise ValueError("weights must cover every date and ticker of df")
    prices = frame.to_numpy()
    targets = weights.to_numpy(dtype=float)
    positions = np.zeros(prices.shape[1])
    total = float(initial_amount)
    values = np.empty(len(prices))
    for t in range(len(prices)):
        if t > 0:
            new_positions = positions * prices[t] / prices[t - 1]
            total += (new_positions - positions).sum()
            positions = new_positions
        target = total * targets[t]
        total -= transaction_cost_pct * np.abs(target - positions).sum()
        positions = total * targets[t]
        values[t] = total
    return pd.DataFrame({"date": frame.index, "account_value": values})


def baseline_account_values(df, initial_amount=1_000_000, transaction_cost_pct=0.0):
    """Account values of every built-in baseline, keyed by a display name."""
    return {
        "Equal-weight buy & hold": buy_and_hold(df, initial_amount, transaction_cost_pct),
        "Equal-weight monthly rebalance": equal_weight_rebalanced(
            df, initial_amount, transaction_cost_pct, rebalance_every=21
        ),
    }
