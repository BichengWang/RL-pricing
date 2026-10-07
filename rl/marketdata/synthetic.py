"""Synthetic daily price data for offline runs, demos and tests.

Prices follow correlated geometric Brownian motions driven by one shared
market factor plus stock-specific noise. The output has the same columns as
:class:`rl.marketdata.yahoodownloader.YahooDownloader`.
"""

import numpy as np
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume", "tic", "day"]


def generate_synthetic_data(
    ticker_list,
    start_date,
    end_date,
    seed=0,
    annual_drift=0.08,
    annual_volatility=0.25,
    market_correlation=0.5,
    start_price=100.0,
):
    """Return business-day OHLCV data for ``ticker_list`` in [start, end).

    Parameters
    ----------
    annual_drift, annual_volatility :
        Average drift and volatility; each ticker gets its own values
        scattered around them.
    market_correlation :
        Share of each stock's return variance explained by the market factor
        (0 gives independent stocks, 1 identical ones).
    """
    if not 0 <= market_correlation <= 1:
        raise ValueError("market_correlation must be between 0 and 1")
    tickers = list(ticker_list)
    dates = pd.bdate_range(start_date, end_date, inclusive="left")
    if not tickers or len(dates) == 0:
        return pd.DataFrame(columns=COLUMNS)
    rng = np.random.default_rng(seed)
    n_days, n_tickers = len(dates), len(tickers)
    dt = 1.0 / 252

    drift = rng.normal(annual_drift, 0.05, n_tickers)
    volatility = annual_volatility * rng.uniform(0.6, 1.4, n_tickers)
    initial = start_price * rng.uniform(0.5, 2.0, n_tickers)
    # Draw each day's random numbers as one row so that a longer date range
    # extends, rather than reshuffles, the series for a given start date.
    draws = rng.standard_normal((n_days, 1 + 5 * n_tickers))
    market = draws[:, :1]
    idiosyncratic, gap, high_noise, low_noise, volume_noise = np.split(draws[:, 1:], 5, axis=1)

    shocks = np.sqrt(market_correlation) * market + np.sqrt(1 - market_correlation) * idiosyncratic
    log_returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_returns[0] = 0.0
    close = initial * np.exp(np.cumsum(log_returns, axis=0))

    previous_close = np.vstack([close[:1], close[:-1]])
    open_ = previous_close * np.exp(0.003 * gap)
    high = np.maximum(open_, close) * (1 + np.abs(0.006 * high_noise))
    low = np.minimum(open_, close) * (1 - np.abs(0.006 * low_noise))
    volume = np.round(1e6 * np.exp(0.3 * volume_noise))

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.strftime("%Y-%m-%d").to_numpy(), n_tickers),
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "volume": volume.ravel(),
            "tic": np.tile(tickers, n_days),
            "day": np.repeat(dates.dayofweek.to_numpy(), n_tickers),
        }
    )
    return frame[COLUMNS].sort_values(["date", "tic"]).reset_index(drop=True)
