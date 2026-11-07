#!/usr/bin/env python3
"""Smallest detectable effects for R2 on the realised 2024-Q1 returns (pre-registration §8).

No model is called. Daily close-to-close returns of AAPL, GOOGL and AMZN from
the 2023-12-29 close to the 2024-03-28 close (61 returns, Yahoo, dividend
adjusted) stand in for the sessions R2 trades. The stationary bootstrap
(Politis and Romano 1994, mean block round(n^(1/3)) = 4, seed 0) resamples the
same dates for all three stocks. No-skill agents hold a rating that is redrawn
with probability 0.2 per session (mean run five sessions), mapped long/short
(+1/-1) or long/flat (1/0).

Run: python reports/r2-tradingagents-replication/prereg-observations/mde.py
"""

from __future__ import annotations

import math

import numpy as np
import yfinance as yf

TICKERS = ("AAPL", "GOOGL", "AMZN")
RESAMPLES = 4000
AGENTS = 150
AGENT_RESAMPLES = 800


def returns() -> np.ndarray:
    rows = []
    for ticker in TICKERS:
        bars = yf.download(ticker, start="2023-12-20", end="2024-04-05", auto_adjust=True, progress=False)
        bars.columns = [c[0] for c in bars.columns]
        close = bars.loc["2023-12-29":"2024-03-28", "Close"].to_numpy()
        rows.append(close[1:] / close[:-1] - 1)
    return np.vstack(rows)


def sharpe(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1, axis=-1)
    return np.sqrt(252) * x.mean(axis=-1) / np.where(sd > 0, sd, np.nan)


def cumulative(x: np.ndarray) -> np.ndarray:
    return np.prod(1 + x, axis=-1) - 1


def max_drawdown(x: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1 + x, axis=-1)
    return np.max(1 - equity / np.maximum.accumulate(equity, axis=-1), axis=-1)


def stationary_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < 1 / block else (idx[i - 1] + 1) % n
    return idx


def positions(mapping: str, n: int, rng: np.random.Generator) -> np.ndarray:
    states = [1.0, -1.0] if mapping == "long/short" else [1.0, 0.0]
    out = np.empty((len(TICKERS), n))
    for k in range(len(TICKERS)):
        current = rng.choice(states)
        for i in range(n):
            if rng.random() < 0.2:
                current = rng.choice(states)
            out[k, i] = current
    return out


def main() -> None:
    r = returns()
    n = r.shape[1]
    years = n / 252
    print("n = {} daily returns, {:.3f} years".format(n, years))
    print("\nLo (2002) standard error of an annualised Sharpe from daily data:")
    for sr in (0.0, 1.0, 3.53, 5.60, 6.39, 8.21):
        daily_form = math.sqrt((1 + sr * sr / (2 * 252)) / years)
        annual_form = math.sqrt((1 + sr * sr / 2) / years)
        print("  SR {:5.2f}: {:5.2f} (daily SR in the SR^2/2 term); {:5.2f} (annual SR in that term)".format(
            sr, daily_form, annual_form))

    rng = np.random.default_rng(0)
    block = round(n ** (1 / 3))
    boot = [stationary_indices(n, block, rng) for _ in range(RESAMPLES)]
    bh_sr = np.array([sharpe(r[:, ix]) for ix in boot])
    bh_cr = np.array([cumulative(r[:, ix]) for ix in boot])
    print("\nblock {}, {} resamples".format(block, RESAMPLES))
    print("buy and hold realised: Sharpe", np.round(sharpe(r), 2), "CR %", np.round(100 * cumulative(r), 2),
          "MDD %", np.round(100 * max_drawdown(r), 2))
    print("buy and hold bootstrap SD: Sharpe", np.round(bh_sr.std(axis=0), 2),
          "pooled", round(float(bh_sr.mean(axis=1).std()), 2),
          "| CR points", np.round(100 * bh_cr.std(axis=0), 2), "pooled", round(float(100 * bh_cr.mean(axis=1).std()), 2))

    for mapping in ("long/short", "long/flat"):
        sd_sr, sd_gap_sr, sd_gap_cr, pool_sr, pool_gap_sr, pool_gap_cr = [], [], [], [], [], []
        for _ in range(AGENTS):
            agent = positions(mapping, n, rng) * r
            s_a = np.array([sharpe(agent[:, ix]) for ix in boot[:AGENT_RESAMPLES]])
            c_a = np.array([cumulative(agent[:, ix]) for ix in boot[:AGENT_RESAMPLES]])
            s_b, c_b = bh_sr[:AGENT_RESAMPLES], bh_cr[:AGENT_RESAMPLES]
            sd_sr.append(np.nanstd(s_a, axis=0))
            sd_gap_sr.append(np.nanstd(s_a - s_b, axis=0))
            sd_gap_cr.append(np.std(c_a - c_b, axis=0))
            pool_sr.append(np.nanstd(np.nanmean(s_a, axis=1)))
            pool_gap_sr.append(np.nanstd(np.nanmean(s_a - s_b, axis=1)))
            pool_gap_cr.append(np.std(np.mean(c_a - c_b, axis=1)))
        print("\n{} ({} no-skill agents x {} resamples), bootstrap SD per stock and of the three-stock mean:".format(
            mapping, AGENTS, AGENT_RESAMPLES))
        print("  agent Sharpe          ", np.round(np.nanmean(sd_sr, axis=0), 2), round(float(np.nanmean(pool_sr)), 2))
        print("  Sharpe minus B&H      ", np.round(np.nanmean(sd_gap_sr, axis=0), 2), round(float(np.nanmean(pool_gap_sr)), 2))
        print("  CR minus B&H, points  ", np.round(100 * np.mean(sd_gap_cr, axis=0), 2), round(float(100 * np.mean(pool_gap_cr)), 2))


if __name__ == "__main__":
    main()
