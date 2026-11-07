#!/usr/bin/env python3
"""Checks of the TradingAgents paper's Table 1 made before any R2 run (pre-registration §3).

No model is called. Yahoo prices are read with yfinance and never written to disk.

1. Relation between Table 1's "CR%" and "ARR%" columns, and the final broker
   values printed in Figs. 6, S1 and S3 (assumed initial capital 100,000).
2. Buy and hold over the paper's window under four start conventions.
3. The dividend-adjustment factor implied by the last price bar printed in
   Figs. 6 (AAPL) and S3 (GOOGL), which dates the authors' price download.

Run: python reports/r2-tradingagents-replication/prereg-observations/table1_checks.py
"""

from __future__ import annotations

import math
from datetime import timedelta

import yfinance as yf

# Table 1, arXiv 2412.20138 v7 p. 11 (identical in v1-v6): (CR%, ARR%) per cell.
TABLE1 = {
    "AAPL": {"B&H": (-5.23, -5.09), "MACD": (-1.49, -1.48), "KDJ&RSI": (2.05, 2.07), "ZMR": (0.57, 0.57),
             "SMA": (-3.2, -2.97), "TradingAgents": (26.62, 30.5)},
    "GOOGL": {"B&H": (7.78, 8.09), "MACD": (6.20, 6.26), "KDJ&RSI": (0.4, 0.4), "ZMR": (-0.58, 0.58),
              "SMA": (6.23, 6.43), "TradingAgents": (24.36, 27.58)},
    "AMZN": {"B&H": (17.1, 17.6), "KDJ&RSI": (-0.77, -0.76), "ZMR": (-0.77, -0.77),
             "SMA": (11.01, 11.6), "TradingAgents": (23.21, 24.90)},
}
# Broker "value" in Figs. 6 (AAPL), S3 (GOOGL), S1 (AMZN); last bar's close in Figs. 6 and S3.
FIG_VALUE = {"AAPL": 130501.44, "GOOGL": 127586.29, "AMZN": 124872.71}
FIG_LAST_CLOSE = {"AAPL": 170.86, "GOOGL": 150.56}


def column_relation() -> None:
    print("1. Table 1: CR% against 100*ln(1 + ARR%/100)")
    matches = total = 0
    for ticker, rows in TABLE1.items():
        for model, (cr, arr) in rows.items():
            log_arr = 100 * math.log(1 + arr / 100)
            total += 1
            matches += abs(cr - log_arr) <= 0.015
            print("   {:5s} {:13s} CR {:6.2f}  ARR {:6.2f}  100*ln(1+ARR) {:6.2f}  diff {:+.2f}".format(
                ticker, model, cr, arr, log_arr, cr - log_arr))
    print("   cells where CR = 100*ln(1+ARR) within rounding (0.015): {} of {}".format(matches, total))
    for ticker, value in FIG_VALUE.items():
        ratio = value / 100000
        print("   {} figure broker value {:.2f}: simple {:.2f}%, log {:.2f}%".format(
            ticker, value, 100 * (ratio - 1), 100 * math.log(ratio)))


def buy_and_hold() -> None:
    print("\n2. Buy and hold, Yahoo dividend-adjusted (auto_adjust=True)")
    for ticker in TABLE1:
        bars = yf.download(ticker, start="2023-12-20", end="2024-04-10", auto_adjust=True, progress=False)
        bars.columns = [c[0] for c in bars.columns]
        window = bars.loc["2024-01-01":"2024-03-29"]
        last_close = window["Close"].iloc[-1]
        print("   {}: {} sessions, {} to {}".format(
            ticker, len(window), window.index[0].date(), window.index[-1].date()))
        variants = {
            "close 2023-12-29": bars.loc[:"2023-12-29", "Close"].iloc[-1],
            "close 2024-01-02": window["Close"].iloc[0],
            "open 2024-01-02": window["Open"].iloc[0],
            "open 2024-01-03": window["Open"].iloc[1],
        }
        for label, start in variants.items():
            r = last_close / start - 1
            print("     from {:17s} to close 2024-03-28: simple {:6.2f}%  log {:6.2f}%".format(
                label, 100 * r, 100 * math.log(1 + r)))
        for start in ("2023-12-29", "2024-01-02"):
            s = bars.loc[start:"2024-03-28", "Close"]
            r = s.pct_change().dropna()
            eq = s / s.iloc[0]
            print("     closes from {}: n={} Sharpe (sqrt 252, rf 0) {:.2f}, max drawdown {:.2f}%".format(
                start, len(r), math.sqrt(252) * r.mean() / r.std(ddof=1), 100 * float((1 - eq / eq.cummax()).max())))
        raw = yf.download(ticker, start="2024-03-27", end="2024-03-30", auto_adjust=False, progress=False)
        raw.columns = [c[0] for c in raw.columns]
        print("     raw bar 2024-03-28 O/H/L/C:", [round(float(raw.loc["2024-03-28", c]), 2)
                                                  for c in ("Open", "High", "Low", "Close")])


def adjustment_dates() -> None:
    print("\n3. Dividend adjustment implied by the figures' last close (2024-03-28)")
    for ticker, fig_close in FIG_LAST_CLOSE.items():
        raw = yf.download(ticker, start="2024-03-01", end="2026-11-01", auto_adjust=False, progress=False)
        raw.columns = [c[0] for c in raw.columns]
        raw.index = raw.index.tz_localize(None)
        dividends = yf.Ticker(ticker).dividends
        dividends.index = dividends.index.tz_localize(None)
        close = float(raw.loc["2024-03-28", "Close"])
        print("   {}: raw close {:.2f}, figure {:.2f}, ratio {:.5f}".format(ticker, close, fig_close, fig_close / close))
        factor = 1.0
        for day, amount in dividends.loc["2024-03-29":"2025-06-30"].items():
            previous = float(raw.loc[:day - timedelta(days=1), "Close"].iloc[-1])
            factor *= 1 - amount / previous
            print("     after ex-dividend {} ({:.2f}): adjusted close {:.2f}".format(day.date(), amount, close * factor))


if __name__ == "__main__":
    column_relation()
    buy_and_hold()
    adjustment_dates()
