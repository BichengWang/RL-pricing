"""Benchmark tasks: a fixed universe, data source, splits and cost model.

Universes are chosen by rules evaluated before the data starts, so no task
selects assets by later survival:

* ``us-sectors``: the nine original Select Sector SPDR ETFs, all listed
  since December 1998.
* ``us-multi-asset``: ten liquid US-listed ETFs spanning equities, bonds,
  gold and real estate, all listed by November 2004.
* ``crypto-majors``: the ten Binance spot USDT pairs with the highest quote
  volume from April to June 2018, excluding stablecoins. Two were later
  delisted (BCCUSDT in November 2018, EOSUSDT in May 2025); the engine sells
  delisted holdings at their last close.

One task deliberately breaks that rule:

* ``paper-2021``: the 20 large-cap stocks, dates and test window of Wang &
  Zhang (2021). The universe was chosen in 2021 by market capitalisation, so
  it is survivorship-biased; the task exists to re-evaluate that paper's
  setup with multiple seeds and the benchmark's execution model, not to rank
  strategies.

Three more tasks are single-asset reproduction tasks, not ranked universes:

* ``us-aapl``, ``us-googl``, ``us-amzn``: one stock each, with the test
  window of the TradingAgents paper (Xiao et al., arXiv 2412.20138, Table 1:
  2024-01-01 to 2024-03-29, whose last session is 2024-03-28). The tickers
  are the paper's and were large caps long before the window, but a
  one-asset "universe" is a survivorship-biased pick by construction: the
  benchmark is that stock bought and held, and a one-asset score says
  nothing about a strategy's breadth. They exist so the paper's claims can
  be re-tested with this engine's execution model and followed forward on
  the live decision ledger. They have no frozen-policy live track
  (``live=False``), so merges don't freeze every submission on them.

Historical splits: train on ``[start, validation_start)``, select models on
``[validation_start, test_start)`` and report ``[test_start, test_end)``.
The live track instead fits on all data up to a freeze date, holding out the
same validation length, and reports every later session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.bench.market import as_day
from rl.bench.info import validate_sources


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    source: str
    tickers: tuple[str, ...]
    start: str
    validation_start: str
    test_start: str
    test_end: str
    transaction_cost: float
    periods_per_year: int
    description: str
    #: Whether the task has a live forward-test track (frozen submissions
    #: scored on sessions after their freeze date).
    live: bool = True
    info_sources: tuple[dict, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "info_sources", validate_sources(self.info_sources))
        if self.source not in ("yahoo", "binance", "synthetic"):
            raise ValueError("Unknown data source {!r}".format(self.source))
        days = [as_day(d) for d in (self.start, self.validation_start, self.test_start, self.test_end)]
        if not all(a < b for a, b in zip(days, days[1:])):
            raise ValueError("Task {} needs start < validation_start < test_start < test_end".format(self.id))

    @property
    def validation_length(self) -> np.timedelta64:
        return as_day(self.test_start) - as_day(self.validation_start)


TASKS = {
    task.id: task
    for task in (
        Task(
            id="us-sectors",
            title="US sector rotation",
            source="yahoo",
            tickers=("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"),
            start="1999-01-01",
            validation_start="2015-01-01",
            test_start="2018-01-01",
            test_end="2026-01-01",
            transaction_cost=0.0005,
            periods_per_year=252,
            description="Daily long-only allocation across the nine original SPDR sector ETFs.",
        ),
        Task(
            id="us-multi-asset",
            title="US multi-asset allocation",
            source="yahoo",
            tickers=("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ"),
            start="2005-01-01",
            validation_start="2015-01-01",
            test_start="2018-01-01",
            test_end="2026-01-01",
            transaction_cost=0.0005,
            periods_per_year=252,
            description="Daily long-only allocation across equity, bond, gold and REIT ETFs.",
        ),
        Task(
            id="crypto-majors",
            title="Crypto majors",
            source="binance",
            tickers=(
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "EOSUSDT", "BCCUSDT",
                "NEOUSDT", "ADAUSDT", "LTCUSDT", "XRPUSDT", "ONTUSDT",
            ),
            start="2018-07-01",
            validation_start="2022-01-01",
            test_start="2023-01-01",
            test_end="2026-01-01",
            transaction_cost=0.001,
            periods_per_year=365,
            description="Daily (UTC) long-only allocation across a point-in-time top-10 Binance USDT universe.",
        ),
        Task(
            id="paper-2021",
            title="Wang & Zhang (2021) setup",
            source="yahoo",
            tickers=(
                "AAPL", "MSFT", "AMZN", "BRK-B", "JPM", "JNJ", "UNH", "HD", "PG", "NVDA",
                "DIS", "BAC", "CMCSA", "XOM", "VZ", "T", "ADBE", "PFE", "CSCO", "INTC",
            ),
            start="2000-01-01",
            validation_start="2017-01-01",
            test_start="2019-01-01",
            test_end="2021-01-01",
            transaction_cost=0.001,
            periods_per_year=252,
            description="The 2021 paper's 20 large caps and 2019-2020 test; survivorship-biased by design.",
            live=False,
        ),
        Task(
            id="us-aapl",
            title="AAPL (TradingAgents paper window)",
            source="yahoo",
            tickers=("AAPL",),
            start="2015-01-01",
            validation_start="2023-07-01",
            test_start="2024-01-01",
            test_end="2024-03-30",
            transaction_cost=0.0005,
            periods_per_year=252,
            description="Single-asset long/flat on Apple with the TradingAgents paper's 2024 Q1 test window; a reproduction task, not a ranked universe.",
            live=False,
        ),
        Task(
            id="us-googl",
            title="GOOGL (TradingAgents paper window)",
            source="yahoo",
            tickers=("GOOGL",),
            start="2015-01-01",
            validation_start="2023-07-01",
            test_start="2024-01-01",
            test_end="2024-03-30",
            transaction_cost=0.0005,
            periods_per_year=252,
            description="Single-asset long/flat on Alphabet (class A) with the TradingAgents paper's 2024 Q1 test window; a reproduction task, not a ranked universe.",
            live=False,
        ),
        Task(
            id="us-amzn",
            title="AMZN (TradingAgents paper window)",
            source="yahoo",
            tickers=("AMZN",),
            start="2015-01-01",
            validation_start="2023-07-01",
            test_start="2024-01-01",
            test_end="2024-03-30",
            transaction_cost=0.0005,
            periods_per_year=252,
            description="Single-asset long/flat on Amazon with the TradingAgents paper's 2024 Q1 test window; a reproduction task, not a ranked universe.",
            live=False,
        ),
        Task(
            id="synthetic",
            title="Synthetic demo",
            source="synthetic",
            tickers=("AAA", "BBB", "CCC", "DDD", "EEE"),
            start="2010-01-01",
            validation_start="2016-01-01",
            test_start="2018-01-01",
            test_end="2020-01-01",
            transaction_cost=0.001,
            periods_per_year=252,
            description="Correlated random walks for offline runs; no strategy has an edge by construction.",
            live=False,
        ),
    )
}


def get_task(task_id: str) -> Task:
    try:
        return TASKS[task_id]
    except KeyError:
        raise KeyError("Unknown task {!r}; choose from {}".format(task_id, sorted(TASKS))) from None
