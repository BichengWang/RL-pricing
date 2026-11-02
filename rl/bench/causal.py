"""Reusable lookahead check: truncation and future poisoning on the shared engine.

A policy is fitted once on pre-test data and saved. Each replay loads a fresh
instance from that artifact (the path frozen live replay takes) and runs it
through :func:`rl.bench.engine.simulate`, so no second execution path exists.
The check sees what a policy *returns*; it cannot see side channels such as
files or the network.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from rl.bench import runner
from rl.bench.data import load_task_data
from rl.bench.engine import simulate
from rl.bench.info import InfoPanel, load_task_info, session_close
from rl.bench.market import FIELDS, MarketData, MarketView, as_day

CUTS = 3


def _replay(submission, task, saved: Path, seed, prices: MarketData, panel: InfoPanel, start: int, last: int) -> list:
    """Decisions at sessions ``start - 1`` through ``last`` of ``prices``."""
    policy = runner.build_policy(submission, task)
    policy.load(saved)
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)
    policy.reset()
    crypto = task.source == "binance"
    decisions = []

    def act(view):
        decision = policy.act(MarketView(view.history, view.weights, panel.at_session(view.date, crypto=crypto)))
        decisions.append(None if decision is None else np.array(decision, dtype=float))
        return decision

    result = simulate(prices, act, start, task.transaction_cost, stop=last + 1)
    # The engine stops one decision short of the last session: take it from the final mark.
    act(MarketView(prices.rows(0, last + 1), result.weights[-1]))
    return decisions


def _same(actual: list, expected: list) -> bool:
    return len(actual) == len(expected) and all(
        (a is None) == (b is None) and (a is None or np.array_equal(a, b)) for a, b in zip(actual, expected)
    )


def check_causal(task, submission, *, data: MarketData | None = None, info: InfoPanel | None = None, seed=0) -> list[str]:
    """Return the decision dates cut at; raise ``ValueError`` listing every failed check.

    *Truncation*: the test window is cut at several dates (the cut data is a
    fresh copy, so nothing later is reachable through array bases) and the
    decisions up to each cut must equal the full run's.
    *Future poisoning*: prices after each cut are scaled tenfold and information
    items from the cut's close on are replaced, with new items injected at and
    just after the close; decisions up to the cut must not change.
    """
    if data is None:
        data = load_task_data(task, end=task.test_end)
    data = data.select(task.tickers).between(None, task.test_end)
    if info is None:
        info = load_task_info(task, submission) if task.info_sources or submission.info_sources else InfoPanel()
    start = data.index_of(task.test_start)
    if not 1 <= start < len(data) - 1:
        raise ValueError("Causal checks need pre-test history and at least two test sessions")
    policy = runner.build_policy(submission, task)
    # A copy, so the training history shares no array with the later sessions.
    history = MarketData(data.dates[:start], data.tickers, **{f: getattr(data, f)[:start] for f in FIELDS})
    policy.fit(history, as_day(task.validation_start), seed)

    crypto = task.source == "binance"
    cuts = sorted({int(i) for i in np.linspace(start, len(data) - 2, CUTS + 2)[1:-1]})
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        saved = Path(directory)
        policy.save(saved)
        reference = _replay(submission, task, saved, seed, data, info, start, len(data) - 1)
        for cut in cuts:
            close = session_close(data.dates[cut], crypto=crypto)
            truncated = MarketData(
                data.dates[:cut + 1], data.tickers, **{f: getattr(data, f)[:cut + 1] for f in FIELDS}
            )
            scaled = {f: getattr(data, f).copy() for f in FIELDS}
            for name in ("open", "high", "low", "close"):
                scaled[name][cut + 1:] *= 10.0
            poisoned = MarketData(data.dates, data.tickers, **scaled)
            expected = reference[:cut - start + 2]
            for check, prices, panel in (
                ("truncation", truncated, info.truncated(close)),
                ("future poisoning", poisoned, info.poisoned(close, data.tickers)),
            ):
                try:
                    actual = _replay(submission, task, saved, seed, prices, panel, start, cut)
                except Exception as error:  # a policy that needs later data fails here, not only by differing
                    failures.append("{} at {}: {}: {}".format(check, data.dates[cut], type(error).__name__, error))
                else:
                    if not _same(actual, expected):
                        failures.append("{} at {}: decisions changed".format(check, data.dates[cut]))
    if failures:
        raise ValueError("{} is not causal on {}: {}".format(submission.id, task.id, "; ".join(failures)))
    return [str(data.dates[cut]) for cut in cuts]
