import json
import math

import numpy as np
import pandas as pd
import pytest

from rl.bench.data import load_task_data
from rl.bench.engine import simulate
from rl.bench.fees import _crossings, break_even_fee, build, render_markdown, write_fees
from rl.bench.policies.baselines import Momentum
from rl.bench.scoring import annualised_sharpe, reprice
from rl.bench.tasks import TASKS, get_task
from rl.trade.execution import rebalance_positions

TASK = get_task("synthetic")


def _churn(view):
    """Trades most of the book every session, from its current weights and the date."""
    rng = np.random.default_rng(len(view.history))
    return 0.3 * np.roll(view.weights, 1) + 0.7 * rng.dirichlet(np.ones(len(view.tickers)))


@pytest.mark.parametrize("make, min_turnover", [(lambda: Momentum().act, 1), (lambda: _churn, 150)])
def test_repriced_returns_match_exact_engine_runs_at_other_fees(make, min_turnover):
    data = load_task_data(TASK)
    start = int(np.searchsorted(data.dates, np.datetime64(TASK.test_start)))
    charged = TASK.transaction_cost
    base = simulate(data, make(), start, charged)
    assert base.turnover.mean() * TASK.periods_per_year > min_turnover
    for fee in (0.0, 0.0001, 0.0005, 0.0025, 0.005):
        exact = simulate(data, make(), start, fee)
        # The weights a policy sees don't depend on the fee, so it makes the same decisions.
        np.testing.assert_allclose(exact.weights, base.weights, rtol=0, atol=1e-12)
        repriced = annualised_sharpe(reprice(base.equity, base.turnover, charged, fee), TASK.periods_per_year)
        tolerance = 1e-9 if fee == 0 else 0.01
        assert repriced == pytest.approx(annualised_sharpe(exact.returns, TASK.periods_per_year), abs=tolerance)
    assert np.array_equal(reprice(base.equity, base.turnover, charged, charged), base.returns)


def test_break_even_fee_is_where_the_extra_return_is_spent():
    # Entry: the benchmark's returns times (1 + edge) per session, trading `turnover` a session.
    # At fee f its gross factor is (1 + edge)(1 - f turnover) / (1 - c turnover); Sharpe ties at factor 1.
    benchmark = np.random.default_rng(0).normal(0.0004, 0.01, 750)
    charged, turnover = 0.001, np.full(750, 0.02)
    bench = (np.cumprod(np.r_[1.0, 1 + benchmark]), np.zeros(750))

    def runs(edge):
        return [(np.cumprod(np.r_[1.0, (1 + benchmark) * (1 + edge)]), turnover)]

    edge = 0.00005
    expected = (1 - (1 - charged * 0.02) / (1 + edge)) / 0.02  # about 35 bp
    assert break_even_fee(runs(edge), bench, charged, 252) == pytest.approx(expected, abs=1e-10)
    assert break_even_fee(runs(-0.0001), bench, charged, 252) is None  # trails even at 0 bp
    assert break_even_fee(runs(0.001), bench, charged, 252) is None  # still leads at 50 bp


# --- the fee table: build(), write_fees() and the markdown ----------------

CHARGED, TURNOVER, SESSIONS = 0.001, 0.1, 60
# Daily returns of 1/N: deterministic, not constant.
BENCH_RETURNS = np.tile([0.01, -0.004, 0.006, -0.002], SESSIONS // 4)


def _equity(returns):
    return [float(x) for x in np.cumprod(np.r_[1.0, 1 + np.asarray(returns)])]


def _record(submission, name, kind, edge, turnover=TURNOVER, task="us-sectors", **overrides):
    """A historical record whose returns are 1/N's times (1 + edge), trading ``turnover`` a session."""
    returns = (1 + BENCH_RETURNS) * (1 + edge) - 1
    record = {
        "schema": 1, "task": task, "track": "historical", "submission": submission, "name": name,
        "kind": kind, "policy": "rl.bench.policies.sb3:SB3Policy" if kind == "rl" else "x:Policy",
        "transaction_cost": CHARGED, "periods_per_year": 252,
        "dates": [str(d) for d in pd.bdate_range("2020-01-01", periods=SESSIONS + 1).date],
        "runs": [{"seed": 0, "equity": _equity(returns), "turnover": [turnover] * SESSIONS}],
        "benchmark": {"name": "1/N", "equity": _equity(BENCH_RETURNS)},
    }
    return {**record, **overrides}


def _write(root, *records):
    for record in records:
        directory = root / record["track"] / record["task"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (record["submission"] + ".json")).write_text(json.dumps(record))


@pytest.fixture
def fee_root(tmp_path):
    """1/N plus four entries: one that breaks even at 20 bp, one that never leads, one that always does, cash."""
    # At fee f an entry's gross factor is (1 + edge)(1 - f turnover)/(1 - charged turnover); it ties 1/N at 1.
    edge_20bp = (1 - CHARGED * TURNOVER) / (1 - 0.002 * TURNOVER) - 1
    flat = [1.0] * (SESSIONS + 1)
    root = tmp_path / "leaderboard"
    _write(
        root,
        _record("equal-weight-buy-hold", "1/N", "baseline", 0.0, turnover=0.0),
        _record("a2c", "A2C", "rl", edge_20bp),
        _record("trailer", "Trailer", "baseline", -0.0005),
        _record("leader", "Leader", "llm", 0.01),
        _record("cash", "Cash", "baseline", 0.0, turnover=0.0, benchmark={"name": "1/N", "equity": flat},
                runs=[{"seed": 0, "equity": flat, "turnover": [0.0] * SESSIONS}]),
    )
    return root


def test_build_finds_the_break_even_fee_and_crossings_of_each_entry(fee_root):
    [section] = build(fee_root)["sections"]
    assert (section["task"], section["role"], section["charged_bp"]) == ("us-sectors", "ranked", 10.0)
    assert section["fees_bp"] == [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    assert (section["first"], section["last"], section["sessions"]) == ("2020-01-02", "2020-03-25", SESSIONS)
    assert len(set(section["benchmark"]["sharpe"])) == 1  # 1/N trades nothing, so its Sharpe ignores the fee
    entries = {e["id"]: e for e in section["entries"]}
    assert entries["a2c"]["break_even_bp"] == pytest.approx(20.0, abs=1e-6)
    assert entries["a2c"]["turnover"] == pytest.approx(TURNOVER * 252)
    # Ahead of 1/N through 10 bp, behind from 25 bp: one sign change.
    assert [d > 0 for d in entries["a2c"]["delta_sharpe"]] == [True] * 5 + [False] * 2
    assert _crossings(entries["a2c"]) == 1
    assert entries["trailer"]["break_even_bp"] is None and all(d < 0 for d in entries["trailer"]["delta_sharpe"])
    assert entries["leader"]["break_even_bp"] is None and all(d > 0 for d in entries["leader"]["delta_sharpe"])
    assert not math.isfinite(entries["cash"]["sharpe"][0]) and entries["cash"]["break_even_bp"] is None
    # Rows follow the Sharpe ratio at the charged fee; cash has none and goes last.
    assert [e["id"] for e in section["entries"]] == ["leader", "a2c", "trailer", "cash"]
    assert {e["id"]: e["group"] for e in section["entries"]} == {
        "a2c": "SB3 RL and LSTM", "leader": "Other learned", "trailer": "Rule-based", "cash": "Rule-based"}


def test_build_counts_entries_ahead_of_one_over_n_per_group_and_fee(fee_root):
    headline = build(fee_root)["headline"]
    assert list(headline) == ["ranked"]
    rows = {row["fee_bp"]: row["groups"] for row in headline["ranked"]["rows"]}
    assert list(rows) == ["charged", 0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    for fee, ahead in {"charged": 1, 0.0: 1, 10.0: 1, 25.0: 0, 50.0: 0}.items():
        assert (rows[fee]["SB3 RL and LSTM"]["entries"], rows[fee]["SB3 RL and LSTM"]["ahead"]) == (1, ahead)
    # Always ahead, never ahead; cash (no Sharpe) is not counted.
    assert all(row["Other learned"]["ahead"] == 1 for row in rows.values())
    assert all(row["Rule-based"] == {"entries": 1, "ahead": 0, "significant": 0} for row in rows.values())


def test_build_leaves_out_retired_demo_and_non_historical_records_and_tasks_without_one_over_n(fee_root):
    _write(
        fee_root,
        _record("equal-weight-buy-hold", "1/N", "baseline", 0.0, turnover=0.0, task="synthetic"),
        _record("a2c", "A2C", "rl", 0.0, track="live"),
        _record("orphan", "Orphan", "rl", 0.0, task="crypto-majors"),  # no 1/N record: skipped with a warning
    )
    summary = build(fee_root, retired={"trailer", "cash"})
    assert [s["task"] for s in summary["sections"]] == ["us-sectors"]
    assert [e["id"] for e in summary["sections"][0]["entries"]] == ["leader", "a2c"]


def test_a_record_that_does_not_share_the_benchmarks_fee_is_rejected(fee_root):
    _write(fee_root, _record("odd", "Odd", "rl", 0.0, transaction_cost=0.0005))
    with pytest.raises(ValueError, match="us-sectors/odd does not share the fee and sessions of 1/N"):
        build(fee_root)


def test_write_fees_writes_the_json_and_markdown_that_leaderboard_publishes(fee_root, tmp_path):
    summary = write_fees(fee_root, submissions_root=tmp_path / "no-submissions")
    saved = json.loads((fee_root / "fees.json").read_text())
    assert saved["schema"] == 1 and saved["fees_bp"] == [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    entries = {e["id"]: e for e in saved["sections"][0]["entries"]}
    assert entries["a2c"]["break_even_bp"] == 20.0  # rounded to 4 decimals
    # Undefined floats are written as null, not NaN.
    assert entries["cash"]["sharpe"][0] is None and entries["cash"]["delta_sharpe"][0] is None
    assert (fee_root / "FEES.md").read_text() == render_markdown(summary)


def test_markdown_renders_each_break_even_case_and_the_headline_table(fee_root, tmp_path):
    write_fees(fee_root, submissions_root=tmp_path / "no-submissions")
    text = (fee_root / "FEES.md").read_text()
    lines = text.splitlines()
    assert lines[0] == "# Fee sensitivity"
    assert "**Ranked tasks**: `us-sectors`" in lines
    assert "| Fee per side | SB3 RL and LSTM | Other learned | Rule-based |" in lines
    charged = next(line for line in lines if line.startswith("| as charged (10 bp) |"))
    assert charged.startswith("| as charged (10 bp) | 1 of 1 (") and charged.endswith(" | 0 of 1 (0) |")
    assert next(line for line in lines if line.startswith("| 50 bp |")).startswith("| 50 bp | 0 of 1 (0) | 1 of 1 (")
    assert ("Groups: *SB3 RL and LSTM*: A2C; *Other learned*: Leader; *Rule-based*: Trailer. "
            "Constant-return entries (cash) have no Sharpe ratio and are left out.") in text
    assert "(no entry's ΔSharpe changes sign more than once on the fee grid)" in text
    assert "### {} — `us-sectors`".format(TASKS["us-sectors"].title) in lines
    assert "2020-01-02 to 2020-03-25 (60 sessions) · fee charged 10 bp per side" in lines
    row = {line.split(" | ")[0]: line for line in lines if line.startswith("| ")}
    assert row["| A2C"].startswith("| A2C | rl | 25.2× | +") and row["| A2C"].endswith(" | 20.0 bp |")
    assert row["| Trailer"].startswith("| Trailer | baseline | 25.2× | -") and row["| Trailer"].endswith(" | never |")
    assert row["| Leader"].endswith(" | > 50 bp |")
    assert row["| Cash"].endswith(" | – | – | – | – | – | – | – | – |")  # no Sharpe at any fee, so no break-even
    assert row["| **1/N** ◆"].startswith("| **1/N** ◆ | baseline | 0.0× | ")


def test_markdown_names_the_entries_whose_delta_sharpe_changes_sign_twice(fee_root):
    summary = build(fee_root)
    entry = summary["sections"][0]["entries"][1]
    assert entry["id"] == "a2c"
    entry["delta_sharpe"] = [1.0, -1.0, 1.0, math.nan, 1.0, 1.0, 1.0]  # the NaN is skipped, not a sign change
    assert _crossings(entry) == 2
    assert "ΔSharpe changes sign more than once for A2C on `us-sectors`" in render_markdown(summary)


# --- fee-adjusted rebalancing ----------------------------------------------


def test_rebalance_pays_the_fee_on_what_it_trades_out_of_post_fee_equity():
    # Sell half of a 1.0 holding into two equal targets at 1%: V + 0.01 (|0.5 V - 1| + |0.5 V|) = 1 gives V = 0.99.
    target, cash, cost = rebalance_positions(0.0, np.array([1.0, 0.0]), np.array([0.5, 0.5]), 0.01)
    np.testing.assert_allclose(target, [0.495, 0.495], rtol=0, atol=1e-15)
    assert cost == pytest.approx(0.01, abs=1e-15)
    assert cash == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("fee", [-0.001, float("nan"), float("inf")])
def test_rebalance_rejects_a_negative_or_non_finite_fee(fee):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rebalance_positions(0.0, np.array([1.0, 0.0]), np.array([0.5, 0.5]), fee)


@pytest.mark.parametrize(
    "weights, fee",
    [
        ([0.5, 0.5], 1.5),  # selling 1.0 costs 1.5: post-fee equity would be negative
        ([0.8, 0.2], 2.0),  # the cost of the targets grows faster than equity: no solution at all
    ],
)
def test_rebalance_with_a_fee_too_large_for_any_self_financing_portfolio_is_an_error(weights, fee):
    with pytest.raises(ValueError, match="no self-financing portfolio"):
        rebalance_positions(0.0, np.array([1.0, 0.0]), np.array(weights), fee)
