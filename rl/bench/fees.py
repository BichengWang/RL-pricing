"""Fee sensitivity of the historical track, from the committed records.

``write_fees`` re-prices every historical record on the ranked tasks and the
re-evaluations (``<root>/historical/<task>/<id>.json``) at the fees in
:data:`FEES_BP` and the task's own fee, with :func:`rl.bench.scoring.reprice`,
and writes ``<root>/fees.json`` and ``<root>/FEES.md``. 1/N buy-and-hold is
re-priced the same way from its own record's turnover.

As on the leaderboard, an entry is scored on the equal-weight average of its
seeds' daily returns: its Sharpe ratio, and its ΔSharpe against each record's
own 1/N benchmark with the one-sided paired stationary-bootstrap p-value of
:func:`rl.bench.scoring.sharpe_difference` (default resamples and seed, so a
rerun on the same records writes the same files byte for byte). At the
charged fee these equal the leaderboard's ΔSharpe and p. The break-even fee is
where the entry's Sharpe meets 1/N's, found by bisection on ``[0, 50]`` bp.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from rl.bench import leaderboard, scoring
from rl.bench.submission import SUBMISSIONS_DIR, discover
from rl.bench.tasks import TASKS

log = logging.getLogger(__name__)

#: Fees per side, in basis points, at which every entry is re-priced (plus the task's own).
FEES_BP = (0, 1, 2, 5, 10, 25, 50)
_BP = 1e-4
_BISECTIONS = 40
GROUPS = ("SB3 RL and LSTM", "Other learned", "Rule-based")


def _group(record: dict) -> str:
    if record["kind"] == "baseline":
        return GROUPS[2]
    if record["kind"] == "forecast" or record["policy"].startswith("rl.bench.policies.sb3:"):
        return GROUPS[0]
    return GROUPS[1]


def _averaged(runs, charged: float, fee: float) -> np.ndarray:
    return np.mean([scoring.reprice(equity, turnover, charged, fee) for equity, turnover in runs], axis=0)


def break_even_fee(runs, benchmark, charged: float, periods_per_year: float, high: float = FEES_BP[-1] * _BP):
    """Fee in ``[0, high]`` at which the seed-averaged Sharpe of ``runs`` equals ``benchmark``'s.

    ``runs`` is a sequence of ``(equity, turnover)`` pairs and ``benchmark``
    one such pair, all recorded at fee ``charged``; both sides are re-priced
    with :func:`rl.bench.scoring.reprice`. Returns ``None`` when the Sharpe
    gap has the same sign (or is undefined) at both ends: the entry trails the
    benchmark even at zero fees, or still leads it at ``high``.
    """

    def gap(fee: float) -> float:
        entry = scoring.annualised_sharpe(_averaged(runs, charged, fee), periods_per_year)
        return entry - scoring.annualised_sharpe(scoring.reprice(*benchmark, charged, fee), periods_per_year)

    lo, hi = 0.0, float(high)
    ahead = gap(lo)
    end = gap(hi)
    if not (math.isfinite(ahead) and math.isfinite(end)) or (ahead > 0) == (end > 0):
        return None
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2
        if (gap(mid) > 0) == (ahead > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _section(task_id: str, records: list[dict]) -> dict | None:
    bench = next((r for r in records if r["submission"] == leaderboard.BENCHMARK_SUBMISSION), None)
    if bench is None:
        log.warning("%s has no %s record; skipped", task_id, leaderboard.BENCHMARK_SUBMISSION)
        return None
    charged, ppy = bench["transaction_cost"], bench["periods_per_year"]
    for record in records:
        if (record["transaction_cost"], record["periods_per_year"], record["dates"]) != (charged, ppy, bench["dates"]):
            raise ValueError("{}/{} does not share the fee and sessions of 1/N".format(task_id, record["submission"]))
    charged_bp = round(charged / _BP, 6)
    fees_bp = sorted({float(f) for f in FEES_BP} | {charged_bp})
    bench_turnover = bench["runs"][0]["turnover"]
    bench_run = (bench["runs"][0]["equity"], bench_turnover)

    def turnover(record) -> float:
        return float(np.mean([np.mean(run["turnover"]) for run in record["runs"]]) * ppy)

    entries = []
    for record in records:
        if record is bench:
            continue
        runs = [(run["equity"], run["turnover"]) for run in record["runs"]]
        own = (record["benchmark"]["equity"], bench_turnover)
        sharpe, delta, p_value = [], [], []
        for bp in fees_bp:
            averaged = _averaged(runs, charged, bp * _BP)
            sharpe.append(scoring.annualised_sharpe(averaged, ppy))
            d, p = scoring.sharpe_difference(averaged, scoring.reprice(*own, charged, bp * _BP), ppy)
            delta.append(d)
            p_value.append(p)
        even = break_even_fee(runs, own, charged, ppy)
        entries.append(
            {
                "id": record["submission"],
                "name": record["name"],
                "kind": record["kind"],
                "group": _group(record),
                "seeds": len(runs),
                "turnover": turnover(record),
                "sharpe": sharpe,
                "delta_sharpe": delta,
                "p_value": p_value,
                "break_even_bp": None if even is None else even / _BP,
            }
        )
    at_charged = fees_bp.index(charged_bp)
    entries.sort(key=lambda e: (not math.isfinite(e["sharpe"][at_charged]), -np.nan_to_num(e["sharpe"][at_charged]), e["id"]))
    task = TASKS.get(task_id)
    return {
        "task": task_id,
        "title": task.title if task else task_id,
        "role": leaderboard._role(task_id),
        "first": bench["dates"][1],
        "last": bench["dates"][-1],
        "sessions": len(bench["dates"]) - 1,
        "periods_per_year": ppy,
        "charged_bp": charged_bp,
        "fees_bp": fees_bp,
        "benchmark": {
            "id": bench["submission"],
            "name": bench["name"],
            "turnover": turnover(bench),
            "sharpe": [scoring.annualised_sharpe(scoring.reprice(*bench_run, charged, bp * _BP), ppy) for bp in fees_bp],
        },
        "entries": entries,
    }


def _headline(sections: list[dict]) -> list[dict]:
    rows = []
    for fee in ("charged", *(float(f) for f in FEES_BP)):
        counts = {group: {"entries": 0, "ahead": 0, "significant": 0} for group in GROUPS}
        for section in sections:
            at = section["fees_bp"].index(section["charged_bp"] if fee == "charged" else fee)
            for entry in section["entries"]:
                if not math.isfinite(entry["delta_sharpe"][at]):
                    continue
                count = counts[entry["group"]]
                count["entries"] += 1
                count["ahead"] += entry["delta_sharpe"][at] > 0
                count["significant"] += entry["p_value"][at] < 0.05
        rows.append({"fee_bp": fee, "groups": counts})
    return rows


def build(root="leaderboard", retired=()) -> dict:
    """Re-price the historical records under ``root`` (except ``retired`` submissions) into the fees summary."""
    by_task: dict[str, list[dict]] = {}
    for record in leaderboard.load_records(root):
        if record["track"] == "historical" and record["submission"] not in retired:
            if leaderboard._role(record["task"]) != "demo":
                by_task.setdefault(record["task"], []).append(record)
    order = sorted(by_task, key=lambda t: (leaderboard._ROLE_ORDER[leaderboard._role(t)], t))
    sections = [s for s in (_section(task_id, by_task[task_id]) for task_id in order) if s is not None]
    return {"schema": 1, "fees_bp": [float(f) for f in FEES_BP], "headline": _headline(sections), "sections": sections}


# --- rendering -------------------------------------------------------------


def _bp(value: float) -> str:
    return "{:g} bp".format(value)


def _delta(delta, p_value) -> str:
    if not math.isfinite(delta):
        return "–"
    return "{:+.2f} ({})".format(delta, "p<0.001" if p_value < 0.001 else "p={:.3f}".format(p_value))


def _break_even(entry: dict) -> str:
    if entry["break_even_bp"] is not None:
        return "{:.1f} bp".format(entry["break_even_bp"])
    if not math.isfinite(entry["delta_sharpe"][0]):
        return "–"
    return "> {}".format(_bp(FEES_BP[-1])) if entry["delta_sharpe"][0] > 0 else "never"


def render_markdown(summary: dict) -> str:
    """The full ``FEES.md``."""
    sections = summary["sections"]
    charged = sorted({s["charged_bp"] for s in sections})
    members = {group: set() for group in GROUPS}
    for section in sections:
        for entry in section["entries"]:
            if math.isfinite(entry["delta_sharpe"][0]):
                members[entry["group"]].add(entry["name"])
    lines = [
        "# Fee sensitivity",
        "",
        "How much of the historical track's verdict is the fee? Every historical record is re-priced at fees "
        "from 0 to {} per side, and so is 1/N buy-and-hold. Written by `rl-pricing bench fees` from "
        "[`historical/`](historical/); every number is also in [`fees.json`](fees.json).".format(_bp(FEES_BP[-1])),
        "",
        "**Method.** Each record keeps, per seed and session, equity net of the charged fee *c* and turnover "
        "*τ* (traded notional over pre-trade equity). A fill leaves post-fee equity *B*(1 − *cτ*), so at fee "
        "*c′* a session's return becomes 1 + *r′* = (1 + *r*)(1 − *c′τ*)/(1 − *cτ*). "
        "Decisions are held fixed, which is what these policies would do at another fee: they decide from "
        "prices and their current weights, and the engine's weights do not depend on the fee, so the one "
        "approximation is that *τ* keeps its charged-fee value (exact at 0 bp; the error grows with turnover "
        "and fee, and `tests/test_bench_fees.py` holds it within 0.01 Sharpe of exact engine runs). "
        "This is not retraining: the learned policies were trained and selected on returns net of the charged "
        "fee, and a policy trained at another fee could trade differently.",
        "",
        "*Sharpe* is that of the seed-averaged daily returns (the basis of the leaderboard's ΔSharpe, so it can "
        "differ from the leaderboard's mean per-seed Sharpe). *ΔSharpe* is that minus 1/N's, with the "
        "leaderboard's one-sided paired stationary-bootstrap p-value (uncorrected for the many comparisons). "
        "*Break-even*: the fee at which the entry's Sharpe equals 1/N's; *never* means it trails 1/N even at "
        "0 bp, *> {}* that it leads at every fee shown.".format(_bp(FEES_BP[-1])),
        "",
        "## Entries ahead of 1/N",
        "",
        "Entries with a higher Sharpe ratio than 1/N at each fee, out of all entries in the group, over the "
        "ranked tasks and the re-evaluations ({}); in brackets, how many of them have p<0.05.".format(
            ", ".join("`{}`".format(s["task"]) for s in sections)
        ),
        "",
    ]
    rows = []
    for row in summary["headline"]:
        if row["fee_bp"] == "charged":
            label = "as charged ({})".format(" or ".join(_bp(c) for c in charged))
        else:
            label = _bp(row["fee_bp"])
        rows.append(
            [label]
            + [
                "{} of {} ({})".format(c["ahead"], c["entries"], c["significant"])
                for c in (row["groups"][group] for group in GROUPS)
            ]
        )
    lines += leaderboard._table(["Fee per side", *GROUPS], rows) + [""]
    lines += [
        "Groups: {}. Constant-return entries (cash) have no Sharpe ratio and are left out.".format(
            "; ".join("*{}*: {}".format(group, ", ".join(sorted(members[group]))) for group in GROUPS if members[group])
        ),
        "",
        "## By task",
        "",
        "Fee columns give ΔSharpe vs 1/N (one-sided p) at that fee; the ◆ row gives 1/N's own Sharpe, so an "
        "entry's Sharpe is the ◆ value plus its ΔSharpe. Rows follow the Sharpe ratio at the charged fee.",
        "",
    ]
    for section in sections:
        title = section["title"]
        if section["role"] == "reevaluation":
            title = "Re-evaluation: " + title
        fees = [_bp(f) + (" (charged)" if f == section["charged_bp"] else "") for f in section["fees_bp"]]
        bench = section["benchmark"]
        table = [
            ["**{}** ◆".format(bench["name"]), "baseline", leaderboard._fmt(bench["turnover"], "{:.1f}×")]
            + [leaderboard._fmt(s, "{:.2f}") for s in bench["sharpe"]]
            + ["–"]
        ]
        for entry in section["entries"]:
            table.append(
                [entry["name"], entry["kind"], leaderboard._fmt(entry["turnover"], "{:.1f}×")]
                + [_delta(d, p) for d, p in zip(entry["delta_sharpe"], entry["p_value"])]
                + [_break_even(entry)]
            )
        lines += [
            "### {} — `{}`".format(title, section["task"]),
            "",
            "{} to {} ({} sessions) · fee charged {} per side".format(
                section["first"], section["last"], section["sessions"], _bp(section["charged_bp"])
            ),
            "",
            *(
                ["*Survivorship-biased universe: a re-evaluation of a published setup, not a ranking.*", ""]
                if section["role"] == "reevaluation"
                else []
            ),
            *leaderboard._table(["Strategy", "Kind", "Turnover/yr", *fees, "Break-even"], table),
            "",
        ]
    return "\n".join(lines)


def _rounded(value):
    """``value`` with floats rounded to 4 decimals and undefined ones as ``None``, for the JSON."""
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def write_fees(root="leaderboard", *, submissions_root=SUBMISSIONS_DIR) -> dict:
    """Write ``fees.json`` and ``FEES.md`` under ``root``; retired submissions are left out."""
    root = Path(root)
    retired = {s.id for s in discover(submissions_root).values() if s.is_retired()}
    summary = build(root, retired)
    (root / "fees.json").write_text(json.dumps(_rounded(summary), indent=1, ensure_ascii=False) + "\n")
    (root / "FEES.md").write_text(render_markdown(summary))
    return summary


def _cmd_fees(options) -> int:
    summary = write_fees(options.out_root, submissions_root=options.submissions_root)
    print((Path(options.out_root) / "FEES.md").read_text())
    print("{} tasks written to {}".format(len(summary["sections"]), options.out_root))
    return 0


def add_cli(subparsers, common) -> None:
    """Register ``fees`` on the bench subparsers; its ``handler`` default runs it."""
    parser = subparsers.add_parser(
        "fees",
        parents=[common],
        help="re-price the historical records at other fees: render FEES.md and fees.json",
    )
    parser.set_defaults(handler=_cmd_fees)
