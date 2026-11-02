"""Monthly live report.

``write_monthly_report`` turns the live-track records (``<root>/live/<task>/<id>.json``)
and, when present, the ledger-track records (``<root>/ledger/<task>/<id>.json``)
into ``<root>/reports/<YYYY-MM>.md`` plus a figure and a small
``<YYYY-MM>.json`` summary that the next month's report diffs against.

A report covers the sessions up to and including the last day of its month, so
re-rendering a month later gives the same numbers (the same files byte for
byte); with no ``month`` the report is for the last completed UTC month. Every
number is computed from the equity curves in the records. A record's window
starts at its own publication (live) or first verified decision (ledger), so
rows of one task can have different start dates; each row is compared with
the 1/N benchmark over its own window. Strategies with several seeds are
scored on the equal-weight average of their seeds' daily returns, as on the
leaderboard.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
from pathlib import Path

import numpy as np

from rl.bench import scoring
from rl.bench.runner import READABLE_SCHEMAS
from rl.bench.submission import SUBMISSIONS_DIR, discover
from rl.bench.tasks import TASKS

log = logging.getLogger(__name__)

TRACKS = ("live", "ledger")
REPORTS_DIR = "reports"
#: Sharpe ratios are only reported from this many scored sessions.
MIN_SHARPE_SESSIONS = 20
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# --- data ------------------------------------------------------------------


def _previous_month(today: dt.date) -> str:
    last = today.replace(day=1) - dt.timedelta(days=1)
    return "{:04d}-{:02d}".format(last.year, last.month)


def _month_end(month: str) -> str:
    year, number = int(month[:4]), int(month[5:])
    first_of_next = dt.date(year + (number == 12), number % 12 + 1, 1)
    return str(first_of_next - dt.timedelta(days=1))


def _load(root: Path) -> list[dict]:
    records = []
    for track in TRACKS:
        for path in sorted((root / track).glob("*/*.json")):
            record = json.loads(path.read_text())
            if record.get("schema") not in READABLE_SCHEMAS:
                raise ValueError("{} has unsupported schema {!r}".format(path, record.get("schema")))
            records.append(record)
    return records


def _drawdown(equity: np.ndarray) -> float:
    return float(np.min(equity / np.maximum.accumulate(equity) - 1))


def _entry(record: dict, as_of: str) -> dict | None:
    """The record cut at ``as_of``, or ``None`` if no session is scored by then."""
    count = sum(1 for day in record["dates"] if day <= as_of)
    if count < 2:
        return None
    runs = np.array([np.asarray(run["equity"][:count], dtype=float) for run in record["runs"]])
    returns = np.mean(runs[:, 1:] / runs[:, :-1] - 1, axis=0)
    equity = np.concatenate(([1.0], np.cumprod(1 + returns)))
    benchmark = np.asarray(record["benchmark"]["equity"][:count], dtype=float)
    benchmark = benchmark / benchmark[0]
    sessions = count - 1
    sharpe = None
    if sessions >= MIN_SHARPE_SESSIONS:
        value = scoring.annualised_sharpe(returns, record["periods_per_year"])
        sharpe = float(value) if math.isfinite(value) else None
    fit = record.get("fit") or {}
    entry = {
        "track": record["track"],
        "task": record["task"],
        "submission": record["submission"],
        "name": record["name"],
        "kind": record["kind"],
        "seeds": len(record["runs"]),
        "published": str(fit.get("published_at") or fit.get("first_decision") or record["dates"][0])[:10],
        "start": record["dates"][1],
        "last": record["dates"][count - 1],
        "sessions": sessions,
        "total_return": float(equity[-1] - 1),
        "benchmark_return": float(benchmark[-1] - 1),
        "vs_benchmark": float(equity[-1] - benchmark[-1]),
        "sharpe": sharpe,
        "max_drawdown": _drawdown(equity),
        "_dates": record["dates"][:count],
        "_equity": equity,
        "_benchmark": benchmark,
    }
    if record.get("cohort", 1) > 1:
        entry["cohort"] = record["cohort"]
    if record.get("code_changed"):
        entry["code_changed"] = True
    if record.get("policy_versions", 1) > 1:
        entry["policy_versions"] = record["policy_versions"]
    return entry


def _key(entry: dict) -> str:
    cohort = "/{}".format(entry["cohort"]) if entry.get("cohort") else ""
    return "{}/{}/{}{}".format(entry["track"], entry["task"], entry["submission"], cohort)


def _order(entry: dict) -> tuple:
    return (entry["task"], -entry["vs_benchmark"], entry["track"], entry["submission"])


def _public(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def _summary(month: str, as_of: str, entries: list[dict]) -> dict:
    return {
        "schema": 1,
        "month": month,
        "as_of": as_of,
        "entries": {_key(entry): _public(entry) for entry in sorted(entries, key=_key)},
    }


def _previous_summary(directory: Path, month: str) -> dict | None:
    """The most recent summary of an earlier month, if any."""
    earlier = sorted(p for p in directory.glob("????-??.json") if _MONTH.match(p.stem) and p.stem < month)
    return json.loads(earlier[-1].read_text()) if earlier else None


# --- rendering -------------------------------------------------------------


def _pct(value) -> str:
    return "{:+.1f}%".format(round(value * 100, 1) + 0.0)  # + 0.0: no "-0.0"


def _pp(value) -> str:
    return "{:+.1f} pp".format(round(value * 100, 1) + 0.0)


def _table(headers, rows) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
        *("| " + " | ".join(cells) + " |" for cells in rows),
    ]


def _title(task_id: str) -> str:
    task = TASKS.get(task_id)
    return task.title if task else task_id


def _label(entry: dict) -> str:
    return (
        entry["name"]
        + (" (ledger)" if entry["track"] == "ledger" else "")
        + (" (cohort {})".format(entry["cohort"]) if entry.get("cohort") else "")
        + (" (code changed)" if entry.get("code_changed") else "")
        + (" (decisions span {} policy versions)".format(entry["policy_versions"]) if entry.get("policy_versions") else "")
    )


def _where(entry: dict) -> str:
    return "{}: {}".format(entry["task"], _label(entry))


def _findings(entries: list[dict]) -> list[str]:
    ahead = [e for e in entries if e["vs_benchmark"] > 0]
    best = max(entries, key=lambda e: e["vs_benchmark"])
    worst = min(entries, key=lambda e: e["vs_benchmark"])

    def line(prefix: str, entry: dict) -> str:
        return "{}: `{}` {:+.1f} pp ({} vs {} for 1/N, {} sessions).".format(
            prefix, _where(entry), entry["vs_benchmark"] * 100, _pct(entry["total_return"]),
            _pct(entry["benchmark_return"]), entry["sessions"],
        )

    findings = [
        "{} of {} strategy-task pairs are ahead of 1/N since publication.".format(len(ahead), len(entries)),
        line("Best versus 1/N", best),
    ]
    if len(entries) > 1:
        findings.append(line("Worst versus 1/N", worst))
    else:
        findings.append("Only one strategy-task pair is scored so far.")
    return findings


def _changes(entries: list[dict], previous: dict) -> list[str]:
    before = previous["entries"]
    rows = []
    for entry in sorted(entries, key=_order):
        old = before.get(_key(entry))
        now = [_pct(entry["total_return"]), _pp(entry["vs_benchmark"])]
        if old is None or old["start"] != entry["start"]:
            # Absent last time, or re-frozen and re-published since: no like-for-like comparison.
            rows.append([_where(entry), "{} (new)".format(entry["sessions"]), *now, "–"])
        else:
            rows.append(
                [
                    _where(entry),
                    "{} → {}".format(old["sessions"], entry["sessions"]),
                    "{} → {}".format(_pct(old["total_return"]), now[0]),
                    "{} → {}".format(_pp(old["vs_benchmark"]), now[1]),
                    _pp(entry["vs_benchmark"] - old["vs_benchmark"]),
                ]
            )
    return [
        "## Change since {}".format(previous["month"]),
        "",
        *_table(["Strategy", "Sessions", "Return", "vs 1/N", "Δ vs 1/N"], rows),
        "",
    ]


def render_markdown(month: str, as_of: str, entries: list[dict], previous: dict | None, figure: str | None) -> str:
    lines = ["# Live report {}".format(month), ""]
    if not entries:
        lines += [
            "No sessions scored yet: no live or ledger record has a scored session on or before {}.".format(as_of),
            "",
            "Full tables: [LEADERBOARD.md](../LEADERBOARD.md).",
            "",
        ]
        return "\n".join(lines)

    tasks = sorted({e["task"] for e in entries})
    lines += [
        "Window: {} (earliest publication or first decision) → {} (last session scored). "
        "Sessions scored: {}. {} strategy-task pairs, {} task{}.".format(
            min(e["published"] for e in entries),
            max(e["last"] for e in entries),
            "{}–{}".format(min(e["sessions"] for e in entries), max(e["sessions"] for e in entries))
            if len({e["sessions"] for e in entries}) > 1
            else str(entries[0]["sessions"]),
            len(entries),
            len(tasks),
            "" if len(tasks) == 1 else "s",
        ),
        "",
        "Full tables: [LEADERBOARD.md](../LEADERBOARD.md).",
        "",
        "## Findings",
        "",
        *("- " + finding for finding in _findings(entries)),
        "",
    ]
    if figure:
        lines += ["![Live equity curves, 1/N dashed]({})".format(figure), ""]
    for task in tasks:
        chosen = sorted((e for e in entries if e["task"] == task), key=_order)
        lines += [
            "## {} — `{}`".format(_title(task), task),
            "",
            *_table(
                ["Strategy", "Kind", "From", "Return", "1/N", "vs 1/N", "Sharpe", "Max DD"],
                [
                    [
                        _label(e),
                        e["kind"],
                        e["start"],
                        _pct(e["total_return"]),
                        _pct(e["benchmark_return"]),
                        _pp(e["vs_benchmark"]),
                        "{:.2f}".format(e["sharpe"]) if e["sharpe"] is not None
                        else "– ({} sessions)".format(e["sessions"]) if e["sessions"] < MIN_SHARPE_SESSIONS
                        else "– (no variance)",
                        "{:.1f}%".format(e["max_drawdown"] * 100),
                    ]
                    for e in chosen
                ],
            ),
            "",
        ]
    if previous is not None:
        lines += _changes(entries, previous)
    lines += [
        "*Return* is since the first scored session after publication (live) or after the first verified "
        "decision (ledger), net of fees. *1/N* is equal-weight buy and hold over the same sessions and "
        "*vs 1/N* is the difference of the two returns. *Sharpe* (daily returns, zero risk-free rate, "
        "annualised) is shown from {} sessions. *Max DD* is peak-to-trough including the start.".format(
            MIN_SHARPE_SESSIONS
        ),
        "",
    ]
    return "\n".join(lines)


def _figure(entries: list[dict], path: Path, month: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({e["task"] for e in entries})
    columns = min(2, len(tasks))
    grid_rows = -(-len(tasks) // columns)
    fig, axes = plt.subplots(grid_rows, columns, figsize=(6.2 * columns, 3.6 * grid_rows), squeeze=False)
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    for ax, task in zip(axes.flat, tasks):
        chosen = sorted((e for e in entries if e["task"] == task), key=_order)
        drawn = set()
        for number, entry in enumerate(chosen):
            days = np.array(entry["_dates"], dtype="datetime64[D]")
            color = "C{}".format(number % 10)
            ax.plot(days, entry["_equity"], color=color, linewidth=1.4, label=_label(entry))
            # One dashed 1/N line per distinct window start (rows can start on different days).
            if entry["start"] not in drawn:
                drawn.add(entry["start"])
                ax.plot(
                    days, entry["_benchmark"], color="black", linestyle="--", linewidth=1,
                    label="1/N buy and hold" if len(drawn) == 1 else None,
                )
        ax.axhline(1.0, color="grey", linewidth=0.5)
        ax.set_title(_title(task), fontsize=9)
        ax.set_ylabel("equity (1 = window start)", fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Live equity curves since publication, {}".format(month), fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_monthly_report(
    root="leaderboard", month: str | None = None, *, now: dt.datetime | None = None, submissions_root=SUBMISSIONS_DIR
) -> Path:
    """Write ``<root>/reports/<month>.md`` (+ ``.png``, ``.json``) and return the markdown path.

    ``month`` is ``YYYY-MM``; the default is the last completed UTC month.
    Sessions after the month's last day are ignored. Re-running is idempotent.
    Submissions under ``submissions_root`` that are retired today are left out.
    """
    root = Path(root)
    if month is None:
        month = _previous_month((now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).date())
    if not _MONTH.match(month):
        raise ValueError("month must look like YYYY-MM, got {!r}".format(month))
    as_of = _month_end(month)
    retired = {s.id for s in discover(submissions_root).values() if s.is_retired()}
    records = [record for record in _load(root) if record["submission"] not in retired]
    entries = [e for e in (_entry(record, as_of) for record in records) if e is not None]
    entries.sort(key=_order)

    directory = root / REPORTS_DIR
    previous = _previous_summary(directory, month)
    figure_path = directory / (month + ".png")
    figure = None
    if entries:
        _figure(entries, figure_path, month)
        figure = figure_path.name
    elif figure_path.exists():
        figure_path.unlink()
    path = directory / (month + ".md")
    _write_text(path, render_markdown(month, as_of, entries, previous, figure))
    _write_text(directory / (month + ".json"), json.dumps(_summary(month, as_of, entries), indent=1) + "\n")
    log.info("Wrote %s (%d strategy-task pairs)", path, len(entries))
    return path


# --- CLI -------------------------------------------------------------------


def _cmd_monthly(options) -> int:
    path = write_monthly_report(options.out_root, options.month, submissions_root=options.submissions_root)
    print(path.read_text())
    print("Wrote {}".format(path))
    return 0


def add_cli(subparsers, common) -> None:
    """Register ``monthly`` on the bench subparsers; its ``handler`` default runs it."""
    parser = subparsers.add_parser(
        "monthly",
        parents=[common],
        help="render leaderboard/reports/<YYYY-MM>.md from the live (and ledger) records",
    )
    parser.add_argument("--month", help="YYYY-MM (default: the last completed UTC month)")
    parser.set_defaults(handler=_cmd_monthly)
