"""Aggregate result records into the leaderboard.

All metrics are recomputed from the equity curves stored in the records
(``leaderboard/<track>/<task>/<submission>.json``, schema 1 or 2):

* Per seed, :func:`rl.bench.scoring.metrics` against the record's own 1/N
  benchmark; the table shows the mean (and, for Sharpe, the standard
  deviation) across seeds.
* Significance uses the seed-averaged return series (the equal-weight
  portfolio of the seeds, which is what a user averaging seeds would get):
  ``ΔSharpe`` is its annualised Sharpe minus the benchmark's with a one-sided
  paired stationary-bootstrap p-value, and ``DSR`` is the deflated Sharpe
  ratio with ``trials`` taken from the submission.
* The DSR's ``sharpe_variance`` is the cross-sectional variance of the
  per-period Sharpe ratios of all records in the same table. With fewer than
  two records, or when that variance is below ``1/n`` (the sampling variance
  of a per-period Sharpe over ``n`` sessions under i.i.d. returns, which a
  handful of near-identical strategies can undercut by chance), ``1/n`` is used.

A live row whose frozen policy replayed on different code (``code_changed``)
is marked, and so is a ledger row whose decisions come from several versions
of the policy's own source (``policy_versions``). Once a live submission has
a later cohort, its earlier cohorts are labelled. Submissions with a
``retired`` date are not ranked; their records are listed apart.

Live results are grouped by freeze date and ledger results by the date of the
first verified decision, so every table covers the same window. Tasks are
ordered ranked first, then re-evaluations
(``paper-2021``: a survivorship-biased universe, shown for comparison with a
published setup, not as a ranking), then the synthetic demo.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import math
import re
from pathlib import Path

import numpy as np

from rl.bench import figures, scoring
from rl.bench.runner import READABLE_SCHEMAS
from rl.bench.submission import SUBMISSIONS_DIR, discover
from rl.bench.tasks import TASKS

log = logging.getLogger(__name__)

TRACKS = ("historical", "live", "ledger")
_TRACK_INTRO = {
    "historical": ("## Historical track", "Fixed public split: fit before the test window, then trade it."),
    "live": (
        "## Live track",
        "Policies frozen at the date shown and scored on the sessions after it; "
        "they cannot have been tuned on the results.",
    ),
    "ledger": (
        "## Ledger track: decisions committed before each open",
        "Each decision is committed to the repository by a scheduled workflow before the session it "
        "trades opens; entries whose workflow run cannot be verified, and sessions without an entry, "
        "count as holds.",
    ),
}
README_START = "<!-- leaderboard:start -->"
README_END = "<!-- leaderboard:end -->"
HEADLINE_ROWS = 3
REEVALUATIONS = ("paper-2021",)
#: The submission whose policy is every record's benchmark; its row is shown
#: as the benchmark instead of a duplicate synthesized row.
BENCHMARK_SUBMISSION = "equal-weight-buy-hold"
_METRICS = ("cagr", "max_drawdown", "turnover", "beta", "alpha", "alpha_t", "volatility", "total_return")
#: Submission kinds the README headline always represents by their best row.
LEARNED_KINDS = ("rl", "forecast", "llm")


def load_records(root="leaderboard") -> list[dict]:
    """Every schema 1 or 2 record under ``root/<track>/<task>/*.json``."""
    records = []
    for path in sorted(Path(root).glob("*/*/*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") not in READABLE_SCHEMAS:
            raise ValueError("{} has unsupported schema {!r}".format(path, record.get("schema")))
        records.append(record)
    return records


def _returns(equity) -> np.ndarray:
    equity = np.asarray(equity, dtype=float)
    return equity[1:] / equity[:-1] - 1


def _nan_safe(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ValueError as error:
        log.warning("%s unavailable: %s", getattr(function, "__name__", function), error)
        return None


def _per_period_sharpe(returns: np.ndarray) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std) if std > 0 else float("nan")


def _mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def _analyze_record(record: dict) -> dict:
    ppy = record["periods_per_year"]
    benchmark = _returns(record["benchmark"]["equity"])
    per_seed = []
    seed_returns = []
    for run in record["runs"]:
        returns = _returns(run["equity"])
        seed_returns.append(returns)
        per_seed.append(scoring.metrics(returns, ppy, benchmark=benchmark, turnover=run["turnover"]))
    row = {
        "id": record["submission"],
        "name": record["name"],
        "kind": record["kind"],
        "trials": record["trials"],
        "seeds": len(record["runs"]),
        "benchmark": False,
        "created": record["created"],
    }
    if record.get("contamination"):
        row["contamination"] = record["contamination"]
    if record.get("code_changed"):
        row["code_changed"] = True
    if record.get("policy_versions", 1) > 1:
        row["policy_versions"] = record["policy_versions"]
    if record.get("cohort", 1) > 1 or record.get("superseded"):
        row["cohort"] = record.get("cohort", 1)
    keys = per_seed[0].keys()
    stats = {key: _mean_sd([m[key] for m in per_seed]) for key in keys}
    row["sharpe"], row["sharpe_sd"] = stats["sharpe"]
    for key in _METRICS:
        if key in stats:
            row[key] = stats[key][0]
    averaged = np.mean(seed_returns, axis=0)
    row["_returns"] = averaged
    row["_benchmark"] = benchmark
    difference = _nan_safe(scoring.sharpe_difference, averaged, benchmark, ppy)
    row["delta_sharpe"], row["p_value"] = difference if difference else (float("nan"), float("nan"))
    return row


def _benchmark_row(record: dict) -> dict:
    ppy = record["periods_per_year"]
    benchmark = _returns(record["benchmark"]["equity"])
    values = scoring.metrics(benchmark, ppy)
    return {
        "id": None,
        "name": record["benchmark"]["name"],
        "kind": "benchmark",
        "trials": 1,
        "seeds": 1,
        "benchmark": True,
        "sharpe": values["sharpe"],
        "sharpe_sd": 0.0,
        "delta_sharpe": float("nan"),
        "p_value": float("nan"),
        "dsr": float("nan"),
        **{key: values[key] for key in ("cagr", "max_drawdown", "volatility", "total_return")},
    }


def _role(task_id: str) -> str:
    task = TASKS.get(task_id)
    if task_id in REEVALUATIONS:
        return "reevaluation"
    return "demo" if task is not None and task.source == "synthetic" else "ranked"


_ROLE_ORDER = {"ranked": 0, "reevaluation": 1, "demo": 2}


def _sections(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (
            TRACKS.index(record["track"]) if record["track"] in TRACKS else len(TRACKS),
            _ROLE_ORDER[_role(record["task"])],
            record["task"],
            record["dates"][0] if record["track"] in ("live", "ledger") else "",
        )
        groups.setdefault(key, []).append(record)

    sections = []
    for key in sorted(groups):
        members = groups[key]
        latest = max(members, key=lambda r: r["created"])
        rows = [_analyze_record(record) for record in members]
        n = min(len(row["_returns"]) for row in rows)
        sharpes = np.array([_per_period_sharpe(row["_returns"]) for row in rows])
        sharpes = sharpes[np.isfinite(sharpes)]
        variance = float(sharpes.var(ddof=1)) if sharpes.size > 1 else 0.0
        variance = max(variance, 1.0 / n)
        for row in rows:
            dsr = _nan_safe(scoring.deflated_sharpe, row["_returns"], row["trials"], variance)
            row["dsr"] = float(dsr) if dsr is not None else float("nan")
            del row["_returns"], row["_benchmark"]
        listed = [row for row in rows if row["id"] == BENCHMARK_SUBMISSION]
        if listed:
            listed[0]["benchmark"] = True
        else:
            rows.append(_benchmark_row(latest))
        rows.sort(key=lambda r: (not math.isfinite(r["sharpe"]), -r["sharpe"] if math.isfinite(r["sharpe"]) else 0.0))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        task = TASKS.get(latest["task"])
        sections.append(
            {
                "track": latest["track"],
                "task": latest["task"],
                "title": task.title if task else latest["task"],
                "role": _role(latest["task"]),
                "first": latest["dates"][1],
                "last": latest["dates"][-1],
                "sessions": len(latest["dates"]) - 1,
                "transaction_cost": latest["transaction_cost"],
                "periods_per_year": latest["periods_per_year"],
                "universe": latest["data"].get("tickers") or (list(task.tickers) if task else []),
                "frozen_through": latest["fit"]["history_end"] if latest["track"] == "live" else None,
                "frozen_at": latest["fit"].get("frozen_at"),
                "decisions_from": latest["dates"][0] if latest["track"] == "ledger" else None,
                "sharpe_variance": variance,
                "rows": rows,
            }
        )
    return sections


def _retired_row(record: dict, retired: str) -> dict:
    row = _analyze_record(record)
    del row["_returns"], row["_benchmark"]
    return {**row, "track": record["track"], "task": record["task"], "retired": retired}


def build(root="leaderboard", retired: Mapping[str, str] | None = None) -> dict:
    """Analyse every record under ``root`` into the summary document.

    ``retired`` maps retired submission ids to their retirement dates: their
    records stay out of the ranked sections and are listed under ``retired``.
    """
    retired = retired or {}
    records = load_records(root)
    latest: dict[tuple, int] = {}
    for r in records:
        key = (r["track"], r["task"], r["submission"])
        latest[key] = max(latest.get(key, 1), r.get("cohort", 1))
    records = [
        {**r, "superseded": True} if r.get("cohort", 1) < latest[(r["track"], r["task"], r["submission"])] else r
        for r in records
    ]
    summary = {"schema": 1, "sections": _sections([r for r in records if r["submission"] not in retired])}
    gone = [r for r in records if r["submission"] in retired]
    if gone:
        gone.sort(key=lambda r: (TRACKS.index(r["track"]), r["task"], r["submission"], r.get("cohort", 1)))
        summary["retired"] = [_retired_row(r, retired[r["submission"]]) for r in gone]
    return summary


# --- rendering -------------------------------------------------------------


def _fmt(value, template: str, missing: str = "–") -> str:
    return missing if value is None or not math.isfinite(value) else template.format(value)


def _pct(value) -> str:
    return _fmt(None if value is None else value * 100, "{:.1f}%")


def _row_cells(row: dict, link_prefix: str | None, ranked: bool = True) -> list[str]:
    name = row["name"]
    if row["benchmark"]:
        name = "**{}** ◆".format(name)
    elif row["id"] and link_prefix is not None:
        name = "[{}]({}/{}/)".format(name, link_prefix, row["id"])
    if row.get("contamination") == "possible":
        name += " (contamination: possible)"
    if row.get("cohort"):
        name += " (cohort {})".format(row["cohort"])
    if row.get("code_changed"):
        name += " (code changed)"
    if row.get("policy_versions"):
        name += " (decisions span {} policy versions)".format(row["policy_versions"])
    sharpe = _fmt(row["sharpe"], "{:.2f}")
    if row["sharpe"] is not None and math.isfinite(row["sharpe"]) and row.get("seeds", 1) > 1:
        sharpe += " ± {:.2f}".format(row["sharpe_sd"])
    alpha = "–"
    if "alpha" in row and math.isfinite(row["alpha"]):
        alpha = _pct(row["alpha"]) + _fmt(row.get("alpha_t"), " (t={:.1f})", "")
    delta = "–"
    if not row["benchmark"] and "delta_sharpe" in row and math.isfinite(row["delta_sharpe"]):
        p_value = row["p_value"]
        shown_p = "p<0.001" if p_value is not None and p_value < 0.001 else "p=" + _fmt(p_value, "{:.3f}")
        delta = "{:+.2f} ({})".format(row["delta_sharpe"], shown_p)
    return [
        str(row["rank"]) if ranked else "–",
        name,
        row["kind"],
        sharpe,
        _pct(row.get("cagr")),
        _pct(row.get("max_drawdown")),
        _fmt(row.get("turnover"), "{:.1f}×"),
        _fmt(row.get("beta"), "{:.2f}"),
        alpha,
        delta,
        _fmt(row.get("dsr"), "{:.2f}"),
        "–" if row["benchmark"] else str(row["seeds"]),
    ]


_COLUMNS = (
    "#", "Strategy", "Kind", "Sharpe", "CAGR", "Max DD", "Turnover/yr", "Beta", "Alpha (t)",
    "ΔSharpe vs 1/N", "DSR", "Seeds",
)


def _table(headers, rows) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
        *("| " + " | ".join(cells) + " |" for cells in rows),
    ]


def _heading(section: dict) -> str:
    title = section["title"]
    if section["role"] == "reevaluation":
        title = "Re-evaluation: " + title
    elif section["role"] == "demo":
        title = "Demo: " + title
    if section["track"] == "live":
        title = "{} (frozen {})".format(title, section["frozen_through"])
    elif section["track"] == "ledger":
        title = "{} (decisions from {})".format(title, section["decisions_from"])
    return title


def render_markdown(summary: dict, link_prefix: str | None = "../submissions") -> str:
    """The full ``LEADERBOARD.md``."""
    lines = [
        "# Leaderboard",
        "",
        "Can deep RL beat 1/N? Every row is traded by the same causal execution model "
        "(signal at close *t*, fill at open *t+1*, proportional fees, long-only, no leverage). "
        "Negative results stay in.",
        "",
    ]
    sections = summary["sections"]
    if not sections:
        lines += ["No results yet.", ""]
    for track in TRACKS:
        chosen = [s for s in sections if s["track"] == track]
        if not chosen:
            continue
        heading, intro = _TRACK_INTRO[track]
        lines += [heading, "", intro, ""]
        for section in chosen:
            lines += [
                "### {} — `{}`".format(_heading(section), section["task"]),
                "",
                "{} to {} ({} sessions) · fee {:.3g}% per side · {}".format(
                    section["first"],
                    section["last"],
                    section["sessions"],
                    section["transaction_cost"] * 100,
                    ", ".join(section["universe"]),
                ),
                "",
                *(
                    [
                        "*Survivorship-biased universe (chosen in hindsight): this re-tests a published "
                        "setup with this benchmark's execution model and is not a ranking.*",
                        "",
                    ]
                    if section["role"] == "reevaluation"
                    else []
                ),
                *_table(
                    _COLUMNS,
                    [_row_cells(row, link_prefix, section["role"] != "reevaluation") for row in section["rows"]],
                ),
                "",
            ]
    if summary.get("retired"):
        lines += [
            "## Retired",
            "",
            "Retired submissions are not ranked and are not updated; their records stay as last written.",
            "",
            *_table(
                ("Strategy", "Kind", "Track", "Task", "Retired after", "Sharpe", "CAGR", "Max DD", "Seeds"),
                [
                    [
                        row["name"] if link_prefix is None else "[{}]({}/{}/)".format(row["name"], link_prefix, row["id"]),
                        row["kind"],
                        row["track"],
                        "`{}`".format(row["task"]),
                        row["retired"],
                        _fmt(row["sharpe"], "{:.2f}"),
                        _pct(row.get("cagr")),
                        _pct(row.get("max_drawdown")),
                        str(row["seeds"]),
                    ]
                    for row in summary["retired"]
                ],
            ),
            "",
        ]
    if any(row.get("code_changed") for section in sections for row in section["rows"]):
        lines += [
            "*(code changed)*: the frozen policy was replayed on code that differs from the code at its "
            "freeze; the record lists both hashes, and the row does not show the frozen policy alone.",
            "",
        ]
    if any(row.get("policy_versions") for section in sections for row in section["rows"]):
        lines += [
            "*(decisions span N policy versions)*: the policy's own source changed while this ledger row was "
            "being decided; the record lists the code behind each decision.",
            "",
        ]
    lines += [
        "◆ marks the 1/N buy-and-hold benchmark (equal weights at the start of the period, never "
        "rebalanced); every ΔSharpe is measured against it over the same sessions.",
        "",
        "**Metrics.** Computed from daily simple returns with a zero risk-free rate. "
        "*Sharpe*: mean/std × √(periods per year); with several seeds, mean ± standard deviation of "
        "per-seed Sharpe ratios. *CAGR*, *Max DD* (peak-to-trough, including the starting value) and "
        "*Beta*/*Alpha* (OLS of daily returns on the benchmark's; alpha annualised, *t* is "
        "Newey-West) are means across seeds. *Turnover/yr*: annualised one-way traded fraction of "
        "equity. *ΔSharpe*: annualised Sharpe of the seed-averaged returns minus the benchmark's, "
        "with a one-sided paired stationary-bootstrap p-value (H1: better than 1/N). *DSR*: "
        "deflated Sharpe ratio of the seed-averaged returns, i.e. the probability the true Sharpe is "
        "positive after accounting for the submission's `trials` and for the dispersion of Sharpe "
        "ratios across this table (Bailey & López de Prado, 2014).",
        "",
    ]
    return "\n".join(lines)


def render_headline(summary: dict, link: str = "leaderboard/LEADERBOARD.md") -> str:
    """Compact table for the README: per task, the top rows, the benchmark
    and the best learned (RL or forecasting) policy."""
    rows = []
    for section in summary["sections"]:
        if section["role"] == "demo":
            continue
        shown = section["rows"][:HEADLINE_ROWS]
        learned = [r for r in section["rows"] if r["kind"] in LEARNED_KINDS]
        extra = [r for r in section["rows"] if r["benchmark"]] + learned[:1]
        shown += [r for r in section["rows"] if r in extra and r not in shown]
        for row in shown:
            cells = _row_cells(row, None, section["role"] != "reevaluation")
            rows.append(
                [
                    section["track"],
                    "`{}`{}".format(section["task"], " (re-evaluation)" if section["role"] == "reevaluation" else ""),
                    cells[0], cells[1], cells[3], cells[4], cells[5], cells[9], cells[10],
                ]
            )
    if not rows:
        return "No results yet."
    headers = ("Track", "Task", "#", "Strategy", "Sharpe", "CAGR", "Max DD", "ΔSharpe vs 1/N", "DSR")
    return "\n".join([*_table(headers, rows), "", "Full tables: [{}]({}).".format(link, link)])


def update_readme(path, text: str) -> bool:
    """Replace the text between the README markers; ``False`` if they are absent."""
    path = Path(path)
    if not path.exists():
        return False
    content = path.read_text()
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(content):
        return False
    block = "{}\n{}\n{}".format(README_START, text.strip("\n"), README_END)
    updated = pattern.sub(lambda _: block, content, count=1)
    if updated != content:
        path.write_text(updated)
    return True


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def write_leaderboard(
    root="leaderboard",
    *,
    readme="README.md",
    link_prefix: str | None = "../submissions",
    submissions_root=SUBMISSIONS_DIR,
) -> dict:
    """Write ``LEADERBOARD.md`` and ``summary.json`` under ``root``; refresh the README block.

    Submissions under ``submissions_root`` that are retired today are listed apart, not ranked.
    """
    root = Path(root)
    retired = {s.id: s.retired for s in discover(submissions_root).values() if s.is_retired()}
    summary = build(root, retired)
    root.mkdir(parents=True, exist_ok=True)
    (root / "LEADERBOARD.md").write_text(render_markdown(summary, link_prefix))
    (root / "summary.json").write_text(json.dumps(_clean(summary), indent=1) + "\n")
    ranked = [r for r in load_records(root) if r["submission"] not in retired]
    figures.render_sharpe_figure(ranked, root / "sharpe.png", figures.load_reported(root))
    if readme is not None and not update_readme(readme, render_headline(summary)):
        log.info("No leaderboard markers in %s; README left unchanged", readme)
    return summary
