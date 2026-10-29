"""The decision ledger: forward-tested decisions committed before each open.

Some strategies cannot be replayed or frozen (an LLM agent's output depends on
live news and a non-deterministic model), so they are scored on decisions
committed *before the fact*. A scheduled GitHub Actions workflow
(``.github/workflows/ledger.yml``) calls :func:`record_all` three times each
weekday. Each submission whose ``submission.json`` lists the task under
``"ledger"`` gets one entry per decision session::

    submissions/<id>/ledger/<task>/<decision_date>.json

``decision_date`` is the last session close in the data the decision used; the
fill is the next session's open, exactly as in :mod:`rl.bench.engine`.

Recording (:func:`record_decision`)
    * Runs are idempotent: an existing entry is never overwritten, and a
      session older than the newest entry is never back-filled.
    * The policy is built from the submission, ``reset()`` and then ``act`` is
      called once with the history through ``decision_date`` and the account's
      current weights, obtained by replaying the earlier entries through the
      engine. Ledger policies are **not fitted**: ``fit`` is never called
      because fitted state is not stored in an entry and could not be audited.
      A policy that needs fitting should use the frozen live track instead.
    * Optional policy attributes read after ``act``: ``last_cost_usd`` (spend of
      the call, counted against the budget) and ``ledger_inputs`` (JSON-able
      provenance: model ids, agent commit, raw decision...).
    * An LLM submission requires ``OPENAI_API_KEY``, ``FRED_API_KEY``,
      ``SEC_EDGAR_USER_AGENT`` and a monthly budget cap. Its policy is built
      before recording even a budget hold, so missing credentials, a key or
      model the provider rejects, a mismatched TradingAgents pin or graph
      construction failure writes no entry. Other participants still record,
      and the command reports failure.
    * If this month's summed ``cost_usd`` of the submission is at or above
      ``LEDGER_MONTHLY_BUDGET_USD`` the entry is a hold with reason
      ``"budget cap"`` (non-LLM policies are not built). Baselines cost nothing
      and ignore the cap. A policy whose ``act`` raises (or
      returns weights the engine rejects) yields a hold with reason
      ``"error: ..."``. A policy that cannot even be constructed is an
      infrastructure failure: nothing is written, so the job fails visibly
      and can be re-dispatched before the open.
    * The entry carries the Actions run identity from the environment.
    * The weights a policy is shown replay every earlier entry as applied
      (verification needs the GitHub API and happens at scoring). An entry
      later found unverifiable is scored as a hold, which does not change
      the weights of later entries, only the account they trade from.

Scoring (:func:`score_ledger`)
    An entry counts only if its ``run`` is a real run of
    ``.github/workflows/ledger.yml`` on ``main`` of this repository, started
    before the fill session's open (US: 09:30 America/New_York; Binance:
    00:00 UTC), checked against the GitHub API; a run id may back only one
    entry per submission and task. Everything else is treated as a hold and
    listed in the record. A session without an entry is a hold too. The
    scored window starts at the fill after the first verified entry. The
    GitHub API is reached through an injectable callable
    (``github(path) -> dict | None``, ``None`` for a 404, raising on any other
    failure so a flaky API cannot silently lower a score).

Crypto note: Binance sessions open at 00:00 UTC, so the weekday morning job
decides after the open of the day's session and its entries cannot verify.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from rl.bench import runner
from rl.bench.engine import Backtest, check_weights, simulate
from rl.bench.market import MarketData, MarketView, as_day
from rl.bench.submission import SUBMISSIONS_DIR, Submission, discover
from rl.bench.tasks import Task, get_task

log = logging.getLogger(__name__)

SCHEMA = 1
BUDGET_ENV = "LEDGER_MONTHLY_BUDGET_USD"
WORKFLOW_PATH = ".github/workflows/ledger.yml"
BRANCH = "main"
EVENTS = ("schedule", "workflow_dispatch")
GITHUB_API = "https://api.github.com"
US_OPEN = (dt.time(9, 30), ZoneInfo("America/New_York"))

GitHub = Callable[[str], "dict | None"]


# --- entries -----------------------------------------------------------------


def _directory(submission: Submission, task_id: str, root=None) -> Path:
    if root is not None:
        return Path(root) / submission.id / "ledger" / task_id
    return submission.ledger_dir(task_id)


def load_entries(submission: Submission, task_id: str, *, root=None) -> dict[str, dict]:
    """Every entry of ``submission`` on ``task_id`` keyed by file stem (decision date), in date order."""
    directory = _directory(submission, task_id, root)
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}


def _entry_target(entry: dict, tickers: tuple[str, ...]) -> np.ndarray | None:
    """The weights an entry trades (``None`` = keep holdings); raises if the engine would reject them."""
    if entry.get("hold") or entry.get("weights") is None:
        return None
    weights = entry["weights"]
    if not isinstance(weights, dict) or set(weights) != set(tickers):
        raise ValueError("weights must name exactly {}".format(list(tickers)))
    return check_weights([weights[t] for t in tickers], len(tickers))


def _targets(entries: dict[str, dict], tickers: tuple[str, ...]) -> tuple[dict, dict]:
    """Per decision date the weights to trade, and the dates whose entries are invalid."""
    targets, invalid = {}, {}
    for date, entry in entries.items():
        try:
            targets[date] = _entry_target(entry, tickers)
        except (ValueError, TypeError, KeyError) as error:
            invalid[date] = "invalid entry: {}".format(error)
    return targets, invalid


def _simulate(task: Task, data: MarketData, targets: dict) -> Backtest | None:
    """Trade ``targets`` (decision date -> weights or ``None``) through the engine.

    The account starts at the first decision date present in ``data``; a
    session without a target holds. ``None`` if no decision can be filled yet.
    """
    index = {str(d): i for i, d in enumerate(data.dates)}
    first = min((index[d] for d in targets if d in index), default=None)
    if first is None or first + 1 >= len(data):
        return None
    return simulate(
        data, lambda view: targets.get(str(view.date)), start=first + 1, transaction_cost=task.transaction_cost
    )


def _finite(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value >= 0 else default


def _jsonable(value) -> dict:
    return json.loads(json.dumps(value, default=str))


# --- recording ---------------------------------------------------------------


def _cap(budget) -> float | None:
    if budget is None:
        budget = os.environ.get(BUDGET_ENV, "").strip()
    if budget == "":
        return None
    try:
        cap = float(budget)
    except ValueError:
        cap = math.nan
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("{} must be a non-negative number, got {!r}".format(BUDGET_ENV, budget))
    return cap


def month_cost(submission: Submission, month: str, *, root=None) -> float:
    """USD the submission's entries (all tasks) created in ``YYYY-MM`` (UTC) report spending."""
    base = _directory(submission, "x", root).parent
    total = 0.0
    for path in sorted(base.glob("*/*.json")):
        try:
            entry = json.loads(path.read_text())
        except ValueError:
            log.warning("Unreadable ledger entry %s ignored for the budget", path)
            continue
        if str(entry.get("created_at", ""))[:7] == month:
            total += _finite(entry.get("cost_usd"))
    return total


def _run_identity(run: Mapping | None, now: dt.datetime | None) -> dict | None:
    if run is not None:
        return dict(run)
    env = os.environ
    if not env.get("GITHUB_RUN_ID"):
        return None
    return {
        "repository": env.get("GITHUB_REPOSITORY"),
        "run_id": env["GITHUB_RUN_ID"],
        "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "workflow": env.get("GITHUB_WORKFLOW_REF"),
        "started_at": runner._utc(now),
    }


def record_decision(
    task: Task,
    submission: Submission,
    *,
    data: MarketData | None = None,
    now: dt.datetime | None = None,
    run: Mapping | None = None,
    budget=None,
    refresh: bool = False,
    root=None,
) -> Path | None:
    """Write the entry for the latest session in ``data`` (default: all data available).

    ``run`` overrides the Actions run identity read from the environment;
    ``budget`` (USD per month) overrides ``LEDGER_MONTHLY_BUDGET_USD``;
    ``root`` replaces the submissions directory. Returns the entry's path
    (existing or new), or ``None`` if recording would back-fill a session.
    """
    data = runner._data_for(task, data, refresh=refresh)
    if len(data) < 1:
        raise ValueError("No data for {}".format(task.id))
    t = len(data) - 1
    decision_date = str(data.dates[t])
    history = data.rows(0, t + 1)
    directory = _directory(submission, task.id, root)
    path = directory / "{}.json".format(decision_date)
    if path.exists():
        log.info("%s on %s: %s already recorded", submission.id, task.id, decision_date)
        return path
    entries = load_entries(submission, task.id, root=root)
    if any(date > decision_date for date in entries):
        log.warning(
            "%s on %s: entries exist after %s; refusing to back-fill", submission.id, task.id, decision_date
        )
        return None

    entry = {
        "schema": SCHEMA,
        "submission": submission.id,
        "task": task.id,
        "decision_date": decision_date,
        "weights": None,
        "hold": True,
        "reason": None,
        "cost_usd": 0.0,
        "inputs": {},
        "data_sha256": history.sha256(),
        "run": _run_identity(run, now),
        "created_at": runner._utc(now),
    }
    cap = None if submission.kind == "baseline" else _cap(budget)
    policy = None
    if submission.kind == "llm":
        missing = [name for name in ("OPENAI_API_KEY", "FRED_API_KEY", "SEC_EDGAR_USER_AGENT")
                   if not os.environ.get(name, "").strip()]
        if cap is None:
            missing.append(BUDGET_ENV)
        if missing:
            raise ValueError("LLM ledger preflight requires {}".format(", ".join(missing)))
        policy = runner.build_policy(submission, task)
    if cap is not None and month_cost(submission, entry["created_at"][:7], root=root) >= cap:
        entry["reason"] = "budget cap"
    else:
        if policy is None:
            policy = runner.build_policy(submission, task)
        targets, invalid = _targets(entries, history.tickers)
        for date, reason in invalid.items():
            log.warning("%s on %s: entry %s treated as a hold (%s)", submission.id, task.id, date, reason)
        backtest = _simulate(task, history, {d: v for d, v in targets.items() if d not in invalid})
        current = np.zeros(len(history.tickers)) if backtest is None else backtest.weights[-1].copy()
        current.setflags(write=False)
        try:
            policy.reset()
            decision = policy.act(MarketView(history, current))
            if decision is not None:
                weights = check_weights(decision, len(history.tickers), decision_date)
                entry["weights"] = {ticker: float(w) for ticker, w in zip(history.tickers, weights)}
                entry["hold"] = False
        except Exception as error:  # a failing policy holds; it never stops the ledger
            log.exception("%s on %s: policy failed on %s", submission.id, task.id, decision_date)
            entry["reason"] = "error: {}: {}".format(type(error).__name__, error)
        entry["cost_usd"] = _finite(getattr(policy, "last_cost_usd", 0.0))
        inputs = getattr(policy, "ledger_inputs", None)
        entry["inputs"] = _jsonable(inputs) if isinstance(inputs, dict) else {}
        entry["inputs"].setdefault("policy", submission.policy)
        entry["inputs"].setdefault("params", submission.params)
    runner._write_json(path, entry, indent=2)
    return path


def record_all(
    root=SUBMISSIONS_DIR,
    *,
    task: str | None = None,
    submission: str | None = None,
    data: Mapping[str, MarketData] | None = None,
    now: dt.datetime | None = None,
    run: Mapping | None = None,
    budget=None,
    refresh: bool = False,
) -> tuple[list[Path], list[str]]:
    """:func:`record_decision` for every ``(task, submission)`` listed under ``"ledger"``.

    ``data`` optionally maps task ids to panels. Returns the entry paths and
    one message per pair that failed (unknown task, construction or data
    errors); a failing pair never stops the others.
    """
    found = discover(root)
    if submission is not None and submission not in found:
        raise KeyError("Unknown submission {!r}; found {}".format(submission, sorted(found)))
    written, failures = [], []
    loaded: dict[str, MarketData] = dict(data or {})
    for sub in found.values():
        if submission not in (None, sub.id):
            continue
        for task_id in sub.ledger:
            if task not in (None, task_id):
                continue
            try:
                spec = get_task(task_id)
                if task_id not in loaded:
                    loaded[task_id] = runner._data_for(spec, None, refresh=refresh)
                path = record_decision(
                    spec, sub, data=loaded[task_id], now=now, run=run, budget=budget, root=root
                )
            except Exception as error:
                log.exception("Ledger recording failed for %s on %s", sub.id, task_id)
                failures.append("{}/{}: {}".format(task_id, sub.id, error))
            else:
                if path is not None:
                    written.append(path)
    return written, failures


# --- verification ------------------------------------------------------------


def _open_utc(task: Task, fill_date: str) -> dt.datetime:
    day = dt.date.fromisoformat(fill_date)
    if task.source == "binance":
        return dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    clock, zone = US_OPEN
    return dt.datetime.combine(day, clock, tzinfo=zone).astimezone(dt.timezone.utc)


def _parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _repository(explicit: str | None = None) -> str:
    """``owner/name`` of this repository: argument, ``GITHUB_REPOSITORY``, or the ``origin`` remote."""
    repository = explicit or os.environ.get("GITHUB_REPOSITORY")
    if repository:
        return repository
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        url = ""
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise ValueError("Cannot tell which repository to verify runs against; set GITHUB_REPOSITORY")
    return match.group(1)


def github_get(path: str) -> dict | None:
    """GET ``path`` from the GitHub API (token from ``GITHUB_TOKEN``); ``None`` on 404."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "rl-bench-ledger"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(GITHUB_API + path, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code < 500 or attempt == 2:
                raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _check_run(entry_run, repository: str, opens: dt.datetime, fetch: Callable[[str], dict | None]) -> str | None:
    """Why the entry's run does not prove a pre-open commit, or ``None`` if it does."""
    if not isinstance(entry_run, dict):
        return "no run identity"
    run_id = str(entry_run.get("run_id") or "")
    if not run_id.isdigit():
        return "no run identity"
    if str(entry_run.get("repository") or "").lower() != repository.lower():
        return "run belongs to another repository"
    actual = fetch(run_id)
    if actual is None:
        return "run {} not found".format(run_id)
    if str(actual.get("path") or "").split("@", 1)[0] != WORKFLOW_PATH:
        return "run {} is not {}".format(run_id, WORKFLOW_PATH)
    if actual.get("head_branch") != BRANCH:
        return "run {} is not on {}".format(run_id, BRANCH)
    if actual.get("event") not in EVENTS:
        return "run {} was triggered by {!r}".format(run_id, actual.get("event"))
    full_name = (actual.get("repository") or {}).get("full_name")
    if full_name is not None and full_name.lower() != repository.lower():
        return "run belongs to another repository"
    try:
        started = _parse_time(actual["run_started_at"])
    except (KeyError, ValueError, TypeError):
        return "run {} has no start time".format(run_id)
    if started >= opens:
        return "run started {} at or after the open {}".format(
            started.isoformat(), opens.replace(microsecond=0).isoformat()
        )
    return None


# --- scoring -----------------------------------------------------------------


def score_ledger(
    task: Task,
    submission: Submission,
    *,
    data: MarketData | None = None,
    out_root=runner.LEADERBOARD_DIR,
    github: GitHub | None = None,
    repository: str | None = None,
    refresh: bool = False,
    root=None,
) -> Path | None:
    """Score the verified entries into ``leaderboard/ledger/<task>/<id>.json``.

    Returns ``None`` while no verified entry has a filled session yet.
    """
    entries = load_entries(submission, task.id, root=root)
    if not entries:
        log.info("%s on %s: no ledger entries", submission.id, task.id)
        return None
    data = runner._data_for(task, data, refresh=refresh)
    repository = _repository(repository)
    client = github or github_get
    runs: dict[str, dict | None] = {}

    def fetch(run_id: str) -> dict | None:
        if run_id not in runs:
            runs[run_id] = client("/repos/{}/actions/runs/{}".format(repository, run_id))
        return runs[run_id]

    index = {str(d): i for i, d in enumerate(data.dates)}
    targets, invalid = _targets(entries, data.tickers)
    verified: dict[str, np.ndarray | None] = {}
    unverified: dict[str, str] = {}
    pending = 0
    used_runs: set[str] = set()
    for date, entry in entries.items():
        position = index.get(date)
        if position == len(data) - 1 or (position is None and date > str(data.dates[-1])):
            pending += 1  # its fill session has not happened yet
            continue
        if position is None:
            unverified[date] = "decision date is not a session in the data"
            continue
        if (entry.get("submission"), entry.get("task"), entry.get("decision_date")) != (
            submission.id,
            task.id,
            date,
        ):
            unverified[date] = "entry does not match its path"
            continue
        if date in invalid:
            unverified[date] = invalid[date]
            continue
        reason = _check_run(entry.get("run"), repository, _open_utc(task, str(data.dates[position + 1])), fetch)
        if reason is None:
            run_id = str(entry["run"]["run_id"])
            if run_id in used_runs:
                reason = "run {} already backs another entry".format(run_id)
            used_runs.add(run_id)
        if reason is not None:
            unverified[date] = reason
            continue
        verified[date] = targets[date]
    for date, reason in unverified.items():
        log.warning("%s on %s: entry %s treated as a hold: %s", submission.id, task.id, date, reason)

    backtest = _simulate(task, data, verified)
    if backtest is None:
        log.info("%s on %s: no verified entry with a filled session yet", submission.id, task.id)
        return None
    start = data.index_of(backtest.dates[0])
    missed = sum(1 for i in range(start, len(data) - 1) if str(data.dates[i]) not in entries)
    history = data.rows(0, start + 1)
    benchmark = runner._benchmark(task, data, history, start + 1, as_day(task.validation_start))
    fit = {
        "first_decision": str(backtest.dates[0]),
        "entries": len(entries),
        "verified": len(verified),
        "unverified": len(unverified),
        "missed_sessions": missed,
        "pending": pending,
        "unverified_entries": [{"decision_date": d, "reason": r} for d, r in unverified.items()],
    }
    record = runner._record(task, submission, "ledger", data, fit, [(0, backtest)], benchmark)
    return runner._store(Path(out_root) / "ledger" / task.id / "{}.json".format(submission.id), record)


def score_all(
    root=SUBMISSIONS_DIR,
    *,
    out_root=runner.LEADERBOARD_DIR,
    task: str | None = None,
    submission: str | None = None,
    data: Mapping[str, MarketData] | None = None,
    github: GitHub | None = None,
    repository: str | None = None,
    refresh: bool = False,
) -> tuple[list[Path], list[str]]:
    """:func:`score_ledger` for every submission with entries; same return shape as :func:`record_all`."""
    found = discover(root)
    if submission is not None and submission not in found:
        raise KeyError("Unknown submission {!r}; found {}".format(submission, sorted(found)))
    written, failures = [], []
    loaded: dict[str, MarketData] = dict(data or {})
    for sub in found.values():
        if submission not in (None, sub.id):
            continue
        for task_id in sub.ledger:
            if task not in (None, task_id):
                continue
            try:
                spec = get_task(task_id)
                if not _directory(sub, task_id).is_dir():
                    continue
                if task_id not in loaded:
                    loaded[task_id] = runner._data_for(spec, None, refresh=refresh)
                path = score_ledger(
                    spec, sub, data=loaded[task_id], out_root=out_root, github=github, repository=repository
                )
            except Exception as error:
                log.exception("Ledger scoring failed for %s on %s", sub.id, task_id)
                failures.append("{}/{}: {}".format(task_id, sub.id, error))
            else:
                if path is not None:
                    written.append(path)
    return written, failures


# --- command line ------------------------------------------------------------


def _report(verb: str, written: list[Path], failures: list[str]) -> int:
    for path in written:
        print("{} {}".format(verb, path))
    if not written:
        print("Nothing to do")
    if failures:
        print("Failed: {}".format("; ".join(failures)), file=sys.stderr)
        return 1
    return 0


def _cmd_record(options) -> int:
    written, failures = record_all(
        options.submissions_root,
        task=options.task,
        submission=options.submission,
        budget=options.budget,
        refresh=options.refresh,
    )
    return _report("Recorded", written, failures)


def _cmd_score(options) -> int:
    written, failures = score_all(
        options.submissions_root,
        out_root=options.out_root,
        task=options.task,
        submission=options.submission,
        refresh=options.refresh,
    )
    return _report("Wrote", written, failures)


def add_cli(commands, common) -> None:
    """Register ``ledger record`` and ``ledger score`` on the bench subparsers.

    The ``ledger`` parser stores the chosen action in ``options.handler``;
    ``cli.main`` must call ``options.handler(options)`` for ``ledger``.
    """
    parser = commands.add_parser("ledger", help="decision ledger: record or score pre-open decisions")
    actions = parser.add_subparsers(dest="ledger_command", required=True)

    record = actions.add_parser(
        "record", parents=[common], help="write today's entry for every submission with ledger tasks (CI)"
    )
    record.add_argument("--task")
    record.add_argument("--submission")
    record.add_argument("--budget", type=float, help="monthly USD cap per submission (default: ${})".format(BUDGET_ENV))
    record.add_argument("--refresh", action="store_true", help="ignore the local data cache")
    record.set_defaults(handler=_cmd_record)

    score = actions.add_parser(
        "score", parents=[common], help="verify entries against GitHub and write leaderboard/ledger records"
    )
    score.add_argument("--task")
    score.add_argument("--submission")
    score.add_argument("--refresh", action="store_true", help="ignore the local data cache")
    score.set_defaults(handler=_cmd_score)
