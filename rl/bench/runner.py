"""Run submissions on benchmark tasks and write result records.

Two tracks share one execution model (:func:`rl.bench.engine.simulate`):

* ``historical`` — fit on the sessions before ``task.test_start`` and trade
  the fixed test window ``[task.test_start, task.test_end)``.
* ``live`` — :func:`freeze` fits a policy on every session available today
  and stores it under the submission. :func:`publish`, run only by CI on the
  main branch, stamps each new freeze with the time it became public, and
  :func:`update_live` replays the frozen policy over the sessions *after*
  that publication, so it cannot have been tuned or cherry-picked on them.

Every record also carries the equal-weight buy-and-hold benchmark simulated
identically. Records are the schema-1 JSON documents the leaderboard reads.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from rl.bench.engine import Backtest, simulate
from rl.bench.market import MarketData, as_day
from rl.bench.policies.baselines import EqualWeightBuyHold
from rl.bench.policy import Policy
from rl.bench.submission import Submission
from rl.bench.tasks import Task

log = logging.getLogger(__name__)

SCHEMA = 1
LEADERBOARD_DIR = Path("leaderboard")
STOCHASTIC_SEEDS = {"historical": 5, "live": 3}
BENCHMARK_NAME = "1/N buy and hold"
FREEZE_FILE = "freeze.json"
PUBLICATIONS_FILE = "publications.json"
_PACKAGES = ("rl-pricing", "numpy", "pandas", "torch", "gymnasium", "stable-baselines3")


def build_policy(submission: Submission, task: Task) -> Policy:
    """Construct the policy, passing the task's fee to policies that take one.

    RL and forecasting policies train against the execution model and need
    ``transaction_cost``; a submission may still pin its own value.
    """
    cls = submission.policy_class()
    params = dict(submission.params)
    if "transaction_cost" not in params and "transaction_cost" in inspect.signature(cls).parameters:
        params["transaction_cost"] = task.transaction_cost
    return cls(**params)


def _seeds(policy: Policy, seeds, track: str) -> list[int]:
    if seeds is not None:
        seeds = [int(s) for s in seeds]
        if not seeds:
            raise ValueError("At least one seed is required")
        return seeds
    return list(range(STOCHASTIC_SEEDS[track])) if policy.stochastic else [0]


def _round(values, digits: int = 10) -> list[float]:
    return [float("{:.{}g}".format(float(v), digits)) for v in values]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _versions() -> dict:
    found = {}
    for name in _PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def _utc(now: dt.datetime | None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def _day(value) -> str:
    return str(as_day(value))


def _write_json(path: Path, document: dict, **dumps) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, **dumps) + "\n")
    os.replace(temporary, path)


def _data_for(task: Task, data: MarketData | None, end=None, refresh: bool = False) -> MarketData:
    if data is not None:
        if data.tickers != task.tickers:
            data = data.select(task.tickers)
        return data
    from rl.bench.data import load_task_data

    return load_task_data(task, end=end, refresh=refresh)


def _run(task: Task, policy: Policy, data: MarketData, start: int, seed, fit_history, validation_start) -> Backtest:
    if seed is not None:
        policy.fit(fit_history, validation_start, seed)
    policy.reset()
    return simulate(data, policy.act, start=start, transaction_cost=task.transaction_cost)


def _benchmark(task: Task, data: MarketData, history: MarketData, start: int, validation_start) -> Backtest:
    policy = EqualWeightBuyHold()
    return _run(task, policy, data, start, 0, history, validation_start)


def _record(
    task: Task,
    submission: Submission,
    track: str,
    data: MarketData,
    fit: dict,
    runs: list[tuple[int, Backtest]],
    benchmark: Backtest,
) -> dict:
    return {
        "schema": SCHEMA,
        "task": task.id,
        "track": track,
        "submission": submission.id,
        "name": submission.name,
        "kind": submission.kind,
        "trials": submission.trials,
        "policy": submission.policy,
        "params": submission.params,
        "transaction_cost": task.transaction_cost,
        "periods_per_year": task.periods_per_year,
        "data": {
            "source": task.source,
            "sha256": data.sha256(),
            "first": str(data.dates[0]),
            "last": str(data.dates[-1]),
            "tickers": list(data.tickers),
        },
        "fit": fit,
        "dates": [str(d) for d in benchmark.dates],
        "runs": [
            {"seed": seed, "equity": _round(bt.equity), "turnover": _round(bt.turnover)}
            for seed, bt in runs
        ],
        "benchmark": {"name": BENCHMARK_NAME, "equity": _round(benchmark.equity)},
        "created": _utc(None),
        "commit": _commit(),
        "versions": _versions(),
    }


def _store(path: Path, record: dict) -> Path:
    if path.exists():
        try:
            previous = json.loads(path.read_text())["data"]["sha256"]
        except (ValueError, KeyError, TypeError):
            previous = None
        if previous != record["data"]["sha256"]:
            log.warning(
                "Overwriting %s: data sha256 changed from %s to %s",
                path, previous, record["data"]["sha256"],
            )
    _write_json(path, record, separators=(",", ":"))
    return path


def evaluate(
    task: Task,
    submission: Submission,
    *,
    data: MarketData | None = None,
    seeds=None,
    out_root=LEADERBOARD_DIR,
) -> Path:
    """Historical track: fit before ``task.test_start``, trade the test window."""
    _require_replayable(submission)
    data = _data_for(task, data, end=task.test_end).between(None, task.test_end)
    start = data.index_of(task.test_start)
    history = data.rows(0, start)
    validation_start = as_day(task.validation_start)

    seeds = _seeds(build_policy(submission, task), seeds, "historical")
    runs = []
    for seed in seeds:
        log.info("%s on %s: seed %d", submission.id, task.id, seed)
        runs.append(
            (seed, _run(task, build_policy(submission, task), data, start, seed, history, validation_start))
        )
    benchmark = _benchmark(task, data, history, start, validation_start)
    fit = {"history_end": str(history.dates[-1]), "validation_start": str(validation_start)}
    record = _record(task, submission, "historical", data, fit, runs, benchmark)
    record["submission_sha256"] = submission.fingerprint()
    return _store(_historical_path(out_root, task, submission), record)


def _historical_path(out_root, task: Task, submission: Submission) -> Path:
    return Path(out_root) / "historical" / task.id / "{}.json".format(submission.id)


def historical_is_current(task: Task, submission: Submission, out_root=LEADERBOARD_DIR) -> bool:
    """Whether the stored historical record was computed from this submission as it is now."""
    try:
        record = json.loads(_historical_path(out_root, task, submission).read_text())
    except FileNotFoundError:
        return False
    return record.get("submission_sha256") == submission.fingerprint()


def freeze(
    task: Task,
    submission: Submission,
    *,
    data: MarketData | None = None,
    seeds=None,
    now: dt.datetime | None = None,
    force: bool = False,
    refresh: bool = False,
) -> Path:
    """Live track: fit on all available sessions and store the frozen policy."""
    _require_live(task)
    _require_replayable(submission)
    directory = submission.frozen_dir(task.id)
    marker = directory / FREEZE_FILE
    if marker.exists() and not force:
        raise FileExistsError("{} is already frozen for {}; pass force=True to replace".format(submission.id, task.id))
    data = _data_for(task, data, refresh=refresh)
    if len(data) < 2:
        raise ValueError("Not enough data to freeze {}".format(task.id))
    fitted_through = data.dates[-1]
    validation_start = data.dates[data.index_of(fitted_through - task.validation_length)]

    seeds = _seeds(build_policy(submission, task), seeds, "live")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for seed in seeds:
        log.info("Freezing %s on %s: seed %d", submission.id, task.id, seed)
        policy = build_policy(submission, task)
        policy.fit(data, validation_start, seed)
        seed_dir = directory / "seed{}".format(seed)
        seed_dir.mkdir()
        policy.save(seed_dir)
    freeze_record = {
        "task": task.id,
        "submission": submission.id,
        "policy": submission.policy,
        "params": submission.params,
        "fitted_through": str(fitted_through),
        "validation_start": str(validation_start),
        "frozen_at": _utc(now),
        "seeds": seeds,
        "data_sha256": data.sha256(),
        "commit": _commit(),
        "artifacts": _artifact_hashes(directory),
    }
    _write_json(marker, freeze_record, indent=2, sort_keys=False)
    return marker


def _require_live(task: Task) -> None:
    if not task.live:
        raise ValueError("Task {} has no live track".format(task.id))


def _require_replayable(submission: Submission) -> None:
    """Ledger participants (e.g. LLM agents) are scored only from decisions
    committed before each open: replaying them on past data would pay for
    calls on a period their models may remember."""
    if submission.ledger:
        raise ValueError(
            "{} takes part only through the decision ledger (tasks: {})".format(
                submission.id, ", ".join(submission.ledger)
            )
        )


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != FREEZE_FILE
    }


def verify_freeze(directory: Path) -> dict:
    """Load ``freeze.json`` and check that the stored artifacts are unchanged."""
    freeze_record = json.loads((directory / FREEZE_FILE).read_text())
    expected = freeze_record["artifacts"]
    actual = _artifact_hashes(directory)
    problems = sorted(
        name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name)
    )
    if problems:
        raise ValueError("Frozen artifacts in {} changed since the freeze: {}".format(directory, problems))
    return freeze_record


def update_live(
    task: Task,
    submission: Submission,
    *,
    data: MarketData | None = None,
    out_root=LEADERBOARD_DIR,
    refresh: bool = False,
) -> Path | None:
    """Replay the frozen policy over every session after its publication."""
    _require_live(task)
    directory = submission.frozen_dir(task.id)
    if not (directory / FREEZE_FILE).exists():
        raise FileNotFoundError("{} has no freeze for {}".format(submission.id, task.id))
    frozen = verify_freeze(directory)
    if frozen["policy"] != submission.policy or frozen["params"] != submission.params:
        raise ValueError(
            "{} changed since it was frozen on {}; re-freeze it to evaluate the new configuration".format(
                submission.id, frozen["frozen_at"]
            )
        )
    data = _data_for(task, data, refresh=refresh)
    fitted_through = as_day(frozen["fitted_through"])
    anchor = data.index_of(fitted_through)
    if anchor >= len(data) or data.dates[anchor] != fitted_through:
        raise ValueError("Data for {} lacks the freeze session {}".format(task.id, fitted_through))
    if data.rows(0, anchor + 1).sha256() != frozen["data_sha256"]:
        log.warning(
            "%s: history through %s differs from the data fitted on (revised prices?)",
            task.id, fitted_through,
        )
    published = _publication(Path(out_root), submission.id, task.id, directory / FREEZE_FILE)
    if published is None:
        log.info("%s on %s: frozen but not yet published", submission.id, task.id)
        return None
    # The first evaluated session fills after the publication day.
    start = int(np.searchsorted(data.dates, max(fitted_through, as_day(published)), side="right"))
    if start >= len(data):
        log.info("%s on %s: no sessions after its publication on %s yet", submission.id, task.id, published)
        return None
    history = data.rows(0, start)
    runs = []
    for seed in frozen["seeds"]:
        policy = build_policy(submission, task)
        policy.load(directory / "seed{}".format(seed))
        runs.append((seed, _run(task, policy, data, start, None, history, None)))
    validation_start = as_day(frozen["validation_start"])
    benchmark = _benchmark(task, data, history, start, validation_start)
    fit = {
        "history_end": frozen["fitted_through"],
        "validation_start": frozen["validation_start"],
        "frozen_at": frozen["frozen_at"],
        "published_at": published,
        "frozen_data_sha256": frozen["data_sha256"],
    }
    record = _record(task, submission, "live", data, fit, runs, benchmark)
    return _store(Path(out_root) / "live" / task.id / "{}.json".format(submission.id), record)


def _publication(out_root: Path, submission_id: str, task_id: str, marker: Path) -> str | None:
    """When this exact freeze was published, or ``None`` if it has not been."""
    path = out_root / PUBLICATIONS_FILE
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get("{}/{}".format(submission_id, task_id))
    if entry is None or entry["freeze_sha256"] != _sha256_file(marker):
        return None
    return entry["published_at"]


def publish(submissions_root, *, out_root=LEADERBOARD_DIR, now: dt.datetime | None = None) -> list[str]:
    """Stamp every unpublished freeze with the current UTC time.

    Only the CI job on the main branch runs this, so a publication time is
    when the frozen policy became public; live scoring starts after it. A
    re-frozen policy (a changed ``freeze.json``) is published anew. Returns
    the ``submission/task`` keys stamped.
    """
    path = Path(out_root) / PUBLICATIONS_FILE
    publications = json.loads(path.read_text()) if path.exists() else {}
    stamped = []
    for marker in sorted(Path(submissions_root).glob("*/frozen/*/{}".format(FREEZE_FILE))):
        verify_freeze(marker.parent)
        key = "{}/{}".format(marker.parent.parent.parent.name, marker.parent.name)
        digest = _sha256_file(marker)
        if publications.get(key, {}).get("freeze_sha256") == digest:
            continue
        publications[key] = {"published_at": _utc(now), "freeze_sha256": digest}
        stamped.append(key)
    if stamped:
        _write_json(path, dict(sorted(publications.items())), indent=2)
    return stamped
