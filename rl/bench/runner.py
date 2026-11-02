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
identically. Records are JSON documents the leaderboard reads: schema 2 adds
``protocol_version`` and the ``code`` identity; schema 1 records stay readable.
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

from rl import __version__
from rl.bench.engine import Backtest, simulate
from rl.bench.info import InfoPanel, historical_contamination, load_task_info, session_close
from rl.bench.market import MarketData, MarketView, as_day
from rl.bench.policies.baselines import EqualWeightBuyHold
from rl.bench.policy import Policy
from rl.bench.submission import Submission
from rl.bench.tasks import Task

log = logging.getLogger(__name__)

SCHEMA = 2
PROTOCOL_VERSION = "2"
READABLE_SCHEMAS = (1, 2)
LEADERBOARD_DIR = Path("leaderboard")
STOCHASTIC_SEEDS = {"historical": 5, "live": 3}
BENCHMARK_NAME = "1/N buy and hold"
FREEZE_FILE = "freeze.json"
COHORTS_DIR = "cohorts"
PUBLICATIONS_FILE = "publications.json"
_PACKAGES = ("rl-pricing", "numpy", "pandas", "torch", "gymnasium", "stable-baselines3")


def policy_params(submission: Submission, task: Task | None = None) -> dict:
    """Constructor arguments of the submission's policy.

    ``max_cost_per_decision_usd`` is a ledger declaration and is passed on only
    to a policy that accepts it. RL and forecasting policies train against the
    execution model and need ``transaction_cost``; a submission may still pin
    its own value.
    """
    parameters = inspect.signature(submission.policy_class()).parameters
    params = dict(submission.params)
    if "max_cost_per_decision_usd" not in parameters:
        params.pop("max_cost_per_decision_usd", None)
    if task is not None and "transaction_cost" not in params and "transaction_cost" in parameters:
        params["transaction_cost"] = task.transaction_cost
    if task is not None and "crypto" not in params and "crypto" in parameters:
        params["crypto"] = task.source == "binance"
    return params


def build_policy(submission: Submission, task: Task) -> Policy:
    """Construct the policy, passing the task's fee to policies that take one."""
    return submission.policy_class()(**policy_params(submission, task))


def fit_policy(policy: Policy, history: MarketData, validation_start, seed: int, info: InfoPanel | None = None,
               *, crypto: bool = False) -> None:
    """Fit ``policy``, giving policies that accept ``info`` only the fitted sessions' items.

    The panel is cut at the close of ``history``'s last session, so a policy
    cannot see an item published during the validation or test period through
    its training data. Policies without an ``info`` parameter are fitted as before.
    """
    parameters = inspect.signature(policy.fit).parameters
    kwargs = {}
    if "info" in parameters:
        kwargs["info"] = None if info is None else info.truncated(
            session_close(history.dates[-1], crypto=crypto))
    if "crypto" in parameters:
        kwargs["crypto"] = crypto
    policy.fit(history, validation_start, seed, **kwargs)


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


def code_identity(submission: Submission) -> dict:
    """Package version, git commit and policy source hash that produced a result."""
    return {"package_version": __version__, "git_commit": _commit(), "code_sha256": submission.code_sha256()}


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


def _run(
    task: Task, policy: Policy, data: MarketData, start: int, seed, fit_history, validation_start,
    info: InfoPanel | None = None,
) -> Backtest:
    if seed is not None:
        fit_policy(
            policy, fit_history, validation_start, seed, info,
            crypto=task.source == "binance",
        )
    policy.reset()
    act = policy.act
    if info is not None:
        def act(view):
            return policy.act(MarketView(
                view.history, view.weights, info.at_session(view.date, crypto=task.source == "binance")
            ))
    return simulate(data, act, start=start, transaction_cost=task.transaction_cost)


def _load_info(task: Task, submission: Submission, data: MarketData, refresh: bool = False) -> InfoPanel | None:
    """The declared sources as of the last decision close in ``data``, or ``None`` if none are declared.

    Later items cannot reach any decision, so leaving them out keeps ``info_sha256``
    a property of the information the record used, not of when the source was downloaded.
    """
    if not (task.info_sources or submission.info_sources):
        return None
    panel = load_task_info(task, submission, refresh=refresh)
    return panel.truncated(session_close(data.dates[-2], crypto=task.source == "binance"))


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
        "protocol_version": PROTOCOL_VERSION,
        "code": code_identity(submission),
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
    _require_task(submission, task)
    _require_replayable(submission)
    data = _data_for(task, data, end=task.test_end).between(None, task.test_end)
    info = _load_info(task, submission, data)
    contamination = historical_contamination(submission, task, info or InfoPanel())
    start = data.index_of(task.test_start)
    history = data.rows(0, start)
    validation_start = as_day(task.validation_start)

    seeds = _seeds(build_policy(submission, task), seeds, "historical")
    runs = []
    for seed in seeds:
        log.info("%s on %s: seed %d", submission.id, task.id, seed)
        runs.append(
            (seed, _run(task, build_policy(submission, task), data, start, seed, history, validation_start, info))
        )
    benchmark = _benchmark(task, data, history, start, validation_start)
    fit = {"history_end": str(history.dates[-1]), "validation_start": str(validation_start)}
    record = _record(task, submission, "historical", data, fit, runs, benchmark)
    record["submission_sha256"] = submission.fingerprint()
    if info is not None:
        record["info_sha256"] = info.sha256()
    if contamination:
        record["contamination"] = contamination
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


def freeze_is_current(task: Task, submission: Submission, out_root=LEADERBOARD_DIR) -> bool:
    """Whether the stored freeze was made from this submission as it is now.

    Edits to the shared ``rl.bench`` code do not make a freeze stale: replay
    reports them as ``code_changed``. A schema-1 freeze stores no fingerprint,
    so its historical record stands in for it.
    """
    marker = submission.frozen_dir(task.id) / FREEZE_FILE
    if not marker.exists():
        return False
    frozen = verify_freeze(marker.parent)
    if frozen["policy"] != submission.policy or frozen["params"] != submission.params:
        return False
    if "submission_sha256" in frozen:
        return frozen["submission_sha256"] == submission.fingerprint()
    return not _historical_path(out_root, task, submission).exists() or historical_is_current(task, submission, out_root)


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
    """Live track: fit on all available sessions and store the frozen policy.

    Replacing an existing freeze (``force``) starts the next cohort: the
    earlier cohorts move to ``frozen/<task>/cohorts/<n>/`` and keep their
    own publication and live record.
    """
    _require_live(task)
    _require_task(submission, task)
    _require_replayable(submission)
    directory = submission.frozen_dir(task.id)
    marker = directory / FREEZE_FILE
    # Beside frozen/, so a crash cannot leave a stray freeze.json for publish(); a failed fit leaves the old freeze alone.
    staging = submission.directory / ".freeze-{}.tmp".format(task.id)
    if _finish_interrupted_freeze(directory, staging):
        return marker
    if marker.exists() and not force:
        raise FileExistsError("{} is already frozen for {}; pass force=True to replace".format(submission.id, task.id))
    previous = json.loads(marker.read_text()) if marker.exists() else None  # replaced even if its artifacts are damaged
    data = _data_for(task, data, refresh=refresh)
    if len(data) < 2:
        raise ValueError("Not enough data to freeze {}".format(task.id))
    fitted_through = data.dates[-1]
    validation_start = data.dates[data.index_of(fitted_through - task.validation_length)]

    seeds = _seeds(build_policy(submission, task), seeds, "live")
    if (staging / COHORTS_DIR).exists():
        raise FileExistsError("{} holds archived cohorts without a complete freeze; restore them first".format(staging))
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    info = _load_info(task, submission, data, refresh=refresh)
    for seed in seeds:
        log.info("Freezing %s on %s: seed %d", submission.id, task.id, seed)
        policy = build_policy(submission, task)
        fit_policy(policy, data, validation_start, seed, info, crypto=task.source == "binance")
        seed_dir = staging / "seed{}".format(seed)
        seed_dir.mkdir()
        policy.save(seed_dir)
    freeze_record = {
        "schema": SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "code": code_identity(submission),
        "cohort": previous.get("cohort", 1) + 1 if previous else 1,
        "submission_sha256": submission.fingerprint(),
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
        "artifacts": _artifact_hashes(staging),
    }
    _write_json(staging / FREEZE_FILE, freeze_record, indent=2, sort_keys=False)
    if previous:
        # Old files go into cohorts/<n>/ inside the live directory, freeze.json last, so that every step can be
        # repeated after a crash and no state holds the only copy of a cohort in the staging directory.
        archived = directory / COHORTS_DIR / str(previous.get("cohort", 1))
        archived.mkdir(parents=True, exist_ok=True)
        for item in sorted(directory.iterdir(), key=lambda path: path.name == FREEZE_FILE):
            if item.name != COHORTS_DIR:
                shutil.move(str(item), archived / item.name)
        shutil.move(str(directory / COHORTS_DIR), staging / COHORTS_DIR)
    if directory.exists():
        shutil.rmtree(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(directory)
    return marker


def _finish_interrupted_freeze(directory: Path, staging: Path) -> bool:
    """Complete a re-freeze that stopped after archiving the old cohorts; ``True`` if it did.

    Then ``directory`` has no ``freeze.json`` but holds (or already handed to
    the finished ``staging`` directory) the earlier cohorts. Nothing is deleted
    except the empty shell of ``directory``.
    """
    if (directory / FREEZE_FILE).exists() or not (staging / FREEZE_FILE).exists():
        return False
    if not ((directory / COHORTS_DIR).exists() or (staging / COHORTS_DIR).exists()):
        return False
    if (directory / COHORTS_DIR).exists():
        shutil.move(str(directory / COHORTS_DIR), staging / COHORTS_DIR)
    if directory.exists():
        shutil.rmtree(directory)
    staging.rename(directory)
    log.warning("Completed an interrupted re-freeze of %s", directory)
    return True


def _require_task(submission: Submission, task: Task) -> None:
    if not submission.applies_to(task.id):
        raise ValueError(
            "{} is declared for {}".format(submission.id, ", ".join(submission.tasks))
        )


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
        if path.is_file() and path.name != FREEZE_FILE and path.relative_to(directory).parts[0] != COHORTS_DIR
    }


def verify_freeze(directory: Path) -> dict:
    """Load ``freeze.json`` and check that the stored artifacts are unchanged."""
    freeze_record = json.loads((directory / FREEZE_FILE).read_text())
    if freeze_record.get("schema", 1) not in READABLE_SCHEMAS:
        raise ValueError("{} has unsupported schema {!r}".format(directory / FREEZE_FILE, freeze_record["schema"]))
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
    _require_task(submission, task)
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
    cohort = frozen.get("cohort", 1)
    published = _publication(Path(out_root), submission.id, task.id, directory / FREEZE_FILE, cohort)
    if published is None:
        log.info("%s on %s: frozen but not yet published", submission.id, task.id)
        return None
    # The first evaluated session fills after the publication day.
    start = int(np.searchsorted(data.dates, max(fitted_through, as_day(published)), side="right"))
    if start >= len(data):
        log.info("%s on %s: no sessions after its publication on %s yet", submission.id, task.id, published)
        return None
    history = data.rows(0, start)
    info = _load_info(task, submission, data, refresh=refresh)
    runs = []
    for seed in frozen["seeds"]:
        policy = build_policy(submission, task)
        policy.load(directory / "seed{}".format(seed))
        runs.append((seed, _run(task, policy, data, start, None, history, None, info)))
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
    if info is not None:
        record["info_sha256"] = info.sha256()
    record["cohort"] = cohort
    frozen_code = (frozen.get("code") or {}).get("code_sha256")
    if frozen_code is None:
        record["code_identity"] = "unrecorded"  # frozen before schema 2: nothing to compare
    else:
        record["frozen_code"] = frozen["code"]
        record["code_changed"] = record["code"]["code_sha256"] != frozen_code
    return _store(live_path(out_root, task, submission, cohort), record)


def live_path(out_root, task: Task, submission: Submission, cohort: int = 1) -> Path:
    name = submission.id if cohort == 1 else "{}.cohort-{}".format(submission.id, cohort)
    return Path(out_root) / "live" / task.id / (name + ".json")


def _publication_key(submission_id: str, task_id: str, cohort: int) -> str:
    """``submission/task`` for cohort 1 (the keys already published), ``submission/task/<n>`` after."""
    key = "{}/{}".format(submission_id, task_id)
    return key if cohort == 1 else "{}/{}".format(key, cohort)


def _publication(out_root: Path, submission_id: str, task_id: str, marker: Path, cohort: int = 1) -> str | None:
    """When this exact freeze was published, or ``None`` if it has not been."""
    path = out_root / PUBLICATIONS_FILE
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get(_publication_key(submission_id, task_id, cohort))
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
        frozen = verify_freeze(marker.parent)
        key = _publication_key(marker.parent.parent.parent.name, marker.parent.name, frozen.get("cohort", 1))
        digest = _sha256_file(marker)
        if publications.get(key, {}).get("freeze_sha256") == digest:
            continue
        publications[key] = {"published_at": _utc(now), "freeze_sha256": digest}
        stamped.append(key)
    if stamped:
        _write_json(path, dict(sorted(publications.items())), indent=2)
    return stamped
