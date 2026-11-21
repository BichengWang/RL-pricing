"""Declared hyperparameter sweeps, selected on validation data only.

``sweep`` takes a base submission and a *declared* grid of parameter
overrides. Every configuration is fitted on the training window of each task
and scored on that task's validation window ``[validation_start,
test_start)`` for each seed. The scoring is policy-agnostic: the fitted
policy is simulated over the validation sessions with the task's cost model
and scored by its per-period Sharpe ratio. Worker processes are only ever
given sessions before ``task.test_start``, so no configuration's test
performance can be computed, selected on or leaked.

One configuration is selected for all tasks: the one with the highest mean,
over tasks, of the rank of its seed-averaged validation Sharpe among the
grid (rank ``len(grid)`` is best, ties share the average rank, remaining ties
go to the earliest declared configuration). Only the selected configuration
is then evaluated on the historical track (``runner.evaluate``). The new
submission records ``trials = len(grid)``, so the leaderboard deflates its
Sharpe ratio for the search.

Outputs, under ``<submissions_root>/<out_id>/``:

* ``submission.json`` — the selected parameters, ``trials``, a description of
  the grid and of the selection rule;
* ``sweep.json`` — every configuration with its per-task, per-seed
  validation scores, the selection, failures, and timestamps.

Scores are appended to ``sweep-progress.jsonl`` as they finish, so an
interrupted sweep resumes where it stopped; the file is removed once
``sweep.json`` is complete.
"""

from __future__ import annotations

import concurrent.futures as futures
import dataclasses
import hashlib
import json
import logging
import multiprocessing
import os
import time
import traceback
from argparse import ArgumentParser
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from rl.bench import runner
from rl.bench.engine import simulate
from rl.bench.market import MarketData, as_day
from rl.bench.submission import SUBMISSIONS_DIR, Submission, discover
from rl.bench.tasks import TASKS, Task, get_task

log = logging.getLogger(__name__)

SCHEMA = 1
SWEEP_FILE = "sweep.json"
PROGRESS_FILE = "sweep-progress.jsonl"
SELECTION_RULE = (
    "Each configuration is fitted on the training window and scored on the validation window "
    "[validation_start, test_start) of every task, for every seed, by the per-period Sharpe ratio "
    "of the fitted policy simulated there with the task's costs. Scores are averaged over seeds, "
    "ranked within each task (the best of N configurations gets rank N, ties share the mean rank), "
    "and the single configuration with the highest mean rank across tasks is selected; remaining "
    "ties go to the earliest declared configuration. Test sessions are never loaded by the search."
)
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_ROUNDS = 2  # a job lost with a dying worker is retried once; job errors are reported, not retried

#: Declared grid for the reference RL agents (2 x 2 x 2 = 8 configurations).
#: The first value of every axis is the untuned default of the reference
#: submissions, so the default configuration is one of the candidates.
_RL_GRID = [
    {"learning_rate": lr, "net_arch": list(arch), "episode_length": length, "gamma": gamma}
    for lr in (3e-4, 1e-3)
    for arch in ((64, 64), (128, 128))
    for length, gamma in ((252, 0.99), (63, 0.95))
]
GRIDS = {"ppo": _RL_GRID, "sac": _RL_GRID}

RATIONALES = {
    algo: (
        "Learning rate: the SB3 default 3e-4 against 1e-3; daily-return rewards are noisy, so the "
        "question is whether faster learning finds better policies within the fixed timestep budget. "
        "Network width: the default [64, 64] against [128, 128], to test whether more capacity helps "
        "on 8-10 asset features or only overfits. Horizon: 252-session episodes with gamma 0.99 "
        "(default; effective horizon about 100 sessions) against 63-session (one quarter) episodes "
        "with gamma 0.95 (effective horizon about 20 sessions), because the reward is a per-session "
        "log return and a shorter credit horizon reduces variance. Timesteps, features, reward and "
        "checkpoint selection are held at the base submission's values. The grid is the full "
        "2 x 2 x 2 factorial and includes the untuned default. Note: SB3Policy itself keeps the "
        "checkpoint with the best validation Sharpe, so every configuration's validation score is "
        "an optimistic, checkpoint-selected number; the bias is the same for all configurations "
        "and the leaderboard's deflation uses trials = 8."
    )
    for algo in GRIDS
}


class SweepError(RuntimeError):
    """The sweep could not produce a valid selection."""


# --------------------------------------------------------------------------
# Scoring


def _digest(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _sharpe(returns: np.ndarray) -> float:
    spread = returns.std(ddof=1) if returns.size > 1 else 0.0
    return float(returns.mean() / spread) if spread > 1e-12 else 0.0


_STATE: dict = {"data": None}


def _init_worker(data) -> None:
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    _STATE["data"] = data
    _STATE["history"] = {}


def _history(task: Task) -> MarketData:
    """Sessions before ``task.test_start`` only: the search never sees the test window."""
    cached = _STATE["history"].get(task.id)
    if cached is None:
        provided = (_STATE["data"] or {}).get(task.id)
        if provided is not None:
            if provided.tickers != task.tickers:
                provided = provided.select(task.tickers)
            cached = provided.between(None, task.test_start)
        else:
            from rl.bench.data import load_task_data

            cached = load_task_data(task, end=task.test_start).between(None, task.test_start)
        _STATE["history"][task.id] = cached
    return cached


def _score(job: dict) -> dict:
    """Fit one (task, configuration, seed) and score it on the validation window."""
    key = {k: job[k] for k in ("task", "config", "seed")}
    started = time.time()
    try:
        task, submission = job["task_spec"], job["submission"]
        history = _history(task)
        validation_start = as_day(task.validation_start)
        start = history.index_of(validation_start)
        if not 0 < start < len(history):
            raise ValueError("No validation sessions in [{}, {})".format(task.validation_start, task.test_start))
        policy = runner.build_policy(submission, task)
        policy.fit(history, validation_start, job["seed"])
        policy.reset()
        backtest = simulate(history, policy.act, start=start, transaction_cost=task.transaction_cost)
        sharpe = _sharpe(backtest.returns)
        if not np.isfinite(sharpe):
            raise ValueError("Non-finite validation Sharpe")
        return {
            **key, "digest": job["digest"], "ok": True, "sharpe": sharpe,
            "validation_sessions": len(history) - start, "data_sha256": history.sha256(),
            "seconds": round(time.time() - started, 1), "finished_at": runner._utc(None),
        }
    except Exception as error:
        return {
            **key, "digest": job["digest"], "ok": False,
            "error": "{}: {}".format(type(error).__name__, error),
            "traceback": traceback.format_exc(limit=8),
            "seconds": round(time.time() - started, 1), "finished_at": runner._utc(None),
        }


def _evaluate(job: dict) -> dict:
    try:
        path = runner.evaluate(
            job["task_spec"], job["submission"], data=(_STATE["data"] or {}).get(job["task"]),
            seeds=job["seeds"], out_root=job["out_root"],
        )
        return {"task": job["task"], "ok": True, "path": str(path)}
    except Exception as error:
        return {
            "task": job["task"], "ok": False, "error": "{}: {}".format(type(error).__name__, error),
            "traceback": traceback.format_exc(limit=8),
        }


def _run_jobs(worker, jobs: list[dict], processes: int, data, on_result=None) -> list[dict]:
    """Run ``worker`` over ``jobs``; return one result per job, failures included.

    A job lost with a dying worker is retried once in a fresh pool; errors
    raised by a job are returned as failures. Nothing is dropped.
    """
    if processes <= 1 or len(jobs) <= 1:
        _init_worker(data)
        results = []
        for job in jobs:
            result = worker(job)
            results.append(result)
            if on_result:
                on_result(result)
        return results
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    pending = list(range(len(jobs)))
    final: dict[int, dict] = {}
    for attempt in range(_ROUNDS):
        last = attempt == _ROUNDS - 1
        failed = []
        with futures.ProcessPoolExecutor(
            max_workers=min(processes, len(pending)), mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(data,),
        ) as pool:
            running = {pool.submit(worker, jobs[i]): i for i in pending}
            for done in futures.as_completed(running):
                i = running[done]
                lost = False
                try:
                    result = done.result()
                except Exception as error:  # the worker died or the job could not be run at all
                    lost = isinstance(error, BrokenProcessPool)  # every job in flight is lost with it
                    result = {**{k: jobs[i].get(k) for k in ("task", "config", "seed")}, "ok": False,
                              "error": "worker lost: {}".format(error) if lost else repr(error)}
                if lost and not last:
                    log.warning("Retrying %s after: %s", result.get("task"), result["error"])
                    failed.append(i)
                else:
                    final[i] = result
                    if on_result:
                        on_result(result)
        pending = sorted(failed)
        if not pending:
            break
    return [final[i] for i in range(len(jobs))]


# --------------------------------------------------------------------------
# Selection


def _average_ranks(values: list[float]) -> list[float]:
    """Ranks 1..n, the largest value ranks n; ties share the mean rank."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks.tolist()


def _select(scores: dict[str, list[float | None]], n_configs: int) -> tuple[int | None, dict]:
    """The best configuration by mean rank over tasks; ``None`` if none is complete.

    ``scores[task][i]`` is the seed-averaged validation Sharpe of configuration
    ``i`` (``None`` if any seed failed). Incomplete configurations are not
    ranked or selectable.
    """
    eligible = [i for i in range(n_configs) if all(scores[t][i] is not None for t in scores)]
    ranks = {t: dict(zip(eligible, _average_ranks([scores[t][i] for i in eligible]))) for t in scores}
    mean_rank = {i: float(np.mean([ranks[t][i] for t in scores])) for i in eligible}
    if not eligible:
        return None, {"ranks": ranks, "mean_rank": mean_rank}
    best = max(eligible, key=lambda i: (round(mean_rank[i], 12), -i))
    return best, {"ranks": ranks, "mean_rank": mean_rank}


# --------------------------------------------------------------------------
# Output


def _describe(base: Submission, grid: list[dict], selected: dict, task_ids: list[str]) -> str:
    axes = {}
    for overrides in grid:
        for key, value in overrides.items():
            values = axes.setdefault(key, [])
            if value not in values:
                values.append(value)
    declared = "; ".join("{}: {}".format(key, json.dumps(values)) for key, values in axes.items())
    return (
        "{name} tuned over a declared grid of {n} configurations ({declared}), all other parameters as in "
        "the '{base}' submission. One configuration is shared by all tasks and was selected on validation "
        "data only: the highest mean across tasks ({tasks}) of the rank of its seed-averaged validation "
        "Sharpe ratio; the test window was never used for selection. Selected: {chosen}. trials = {n}. "
        "Every configuration's scores are in sweep.json."
    ).format(
        name=base.name, n=len(grid), declared=declared, base=base.id, tasks=", ".join(task_ids),
        chosen=json.dumps(selected, sort_keys=True),
    )


def _write(path: Path, document: dict) -> None:
    runner._write_json(path, document, indent=2, sort_keys=False)


def _load_progress(path: Path) -> dict[tuple, dict]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                result = json.loads(line)
                if result.get("ok"):
                    done[(result["task"], result["digest"], result["seed"])] = result
    return done


def sweep(
    base: Submission,
    grid: list[dict],
    tasks,
    *,
    seeds: int = 5,
    out_id: str,
    processes: int | None = None,
    submissions_root=SUBMISSIONS_DIR,
    out_root=runner.LEADERBOARD_DIR,
    data: dict[str, MarketData] | None = None,
    rationale: str = "",
) -> dict:
    """Run the declared sweep, select on validation, write the submission and records.

    ``grid`` is a list of parameter overrides applied to ``base.params``;
    ``tasks`` are tasks or task ids. ``data`` optionally maps task ids to
    panels (used instead of downloading; test sessions in it are never read
    by the search). Returns the ``sweep.json`` document.

    Raises :class:`SweepError` after writing ``sweep.json`` if no
    configuration completed every task and seed, or if a final evaluation
    failed; individual failures are always listed in ``sweep.json``.
    """
    started_at = runner._utc(None)
    task_list = [get_task(t) if isinstance(t, str) else t for t in tasks]
    if not grid or not task_list:
        raise ValueError("A sweep needs at least one configuration and one task")
    if len({t.id for t in task_list}) != len(task_list):
        raise ValueError("Duplicate tasks")
    if seeds < 1:
        raise ValueError("seeds must be positive")
    if not all(isinstance(config, dict) and config for config in grid):
        raise ValueError("Every configuration must be a non-empty dict of parameter overrides")
    params = [{**base.params, **config} for config in grid]
    digests = [_digest(p) for p in params]
    if len(set(digests)) != len(digests):
        raise ValueError("The grid contains duplicate configurations")
    json.dumps(params)
    out_dir = Path(submissions_root) / out_id
    if out_id == base.id or (out_dir / "submission.json").exists():
        raise FileExistsError("Submission {!r} already exists; choose a new out_id".format(out_id))
    if processes is None:
        processes = max(1, (os.cpu_count() or 2) - 1)

    candidates = [dataclasses.replace(base, id=out_id, params=p, directory=out_dir) for p in params]
    stochastic = candidates[0].policy_class().stochastic
    seed_list = list(range(seeds)) if stochastic else [0]

    progress_path = out_dir / PROGRESS_FILE
    done = _load_progress(progress_path)
    results: dict[tuple, dict] = {}
    jobs = []
    for task in task_list:
        for i, candidate in enumerate(candidates):
            for seed in seed_list:
                key = (task.id, digests[i], seed)
                if key in done:
                    results[(task.id, i, seed)] = done[key]
                    continue
                jobs.append({
                    "task": task.id, "task_spec": task, "config": i, "seed": seed,
                    "digest": digests[i], "submission": candidate,
                })
    log.info(
        "Sweep %s: %d configurations x %d tasks x %d seeds; %d jobs to run, %d resumed, %d processes",
        out_id, len(grid), len(task_list), len(seed_list), len(jobs), len(results), processes,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(jobs)
    counter = {"n": 0}

    def record(result: dict) -> None:
        counter["n"] += 1
        if result["ok"]:
            with progress_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            log.info("[%d/%d] %s config %s seed %s: validation Sharpe %.4f (%.0fs)",
                     counter["n"], total, result["task"], result["config"], result["seed"],
                     result["sharpe"], result["seconds"])
        else:
            log.error("[%d/%d] %s config %s seed %s FAILED: %s",
                      counter["n"], total, result["task"], result["config"], result["seed"], result.get("error"))

    fresh = _run_jobs(_score, jobs, processes, data, on_result=record) if jobs else []
    failures = []
    for job, result in zip(jobs, fresh):
        if result["ok"]:
            results[(job["task"], job["config"], job["seed"])] = result
        else:
            failures.append({k: result.get(k) for k in ("task", "config", "seed", "error", "traceback")})

    # Per-config, per-task scores.
    scores: dict[str, list[float | None]] = {t.id: [] for t in task_list}
    detail: list[dict] = [{} for _ in grid]
    for task in task_list:
        for i in range(len(grid)):
            found = [results.get((task.id, i, s)) for s in seed_list]
            values = [r["sharpe"] if r else None for r in found]
            complete = all(v is not None for v in values)
            mean = float(np.mean(values)) if complete else None
            scores[task.id].append(mean)
            detail[i][task.id] = {
                "sharpe_mean": mean,
                "sharpe_per_seed": {str(s): v for s, v in zip(seed_list, values)},
                "failed_seeds": [s for s, v in zip(seed_list, values) if v is None],
                "seconds": round(sum(r["seconds"] for r in found if r), 1),
            }
    best, ranking = _select(scores, len(grid))

    document = {
        "schema": SCHEMA,
        "submission": out_id,
        "base": base.id,
        "started_at": started_at,
        "finished_at": None,
        "tasks": [t.id for t in task_list],
        "seeds": seed_list,
        "metric": "per-period (not annualised) Sharpe ratio of the fitted policy on [validation_start, test_start)",
        "validation_windows": {t.id: [t.validation_start, t.test_start] for t in task_list},
        "validation_data_sha256": {
            t.id: next((r["data_sha256"] for (task_id, _, _), r in results.items() if task_id == t.id), None)
            for t in task_list
        },
        "selection_rule": SELECTION_RULE,
        "rationale": rationale,
        "configs": [
            {
                "index": i,
                "overrides": grid[i],
                "params": params[i],
                "validation": detail[i],
                "eligible": i in ranking["mean_rank"],
                "ranks": {t: ranking["ranks"][t].get(i) for t in ranking["ranks"]},
                "mean_rank": ranking["mean_rank"].get(i),
                "selected": i == best,
            }
            for i in range(len(grid))
        ],
        "selection": None
        if best is None
        else {"index": best, "overrides": grid[best], "params": params[best], "mean_rank": ranking["mean_rank"][best]},
        "failures": failures,
        "records": {},
    }
    if best is None:
        document["finished_at"] = runner._utc(None)
        _write(out_dir / SWEEP_FILE, document)
        raise SweepError(
            "No configuration completed every task and seed ({} failed jobs); see {}".format(
                len(failures), out_dir / SWEEP_FILE
            )
        )

    final = dataclasses.replace(base, id=out_id, name="{} (tuned)".format(base.name), params=params[best],
                                trials=len(grid), directory=out_dir)
    final = dataclasses.replace(
        final, description=_describe(base, grid, grid[best], [t.id for t in task_list])
    )
    spec = {
        "name": final.name, "kind": final.kind, "policy": final.policy, "params": final.params,
        "trials": final.trials, "authors": list(final.authors), "url": final.url,
        "description": final.description,
    }
    # The records are computed from ``final``: the written definition must load back to it.
    for name in ("tasks", "ledger", "info_sources", "model_cutoffs", "contamination", "retired"):
        if getattr(final, name):
            spec[name] = getattr(final, name)
    _write(out_dir / "submission.json", spec)
    _write(out_dir / SWEEP_FILE, document)
    if failures:
        log.error("%d scoring jobs failed; incomplete configurations were excluded from selection", len(failures))

    evaluations = _run_jobs(
        _evaluate,
        [{"task": t.id, "task_spec": t, "submission": final, "seeds": seed_list, "out_root": str(out_root)}
         for t in task_list],
        min(processes, len(task_list)), data,
    )
    for result in evaluations:
        document["records"][result["task"]] = (
            result["path"] if result["ok"] else {"error": result["error"], "traceback": result["traceback"]}
        )
    document["finished_at"] = runner._utc(None)
    _write(out_dir / SWEEP_FILE, document)
    broken = [r["task"] for r in evaluations if not r["ok"]]
    if broken:
        raise SweepError("Evaluation of the selected configuration failed on {}; see {}".format(
            ", ".join(broken), out_dir / SWEEP_FILE))
    if progress_path.exists():
        progress_path.unlink()
    return document


# --------------------------------------------------------------------------
# CLI


def add_cli(subparsers, common) -> None:
    """Register ``sweep`` on a bench subparsers action; ``common`` holds --submissions-root/--out-root."""
    parser = subparsers.add_parser(
        "sweep", parents=[common],
        help="declared hyperparameter sweep selected on validation only; writes <out-id> and its records",
    )
    parser.add_argument("--base", required=True, help="submission whose parameters are the base")
    parser.add_argument("--out-id", required=True, help="id of the new tuned submission")
    parser.add_argument("--grid", help="named declared grid: {}".format(", ".join(sorted(GRIDS))))
    parser.add_argument("--grid-file", help="JSON list of parameter-override objects")
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASKS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--processes", type=int, help="worker processes (default: cpu count - 1)")
    parser.set_defaults(handler=_cmd_sweep)


def _cmd_sweep(options) -> int:
    found = discover(options.submissions_root)
    if options.base not in found:
        raise SystemExit("Unknown submission {!r}; found {}".format(options.base, sorted(found) or "none"))
    if bool(options.grid) == bool(options.grid_file):
        raise SystemExit("Give exactly one of --grid and --grid-file")
    if options.grid:
        if options.grid not in GRIDS:
            raise SystemExit("Unknown grid {!r}; choose from {}".format(options.grid, sorted(GRIDS)))
        grid, rationale = GRIDS[options.grid], RATIONALES[options.grid]
    else:
        grid, rationale = json.loads(Path(options.grid_file).read_text()), ""
    document = sweep(
        found[options.base], grid, options.tasks, seeds=options.seeds, out_id=options.out_id,
        processes=options.processes, submissions_root=options.submissions_root, out_root=options.out_root,
        rationale=rationale,
    )
    chosen = document["selection"]
    print("Selected config {} of {}: {}".format(chosen["index"], len(grid), json.dumps(chosen["overrides"])))
    for task_id, path in document["records"].items():
        print("{}: {}".format(task_id, path))
    return 1 if document["failures"] else 0


def main(argv=None) -> int:
    """``python -m rl.bench.sweep``: the sweep command without the full bench CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = ArgumentParser(prog="python -m rl.bench.sweep")
    common = ArgumentParser(add_help=False)
    common.add_argument("--submissions-root", default=str(SUBMISSIONS_DIR))
    common.add_argument("--out-root", default=str(runner.LEADERBOARD_DIR))
    add_cli(parser.add_subparsers(dest="command", required=True), common)
    options = parser.parse_args(argv)
    return options.handler(options)


if __name__ == "__main__":
    raise SystemExit(main())
