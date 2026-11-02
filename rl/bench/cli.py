"""``rl-pricing bench ...``: run, freeze and rank benchmark submissions."""

from __future__ import annotations

import inspect
import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from rl.bench import ledger, leaderboard, monthly, runner, sweep, verdicts
from rl.bench.submission import SUBMISSIONS_DIR, Submission, discover
from rl.bench.tasks import TASKS, get_task

log = logging.getLogger("rl.bench")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="rl-pricing bench",
        description="Honest, live-forward-tested benchmark for portfolio policies.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    common = ArgumentParser(add_help=False)
    common.add_argument("--submissions-root", default=str(SUBMISSIONS_DIR), help="directory of submissions")
    common.add_argument("--out-root", default=str(runner.LEADERBOARD_DIR), help="directory of result records")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("tasks", help="list benchmark tasks")

    data = commands.add_parser("data", help="download or verify a task's data and print its coverage")
    data.add_argument("task", choices=sorted(TASKS))
    data.add_argument("--refresh", action="store_true", help="ignore the local cache")
    data.add_argument("--end", help="last session is before this date (default: all available)")

    run = commands.add_parser("run", parents=[common], help="historical track: one submission on one task")
    run.add_argument("--task", required=True, choices=sorted(TASKS))
    run.add_argument("--submission", required=True)
    run.add_argument("--seeds", type=int, help="number of seeds (default: 5 if stochastic, else 1)")

    causal = commands.add_parser("check-causal", parents=[common], help="check truncation and future poisoning")
    causal.add_argument("--task", required=True, choices=sorted(TASKS))
    causal.add_argument("--submission", required=True)
    causal.add_argument("--seed", type=int, default=0, help="fixed fit and decision seed")

    run_all = commands.add_parser(
        "run-all",
        parents=[common],
        help="historical track: every submission on a task (omit --task for every task with a live track)",
    )
    run_all.add_argument("--task", choices=sorted(TASKS))
    run_all.add_argument("--kind", help="only submissions of this kind")
    run_all.add_argument("--seeds", type=int, help="number of seeds (default: 5 if stochastic, else 1)")
    run_all.add_argument(
        "--stale", action="store_true",
        help="skip submissions whose record was computed from their current definition (CI after a merge)",
    )

    freeze = commands.add_parser(
        "freeze",
        parents=[common],
        help="live track: fit on all data and freeze (omit --task/--submission to freeze every unfrozen pair)",
    )
    freeze.add_argument("--task", choices=sorted(TASKS))
    freeze.add_argument("--submission")
    freeze.add_argument("--seeds", type=int, help="number of seeds (default: 3 if stochastic, else 1)")
    freeze.add_argument("--force", action="store_true", help="replace an existing freeze (a new cohort)")
    freeze.add_argument(
        "--stale", action="store_true", help="also re-freeze, as a new cohort, submissions changed since their freeze"
    )
    freeze.add_argument("--refresh", action="store_true", help="ignore the local data cache")

    live = commands.add_parser("live", parents=[common], help="live track: score frozen policies on new sessions")
    live.add_argument("--task", choices=sorted(TASKS))
    live.add_argument("--submission")
    live.add_argument("--refresh", action="store_true", help="ignore the local data cache")

    commands.add_parser(
        "publish",
        parents=[common],
        help="stamp new freezes with the current time (CI on main only); live scoring starts after it",
    )

    board = commands.add_parser("leaderboard", parents=[common], help="render LEADERBOARD.md and summary.json")
    board.add_argument("--readme", default="README.md", help="README whose leaderboard markers are refreshed")
    board.add_argument("--no-readme", action="store_true")

    commands.add_parser(
        "validate", parents=[common],
        help="check submissions (paid ledger policies: class and constructor arguments only)",
    )

    # Commands that register themselves and carry their handler.
    ledger.add_cli(commands, common)
    monthly.add_cli(commands, common)
    sweep.add_cli(commands, common)
    verdicts.add_cli(commands, common)
    return parser


def _submission(root, submission_id: str) -> Submission:
    found = discover(root)
    if submission_id not in found:
        raise SystemExit("Unknown submission {!r}; found {}".format(submission_id, sorted(found) or "none"))
    return found[submission_id]


def _seeds(count):
    if count is not None and count < 1:
        raise SystemExit("--seeds must be positive")
    return None if count is None else range(count)


def _frozen_pairs(root, task_id, submission_id):
    pairs = []
    for marker in sorted(Path(root).glob("*/frozen/*/freeze.json")):
        sub_id, frozen_task = marker.parent.parent.parent.name, marker.parent.name
        if task_id in (None, frozen_task) and submission_id in (None, sub_id) and frozen_task in TASKS:
            pairs.append((get_task(frozen_task), sub_id))
    return pairs


def _cmd_tasks(options) -> int:
    print("{:<16} {:<9} {:>7} {:<23} {:>6}  {:>4}  {}".format(
        "id", "source", "assets", "test window", "fee", "live", "title"))
    for task in TASKS.values():
        print("{:<16} {:<9} {:>7} {:<23} {:>5.2f}%  {:>4}  {}".format(
            task.id, task.source, len(task.tickers), "{} .. {}".format(task.test_start, task.test_end),
            task.transaction_cost * 100, "yes" if task.live else "no", task.title))
    return 0


def _cmd_data(options) -> int:
    from rl.bench.data import load_task_data

    task = get_task(options.task)
    data = load_task_data(task, end=options.end, refresh=options.refresh)
    print("task      {}".format(task.id))
    print("sessions  {} ({} .. {})".format(len(data), data.dates[0], data.dates[-1]))
    print("sha256    {}".format(data.sha256()))
    print("{:<10} {:>8} {:>12} {:>12}".format("ticker", "coverage", "first", "last"))
    import numpy as np

    for i, ticker in enumerate(data.tickers):
        listed = np.flatnonzero(np.isfinite(data.close[:, i]))
        if listed.size:
            print("{:<10} {:>7.1%} {:>12} {:>12}".format(
                ticker, listed.size / len(data), data.dates[listed[0]], data.dates[listed[-1]]))
        else:
            print("{:<10} {:>7.1%} {:>12} {:>12}".format(ticker, 0.0, "-", "-"))
    return 0


def _cmd_run(options) -> int:
    submission = _submission(options.submissions_root, options.submission)
    path = runner.evaluate(get_task(options.task), submission, seeds=_seeds(options.seeds), out_root=options.out_root)
    print("Wrote {}".format(path))
    return 0


def _cmd_check_causal(options) -> int:
    from rl.bench.causal import check_causal

    submission = _submission(options.submissions_root, options.submission)
    try:
        cuts = check_causal(get_task(options.task), submission, seed=options.seed)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print("{}: truncation and future poisoning passed at {} (seed {})".format(
        submission.id, ", ".join(cuts), options.seed
    ))
    return 0


def _cmd_run_all(options) -> int:
    from rl.bench.data import load_task_data

    tasks = [get_task(options.task)] if options.task else [t for t in TASKS.values() if t.live]
    submissions = [
        s for s in discover(options.submissions_root).values()
        if not s.ledger and not s.is_retired() and options.kind in (None, s.kind)
    ]
    written, failures = 0, []
    for task in tasks:
        data = None
        for submission in submissions:
            if options.stale and runner.historical_is_current(task, submission, options.out_root):
                continue
            if data is None:
                data = load_task_data(task, end=task.test_end)
            try:
                path = runner.evaluate(
                    task, submission, data=data, seeds=_seeds(options.seeds), out_root=options.out_root
                )
            except Exception:
                log.exception("%s failed on %s", submission.id, task.id)
                failures.append("{}/{}".format(task.id, submission.id))
            else:
                written += 1
                print("Wrote {}".format(path))
    if not written and not failures:
        print("Nothing to do")
    if failures:
        print("Failed: {}".format(", ".join(failures)), file=sys.stderr)
        return 1
    return 0


def _should_freeze(task, submission, options) -> bool:
    if not (submission.frozen_dir(task.id) / runner.FREEZE_FILE).exists():
        return True
    return options.force or (options.stale and not runner.freeze_is_current(task, submission, options.out_root))


def _cmd_freeze(options) -> int:
    from rl.bench.data import load_task_data

    if options.task and options.submission:
        task, submission = get_task(options.task), _submission(options.submissions_root, options.submission)
        try:
            if submission.is_retired() or (options.stale and not _should_freeze(task, submission, options)):
                print("{} is retired or up to date on {}".format(submission.id, task.id))
                return 0
            path = runner.freeze(
                task, submission, seeds=_seeds(options.seeds), force=options.force or options.stale,
                refresh=options.refresh,
            )
        except Exception:
            log.exception("%s failed to freeze on %s", submission.id, task.id)
            print("Failed: {}/{}".format(task.id, submission.id), file=sys.stderr)
            return 1
        print("Froze {}".format(path))
        return 0
    tasks = [get_task(options.task)] if options.task else [t for t in TASKS.values() if t.live]
    submissions = (
        [_submission(options.submissions_root, options.submission)]
        if options.submission
        else [s for s in discover(options.submissions_root).values() if not s.ledger]
    )
    failures = []
    for task in tasks:
        data = None
        for submission in submissions:
            try:
                if submission.is_retired() or not _should_freeze(task, submission, options):
                    continue
                if data is None:
                    data = load_task_data(task, refresh=options.refresh)
                path = runner.freeze(
                    task, submission, data=data, seeds=_seeds(options.seeds), force=options.force or options.stale
                )
            except Exception:
                log.exception("%s failed to freeze on %s", submission.id, task.id)
                failures.append("{}/{}".format(task.id, submission.id))
            else:
                print("Froze {}".format(path))
    if failures:
        print("Failed: {}".format(", ".join(failures)), file=sys.stderr)
        return 1
    return 0


def _cmd_live(options) -> int:
    from rl.bench.data import load_task_data

    submissions = discover(options.submissions_root)
    failures = []
    loaded = {}
    for task, submission_id in _frozen_pairs(options.submissions_root, options.task, options.submission):
        if submission_id not in submissions or submissions[submission_id].is_retired() or not task.live:
            continue
        if task.id not in loaded:
            loaded[task.id] = load_task_data(task, refresh=options.refresh)
        try:
            path = runner.update_live(task, submissions[submission_id], data=loaded[task.id], out_root=options.out_root)
        except Exception:
            log.exception("%s failed on live %s", submission_id, task.id)
            failures.append("{}/{}".format(task.id, submission_id))
        else:
            print("Wrote {}".format(path) if path else "{}/{}: unpublished or no new sessions".format(task.id, submission_id))
    if failures:
        print("Failed: {}".format(", ".join(failures)), file=sys.stderr)
        return 1
    return 0


def _cmd_leaderboard(options) -> int:
    readme = None if options.no_readme else options.readme
    summary = leaderboard.write_leaderboard(options.out_root, readme=readme, submissions_root=options.submissions_root)
    print((Path(options.out_root) / "LEADERBOARD.md").read_text())
    print("{} tables written to {}".format(len(summary["sections"]), options.out_root))
    return 0


def _cmd_publish(options) -> int:
    stamped = runner.publish(options.submissions_root, out_root=options.out_root)
    print("Published {}".format(", ".join(stamped)) if stamped else "Nothing new to publish")
    return 0


def _cmd_validate(options) -> int:
    found = discover(options.submissions_root)
    failures = []
    for submission in found.values():
        for task in TASKS.values():
            try:
                if submission.ledger:
                    ledger.declared_max_cost(submission)
                if submission.kind == "llm" and submission.ledger:
                    # Paid graphs require credentials; record performs the live preflight.
                    inspect.signature(submission.policy_class()).bind(**runner.policy_params(submission))
                else:
                    runner.build_policy(submission, task)
            except Exception as error:
                failures.append("{} on {}: {}".format(submission.id, task.id, error))
    for message in failures:
        print(message, file=sys.stderr)
    print("{} submissions, {} failures".format(len(found), len(failures)))
    return 1 if failures else 0


_COMMANDS = {
    "tasks": _cmd_tasks,
    "data": _cmd_data,
    "run": _cmd_run,
    "check-causal": _cmd_check_causal,
    "run-all": _cmd_run_all,
    "freeze": _cmd_freeze,
    "live": _cmd_live,
    "leaderboard": _cmd_leaderboard,
    "publish": _cmd_publish,
    "validate": _cmd_validate,
}


def main(argv=None) -> int:
    """Run a bench subcommand; returns the process exit code."""
    options = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = getattr(options, "handler", None)
    return handler(options) if handler is not None else _COMMANDS[options.command](options)
