#!/usr/bin/env python3
"""Run every registered R5 L2 arm on ``us-ndx80`` and write the records to ``--out-root``.

The pre-registration (§5.2) fixes the arms, five seeds (0-4) and the task. The
score files are downloaded and hash-checked into ``datasets/`` on first use,
converted to the M2 jsonl variants the arms declare, and then every arm is
fitted and tested by :func:`rl.bench.runner.evaluate` on the shared engine.
The two equal-weight baselines are run the same way, and QQQ and ``^NDX``
closes are written to ``<out-root>/baselines/us-ndx80.json`` with their
hashes. Nothing is written into the repository unless ``--out-root`` points
there; run outputs are never committed.

The pre-registration's §9 recall probe must have been run before a real run
that uses scores. ``--timesteps`` and ``--seeds`` are for a local smoke test
only: those runs are not the registered experiment.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scores  # noqa: E402

from rl.bench import runner  # noqa: E402
from rl.bench.data import _adjust_forward, load_task_data  # noqa: E402
from rl.bench.submission import discover  # noqa: E402
from rl.bench.tasks import get_task  # noqa: E402
from rl.marketdata.yahoodownloader import YahooDownloader  # noqa: E402

TASK = "us-ndx80"
#: The two equal-weight baselines are engine submissions; QQQ and ^NDX are
#: market series written beside the records.
BASELINES = ("equal-weight-buy-hold", "equal-weight-monthly")
INDEXES = ("QQQ", "^NDX")
#: The registered arm order (§5.2): P0, P10/P1/P01, C0, C10/C1/C01, the Llama
#: arms, the placebo arms and the same-day arms.
ARMS = (
    "r5-p0", "r5-p10", "r5-p1", "r5-p01",
    "r5-c0", "r5-c10", "r5-c1", "r5-c01",
    "r5-p10-l", "r5-c10-l",
    "r5-p10-s", "r5-c10-s",
    "r5-p10-d", "r5-c10-d",
)


def prepare_scores(arms, task, sessions, cache, *, refresh=False):
    """Build every declared jsonl variant that is missing; return variant manifests."""
    from rl.bench.info import validate_sources

    needed = {}
    for submission in arms:
        for spec in validate_sources(submission.info_sources):
            path = (submission.directory / spec["path"]).resolve()
            if path.exists() and not refresh:
                continue
            variant = path.stem
            if variant not in scores.VARIANTS:
                raise ValueError("{} declares an unknown score variant {}".format(submission.id, path))
            needed[variant] = path
    manifests = []
    for variant, path in needed.items():
        started = time.monotonic()
        manifest = scores.build(variant, task.tickers, sessions, cache, output=path, refresh=refresh)
        manifest["seconds"] = round(time.monotonic() - started, 1)
        manifests.append(manifest)
        print("scores {}: {} items, sha256 {}, {:.1f}s".format(
            variant, manifest["items"], manifest["sha256"], manifest["seconds"]))
    return manifests


def market_baselines(task, data, out_root):
    """QQQ and ^NDX closes on the record's sessions, with hashes, under ``out_root``."""
    first = max(0, data.index_of(task.test_start) - 1)  # the first decision session
    last = data.index_of(task.test_end)
    dates = [str(day) for day in data.dates[first:last]]
    series = {}
    for ticker in INDEXES:
        frame = YahooDownloader(str(task.start), str(task.test_end), [ticker]).fetch_unadjusted()
        if frame.empty:
            raise ValueError("Yahoo Finance returned no data for {}".format(ticker))
        adjusted = _adjust_forward(frame, task.start, task.test_end)
        closes = dict(zip(adjusted["date"], adjusted["close"]))
        values = [None if day not in closes else float(closes[day]) for day in dates]
        payload = {"ticker": ticker, "dates": dates, "close": values}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        series[ticker] = {**payload, "missing": sum(value is None for value in values), "sha256": digest}
    document = {
        "task": task.id,
        "window": {"first": dates[0], "last": dates[-1]},
        "source": "Yahoo Finance, forward dividend-adjusted from the task start like the task's prices",
        "series": series,
    }
    path = Path(out_root) / "baselines" / "{}.json".format(task.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    print("market baselines: " + ", ".join(
        "{} {} closes sha256 {}".format(ticker, len(dates), series[ticker]["sha256"][:12])
        for ticker in INDEXES))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", required=True, help="directory for the historical records")
    parser.add_argument("--seeds", type=int, default=5, help="seeds 0..N-1 (the registration uses 5)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--timesteps", type=int, help="override every arm's training budget (smoke runs only)")
    parser.add_argument("--cache", default=str(scores.DEFAULT_CACHE), help="raw download cache")
    parser.add_argument("--refresh-scores", action="store_true", help="download and convert the scores again")
    options = parser.parse_args(argv)
    if options.seeds < 1:
        parser.error("--seeds must be positive")

    task = get_task(TASK)
    found = discover(ROOT / "submissions")
    arms = []
    for arm in options.arms:
        if arm not in found:
            parser.error("missing submission {}".format(arm))
        submission = found[arm]
        if not submission.applies_to(task.id):
            parser.error("{} is not declared for {}".format(arm, task.id))
        if options.timesteps:
            params = {**submission.params, "timesteps": options.timesteps}
            submission = dataclasses.replace(submission, params=params)
        arms.append(submission)

    started = time.monotonic()
    data = load_task_data(task, end=task.test_end)
    print("data: {} sessions {} .. {} sha256 {}".format(
        len(data), data.dates[0], data.dates[-1], data.sha256()))
    print("data loaded in {:.1f}s".format(time.monotonic() - started))
    prepare_scores(arms, task, data.dates, Path(options.cache), refresh=options.refresh_scores)
    market_baselines(task, data, options.out_root)

    for name in BASELINES:
        if name not in found:
            parser.error("missing baseline submission {}".format(name))
        started = time.monotonic()
        path = runner.evaluate(task, found[name], data=data, seeds=[0], out_root=options.out_root)
        print("{}: {:.1f}s -> {}".format(name, time.monotonic() - started, path))

    seeds = list(range(options.seeds))
    for submission in arms:
        started = time.monotonic()
        path = runner.evaluate(task, submission, data=data, seeds=seeds, out_root=options.out_root)
        print("{}: seeds {} in {:.1f}s -> {}".format(
            submission.id, seeds, time.monotonic() - started, path))
    print("wrote {} arm records and {} baseline records to {}".format(
        len(arms), len(BASELINES), options.out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
