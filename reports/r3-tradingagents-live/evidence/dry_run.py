#!/usr/bin/env python3
"""Run one disposable ledger decision for each R3 stock and report usage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SUBMISSION_FILE = REPO / "submissions" / "tradingagents" / "submission.json"
TASKS = ("us-aapl", "us-googl", "us-amzn")
REQUIRED_ENV = ("OPENAI_API_KEY", "FRED_API_KEY", "SEC_EDGAR_USER_AGENT", "LEDGER_MONTHLY_BUDGET_USD")


def run(refresh: bool = False) -> int:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print("dry run requires local environment values for: {}".format(", ".join(missing)), file=sys.stderr)
        return 2
    try:
        budget = float(os.environ["LEDGER_MONTHLY_BUDGET_USD"])
        if budget <= 0:
            raise ValueError
    except ValueError:
        print("LEDGER_MONTHLY_BUDGET_USD must be a positive number", file=sys.stderr)
        return 2

    # Use the public ledger recording path so the same policy factory and any
    # constructor preflight run as they do for a live entry. Every file and
    # downloaded price cache is below this temporary directory.
    from rl.bench import ledger

    with tempfile.TemporaryDirectory(prefix="r3-dry-run-") as temporary:
        scratch = Path(temporary)
        submissions_root = scratch / "submissions"
        copied_submission = submissions_root / "tradingagents"
        copied_submission.mkdir(parents=True)
        shutil.copy2(SUBMISSION_FILE, copied_submission / "submission.json")
        os.environ["RL_PRICING_DATA"] = str(scratch / "data")
        written, failures = ledger.record_all(
            root=submissions_root,
            submission="tradingagents",
            budget=budget,
            refresh=refresh,
        )
        entries = {}
        for path in written:
            entry = json.loads(path.read_text())
            entries[entry["task"]] = entry

        errors = list(failures)
        decision_dates = set()
        for task in TASKS:
            entry = entries.get(task)
            if entry is None:
                errors.append("no entry produced for {}".format(task))
                continue
            decision_dates.add(entry["decision_date"])
            inputs = entry.get("inputs") or {}
            usage = inputs.get("token_usage") or {}
            totals = {
                "input": sum(row.get("input", 0) for row in usage.values()),
                "cached_input": sum(row.get("cached_input", 0) for row in usage.values()),
                "output": sum(row.get("output", 0) for row in usage.values()),
            }
            print("{} {}: signal={}, cost_usd={:.6f}, tokens={} ({}), reason={}".format(
                task, entry["decision_date"], inputs.get("signal", "unavailable"),
                entry.get("cost_usd", 0.0), json.dumps(totals, sort_keys=True),
                ", ".join("{}:{}".format(model, row.get("calls", 0)) for model, row in sorted(usage.items())) or "no usage reported",
                entry.get("reason") or "none"))
            if entry.get("reason") or not usage:
                errors.append("{} did not complete with reported token usage ({})".format(
                    task, entry.get("reason") or "usage missing"))
        if len(decision_dates) != 1:
            errors.append("the three decisions did not use one common session: {}".format(sorted(decision_dates)))
        print("scratch entries and price cache discarded")
        if errors:
            for error in errors:
                print("dry run failed: {}".format(error), file=sys.stderr)
            return 1
        print("three decisions completed for {}".format(next(iter(decision_dates))))
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refresh Yahoo prices in the scratch cache")
    return run(parser.parse_args(argv).refresh)


if __name__ == "__main__":
    raise SystemExit(main())
