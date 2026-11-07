#!/usr/bin/env python3
"""Run R1's L2/L3 arms locally on the licensed data (pre-registration §5.2).

    python reports/r1-finrl-ensemble/evidence/l2_runs.py --data-file <toolkit CSV> --workers 8

The CSV comes from ``python -m rl.marketdata.sharadar`` and must hold the
stated 2016 Dow 30 plus WBA (arm C). It is vendor data: it and the run
directories stay under ``.cache/`` and are never committed.

Order (§5.2): arms A and B at seeds 0-4, then seeds 5-9, then arm C at seeds
0-4. Before them, ``--prefix-check`` runs arm A seed 0 to 2020-05-08 only;
:mod:`l2` compares it with the long run up to that date. A finished run
(``agent_accounts.csv`` present) is skipped, so the command can be resumed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / ".cache" / "r1-finrl-ensemble" / "l2-runs"

STATED = [  # Dow 30 on 2016-01-01 (§2.3)
    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
    "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
    "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM",
]
SHIPPED = sorted(set(STATED) - {"GE"} | {"WBA"})  # the authors' CSV: WBA for GE

ARMS = {
    "A": {"tickers": STATED, "turbulence": True},
    "B": {"tickers": STATED, "turbulence": False},
    "C": {"tickers": SHIPPED, "turbulence": True},
}
PREFIX_END = "2020-05-09"  # exclusive: the last session is 2020-05-08


def plan(prefix_check: bool) -> list[dict]:
    jobs = []
    if prefix_check:
        jobs.append({"name": "r1_A_s0_to20200508", "arm": "A", "seed": 0, "end": PREFIX_END})
    for seeds in (range(0, 5), range(5, 10)):
        for seed in seeds:
            for arm in ("A", "B"):
                jobs.append({"name": f"r1_{arm}_s{seed}", "arm": arm, "seed": seed})
    jobs += [{"name": f"r1_C_s{seed}", "arm": "C", "seed": seed} for seed in range(5)]
    return jobs


def command(job: dict, data_file: Path, results: Path, timesteps: str | None) -> list[str]:
    arm = ARMS[job["arm"]]
    if timesteps is None:
        cmd = [sys.executable, "-m", "rl.autotrain.r1"]
    else:  # smoke tests only: same driver with fewer steps
        cmd = [sys.executable, "-c",
               "import json,sys; from rl.autotrain.r1 import main;"
               f" import rl.autotrain.r1 as m; m.AUTHORS_TIMESTEPS = json.loads({timesteps!r}); main(sys.argv[1:])"]
    cmd += ["--data-file", str(data_file), "--seed", str(job["seed"]),
            "--run-name", job["name"], "--results-dir", str(results), "--tickers", *arm["tickers"]]
    if not arm["turbulence"]:
        cmd.append("--no-turbulence")
    if "end" in job:
        cmd += ["--end-date", job["end"]]
    return cmd


def run(job: dict, data_file: Path, results: Path, timesteps: str | None) -> tuple[str, int]:
    out = results / job["name"]
    if (out / "agent_accounts.csv").exists():
        return job["name"], 0
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    with open(out / "run.log", "w") as log:
        code = subprocess.call(command(job, data_file, results, timesteps), cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    return job["name"], code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-file", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=RUNS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-prefix-check", action="store_true")
    ap.add_argument("--only", nargs="+", help="run names to run (default: all)")
    ap.add_argument("--timesteps", help=argparse.SUPPRESS)  # JSON, smoke tests only
    args = ap.parse_args(argv)

    jobs = plan(not args.no_prefix_check)
    if args.only:
        jobs = [j for j in jobs if j["name"] in set(args.only)]
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "plan.json").write_text(json.dumps(
        {"data_file": str(args.data_file), "jobs": jobs, "smoke_timesteps": args.timesteps}, indent=2))
    failed = []
    # Submitted in plan order; workers pick them up in that order.
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, j, args.data_file, args.results, args.timesteps) for j in jobs]
        for future in cf.as_completed(futures):
            name, code = future.result()
            print(f"{name}: {'ok' if code == 0 else f'exit {code}'}", flush=True)
            if code:
                failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
