#!/usr/bin/env python3
"""Score R1's L1 runs: the authors' unmodified code, five unseeded runs.

Implements the pre-registration's §6 metrics and §7 verdict rules with the
choices fixed in ``addendum-2026-11-07.md`` (dates of the account-value rows,
annual return over sessions spanned, ``^DJI.csv`` paired by date, the crash
bootstrap within its window, Holm over five gaps, picks against Table I).

    python reports/r1-finrl-ensemble/evidence/fetch_inputs.py
    gh run download <run id> -D .cache/r1-finrl-ensemble/l1-runs
    python reports/r1-finrl-ensemble/evidence/l1.py .cache/r1-finrl-ensemble/l1-runs --workflow-run <run id>

Reads the authors' checkout that ``fetch_inputs.py`` caches and checks its
hashes. Writes ``evidence/l1/summary.json`` and ``evidence/l1/account_values.csv``
(the runs' account values only; no vendor prices).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rl.bench.scoring import _stationary_indices, deflated_sharpe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "5dbb44dfcb8a01a94762dc44fca2bcb2c626d1dc"
AUTHORS = ROOT / ".cache" / "r1-finrl-ensemble" / f"FinRL-Trading-{PIN}"
OUT = HERE / "l1"

REBALANCE = 63
FIRST, CUT = 20160104, 20200508
CRASH = (20200212, 20200323)  # ^DJI.csv peak and trough in the window (§6)
PPY = 252
N_BOOT = 10_000
CHUNK = 500

TABLE_I = [  # arXiv 2511.12120v1, p. 6 (addendum §6)
    "PPO", "DDPG", "DDPG", "PPO", "PPO", "A2C", "PPO", "DDPG", "PPO",
    "DDPG", "A2C", "A2C", "DDPG", "PPO", "PPO", "A2C", "A2C", "A2C",
]

# id: (metric, claimed level or None, claimed gap)   (§7; drawdowns as depth)
CLAIMS = {
    "ens-sharpe-l1": ("sharpe", 1.30, 0.83),
    "ens-cumulative-return-l1": ("cumulative_return", 0.704, 0.318),
    "ens-annual-return-l1": ("annual_return", 0.130, 0.052),
    "ens-max-drawdown-l1": ("max_drawdown", -0.097, 0.274),
    "ens-crash-2020-l1": ("crash_drawdown", None, None),  # c_d = half the DJIA's crash depth
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs(authors: Path) -> dict:
    manifest = json.loads((HERE / "hashes.json").read_text())
    files = next(e["files"] for e in manifest["authors_repository"] if "files" in e)
    expected = {f["path"]: f["sha256"] for f in files}
    out = {}
    for rel in ("done_data.csv", "data/^DJI.csv"):
        got = sha256(authors / rel)
        if got != expected[rel]:
            raise SystemExit(f"{rel}: sha256 {got} does not match hashes.json {expected[rel]}")
        out[rel] = got
    return out


def trade_dates(authors: Path) -> np.ndarray:
    """``unique_trade_date`` exactly as ``run_DRL.py`` builds it."""
    df = pd.read_csv(authors / "done_data.csv", index_col=0, usecols=[0, 1])
    d = df["datadate"]
    return d[(d > 20151001) & (d <= 20200707)].unique()


def windows(u: np.ndarray) -> list[int]:
    return list(range(2 * REBALANCE, len(u), REBALANCE))


def load_run(run_dir: Path, u: np.ndarray) -> pd.Series:
    """Account values by date with the 17 seam rows dropped (addendum §1).

    Row 0 of window k's file is the previous window's d62 value (or the
    initial $1,000,000 at d0 for the first window); rows 1-62 are d1-d62.
    """
    dates, values, previous = [], [], None
    for k, i in enumerate(windows(u)):
        col = pd.read_csv(run_dir / f"account_value_trade_ensemble_{i}.csv", index_col=0).iloc[:, 0]
        v = col.to_numpy(dtype=float)
        w = u[i - REBALANCE : i]
        if len(v) != REBALANCE or len(w) != REBALANCE:
            raise ValueError(f"{run_dir.name} window {i}: {len(v)} rows for {len(w)} sessions")
        if previous is None:
            dates.append(int(w[0]))
            values.append(v[0])
        elif not math.isclose(v[0], previous, rel_tol=1e-9):
            raise ValueError(f"{run_dir.name} window {i}: seam {v[0]} != previous last value {previous}")
        dates.extend(int(x) for x in w[1:])
        values.extend(v[1:])
        previous = v[-1]
    return pd.Series(values, index=dates, name=run_dir.name)


def djia(authors: Path) -> pd.Series:
    df = pd.read_csv(authors / "data/^DJI.csv")
    idx = df["Date"].str.replace("-", "").astype(int)
    return pd.Series(df["Adj Close"].to_numpy(dtype=float), index=idx, name="djia")


def depth(paths: np.ndarray) -> np.ndarray:
    """Max drawdown depth (positive) of value paths along the last axis."""
    return np.max(1.0 - paths / np.maximum.accumulate(paths, axis=-1), axis=-1)


def path_metrics(r: np.ndarray, sessions: int) -> dict[str, np.ndarray]:
    """§6 metrics of return arrays along the last axis (any leading shape)."""
    growth = np.cumprod(1.0 + r, axis=-1)
    ones = np.ones(r.shape[:-1] + (1,))
    sd = r.std(axis=-1, ddof=1)
    cum = growth[..., -1] - 1.0
    return {
        "sharpe": math.sqrt(PPY) * np.divide(r.mean(axis=-1), sd, out=np.zeros_like(sd), where=sd > 0),
        "cumulative_return": cum,
        "annual_return": (1.0 + cum) ** (PPY / sessions) - 1.0,
        "annual_volatility": math.sqrt(PPY) * sd,
        "max_drawdown": depth(np.concatenate([ones, growth], axis=-1)),
    }


def gap(metric: str, ens: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Larger is better: drawdown gaps are baseline depth minus ensemble depth."""
    return base - ens if metric in ("max_drawdown", "crash_drawdown") else ens - base


def level(metric: str, value):
    """Level oriented larger-is-better: a drawdown is reported as minus its depth."""
    return -value if metric == "max_drawdown" else value


def bootstrap(R: np.ndarray, b: np.ndarray, sessions: int, block: int, metrics, seed: int = 0) -> dict:
    """Paired stationary bootstrap of seed-mean levels and gaps (§6).

    One draw of dates per resample is applied to every run and the baseline.
    """
    n = R.shape[1]
    rng = np.random.default_rng(seed)
    out = {m: {"level": np.empty(N_BOOT), "gap": np.empty(N_BOOT)} for m in metrics}
    for lo in range(0, N_BOOT, CHUNK):
        rows = min(CHUNK, N_BOOT - lo)
        idx = _stationary_indices(rng, rows, n, block)
        ens = path_metrics(R[:, idx], sessions)  # (runs, rows)
        base = path_metrics(b[idx], sessions)  # (rows,)
        for m in metrics:
            key = "max_drawdown" if m == "crash_drawdown" else m
            mean = ens[key].mean(axis=0)
            out[m]["level"][lo : lo + rows] = level(m, mean)
            out[m]["gap"][lo : lo + rows] = gap(m, mean, base[key])
    return out


def verdict(d_hat, l_d, u_d, p_d, c_d, m_runs=None, u_m=None, c=None) -> str:
    """§7 rules, applied in order; names as in ``rl.bench.verdicts``."""
    has_level = c is not None
    if p_d < 0.05 and u_d >= c_d and (not has_level or u_m >= c):
        return "holds"
    if l_d <= 0 < c_d <= u_d:
        return "inconclusive"
    if (d_hat <= 0 and u_d < c_d) or (has_level and all(m < c for m in m_runs) and u_m < c):
        return "does-not-hold"
    return "partly"


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=p.get)
    out, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[key]))
        out[key] = running
    return out


VAL_RE = re.compile(r"^(A2C|PPO) Sharpe Ratio:\s+(\S+)", re.M)


def picks(run_dir: Path, u: np.ndarray) -> list[str]:
    """Rebuild the code's pick per quarter (addendum §6)."""
    logged = {"A2C": [], "PPO": []}
    for name, value in VAL_RE.findall((run_dir / "run_DRL.log").read_text(errors="replace")):
        logged[name].append(float(value))
    out = []
    for k, i in enumerate(windows(u)):
        v = pd.read_csv(run_dir / f"account_value_validation_{i}.csv", index_col=0).iloc[:, 0]
        r = v.pct_change(1)
        ddpg = (4 ** 0.5) * r.mean() / r.std()
        a2c, ppo = logged["A2C"][k], logged["PPO"][k]
        if ppo >= a2c and ppo >= ddpg:
            out.append("PPO")
        elif a2c > ppo and a2c > ddpg:
            out.append("A2C")
        else:
            out.append("DDPG")
    return out


def update_claims(path: Path, claims: dict) -> None:
    """Write the L1 results into ``claims.json`` (level where there is one, else gap)."""
    entries = json.loads(path.read_text())
    for entry in entries:
        result = claims.get(entry["id"])
        if result is None:
            continue
        g, lv = result["gap"], result.get("level")
        shown = lv if lv is not None else g
        if lv is None:
            entry["claimed"] = round(g["claimed"], 4)
        entry["ours"] = round(shown["estimate"], 4)
        entry["ci95"] = [round(x, 4) for x in shown["ci95"]]
        entry["verdict"] = result["verdict"]
        runs = ", ".join(f"{x:.3f}" for x in result["per_run"])
        entry["note"] = (
            f"L1, the authors' code, 5 unseeded runs ({runs}); gap to the DJIA "
            f"{g['estimate']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] against "
            f"{g['claimed']:+.3f} claimed, one-sided p {g['p_one_sided']:.4f} "
            f"(Holm {g['p_holm']:.4f}); preregistration §7 with addendum 2026-11-07."
        )
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, help="directory holding r1-l1-run-<n>/ artifacts")
    ap.add_argument("--authors", type=Path, default=AUTHORS)
    ap.add_argument("--workflow-run", type=int, action="append", default=[],
                    help="Actions run id(s) the artifacts came from")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--claims", type=Path, help="fill this claims.json's L1 rows")
    args = ap.parse_args(argv)

    inputs = check_inputs(args.authors)
    u = trade_dates(args.authors)
    run_dirs = sorted(p for p in args.runs.glob("r1-l1-run-*") if p.is_dir())
    if not run_dirs:
        raise SystemExit(f"no r1-l1-run-* directories under {args.runs}")
    runs = pd.concat([load_run(p, u) for p in run_dirs], axis=1)
    dji = djia(args.authors)
    missing = runs.index.difference(dji.index)
    if len(missing):
        raise SystemExit(f"^DJI.csv lacks {len(missing)} value dates, first {missing[0]}")

    window = runs.loc[FIRST:CUT]
    sessions = int(((u >= FIRST) & (u <= CUT)).sum()) - 1
    V = window.to_numpy().T  # (runs, values)
    R = V[:, 1:] / V[:, :-1] - 1.0
    bv = dji.loc[window.index].to_numpy()
    b = bv[1:] / bv[:-1] - 1.0
    n = R.shape[1]
    block = round(n ** (1 / 3))

    per_run = path_metrics(R, sessions)
    base = path_metrics(b, sessions)

    crash = window.loc[CRASH[0]:CRASH[1]]
    CV = crash.to_numpy().T
    CR = CV[:, 1:] / CV[:, :-1] - 1.0
    cb = dji.loc[crash.index].to_numpy()
    cbr = cb[1:] / cb[:-1] - 1.0
    per_run["crash_drawdown"] = depth(CV)
    base["crash_drawdown"] = float(depth(cb))
    crash_block = round(CR.shape[1] ** (1 / 3))

    boot = bootstrap(R, b, sessions, block, [m for m, *_ in CLAIMS.values() if m != "crash_drawdown"])
    boot.update(bootstrap(CR, cbr, CR.shape[1], crash_block, ["crash_drawdown"]))

    claims, p = {}, {}
    for cid, (metric, c, c_d) in CLAIMS.items():
        if metric == "crash_drawdown":
            c_d = 0.5 * base["crash_drawdown"]
        m_runs = [float(level(metric, x)) for x in per_run[metric]]
        d_hat = float(gap(metric, np.mean(per_run[metric]), base[metric]))
        lv, gp = boot[metric]["level"], boot[metric]["gap"]
        l_d, u_d = (float(x) for x in np.percentile(gp, [2.5, 97.5]))
        l_m, u_m = (float(x) for x in np.percentile(lv, [2.5, 97.5]))
        p_d = float(np.mean(gp <= 0))
        p[cid] = p_d
        entry = {
            "metric": metric,
            "gap": {"claimed": c_d, "estimate": d_hat, "ci95": [l_d, u_d], "p_one_sided": p_d},
            "baseline": {"name": "DJIA (^DJI.csv, price index), paired on the ensemble's dates",
                         "value": float(level(metric, base[metric]))},
            "per_run": m_runs,
            "verdict": verdict(d_hat, l_d, u_d, p_d, c_d, m_runs, u_m if c is not None else None, c),
        }
        if c is not None:
            entry["level"] = {"claimed": c, "estimate": float(np.mean(m_runs)), "ci95": [l_m, u_m],
                              "exceeds": bool(np.mean(m_runs) > c)}
        claims[cid] = entry
    for cid, adj in holm(p).items():
        claims[cid]["gap"]["p_holm"] = adj

    # Secondary (§11): DJIA on every session, picks, flat sessions, DSR, averaged series.
    full = dji.loc[FIRST:CUT].to_numpy()
    full_r = full[1:] / full[:-1] - 1.0
    djia_full = {k: float(v) for k, v in path_metrics(full_r, len(full_r)).items()}
    run_picks = {p_.name: picks(p_, u) for p_ in run_dirs}
    sharpe_pp = R.mean(axis=1) / R.std(axis=1, ddof=1)
    secondary = {
        "djia_every_session": djia_full,
        "per_run_metrics": {name: {k: float(v[j]) for k, v in per_run.items()}
                            for j, name in enumerate(window.columns)},
        "flat_sessions": {name: int(np.sum(R[j] == 0.0)) for j, name in enumerate(window.columns)},
        "picks": {
            "table_i": TABLE_I,
            "runs": run_picks,
            "agreement_with_table_i": {k: sum(a == b_ for a, b_ in zip(v, TABLE_I)) for k, v in run_picks.items()},
            "counts": {k: {a: v.count(a) for a in ("PPO", "A2C", "DDPG")} for k, v in run_picks.items()},
        },
        "deflated_sharpe_trials_3": {
            name: deflated_sharpe(R[j], 3, float(np.var(sharpe_pp, ddof=1)))
            for j, name in enumerate(window.columns)
        },
        "sharpe_of_run_averaged_returns": float(path_metrics(R.mean(axis=0), sessions)["sharpe"]),
    }

    envs = {}
    for p_ in run_dirs:
        env = p_ / "environment.txt"
        envs[p_.name] = env.read_text().splitlines() if env.exists() else None

    summary = {
        "schema_version": 1,
        "report": "r1-finrl-ensemble",
        "level": "L1",
        "authors_commit": PIN,
        "inputs_sha256": inputs,
        "workflow_runs": args.workflow_run,
        "runs": [p_.name for p_ in run_dirs],
        "environment": envs,
        "window": {"first": FIRST, "last": int(window.index[-1]), "values": int(len(window)),
                   "returns": int(n), "sessions_spanned": sessions, "bootstrap_block": block},
        "crash_window": {"first": CRASH[0], "last": CRASH[1], "returns": int(CR.shape[1]),
                         "bootstrap_block": crash_block},
        "bootstrap": {"method": "stationary (Politis-Romano)", "resamples": N_BOOT, "seed": 0},
        "claims": claims,
        "secondary": secondary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runs.index.name = "date"
    runs.to_csv(args.out / "account_values.csv", float_format="%.6f")
    if args.claims:
        update_claims(args.claims, claims)
    for cid, e in claims.items():
        g = e["gap"]
        print(f"{cid:28s} gap {g['estimate']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] "
              f"p={g['p_one_sided']:.3f} -> {e['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
