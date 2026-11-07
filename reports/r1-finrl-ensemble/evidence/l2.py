#!/usr/bin/env python3
"""Score R1's L2 and L3 runs (pre-registration §5.2-§8, §11).

    python reports/r1-finrl-ensemble/evidence/l2_runs.py --data-file <toolkit CSV>
    python reports/r1-finrl-ensemble/evidence/l2.py --data-file <toolkit CSV> \\
        --claims reports/r1-finrl-ensemble/claims.json

Reads the run directories of ``l2_runs.py``, recomputes the min-variance and
1/N baselines from the same price file, and takes DIA (total-return proxy) and
``^DJI`` (price index, crash dates) from Yahoo's chart API, cached under
``.cache/r1-finrl-ensemble`` with their hashes recorded in the summary.
Writes ``evidence/l2/summary.json``: metrics only, no prices and no account
values (whether the licence allows publishing equity curves is checked
separately, §12).

Arm A gives the verdicts (seeds 0-9, or those that finished). L2 is
2016-01-04 to 2020-05-08 (levels and gaps); L3 is 2020-05-11 to 2025-12-31
(gaps only). Arms B and C are reported as paired differences from arm A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l1  # noqa: E402  shared metrics, verdict rules, Holm, Table I
from l2_runs import ARMS  # noqa: E402
from rl.bench.scoring import _stationary_indices, deflated_sharpe  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".cache" / "r1-finrl-ensemble"
RUNS = CACHE / "l2-runs"
OUT = HERE / "l2"
FEE = 0.001  # per side (§5.2)
CAPITAL = 1_000_000
AGENTS = ("A2C", "PPO", "DDPG")
LEVELS = {"L2": (20160104, 20200508), "L3": (20200508, 20251231)}  # first value, last value
PREFIX_RUN = "r1_A_s0_to20200508"

# claim: (metric, claimed level or None, claimed gap, baseline)   §7
CLAIMS = {
    "ens-sharpe": ("sharpe", 1.30, 0.83, "dia"),
    "ens-sharpe-vs-minvar": ("sharpe", None, 0.85, "minvar"),
    "ens-sharpe-vs-agents": ("sharpe", None, 0.18, "best_agent"),
    "ens-cumulative-return": ("cumulative_return", 0.704, 0.318, "dia"),
    "ens-annual-return": ("annual_return", 0.130, 0.052, "dia"),
    "ens-max-drawdown": ("max_drawdown", -0.097, 0.274, "dia"),
    "ens-crash-2020": ("crash_drawdown", None, None, "dia"),
}
BASELINE_NAMES = {
    "dia": "DIA adjusted close (total-return proxy for the DJIA), Yahoo",
    "minvar": "long-only minimum variance of the same names, 252-session covariance, every 63 sessions, next-open fills, 0.1% per side",
    "best_agent": "best standalone agent by seed-mean Sharpe, same seeds",
}


def _date(x) -> int:
    return int(str(x).replace("-", "")[:8])


def yahoo(symbol: str, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Daily close and adjusted close from Yahoo's chart API (cached)."""
    path = CACHE / f"yahoo-{symbol.replace('^', '')}.json"
    query = urllib.parse.urlencode({"period1": 1230768000, "period2": 1767312000, "interval": "1d",
                                    "events": "history", "includeAdjustedClose": "true"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol, safe='')}?{query}"
    if refresh or not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "RL-pricing-reproducibility-lab/1.0"})
        with urllib.request.urlopen(req, timeout=90) as response:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.read())
    raw = path.read_bytes()
    result = json.loads(raw)["chart"]["result"][0]
    dates = [int(datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%d")) for t in result["timestamp"]]
    frame = pd.DataFrame({
        "close": result["indicators"]["quote"][0]["close"],
        "adjclose": result["indicators"]["adjclose"][0]["adjclose"],
    }, index=dates, dtype=float).dropna()
    meta = {"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    return frame, meta


def load_runs(root: Path) -> dict[str, dict[int, pd.DataFrame]]:
    """``arm -> seed -> frame`` (index: date; columns ENSEMBLE, A2C, PPO, DDPG)."""
    arms: dict[str, dict[int, pd.DataFrame]] = {}
    for path in sorted(root.glob("r1_[ABC]_s*/agent_accounts.csv")):
        name = path.parent.name
        if name == PREFIX_RUN:
            continue
        arm, seed = name.split("_")[1], int(name.split("_s")[1])
        frame = pd.read_csv(path)
        frame.index = frame.pop("date").map(_date)
        arms.setdefault(arm, {})[seed] = frame
    return arms


def prefix_check(root: Path, arms) -> dict | None:
    """§5.2: the run ended on 2020-05-08 must equal the long run up to that date."""
    path = root / PREFIX_RUN / "agent_accounts.csv"
    if not path.exists() or 0 not in arms.get("A", {}):
        return None
    short = pd.read_csv(path)
    short.index = short.pop("date").map(_date)
    long = arms["A"][0].loc[short.index]
    diff = float(np.max(np.abs(short.to_numpy() / long.to_numpy() - 1.0)))
    return {"sessions": len(short), "max_relative_difference": diff, "identical": diff < 1e-9}


def baselines(data_file: Path, tickers: list[str], dates: pd.Index) -> dict[str, pd.Series]:
    """Min-variance and 1/N on the arm's names, from the same price file."""
    from rl.trade.baselines import buy_and_hold, equal_weight_rebalanced, min_variance_portfolio

    prices = pd.read_csv(data_file)
    prices = prices[prices["tic"].isin(tickers)].copy()
    prices["date"] = prices["date"].astype(str)
    first = f"{dates[0]:08d}"
    first = f"{first[:4]}-{first[4:6]}-{first[6:]}"
    last = f"{dates[-1]:08d}"
    last = f"{last[:4]}-{last[4:6]}-{last[6:]}"
    history = prices[prices["date"] < first]
    trade = prices[(prices["date"] >= first) & (prices["date"] <= last)]
    out = {
        "minvar": min_variance_portfolio(trade, history, CAPITAL, FEE),
        "one_over_n": buy_and_hold(trade, CAPITAL, FEE),
        "one_over_n_monthly": equal_weight_rebalanced(trade, CAPITAL, FEE, rebalance_every=21),
    }
    series = {}
    for key, frame in out.items():
        s = pd.Series(frame["account_value"].to_numpy(dtype=float), index=frame["date"].map(_date))
        series[key] = s.reindex(dates)
        if series[key].isna().any():
            raise SystemExit(f"{key} baseline misses {int(series[key].isna().sum())} run dates")
    return series


def crash_dates(index_close: pd.Series) -> tuple[int, int]:
    """Peak and trough of the largest peak-to-trough fall of a price series."""
    v = index_close.to_numpy()
    running = np.maximum.accumulate(v)
    trough = int(np.argmax(1 - v / running))
    peak = int(np.argmax(v[: trough + 1]))
    return int(index_close.index[peak]), int(index_close.index[trough])


def returns(values: np.ndarray) -> np.ndarray:
    return values[..., 1:] / values[..., :-1] - 1.0


def score_level(level, ens, agents, base, crash, ppy_sessions, n_boot=l1.N_BOOT, chunk=200, seed=0):
    """Claims of one level for arm A. ``ens``: (seeds, values); ``agents``: name -> (seeds, values);
    ``base``: name -> (values,); ``crash``: (first, last) positions in the value arrays."""
    R = returns(ens)
    AR = {a: returns(v) for a, v in agents.items()}
    B = {k: returns(v) for k, v in base.items()}
    n = R.shape[1]
    block = round(n ** (1 / 3))
    obs_ens = l1.path_metrics(R, ppy_sessions)
    obs_agents = {a: l1.path_metrics(r, ppy_sessions) for a, r in AR.items()}
    obs_base = {k: l1.path_metrics(r, ppy_sessions) for k, r in B.items()}
    best = max(AGENTS, key=lambda a: obs_agents[a]["sharpe"].mean())
    c0, c1 = crash
    obs_ens["crash_drawdown"] = l1.depth(ens[:, c0 : c1 + 1])
    obs_base["dia"]["crash_drawdown"] = float(l1.depth(base["dia"][c0 : c1 + 1]))

    def baseline_value(metric, kind, metrics_base, metrics_agents):
        if kind == "best_agent":
            return metrics_agents[best][metric].mean(axis=0)
        return metrics_base[kind][metric]

    # Full-window bootstrap: one draw of dates for every seed, agent and baseline.
    rng = np.random.default_rng(seed)
    boot = {cid: {"level": np.empty(n_boot), "gap": np.empty(n_boot)} for cid in CLAIMS}
    for lo in range(0, n_boot, chunk):
        rows = min(chunk, n_boot - lo)
        idx = _stationary_indices(rng, rows, n, block)
        e = l1.path_metrics(R[:, idx], ppy_sessions)
        ag = {a: l1.path_metrics(r[:, idx], ppy_sessions) for a, r in AR.items()}
        bs = {k: l1.path_metrics(r[idx], ppy_sessions) for k, r in B.items()}
        for cid, (metric, _, _, kind) in CLAIMS.items():
            if metric == "crash_drawdown":
                continue
            mean = e[metric].mean(axis=0)
            boot[cid]["level"][lo : lo + rows] = l1.level(metric, mean)
            boot[cid]["gap"][lo : lo + rows] = l1.gap(metric, mean, baseline_value(metric, kind, bs, ag))
    # Crash: dates resampled within the crash window (addendum 2026-11-07 §4).
    CR, CB = R[:, c0:c1], B["dia"][c0:c1]
    m = CR.shape[1]
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(rng, n_boot, m, round(m ** (1 / 3)))
    ce = l1.path_metrics(CR[:, idx], m)["max_drawdown"]  # (seeds, n_boot)
    cb = l1.path_metrics(CB[idx], m)["max_drawdown"]
    boot["ens-crash-2020"]["gap"][:] = cb - ce.mean(axis=0)

    claims, p = {}, {}
    for cid, (metric, c, c_d, kind) in CLAIMS.items():
        if level == "L3":
            c = None  # §5.3: gap form only
        if metric == "crash_drawdown":
            c_d = 0.5 * obs_base["dia"]["crash_drawdown"]
            base_value = obs_base["dia"]["crash_drawdown"]
        else:
            base_value = float(np.mean(baseline_value(metric, kind, obs_base, obs_agents)))
        per_seed = [float(l1.level(metric, x)) for x in obs_ens[metric]]
        d_hat = float(l1.gap(metric, np.mean(obs_ens[metric]), base_value))
        gp = boot[cid]["gap"]
        l_d, u_d = (float(x) for x in np.percentile(gp, [2.5, 97.5]))
        p_d = float(np.mean(gp <= 0))
        p[cid] = p_d
        entry = {
            "metric": metric,
            "gap": {"claimed": c_d, "estimate": d_hat, "ci95": [l_d, u_d], "p_one_sided": p_d},
            "baseline": {"name": BASELINE_NAMES[kind] + (f" ({best})" if kind == "best_agent" else ""),
                         "value": float(l1.level(metric, base_value))},
            "per_seed": per_seed,
            "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        }
        u_m = None
        if c is not None:
            l_m, u_m = (float(x) for x in np.percentile(boot[cid]["level"], [2.5, 97.5]))
            entry["level"] = {"claimed": c, "estimate": float(np.mean(per_seed)), "ci95": [l_m, u_m],
                              "exceeds": bool(np.mean(per_seed) > c)}
        entry["verdict"] = l1.verdict(d_hat, l_d, u_d, p_d, c_d, per_seed, u_m, c)
        claims[f"{cid}-{level.lower()}"] = entry
    for cid, adj in l1.holm({f"{k}-{level.lower()}": v for k, v in p.items()}).items():
        claims[cid]["gap"]["p_holm"] = adj
    context = {
        "best_agent": best,
        "agents_seed_mean": {a: {k: float(v.mean()) for k, v in obs_agents[a].items()} for a in AGENTS},
        "baselines": {k: {m_: float(v) for m_, v in obs_base[k].items()} for k in obs_base},
        "returns": int(n), "sessions_spanned": int(ppy_sessions), "bootstrap_block": int(block),
        "sharpe_of_seed_averaged_returns": float(l1.path_metrics(R.mean(axis=0), ppy_sessions)["sharpe"]),
    }
    return claims, context


def paired_difference(arm, ref, ppy_sessions, n_boot=l1.N_BOOT, chunk=200, seed=0) -> dict:
    """Seed-mean metric of ``arm`` minus ``ref`` on shared seeds, paired date bootstrap."""
    Ra, Rr = returns(arm), returns(ref)
    n = Ra.shape[1]
    block = round(n ** (1 / 3))
    obs_a, obs_r = l1.path_metrics(Ra, ppy_sessions), l1.path_metrics(Rr, ppy_sessions)
    rng = np.random.default_rng(seed)
    keys = ("sharpe", "annual_return", "max_drawdown", "annual_volatility")
    draws = {k: np.empty(n_boot) for k in keys}
    for lo in range(0, n_boot, chunk):
        rows = min(chunk, n_boot - lo)
        idx = _stationary_indices(rng, rows, n, block)
        a, r = l1.path_metrics(Ra[:, idx], ppy_sessions), l1.path_metrics(Rr[:, idx], ppy_sessions)
        for k in keys:
            draws[k][lo : lo + rows] = a[k].mean(axis=0) - r[k].mean(axis=0)
    return {k: {"difference": float(obs_a[k].mean() - obs_r[k].mean()),
                "ci95": [float(x) for x in np.percentile(draws[k], [2.5, 97.5])]} for k in keys}


def flat_sessions(values: np.ndarray, dates) -> dict:
    """Sessions with unchanged equity per seed and how many distinct sets there are (R0, §11)."""
    r = returns(values)
    sets = [tuple(int(d) for d, z in zip(dates[1:], row == 0.0) if z) for row in r]
    return {"per_seed": [len(s) for s in sets], "distinct_sets": len(set(sets))}


def picks(root: Path, names: list[str]) -> dict:
    """Picked agent per quarter for the first 18 windows against Table I (§11)."""
    runs = {}
    for name in names:
        path = root / name / "ensemble_summary.csv"
        if path.exists():
            runs[name] = pd.read_csv(path)["Model Used"].astype(str).str.upper().tolist()[:18]
    agree = {k: sum(a == b for a, b in zip(v, l1.TABLE_I)) for k, v in runs.items()}
    seqs = list(runs.values())
    between = [sum(a == b for a, b in zip(seqs[i], seqs[j]))
               for i in range(len(seqs)) for j in range(i + 1, len(seqs))]
    return {"runs": runs, "agreement_with_table_i": agree,
            "mean_pairwise_agreement_between_seeds": float(np.mean(between)) if between else None}


def update_claims(path: Path, claims: dict) -> None:
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
        seeds = ", ".join(f"{x:.3f}" for x in result["per_seed"])
        entry["note"] = (
            f"{entry['level']}, arm A, {len(result['per_seed'])} seeds ({seeds}); gap to the "
            f"{result['baseline']['name'].split(' (')[0].split(',')[0]} {g['estimate']:+.3f} "
            f"[{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] against {g['claimed']:+.3f} claimed, "
            f"one-sided p {g['p_one_sided']:.4f} (Holm {g['p_holm']:.4f}); preregistration §7."
        )
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--data-file", type=Path, required=True, help="the toolkit CSV the runs used")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--claims", type=Path, help="fill this claims.json's L2 and L3 rows")
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--allow-prefix-mismatch", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    arms = load_runs(args.runs)
    if "A" not in arms:
        raise SystemExit(f"no arm A runs under {args.runs}")
    prefix = prefix_check(args.runs, arms)
    if prefix is not None and not prefix["identical"] and not args.allow_prefix_mismatch:
        raise SystemExit(f"prefix check failed ({prefix}); §5.2: L2 then uses separate runs ended on 2020-05-08")
    dia, dia_meta = yahoo("DIA", args.refresh_yahoo)
    dji, dji_meta = yahoo("^DJI", args.refresh_yahoo)
    rate, rate_meta = l1.dtb3(args.refresh_yahoo)

    A = arms["A"]
    seeds = sorted(A)
    dates = A[seeds[0]].index
    for s in seeds:
        if not A[s].index.equals(dates):
            raise SystemExit(f"arm A seed {s} has different dates")
    base_all = baselines(args.data_file, ARMS["A"]["tickers"], dates)
    base_all["dia"] = dia["adjclose"].reindex(dates)
    if base_all["dia"].isna().any():
        raise SystemExit("DIA misses run dates")

    claims, context = {}, {}
    for level, (first, last) in LEVELS.items():
        sel = (dates >= first) & (dates <= last)
        d = dates[sel]
        sessions = len(d) - 1
        ens = np.stack([A[s].loc[d, "ENSEMBLE"].to_numpy() for s in seeds])
        agents = {a: np.stack([A[s].loc[d, a].to_numpy() for s in seeds]) for a in AGENTS}
        base = {k: base_all[k].loc[d].to_numpy() for k in ("dia", "minvar")}
        peak, trough = crash_dates(dji["close"].loc[first:last])
        crash = (int(np.searchsorted(d, peak)), int(np.searchsorted(d, trough)))
        lv_claims, ctx = score_level(level, ens, agents, base, crash, sessions)
        ctx["crash_window"] = {"peak": peak, "trough": trough, "returns": crash[1] - crash[0]}
        ctx["one_over_n"] = {k: {m: float(v) for m, v in l1.path_metrics(returns(base_all[k].loc[d].to_numpy()), sessions).items()}
                             for k in ("one_over_n", "one_over_n_monthly")}
        djp = dji["close"].reindex(d).to_numpy()
        ctx["dji_price_index"] = {m: float(v) for m, v in l1.path_metrics(returns(djp), sessions).items()}
        ctx["dividend_bridge_annual_return"] = (ctx["baselines"]["dia"]["annual_return"]
                                                - ctx["dji_price_index"]["annual_return"])
        ctx["flat_sessions_arm_A"] = flat_sessions(ens, d)
        ctx["sharpe_excess_dtb3"] = {"ensemble_seed_mean": float(l1.excess_sharpe(ens, d, rate).mean()),
                                     "dia": float(l1.excess_sharpe(base["dia"], d, rate))}
        for arm in ("B", "C"):
            shared = sorted(set(seeds) & set(arms.get(arm, {})))
            if not shared:
                continue
            other = np.stack([arms[arm][s].loc[d, "ENSEMBLE"].to_numpy() for s in shared])
            mine = np.stack([A[s].loc[d, "ENSEMBLE"].to_numpy() for s in shared])
            ctx[f"arm_{arm}_minus_A"] = {"seeds": shared, **paired_difference(other, mine, sessions)}
            if arm == "B":
                ctx["flat_sessions_arm_B"] = flat_sessions(other, d)
        # DSR with N = 3 (arms A, B, C): variance of per-session Sharpe across every run.
        all_runs = [returns(arms[a][s].loc[d, "ENSEMBLE"].to_numpy()) for a in arms for s in arms[a]]
        sr = np.array([r.mean() / r.std(ddof=1) for r in all_runs])
        ctx["deflated_sharpe_trials_3"] = [deflated_sharpe(returns(e), 3, float(np.var(sr, ddof=1))) for e in ens]
        claims.update(lv_claims)
        context[level] = ctx

    summary = {
        "schema_version": 1,
        "report": "r1-finrl-ensemble",
        "levels": ["L2", "L3"],
        "arm_a_seeds": seeds,
        "arms": {a: sorted(v) for a, v in arms.items()},
        "prefix_check": prefix,
        "inputs": {"data_file_sha256": hashlib.sha256(args.data_file.read_bytes()).hexdigest(),
                   "yahoo_DIA": dia_meta, "yahoo_DJI": dji_meta, "fred_DTB3": rate_meta},
        "bootstrap": {"method": "stationary (Politis-Romano)", "resamples": l1.N_BOOT, "seed": 0},
        "claims": claims,
        "context": context,
        "picks_arm_A": picks(args.runs, [f"r1_A_s{s}" for s in seeds]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.claims:
        update_claims(args.claims, claims)
    for cid, e in claims.items():
        g = e["gap"]
        print(f"{cid:28s} gap {g['estimate']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] "
              f"p={g['p_one_sided']:.3f} -> {e['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
