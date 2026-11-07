"""Recompute every number of report R0 from its evidence sources.

Sources (nothing else is read):

* ``evidence/paper-2021-share-env.json``: 50 runs (5 agents x 5 seeds x with and
  without the environment's turbulence rule) of the paper's share-trading
  environment, per-run Sharpe / annual return / max drawdown, plus 1/N.
* ``evidence/paper-2021-share-env-equity.json``: the daily account values of the
  same 50 runs and of 1/N (505 sessions); gives seed-averaged Sharpe, the paired
  bootstrap against 1/N, alpha/beta, and the unchanged-equity ("cash day") check.
  It must reproduce the per-run metrics of the file above (asserted).
* ``leaderboard/historical/paper-2021/*.json``: benchmark records (``paper-2021``
  task); metrics via :mod:`rl.bench.scoring`, aggregation via
  :mod:`rl.bench.leaderboard`.

Writes ``evidence/summary.json``, ``evidence/tables.md`` and ``claims.json``::

    python reports/r0-wang-zhang-2026/evidence/analyze.py

Interval convention: 95% Student-t interval of the mean over seeds (5 seeds,
4 degrees of freedom).  Verdict rule (post hoc, see ``preregistration.md``):

1. does not hold: the sign is opposite or no single seed reaches the claimed
   value (every metric here is signed so that larger is better; a drawdown of
   -0.20 is better than -0.30);
2. holds: the claimed value lies inside the interval and the seed-averaged
   Sharpe beats 1/N with bootstrap p < 0.05;
3. partly holds: otherwise (a seed reaches the claim, but the interval excludes
   it or significance is absent or untestable);
4. inconclusive: the setup run is not the paper's pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rl.bench import leaderboard, scoring

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ROOT = REPORT.parents[1]
SHARE = HERE / "paper-2021-share-env.json"
SHARE_EQUITY = HERE / "paper-2021-share-env-equity.json"
PPY = 252
HISTORICAL = ROOT / "leaderboard"
TASK = "paper-2021"

AGENTS = ("ddpg", "td3", "sac", "ppo", "a2c")
SETUPS = {
    "share-rule": "Share env + turbulence rule",
    "share-no-rule": "Share env, no rule",
    "benchmark": "Benchmark `paper-2021`",
}
#: two-sided 97.5% Student-t quantiles by degrees of freedom
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
ALPHA = 0.05
SYMBOL = {"holds": "✅", "partly": "◐", "does-not-hold": "✗", "inconclusive": "?"}


def seed_stats(values) -> dict:
    values = [float(v) for v in values]
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half = _T975[n - 1] * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "values": values,
        "mean": mean,
        "sd": sd,
        "ci95": [mean - half, mean + half],
        "min": min(values),
        "max": max(values),
        "median": float(np.median(values)),
    }


def share_env() -> tuple[dict, dict]:
    data = json.loads(SHARE.read_text())
    out: dict = {"share-rule": {}, "share-no-rule": {}}
    for run in sorted(data["runs"], key=lambda r: (r["agent"], r["seed"])):
        setup = "share-rule" if run["turbulence_rule"] else "share-no-rule"
        entry = out[setup].setdefault(run["agent"], {"sharpe": [], "annual_return": [], "max_drawdown": [], "seeds": []})
        entry["seeds"].append(run["seed"])
        for key in ("sharpe", "annual_return", "max_drawdown"):
            entry[key].append(run[key])
    stats = {
        setup: {
            agent: {key: seed_stats(vals[key]) for key in ("sharpe", "annual_return", "max_drawdown")}
            | {"seeds": vals["seeds"]}
            for agent, vals in agents.items()
        }
        for setup, agents in out.items()
    }
    return stats, data


def _daily(equity) -> np.ndarray:
    e = np.asarray(equity, dtype=float)
    return e[1:] / e[:-1] - 1


def share_env_equity() -> dict:
    """Seed-averaged metrics and cash-day check from the share-env daily equity curves."""
    data = json.loads(SHARE_EQUITY.read_text())
    reference = {(r["turbulence_rule"], r["agent"], r["seed"]): r for r in json.loads(SHARE.read_text())["runs"]}
    dates = data["dates"]
    one_n = _daily(data["equal_weight_buy_and_hold"])
    one_n_metrics = scoring.metrics(one_n, PPY)
    grouped: dict = {}
    cash_sets: dict = {True: [], False: []}
    worst = 0.0
    for run in data["runs"]:
        returns = _daily(run["equity"])
        grouped.setdefault(("share-rule" if run["turbulence_rule"] else "share-no-rule", run["agent"]), []).append(returns)
        cash_sets[run["turbulence_rule"]].append(tuple(dates[i + 1] for i in np.flatnonzero(np.abs(returns) < 1e-9)))
        # the curves must reproduce the per-run metrics stored in the other evidence file
        m, ref = scoring.metrics(returns, PPY), reference[(run["turbulence_rule"], run["agent"], run["seed"])]
        worst = max(worst, abs(m["sharpe"] - ref["sharpe"]), abs(m["cagr"] - ref["annual_return"]),
                    abs(m["max_drawdown"] - ref["max_drawdown"]))
    agents: dict = {"share-rule": {}, "share-no-rule": {}}
    for (setup, agent), runs in grouped.items():
        averaged = np.mean(runs, axis=0)
        m = scoring.metrics(averaged, PPY, benchmark=one_n)
        delta, p = scoring.sharpe_difference(averaged, one_n, PPY)
        agents[setup][agent] = {
            "avg_sharpe": m["sharpe"], "delta_sharpe": delta, "p_value": p,
            "alpha": m["alpha"], "alpha_t": m["alpha_t"], "beta": m["beta"],
        }
    cash = {}
    for key, flag in (("rule", True), ("no-rule", False)):
        sets = cash_sets[flag]
        cash[key] = {
            "runs": len(sets),
            "runs_without": sum(not s for s in sets),
            "distinct_sets": len(set(sets)),
            "sessions": sorted(set().union(*map(set, sets))),
            "per_run_counts": [len(s) for s in sets],
        }
    assert worst < 1e-3, worst
    return {
        "agents": agents,
        "one_over_n": {"sharpe": one_n_metrics["sharpe"], "annual_return": one_n_metrics["cagr"],
                       "max_drawdown": one_n_metrics["max_drawdown"]},
        "cash_days": cash,
        "sessions": len(dates),
        "max_abs_diff_vs_per_run_metrics": worst,
    }


def stationary_se(returns, ppy, *, n_boot=2000, seed=0) -> float:
    """Bootstrap standard error of the annualised Sharpe ratio (stationary bootstrap)."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    block = round(n ** (1 / 3))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        restart = rng.random(n) < 1.0 / block
        jump = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = jump[t] if restart[t] else (idx[t - 1] + 1) % n
        s = r[idx]
        out[b] = s.mean() / s.std(ddof=1) * math.sqrt(ppy)
    return float(out.std(ddof=1))


def benchmark_task() -> dict:
    records = {
        r["submission"]: r
        for r in leaderboard.load_records(HISTORICAL)
        if r["track"] == "historical" and r["task"] == TASK
    }
    section = next(s for s in leaderboard.build(HISTORICAL)["sections"] if s["task"] == TASK)
    rows = {row["id"]: row for row in section["rows"] if row["id"]}
    bench_row = next(row for row in section["rows"] if row["benchmark"])
    out: dict = {"agents": {}, "baselines": {}, "section": {k: section[k] for k in ("first", "last", "sessions", "periods_per_year")}}
    for sub_id, record in records.items():
        ppy = record["periods_per_year"]
        bench = np.asarray(record["benchmark"]["equity"], dtype=float)
        bench_returns = bench[1:] / bench[:-1] - 1
        per_seed = []
        for run in record["runs"]:
            eq = np.asarray(run["equity"], dtype=float)
            per_seed.append(scoring.metrics(eq[1:] / eq[:-1] - 1, ppy, benchmark=bench_returns))
        entry = {
            "name": record["name"],
            "kind": record["kind"],
            "seeds": [run["seed"] for run in record["runs"]],
            "sharpe": seed_stats([m["sharpe"] for m in per_seed]),
            "annual_return": seed_stats([m["cagr"] for m in per_seed]),
            "max_drawdown": seed_stats([m["max_drawdown"] for m in per_seed]),
            "delta_sharpe": rows[sub_id]["delta_sharpe"],
            "p_value": rows[sub_id]["p_value"],
            "alpha": rows[sub_id]["alpha"],
            "alpha_t": rows[sub_id]["alpha_t"],
            "beta": rows[sub_id]["beta"],
            "avg_sharpe": scoring.metrics(
                np.mean([np.asarray(r["equity"], dtype=float)[1:] / np.asarray(r["equity"], dtype=float)[:-1] - 1
                         for r in record["runs"]], axis=0), ppy)["sharpe"],
            "created": record["created"],
            "commit": record["commit"],
            "data_sha256": record["data"]["sha256"],
        }
        # the leaderboard aggregation must agree with the per-seed numbers computed here
        if sub_id in AGENTS or sub_id == "lstm-forecast":
            assert math.isclose(entry["sharpe"]["mean"], rows[sub_id]["sharpe"], rel_tol=1e-9), sub_id
            assert math.isclose(entry["max_drawdown"]["mean"], rows[sub_id]["max_drawdown"], rel_tol=1e-9), sub_id
        (out["agents"] if sub_id in AGENTS or sub_id == "lstm-forecast" else out["baselines"])[sub_id] = entry
    sample = records["ddpg"]
    bench = np.asarray(sample["benchmark"]["equity"], dtype=float)
    bench_returns = bench[1:] / bench[:-1] - 1
    ppy = sample["periods_per_year"]
    out["one_over_n"] = {
        "sharpe": bench_row["sharpe"],
        "annual_return": bench_row["cagr"],
        "max_drawdown": bench_row["max_drawdown"],
        "sharpe_bootstrap_se": stationary_se(bench_returns, ppy),
        # Lo (2002) with the per-session Sharpe (corrected 2026-11-07; report § Corrections).
        "sharpe_lo_se": math.sqrt((1 + bench_row["sharpe"] ** 2 / ppy / 2) / (len(bench_returns) / ppy)),
    }
    out["data_sha256"] = sample["data"]["sha256"]
    out["commit"] = sample["commit"]
    out["trading_days"] = len(bench_returns)
    return out


def judge(claimed, stats, *, p=None, comparable=True) -> str:
    """Verdict of one claim against seed statistics (module docstring: rules 1-4)."""
    if not comparable:
        return "inconclusive"
    if (claimed > 0 and stats["mean"] <= 0) or stats["max"] < claimed:
        return "does-not-hold"
    inside = stats["ci95"][0] <= claimed <= stats["ci95"][1]
    if inside and p is not None and p < ALPHA:
        return "holds"
    return "partly"


# claim id, agent, metric, claimed, quote, source
CLAIMS = (
    ("ddpg-annual-return", "ddpg", "annual_return", 0.5551,
     "DDPG attains the highest annual return (55.51%)", "Table 4 and Section 5.2 (p. 6); abstract: 55.5% (p. 1)"),
    ("ddpg-sharpe", "ddpg", "sharpe", 1.38,
     "DDPG attains the highest annual return (55.51%), cumulative return (142.26%), Sharpe ratio (1.38)",
     "Table 4 and Section 5.2 (p. 6); abstract (p. 1)"),
    ("td3-sharpe", "td3", "sharpe", 1.37,
     "TD3 and SAC offer a better balance between return and risk, with Sharpe ratios of 1.37 and 1.33",
     "Table 4 and Section 5.2 (p. 6)"),
    ("sac-sharpe", "sac", "sharpe", 1.33,
     "TD3 and SAC offer a better balance between return and risk, with Sharpe ratios of 1.37 and 1.33",
     "Table 4 and Section 5.2 (p. 6)"),
    ("ppo-sharpe", "ppo", "sharpe", 1.09,
     "PPO 27.89% 63.71% 1.09 −25.33% 0.05 0.91", "Table 4, PPO row (p. 6): Sharpe ratio 1.09"),
    ("a2c-sharpe", "a2c", "sharpe", 0.82,
     "A2C 20.47% 45.23% 0.82 −31.11% −0.03 1.02", "Table 4, A2C row (p. 6): Sharpe ratio 0.82"),
)
LSTM_CLAIMS = (
    ("lstm-max-drawdown", "max_drawdown", -0.0957,
     "its Sharpe ratio of 1.19 lies in the middle of the range, and it has the smallest maximum drawdown (−9.57%)",
     "Table 4 and Section 5.2 (p. 6); abstract: −9.6% (p. 1)"),
    ("lstm-sharpe", "sharpe", 1.19,
     "its Sharpe ratio of 1.19 lies in the middle of the range",
     "Table 4 and Section 5.2 (p. 6)"),
)


def _f(metric, value) -> str:
    if metric == "sharpe":
        return "{:.2f}".format(value)
    return "{:.1f}%".format(value * 100)


def _where(setup) -> str:
    return "benchmark" if setup == "benchmark" else "share"


def build() -> dict:
    share, share_raw = share_env()
    equity = share_env_equity()
    bench = benchmark_task()
    one_n_share = share_raw["equal_weight_buy_and_hold"]
    per_setup: dict = {}
    for setup in SETUPS:
        per_setup[setup] = {}
        for agent in AGENTS:
            if setup == "benchmark":
                e = bench["agents"][agent]
                per_setup[setup][agent] = {
                    "sharpe": e["sharpe"], "annual_return": e["annual_return"], "max_drawdown": e["max_drawdown"],
                    "p_value": e["p_value"], "delta_sharpe": e["delta_sharpe"], "alpha": e["alpha"], "alpha_t": e["alpha_t"],
                    "beta": e["beta"], "avg_sharpe": e["avg_sharpe"],
                }
            else:
                per_setup[setup][agent] = dict(share[setup][agent], **equity["agents"][setup][agent])
    summary = {
        "sources": {
            "share_env": "reports/r0-wang-zhang-2026/evidence/paper-2021-share-env.json",
            "benchmark_records": "leaderboard/historical/paper-2021/*.json",
        },
        "setups": SETUPS,
        "share_env_data_sha256": share_raw["data"]["sha256"],
        "share_env_rows": share_raw["data"]["rows"],
        "share_env_one_over_n": one_n_share,
        "share_env_equity": {k: v for k, v in equity.items() if k != "agents"},
        "benchmark": {k: v for k, v in bench.items() if k not in ("agents", "baselines")},
        "per_setup": per_setup,
        "lstm": bench["agents"]["lstm-forecast"],
        "baselines": bench["baselines"],
    }

    # pooled share-env statistics (all 50 runs)
    pooled = [r for r in share_raw["runs"]]
    returns = [r["annual_return"] for r in pooled]
    summary["share_env_pooled"] = {
        "runs": len(pooled),
        "annual_return_max": max(returns),
        "annual_return_median": float(np.median(returns)),
        "annual_return_max_run": max(pooled, key=lambda r: r["annual_return"]),
        "sharpe_beating_one_over_n": {
            setup: {
                agent: sum(v > one_n_share["sharpe"] for v in share[setup][agent]["sharpe"]["values"]) for agent in AGENTS
            }
            for setup in ("share-rule", "share-no-rule")
        },
        "sharpe_max": {s: {a: share[s][a]["sharpe"]["max"] for a in AGENTS} for s in ("share-rule", "share-no-rule")},
    }
    summary["benchmark_beating_one_over_n"] = {
        agent: sum(v > bench["one_over_n"]["sharpe"] for v in bench["agents"][agent]["sharpe"]["values"]) for agent in AGENTS
    }

    # verdicts per setup (same rule everywhere)
    verdicts: dict = {}
    for claim_id, agent, metric, claimed, _, _ in CLAIMS:
        verdicts[claim_id] = {}
        for setup in SETUPS:
            cell = per_setup[setup][agent]
            use_p = cell["p_value"] if metric == "sharpe" else None
            verdicts[claim_id][setup] = judge(claimed, cell[metric], p=use_p)
    summary["verdicts_by_setup"] = verdicts

    # smallest detectable effect over this window
    summary["detectability"] = {
        "years": bench["trading_days"] / 252,
        "sharpe_se_bootstrap": bench["one_over_n"]["sharpe_bootstrap_se"],
        "sharpe_se_lo2002": bench["one_over_n"]["sharpe_lo_se"],
        "one_sided_5pct_threshold": 1.645 * bench["one_over_n"]["sharpe_bootstrap_se"],
    }

    # claims.json: primary setup = share env + turbulence rule (see preregistration.md)
    claims = []
    primary = "share-rule"
    base_share = one_n_share["sharpe"]
    for claim_id, agent, metric, claimed, quote, source in CLAIMS:
        cell = per_setup[primary][agent][metric]
        bcell = per_setup["benchmark"][agent]
        other = per_setup["share-no-rule"][agent][metric]
        by = verdicts[claim_id]
        acell, ncell = per_setup[primary][agent], per_setup["share-no-rule"][agent]
        note = (
            "Share env + rule, 5 seeds: mean {m} (sd {sd}), best seed {mx}"
            "{base}. No rule: mean {om}, best {omx}. Benchmark task: mean {bm}, best {bmx}. "
            "{sig}Verdict by setup: rule {v1}, no rule {v2}, benchmark {v3}."
        ).format(
            m=_f(metric, cell["mean"]), sd=_f(metric, cell["sd"]), mx=_f(metric, cell["max"]),
            base="; 1/N Sharpe {:.2f}".format(base_share) if metric == "sharpe" else "; 1/N annual return {:.1f}%".format(
                one_n_share["annual_return"] * 100),
            om=_f(metric, other["mean"]), omx=_f(metric, other["max"]),
            bm=_f(metric, bcell[metric]["mean"]), bmx=_f(metric, bcell[metric]["max"]),
            v1=SYMBOL[by["share-rule"]], v2=SYMBOL[by["share-no-rule"]], v3=SYMBOL[by["benchmark"]],
            sig=(
                "Seed-averaged returns against 1/N, ΔSharpe (one-sided bootstrap p): rule {:+.2f} (p = {:.3f}), "
                "no rule {:+.2f} (p = {:.3f}), benchmark {:+.2f} (p = {:.3f}). ".format(
                    acell["delta_sharpe"], acell["p_value"], ncell["delta_sharpe"], ncell["p_value"],
                    bcell["delta_sharpe"], bcell["p_value"])
                if metric == "sharpe" else
                "No seed of any setup reaches the claimed value. " if by["share-rule"] == "does-not-hold" else ""
            ),
        )
        claims.append({
            "id": claim_id, "quote": quote, "source": source, "metric": metric,
            "claimed": claimed, "ours": round(cell["mean"], 4), "ci95": [round(x, 4) for x in cell["ci95"]],
            "level": "L2", "verdict": by[primary], "note": note.strip(),
        })
    lstm = summary["lstm"]
    for claim_id, metric, claimed, quote, source in LSTM_CLAIMS:
        cell = lstm[metric]
        claims.append({
            "id": claim_id, "quote": quote, "source": source, "metric": metric,
            "claimed": claimed, "ours": round(cell["mean"], 4), "ci95": [round(x, 4) for x in cell["ci95"]],
            "level": "L2",
            "verdict": judge(claimed, cell, comparable=False),
            "note": (
                "Benchmark task `paper-2021`, submission `lstm-forecast`, 5 seeds: mean {m} (sd {sd}), best seed {mx}; "
                "that submission forecasts next-session returns and holds the assets expected to rise, not the paper's "
                "price-forecast pipeline, and the authors' LSTM code was not run (see report, Deviations). The test cannot "
                "confirm or refute the paper's LSTM numbers; deciding it needs a run of the paper's LSTM pipeline "
                "(rl/forecast/lstm.py) under this engine."
            ).format(m=_f(metric, cell["mean"]), sd=_f(metric, cell["sd"]), mx=_f(metric, cell["max"])),
        })
    return {"summary": summary, "claims": claims, "share": share, "bench": bench}


def _table_header(columns) -> list[str]:
    return ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]


def tables(result: dict) -> str:
    s = result["summary"]
    per_setup = s["per_setup"]
    out: list[str] = []

    out += ["### Sharpe ratio, mean ± sd over 5 seeds [min, max]", ""]
    out += _table_header(["Agent", "Reported", *SETUPS.values()])
    reported = {"ddpg": 1.38, "td3": 1.37, "sac": 1.33, "ppo": 1.09, "a2c": 0.82}
    for agent in AGENTS:
        cells = []
        for setup in SETUPS:
            st = per_setup[setup][agent]["sharpe"]
            cells.append("{:.2f} ± {:.2f} [{:.2f}, {:.2f}]".format(st["mean"], st["sd"], st["min"], st["max"]))
        out.append("| {} | {:.2f} | {} |".format(agent.upper(), reported[agent], " | ".join(cells)))
    lstm = s["lstm"]["sharpe"]
    out.append("| LSTM forecast | 1.19 | – | – | {:.2f} ± {:.2f} [{:.2f}, {:.2f}] |".format(
        lstm["mean"], lstm["sd"], lstm["min"], lstm["max"]))
    out.append("| *1/N buy and hold* | – | *{:.2f}* | *{:.2f}* | *{:.2f}* |".format(
        s["share_env_one_over_n"]["sharpe"], s["share_env_one_over_n"]["sharpe"], s["benchmark"]["one_over_n"]["sharpe"]))

    out += ["", "### Annual return, mean ± sd over 5 seeds [min, max]", ""]
    out += _table_header(["Agent", "Reported", *SETUPS.values()])
    reported_return = {"ddpg": 0.5551, "td3": 0.4200, "sac": 0.3895, "ppo": 0.2789, "a2c": 0.2047}
    for agent in AGENTS:
        cells = []
        for setup in SETUPS:
            st = per_setup[setup][agent]["annual_return"]
            cells.append("{:.1f}% ± {:.1f} [{:.1f}%, {:.1f}%]".format(
                st["mean"] * 100, st["sd"] * 100, st["min"] * 100, st["max"] * 100))
        out.append("| {} | {:.2f}% | {} |".format(agent.upper(), reported_return[agent] * 100, " | ".join(cells)))
    lr = s["lstm"]["annual_return"]
    out.append("| LSTM forecast | 16.99% | – | – | {:.1f}% ± {:.1f} [{:.1f}%, {:.1f}%] |".format(
        lr["mean"] * 100, lr["sd"] * 100, lr["min"] * 100, lr["max"] * 100))
    out.append("| *1/N buy and hold* | – | *{:.1f}%* | *{:.1f}%* | *{:.1f}%* |".format(
        s["share_env_one_over_n"]["annual_return"] * 100, s["share_env_one_over_n"]["annual_return"] * 100,
        s["benchmark"]["one_over_n"]["annual_return"] * 100))

    out += ["", "### Maximum drawdown, mean over 5 seeds [worst, best]", ""]
    out += _table_header(["Agent", "Reported", *SETUPS.values()])
    reported_dd = {"ddpg": -0.3352, "td3": -0.2524, "sac": -0.2419, "ppo": -0.2533, "a2c": -0.3111}
    for agent in AGENTS:
        cells = []
        for setup in SETUPS:
            st = per_setup[setup][agent]["max_drawdown"]
            cells.append("{:.1f}% [{:.1f}%, {:.1f}%]".format(st["mean"] * 100, st["min"] * 100, st["max"] * 100))
        out.append("| {} | {:.2f}% | {} |".format(agent.upper(), reported_dd[agent] * 100, " | ".join(cells)))
    ld = s["lstm"]["max_drawdown"]
    out.append("| LSTM forecast | −9.57% | – | – | {:.1f}% [{:.1f}%, {:.1f}%] |".format(
        ld["mean"] * 100, ld["min"] * 100, ld["max"] * 100))
    out.append("| *1/N buy and hold* | – | – | – | *{:.1f}%* |".format(s["benchmark"]["one_over_n"]["max_drawdown"] * 100))

    out += ["", "### Sharpe against 1/N, seed-averaged returns (one-sided paired stationary bootstrap)", ""]
    out += ["Share env (A, B): from `paper-2021-share-env-equity.json`, alpha, beta and the Newey-West t of the",
            "seed-averaged daily returns. Benchmark (C): alpha, beta and t as in `leaderboard/LEADERBOARD.md`",
            "(per-seed values averaged). 'Seeds above' counts seeds whose own Sharpe exceeds that setup's 1/N Sharpe.", ""]
    out += _table_header(["Setup", "Agent", "Sharpe of seed-avg returns", "ΔSharpe vs 1/N", "p", "Alpha", "Alpha t (HAC)", "Beta", "Seeds above 1/N"])
    for setup in SETUPS:
        one_n = s["benchmark"]["one_over_n"]["sharpe"] if setup == "benchmark" else s["share_env_equity"]["one_over_n"]["sharpe"]
        for agent in (*AGENTS, "lstm-forecast"):
            if agent == "lstm-forecast":
                if setup != "benchmark":
                    continue
                e, name = s["lstm"], "LSTM forecast"
            else:
                e, name = per_setup[setup][agent], agent.upper()
            count = sum(v > one_n for v in e["sharpe"]["values"])
            out.append("| {} | {} | {:.2f} | {:+.2f} | {:.3f} | {:.1f}% | {:.1f} | {:.2f} | {}/5 |".format(
                SETUPS[setup], name, e["avg_sharpe"], e["delta_sharpe"], e["p_value"], e["alpha"] * 100,
                e["alpha_t"], e["beta"], count))

    cash = s["share_env_equity"]["cash_days"]
    out += ["", "### Sessions with unchanged equity (|daily return| < 1e-9)", ""]
    out += _table_header(["Setting", "Runs", "Runs with no such session", "Distinct sets of such sessions", "Sessions"])
    for key, label in (("rule", "Share env + turbulence rule"), ("no-rule", "Share env, no rule")):
        c = cash[key]
        out.append("| {} | {} | {} | {} | {} |".format(
            label, c["runs"], c["runs_without"], c["distinct_sets"], ", ".join(c["sessions"]) or "–"))

    out += ["", "### Per-seed Sharpe ratio", ""]
    out += _table_header(["Setup", "Agent", "Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4"])
    for setup in SETUPS:
        for agent in AGENTS:
            vals = per_setup[setup][agent]["sharpe"]["values"]
            out.append("| {} | {} | {} |".format(SETUPS[setup], agent.upper(), " | ".join("{:.2f}".format(v) for v in vals)))
    out.append("| Benchmark `paper-2021` | LSTM forecast | {} |".format(
        " | ".join("{:.2f}".format(v) for v in s["lstm"]["sharpe"]["values"])))

    out += ["", "### Per-seed annual return", ""]
    out += _table_header(["Setup", "Agent", "Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4"])
    for setup in SETUPS:
        for agent in AGENTS:
            vals = per_setup[setup][agent]["annual_return"]["values"]
            out.append("| {} | {} | {} |".format(SETUPS[setup], agent.upper(), " | ".join("{:.1f}%".format(v * 100) for v in vals)))

    out += ["", "### Verdict by setup (same rule everywhere)", ""]
    out += _table_header(["Claim", *SETUPS.values()])
    for claim_id, by in s["verdicts_by_setup"].items():
        out.append("| `{}` | {} |".format(claim_id, " | ".join(SYMBOL[by[k]] for k in SETUPS)))
    return "\n".join(out) + "\n"


def main() -> None:
    def _clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        return value

    result = build()
    (HERE / "summary.json").write_text(json.dumps(_clean(result["summary"]), indent=1, sort_keys=True) + "\n")
    (HERE / "tables.md").write_text(tables(result))
    (REPORT / "claims.json").write_text(json.dumps(result["claims"], indent=2, ensure_ascii=False) + "\n")
    print((HERE / "tables.md").read_text())
    pooled = result["summary"]["share_env_pooled"]
    print("pooled", {k: v for k, v in pooled.items() if k != "annual_return_max_run"})
    print("detectability", result["summary"]["detectability"])
    print("1/N", result["summary"]["benchmark"]["one_over_n"], result["summary"]["share_env_one_over_n"])


if __name__ == "__main__":
    main()
