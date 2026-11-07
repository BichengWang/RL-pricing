"""R1 L2/L3 scoring helpers and the run plan (offline, synthetic arrays)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVIDENCE = Path(__file__).resolve().parents[1] / "reports/r1-finrl-ensemble/evidence"
sys.path.insert(0, str(EVIDENCE))
SPEC = importlib.util.spec_from_file_location("r1_l2", EVIDENCE / "l2.py")
l2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(l2)
import l2_runs  # noqa: E402


def test_plan_runs_core_seeds_first_and_the_prefix_check_before_them():
    names = [j["name"] for j in l2_runs.plan(prefix_check=True)]
    assert names[0] == "r1_A_s0_to20200508"
    assert names[1:11] == [f"r1_{a}_s{s}" for s in range(5) for a in "AB"]
    assert names[-5:] == [f"r1_C_s{s}" for s in range(5)]
    assert len(names) == 1 + 25
    assert "GE" in l2_runs.STATED and "WBA" in l2_runs.SHIPPED and "GE" not in l2_runs.SHIPPED


def test_crash_dates_are_the_largest_peak_to_trough_fall():
    close = pd.Series([10, 12, 9, 11, 13, 8, 9, 14], index=range(20200101, 20200109))
    assert l2.crash_dates(close) == (20200105, 20200106)


def test_flat_sessions_counts_unchanged_equity_and_distinct_sets():
    values = np.array([[1.0, 1.0, 1.1, 1.1], [1.0, 1.0, 1.2, 1.2], [1.0, 1.1, 1.2, 1.3]])
    out = l2.flat_sessions(values, [1, 2, 3, 4])
    assert out == {"per_seed": [2, 2, 0], "distinct_sets": 2}


def test_score_level_picks_the_best_agent_and_adjusts_seven_gaps():
    rng = np.random.default_rng(0)
    n = 300

    def paths(drift, seeds=3):
        return np.cumprod(1 + rng.normal(drift, 0.01, (seeds, n)), axis=1)

    agents = {"A2C": paths(0.0), "PPO": paths(0.002), "DDPG": paths(-0.001)}
    base = {"dia": paths(0.0, 1)[0], "minvar": paths(0.0, 1)[0]}
    claims, ctx = l2.score_level("L2", paths(0.001), agents, base, (100, 130), n - 1, n_boot=200)
    assert ctx["best_agent"] == "PPO"
    assert sorted(claims) == sorted(f"{c}-l2" for c in l2.CLAIMS)
    assert all(c["verdict"] in ("holds", "partly", "does-not-hold", "inconclusive") for c in claims.values())
    assert all("p_holm" in c["gap"] for c in claims.values())
    assert "level" in claims["ens-sharpe-l2"] and "level" not in claims["ens-sharpe-vs-agents-l2"]
    l3, _ = l2.score_level("L3", paths(0.001), agents, base, (100, 130), n - 1, n_boot=200)
    assert all("level" not in c for c in l3.values())
