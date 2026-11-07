"""R1 L1 scoring: dates of the authors' account rows, seams, verdict order (offline)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "reports/r1-finrl-ensemble/evidence/l1.py"
SPEC = importlib.util.spec_from_file_location("r1_l1", SCRIPT)
l1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(l1)

U = np.arange(20150000, 20150000 + 4 * 63)  # validation quarter, then 3 trade windows: 2 files


def write_run(tmp_path, seam_error=0.0):
    run = tmp_path / "r1-l1-run-1"
    run.mkdir()
    last = 1_000_000.0
    for i in l1.windows(U):
        values = [last + seam_error * (i > 126)] + [last + 10.0 * j for j in range(1, 63)]
        pd.DataFrame(values).to_csv(run / f"account_value_trade_ensemble_{i}.csv")
        last = values[-1]
    return run


def test_rows_are_dated_by_session_and_seams_dropped(tmp_path):
    series = l1.load_run(write_run(tmp_path), U)
    assert l1.windows(U) == [126, 189]
    assert len(series) == 63 + 62
    assert series.index[0] == U[63] and series.iloc[0] == 1_000_000.0
    assert U[126] not in series.index  # d0 of the second window has no value
    assert series.index[63] == U[127]
    assert series.index[-1] == U[188]


def test_a_seam_that_does_not_repeat_the_last_value_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="seam"):
        l1.load_run(write_run(tmp_path, seam_error=5.0), U)


def test_verdict_rules_apply_in_preregistered_order():
    # Significant, the claim inside both intervals: holds.
    assert l1.verdict(0.9, 0.1, 1.5, 0.01, 0.83, [1.4], 1.6, 1.30) == "holds"
    # Interval spans 0 and the claimed gap: inconclusive, even with every run below the level.
    assert l1.verdict(0.2, -0.3, 0.9, 0.2, 0.83, [0.5], 0.8, 1.30) == "inconclusive"
    # Wrong direction and the claimed gap above the interval: does not hold.
    assert l1.verdict(-0.1, -0.6, 0.4, 0.6, 0.83) == "does_not_hold"
    # Every run below the level and the level above its interval: does not hold.
    assert l1.verdict(0.5, 0.1, 0.7, 0.01, 0.83, [0.9, 1.0], 1.1, 1.30) == "does_not_hold"
    # Positive and significant, claimed gap above the interval, one run reaches the level.
    assert l1.verdict(0.5, 0.1, 0.7, 0.01, 0.83, [0.9, 1.4], 1.1, 1.30) == "partly_holds"


def test_holm_is_monotone_and_capped():
    adjusted = l1.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": 0.2})
    assert adjusted == pytest.approx({"a": 0.05, "c": 0.12, "b": 0.12, "e": 0.4, "d": 0.5})
