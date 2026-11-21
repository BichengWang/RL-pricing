import re

import numpy as np
import pandas as pd
import pytest
from stockstats import StockDataFrame as Sdf

from conftest import TECH, TICKERS
from rl.preprocessing.data import data_split, load_dataset
from rl.preprocessing.preprocessors import (
    FeatureEngineer,
    add_covariance_matrix,
    balance_panel,
    calculate_turbulence,
    require_complete_prices,
)


def test_preprocess_adds_features_without_missing_values(processed, raw_prices):
    for column in TECH + ["turbulence"]:
        assert column in processed.columns
    assert not processed[TECH + ["turbulence"]].isna().any().any()
    assert len(processed) == len(raw_prices)
    assert processed.equals(processed.sort_values(["date", "tic"]).reset_index(drop=True))


def test_indicators_are_computed_per_ticker(processed, raw_prices):
    for tic in TICKERS:
        own = raw_prices[raw_prices.tic == tic].sort_values("date")
        expected = Sdf.retype(own[["date", "open", "high", "low", "close", "volume"]].copy())
        got = processed[processed.tic == tic].sort_values("date")
        for indicator in TECH:
            want = np.asarray(expected[indicator], dtype=float)
            # Warm-up NaNs are zero-filled; compare the rest exactly.
            mask = ~np.isnan(want)
            np.testing.assert_allclose(got[indicator].to_numpy()[mask], want[mask])


def test_features_do_not_use_future_data(raw_prices):
    """Changing prices after a cut-off must not change any earlier feature."""
    fe = FeatureEngineer(tech_indicator_list=TECH, use_turbulence=True)
    cutoff = "2017-09-01"
    full = fe.preprocess_data(raw_prices)
    shocked = raw_prices.copy()
    later = shocked.date >= cutoff
    for column in ["open", "high", "low", "close"]:
        shocked.loc[later, column] *= 3.0
    changed = fe.preprocess_data(shocked)
    before = full.date < cutoff
    pd.testing.assert_frame_equal(full[before], changed[before])
    assert not np.allclose(full.loc[~before, "close"], changed.loc[~before, "close"])


def test_balance_panel_drops_late_tickers_and_gap_dates(raw_prices):
    df = raw_prices.copy()
    dates = sorted(df.date.unique())
    late = df[(df.tic == "AAA") & (df.date >= dates[400])].assign(tic="NEW")
    gap = (df.tic == "BBB") & (df.date == dates[10])
    df = pd.concat([df[~gap], late], ignore_index=True)
    balanced = balance_panel(df)
    assert sorted(balanced.tic.unique()) == TICKERS
    assert dates[10] not in set(balanced.date)
    counts = balanced.groupby("date").tic.nunique()
    assert (counts == len(TICKERS)).all()
    assert balanced.date.nunique() == len(dates) - 1


def test_turbulence_starts_after_lookback_and_is_non_negative(raw_prices):
    turbulence = calculate_turbulence(raw_prices, lookback=100)
    assert len(turbulence) == raw_prices.date.nunique()
    assert (turbulence.turbulence.iloc[:100] == 0).all()
    assert (turbulence.turbulence >= 0).all()
    assert (turbulence.turbulence.iloc[103:] > 0).all()


def test_turbulence_matches_mahalanobis_distance(raw_prices):
    lookback = 60
    turbulence = calculate_turbulence(raw_prices, lookback=lookback, warmup=0)
    returns = raw_prices.pivot(index="date", columns="tic", values="close").pct_change()
    i = 200
    hist = returns.iloc[i - lookback : i]
    diff = returns.iloc[i] - hist.mean()
    expected = diff @ np.linalg.inv(hist.cov()) @ diff
    assert turbulence.turbulence.iloc[i] == pytest.approx(expected)


def test_daily_return_feature_is_per_ticker(raw_prices):
    fe = FeatureEngineer(use_technical_indicator=False, user_defined_feature=True)
    out = fe.preprocess_data(raw_prices)
    aaa = out[out.tic == "AAA"].sort_values("date")
    np.testing.assert_allclose(
        aaa.daily_return.iloc[1:], aaa.close.pct_change().iloc[1:].to_numpy()
    )
    # The first day has no previous close, so it is zero-filled rather than
    # computed against another ticker's price.
    assert (out.groupby("tic").daily_return.first() == 0).all()


def test_data_split_indexes_by_day(processed):
    part = data_split(processed, "2017-01-01", "2017-02-01")
    assert part.date.min() >= "2017-01-01" and part.date.max() < "2017-02-01"
    assert list(part.index.unique()) == list(range(part.date.nunique()))
    assert (part.groupby(level=0).tic.count() == len(TICKERS)).all()


def test_add_covariance_matrix(processed):
    out = add_covariance_matrix(processed, lookback=30)
    assert out.date.nunique() == processed.date.nunique() - 30
    cov = out.cov_list.iloc[0]
    assert cov.shape == (len(TICKERS), len(TICKERS))
    np.testing.assert_allclose(cov, cov.T)


def test_load_dataset_round_trip(tmp_path, raw_prices):
    path = tmp_path / "prices.csv"
    raw_prices.to_csv(path)  # with the index column, as download_data used to
    loaded = load_dataset(file_name=str(path))
    assert list(loaded.columns) == list(raw_prices.columns)
    assert loaded.date.iloc[0] == raw_prices.date.iloc[0]


def test_load_dataset_normalises_dates_and_rejects_ambiguous_ones(tmp_path, raw_prices):
    path = tmp_path / "prices.csv"
    raw_prices.to_csv(path, index=False)
    iso = load_dataset(file_name=str(path))
    pd.testing.assert_frame_equal(iso, pd.read_csv(path))
    # Compared as strings, 20160701 sorts before 2016-07-01 and moves the split.
    dates = pd.to_datetime(raw_prices.date)
    raw_prices.assign(date=dates.dt.strftime("%Y%m%d")).to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_dataset(file_name=str(path)), iso)
    for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
        bad = raw_prices.assign(date=dates.dt.strftime(fmt))
        bad.to_csv(path, index=False)
        with pytest.raises(ValueError, match="'date'.*" + re.escape(bad.date.iloc[0])):
            load_dataset(file_name=str(path))


@pytest.mark.parametrize("column", ["open", "close"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_frozen_panel_rejects_bad_raw_prices_before_feature_fill(column, value):
    prices = pd.DataFrame({
        "date": ["2020-01-02"], "tic": ["AAA"], "open": [10.0], "close": [11.0],
    })
    prices.loc[0, column] = value
    with pytest.raises(ValueError, match=column + ".*AAA.*2020-01-02"):
        require_complete_prices(prices, ["AAA"], ["2020-01-02"])


def test_frozen_panel_requires_raw_open_and_reports_missing_rows():
    prices = pd.DataFrame({
        "date": ["2020-01-02"], "tic": ["AAA"], "open": [10.0], "close": [11.0],
    })
    with pytest.raises(ValueError, match="requires.*open"):
        require_complete_prices(prices.drop(columns="open"), ["AAA"], ["2020-01-02"])
    with pytest.raises(ValueError, match="BBB.*2020-01-02"):
        require_complete_prices(prices, ["AAA", "BBB"], ["2020-01-02"])
    with pytest.raises(ValueError, match="Duplicate.*AAA.*2020-01-02"):
        require_complete_prices(pd.concat([prices, prices]), ["AAA"], ["2020-01-02"])


def test_feature_engineer_can_preserve_a_preselected_panel(raw_prices):
    # prepare_data owns eligibility and passes clean_data=False: the feature
    # stage must not make a second, full-period ticker eligibility decision.
    dates = sorted(raw_prices.date.unique())
    late = raw_prices[(raw_prices.tic == "AAA") & (raw_prices.date >= dates[400])].assign(tic="NEW")
    selected = pd.concat([raw_prices, late], ignore_index=True)
    processed = FeatureEngineer(tech_indicator_list=TECH, clean_data=False).preprocess_data(selected)
    assert sorted(processed.tic.unique()) == sorted(TICKERS + ["NEW"])
    assert len(processed) == len(selected)
