import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from rl.marketdata.synthetic import generate_synthetic_data  # noqa: E402
from rl.preprocessing.preprocessors import FeatureEngineer  # noqa: E402

TICKERS = ["AAA", "BBB", "CCC"]
TECH = ["macd", "rsi_10", "close_20_sma"]


@pytest.fixture(scope="session")
def raw_prices():
    return generate_synthetic_data(TICKERS, "2016-01-01", "2018-07-01", seed=7)


@pytest.fixture(scope="session")
def processed(raw_prices):
    fe = FeatureEngineer(tech_indicator_list=TECH, use_turbulence=True)
    return fe.preprocess_data(raw_prices)


def make_panel(closes, tickers=None, start="2020-01-01", **columns):
    """Build a tiny (day, tic) frame from a list of per-day close lists."""
    closes = [list(row) if hasattr(row, "__len__") else [row] for row in closes]
    tickers = tickers or ["T{}".format(i) for i in range(len(closes[0]))]
    dates = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    rows = []
    for d, (date, row) in enumerate(zip(dates, closes)):
        for i, (tic, close) in enumerate(zip(tickers, row)):
            record = {"date": date, "tic": tic, "close": float(close)}
            for name, values in columns.items():
                value = values[d]
                record[name] = float(value[i] if hasattr(value, "__len__") else value)
            rows.append(record)
    df = pd.DataFrame(rows)
    df.index = df.date.factorize()[0]
    return df
