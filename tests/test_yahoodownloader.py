"""Offline regression tests for Yahoo Finance's historical data layouts."""

from unittest.mock import call, patch

import pandas as pd
import pytest

from rl.marketdata.yahoodownloader import YahooDownloader


OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume", "tic", "day"]


def prices():
    # Distinct raw and adjusted prices expose accidental column remapping.
    return pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [105.0, 115.0],
            "Low": [95.0, 105.0],
            "Close": [102.0, 112.0],
            "Adj Close": [51.0, 56.0],
            "Volume": [1000, 2000],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-02"]),
    ).rename_axis("Date")


@pytest.mark.parametrize("multi_index", [False, True])
def test_download_preserves_prices_and_normalizes_columns(multi_index):
    source = prices()
    # Modern responses use alphabetic price order rather than legacy OHLC.
    if multi_index:
        source = source[sorted(source.columns)]
        source.columns = pd.MultiIndex.from_product(
            [source.columns, ["AAA"]], names=["Price", "Ticker"]
        )
    original = source.copy(deep=True)
    downloader = YahooDownloader("2024-01-01", "2024-01-04", ["AAA"])

    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=source) as download:
        result = downloader.fetch_data()

    download.assert_called_once_with(
        "AAA", start="2024-01-01", end="2024-01-04", auto_adjust=False
    )
    expected = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [110.0, 100.0],
            "high": [115.0, 105.0],
            "low": [105.0, 95.0],
            "close": [56.0, 51.0],
            "volume": [2000, 1000],
            "tic": ["AAA", "AAA"],
            "day": [1, 2],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    pd.testing.assert_frame_equal(source, original)


def test_multiple_tickers_skip_empty_download_and_sort_rows():
    aaa = prices()
    bbb = prices() * 2
    bbb.loc[pd.Timestamp("2024-01-03"), "Adj Close"] = float("nan")
    downloader = YahooDownloader("2024-01-01", "2024-01-04", ["BBB", "EMPTY", "AAA"])

    with patch(
        "rl.marketdata.yahoodownloader.yf.download",
        side_effect=[bbb, pd.DataFrame(), aaa],
    ) as download:
        result = downloader.fetch_data()

    assert download.call_args_list == [
        call(tic, start="2024-01-01", end="2024-01-04", auto_adjust=False)
        for tic in ["BBB", "EMPTY", "AAA"]
    ]
    assert list(result.columns) == OUTPUT_COLUMNS
    assert result[["date", "tic"]].values.tolist() == [
        ["2024-01-02", "AAA"],
        ["2024-01-02", "BBB"],
        ["2024-01-03", "AAA"],
    ]
    assert result["close"].tolist() == [56.0, 112.0, 51.0]
    assert result.index.tolist() == [0, 1, 2]


@pytest.mark.parametrize("tickers,response", [([], None), (["AAA"], None), (["AAA"], pd.DataFrame())])
def test_no_data_returns_stable_empty_schema(tickers, response):
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=response) as download:
        result = YahooDownloader("2024-01-01", "2024-01-04", tickers).fetch_data()
    assert result.empty
    assert list(result.columns) == OUTPUT_COLUMNS
    assert download.call_count == len(tickers)


def test_missing_adjusted_close_reports_ticker_and_column():
    with patch(
        "rl.marketdata.yahoodownloader.yf.download",
        return_value=prices().drop(columns="Adj Close"),
    ):
        with pytest.raises(ValueError, match="Missing Yahoo Finance columns for AAA: Adj Close"):
            YahooDownloader("2024-01-01", "2024-01-04", ["AAA"]).fetch_data()


def test_multi_index_accepts_normalized_ticker_case():
    source = prices()
    source.columns = pd.MultiIndex.from_product([source.columns, ["AAA"]])
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=source):
        result = YahooDownloader("2024-01-01", "2024-01-04", ["aaa"]).fetch_data()
    assert result["tic"].tolist() == ["aaa", "aaa"]
    assert result["close"].tolist() == [56.0, 51.0]


def test_unexpected_multiple_tickers_are_rejected():
    source = pd.concat([prices(), prices()], axis=1, keys=["AAA", "BBB"])
    source = source.swaplevel(axis=1)
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=source):
        with pytest.raises(ValueError, match="Expected one Yahoo Finance ticker for AAA"):
            YahooDownloader("2024-01-01", "2024-01-04", ["AAA"]).fetch_data()
