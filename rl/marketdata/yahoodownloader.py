"""Daily prices from Yahoo Finance.

Yahoo's ``Adj Close`` is recomputed on every request, and its last digits
differ between identical requests (relative differences up to about 1e-6),
so a re-download never reproduces the same bytes. It is not used. Prices are
adjusted here from fields that do repeat exactly: ``Open``, ``High``, ``Low``
and ``Close`` (adjusted for splits only) and ``Dividends`` (per share, also
split-adjusted), with the rule behind Yahoo's adjusted close (see
:func:`total_return_factors`).
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close"]
COLUMNS = ["date", "open", "high", "low", "close", "volume", "tic", "day"]
UNADJUSTED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "dividends", "tic", "day"]
_SOURCE_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Dividends": "dividends",
}


def total_return_factors(close, dividends) -> np.ndarray:
    """Cumulative dividend reinvestment factor of each row, 1.0 at the first row.

    ``close`` is adjusted for splits but not dividends, in date order. A
    dividend whose ex-date is row ``s`` is reinvested at the close before
    ``s``. ``prices * factors`` is then adjusted forward from the first row:
    extending the rows never changes earlier values. ``prices * factors /
    factors[-1]`` is adjusted back from the last row, Yahoo's convention.
    Either way, returns include the dividends. A dividend on the first row
    has no earlier close in the rows and is ignored.
    """
    close = np.asarray(close, dtype=float)
    dividends = np.nan_to_num(np.asarray(dividends, dtype=float))
    previous = pd.Series(close).ffill().shift(1).to_numpy()
    paid = (dividends > 0) & np.isfinite(previous)
    ratio = np.ones(len(close))
    with np.errstate(divide="ignore"):
        ratio[paid] = previous[paid] / (previous[paid] - dividends[paid])
    if not (np.isfinite(ratio) & (ratio > 0)).all():
        raise ValueError("A dividend is not below the close before its ex-date")
    return np.cumprod(ratio)


class YahooDownloader:
    """Daily prices of ``ticker_list`` for ``[start_date, end_date)`` from Yahoo Finance."""

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_unadjusted(self) -> pd.DataFrame:
        """Prices adjusted for splits only, with the dividend per share of each ex-date.

        Columns ``UNADJUSTED_COLUMNS``, sorted by date and ticker. Rows
        without prices or volume are dropped; a dividend on such a row moves
        to the ticker's next row that has them. Empty downloads are skipped;
        if none succeed, an empty frame with the same columns is returned.
        """
        frames = [frame for frame in map(self._download, self.ticker_list) if frame is not None]
        if not frames:
            return pd.DataFrame(columns=UNADJUSTED_COLUMNS)
        data_df = pd.concat(frames, ignore_index=True)
        data_df["day"] = data_df["date"].dt.day_of_week  # Monday = 0
        data_df["date"] = data_df["date"].dt.strftime("%Y-%m-%d")
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        logger.info("Downloaded %d rows for %d tickers", len(data_df), data_df["tic"].nunique())
        return data_df[UNADJUSTED_COLUMNS]

    def fetch_data(self) -> pd.DataFrame:
        """Prices adjusted for splits and dividends back from the last row, as Yahoo adjusts.

        Columns ``COLUMNS``, sorted by date and ticker. Open, high, low and
        close are all scaled by the same factor, so open-to-close returns do
        not mix raw and adjusted prices. Identical requests return identical
        values, unlike Yahoo's own ``Adj Close``.
        """
        data_df = self.fetch_unadjusted()
        for _, rows in data_df.groupby("tic", sort=False):
            factors = total_return_factors(rows["close"], rows["dividends"])
            data_df.loc[rows.index, PRICE_COLUMNS] = rows[PRICE_COLUMNS].mul(factors / factors[-1], axis=0)
        return data_df[COLUMNS]

    def _download(self, tic) -> pd.DataFrame | None:
        """One ticker's rows with columns date, prices, volume, dividends and tic."""
        temp_df = yf.download(
            tic, start=self.start_date, end=self.end_date, auto_adjust=False, actions=True
        )
        if temp_df is None or temp_df.empty:
            return None
        temp_df = temp_df.copy()
        # Recent yfinance releases return (price, ticker) columns even
        # for one ticker. Older releases return a flat column index.
        if isinstance(temp_df.columns, pd.MultiIndex):
            if len(temp_df.columns.get_level_values(-1).unique()) != 1:
                raise ValueError("Expected one Yahoo Finance ticker for {}".format(tic))
            # Yahoo normalizes ticker case; retain the requested tic in
            # the output without using its spelling as a column key.
            temp_df.columns = temp_df.columns.droplevel(-1)
        missing = set(_SOURCE_COLUMNS).difference(temp_df.columns)
        if missing:
            raise ValueError(
                "Missing Yahoo Finance columns for {}: {}".format(tic, ", ".join(sorted(missing)))
            )
        temp_df = temp_df[list(_SOURCE_COLUMNS)].rename(columns=_SOURCE_COLUMNS)
        temp_df.columns.name = None
        temp_df = temp_df.rename_axis("date").reset_index().sort_values("date", ignore_index=True)
        temp_df["dividends"] = temp_df["dividends"].fillna(0.0)
        valid = temp_df[PRICE_COLUMNS + ["volume"]].notna().all(axis=1)
        # Each row and the invalid rows just before it share a group, so the
        # group's dividends land on its one valid row.
        group = valid[::-1].cumsum()[::-1]
        temp_df["dividends"] = temp_df.groupby(group)["dividends"].transform("sum")
        temp_df = temp_df[valid]
        if temp_df.empty:
            return None
        temp_df["tic"] = tic
        return temp_df
