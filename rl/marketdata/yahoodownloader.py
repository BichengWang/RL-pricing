"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            date, open, high, low, close, volume, tic and day columns.
            The close column contains the adjusted closing price. Empty
            downloads are skipped; if none succeed, the same columns are
            returned in an empty frame.
        """
        # Download and save the data in a pandas DataFrame:
        columns = ["date", "open", "high", "low", "close", "volume", "tic", "day"]
        price_columns = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Adj Close": "close",
            "Volume": "volume",
        }
        frames = []
        for tic in self.ticker_list:
            # Keep raw OHLC and adjusted close separate, as in the original
            # downloader, regardless of yfinance's auto_adjust default.
            temp_df = yf.download(
                tic, start=self.start_date, end=self.end_date, auto_adjust=False
            )
            if temp_df is None or temp_df.empty:
                continue
            temp_df = temp_df.copy()
            # Recent yfinance releases return (price, ticker) columns even
            # for one ticker. Older releases return a flat column index.
            if isinstance(temp_df.columns, pd.MultiIndex):
                if len(temp_df.columns.get_level_values(-1).unique()) != 1:
                    raise ValueError("Expected one Yahoo Finance ticker for {}".format(tic))
                # Yahoo normalizes ticker case; retain the requested tic in
                # the output without using its spelling as a column key.
                temp_df.columns = temp_df.columns.droplevel(-1)
            missing = set(price_columns).difference(temp_df.columns)
            if missing:
                raise ValueError(
                    "Missing Yahoo Finance columns for {}: {}".format(
                        tic, ", ".join(sorted(missing))
                    )
                )
            temp_df = temp_df[list(price_columns)].rename(columns=price_columns)
            temp_df.columns.name = None
            temp_df = temp_df.rename_axis("date").reset_index()
            temp_df["tic"] = tic
            frames.append(temp_df)
        if not frames:
            return pd.DataFrame(columns=columns)
        data_df = pd.concat(frames, ignore_index=True)
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df["date"].dt.strftime("%Y-%m-%d")
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        logger.info("Downloaded %d rows for %d tickers", len(data_df), data_df["tic"].nunique())

        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

        return data_df[columns]
