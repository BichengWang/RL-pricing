import logging

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from rl.config import config

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def balance_panel(df, max_missing_frac=0.05):
    """Make every date contain every ticker.

    Trading environments need the same set of tickers on every day. Tickers
    missing more than ``max_missing_frac`` of the dates (for example stocks
    that listed part-way through the period) are dropped; then dates on which
    any remaining ticker has no row are dropped.
    """
    df = df.drop_duplicates(subset=["date", "tic"])
    n_dates = df["date"].nunique()
    if n_dates == 0:
        return df.copy()
    coverage = df.groupby("tic")["date"].nunique() / n_dates
    keep = coverage[coverage >= 1 - max_missing_frac].index
    dropped = sorted(set(coverage.index) - set(keep))
    if dropped:
        logger.warning(
            "Dropping tickers missing more than %.0f%% of dates: %s",
            100 * max_missing_frac,
            ", ".join(map(str, dropped)),
        )
    df = df[df["tic"].isin(keep)]
    per_date = df.groupby("date")["tic"].nunique()
    full_dates = per_date[per_date == len(keep)].index
    n_partial = len(per_date) - len(full_dates)
    if n_partial:
        logger.warning("Dropping %d dates on which some tickers have no data", n_partial)
    df = df[df["date"].isin(full_dates)]
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def calculate_turbulence(df, lookback=252, warmup=2):
    """Turbulence index (Kritzman & Li, 2010) for each date.

    For every date after the first ``lookback`` dates, this is the Mahalanobis
    distance of that day's stock returns from the mean of the previous
    ``lookback`` days of returns. Only past data is used. The first ``warmup``
    positive values are set to zero to avoid outliers while the estimate
    stabilises.
    """
    pivot = df.pivot_table(index="date", columns="tic", values="close").sort_index()
    returns = pivot.pct_change().to_numpy()
    turbulence = np.zeros(len(pivot))
    count = 0
    for i in range(lookback, len(pivot)):
        hist = returns[i - lookback : i]
        # Skip the leading rows where even the longest-listed ticker had no data,
        # then keep the tickers with full history and a return today.
        hist = hist[int(np.isnan(hist).sum(axis=0).min()) :]
        current = returns[i]
        usable = ~np.isnan(hist).any(axis=0) & ~np.isnan(current)
        if hist.shape[0] < 2 or not usable.any():
            continue
        hist = hist[:, usable]
        diff = current[usable] - hist.mean(axis=0)
        cov = np.atleast_2d(np.cov(hist, rowvar=False))
        value = float(diff @ np.linalg.pinv(cov) @ diff)
        if value > 0:
            count += 1
            if count > warmup:
                turbulence[i] = value
    return pd.DataFrame({"date": pivot.index, "turbulence": turbulence})


def add_covariance_matrix(df, lookback=252):
    """Add a ``cov_list`` column holding each date's return covariance matrix.

    The matrix for a date uses returns from the ``lookback`` days up to and
    including that date (its close is known when the agent acts). The first
    ``lookback`` dates, which lack a full window, are dropped. This is the
    input expected by :class:`rl.env.env_portfolio.StockPortfolioEnv`.
    """
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    pivot = df.pivot_table(index="date", columns="tic", values="close").sort_index()
    returns = pivot.pct_change()
    dates = pivot.index
    covs = []
    for i in range(lookback, len(dates)):
        window = returns.iloc[i - lookback + 1 : i + 1].dropna()
        covs.append(window.cov().to_numpy())
    df_cov = pd.DataFrame({"date": dates[lookback:], "cov_list": covs})
    df = df.merge(df_cov, on="date")
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            add technical indicators or not
        tech_indicator_list : list
            a list of stockstats indicator names (see config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature : boolean
            add user defined features or not
        clean_data : boolean
            make every date contain every ticker (see :func:`balance_panel`)

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
        clean_data=True,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = list(tech_indicator_list)
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
        self.clean_data = clean_data

    def preprocess_data(self, df):
        """Add the configured features to a long (date, tic) price frame."""
        df = df.copy()
        df["date"] = df["date"].astype(str)
        if self.clean_data:
            df = balance_panel(df)

        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            logger.info("Added technical indicators")

        if self.use_turbulence:
            df = self.add_turbulence(df)
            logger.info("Added turbulence index")

        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            logger.info("Added user defined features")

        return self.fill_missing(df)

    @staticmethod
    def fill_missing(df):
        """Forward-fill each ticker's own history, then zero-fill warm-up gaps.

        Only past values are used, so no future information leaks into
        earlier rows (and no ticker borrows another ticker's values).
        """
        df = df.sort_values(["tic", "date"])
        value_columns = [
            c for c in df.columns if c not in ("date", "tic") and pd.api.types.is_numeric_dtype(df[c])
        ]
        df[value_columns] = df.groupby("tic")[value_columns].ffill().fillna(0)
        return df.sort_values(["date", "tic"]).reset_index(drop=True)

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical indicators
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        frames = []
        for tic, group in data.sort_values(["tic", "date"]).groupby("tic", sort=False):
            group = group.copy()
            available = [c for c in OHLCV_COLUMNS if c in group.columns]
            stock = Sdf.retype(group[["date"] + available].copy())
            for indicator in self.tech_indicator_list:
                group[indicator] = np.asarray(stock[indicator], dtype=float)
            frames.append(group)
        if not frames:
            return data.assign(**{indicator: np.nan for indicator in self.tech_indicator_list})
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["date", "tic"]).reset_index(drop=True)

    def add_user_defined_feature(self, data):
        """
        add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.sort_values(["tic", "date"]).copy()
        df["daily_return"] = df.groupby("tic")["close"].pct_change()
        return df.sort_values(["date", "tic"]).reset_index(drop=True)

    def add_turbulence(self, data):
        """
        add turbulence index (see :func:`calculate_turbulence`)
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on the tickers in ``data``"""
        return calculate_turbulence(data)
