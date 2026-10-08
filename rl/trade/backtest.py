import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from rl.config import config
from rl.marketdata.yahoodownloader import YahooDownloader
from rl.trade import metrics

logger = logging.getLogger(__name__)


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    ts = pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)
    return ts


def backtest_stats(account_value, value_col_name="account_value"):
    """Performance statistics (pyfolio's ``perf_stats`` set) for an account."""
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = metrics.perf_stats(dr_test)
    logger.info("Performance statistics:\n%s", perf_stats_all.to_string())
    return perf_stats_all


def compare_stats(account_values):
    """Side-by-side statistics for ``{name: account value frame}``."""
    columns = {}
    for name, frame in account_values.items():
        value_col = "account_value" if "account_value" in frame.columns else "close"
        stats = metrics.perf_stats(get_daily_return(frame, value_col_name=value_col))
        stats["Final value"] = float(frame[value_col].iloc[-1])
        columns[name] = stats
    return pd.DataFrame(columns)


def plot_account_values(account_values, save_path=None, title="Backtest"):
    """Plot cumulative return and drawdown of each ``{name: account frame}``."""
    if save_path is None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(11, 7))
    else:
        # A bare Figure renders without touching the global pyplot backend.
        from matplotlib.figure import Figure

        fig = Figure(figsize=(11, 7))
    ax_ret, ax_dd = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, frame in account_values.items():
        value_col = "account_value" if "account_value" in frame.columns else "close"
        returns = get_daily_return(frame, value_col_name=value_col).dropna()
        wealth = (1 + returns).cumprod() - 1
        ax_ret.plot(wealth.index, 100 * wealth.to_numpy(), label=name, linewidth=1.4)
        drawdown = metrics.drawdown_series(returns)
        ax_dd.plot(drawdown.index, 100 * drawdown.to_numpy(), linewidth=1.0)
    ax_ret.set_ylabel("Cumulative return (%)")
    ax_ret.set_title(title)
    ax_ret.axhline(0, color="grey", linewidth=0.6)
    ax_ret.legend(loc="upper left", frameon=False)
    ax_ret.grid(alpha=0.3)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    return fig


def backtest_plot(
    account_value,
    baseline_start=config.START_TRADE_DATE,
    baseline_end=config.END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
    save_path=None,
):
    """Plot an account against a Yahoo Finance benchmark ticker."""
    baseline_df = get_baseline(ticker=baseline_ticker, start=baseline_start, end=baseline_end)
    strategy = account_value.rename(columns={value_col_name: "account_value"})
    frames = {"Strategy": strategy}
    if not baseline_df.empty:
        frames[baseline_ticker] = baseline_df[["date", "close"]]
    return plot_account_values(frames, save_path=save_path)


def get_baseline(ticker, start, end):
    dji = YahooDownloader(start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()
    return dji


def trx_plot(df_trade, df_actions, ticker_list):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    df_trx = pd.DataFrame(np.array(df_actions["transactions"].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions["date"]
    df_trx.index.name = ""

    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:, i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: True if x > 0 else False)
        selling_signal = df_trx_temp_sign.apply(lambda x: True if x < 0 else False)

        tic_plot = df_trade[
            (df_trade["tic"] == df_trx_temp.name) & (df_trade["date"].isin(df_trx.index))
        ]["close"]
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize=(10, 8))
        plt.plot(tic_plot, color="g", lw=2.0)
        plt.plot(tic_plot, "^", markersize=10, color="m", label="buying signal", markevery=buying_signal)
        plt.plot(tic_plot, "v", markersize=10, color="k", label="selling signal", markevery=selling_signal)
        plt.title(
            f"{df_trx_temp.name} Num Transactions: "
            f"{len(buying_signal[buying_signal == True]) + len(selling_signal[selling_signal == True])}"
        )
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
        plt.xticks(rotation=45, ha="right")
        plt.show()
