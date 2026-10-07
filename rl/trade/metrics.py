"""Portfolio performance statistics.

These follow the definitions used by pyfolio/empyrical so results stay
comparable with earlier runs, without depending on those unmaintained
packages. All functions take simple (not log) periodic returns.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _clean(returns):
    returns = pd.Series(returns, dtype=float).dropna()
    return returns


def cumulative_returns(returns):
    """Total compounded return over the whole period."""
    returns = _clean(returns)
    return float(np.prod(1 + returns.to_numpy()) - 1)


def annual_return(returns, periods=TRADING_DAYS_PER_YEAR):
    """Compound annual growth rate."""
    returns = _clean(returns)
    if returns.empty:
        return float("nan")
    ending_value = 1 + cumulative_returns(returns)
    years = len(returns) / periods
    return float(ending_value ** (1 / years) - 1) if ending_value > 0 else -1.0


def annual_volatility(returns, periods=TRADING_DAYS_PER_YEAR):
    returns = _clean(returns)
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns, risk_free=0.0, periods=TRADING_DAYS_PER_YEAR):
    """Annualised Sharpe ratio; ``risk_free`` is a per-period rate."""
    excess = _clean(returns) - risk_free
    if len(excess) < 2:
        return float("nan")
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods))


def sortino_ratio(returns, required_return=0.0, periods=TRADING_DAYS_PER_YEAR):
    excess = _clean(returns) - required_return
    if len(excess) < 2:
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0) ** 2)) * np.sqrt(periods)
    if downside == 0:
        return float("nan")
    return float(excess.mean() * periods / downside)


def drawdown_series(returns):
    """Fractional drawdown from the running peak, including the starting value."""
    returns = _clean(returns)
    wealth = np.concatenate(([1.0], np.cumprod(1 + returns.to_numpy())))
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1
    return pd.Series(drawdown[1:], index=returns.index)


def max_drawdown(returns):
    drawdown = drawdown_series(returns)
    return float(drawdown.min()) if not drawdown.empty else float("nan")


def calmar_ratio(returns, periods=TRADING_DAYS_PER_YEAR):
    mdd = max_drawdown(returns)
    if not mdd < 0:
        return float("nan")
    return float(annual_return(returns, periods) / abs(mdd))


def omega_ratio(returns, threshold=0.0):
    excess = _clean(returns).to_numpy() - threshold
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("nan")
    return float(excess[excess > 0].sum() / losses)


def stability_of_timeseries(returns):
    """R-squared of a linear fit to cumulative log returns."""
    returns = _clean(returns)
    if len(returns) < 2:
        return float("nan")
    cum_log = np.cumsum(np.log1p(returns.to_numpy()))
    if np.std(cum_log) == 0:
        return float("nan")
    r = np.corrcoef(np.arange(len(cum_log)), cum_log)[0, 1]
    return float(r**2)


def tail_ratio(returns):
    returns = _clean(returns).to_numpy()
    if returns.size == 0:
        return float("nan")
    left = abs(np.percentile(returns, 5))
    if left == 0:
        return float("nan")
    return float(abs(np.percentile(returns, 95)) / left)


def value_at_risk(returns, cutoff=0.05):
    returns = _clean(returns).to_numpy()
    if returns.size == 0:
        return float("nan")
    return float(np.percentile(returns, 100 * cutoff))


def skew(returns):
    x = _clean(returns).to_numpy()
    if x.size < 2 or x.std() == 0:
        return float("nan")
    return float(np.mean((x - x.mean()) ** 3) / x.std() ** 3)


def kurtosis(returns):
    """Excess kurtosis (0 for a normal distribution)."""
    x = _clean(returns).to_numpy()
    if x.size < 2 or x.std() == 0:
        return float("nan")
    return float(np.mean((x - x.mean()) ** 4) / x.std() ** 4 - 3)


def perf_stats(returns):
    """The statistics reported by ``pyfolio.timeseries.perf_stats``."""
    return pd.Series(
        {
            "Annual return": annual_return(returns),
            "Cumulative returns": cumulative_returns(returns),
            "Annual volatility": annual_volatility(returns),
            "Sharpe ratio": sharpe_ratio(returns),
            "Calmar ratio": calmar_ratio(returns),
            "Stability": stability_of_timeseries(returns),
            "Max drawdown": max_drawdown(returns),
            "Omega ratio": omega_ratio(returns),
            "Sortino ratio": sortino_ratio(returns),
            "Skew": skew(returns),
            "Kurtosis": kurtosis(returns),
            "Tail ratio": tail_ratio(returns),
            "Daily value at risk": value_at_risk(returns),
        }
    )
