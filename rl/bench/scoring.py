"""Benchmark scoring: performance metrics and Sharpe-ratio inference.

Pure numpy functions on simple per-period return arrays (``equity[1:] /
equity[:-1] - 1``).  Definitions shared with :mod:`rl.trade.metrics` are
imported from there (Sharpe = mean / std(ddof=1) * sqrt(periods), risk-free
rate 0; max drawdown includes the starting value).

Sharpe-ratio inference:

* :func:`probabilistic_sharpe` - Bailey & Lopez de Prado (2012), PSR.
* :func:`deflated_sharpe` - Bailey & Lopez de Prado (2014), DSR: PSR against
  the Sharpe ratio expected from the best of ``trials`` skill-less strategies.
* :func:`sharpe_difference` - one-sided paired stationary-bootstrap test
  (Politis & Romano 1994) of "policy Sharpe > benchmark Sharpe".

:func:`reprice` gives a recorded backtest's returns at another fee, with its
decisions held fixed.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from rl.trade import metrics as _m

__all__ = [
    "annualised_sharpe",
    "deflated_sharpe",
    "metrics",
    "probabilistic_sharpe",
    "reprice",
    "sharpe_difference",
]

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329
# Bootstrap resamples are generated in chunks so memory stays bounded.
_CHUNK_ELEMENTS = 4_000_000


def _series(x, name: str, *, minimum: int = 2, floor: float | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size < minimum:
        raise ValueError(f"{name} needs at least {minimum} observations, got {a.size}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if floor is not None and (a < floor).any():
        raise ValueError(f"{name} contains values below {floor}")
    return a


def _returns(x, name: str = "returns") -> np.ndarray:
    return _series(x, name, floor=-1.0)


def _periods(periods_per_year) -> float:
    ppy = float(periods_per_year)
    if not (math.isfinite(ppy) and ppy > 0):
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year!r}")
    return ppy


def _paired(returns, benchmark) -> tuple[np.ndarray, np.ndarray]:
    r = _returns(returns)
    b = _returns(benchmark, "benchmark")
    if r.shape != b.shape:
        raise ValueError(f"returns ({r.size}) and benchmark ({b.size}) must have the same length")
    return r, b


def _sharpe(r: np.ndarray) -> float:
    """Per-period Sharpe ratio (mean / std with ddof=1); NaN when undefined."""
    std = r.std(ddof=1)
    return float(r.mean() / std) if std > 0 else float("nan")


def annualised_sharpe(returns, periods_per_year: float) -> float:
    """Annualised Sharpe ratio (risk-free 0); NaN for constant returns."""
    r = _returns(returns)
    return _m.sharpe_ratio(r, 0.0, _periods(periods_per_year))


def _hac_alpha_t(r: np.ndarray, b: np.ndarray, coef: np.ndarray) -> float:
    """Newey-West (Bartlett kernel) t-statistic of the OLS intercept."""
    n = r.size
    x = np.column_stack((np.ones(n), b))
    resid = r - x @ coef
    lag = min(int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), n - 1)
    scores = x * resid[:, None]
    s = scores.T @ scores
    for k in range(1, lag + 1):
        gamma = scores[k:].T @ scores[:-k]
        s += (1.0 - k / (lag + 1.0)) * (gamma + gamma.T)
    bread = np.linalg.inv(x.T @ x)
    var_alpha = (bread @ s @ bread)[0, 0]
    if not var_alpha > 0:
        return float("nan")
    return float(coef[0] / math.sqrt(var_alpha))


def metrics(
    returns,
    periods_per_year: float,
    *,
    benchmark=None,
    turnover=None,
) -> dict[str, float]:
    """Standard performance statistics of a per-period return series.

    Always: ``total_return, cagr, volatility, sharpe, sortino, max_drawdown,
    calmar`` (definitions from :mod:`rl.trade.metrics`; undefined ratios are
    NaN).  With ``benchmark`` (same length): ``excess_cagr`` (cagr minus the
    benchmark's), ``beta`` and ``alpha`` (OLS of returns on benchmark returns;
    alpha is the per-period intercept times ``periods_per_year``), ``alpha_t``
    (Newey-West HAC t-statistic of the intercept, lag
    ``floor(4 (n/100)^(2/9))``), ``tracking_error`` (annualised std of active
    returns) and ``information_ratio`` (annualised mean active return over
    tracking error).  With ``turnover`` (per-session one-way traded fraction of
    equity, same length): ``turnover`` = mean * ``periods_per_year``.
    """
    r = _returns(returns)
    ppy = _periods(periods_per_year)
    out = {
        "total_return": _m.cumulative_returns(r),
        "cagr": _m.annual_return(r, ppy),
        "volatility": _m.annual_volatility(r, ppy),
        "sharpe": _m.sharpe_ratio(r, 0.0, ppy),
        "sortino": _m.sortino_ratio(r, 0.0, ppy),
        "max_drawdown": _m.max_drawdown(r),
        "calmar": _m.calmar_ratio(r, ppy),
    }
    if benchmark is not None:
        r, b = _paired(r, benchmark)
        out["excess_cagr"] = out["cagr"] - _m.annual_return(b, ppy)
        var_b = b.var()
        if var_b > 0:
            beta = float(np.mean((r - r.mean()) * (b - b.mean())) / var_b)
            intercept = float(r.mean() - beta * b.mean())
            alpha_t = _hac_alpha_t(r, b, np.array([intercept, beta]))
        else:
            beta, intercept, alpha_t = float("nan"), float("nan"), float("nan")
        active = r - b
        te = float(active.std(ddof=1) * math.sqrt(ppy))
        out["beta"] = beta
        out["alpha"] = intercept * ppy
        out["alpha_t"] = alpha_t
        out["tracking_error"] = te
        out["information_ratio"] = float(active.mean() * ppy / te) if te > 0 else float("nan")
    if turnover is not None:
        t = _series(turnover, "turnover", minimum=1, floor=0.0)
        if t.size != r.size:
            raise ValueError(f"turnover ({t.size}) and returns ({r.size}) must have the same length")
        out["turnover"] = float(t.mean() * ppy)
    return out


def probabilistic_sharpe(returns, benchmark_sharpe: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012).

    Probability that the true per-period Sharpe ratio exceeds
    ``benchmark_sharpe`` (also per-period), given the sample length, skewness
    and kurtosis of ``returns``.  Returns NaN for constant returns.
    """
    r = _returns(returns)
    sr_star = float(benchmark_sharpe)
    if not math.isfinite(sr_star):
        raise ValueError("benchmark_sharpe must be finite")
    sr = _sharpe(r)
    if math.isnan(sr):
        return float("nan")
    z = (r - r.mean()) / r.std()
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # non-excess: 3 for a normal distribution
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if not denom > 0:
        return float("nan")
    stat = (sr - sr_star) * math.sqrt(r.size - 1) / math.sqrt(denom)
    return _NORMAL.cdf(stat)


def deflated_sharpe(returns, trials: int, sharpe_variance: float) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014).

    PSR of ``returns`` against the Sharpe ratio expected from the best of
    ``trials`` independent skill-less strategies whose per-period Sharpe ratios
    have variance ``sharpe_variance``::

        SR0 = sqrt(V) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))

    with ``g`` the Euler-Mascheroni constant.  ``trials == 1`` gives PSR
    against 0.
    """
    n_trials = int(trials)
    if n_trials != trials or n_trials < 1:
        raise ValueError(f"trials must be a positive integer, got {trials!r}")
    variance = float(sharpe_variance)
    if not (math.isfinite(variance) and variance >= 0):
        raise ValueError(f"sharpe_variance must be finite and non-negative, got {sharpe_variance!r}")
    if n_trials == 1:
        return probabilistic_sharpe(returns, 0.0)
    expected_max = math.sqrt(variance) * (
        (1.0 - _EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    return probabilistic_sharpe(returns, expected_max)


def _stationary_indices(rng: np.random.Generator, rows: int, n: int, block: float) -> np.ndarray:
    """``(rows, n)`` stationary-bootstrap index paths (mean block length ``block``).

    Each path starts at a uniform index; every next step restarts at a fresh
    uniform index with probability ``1/block`` and otherwise advances by one
    (wrapping circularly).
    """
    starts = rng.integers(0, n, size=(rows, n))
    restart = rng.random((rows, n)) < 1.0 / block
    restart[:, 0] = True
    pos = np.arange(n)
    last = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    anchor = np.take_along_axis(starts, last, axis=1)
    return (anchor + pos - last) % n


def sharpe_difference(
    returns,
    benchmark,
    periods_per_year: float,
    *,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Annualised Sharpe(returns) - Sharpe(benchmark) with a bootstrap p-value.

    The one-sided p-value tests H0: policy Sharpe <= benchmark Sharpe against
    H1: policy > benchmark.  Return pairs are resampled jointly with the
    stationary bootstrap (Politis & Romano 1994; geometric block lengths with
    mean ``block``, default ``round(n ** (1/3))``), preserving cross-correlation
    and serial dependence.  The bootstrap distribution of the difference is
    centred at the observed difference ``d`` (imposing H0), so

        p = (1 + #{b : d*_b - d >= d}) / (n_boot + 1)

    i.e. the share of resamples whose centred difference is at least as large
    as the observed one (the +1 keeps p strictly positive).  Deterministic for
    a fixed ``seed``.  Returns ``(nan, nan)`` if either Sharpe is undefined
    (constant returns).
    """
    r, b = _paired(returns, benchmark)
    ppy = _periods(periods_per_year)
    n = r.size
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    mean_block = round(n ** (1.0 / 3.0)) if block is None else block
    if not (isinstance(mean_block, (int, np.integer)) and mean_block >= 1):
        raise ValueError(f"block must be an integer >= 1, got {block!r}")
    scale = math.sqrt(ppy)
    sr_r, sr_b = _sharpe(r), _sharpe(b)
    if math.isnan(sr_r) or math.isnan(sr_b):
        return float("nan"), float("nan")
    observed = (sr_r - sr_b) * scale

    rng = np.random.default_rng(seed)
    n_boot = int(n_boot)
    chunk = max(1, _CHUNK_ELEMENTS // n)
    diffs = np.empty(n_boot)
    for lo in range(0, n_boot, chunk):
        rows = min(chunk, n_boot - lo)
        idx = _stationary_indices(rng, rows, n, mean_block)
        d = np.zeros(rows)
        for series, sign in ((r, 1.0), (b, -1.0)):
            sample = series[idx]
            std = sample.std(axis=1, ddof=1)
            ratio = np.divide(sample.mean(axis=1), std, out=np.zeros(rows), where=std > 0)
            d += sign * ratio
        diffs[lo : lo + rows] = d * scale
    exceed = int(np.count_nonzero(diffs - observed >= observed))
    return float(observed), (1 + exceed) / (n_boot + 1)


def reprice(equity, turnover, charged: float, fee: float) -> np.ndarray:
    """Per-period returns of a recorded backtest re-priced from fee ``charged`` to ``fee``.

    ``equity`` (``n + 1`` values, net of fees at ``charged``) and ``turnover``
    (``n`` sessions of traded notional over pre-trade equity) are as
    :func:`rl.bench.engine.simulate` records them. A fill turns pre-trade
    equity ``B`` into post-fee equity ``B (1 - charged * turnover)``, and the
    weights after it are the targets whatever the fee, so with the decisions
    held fixed::

        1 + r' = (1 + r) (1 - fee * turnover) / (1 - charged * turnover)

    This is exact at ``fee`` 0. At other fees it keeps the turnover of the
    charged run, which shifts slightly with the fee. To first order,
    ``r' = r + (charged - fee) * turnover``.
    """
    e = _series(equity, "equity", floor=0.0)
    t = _series(turnover, "turnover", minimum=1, floor=0.0)
    if t.size != e.size - 1:
        raise ValueError(f"turnover ({t.size}) must have one value per return ({e.size - 1})")
    for name, value in (("charged", charged), ("fee", fee)):
        if not (math.isfinite(value) and 0 <= value < 1):
            raise ValueError(f"{name} must be in [0, 1), got {value!r}")
    return e[1:] / e[:-1] * ((1.0 - fee * t) / (1.0 - charged * t)) - 1.0
