import math
from statistics import NormalDist

import numpy as np
import pytest

from rl.bench import scoring
from rl.bench.scoring import (
    annualised_sharpe,
    deflated_sharpe,
    metrics,
    probabilistic_sharpe,
    sharpe_difference,
)


def _noise(n, mean, sd, seed):
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_metrics_known_values():
    r = np.array([0.1, -0.5, 0.2])
    m = metrics(r, 3)  # one "year" of three periods
    assert m["total_return"] == pytest.approx(0.66 - 1)
    assert m["cagr"] == pytest.approx(0.66 - 1)
    assert m["max_drawdown"] == pytest.approx(-0.5)  # 1.1 -> 0.55
    assert m["calmar"] == pytest.approx(m["cagr"] / 0.5)
    assert m["volatility"] == pytest.approx(r.std(ddof=1) * math.sqrt(3))
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(3))
    downside = math.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert m["sortino"] == pytest.approx(r.mean() / downside * math.sqrt(3))
    assert set(m) == {"total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar"}


def test_max_drawdown_counts_losses_from_the_starting_value():
    assert metrics(np.array([-0.2, 0.1]), 252)["max_drawdown"] == pytest.approx(-0.2)


def test_cagr_annualises_by_elapsed_years():
    r = np.full(504, 0.0004) + np.tile([0.001, -0.001], 252)
    m = metrics(r, 252)
    assert m["cagr"] == pytest.approx((1 + m["total_return"]) ** 0.5 - 1)


def test_total_loss_gives_minus_one_cagr():
    m = metrics(np.array([0.1, -1.0, 0.05]), 252)
    assert m["cagr"] == -1.0
    assert m["total_return"] == pytest.approx(-1.0)


def test_turnover_is_annualised_mean():
    r = _noise(10, 0.0, 0.01, 0)
    t = np.linspace(0.0, 0.18, 10)
    assert metrics(r, 252, turnover=t)["turnover"] == pytest.approx(0.09 * 252)


def test_beta_and_alpha_for_levered_benchmark():
    b = _noise(500, 0.0004, 0.01, 1)
    m = metrics(2 * b, 252, benchmark=b)
    assert m["beta"] == pytest.approx(2.0)
    assert m["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert m["tracking_error"] == pytest.approx(b.std(ddof=1) * math.sqrt(252))
    assert m["information_ratio"] == pytest.approx(b.mean() * 252 / (b.std(ddof=1) * math.sqrt(252)))


def test_alpha_recovers_intercept_and_hac_t_matches_ols_for_iid_noise():
    n = 4000
    b = _noise(n, 0.0004, 0.01, 2)
    r = 0.0005 + 1.5 * b + _noise(n, 0.0, 0.004, 3)
    m = metrics(r, 252, benchmark=b)
    assert m["beta"] == pytest.approx(1.5, abs=0.05)
    assert m["alpha"] == pytest.approx(0.0005 * 252, abs=0.04)
    x = np.column_stack((np.ones(n), b))
    coef = np.linalg.lstsq(x, r, rcond=None)[0]
    resid = r - x @ coef
    ols_var = resid @ resid / (n - 2) * np.linalg.inv(x.T @ x)[0, 0]
    assert m["alpha_t"] == pytest.approx(coef[0] / math.sqrt(ols_var), rel=0.15)
    assert m["excess_cagr"] == pytest.approx(m["cagr"] - metrics(b, 252)["cagr"])


def test_hac_t_stat_is_deflated_by_autocorrelated_residuals():
    n = 3000
    rng = np.random.default_rng(4)
    e = np.zeros(n)
    shocks = rng.normal(0, 0.004, n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + shocks[i]
    b = _noise(n, 0.0, 0.01, 5)
    r = 0.0002 + b + e
    x = np.column_stack((np.ones(n), b))
    coef = np.linalg.lstsq(x, r, rcond=None)[0]
    resid = r - x @ coef
    ols_t = coef[0] / math.sqrt(resid @ resid / (n - 2) * np.linalg.inv(x.T @ x)[0, 0])
    assert abs(metrics(r, 252, benchmark=b)["alpha_t"]) < 0.7 * abs(ols_t)


def test_psr_is_half_when_sharpe_equals_benchmark():
    r = _noise(300, 0.001, 0.01, 6)
    sr = r.mean() / r.std(ddof=1)
    assert probabilistic_sharpe(r, sr) == pytest.approx(0.5)


def test_psr_matches_closed_form_for_symmetric_two_point_returns():
    # Alternating +/-d around mu: skew 0 and kurtosis 1, so the variance
    # correction is exactly 1 and PSR = Phi(SR * sqrt(n - 1)).
    n, mu, d = 100, 0.002, 0.01
    r = mu + d * np.tile([1.0, -1.0], n // 2)
    sr = mu / (d * math.sqrt(n / (n - 1)))
    assert probabilistic_sharpe(r) == pytest.approx(NormalDist().cdf(sr * math.sqrt(n - 1)))
    assert probabilistic_sharpe(r, sr / 2) == pytest.approx(NormalDist().cdf(sr / 2 * math.sqrt(n - 1)))


def test_psr_grows_with_evidence_and_penalises_negative_skew():
    assert probabilistic_sharpe(_noise(1000, 0.001, 0.01, 7)) > probabilistic_sharpe(_noise(100, 0.001, 0.01, 7))
    assert probabilistic_sharpe(_noise(300, -0.001, 0.01, 8)) < 0.5
    # Same mean and variance, but the second has a crash tail: lower confidence.
    base = np.tile([0.01, 0.0, 0.02, 0.01], 50)
    skewed = base.copy()
    skewed[::50] = -0.1
    skewed += base.mean() - skewed.mean()
    skewed = (skewed - skewed.mean()) / skewed.std() * base.std() + base.mean()
    assert probabilistic_sharpe(skewed) < probabilistic_sharpe(base)


def test_dsr_decreases_with_trials():
    r = _noise(750, 0.001, 0.01, 9)
    values = [deflated_sharpe(r, k, 1e-4) for k in (1, 2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_dsr_reduces_to_psr_without_selection():
    r = _noise(500, 0.001, 0.01, 10)
    assert deflated_sharpe(r, 1, 0.5) == probabilistic_sharpe(r, 0.0)
    assert deflated_sharpe(r, 50, 0.0) == pytest.approx(probabilistic_sharpe(r, 0.0))


def test_dsr_uses_expected_maximum_sharpe_of_null_trials():
    # N=10, V=1e-4: SR0 = 0.01 * (0.4228 * 1.2816 + 0.5772 * 1.7884) = 0.015745
    r = _noise(500, 0.001, 0.01, 11)
    assert deflated_sharpe(r, 10, 1e-4) == pytest.approx(probabilistic_sharpe(r, 0.015745), abs=2e-3)


def test_sharpe_difference_of_identical_series_is_zero_and_insignificant():
    r = _noise(500, 0.0004, 0.01, 12)
    diff, p = sharpe_difference(r, r.copy(), 252)
    assert diff == 0.0
    assert p >= 0.4


def test_sharpe_difference_detects_true_edge():
    b = _noise(1500, 0.0003, 0.01, 13)
    r = _noise(1500, 0.0020, 0.01, 14)
    diff, p = sharpe_difference(r, b, 252)
    assert diff == pytest.approx(annualised_sharpe(r, 252) - annualised_sharpe(b, 252))
    assert diff > 1.5
    assert p < 0.02


def test_sharpe_difference_not_significant_for_equal_skill_and_reversed_for_underperformer():
    b = _noise(1000, 0.0004, 0.01, 15)
    r = _noise(1000, 0.0004, 0.01, 16)
    _, p = sharpe_difference(r, b, 252)
    assert p > 0.05
    diff, p_rev = sharpe_difference(0.5 * b, b, 252)  # half the exposure, same Sharpe
    assert diff == pytest.approx(0.0, abs=1e-9)
    worse = _noise(1000, -0.0010, 0.01, 17)
    diff, p_worse = sharpe_difference(worse, b, 252)
    assert diff < 0 and p_worse > 0.9


def test_sharpe_difference_is_roughly_calibrated_under_the_null():
    rng = np.random.default_rng(18)
    rejections = 0
    trials = 150
    for k in range(trials):
        r = rng.normal(0.0004, 0.01, 250)
        b = rng.normal(0.0004, 0.01, 250)
        if sharpe_difference(r, b, 252, n_boot=200, seed=k)[1] < 0.1:
            rejections += 1
    assert 0.03 < rejections / trials < 0.2


def test_sharpe_difference_is_deterministic_per_seed():
    b = _noise(400, 0.0003, 0.01, 19)
    r = _noise(400, 0.0006, 0.01, 20)
    assert sharpe_difference(r, b, 252, seed=3) == sharpe_difference(r, b, 252, seed=3)
    assert sharpe_difference(r, b, 252, seed=3)[1] != sharpe_difference(r, b, 252, seed=4)[1]


def test_sharpe_difference_undefined_for_constant_series():
    r = _noise(100, 0.0004, 0.01, 21)
    diff, p = sharpe_difference(r, np.zeros(100), 252)
    assert math.isnan(diff) and math.isnan(p)


def test_constant_nonzero_returns_have_no_sharpe():
    r = np.full(30, 0.01)  # std(ddof=1) is 1.8e-18 in floats, not 0
    b = _noise(30, 0.0, 0.01, 0)
    assert math.isnan(annualised_sharpe(r, 252)) and math.isnan(metrics(r, 252)["sharpe"])
    assert math.isnan(probabilistic_sharpe(r))
    assert all(math.isnan(x) for x in sharpe_difference(r, b, 252))
    # Resamples that miss the one different session are constant: zero Sharpe, not ~1e17.
    r[0] = 0.02
    assert sharpe_difference(r, b, 252)[1] < 0.01


def test_stationary_bootstrap_block_lengths():
    rng = np.random.default_rng(0)
    n = 1000
    idx = scoring._stationary_indices(rng, 200, n, 10)
    assert idx.min() >= 0 and idx.max() < n
    consecutive = (idx[:, 1:] == (idx[:, :-1] + 1) % n).mean()
    assert consecutive == pytest.approx(1 - 1 / 10, abs=0.01)
    iid = scoring._stationary_indices(rng, 200, n, 1)
    assert (iid[:, 1:] == (iid[:, :-1] + 1) % n).mean() == pytest.approx(1 / n, abs=0.002)


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics([0.01, np.nan, 0.02], 252),
        lambda: metrics([0.01, np.inf], 252),
        lambda: metrics([0.01], 252),
        lambda: metrics([[0.01, 0.02]], 252),
        lambda: metrics([0.01, -1.5], 252),
        lambda: metrics([0.01, 0.02], 0),
        lambda: metrics([0.01, 0.02, 0.03], 252, benchmark=[0.01, 0.02]),
        lambda: metrics([0.01, 0.02, 0.03], 252, benchmark=[0.01, np.nan, 0.02]),
        lambda: metrics([0.01, 0.02, 0.03], 252, turnover=[0.1, 0.1]),
        lambda: metrics([0.01, 0.02, 0.03], 252, turnover=[0.1, -0.1, 0.1]),
        lambda: probabilistic_sharpe([0.01, 0.02], float("nan")),
        lambda: deflated_sharpe([0.01, 0.02, 0.0], 0, 1e-4),
        lambda: deflated_sharpe([0.01, 0.02, 0.0], 2.5, 1e-4),
        lambda: deflated_sharpe([0.01, 0.02, 0.0], 5, -1e-4),
        lambda: sharpe_difference([0.01, 0.02, 0.0], [0.01, 0.0], 252),
        lambda: sharpe_difference([0.01, 0.02, 0.0], [0.01, 0.0, 0.03], 252, n_boot=0),
        lambda: sharpe_difference([0.01, 0.02, 0.0], [0.01, 0.0, 0.03], 252, block=0),
        lambda: sharpe_difference([0.01, 0.02, 0.0], [0.01, 0.0, 0.03], 252, block=2.5),
        lambda: annualised_sharpe([0.01, np.nan], 252),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
