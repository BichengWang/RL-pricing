import numpy as np
import pytest

from rl.bench.data import load_task_data
from rl.bench.engine import simulate
from rl.bench.fees import break_even_fee
from rl.bench.policies.baselines import Momentum
from rl.bench.scoring import annualised_sharpe, reprice
from rl.bench.tasks import get_task

TASK = get_task("synthetic")


def _churn(view):
    """Trades most of the book every session, from its current weights and the date."""
    rng = np.random.default_rng(len(view.history))
    return 0.3 * np.roll(view.weights, 1) + 0.7 * rng.dirichlet(np.ones(len(view.tickers)))


@pytest.mark.parametrize("make, min_turnover", [(lambda: Momentum().act, 1), (lambda: _churn, 150)])
def test_repriced_returns_match_exact_engine_runs_at_other_fees(make, min_turnover):
    data = load_task_data(TASK)
    start = int(np.searchsorted(data.dates, np.datetime64(TASK.test_start)))
    charged = TASK.transaction_cost
    base = simulate(data, make(), start, charged)
    assert base.turnover.mean() * TASK.periods_per_year > min_turnover
    for fee in (0.0, 0.0001, 0.0005, 0.0025, 0.005):
        exact = simulate(data, make(), start, fee)
        # The weights a policy sees don't depend on the fee, so it makes the same decisions.
        np.testing.assert_allclose(exact.weights, base.weights, rtol=0, atol=1e-12)
        repriced = annualised_sharpe(reprice(base.equity, base.turnover, charged, fee), TASK.periods_per_year)
        tolerance = 1e-9 if fee == 0 else 0.01
        assert repriced == pytest.approx(annualised_sharpe(exact.returns, TASK.periods_per_year), abs=tolerance)
    assert np.array_equal(reprice(base.equity, base.turnover, charged, charged), base.returns)


def test_break_even_fee_is_where_the_extra_return_is_spent():
    # Entry: the benchmark's returns times (1 + edge) per session, trading `turnover` a session.
    # At fee f its gross factor is (1 + edge)(1 - f turnover) / (1 - c turnover); Sharpe ties at factor 1.
    benchmark = np.random.default_rng(0).normal(0.0004, 0.01, 750)
    charged, turnover = 0.001, np.full(750, 0.02)
    bench = (np.cumprod(np.r_[1.0, 1 + benchmark]), np.zeros(750))

    def runs(edge):
        return [(np.cumprod(np.r_[1.0, (1 + benchmark) * (1 + edge)]), turnover)]

    edge = 0.00005
    expected = (1 - (1 - charged * 0.02) / (1 + edge)) / 0.02  # about 35 bp
    assert break_even_fee(runs(edge), bench, charged, 252) == pytest.approx(expected, abs=1e-10)
    assert break_even_fee(runs(-0.0001), bench, charged, 252) is None  # trails even at 0 bp
    assert break_even_fee(runs(0.001), bench, charged, 252) is None  # still leads at 50 bp
