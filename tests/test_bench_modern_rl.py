from pathlib import Path

import numpy as np
import pytest
import torch

from rl.bench.data import load_task_data
from rl.bench.engine import simulate
from rl.bench.market import MarketData, MarketView
from rl.bench.policies.cql import CQLPolicy
from rl.bench.policies.cvar_ppo import CVaRPPO, CVaRPPOPolicy, cvar_penalty
from rl.bench.policies.decision_transformer import DecisionTransformer, DecisionTransformerPolicy
from rl.bench.policies.iql import IQLPolicy
from rl.bench.policies.offline import BEHAVIOURS, BCPolicy, build_dataset
from rl.bench.runner import build_policy
from rl.bench.submission import Submission
from rl.bench.tasks import get_task

COST = 0.001
VALIDATION = np.datetime64("2016-06-01")
TINY = dict(steps=20, eval_every=10, hidden=16, batch_size=16, episodes_per_behaviour=1, episode_length=20)
POLICIES = {
    "bc": (BCPolicy, TINY),
    "iql": (IQLPolicy, TINY),
    "cql": (CQLPolicy, {**TINY, "proposals": 2}),
    "decision-transformer": (DecisionTransformerPolicy, {**TINY, "context": 4, "heads": 2, "layers": 1}),
    "cvar-ppo": (CVaRPPOPolicy, {"timesteps": 64, "episode_length": 20, "net_arch": [8],
                                 "algo_kwargs": {"n_steps": 32, "batch_size": 16}}),
}


@pytest.fixture(scope="module")
def market():
    return load_task_data(get_task("synthetic")).between("2015-06-01", "2017-06-01")


def poisoned(data: MarketData, start: int) -> MarketData:
    """``data`` with every price and volume from session ``start`` on replaced."""
    arrays = {name: getattr(data, name).copy() for name in ("open", "high", "low", "close", "volume")}
    rng = np.random.default_rng(99)
    for array in arrays.values():
        array[start:] = array[start:] * rng.uniform(0.2, 5.0, array[start:].shape)
    return MarketData(data.dates, data.tickers, **arrays)


def test_dataset_ignores_validation_and_test_sessions(market):
    split = market.index_of(VALIDATION)
    spec = (VALIDATION, 3, COST, 1, 20)
    clean = build_dataset(market, *spec)
    other = build_dataset(poisoned(market, split), *spec)
    for name in ("observations", "actions", "rewards", "next_observations", "terminals"):
        assert np.array_equal(getattr(clean, name), getattr(other, name))
    assert clean.provenance == other.provenance
    assert clean.provenance["training_end"] == str(market.dates[split - 1])
    assert max(s["last_fill"] for s in clean.provenance["sessions"]) <= clean.provenance["training_end"]
    mix = [m["policy"] for m in clean.provenance["behaviour_mix"]]
    assert mix == [b.__name__ + noise for b in BEHAVIOURS for noise in ("", "+Dirichlet")]
    assert len(clean.rewards) == 2 * len(BEHAVIOURS) * 20
    assert clean.terminals.reshape(-1, 20)[:, -1].all() and not clean.terminals.reshape(-1, 20)[:, :-1].any()
    assert not np.array_equal(clean.actions, build_dataset(market, *(VALIDATION, 4, COST, 1, 20)).actions)


def test_logged_rewards_are_what_the_engine_pays_for_the_logged_weights(market):
    split = market.index_of(VALIDATION)
    data = build_dataset(market, VALIDATION, 5, COST, 1, 20)
    train = market.rows(0, split)
    start = train.index_of(data.provenance["sessions"][3]["decision_start"])
    chosen = data.actions[3 * 20:4 * 20, :-1].astype(float)
    steps = iter(chosen)
    result = simulate(train, lambda view: next(steps), start + 1, COST, stop=start + 21)
    np.testing.assert_allclose(np.log1p(result.returns), data.rewards[3 * 20:4 * 20], atol=1e-5)
    np.testing.assert_allclose(data.actions.sum(-1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", POLICIES)
def test_saved_policy_replays_the_fitted_one_exactly(name, market, tmp_path):
    cls, params = POLICIES[name]
    fitted = cls(transaction_cost=COST, **params)
    fitted.fit(market.between(None, "2017-01-01"), VALIDATION, 2)
    fitted.save(tmp_path)
    restored = cls(transaction_cost=COST, **params)
    restored.load(tmp_path)
    start = market.index_of("2017-01-01")
    runs = []
    for policy in (fitted, restored):
        policy.reset()
        runs.append(simulate(market, policy.act, start, COST))
    assert runs[0].weights.std(axis=0).max() > 0
    assert np.array_equal(runs[0].weights, runs[1].weights)
    assert np.array_equal(runs[0].equity, runs[1].equity)
    restored.reset()
    assert np.array_equal(simulate(market, restored.act, start, COST).equity, runs[0].equity)


@pytest.mark.parametrize("name", ["bc", "iql", "cql", "decision-transformer"])
def test_checkpoint_is_the_best_validation_sharpe_and_ignores_later_sessions(name, market):
    cls, params = POLICIES[name]
    history = market.between(None, "2017-01-01")
    policy = cls(transaction_cost=COST, **params)
    policy.fit(history, VALIDATION, 1)
    policy.reset()
    returns = simulate(history, policy.act, history.index_of(VALIDATION), COST).returns
    assert returns.mean() / returns.std(ddof=1) == pytest.approx(policy.best_sharpe, abs=1e-9)
    assert policy.best_step in (10, 20)
    assert policy.provenance["training_end"] < str(VALIDATION)
    assert policy.provenance["training_sha256"] == market.rows(0, market.index_of(VALIDATION)).sha256()


def test_policy_rejects_another_universe(market, tmp_path):
    policy = BCPolicy(transaction_cost=COST, **TINY)
    policy.fit(market.between(None, "2017-01-01"), VALIDATION, 0)
    with pytest.raises(ValueError, match="different universe"):
        policy.act(MarketView(market.select(market.tickers[::-1]).rows(0, 300), np.zeros(5)))


def test_decision_transformer_is_causal_and_context_is_bounded():
    torch.manual_seed(0)
    model = DecisionTransformer(inputs=73, assets=6, hidden=16, context=5, layers=2, heads=2).eval()

    def sequence():
        obs = torch.randn(1, 5, 73)
        obs[..., 10:66:11] = 1.0  # every asset listed
        return torch.randn(1, 5, 1), obs, torch.rand(1, 5, 7)

    rtg, obs, actions = sequence()
    base = model(rtg, obs, actions)
    # Steps 3 and 4 and the action at step 2 come after the output of steps 0-2.
    other_rtg, other_obs, other_actions = sequence()
    rtg[:, 3:], obs[:, 3:], actions[:, 2:] = other_rtg[:, 3:], other_obs[:, 3:], other_actions[:, 2:]
    out = model(rtg, obs, actions)
    assert torch.allclose(out[:, :3], base[:, :3], atol=1e-6)
    assert not torch.allclose(out[:, 3:], base[:, 3:], atol=1e-3)
    with pytest.raises(ValueError, match="context"):
        DecisionTransformerPolicy(context=21)
    for bad in (dict(heads=0), dict(heads=3), dict(layers=0)):
        with pytest.raises(ValueError, match="Positive layers and heads"):
            DecisionTransformerPolicy(**bad)


def test_cvar_penalty_hits_the_worst_estimated_returns_in_proportion_to_the_weight():
    # Four environments, four steps, constant rewards; V(s_t) is the remaining return, so the
    # estimate return-so-far + V is each episode's final return: -0.4, 0.1, 0.2 and 0.3.
    final = np.array([-0.4, 0.1, 0.2, 0.3])
    rewards = np.tile(final / 4, (4, 1))
    values = rewards * np.arange(4, 0, -1)[:, None]
    starts = np.zeros((4, 4))
    low, carry = cvar_penalty(rewards, values, starts, np.zeros(4), alpha=0.5, weight=1.0)
    np.testing.assert_allclose(low[:, 0], 0.55 / 0.5)
    np.testing.assert_allclose(low[:, 1], 0.05 / 0.5)
    assert (low[:, 2:] == 0).all()
    np.testing.assert_allclose(carry, final)
    high, _ = cvar_penalty(rewards, values, starts, np.zeros(4), alpha=0.5, weight=3.0)
    np.testing.assert_allclose(high, 3 * low)
    # Identical episodes have no tail to penalize.
    flat, _ = cvar_penalty(rewards[:, :1].repeat(4, 1), values[:, :1].repeat(4, 1), starts, np.zeros(4), 0.5, 1.0)
    assert (flat == 0).all()
    # A restart zeroes the running return: the later steps' estimate is milder than the true loss.
    restarted = starts.copy()
    restarted[2, 0] = 1
    again, _ = cvar_penalty(rewards, values, restarted, np.zeros(4), alpha=0.5, weight=1.0)
    assert again[3, 0] < low[3, 0]
    with pytest.raises(ValueError, match="cvar_alpha"):
        CVaRPPO("MlpPolicy", "Pendulum-v1", cvar_alpha=1.0)


def test_cvar_ppo_lowers_the_advantages_of_the_tail_steps_only():
    seen = {}

    class Probe(CVaRPPO):
        def train(self):
            seen[self.cvar_penalty] = self.rollout_buffer.advantages.copy()
            super().train()

    for weight in (0.0, 2.0):
        model = Probe("MlpPolicy", "Pendulum-v1", n_steps=100, batch_size=50, seed=0, device="cpu",
                      cvar_alpha=0.8, cvar_penalty=weight)
        model.learn(100)
    drop = seen[0.0] - seen[2.0]
    assert (drop >= 0).all() and (drop > 0).any() and (drop == 0).any()


@pytest.mark.parametrize("name", POLICIES)
def test_submission_trains_under_the_task_fee(name):
    task = get_task("us-sectors")
    submission = Submission.load(Path(__file__).parent.parent / "submissions" / name)
    policy = build_policy(submission, task)
    assert policy.stochastic and policy.transaction_cost == task.transaction_cost
    assert submission.kind == "rl" and submission.trials > 1
