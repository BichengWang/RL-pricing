"""Training-only engine trajectories, behaviour cloning and simplex networks.

The ten equally represented behaviours are the five built-in monthly rules
and a Dirichlet-perturbed version of each. Holds become the current weights;
a 1e-4 floor on listed assets and cash gives continuous-action learners an
interior target. The *executed* weights and PortfolioEnv's net log reward
are logged, not hypothetical teacher returns. No prices at or after the
validation boundary enter the environment, features, or provenance hash.

Offline learners share this data/selection/persistence lifecycle. Their
optimizers live in their respective modules; only the selected actor is
needed for replay. All inference uses one CPU thread, including after load.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl.bench.engine import simulate
from rl.bench.features import LOOKBACK, N_FEATURES, feature_panel, latest_features
from rl.bench.gym_env import PortfolioEnv, actions_to_weights, observation
from rl.bench.market import MarketData, MarketView
from rl.bench.policies.baselines import (
    EqualWeightRebalance, InverseVolatility, MinimumVariance, Momentum, TrendFollowing,
)
from rl.bench.policies.sb3 import _single_thread
from rl.bench.policy import Policy

BEHAVIOURS = (EqualWeightRebalance, Momentum, MinimumVariance, InverseVolatility, TrendFollowing)


@dataclass
class Trajectories:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    provenance: dict


def build_dataset(
    history: MarketData, validation_start, seed: int, transaction_cost: float,
    episodes_per_behaviour: int = 2, episode_length: int = 128,
    noise_mix: float = 0.2,
) -> Trajectories:
    """Log equally many episodes per teacher and per noisy teacher, strictly pre-split.

    Episodes are finite training tasks: their last transition is terminal
    (no bootstrap into another independently reset account).
    """
    split = history.index_of(validation_start)
    if split < 2:
        raise ValueError("Need at least two training sessions before validation_start")
    if episodes_per_behaviour < 1 or episode_length < 1 or not 0 <= noise_mix <= 1:
        raise ValueError("Positive episode counts/length and noise_mix in [0, 1] required")
    train = history.rows(0, split)
    env = PortfolioEnv(train, transaction_cost, episode_length)
    rng = np.random.default_rng(seed)
    rows, mix, sessions = [], [], []
    for teacher_class in BEHAVIOURS:
        for noisy in (False, True):
            teacher = teacher_class()
            name = teacher_class.__name__ + ("+Dirichlet" if noisy else "")
            mix.append({"policy": name, "episodes": episodes_per_behaviour})
            for _ in range(episodes_per_behaviour):
                teacher.reset()
                obs, info = env.reset(seed=int(rng.integers(2**31)))
                decision = train.index_of(info["date"])
                first = decision
                done = False
                while not done:
                    view = MarketView(train.rows(0, decision + 1), obs[-len(train.tickers)-1:-1])
                    target = teacher.act(view)
                    target = view.weights if target is None else target
                    full = np.append(target, max(0.0, 1.0 - target.sum()))
                    keep = np.append(view.listed, True)
                    if noisy:
                        noise = np.zeros_like(full)
                        noise[keep] = rng.dirichlet(np.ones(keep.sum()))
                        full = (1 - noise_mix) * full + noise_mix * noise
                    full = np.where(keep, np.maximum(full, 1e-4), 0.0)
                    full /= full.sum()
                    logits = np.log(np.maximum(full, 1e-30))
                    executed = actions_to_weights(logits, view.listed)
                    action = np.append(executed, 1.0 - executed.sum())
                    nxt, reward, terminated, truncated, info = env.step(logits)
                    done = terminated or truncated
                    rows.append((obs, action, reward, nxt, done))
                    obs, decision = nxt, decision + 1
                sessions.append({"behaviour": name, "decision_start": str(train.dates[first]),
                                 "last_fill": str(train.dates[decision]), "steps": decision - first})
    obs, actions, rewards, nxt, terminals = zip(*rows)
    return Trajectories(
        np.asarray(obs, dtype=np.float32), np.asarray(actions, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32), np.asarray(nxt, dtype=np.float32),
        np.asarray(terminals, dtype=np.float32),
        {"seed": int(seed), "behaviour_mix": mix, "sessions": sessions,
         "training_sha256": train.sha256(), "training_start": str(train.dates[0]),
         "training_end": str(train.dates[-1]), "validation_start": str(validation_start),
         "tickers": list(train.tickers), "transaction_cost": transaction_cost,
         "episode_length": env.episode_length, "noise_mix": noise_mix,
         "weight_floor": 1e-4, "reward": "PortfolioEnv net log return"},
    )


def mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, outputs))


def listed_mask(obs: torch.Tensor, assets: int) -> torch.Tensor:
    listed = obs[..., :assets * N_FEATURES].reshape(*obs.shape[:-1], assets, N_FEATURES)[..., -1] > 0
    return torch.cat((listed, torch.ones_like(listed[..., :1])), dim=-1)


class WeightActor(nn.Module):
    def __init__(self, inputs: int, assets: int, hidden: int):
        super().__init__()
        self.assets = assets
        self.net = mlp(inputs, assets + 1, hidden)

    def forward(self, obs):
        return self.net(obs).masked_fill(~listed_mask(obs, self.assets), -torch.inf).softmax(-1)


class SimplexActor(WeightActor):
    """Dirichlet actor on the listed-asset + cash simplex, with exact densities.

    Missing assets are integrated out rather than assigned a tiny allocation.
    This replaces CORL's tanh Gaussian, whose cube support is inappropriate
    for weights. The deterministic evaluation action is the Dirichlet mean.
    """

    def concentration(self, obs):
        return F.softplus(self.net(obs)).clamp(0.05, 100.0)

    def forward(self, obs):
        alpha = self.concentration(obs) * listed_mask(obs, self.assets)
        return alpha / alpha.sum(-1, keepdim=True)

    def log_prob(self, obs, weights):
        keep = listed_mask(obs, self.assets)
        alpha = self.concentration(obs)
        return (torch.lgamma((alpha * keep).sum(-1))
                - (torch.lgamma(alpha) * keep).sum(-1)
                + ((alpha - 1) * weights.clamp_min(1e-12).log() * keep).sum(-1))

    def sample(self, obs):
        alpha = self.concentration(obs)
        draws = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).rsample()
        draws = draws.clamp_min(1e-12) * listed_mask(obs, self.assets)
        weights = draws / draws.sum(-1, keepdim=True)
        return weights, self.log_prob(obs, weights)


def update(loss, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite offline training objective")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def soft_update(target, source, tau):
    with torch.no_grad():
        for old, new in zip(target.parameters(), source.parameters()):
            old.lerp_(new, tau)


class BCPolicy(Policy):
    """Softmax MLP behaviour cloning; select checkpoints by validation net Sharpe."""

    stochastic = True
    actor_class = WeightActor

    def __init__(
        self, steps: int = 1000, batch_size: int = 128, hidden: int = 64,
        learning_rate: float = 3e-4, eval_every: int = 250,
        episodes_per_behaviour: int = 2, episode_length: int = 128,
        noise_mix: float = 0.2, transaction_cost: float = 0.001,
    ):
        if min(steps, batch_size, hidden, eval_every, episodes_per_behaviour, episode_length) < 1:
            raise ValueError("Training sizes and intervals must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0 or not 0 <= noise_mix <= 1:
            raise ValueError("Positive learning rate and noise_mix in [0, 1] required")
        if not np.isfinite(transaction_cost) or not 0 <= transaction_cost < 1:
            raise ValueError("transaction_cost must be in [0, 1)")
        self.steps, self.batch_size, self.hidden = steps, batch_size, hidden
        self.learning_rate, self.eval_every = learning_rate, eval_every
        self.episodes_per_behaviour, self.episode_length = episodes_per_behaviour, episode_length
        self.noise_mix, self.transaction_cost = noise_mix, transaction_cost
        self.actor = None
        self.best_step = self.best_sharpe = None
        self.provenance = None
        self._selection_features = None

    def _make_actor(self, inputs, assets):
        return self.actor_class(inputs, assets, self.hidden)

    def _train(self, data, evaluate):
        obs, actions = torch.from_numpy(data.observations), torch.from_numpy(data.actions)
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        for step in range(1, self.steps + 1):
            idx = torch.randint(len(obs), (self.batch_size,))
            pred = self.actor(obs[idx]).clamp_min(1e-12)
            update(-(actions[idx] * pred.log()).sum(-1).mean(), optimizer)
            if step % self.eval_every == 0 or step == self.steps:
                evaluate(step)

    def _obs(self, view):
        if tuple(view.tickers) != self.tickers:
            raise ValueError("Policy was trained on a different universe or ticker order")
        features = (latest_features(view.history.window(LOOKBACK)) if self._selection_features is None
                    else self._selection_features[len(view.history) - 1])
        return observation(features, view.weights)

    def _weights(self, obs, view):
        with torch.no_grad():
            return self.actor(torch.from_numpy(obs).unsqueeze(0))[0].double().numpy()

    def act(self, view):
        if self.actor is None:
            raise RuntimeError("Call fit() or load() first")
        with _single_thread():
            full = self._weights(self._obs(view), view)
        # Float64 normalization keeps sum <= 1 even for near-zero cash.
        full = np.where(np.append(view.listed, True), np.maximum(full, 0), 0)
        return (full / full.sum())[:-1]

    def _validate(self, history, split):
        """Validation net Sharpe of the current actor and the policy settings it used."""
        self.reset()
        result = simulate(history, self.act, split, self.transaction_cost)
        spread = result.returns.std(ddof=1) if result.returns.size > 1 else 0
        return (float(result.returns.mean() / spread) if spread > 1e-12 else 0.0), {}

    def _choose(self, choice):
        pass

    def fit(self, history: MarketData, validation_start, seed: int):
        split = history.index_of(validation_start)
        if not 2 <= split < len(history):
            raise ValueError("Need training and validation sessions")
        data = build_dataset(history, validation_start, seed, self.transaction_cost,
                             self.episodes_per_behaviour, self.episode_length, self.noise_mix)
        self.tickers = tuple(history.tickers)
        self.provenance = data.provenance
        self.best_sharpe, self.best_step = -np.inf, None
        best = None
        with _single_thread(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = self._make_actor(data.observations.shape[1], len(self.tickers))
            self._selection_features = feature_panel(history)

            def evaluate(step):
                nonlocal best
                self.actor.eval()
                score, choice = self._validate(history, split)
                if score > self.best_sharpe:
                    best = copy.deepcopy(self.actor.state_dict()), choice
                    self.best_sharpe, self.best_step = score, step
                self.actor.train()

            try:
                self._train(data, evaluate)
                self.actor.load_state_dict(best[0])
                self._choose(best[1])
                self.actor.eval()
            finally:
                self._selection_features = None
                self.reset()

    def _metadata(self):
        return {}

    def _restore_metadata(self, meta):
        pass

    def save(self, directory):
        if self.actor is None:
            raise RuntimeError("Call fit() or load() first")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / "actor.pt")
        meta = {"policy": type(self).__name__, "hidden": self.hidden, "tickers": self.tickers,
                "best_step": self.best_step, "best_sharpe": self.best_sharpe,
                "provenance": self.provenance, "transaction_cost": self.transaction_cost,
                **self._metadata()}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def load(self, directory):
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        if meta["policy"] != type(self).__name__:
            raise ValueError("Artifact belongs to a different policy")
        self.hidden, self.tickers = meta["hidden"], tuple(meta["tickers"])
        self.transaction_cost = meta["transaction_cost"]
        self.provenance = meta["provenance"]
        self.best_step, self.best_sharpe = meta["best_step"], meta["best_sharpe"]
        self._restore_metadata(meta)
        self.actor = self._make_actor(len(self.tickers) * (N_FEATURES + 1) + 1, len(self.tickers))
        self.actor.load_state_dict(torch.load(directory / "actor.pt", map_location="cpu", weights_only=True))
        self.actor.eval()
        self.reset()
