"""Deep RL policies trained with Stable-Baselines3 on the benchmark's execution model.

An agent observes causal scale-free features of every asset
(:mod:`rl.bench.features`), its current weights and cash, and emits ``N + 1``
logits that are softmaxed into long-only target weights over the listed
assets and cash (:func:`rl.bench.gym_env.actions_to_weights`). Training runs
on :class:`rl.bench.gym_env.PortfolioEnv`, which trades through the
evaluation engine's account, so fills and fees match scoring.

Model selection uses the validation sessions only: at regular intervals the
deterministic policy is run through :func:`rl.bench.engine.simulate` over
the sessions from ``validation_start`` on, with all earlier history visible,
and the checkpoint with the best net-of-fees Sharpe ratio is restored at the
end of training. Training itself never sees a validation session.

Pass the task's ``transaction_cost`` to the constructor: the policy has no
other way to learn the fee it will be scored with.

After training, the deterministic actor is exported to plain arrays
(:class:`Actor`) and every decision uses that export. ``save`` writes only
those arrays, so a frozen policy needs neither the optimizer state nor a
compatible Stable-Baselines3 version to be replayed years later.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from torch import nn

from rl.bench.engine import Decision, simulate
from rl.bench.features import LOOKBACK, latest_features
from rl.bench.gym_env import PortfolioEnv, actions_to_weights, observation
from rl.bench.info import InfoPanel, session_close
from rl.bench.llm import apply_sentiment, empty_scores, score_panel, update_scores
from rl.bench.market import MarketData, MarketView
from rl.bench.policy import Policy

logger = logging.getLogger(__name__)

ALGORITHMS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "ddpg": DDPG}
_OFF_POLICY = ("sac", "td3", "ddpg")
_EXPLORATION_NOISE = 0.1  # action noise (in units of the action range) for DDPG and TD3
_ACTOR_FILE = "actor.npz"
_META_FILE = "meta.json"
_ACTIVATIONS = {nn.Tanh: "tanh", nn.ReLU: "relu"}


@contextlib.contextmanager
def _single_thread():
    """Run torch on one thread so training does not depend on the machine's core count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _Selection(BaseCallback):
    """Tracks the best validation checkpoint while training."""

    def __init__(self, policy: SB3Policy, every: int):
        super().__init__()
        self.policy = policy
        self.every = every
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.best_parameters = None
        self._last = None

    def _evaluate(self):
        self._last = self.num_timesteps
        sharpe = self.policy._validation_sharpe(self.model)
        logger.info("step %d validation Sharpe %.4f", self.num_timesteps, sharpe)
        if sharpe > self.best_sharpe:
            self.best_sharpe, self.best_step = sharpe, self.num_timesteps
            self.best_parameters = copy.deepcopy(self.model.get_parameters())

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self._last != self.num_timesteps:
            self._evaluate()


@dataclass(frozen=True)
class Actor:
    """A trained agent's deterministic action function as plain arrays.

    ``layers`` alternate dense ``(weight, bias)`` pairs and activation names;
    ``output`` is ``"clip"`` (Gaussian policies: clip the mean to the action
    bounds) or ``"squash"`` (tanh-bounded policies: rescale from [-1, 1]).
    """

    layers: tuple
    output: str
    low: np.ndarray
    high: np.ndarray

    @property
    def input_size(self) -> int:
        return self.layers[0][0].shape[1]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        for layer in self.layers:
            if layer == "tanh":
                x = np.tanh(x)
            elif layer == "relu":
                x = np.maximum(x, 0.0)
            else:
                weight, bias = layer
                x = weight @ x + bias
        if self.output == "squash":
            return self.low + 0.5 * (x + 1.0) * (self.high - self.low)
        return np.clip(x, self.low, self.high)

    @classmethod
    def from_model(cls, algo: str, model) -> Actor:
        policy = model.policy
        if algo in ("ppo", "a2c"):
            modules, extra = [policy.mlp_extractor.policy_net, policy.action_net], []
        elif algo == "sac":
            modules, extra = [policy.actor.latent_pi, policy.actor.mu], ["tanh"]
        else:  # td3, ddpg: the actor network ends in tanh itself
            modules, extra = [policy.actor.mu], []
        layers = [layer for module in modules for layer in _array_layers(module)] + extra
        space = model.action_space
        return cls(
            layers=tuple(layers),
            output="squash" if policy.squash_output else "clip",
            low=np.asarray(space.low, dtype=np.float64),
            high=np.asarray(space.high, dtype=np.float64),
        )

    def save(self, path: Path) -> list[str]:
        """Write the arrays to ``path``; returns the layer spec for metadata."""
        arrays, spec = {"low": self.low, "high": self.high}, []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, str):
                spec.append(layer)
            else:
                arrays["w{}".format(i)], arrays["b{}".format(i)] = layer
                spec.append("dense")
        np.savez_compressed(path, **arrays)
        return spec

    @classmethod
    def load(cls, path: Path, spec: list[str], output: str) -> Actor:
        with np.load(path) as arrays:
            layers = tuple(
                (arrays["w{}".format(i)], arrays["b{}".format(i)]) if kind == "dense" else kind
                for i, kind in enumerate(spec)
            )
            return cls(layers=layers, output=output, low=arrays["low"], high=arrays["high"])


def _array_layers(module: nn.Module) -> list:
    """Dense layers as float64 arrays and activations as names, in forward order."""
    children = list(module.children())
    if children:
        return [layer for child in children for layer in _array_layers(child)]
    if isinstance(module, nn.Linear):
        weight = module.weight.detach().cpu().numpy().astype(np.float64)
        return [(weight, module.bias.detach().cpu().numpy().astype(np.float64))]
    if type(module) in _ACTIVATIONS:
        return [_ACTIVATIONS[type(module)]]
    if isinstance(module, (nn.Flatten, nn.Identity)):
        return []
    raise TypeError("Cannot export layer {!r}".format(module))


class SB3Policy(Policy):
    """A Stable-Baselines3 agent (``ppo``, ``a2c``, ``sac``, ``td3`` or ``ddpg``).

    Parameters
    ----------
    algo, timesteps :
        Algorithm and number of environment steps to train for.
    episode_length :
        Sessions per training episode (random start); ``None`` trains on the
        whole training period in one episode.
    learning_rate, net_arch, gamma :
        Passed to the algorithm; ``net_arch`` is the hidden layer sizes.
    eval_every :
        Environment steps between validation evaluations (default
        ``timesteps // 10``).
    algo_kwargs :
        Extra keyword arguments for the SB3 algorithm class.
    transaction_cost :
        Fee rate used for training and validation. The runner must pass the
        task's cost, otherwise the agent trains under the wrong fees.
    llm_strength :
        Opt-in strength of the LLM score features (default 0: none). When
        positive, the two per-asset scores of :mod:`rl.bench.llm` are appended
        to the observation through the point-in-time rule, and the sentiment
        score scales the trades it agrees or disagrees with. A session
        without a fresh score takes the missing defaults (0 and 3).
    crypto :
        Whether the task's sessions close at midnight UTC (Binance) instead
        of the New York close. The runner injects it from the task.
    """

    stochastic = True

    def __init__(
        self,
        algo: str = "ppo",
        timesteps: int = 100_000,
        episode_length: Optional[int] = 252,
        learning_rate: float = 3e-4,
        net_arch=(64, 64),
        gamma: float = 0.99,
        eval_every: Optional[int] = None,
        algo_kwargs: Optional[dict] = None,
        transaction_cost: float = 0.001,
        llm_strength: float = 0.0,
        crypto: bool = False,
    ):
        algo = str(algo).lower()
        if algo not in ALGORITHMS:
            raise ValueError("algo must be one of {}, got {!r}".format(sorted(ALGORITHMS), algo))
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        if isinstance(llm_strength, bool) or not isinstance(llm_strength, (int, float)) or llm_strength < 0:
            raise ValueError("llm_strength must be a nonnegative number")
        self.llm_strength = float(llm_strength)
        self.crypto = bool(crypto)
        self.algo = algo
        self.timesteps = int(timesteps)
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.net_arch = tuple(int(n) for n in net_arch)
        self.gamma = gamma
        self.eval_every = max(1, self.timesteps // 10) if eval_every is None else int(eval_every)
        self.algo_kwargs = dict(algo_kwargs or {})
        self.transaction_cost = transaction_cost
        self.model = None
        self.actor: Optional[Actor] = None
        self._validation = None
        self._validation_scores = None
        self._scores = None
        self._info_seen = 0
        self._last_date = None
        self.best_step: Optional[int] = None
        self.best_sharpe: Optional[float] = None

    def _algorithm(self):
        return ALGORITHMS[self.algo]

    def _decide(self, act, input_size: int, view: MarketView, scores=None) -> np.ndarray:
        features = latest_features(view.history.window(LOOKBACK))
        if self.llm_strength > 0:
            if scores is None:
                scores = empty_scores(len(view.tickers))
                update_scores(scores, view.info, view.tickers, since=self._previous_close(view))
            features = np.concatenate([features, np.asarray(scores, dtype=float)], axis=-1)
        obs = observation(features, view.weights)
        if obs.shape != (input_size,):
            raise ValueError(
                "Model expects observations of size {}, got {}: it was trained on a different universe".format(
                    input_size, obs.shape
                )
            )
        weights = actions_to_weights(act(obs), view.listed)
        if self.llm_strength > 0:
            weights = apply_sentiment(weights, view.weights, np.asarray(scores, dtype=float)[:, 0], self.llm_strength)
        return weights

    def _model_decide(self, model, view: MarketView, scores=None) -> np.ndarray:
        return self._decide(
            lambda obs: model.predict(obs, deterministic=True)[0], model.observation_space.shape[0], view, scores
        )

    def _validation_sharpe(self, model) -> float:
        """Per-session Sharpe ratio of the deterministic policy on the validation sessions."""
        history, start = self._validation

        def act(view):
            scores = None
            if self._validation_scores is not None:
                scores = self._validation_scores[history.index_of(view.date)]
            return self._model_decide(model, view, scores)

        result = simulate(history, act, start, self.transaction_cost)
        returns = result.returns
        spread = returns.std(ddof=1) if returns.size > 1 else 0.0
        return float(returns.mean() / spread) if spread > 1e-12 else 0.0

    def fit(
        self, history: MarketData, validation_start: np.datetime64, seed: int, info: Optional[InfoPanel] = None,
        crypto: Optional[bool] = None,
    ) -> None:
        split = history.index_of(validation_start)
        if split < 2:
            raise ValueError("Need at least two sessions before validation_start to train")
        if crypto is not None:
            self.crypto = bool(crypto)
        scores = None
        if self.llm_strength > 0:
            if info is None:
                raise ValueError("llm_strength needs the submission's declared information source")
            scores = score_panel(info, history.dates, history.tickers, crypto=self.crypto)
        env = PortfolioEnv(
            history.rows(0, split), self.transaction_cost, self.episode_length,
            extra=None if scores is None else scores[:split], sentiment_strength=self.llm_strength,
        )
        has_validation = split < len(history)
        self._validation = (history, split) if has_validation else None
        self._validation_scores = scores if has_validation else None
        self.reset()
        kwargs = dict(self.algo_kwargs)
        if self.algo in _OFF_POLICY:
            kwargs.setdefault("buffer_size", min(self.timesteps, 1_000_000))
        if self.algo in ("td3", "ddpg"):
            size = env.action_space.shape[0]
            kwargs.setdefault(
                "action_noise", NormalActionNoise(np.zeros(size), np.full(size, _EXPLORATION_NOISE))
            )
        with _single_thread():
            set_random_seed(int(seed))
            model = self._algorithm()(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                policy_kwargs={"net_arch": list(self.net_arch)},
                seed=int(seed),
                device="cpu",
                verbose=0,
                **kwargs,
            )
            if has_validation:
                selection = _Selection(self, self.eval_every)
                model.learn(self.timesteps, callback=selection)
                model.set_parameters(selection.best_parameters, exact_match=True)
                self.best_step, self.best_sharpe = selection.best_step, selection.best_sharpe
            else:
                logger.warning("No validation sessions after %s: keeping the final checkpoint", validation_start)
                model.learn(self.timesteps)
                self.best_step, self.best_sharpe = model.num_timesteps, None
        self._validation = None
        self._validation_scores = None
        self.model = model
        self.actor = Actor.from_model(self.algo, model)

    def reset(self) -> None:
        self._info_seen = 0
        self._last_date = None

    def _previous_close(self, view: MarketView):
        """The previous session's close, or ``None`` when the view has one session."""
        dates = view.history.dates
        return session_close(dates[-2], crypto=self.crypto) if len(dates) >= 2 else None

    def _scores_at(self, view: MarketView) -> np.ndarray:
        """The current session's scores; an older item is not carried forward."""
        if len(view.info) < self._info_seen or (self._last_date is not None and view.date <= self._last_date):
            self._info_seen = 0
        scores = empty_scores(len(view.tickers))
        update_scores(scores, view.info, view.tickers, start=self._info_seen, since=self._previous_close(view))
        self._info_seen = len(view.info)
        self._last_date = view.date
        return scores

    def act(self, view: MarketView) -> Decision:
        if self.actor is None:
            raise RuntimeError("SB3Policy has no model: call fit() or load() first")
        scores = self._scores_at(view) if self.llm_strength > 0 else None
        return self._decide(self.actor, self.actor.input_size, view, scores)

    def save(self, directory: Path) -> None:
        if self.actor is None:
            raise RuntimeError("SB3Policy has no model: call fit() or load() first")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        layers = self.actor.save(directory / _ACTOR_FILE)
        meta = {
            "algo": self.algo,
            "layers": layers,
            "output": self.actor.output,
            "best_step": self.best_step,
            "best_sharpe": self.best_sharpe,
        }
        (directory / _META_FILE).write_text(json.dumps(meta, indent=2))

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        meta = json.loads((directory / _META_FILE).read_text())
        if meta["algo"] != self.algo:
            raise ValueError("Saved model is {!r} but this policy is {!r}".format(meta["algo"], self.algo))
        self.actor = Actor.load(directory / _ACTOR_FILE, meta["layers"], meta["output"])
        self.model = None
        self.best_step, self.best_sharpe = meta["best_step"], meta["best_sharpe"]
