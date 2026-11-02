"""Gymnasium environment that trains agents under the benchmark's execution model.

Every trade goes through :class:`rl.bench.engine.Account`, the account the
evaluation harness uses, so an agent is trained on exactly the fills, fees and
delisting rules it will be scored with. The agent decides at a session's
close and trades at the next open; the reward is the log change in equity
over that session, net of fees.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from rl.bench.engine import Account
from rl.bench.features import LOOKBACK, N_FEATURES, feature_panel
from rl.bench.llm import apply_sentiment
from rl.bench.market import MarketData

LOGIT_BOUND = 5.0  # actions are logits in [-LOGIT_BOUND, LOGIT_BOUND]
_MIN_EQUITY = 1e-12


def actions_to_weights(action, listed) -> np.ndarray:
    """Target asset weights from ``N + 1`` logits (one per asset, then cash).

    A softmax over the listed assets and cash; unlisted assets get zero weight.
    The remainder after the asset weights is held as cash.
    """
    logits = np.asarray(action, dtype=float).reshape(-1)
    listed = np.asarray(listed, dtype=bool)
    if logits.size != listed.size + 1:
        raise ValueError("Expected {} logits, got {}".format(listed.size + 1, logits.size))
    keep = np.append(listed, True)
    scores = np.where(keep, logits, -np.inf)
    exp = np.exp(scores - scores.max())
    return (exp / exp.sum())[:-1]


def observation(features: np.ndarray, weights: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
    """Flattened features, current asset weights and the cash weight.

    ``extra`` holds additional per-asset columns (the LLM scores), appended to
    each asset's features before the flattening.
    """
    weights = np.asarray(weights, dtype=float)
    if extra is not None:
        features = np.concatenate([np.asarray(features, dtype=float), np.asarray(extra, dtype=float)], axis=-1)
    return np.concatenate([np.ravel(features), weights, [1.0 - weights.sum()]]).astype(np.float32)


def observation_size(n_assets: int, extra_columns: int = 0) -> int:
    return n_assets * (N_FEATURES + extra_columns) + n_assets + 1


class PortfolioEnv(gym.Env):
    """Long-only portfolio allocation over a :class:`MarketData` panel.

    Parameters
    ----------
    data :
        Sessions the agent may train on; features and fills use only these.
    transaction_cost :
        Proportional fee rate, normally the task's.
    episode_length :
        Sessions per episode, starting at a random session drawn from the
        generator seeded by ``reset(seed=...)``; ``None`` runs once through
        the whole usable range. A length beyond the usable range is clipped
        to it.
    first_session :
        First session an episode may trade on (the first decision is at the
        previous close). Default :data:`LOOKBACK` when ``data`` has at least
        ``2 * LOOKBACK`` sessions, so features are complete, otherwise 1.
    extra :
        Optional per-session, per-asset columns (LLM scores) of shape
        ``(T, N, k)``, already restricted to what was visible at each
        decision close. They are appended to each asset's observation.
    sentiment_strength :
        When positive, the first extra column (the sentiment score) scales
        the trade it agrees or disagrees with, as
        :func:`rl.bench.llm.apply_sentiment`; 0 leaves actions unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: MarketData,
        transaction_cost: float,
        episode_length: Optional[int] = 252,
        first_session: Optional[int] = None,
        extra: Optional[np.ndarray] = None,
        sentiment_strength: float = 0.0,
    ):
        super().__init__()
        n_sessions, n_assets = len(data), len(data.tickers)
        if first_session is None:
            first_session = LOOKBACK if n_sessions >= 2 * LOOKBACK else 1
        if not 1 <= first_session < n_sessions:
            raise ValueError("first_session must be in [1, {}), got {}".format(n_sessions, first_session))
        if episode_length is not None and episode_length < 1:
            raise ValueError("episode_length must be positive")
        self.data = data
        self.transaction_cost = transaction_cost
        self.first_session = first_session
        usable = n_sessions - first_session
        self.episode_length = usable if episode_length is None else min(episode_length, usable)
        self._random_start = episode_length is not None and self.episode_length < usable
        self._features = feature_panel(data)
        self._listed = np.isfinite(data.close)
        if extra is not None:
            extra = np.asarray(extra, dtype=float)
            if extra.shape[:2] != (n_sessions, n_assets) or extra.ndim != 3 or extra.shape[2] < 1:
                raise ValueError(
                    "extra must have shape (sessions, assets, columns) = ({}, {}, k), got {}".format(
                        n_sessions, n_assets, extra.shape
                    )
                )
            if not np.isfinite(extra).all():
                raise ValueError("extra columns must be finite")
            if sentiment_strength < 0:
                raise ValueError("sentiment_strength must be nonnegative")
        elif sentiment_strength:
            raise ValueError("sentiment_strength needs extra score columns")
        self._extra = extra
        self._sentiment_strength = float(sentiment_strength)
        columns = 0 if extra is None else extra.shape[2]
        self.action_space = gym.spaces.Box(-LOGIT_BOUND, LOGIT_BOUND, (n_assets + 1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (observation_size(n_assets, columns),), dtype=np.float32
        )
        self._account: Optional[Account] = None

    def _observe(self) -> np.ndarray:
        extra = None if self._extra is None else self._extra[self._decision]
        return observation(self._features[self._decision], self._account.weights(self._decision), extra)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        last_start = len(self.data) - self.episode_length
        start = int(self.np_random.integers(self.first_session, last_start + 1)) if self._random_start else self.first_session
        self._decision = start - 1  # the close the next action is decided at
        self._stop = start + self.episode_length  # first session not traded
        self._account = Account(self.data, self.transaction_cost)
        self._equity = self._account.equity(self._decision)
        return self._observe(), {"date": self.data.dates[self._decision]}

    def step(self, action):
        if self._account is None:
            raise RuntimeError("Call reset() before step()")
        fill = self._decision + 1
        target = actions_to_weights(action, self._listed[self._decision])
        if self._sentiment_strength > 0:
            target = apply_sentiment(
                target, self._account.weights(self._decision),
                self._extra[self._decision][:, 0], self._sentiment_strength,
            )
        turnover, cost = self._account.trade(fill, target)
        equity = self._account.equity(fill)
        reward = float(np.log(max(equity, _MIN_EQUITY) / max(self._equity, _MIN_EQUITY)))
        self._equity = equity
        self._decision = fill
        truncated = fill + 1 >= self._stop
        info = {"date": self.data.dates[fill], "equity": equity, "turnover": turnover, "cost": cost}
        return self._observe(), reward, False, truncated, info
