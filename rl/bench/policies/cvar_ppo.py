"""PPO with a CVaR penalty on the episode return (CPPO), on the SB3 PPO path.

Follows the CVaR term of FinRL-DeepSeek's ``train_cppo.py`` (arXiv 2502.07393),
https://github.com/benstaf/FinRL_DeepSeek/blob/5c21a923214bca6370800efd45f8c6c1ef776ae7/train_cppo.py
which lowers the advantage of every step whose estimated episode return,
``return so far + V(s_t)``, falls below a threshold ``nu`` by
``lambda / (1 - alpha) * (nu - estimate)``: the Rockafellar-Uryasev gradient
of ``-lambda * CVaR_alpha`` of the episode return.

Differences from the reference: it runs inside Stable-Baselines3's PPO
(rollouts, GAE, clipped objective, value loss) instead of a copied trainer;
``nu`` is the empirical ``(1 - alpha)`` quantile of the rollout's estimates,
the minimiser of the Rockafellar-Uryasev objective, instead of a delayed
running mean; the penalty weight ``cvar_penalty`` is fixed instead of being
raised by dual ascent; and the reference's cap on the penalty is dropped, so
the declared weight is what is applied. ``cvar_alpha`` is the confidence
level: the penalty targets the worst ``1 - cvar_alpha`` of the estimates.

Only the actor is saved, as for every SB3 policy, so replay is the ordinary
PPO actor export.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from rl.bench.features import N_FEATURES
from rl.bench.gym_env import LOGIT_BOUND
from rl.bench.llm import apply_sentiment, risk_factors
from rl.bench.policies.sb3 import SB3Policy


def cvar_penalty(rewards, values, episode_starts, running, alpha: float, weight: float, factors=None):
    """Per-step advantage penalty and the running episode return after the rollout.

    ``rewards``, ``values`` and ``episode_starts`` are ``(steps, envs)``;
    ``running`` is each environment's return so far before the first step.
    ``factors`` optionally multiplies each step's estimated episode return (the
    LLM risk multiplier).
    """
    before = np.empty(rewards.shape)
    for t in range(len(rewards)):
        running = np.where(episode_starts[t] > 0, 0.0, running)
        before[t] = running
        running = running + rewards[t]
    estimate = before + values
    if factors is not None:
        factors = np.asarray(factors, dtype=estimate.dtype)
        if factors.shape != estimate.shape:
            raise ValueError("factors must have the shape of the return estimates")
        estimate = estimate * factors
    nu = np.quantile(estimate, 1.0 - alpha)
    return weight / (1.0 - alpha) * np.maximum(nu - estimate, 0.0), running


class CVaRPPO(PPO):
    """:class:`PPO` whose tail steps lose advantage, optionally risk-scaled by LLM scores.

    ``risk_strength`` (0 by default, no effect) reads the two LLM score columns
    that :mod:`rl.bench.gym_env` appends to each observation and multiplies the
    estimated return by the holdings-weighted risk multiplier of
    :func:`rl.bench.llm.risk_factors`, using the target weights the action
    implies before the fill.
    """

    def __init__(self, *args, cvar_alpha: float = 0.9, cvar_penalty: float = 1.0,
                 risk_strength: float = 0.0, **kwargs):
        if not 0 < cvar_alpha < 1 or not 0 <= cvar_penalty < float("inf"):
            raise ValueError("cvar_alpha must be in (0, 1) and cvar_penalty finite and nonnegative")
        if risk_strength < 0:
            raise ValueError("risk_strength must be nonnegative")
        self.cvar_alpha, self.cvar_penalty = cvar_alpha, cvar_penalty
        self.risk_strength = float(risk_strength)
        super().__init__(*args, **kwargs)
        self._running = np.zeros(self.n_envs)

    def _risk_factors(self, observations, actions) -> np.ndarray:
        n_assets = actions.shape[-1] - 1
        width = N_FEATURES + 2
        per_asset = np.asarray(observations[..., : n_assets * width], dtype=float).reshape(
            observations.shape[:-1] + (n_assets, width)
        )
        listed = per_asset[..., N_FEATURES - 1] > 0.5
        sentiment = per_asset[..., N_FEATURES]
        risk = per_asset[..., N_FEATURES + 1]
        current = np.asarray(observations[..., -(n_assets + 1):-1], dtype=float)
        keep = np.concatenate([listed, np.ones(listed.shape[:-1] + (1,), dtype=bool)], axis=-1)
        # SB3 clips sampled actions to the action space before env.step but stores the raw ones.
        clipped = np.clip(np.asarray(actions, dtype=float), -LOGIT_BOUND, LOGIT_BOUND)
        logits = np.where(keep, clipped, -np.inf)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        target = (exp / exp.sum(axis=-1, keepdims=True))[..., :-1]
        target = apply_sentiment(target, current, sentiment, self.risk_strength)
        return risk_factors(target, risk, self.risk_strength)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        proceed = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if proceed and self.cvar_penalty > 0:
            factors = None
            if self.risk_strength > 0:
                factors = self._risk_factors(rollout_buffer.observations, rollout_buffer.actions)
            penalty, self._running = cvar_penalty(
                rollout_buffer.rewards, rollout_buffer.values, rollout_buffer.episode_starts,
                self._running, self.cvar_alpha, self.cvar_penalty, factors)
            rollout_buffer.advantages -= penalty.astype(rollout_buffer.advantages.dtype)
        return proceed


class CVaRPPOPolicy(SB3Policy):
    """:class:`SB3Policy` PPO trained with :class:`CVaRPPO`; other parameters as there."""

    def __init__(self, cvar_alpha: float = 0.9, cvar_penalty: float = 1.0,
                 transaction_cost: float = 0.001, **kwargs):
        super().__init__(algo="ppo", transaction_cost=transaction_cost, **kwargs)
        self.algo_kwargs = {"cvar_alpha": cvar_alpha, "cvar_penalty": cvar_penalty, **self.algo_kwargs}
        if self.llm_strength > 0:
            self.algo_kwargs["risk_strength"] = self.llm_strength

    def _algorithm(self):
        return CVaRPPO
