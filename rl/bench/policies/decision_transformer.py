"""Decision Transformer (Chen et al., 2021) over logged training trajectories.

A small causal transformer reads ``(return-to-go, observation, action)``
token triples for up to ``context`` sessions and predicts the next target
weights from the observation token; the loss is the cross-entropy to the
logged weights. The trajectories are :func:`rl.bench.policies.offline.build_dataset`'s:
training sessions only, behaviour policies and Dirichlet-noised copies.

At decision time the policy runs in pseudo-episodes of the logged episode
length. Its return-to-go at position ``j`` is ``rate * (length - j)``: the
target is a *rate* of net log return per session, which is what the
engine's account would earn if the target were met. The rate is picked on
the validation window, together with the checkpoint, from quantiles of the
logged episodes' mean reward; validation never enters training. Because the
return-to-go is not updated from realized rewards, ``act`` depends only on
the market view and the policy's own earlier outputs, and ``reset`` restarts
the episode and the context.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl.bench.policies.offline import BCPolicy, listed_mask, update

TARGET_QUANTILES = (0.5, 0.75, 0.9, 1.0)
MAX_CONTEXT = 20


class DecisionTransformer(nn.Module):
    def __init__(self, inputs: int, assets: int, hidden: int, context: int, layers: int, heads: int):
        super().__init__()
        self.assets, self.context = assets, context
        self.rtg = nn.Linear(1, hidden)
        self.state = nn.Linear(inputs, hidden)
        self.action = nn.Linear(assets + 1, hidden)
        self.position = nn.Embedding(context, hidden)
        block = nn.TransformerEncoderLayer(
            hidden, heads, 4 * hidden, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, assets + 1)
        mask = torch.triu(torch.full((3 * context, 3 * context), -torch.inf), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, rtg, obs, actions):
        """Weights for every step of ``(batch, steps <= context)`` sequences.

        ``actions[:, t]`` is the weights chosen at step ``t``; the last step's
        entry cannot influence its own prediction (the causal mask).
        """
        batch, steps = obs.shape[:2]
        position = self.position(torch.arange(steps))
        tokens = torch.stack(
            (self.rtg(rtg) + position, self.state(obs) + position, self.action(actions) + position), dim=2)
        size = 3 * steps
        out = self.encoder(tokens.reshape(batch, size, -1), mask=self.mask[:size, :size], is_causal=True)
        logits = self.head(self.norm(out[:, 1::3]))
        return logits.masked_fill(~listed_mask(obs, self.assets), -torch.inf).softmax(-1)


class DecisionTransformerPolicy(BCPolicy):
    def __init__(self, context: int = 20, layers: int = 2, heads: int = 2,
                 return_scale: float = 10.0, hidden: int = 64, batch_size: int = 64,
                 steps: int = 1000, eval_every: int = 250, transaction_cost: float = 0.001, **kwargs):
        super().__init__(steps=steps, batch_size=batch_size, hidden=hidden, eval_every=eval_every,
                         transaction_cost=transaction_cost, **kwargs)
        if not 1 <= context <= MAX_CONTEXT:
            raise ValueError("context must be in [1, {}]".format(MAX_CONTEXT))
        if layers < 1 or hidden % heads or not 0 < return_scale < float("inf"):
            raise ValueError("Positive layers, hidden divisible by heads and finite return_scale required")
        self.context, self.layers, self.heads = context, layers, heads
        self.return_scale = return_scale
        self.targets = self.rate = None
        self.reset()

    def _make_actor(self, inputs, assets):
        return DecisionTransformer(inputs, assets, self.hidden, self.context, self.layers, self.heads)

    def reset(self):
        self._steps = 0
        self._rtgs, self._states, self._taken = [], [], []

    def _train(self, data, evaluate):
        length = self.provenance["episode_length"]
        count = len(data.rewards) // length
        rewards = data.rewards.astype(np.float64).reshape(count, length)
        rtg = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
        self.targets = [float(np.quantile(rtg[:, 0] / length, q)) for q in TARGET_QUANTILES]
        rtg = torch.from_numpy((rtg * self.return_scale).astype(np.float32)).unsqueeze(-1)
        obs = torch.from_numpy(data.observations).reshape(count, length, -1)
        actions = torch.from_numpy(data.actions).reshape(count, length, -1)
        steps = torch.arange(self.context)
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        for step in range(1, self.steps + 1):
            episode = torch.randint(count, (self.batch_size, 1))
            at = torch.randint(length, (self.batch_size, 1)) + steps
            valid, at = at < length, at.clamp_max(length - 1)
            pred = self.actor(rtg[episode, at], obs[episode, at], actions[episode, at]).clamp_min(1e-12)
            update(-((actions[episode, at] * pred.log()).sum(-1) * valid).sum() / valid.sum(), optimizer)
            if step % self.eval_every == 0 or step == self.steps:
                evaluate(step)

    def _validate(self, history, split):
        best = (-np.inf, {})
        for rate in self.targets:
            self.rate = rate
            score, _ = super()._validate(history, split)
            if score > best[0]:
                best = score, {"rate": rate}
        return best

    def _choose(self, choice):
        self.rate = choice["rate"]

    def _weights(self, obs, view):
        length = self.provenance["episode_length"]
        if self._steps % length == 0:
            self._rtgs, self._states, self._taken = [], [], []
        remaining = self.rate * (length - self._steps % length)
        self._steps += 1
        self._rtgs.append([remaining * self.return_scale])
        self._states.append(obs)
        self._taken.append(np.zeros(len(self.tickers) + 1, dtype=np.float32))
        del self._rtgs[:-self.context], self._states[:-self.context], self._taken[:-self.context]
        with torch.no_grad():
            out = self.actor(
                torch.tensor(self._rtgs, dtype=torch.float32)[None],
                torch.from_numpy(np.stack(self._states))[None],
                torch.from_numpy(np.stack(self._taken))[None])[0, -1]
        self._taken[-1] = out.numpy()
        return out.double().numpy()

    def _metadata(self):
        return {"context": self.context, "layers": self.layers, "heads": self.heads,
                "return_scale": self.return_scale, "targets": self.targets, "rate": self.rate}

    def _restore_metadata(self, meta):
        self.context, self.layers, self.heads = meta["context"], meta["layers"], meta["heads"]
        self.return_scale, self.targets, self.rate = meta["return_scale"], meta["targets"], meta["rate"]
