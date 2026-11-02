"""Conservative Q-learning (Kumar et al., 2020), CPU simplex-action variant.

Follows the twin-critic SAC/CQL(H) updates in CORL algorithms/offline/cql.py,
commit eeeeef2671235fcce3a3eb255576a207d78840e6:
https://github.com/corl-team/CORL/blob/eeeeef2671235fcce3a3eb255576a207d78840e6/algorithms/offline/cql.py

Differences: masked Dirichlet actors and uniform-simplex proposals replace
cube actions; proposal densities use the same simplex measure. Fixed entropy
and conservative coefficients replace automatic Lagrange tuning. The
conservative log integral uses current-state actor and uniform proposals
(no next-state proposals, which can have a different listing support).
Net log rewards have a declared fixed scale. Replay needs only the actor.
"""

import copy
import math

import torch
from torch.nn import functional as F

from rl.bench.policies.offline import (
    BCPolicy, SimplexActor, listed_mask, mlp, soft_update, update,
)


def conservative_gap(q_samples, log_density, q_data):
    """Importance-sampled log integral minus the logged-action Q (CQL(H))."""
    return (torch.logsumexp(q_samples - log_density, dim=1)
            - math.log(q_samples.shape[1]) - q_data).mean()


class CQLPolicy(BCPolicy):
    actor_class = SimplexActor

    def __init__(self, cql_weight=1.0, entropy_weight=0.01, proposals=5, gamma=0.99,
                 tau=0.005, reward_scale=100.0, transaction_cost=0.001, **kwargs):
        super().__init__(transaction_cost=transaction_cost, **kwargs)
        if not 0 <= cql_weight < float("inf") or not 0 <= entropy_weight < float("inf"):
            raise ValueError("Finite nonnegative CQL and entropy weights required")
        if proposals < 1 or not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError("Invalid proposals, gamma or tau")
        if not 0 < reward_scale < float("inf"):
            raise ValueError("reward_scale must be finite and positive")
        self.cql_weight, self.entropy_weight, self.proposals = cql_weight, entropy_weight, proposals
        self.gamma, self.tau, self.reward_scale = gamma, tau, reward_scale

    def _train(self, data, evaluate):
        obs, actions, rewards, nxt, done = [torch.from_numpy(x) for x in (
            data.observations, data.actions, data.rewards, data.next_observations, data.terminals)]
        inputs, action_size = obs.shape[1], actions.shape[1]
        qs = torch.nn.ModuleList([mlp(inputs + action_size, 1, self.hidden) for _ in range(2)])
        target = copy.deepcopy(qs).requires_grad_(False)
        q_opt = torch.optim.Adam(qs.parameters(), lr=self.learning_rate)
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        for step in range(1, self.steps + 1):
            idx = torch.randint(len(obs), (self.batch_size,))
            s, a, r, ns, d = obs[idx], actions[idx], rewards[idx], nxt[idx], done[idx]
            with torch.no_grad():
                na, logp = self.actor.sample(ns)
                nsa = torch.cat((ns, na), dim=-1)
                next_q = torch.minimum(target[0](nsa), target[1](nsa)).squeeze(-1)
                goal = self.reward_scale * r + self.gamma * (1 - d) * (next_q - self.entropy_weight * logp)
                repeated = s[:, None, :].expand(-1, self.proposals, -1)
                pa, plogp = self.actor.sample(repeated)
                keep = listed_mask(repeated, self.actor.assets)
                uniform = torch.distributions.Exponential(torch.ones_like(pa)).sample() * keep
                uniform /= uniform.sum(-1, keepdim=True)
                # Uniform density on a (k-1)-simplex is (k-1)!.
                ulogp = torch.lgamma(keep.sum(-1).float())
                candidates = torch.cat((uniform, pa), dim=1)
                density = torch.cat((ulogp, plogp), dim=1)
                states = s[:, None, :].expand(-1, 2 * self.proposals, -1)
                pairs = torch.cat((states, candidates), dim=-1)
            loss = 0
            for net in qs:
                q_data = net(torch.cat((s, a), dim=-1)).squeeze(-1)
                q_samples = net(pairs).squeeze(-1)
                loss = loss + F.mse_loss(q_data, goal) + self.cql_weight * conservative_gap(q_samples, density, q_data)
            update(loss, q_opt)
            qs.requires_grad_(False)
            pa, logp = self.actor.sample(s)
            pair = torch.cat((s, pa), dim=-1)
            q = torch.minimum(qs[0](pair), qs[1](pair)).squeeze(-1)
            update((self.entropy_weight * logp - q).mean(), actor_opt)
            qs.requires_grad_(True)
            soft_update(target, qs, self.tau)
            if step % self.eval_every == 0 or step == self.steps:
                evaluate(step)
