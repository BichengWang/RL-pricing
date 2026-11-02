"""Implicit Q-learning (Kostrikov et al., 2021), CPU portfolio-weight variant.

Update order and objectives follow CORL algorithms/offline/iql.py at commit
 eeeeef2671235fcce3a3eb255576a207d78840e6:
https://github.com/corl-team/CORL/blob/eeeeef2671235fcce3a3eb255576a207d78840e6/algorithms/offline/iql.py

Twin target Q, expectile V regression, bootstrapped Q regression and capped
advantage-weighted log likelihood are retained. A masked Dirichlet replaces
the reference Gaussian so the continuous actions are tradable weights.
Rewards are the logged engine net log returns times a declared fixed scale;
finite logged episodes are terminal. Only the selected actor is persisted.
"""

import copy

import torch
from torch.nn import functional as F

from rl.bench.policies.offline import BCPolicy, SimplexActor, mlp, soft_update, update


def expectile_loss(residual, expectile):
    return (torch.where(residual > 0, expectile, 1 - expectile) * residual.square()).mean()


class IQLPolicy(BCPolicy):
    actor_class = SimplexActor

    def __init__(self, expectile=0.7, beta=3.0, gamma=0.99, tau=0.005,
                 reward_scale=100.0, transaction_cost=0.001, **kwargs):
        super().__init__(transaction_cost=transaction_cost, **kwargs)
        if not 0 < expectile < 1 or not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError("Invalid expectile, gamma or tau")
        if not 0 < beta < float("inf") or not 0 < reward_scale < float("inf"):
            raise ValueError("beta and reward_scale must be finite and positive")
        self.expectile, self.beta, self.gamma, self.tau = expectile, beta, gamma, tau
        self.reward_scale = reward_scale

    def _train(self, data, evaluate):
        obs, actions, rewards, nxt, done = [torch.from_numpy(x) for x in (
            data.observations, data.actions, data.rewards, data.next_observations, data.terminals)]
        inputs, action_size = obs.shape[1], actions.shape[1]
        qs = torch.nn.ModuleList([mlp(inputs + action_size, 1, self.hidden) for _ in range(2)])
        target = copy.deepcopy(qs).requires_grad_(False)
        value = mlp(inputs, 1, self.hidden)
        q_opt = torch.optim.Adam(qs.parameters(), lr=self.learning_rate)
        v_opt = torch.optim.Adam(value.parameters(), lr=self.learning_rate)
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        for step in range(1, self.steps + 1):
            idx = torch.randint(len(obs), (self.batch_size,))
            s, a, r, ns, d = obs[idx], actions[idx], rewards[idx], nxt[idx], done[idx]
            sa = torch.cat((s, a), dim=-1)
            with torch.no_grad():
                q = torch.minimum(target[0](sa), target[1](sa)).squeeze(-1)
                next_v = value(ns).squeeze(-1)
            advantage = q - value(s).squeeze(-1)
            update(expectile_loss(advantage, self.expectile), v_opt)
            goal = self.reward_scale * r + self.gamma * (1 - d) * next_v
            update(sum(F.mse_loss(net(sa).squeeze(-1), goal) for net in qs) / 2, q_opt)
            soft_update(target, qs, self.tau)
            weights = (self.beta * advantage.detach()).clamp(max=4.605170186).exp()
            update(-(weights * self.actor.log_prob(s, a)).mean(), actor_opt)
            if step % self.eval_every == 0 or step == self.steps:
                evaluate(step)
