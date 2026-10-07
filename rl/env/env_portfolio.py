import logging
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.config import config

logger = logging.getLogger(__name__)


class StockPortfolioEnv(gym.Env):
    """A portfolio allocation environment with the gymnasium API.

    Each step the agent outputs one score per stock; a softmax turns the
    scores into long-only portfolio weights that are held until the next day.

    Attributes
    ----------
        df: DataFrame
            input data indexed by day number, with ``cov_list`` (see
            :func:`rl.preprocessing.preprocessors.add_covariance_matrix`)
            and the technical indicator columns
        stock_dim : int
            number of unique stocks
        hmax : int
            unused; kept for signature compatibility
        initial_amount : int
            start money
        transaction_cost_pct: float
            cost as a fraction of the value traded when rebalancing
        reward_scaling: float
            scaling factor for the reward
        reward_type: str
            ``"asset_change"`` (default) rewards the change in portfolio
            value, ``"log_return"`` the log of the portfolio's gross return
        state_space: int
            the number of stocks (the covariance matrix is state_space wide)
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            unused; kept for signature compatibility
        day: int
            an increment number to control date

    The observation is the stock covariance matrix stacked on top of the
    indicator matrix: shape ``(state_space + len(tech_indicator_list),
    state_space)``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
        *,
        reward_type="asset_change",
        make_plots=False,
        results_dir=config.RESULTS_DIR,
    ):
        self.day = day
        self.start_day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        if reward_type not in ("asset_change", "log_return"):
            raise ValueError("reward_type must be 'asset_change' or 'log_return'")
        self.reward_type = reward_type
        self.make_plots = make_plots
        self.results_dir = results_dir

        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
            dtype=np.float32,
        )
        self.n_days = len(self.df.index.unique())
        self._reset_memory()

    def _load_day(self):
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )

    def _reset_memory(self):
        self.day = self.start_day
        self._load_day()
        self.terminal = False
        self.reward = 0.0
        self.portfolio_value = self.initial_amount
        self.cost = 0.0
        self.weights = np.full(self.stock_dim, 1.0 / self.stock_dim)
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [self.weights.copy()]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        if self.terminal:
            raise RuntimeError("Episode is over; call reset() before step()")
        weights = self.softmax_normalization(np.asarray(actions, dtype=np.float64))
        # Weights drift with prices between rebalances, so rebalancing costs
        # are charged on the change from the drifted weights.
        turnover = np.abs(weights - self.weights).sum()
        cost = self.portfolio_value * turnover * self.transaction_cost_pct
        self.cost += cost
        self.actions_memory.append(weights)
        last_day_memory = self.data

        self.day += 1
        self._load_day()
        relative = self.data.close.values / last_day_memory.close.values
        portfolio_return = float(np.dot(relative - 1, weights))
        begin_value = self.portfolio_value
        self.portfolio_value = (begin_value - cost) * (1 + portfolio_return)
        drifted = weights * relative
        self.weights = drifted / drifted.sum()

        self.portfolio_return_memory.append(self.portfolio_value / begin_value - 1)
        self.date_memory.append(self.data.date.unique()[0])
        self.asset_memory.append(self.portfolio_value)
        if self.reward_type == "log_return":
            self.reward = float(np.log(self.portfolio_value / begin_value)) * self.reward_scaling
        else:
            self.reward = (self.portfolio_value - begin_value) * self.reward_scaling

        self.terminal = self.day >= self.n_days - 1
        info = {}
        if self.terminal:
            info["episode_stats"] = self._finish_episode()
        return self.state.astype(np.float32), self.reward, self.terminal, False, info

    def _finish_episode(self):
        returns = pd.Series(self.portfolio_return_memory[1:], dtype=float)
        std = returns.std()
        sharpe = float(np.sqrt(252) * returns.mean() / std) if std > 0 else float("nan")
        logger.info(
            "begin_total_asset: %.2f, end_total_asset: %.2f, sharpe: %.3f",
            self.asset_memory[0],
            self.portfolio_value,
            sharpe,
        )
        if self.make_plots:
            from matplotlib.figure import Figure

            os.makedirs(self.results_dir, exist_ok=True)
            fig = Figure()
            fig.subplots().plot(np.cumsum(self.portfolio_return_memory), "r")
            fig.savefig(os.path.join(self.results_dir, "cumulative_reward.png"))
        return {
            "begin_total_asset": float(self.asset_memory[0]),
            "end_total_asset": float(self.portfolio_value),
            "total_cost": float(self.cost),
            "sharpe": sharpe,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_memory()
        return self.state.astype(np.float32), {}

    def render(self):
        return self.state

    def softmax_normalization(self, actions):
        exp = np.exp(actions - np.max(actions))
        return exp / exp.sum()

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame(
            {
                "date": date_list,
                "daily_return": portfolio_return,
                "account_value": self.asset_memory,
            }
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        df_actions = pd.DataFrame(np.asarray(self.actions_memory), columns=self.data.tic.values)
        df_actions.index = pd.Index(self.date_memory, name="date")
        return df_actions

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
