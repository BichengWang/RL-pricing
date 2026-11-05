import logging
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl.config import config
from rl.env.env_stocktrading import panel_arrays
from rl.trade.execution import rebalance_positions

logger = logging.getLogger(__name__)


class StockPortfolioEnv(gym.Env):
    """A portfolio allocation environment with the gymnasium API.

    Each step the agent outputs one score per stock; a softmax turns the
    scores into long-only target weights. Signals use the current close,
    trades fill at the next open, and holdings are marked at that day's close.

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
        # Look up each day's prices and observation in arrays instead of
        # slicing the frame on every step.
        self.dates, self.tickers, arrays = panel_arrays(
            df, ["open", "close"] + list(tech_indicator_list)
        )
        if len(self.tickers) != self.stock_dim:
            raise ValueError(
                "stock_dim is {} but the data has {} tickers".format(
                    self.stock_dim, len(self.tickers)
                )
            )
        if "cov_list" not in df.columns:
            raise ValueError("Data is missing column 'cov_list' (see add_covariance_matrix)")
        self.n_days = len(self.dates)
        if not 0 <= self.start_day < self.n_days - 1:
            raise ValueError("Need at least two days of data after the start day")
        self.open_prices = arrays["open"]
        self.close_prices = arrays["close"]
        covs = np.stack(
            [np.asarray(c, dtype=np.float64) for c in df["cov_list"].groupby(df.index).first()]
        )
        tech = [arrays[name][:, np.newaxis, :] for name in self.tech_indicator_list]
        # (days, stocks + indicators, stocks)
        self.observations = np.concatenate([covs] + tech, axis=1)
        self._reset_memory()

    @property
    def data(self):
        """The rows of ``df`` for the current day."""
        return self.df.loc[self.day, :]

    def _load_day(self):
        self.covs = self.observations[self.day, : self.stock_dim]
        self.state = self.observations[self.day]

    def _reset_memory(self):
        self.day = self.start_day
        self._load_day()
        self.terminal = False
        self.reward = 0.0
        self.portfolio_value = self.initial_amount
        self.cost = 0.0
        self.cash = float(self.initial_amount)
        self.holdings = np.zeros(self.stock_dim)
        self.weights = np.zeros(self.stock_dim)
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        # Target weights remain indexed by the signal close, not the fill date.
        self.actions_memory = []
        self.date_memory = [self.dates[self.day]]

    def step(self, actions):
        if self.terminal:
            raise RuntimeError("Episode is over; call reset() before step()")
        weights = self.softmax_normalization(np.asarray(actions, dtype=np.float64))
        begin_value = self.portfolio_value
        fill_day = self.day + 1
        fill_prices = self.open_prices[fill_day]
        # Existing shares earn the overnight gap before rebalancing. Fees are
        # paid from equity and targets are fractions of the post-fee account.
        positions, self.cash, cost = rebalance_positions(
            self.cash,
            self.holdings * fill_prices,
            weights,
            self.transaction_cost_pct,
        )
        self.holdings = positions / fill_prices
        self.cost += cost
        self.actions_memory.append(weights)

        self.day = fill_day
        self._load_day()
        closing_positions = self.holdings * self.close_prices[self.day]
        self.portfolio_value = float(self.cash + closing_positions.sum())
        self.weights = closing_positions / self.portfolio_value

        self.portfolio_return_memory.append(self.portfolio_value / begin_value - 1)
        self.date_memory.append(self.dates[self.day])
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
        """Target portfolio weights indexed by their signal close."""
        df_actions = pd.DataFrame(
            np.asarray(self.actions_memory).reshape(-1, self.stock_dim),
            columns=self.tickers,
        )
        df_actions.index = pd.Index(self.date_memory[:-1], name="date")
        return df_actions

    def save_execution_memory(self):
        """Signal/fill dates and target weights, one column per ticker."""
        executions = self.save_action_memory().reset_index().rename(columns={"date": "signal_date"})
        executions.insert(1, "fill_date", self.date_memory[1:])
        return executions
