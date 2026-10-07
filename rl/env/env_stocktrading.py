"""Multi-stock trading environment with the gymnasium API.

The agent holds cash and whole shares of ``stock_dim`` stocks. At each step it
outputs one number in [-1, 1] per stock, which is scaled by ``hmax`` into a
number of shares to sell (negative) or buy (positive) at the day's close.

State vector (kept identical to the original FinRL layout so saved
``previous_state`` lists stay valid)::

    [cash, close_1..close_N, holdings_1..holdings_N,
     tech_1(stock_1..stock_N), ..., tech_K(stock_1..stock_N)]
"""

import logging
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.config import config

logger = logging.getLogger(__name__)

REWARD_TYPES = ("asset_change", "log_return")


def panel_arrays(df, columns):
    """Reshape a (day, tic) long frame into per-column ``(days, tickers)`` arrays.

    ``df`` must be indexed by an integer day number (as produced by
    :func:`rl.preprocessing.data.data_split`) and contain every ticker on every
    day. Returns ``(dates, tickers, {column: array})``.
    """
    if "tic" not in df.columns:
        raise ValueError("Data must contain a 'tic' column")
    day_index = np.asarray(df.index)
    order = np.lexsort((df["tic"].astype(str).to_numpy(), day_index))
    frame = df.iloc[order]
    days, counts = np.unique(day_index, return_counts=True)
    n_days, n_tickers = len(days), int(counts[0]) if len(counts) else 0
    if n_days == 0:
        raise ValueError("Data is empty")
    if not (counts == n_tickers).all():
        raise ValueError(
            "Every day must contain the same tickers; found between {} and {} rows "
            "per day. Run FeatureEngineer with clean_data=True (or "
            "rl.preprocessing.preprocessors.balance_panel) first.".format(
                counts.min(), counts.max()
            )
        )
    tics = frame["tic"].astype(str).to_numpy().reshape(n_days, n_tickers)
    if not (tics == tics[0]).all():
        raise ValueError("Every day must contain the same tickers")
    arrays = {}
    for column in columns:
        if column not in frame.columns:
            raise ValueError("Data is missing column '{}'".format(column))
        values = frame[column].to_numpy(dtype=np.float64).reshape(n_days, n_tickers)
        if np.isnan(values).any():
            raise ValueError("Column '{}' contains missing values".format(column))
        arrays[column] = values
    dates = frame["date"].to_numpy().reshape(n_days, n_tickers)[:, 0]
    return dates, list(tics[0]), arrays


class StockTradingEnv(gym.Env):
    """A multi-stock trading environment.

    Parameters keep the names and order of the original FinRL environment.
    New keyword-only options:

    reward_type : ``"asset_change"`` (default, change in total assets as in
        FinRL) or ``"log_return"`` (log of the total-asset growth factor).
        Either is multiplied by ``reward_scaling``.
    results_dir : where per-episode CSV/PNG files go when ``model_name`` and
        ``mode`` are set, or when ``make_plots`` is true.

    An episode starts on ``day`` and ends when the last day of ``df`` is
    reached, so it lasts ``len(days) - 1 - day`` steps. With
    ``turbulence_threshold`` set, every position is liquidated (and buying is
    blocked) on days whose turbulence index is at or above the threshold.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=None,
        model_name="",
        mode="",
        iteration="",
        *,
        reward_type="asset_change",
        results_dir=config.RESULTS_DIR,
    ):
        if reward_type not in REWARD_TYPES:
            raise ValueError(
                "reward_type must be one of {}, got {!r}".format(REWARD_TYPES, reward_type)
            )
        self.df = df
        self.stock_dim = int(stock_dim)
        self.hmax = hmax
        self.initial_amount = float(initial_amount)
        self.buy_cost_pct = float(buy_cost_pct)
        self.sell_cost_pct = float(sell_cost_pct)
        self.reward_scaling = float(reward_scaling)
        self.state_space = int(state_space)
        self.tech_indicator_list = list(tech_indicator_list)
        self.turbulence_threshold = turbulence_threshold
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.start_day = int(day)
        self.initial = initial
        self.previous_state = list(previous_state) if previous_state is not None else []
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.reward_type = reward_type
        self.results_dir = results_dir

        columns = ["close"] + self.tech_indicator_list
        if turbulence_threshold is not None:
            columns.append("turbulence")
        self.dates, self.tickers, arrays = panel_arrays(df, columns)
        if len(self.tickers) != self.stock_dim:
            raise ValueError(
                "stock_dim is {} but the data has {} tickers".format(
                    self.stock_dim, len(self.tickers)
                )
            )
        expected_state = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
        if self.state_space != expected_state:
            raise ValueError(
                "state_space is {} but 1 + 2 * stock_dim + len(tech_indicator_list) "
                "* stock_dim is {}".format(self.state_space, expected_state)
            )
        if int(action_space) != self.stock_dim:
            raise ValueError("action_space must equal stock_dim")
        if not 0 <= self.start_day < len(self.dates) - 1:
            raise ValueError("Need at least two days of data after the start day")
        if not self.initial and len(self.previous_state) != self.state_space:
            raise ValueError("previous_state must have state_space entries when initial=False")
        self.close_prices = arrays["close"]
        # Tech-major layout: all stocks for indicator 1, then indicator 2, ...
        if self.tech_indicator_list:
            self.tech_features = np.concatenate(
                [arrays[tech] for tech in self.tech_indicator_list], axis=1
            )
        else:
            self.tech_features = np.zeros((len(self.dates), 0))
        self.turbulence_index = (
            arrays["turbulence"][:, 0] if turbulence_threshold is not None else None
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )
        self.episode = 0
        self._reset_account()

    # ------------------------------------------------------------------
    # Account bookkeeping
    # ------------------------------------------------------------------
    def _reset_account(self):
        self.day = self.start_day
        if self.initial:
            self.cash = self.initial_amount
            self.holdings = np.zeros(self.stock_dim)
        else:
            n = self.stock_dim
            self.cash = float(self.previous_state[0])
            self.holdings = np.asarray(self.previous_state[n + 1 : 2 * n + 1], dtype=np.float64)
        self.turbulence = self._turbulence_on(self.day)
        self.cost = 0.0
        self.trades = 0
        self.reward = 0.0
        self.terminal = False
        self.state = self._build_state()
        self.asset_memory = [self.total_asset]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.dates[self.day]]

    def _turbulence_on(self, day):
        if self.turbulence_index is None:
            return 0.0
        return float(self.turbulence_index[day])

    @property
    def prices(self):
        return self.close_prices[self.day]

    @property
    def total_asset(self):
        return float(self.cash + np.dot(self.holdings, self.prices))

    def _build_state(self):
        return np.concatenate(
            ([self.cash], self.prices, self.holdings, self.tech_features[self.day])
        )

    def _observation(self):
        return self.state.astype(np.float32)

    def _turbulent(self):
        return (
            self.turbulence_threshold is not None
            and self.turbulence >= self.turbulence_threshold
        )

    def _sell(self, index, shares):
        """Sell up to ``shares`` of stock ``index``; return shares sold."""
        price = self.prices[index]
        if price <= 0 or self.holdings[index] <= 0:
            return 0
        shares = min(shares, self.holdings[index])
        gross = price * shares
        self.cash += gross * (1 - self.sell_cost_pct)
        self.cost += gross * self.sell_cost_pct
        self.holdings[index] -= shares
        self.trades += 1
        return shares

    def _buy(self, index, shares):
        """Buy up to ``shares`` of stock ``index`` with cash on hand."""
        price = self.prices[index]
        if price <= 0:
            return 0
        # Costs are paid in cash too, so they limit how much can be bought.
        affordable = np.floor(self.cash / (price * (1 + self.buy_cost_pct)))
        shares = min(shares, affordable)
        if shares <= 0:
            return 0
        gross = price * shares
        self.cash -= gross * (1 + self.buy_cost_pct)
        self.cost += gross * self.buy_cost_pct
        self.holdings[index] += shares
        self.trades += 1
        return shares

    def _execute(self, actions):
        """Turn raw actions into trades and return the signed shares traded."""
        actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(self.stock_dim), -1, 1)
        # Whole shares only; truncate towards zero as the original env did.
        target = np.trunc(actions * self.hmax)
        if self._turbulent():
            # Market is in turmoil: liquidate everything and do not buy.
            target = -self.holdings.copy()
        executed = np.zeros(self.stock_dim)
        for index in np.where(target < 0)[0]:
            executed[index] = -self._sell(index, -target[index])
        buys = np.where(target > 0)[0]
        # Largest orders first, as in the original environment.
        for index in buys[np.argsort(-target[buys], kind="stable")]:
            executed[index] = self._buy(index, target[index])
        return executed

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------
    def step(self, actions):
        if self.terminal:
            raise RuntimeError("Episode is over; call reset() before step()")
        begin_total_asset = self.total_asset
        executed = self._execute(actions)
        self.actions_memory.append(executed)

        self.day += 1
        self.turbulence = self._turbulence_on(self.day)
        self.state = self._build_state()
        end_total_asset = self.total_asset
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self.dates[self.day])

        if self.reward_type == "log_return":
            if begin_total_asset > 0 and end_total_asset > 0:
                raw_reward = float(np.log(end_total_asset / begin_total_asset))
            else:
                raw_reward = 0.0
        else:
            raw_reward = end_total_asset - begin_total_asset
        self.rewards_memory.append(raw_reward)
        self.reward = raw_reward * self.reward_scaling

        self.terminal = self.day >= len(self.dates) - 1
        info = {}
        if self.terminal:
            info["episode_stats"] = self._finish_episode()
        return self._observation(), self.reward, self.terminal, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self._reset_account()
        return self._observation(), {}

    def render(self):
        return self.state

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Episode summaries
    # ------------------------------------------------------------------
    def _finish_episode(self):
        values = pd.Series(self.asset_memory, dtype=float)
        daily_return = values.pct_change().dropna()
        std = daily_return.std()
        sharpe = float(np.sqrt(252) * daily_return.mean() / std) if std > 0 else float("nan")
        stats = {
            "begin_total_asset": float(self.asset_memory[0]),
            "end_total_asset": float(self.asset_memory[-1]),
            "total_reward": float(self.asset_memory[-1] - self.asset_memory[0]),
            "total_cost": float(self.cost),
            "total_trades": int(self.trades),
            "sharpe": sharpe,
        }
        if self.print_verbosity and self.episode % self.print_verbosity == 0:
            logger.info(
                "day: %d, episode: %d, begin_total_asset: %.2f, end_total_asset: %.2f, "
                "total_reward: %.2f, total_cost: %.2f, total_trades: %d, sharpe: %.3f",
                self.day,
                self.episode,
                stats["begin_total_asset"],
                stats["end_total_asset"],
                stats["total_reward"],
                stats["total_cost"],
                stats["total_trades"],
                sharpe,
            )
        if self.model_name and self.mode:
            self._save_episode_files()
        if self.make_plots:
            self._plot("account_value_trade_{}.png".format(self.episode))
        return stats

    def _save_episode_files(self):
        os.makedirs(self.results_dir, exist_ok=True)
        suffix = "{}_{}_{}".format(self.mode, self.model_name, self.iteration)
        df_total_value = self.save_asset_memory()
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change()
        df_total_value.to_csv(
            os.path.join(self.results_dir, "account_value_{}.csv".format(suffix)), index=False
        )
        self.save_action_memory().to_csv(
            os.path.join(self.results_dir, "actions_{}.csv".format(suffix))
        )
        pd.DataFrame(
            {"date": self.date_memory[:-1], "account_rewards": self.rewards_memory}
        ).to_csv(
            os.path.join(self.results_dir, "account_rewards_{}.csv".format(suffix)),
            index=False,
        )
        self._plot("account_value_{}.png".format(suffix))

    def _plot(self, file_name):
        from matplotlib.figure import Figure

        os.makedirs(self.results_dir, exist_ok=True)
        fig = Figure()
        fig.subplots().plot(self.asset_memory, "r")
        fig.savefig(os.path.join(self.results_dir, file_name))

    def save_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory, "account_value": self.asset_memory})

    def save_action_memory(self):
        """Signed shares traded on each day, one column per ticker."""
        df_actions = pd.DataFrame(
            np.asarray(self.actions_memory).reshape(-1, self.stock_dim),
            columns=self.tickers,
        )
        df_actions.index = pd.Index(self.date_memory[:-1], name="date")
        return df_actions

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
