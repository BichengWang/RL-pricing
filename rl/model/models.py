import copy
import importlib.util
import logging
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.config import config
from rl.env.env_stocktrading import StockTradingEnv
from rl.preprocessing.data import data_split
from rl.trade import metrics

logger = logging.getLogger(__name__)

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: getattr(config, f"{x.upper()}_PARAMS") for x in MODELS}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


def _tensorboard_log(model_name):
    # SB3 refuses a log directory when tensorboard is not installed.
    if importlib.util.find_spec("tensorboard") is None:
        return None
    return f"{config.TENSORBOARD_LOG_DIR}/{model_name}"


def build_model(
    model_name,
    env,
    policy="MlpPolicy",
    policy_kwargs=None,
    model_kwargs=None,
    verbose=1,
    seed=None,
):
    """Create a stable-baselines3 model with this project's defaults.

    ``model_kwargs`` defaults to the ``<NAME>_PARAMS`` dict in config.py and is
    copied, so neither it nor the config is modified. An ``action_noise`` entry
    may name a noise type ("normal" or "ornstein_uhlenbeck").
    """
    if model_name not in MODELS:
        raise NotImplementedError(
            "Unknown model {!r}; choose from {}".format(model_name, ", ".join(MODELS))
        )
    kwargs = copy.deepcopy(MODEL_KWARGS[model_name] if model_kwargs is None else model_kwargs)
    if isinstance(kwargs.get("action_noise"), str):
        n_actions = env.action_space.shape[-1]
        kwargs["action_noise"] = NOISE[kwargs["action_noise"]](
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
    logger.info("Building %s with %s", model_name.upper(), kwargs)
    return MODELS[model_name](
        policy=policy,
        env=env,
        tensorboard_log=_tensorboard_log(model_name),
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        seed=seed,
        **kwargs,
    )


def make_vec_env(env, normalize_obs=False, seed=None):
    """Wrap an environment instance for stable-baselines3.

    With ``normalize_obs`` the observations are standardised with running
    statistics (``VecNormalize``); cash, prices, share counts and indicators
    otherwise differ by several orders of magnitude.
    """
    venv = DummyVecEnv([lambda: env])
    if seed is not None:
        venv.seed(seed)
    if normalize_obs:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


class EpisodeStatsCallback(BaseCallback):
    """Record the environment's end-of-episode statistics in the SB3 logger."""

    def _on_step(self):
        for info in self.locals.get("infos", []):
            for key, value in info.get("episode_stats", {}).items():
                self.logger.record(f"environment/{key}", value)
        return True


def predict_episode(model, environment, deterministic=True, obs_normalizer="auto"):
    """Run ``model`` through one full episode of ``environment``.

    ``obs_normalizer`` is a ``VecNormalize`` whose statistics are applied to
    observations; by default the one the model was trained with (if any).
    Returns the environment's account value and action frames.
    """
    if isinstance(obs_normalizer, str) and obs_normalizer == "auto":
        obs_normalizer = model.get_vec_normalize_env()
    obs, _ = environment.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        model_obs = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
        action, _ = model.predict(model_obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = environment.step(action)
    return environment.save_asset_memory(), environment.save_action_memory()


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            build an A2C, PPO, DDPG, TD3 or SAC model
        train_model()
            train a model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True, obs_normalizer="auto"):
        """Trade one episode of ``environment`` (an unwrapped env instance)."""
        return predict_episode(
            model, environment, deterministic=deterministic, obs_normalizer=obs_normalizer
        )

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
    ):
        return build_model(
            model_name,
            self.env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
            seed=seed,
        )

    def train_model(self, model, tb_log_name, total_timesteps=5000, callback=None):
        callbacks = [EpisodeStatsCallback()] + ([callback] if callback is not None else [])
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=CallbackList(callbacks),
        )
        return model


class DRLEnsembleAgent:
    """Rolling-window ensemble of A2C, PPO and DDPG.

    Every ``rebalance_window`` trading days, each algorithm is trained on all
    data before a validation window, scored by its validation Sharpe ratio,
    and the winner is retrained up to the trading window and used to trade
    it. Positions carry over from one trading window to the next.
    """

    # Ties are broken in this order, as in the original implementation.
    MODEL_NAMES = ("ppo", "a2c", "ddpg")

    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
    ):
        return build_model(
            model_name,
            env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
            seed=seed,
        )

    @staticmethod
    def train_model(model, model_name, tb_log_name, iter_num, total_timesteps=5000, save=True):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=EpisodeStatsCallback(),
        )
        if save:
            os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
            model.save(
                f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}"
            )
        return model

    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """Annualised Sharpe ratio of a saved validation run."""
        df_total_value = pd.read_csv(
            f"{config.RESULTS_DIR}/account_value_validation_{model_name}_{iteration}.csv"
        )
        return metrics.sharpe_ratio(df_total_value["daily_return"])

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
        *,
        normalize_obs=False,
        reward_type="asset_change",
        seed=None,
        save_models=True,
        results_dir=config.RESULTS_DIR,
        verbose=1,
    ):
        if "turbulence" not in df.columns:
            raise ValueError("The ensemble strategy needs a 'turbulence' column")
        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.normalize_obs = normalize_obs
        self.reward_type = reward_type
        self.seed = seed
        self.save_models = save_models
        self.results_dir = results_dir
        self.verbose = verbose
        self.account_value = None

    def _env(self, data, **kwargs):
        return StockTradingEnv(
            data,
            self.stock_dim,
            self.hmax,
            self.initial_amount,
            self.buy_cost_pct,
            self.sell_cost_pct,
            self.reward_scaling,
            self.state_space,
            self.action_space,
            self.tech_indicator_list,
            print_verbosity=self.print_verbosity,
            reward_type=self.reward_type,
            results_dir=self.results_dir,
            **kwargs,
        )

    def _train(self, model_name, data, model_kwargs, total_timesteps, tb_log_name, iter_num):
        venv = make_vec_env(self._env(data), normalize_obs=self.normalize_obs, seed=self.seed)
        model = self.get_model(
            model_name, venv, model_kwargs=model_kwargs, verbose=self.verbose, seed=self.seed
        )
        return self.train_model(
            model,
            model_name,
            tb_log_name=tb_log_name,
            iter_num=iter_num,
            total_timesteps=total_timesteps,
            save=self.save_models,
        )

    def DRL_validation(self, model, test_data, turbulence_threshold, iteration, model_name):
        """Trade the validation window and return its annualised Sharpe ratio."""
        env = self._env(
            test_data,
            turbulence_threshold=turbulence_threshold,
            iteration=iteration,
            model_name=model_name.upper(),
            mode="validation",
        )
        account_value, _ = predict_episode(model, env)
        return metrics.sharpe_ratio(account_value["account_value"].pct_change())

    def DRL_prediction(self, model, name, last_state, iter_num, turbulence_threshold, initial):
        """Trade one rebalance window, continuing from ``last_state``.

        Returns the final state and the window's account values.
        """
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        trade_env = self._env(
            trade_data,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
            previous_state=last_state,
            model_name=name,
            mode="trade",
            iteration=iter_num,
        )
        account_value, _ = predict_episode(model, trade_env)
        last_state = list(trade_env.render())
        os.makedirs(self.results_dir, exist_ok=True)
        pd.DataFrame({"last_state": last_state}).to_csv(
            os.path.join(self.results_dir, "last_state_{}_{}.csv".format(name, iter_num)),
            index=False,
        )
        return last_state, account_value

    def run_ensemble_strategy(
        self, A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict
    ):
        """Ensemble Strategy that combines PPO, A2C and DDPG"""
        logger.info("============Start Ensemble Strategy============")
        model_kwargs = {"a2c": A2C_model_kwargs, "ppo": PPO_model_kwargs, "ddpg": DDPG_model_kwargs}
        # The trading env is fed the last state of the previous window so
        # positions carry over between windows.
        last_state_ensemble = []
        sharpe_lists = {name: [] for name in self.MODEL_NAMES}
        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []
        account_values = []

        daily_turbulence = (
            self.df.drop_duplicates(subset=["date"]).set_index("date")["turbulence"].sort_index()
        )
        insample_turbulence = daily_turbulence[
            (daily_turbulence.index >= self.train_period[0])
            & (daily_turbulence.index < self.train_period[1])
        ].to_numpy()
        insample_turbulence_threshold = np.quantile(insample_turbulence, 0.90)

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]
            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)
            initial = i - self.rebalance_window - self.validation_window == 0

            # Tune the turbulence threshold on the quarter (63 trading days)
            # before the validation window.
            historical_turbulence_mean = daily_turbulence[
                daily_turbulence.index <= validation_start_date
            ].iloc[-63:].mean()
            if historical_turbulence_mean > insample_turbulence_threshold:
                # Volatile market: liquidate above the in-sample 90% quantile.
                turbulence_threshold = insample_turbulence_threshold
            else:
                # Calm market: only liquidate above the in-sample maximum.
                turbulence_threshold = np.quantile(insample_turbulence, 1)
            if not turbulence_threshold > 0:
                # No turbulence history yet; a zero threshold would sell every day.
                turbulence_threshold = None
            logger.info("turbulence_threshold: %s", turbulence_threshold)

            train = data_split(self.df, start=self.train_period[0], end=validation_start_date)
            validation = data_split(self.df, start=validation_start_date, end=validation_end_date)

            logger.info(
                "======Model training from %s to %s", self.train_period[0], validation_start_date
            )
            sharpes = {}
            for name in self.MODEL_NAMES:
                model = self._train(
                    name, train, model_kwargs[name], timesteps_dict[name], f"{name}_{i}", i
                )
                sharpes[name] = self.DRL_validation(
                    model, validation, turbulence_threshold, i, name
                )
                sharpe_lists[name].append(sharpes[name])
                logger.info(
                    "%s validation Sharpe (%s to %s): %.3f",
                    name.upper(),
                    validation_start_date,
                    validation_end_date,
                    sharpes[name],
                )

            best = max(
                self.MODEL_NAMES,
                key=lambda n: -np.inf if np.isnan(sharpes[n]) else sharpes[n],
            )
            model_use.append(best.upper())
            logger.info(
                "======Retraining %s from %s to %s",
                best.upper(),
                self.train_period[0],
                validation_end_date,
            )
            train_full = data_split(self.df, start=self.train_period[0], end=validation_end_date)
            model_ensemble = self._train(
                best, train_full, model_kwargs[best], timesteps_dict[best], f"ensemble_{i}", i
            )

            logger.info(
                "======Trading from %s to %s",
                validation_end_date,
                self.unique_trade_date[i],
            )
            last_state_ensemble, window_values = self.DRL_prediction(
                model=model_ensemble,
                name="ensemble",
                last_state=last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            )
            account_values.append(window_values)

        logger.info("Ensemble Strategy took %.1f minutes", (time.time() - start) / 60)
        if account_values:
            self.account_value = pd.concat(account_values, ignore_index=True)

        df_summary = pd.DataFrame(
            {
                "Iter": iteration_list,
                "Val Start": validation_start_date_list,
                "Val End": validation_end_date_list,
                "Model Used": model_use,
                "A2C Sharpe": sharpe_lists["a2c"],
                "PPO Sharpe": sharpe_lists["ppo"],
                "DDPG Sharpe": sharpe_lists["ddpg"],
            }
        )
        return df_summary
