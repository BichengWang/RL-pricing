"""End-to-end pipelines: load data, engineer features, train, trade and
benchmark against rule-based baselines.

Each run writes to ``<trained_model_dir>/<run_name>/`` (model, observation
normalisation statistics and ``run_config.json``) and
``<results_dir>/<run_name>/`` (account values, trades, statistics and a plot).
"""

import dataclasses
import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from rl.config import config

logger = logging.getLogger(__name__)

DATA_SOURCES = ("yahoo", "csv", "synthetic")
AGENTS = ("a2c", "ppo", "ddpg", "td3", "sac")
# trading: buy and sell whole shares (StockTradingEnv);
# portfolio: choose long-only portfolio weights each day (StockPortfolioEnv).
TASKS = ("trading", "portfolio")


@dataclass
class TrainConfig:
    """Settings for one training / backtest run.

    ``ticker_list`` defaults to the Dow 30 for the Yahoo and synthetic data
    sources and to every ticker in the file for the CSV source.

    With ``task="portfolio"`` the agent allocates the portfolio across the
    tickers instead of trading share counts; ``hmax`` and the turbulence
    settings do not apply, and the first ``cov_lookback`` days of data are
    used to warm up the covariance features.
    """

    task: str = "trading"
    agent: str = "sac"
    total_timesteps: int = 80_000
    data_source: str = "yahoo"
    data_file: Optional[str] = None
    ticker_list: Optional[List[str]] = None
    start_date: str = config.START_DATE
    start_trade_date: str = config.START_TRADE_DATE
    end_date: str = config.END_DATE
    tech_indicator_list: List[str] = field(
        default_factory=lambda: list(config.TECHNICAL_INDICATORS_LIST)
    )
    use_turbulence: bool = True
    # A fixed threshold wins over the quantile of in-sample turbulence.
    turbulence_threshold: Optional[float] = None
    turbulence_quantile: Optional[float] = 0.99
    hmax: int = 100
    initial_amount: float = 1_000_000
    buy_cost_pct: float = 0.001
    sell_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    reward_type: str = "asset_change"
    # portfolio task: days of returns in each covariance matrix
    cov_lookback: int = 252
    normalize_observations: bool = True
    # Hold out the last ``validation_days`` training days; every ``eval_freq``
    # steps the agent trades them and the best checkpoint (by Sharpe ratio)
    # is kept instead of the final one. 0 trains on everything.
    validation_days: int = 0
    eval_freq: int = 10_000
    model_kwargs: Optional[dict] = None
    seed: Optional[int] = None
    benchmark_ticker: Optional[str] = None
    run_name: Optional[str] = None
    results_dir: str = config.RESULTS_DIR
    trained_model_dir: str = config.TRAINED_MODEL_DIR
    # ensemble strategy
    rebalance_window: int = 63
    validation_window: int = 63
    verbose: int = 0

    def validate(self):
        if self.task not in TASKS:
            raise ValueError("task must be one of {}".format(TASKS))
        if self.task == "portfolio" and self.agent == "ensemble":
            raise ValueError("The ensemble strategy supports only the trading task")
        if self.cov_lookback < 2:
            raise ValueError("cov_lookback must be at least 2")
        if self.data_source not in DATA_SOURCES:
            raise ValueError("data_source must be one of {}".format(DATA_SOURCES))
        if self.data_source == "csv" and not self.data_file:
            raise ValueError("data_file is required for the csv data source")
        if self.agent not in AGENTS + ("ensemble",):
            raise ValueError("agent must be one of {}".format(AGENTS + ("ensemble",)))
        if not self.start_date < self.start_trade_date < self.end_date:
            raise ValueError("Need start_date < start_trade_date < end_date")
        if self.turbulence_quantile is not None and not 0 < self.turbulence_quantile <= 1:
            raise ValueError("turbulence_quantile must be in (0, 1]")
        if self.validation_days < 0:
            raise ValueError("validation_days must not be negative")
        if self.eval_freq < 1:
            raise ValueError("eval_freq must be at least 1")
        if self.validation_days and self.agent == "ensemble":
            raise ValueError("The ensemble strategy has its own validation_window")
        return self

    def tickers(self):
        if self.ticker_list:
            return list(self.ticker_list)
        if self.data_source == "csv":
            return None
        return list(config.DOW_30_TICKER)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_market_data(cfg):
    """Raw daily OHLCV data for the configured source, tickers and dates."""
    tickers = cfg.tickers()
    if cfg.data_source == "yahoo":
        from rl.marketdata.yahoodownloader import YahooDownloader

        df = YahooDownloader(cfg.start_date, cfg.end_date, tickers).fetch_data()
    elif cfg.data_source == "synthetic":
        from rl.marketdata.synthetic import generate_synthetic_data

        df = generate_synthetic_data(
            tickers, cfg.start_date, cfg.end_date, seed=0 if cfg.seed is None else cfg.seed
        )
    else:
        from rl.preprocessing.data import load_dataset

        df = load_dataset(file_name=cfg.data_file)
        df = df[(df["date"] >= cfg.start_date) & (df["date"] < cfg.end_date)]
        if tickers is not None:
            df = df[df["tic"].isin(tickers)]
    if df.empty:
        raise ValueError("No market data for the requested tickers and dates")
    return df.reset_index(drop=True)


def prepare_data(cfg, tickers=None):
    """Return ``(processed, train, trade)`` frames for ``cfg``.

    With ``tickers`` given, the processed data must contain exactly those
    tickers (used when reloading a trained model).
    """
    from rl.preprocessing.data import data_split
    from rl.preprocessing.preprocessors import FeatureEngineer, add_covariance_matrix

    raw = load_market_data(cfg)
    if tickers is not None:
        raw = raw[raw["tic"].isin(tickers)]
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=cfg.tech_indicator_list,
        use_turbulence=cfg.use_turbulence and cfg.task == "trading",
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(raw)
    if cfg.task == "portfolio":
        processed = add_covariance_matrix(processed, lookback=cfg.cov_lookback)
    if tickers is not None:
        found = sorted(processed["tic"].unique())
        if found != sorted(tickers):
            raise ValueError(
                "The model was trained on {} but the data has {}".format(sorted(tickers), found)
            )
    train = data_split(processed, cfg.start_date, cfg.start_trade_date)
    trade = data_split(processed, cfg.start_trade_date, cfg.end_date)
    if train.empty or trade.index.nunique() < 2:
        hint = ""
        if cfg.task == "portfolio":
            hint = " (the portfolio task uses the first {} days for covariances)".format(
                cfg.cov_lookback
            )
        raise ValueError("Not enough data on one side of start_trade_date" + hint)
    return processed, train, trade


def env_kwargs(cfg, stock_dim):
    if cfg.task == "portfolio":
        return {
            "hmax": cfg.hmax,
            "initial_amount": cfg.initial_amount,
            # Rebalancing buys and sells in equal measure.
            "transaction_cost_pct": (cfg.buy_cost_pct + cfg.sell_cost_pct) / 2,
            "reward_scaling": cfg.reward_scaling,
            "reward_type": cfg.reward_type,
            "state_space": stock_dim,
            "stock_dim": stock_dim,
            "action_space": stock_dim,
            "tech_indicator_list": cfg.tech_indicator_list,
            "lookback": cfg.cov_lookback,
        }
    return {
        "hmax": cfg.hmax,
        "initial_amount": cfg.initial_amount,
        "buy_cost_pct": cfg.buy_cost_pct,
        "sell_cost_pct": cfg.sell_cost_pct,
        "state_space": 1 + 2 * stock_dim + len(cfg.tech_indicator_list) * stock_dim,
        "stock_dim": stock_dim,
        "tech_indicator_list": cfg.tech_indicator_list,
        "action_space": stock_dim,
        "reward_scaling": cfg.reward_scaling,
        "reward_type": cfg.reward_type,
    }


def make_env(cfg, df, kwargs, turbulence_threshold=None, results_dir=config.RESULTS_DIR):
    """The environment for ``cfg.task`` over ``df``."""
    if cfg.task == "portfolio":
        from rl.env.env_portfolio import StockPortfolioEnv

        return StockPortfolioEnv(df=df, results_dir=results_dir, **kwargs)
    from rl.env.env_stocktrading import StockTradingEnv

    return StockTradingEnv(
        df=df, turbulence_threshold=turbulence_threshold, results_dir=results_dir, **kwargs
    )


def split_validation(train, validation_days):
    """Split the training frame into ``(fit, validation)`` by date.

    The validation part is the last ``validation_days`` trading days, so the
    agent is selected on data that follows everything it was trained on.
    """
    from rl.preprocessing.data import data_split

    dates = sorted(train["date"].unique())
    if validation_days < 2 or validation_days > len(dates) - 2:
        raise ValueError(
            "validation_days must be between 2 and {} (the training period has {} days)".format(
                len(dates) - 2, len(dates)
            )
        )
    split_date = dates[-validation_days]
    end = pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
    return (
        data_split(train, dates[0], split_date),
        data_split(train, split_date, end.strftime("%Y-%m-%d")),
    )


def resolve_turbulence_threshold(cfg, train):
    """Liquidation threshold for trading, or ``None`` to disable it."""
    if not cfg.use_turbulence or cfg.task != "trading":
        return None
    if cfg.turbulence_threshold is not None:
        return float(cfg.turbulence_threshold)
    if cfg.turbulence_quantile is None:
        return None
    daily = train.drop_duplicates(subset=["date"])["turbulence"].to_numpy()
    threshold = float(np.quantile(daily, cfg.turbulence_quantile))
    if threshold <= 0:
        # Turbulence needs a year of history; a zero threshold would sell every day.
        logger.warning("Training period too short for a turbulence threshold; disabling it")
        return None
    return threshold


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
def _run_name(cfg):
    if cfg.run_name:
        return cfg.run_name
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return "{}_{}".format(now, cfg.agent)


def _seed_everything(seed):
    if seed is not None:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)


def _to_jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def evaluate_account(
    account_value, trade, cfg, out_dir, actions=None, title="Backtest", extra=None
):
    """Benchmark an account against the baselines and save the results."""
    from rl.trade.backtest import compare_stats, get_baseline, plot_account_values
    from rl.trade.baselines import baseline_account_values

    os.makedirs(out_dir, exist_ok=True)
    traded = trade[trade["date"].isin(account_value["date"])]
    strategies = {cfg.agent.upper(): account_value}
    strategies.update(
        baseline_account_values(traded, cfg.initial_amount, cfg.buy_cost_pct)
    )
    if cfg.benchmark_ticker:
        try:
            bench = get_baseline(
                cfg.benchmark_ticker,
                str(account_value["date"].iloc[0]),
                # Yahoo's end date is exclusive.
                (pd.Timestamp(account_value["date"].iloc[-1]) + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
            )
            if bench.empty:
                logger.warning("No benchmark data for %s", cfg.benchmark_ticker)
            else:
                strategies[cfg.benchmark_ticker] = bench[["date", "close"]]
        except Exception as exc:  # network problems should not lose the run
            logger.warning("Could not download benchmark %s: %s", cfg.benchmark_ticker, exc)

    stats = compare_stats(strategies)
    account_value.to_csv(os.path.join(out_dir, "account_value.csv"), index=False)
    if actions is not None:
        actions.to_csv(os.path.join(out_dir, "actions.csv"))
    for name, frame in (extra or {}).items():
        frame.to_csv(os.path.join(out_dir, name), index=False)
    stats.to_csv(os.path.join(out_dir, "perf_stats.csv"))
    plot_account_values(strategies, save_path=os.path.join(out_dir, "backtest.png"), title=title)
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 160):
        logger.info("Performance from %s to %s:\n%s", account_value["date"].iloc[0],
                    account_value["date"].iloc[-1], stats.to_string())
    return stats


def run_training(cfg=None):
    """Train ``cfg.agent`` on the training period and trade the rest.

    Returns a dict with the run name, output directories, the agent's account
    values and the statistics table (agent and baselines side by side).
    """
    from rl.model.models import DRLAgent, ValidationCallback, load_model, make_vec_env

    cfg = (cfg or TrainConfig()).validate()
    if cfg.agent == "ensemble":
        return run_ensemble(cfg)
    _seed_everything(cfg.seed)
    run_name = _run_name(cfg)
    model_dir = os.path.join(cfg.trained_model_dir, run_name)
    out_dir = os.path.join(cfg.results_dir, run_name)

    logger.info("==============Loading data and engineering features===========")
    _, train, trade = prepare_data(cfg)
    validation = None
    if cfg.validation_days:
        train, validation = split_validation(train, cfg.validation_days)
    tickers = sorted(train["tic"].unique())
    kwargs = env_kwargs(cfg, len(tickers))
    threshold = resolve_turbulence_threshold(cfg, train)
    logger.info(
        "%s task, %d tickers, %d training days, %d validation days, %d trading days, "
        "turbulence threshold %s",
        cfg.task, len(tickers), train.index.nunique(),
        0 if validation is None else validation.index.nunique(), trade.index.nunique(), threshold,
    )

    logger.info("==============Training %s===========", cfg.agent.upper())
    train_env = make_env(cfg, train, kwargs, results_dir=out_dir)
    venv = make_vec_env(train_env, normalize_obs=cfg.normalize_observations, seed=cfg.seed)
    agent = DRLAgent(env=venv)
    model = agent.get_model(
        cfg.agent, model_kwargs=cfg.model_kwargs, verbose=cfg.verbose, seed=cfg.seed
    )
    os.makedirs(model_dir, exist_ok=True)
    validator = None
    if validation is not None:
        validator = ValidationCallback(
            lambda: make_env(cfg, validation, kwargs, turbulence_threshold=threshold,
                             results_dir=out_dir),
            eval_freq=cfg.eval_freq,
            save_dir=model_dir,
        )
    model = agent.train_model(
        model=model,
        tb_log_name=cfg.agent,
        total_timesteps=cfg.total_timesteps,
        callback=validator,
    )

    run_config = dataclasses.asdict(cfg)
    run_config.update(
        run_name=run_name,
        tickers=tickers,
        resolved_turbulence_threshold=threshold,
    )
    obs_normalizer = "auto"
    if validator is None:
        model.save(os.path.join(model_dir, "model.zip"))
        if model.get_vec_normalize_env() is not None:
            model.get_vec_normalize_env().save(os.path.join(model_dir, "vecnormalize.pkl"))
    else:
        # Trade with the best validation checkpoint, not the final model.
        model, obs_normalizer = load_model(model_dir, cfg.agent, make_env(cfg, trade, kwargs))
        history = pd.DataFrame(validator.history)
        os.makedirs(out_dir, exist_ok=True)
        history.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
        run_config.update(
            best_validation_timesteps=validator.best["timesteps"],
            best_validation_sharpe=validator.best["sharpe"],
        )
        logger.info(
            "Selected the checkpoint at %d of %d steps (validation Sharpe %.3f)",
            validator.best["timesteps"], validator.history[-1]["timesteps"],
            validator.best["sharpe"],
        )
    with open(os.path.join(model_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=_to_jsonable)

    logger.info("==============Trading===========")
    result = _trade(model, cfg, trade, kwargs, threshold, out_dir, obs_normalizer=obs_normalizer)
    result.update(run_name=run_name, model_dir=model_dir, results_dir=out_dir)
    if validator is not None:
        result["validation"] = history
    return result


def _trade(model, cfg, trade, kwargs, threshold, out_dir, obs_normalizer="auto"):
    from rl.model.models import DRLAgent

    trade_env = make_env(cfg, trade, kwargs, turbulence_threshold=threshold, results_dir=out_dir)
    account_value, actions = DRLAgent.DRL_prediction(
        model, trade_env, deterministic=True, obs_normalizer=obs_normalizer
    )
    stats = evaluate_account(
        account_value,
        trade,
        cfg,
        out_dir,
        actions=actions,
        title="{} vs baselines".format(cfg.agent.upper()),
    )
    return {"account_value": account_value, "actions": actions, "stats": stats}


def load_run_config(model_dir):
    with open(os.path.join(model_dir, "run_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in saved.items() if k in fields})
    return cfg, saved


def run_backtest(model_dir, **overrides):
    """Trade with a model saved by :func:`run_training`.

    ``overrides`` replace saved settings, e.g. ``start_trade_date`` and
    ``end_date`` to test another period, or ``data_source``/``data_file``.
    Features are recomputed from ``start_date`` so indicators are warmed up.
    """
    from rl.model.models import load_model

    cfg, saved = load_run_config(model_dir)
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    cfg.validate()
    tickers = saved["tickers"]
    _, _, trade = prepare_data(cfg, tickers=tickers)
    kwargs = env_kwargs(cfg, len(tickers))
    threshold = saved.get("resolved_turbulence_threshold") if cfg.use_turbulence else None
    if cfg.task == "trading" and overrides.get("turbulence_threshold") is not None:
        threshold = float(overrides["turbulence_threshold"])

    model, normalizer = load_model(model_dir, cfg.agent, make_env(cfg, trade, kwargs))
    out_dir = os.path.join(
        cfg.results_dir,
        "{}_backtest_{}".format(saved["run_name"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
    )
    result = _trade(model, cfg, trade, kwargs, threshold, out_dir, obs_normalizer=normalizer)
    result.update(run_name=saved["run_name"], model_dir=model_dir, results_dir=out_dir)
    return result


def run_ensemble(cfg):
    """Rolling-window A2C/PPO/DDPG ensemble over the trading period."""
    from rl.model.models import DRLEnsembleAgent, MODEL_KWARGS

    cfg = dataclasses.replace(
        cfg, agent="ensemble", task="trading", use_turbulence=True
    ).validate()
    _seed_everything(cfg.seed)
    run_name = _run_name(cfg)
    out_dir = os.path.join(cfg.results_dir, run_name)
    processed, train, trade = prepare_data(cfg)
    tickers = sorted(train["tic"].unique())
    kwargs = env_kwargs(cfg, len(tickers))
    ensemble = DRLEnsembleAgent(
        df=processed,
        train_period=(cfg.start_date, cfg.start_trade_date),
        val_test_period=(cfg.start_trade_date, cfg.end_date),
        rebalance_window=cfg.rebalance_window,
        validation_window=cfg.validation_window,
        stock_dim=kwargs["stock_dim"],
        hmax=kwargs["hmax"],
        initial_amount=kwargs["initial_amount"],
        buy_cost_pct=kwargs["buy_cost_pct"],
        sell_cost_pct=kwargs["sell_cost_pct"],
        reward_scaling=kwargs["reward_scaling"],
        state_space=kwargs["state_space"],
        action_space=kwargs["action_space"],
        tech_indicator_list=kwargs["tech_indicator_list"],
        print_verbosity=10,
        normalize_obs=cfg.normalize_observations,
        reward_type=cfg.reward_type,
        seed=cfg.seed,
        save_models=False,
        # Per-window validation and trading files.
        results_dir=os.path.join(out_dir, "windows"),
        verbose=cfg.verbose,
    )
    timesteps = {name: cfg.total_timesteps for name in ("a2c", "ppo", "ddpg")}
    summary = ensemble.run_ensemble_strategy(
        MODEL_KWARGS["a2c"], MODEL_KWARGS["ppo"], MODEL_KWARGS["ddpg"], timesteps
    )
    if ensemble.account_value is None:
        raise ValueError(
            "The trading period is shorter than rebalance_window + validation_window "
            "({} days)".format(cfg.rebalance_window + cfg.validation_window)
        )
    stats = evaluate_account(
        ensemble.account_value,
        trade,
        cfg,
        out_dir,
        title="Ensemble vs baselines",
        extra={"ensemble_summary.csv": summary},
    )
    logger.info("Ensemble model choices:\n%s", summary.to_string(index=False))
    return {
        "run_name": run_name,
        "results_dir": out_dir,
        "account_value": ensemble.account_value,
        "summary": summary,
        "stats": stats,
    }


def train_one():
    """
    train an agent with the default settings
    """
    return run_training(TrainConfig())
