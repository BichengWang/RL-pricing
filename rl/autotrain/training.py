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
ENSEMBLE_AGENTS = ("a2c", "ppo", "ddpg")
# trading: buy and sell whole shares (StockTradingEnv);
# portfolio: choose long-only portfolio weights each day (StockPortfolioEnv).
TASKS = ("trading", "portfolio")


def normalize_date(value, name="date"):
    """``value`` (a date string or datetime) as a ``YYYY-MM-DD`` string."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} is not a valid date: {!r}".format(name, value)) from exc
    if pd.isna(timestamp):
        raise ValueError("{} is not a valid date: {!r}".format(name, value))
    return timestamp.strftime("%Y-%m-%d")


@dataclass
class TrainConfig:
    """Settings for one training / backtest run.

    ``ticker_list`` defaults to the Dow 30 for the Yahoo and synthetic data
    sources and to every ticker in the file for the CSV source.

    With ``task="portfolio"`` the agent allocates the portfolio across the
    tickers instead of trading share counts; ``hmax`` and the turbulence
    settings do not apply, and the first ``cov_lookback`` days of data are
    used to warm up the covariance features.

    ``agent="lstm"`` runs the forecasting strategy instead of a DRL agent: an
    LSTM forecasts each ticker's next close and the portfolio holds the
    stocks forecast to rise (see :mod:`rl.forecast.lstm`). It uses the
    ``lstm_*`` settings and ``leverage``; ``total_timesteps``, ``hmax``, the
    reward and turbulence settings do not apply.
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
    # Per-algorithm training steps, e.g. {"a2c": 30_000, "ppo": 100_000, "ddpg": 10_000};
    # algorithms not listed use ``total_timesteps``.
    ensemble_timesteps: Optional[dict] = None
    # Per-algorithm model settings replacing the config.py defaults, e.g.
    # {"ddpg": {"action_noise": {"type": "ornstein_uhlenbeck", "sigma": 0.5}}}.
    ensemble_model_kwargs: Optional[dict] = None
    # Also trade each algorithm with its own account (refit like the winner).
    standalone_accounts: bool = False
    # LSTM forecasting strategy
    lstm_window: int = 20
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 2
    lstm_epochs: int = 30
    lstm_batch_size: int = 256
    lstm_learning_rate: float = 1e-3
    lstm_loss: str = "mape"
    # Hold out floor(fraction * raw pre-trade sessions), most recent first.
    # The last fitting signal is purged if its next-close label is held out.
    lstm_validation_fraction: float = 0.1
    leverage: float = 1.0
    verbose: int = 0

    def validate(self):
        """Check the settings and normalise the dates to ``YYYY-MM-DD``."""
        # Dates are compared as strings throughout the pipeline, so
        # "2019-1-1" would sort after "2019-09-30" and silently move the
        # split between training and trading.
        for name in ("start_date", "start_trade_date", "end_date"):
            setattr(self, name, normalize_date(getattr(self, name), name))
        if self.task not in TASKS:
            raise ValueError("task must be one of {}".format(TASKS))
        if self.task == "portfolio" and self.agent == "ensemble":
            raise ValueError("The ensemble strategy supports only the trading task")
        if self.cov_lookback < 2:
            raise ValueError("cov_lookback must be at least 2")
        for name in ("buy_cost_pct", "sell_cost_pct"):
            cost = getattr(self, name)
            if not np.isfinite(cost) or not 0 <= cost < 1:
                raise ValueError("{} must be finite and in [0, 1)".format(name))
        if self.data_source not in DATA_SOURCES:
            raise ValueError("data_source must be one of {}".format(DATA_SOURCES))
        if self.data_source == "csv" and not self.data_file:
            raise ValueError("data_file is required for the csv data source")
        if self.agent not in AGENTS + ("ensemble", "lstm"):
            raise ValueError("agent must be one of {}".format(AGENTS + ("ensemble", "lstm")))
        if self.agent == "lstm":
            if self.task != "trading":
                raise ValueError("The LSTM strategy supports only the trading task")
            if self.lstm_loss not in ("mape", "mse"):
                raise ValueError("lstm_loss must be 'mape' or 'mse'")
            if self.lstm_window < 1 or self.lstm_epochs < 1:
                raise ValueError("lstm_window and lstm_epochs must be at least 1")
            if not 0 <= self.lstm_validation_fraction < 1:
                raise ValueError("lstm_validation_fraction must be in [0, 1)")
            if self.leverage <= 0:
                raise ValueError("leverage must be positive")
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
        if self.turbulence_threshold is not None and self.agent == "ensemble":
            raise ValueError("The ensemble strategy sets its own turbulence threshold every window")
        for name in ("ensemble_timesteps", "ensemble_model_kwargs"):
            unknown = set(getattr(self, name) or {}) - set(ENSEMBLE_AGENTS)
            if unknown:
                raise ValueError(
                    "{} has unknown algorithms {}; use {}".format(
                        name, sorted(unknown), ENSEMBLE_AGENTS
                    )
                )
        for agent, steps in (self.ensemble_timesteps or {}).items():
            if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
                raise ValueError(
                    "ensemble_timesteps[{!r}] must be a positive integer".format(agent)
                )
        if self.validation_days and self.agent == "lstm":
            raise ValueError("The LSTM strategy has its own lstm_validation_fraction")
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
    if tickers is not None:
        missing = sorted(set(tickers) - set(df["tic"]))
        if missing:
            # Typos and delisted symbols would otherwise just shrink the universe.
            logger.warning(
                "No %s data between %s and %s for: %s",
                cfg.data_source, cfg.start_date, cfg.end_date, ", ".join(missing),
            )
    return df.reset_index(drop=True)


def _eligibility_cutoff(cfg, dates):
    """Exclusive fitting end, determined only from raw pre-trade sessions.

    LSTM holds out floor(fraction * raw training sessions), not a fraction
    of label-bearing windows. Thus eligibility, scaling and labels all use
    one date boundary, independent of feature warm-up or ticker coverage.
    """
    if cfg.agent == "ensemble":
        n_validation = cfg.validation_window
    elif cfg.agent == "lstm":
        n_validation = int(len(dates) * cfg.lstm_validation_fraction)
    else:
        n_validation = cfg.validation_days
    if n_validation:
        if n_validation < 2 and cfg.agent != "lstm":
            raise ValueError("The validation period must contain at least 2 raw sessions")
        if n_validation > len(dates) - 2:
            raise ValueError("Not enough fitting sessions before the validation period")
        return dates[-n_validation], dates[-n_validation]
    return cfg.start_trade_date, None


def prepare_data(cfg, tickers=None, eligibility_cutoff=None):
    """Return ``(processed, train, trade)`` with a causally frozen universe.

    Eligibility is fitted only before the exclusive ``eligibility_cutoff``.
    By default that boundary excludes the configured validation sessions.
    Saved ``tickers`` bypass eligibility selection; replay can also supply
    the saved cutoff. Fitting history may drop incomplete dates according
    to :func:`balance_panel`, but validation/trading sessions never do.
    All frames carry ``eligibility_cutoff``, ``eligibility_tickers`` and
    ``validation_start_date`` in their attrs.
    """
    from rl.preprocessing.data import data_split
    from rl.preprocessing.preprocessors import (
        FeatureEngineer,
        add_covariance_matrix,
        balance_panel,
        require_complete_prices,
    )

    raw = load_market_data(cfg).copy()
    raw["date"] = raw["date"].astype(str)
    training_dates = sorted(raw.loc[raw["date"] < cfg.start_trade_date, "date"].unique())
    if eligibility_cutoff is None:
        cutoff, validation_start = _eligibility_cutoff(cfg, training_dates)
    else:
        cutoff = normalize_date(eligibility_cutoff, "eligibility_cutoff")
        validation_start = cutoff if cutoff < cfg.start_trade_date else None
    # Replay may move trading earlier than the saved fit boundary. Those
    # observed trading sessions must still be checked, never history-balanced.
    evaluation_start = min(cutoff, cfg.start_trade_date)
    fitting = raw[raw["date"] < evaluation_start]
    if tickers is None:
        fitting = balance_panel(fitting)
        frozen = sorted(fitting["tic"].unique())
    else:
        frozen = sorted(set(tickers))
        fitting = fitting[fitting["tic"].isin(frozen)].drop_duplicates(["date", "tic"])
        counts = fitting.groupby("date")["tic"].nunique()
        fitting = fitting[fitting["date"].isin(counts[counts == len(frozen)].index)]
    if not frozen or fitting.empty:
        raise ValueError("No eligible ticker fitting history before {}".format(cutoff))
    evaluation_dates = sorted(raw.loc[raw["date"] >= evaluation_start, "date"].unique())
    evaluation = raw[(raw["date"] >= evaluation_start) & raw["tic"].isin(frozen)]
    require_complete_prices(fitting, frozen, fitting["date"].unique())
    require_complete_prices(evaluation, frozen, evaluation_dates)
    selected = pd.concat([fitting, evaluation], ignore_index=True)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=cfg.tech_indicator_list,
        use_turbulence=cfg.use_turbulence and cfg.task == "trading",
        user_defined_feature=False,
        clean_data=False,
    )
    processed = fe.preprocess_data(selected)
    if cfg.task == "portfolio":
        if tickers is None and fitting["date"].nunique() < cfg.cov_lookback + 2:
            raise ValueError(
                "Portfolio fitting history must contain at least {} sessions before {}; "
                "covariance warm-up cannot consume validation or trading sessions".format(
                    cfg.cov_lookback + 2, cutoff
                )
            )
        processed = add_covariance_matrix(processed, lookback=cfg.cov_lookback)
        # Covariance warm-up is allowed to trim fitting history, not evaluation.
        require_complete_prices(processed, frozen, evaluation_dates)
    processed.attrs.update(
        eligibility_cutoff=cutoff,
        eligibility_tickers=frozen,
        validation_start_date=validation_start,
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
    account_value, trade, cfg, out_dir, actions=None, title="Backtest", extra=None,
    history=None, agent_accounts=None,
):
    """Benchmark an account against the baselines and save the results.

    ``history`` (rows before ``trade``) adds the minimum-variance baseline;
    ``agent_accounts`` (name -> account frame) adds standalone-agent accounts.
    """
    from rl.trade.backtest import compare_stats, get_baseline, plot_account_values
    from rl.trade.baselines import baseline_account_values, buy_and_hold

    os.makedirs(out_dir, exist_ok=True)
    traded = trade[trade["date"].isin(account_value["date"])]
    strategies = {cfg.agent.upper(): account_value}
    strategies.update({name.upper(): frame for name, frame in (agent_accounts or {}).items()})
    strategies.update(
        baseline_account_values(traded, cfg.initial_amount, cfg.buy_cost_pct, history=history)
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
                benchmark_dates = sorted(account_value["date"].unique())
                bench = bench.set_index("date").reindex(benchmark_dates).reset_index()
                strategies[cfg.benchmark_ticker] = buy_and_hold(
                    bench.assign(tic=cfg.benchmark_ticker),
                    float(account_value["account_value"].iloc[0]), cfg.buy_cost_pct,
                )
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
    if cfg.agent == "lstm":
        return run_lstm(cfg)
    _seed_everything(cfg.seed)
    run_name = _run_name(cfg)
    model_dir = os.path.join(cfg.trained_model_dir, run_name)
    out_dir = os.path.join(cfg.results_dir, run_name)

    logger.info("==============Loading data and engineering features===========")
    processed, train, trade = prepare_data(cfg)
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
        eligibility_cutoff=processed.attrs["eligibility_cutoff"],
        execution_timing="next_open",
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
    executions = trade_env.save_execution_memory()
    stats = evaluate_account(
        account_value,
        trade,
        cfg,
        out_dir,
        actions=actions,
        title="{} vs baselines".format(cfg.agent.upper()),
        extra={"executions.csv": executions},
    )
    return {
        "account_value": account_value, "actions": actions,
        "executions": executions, "stats": stats,
    }


def load_run_config(model_dir):
    with open(os.path.join(model_dir, "run_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    if saved.get("execution_timing") != "next_open":
        raise ValueError(
            "Saved model does not use execution_timing='next_open'; retrain legacy artifacts"
        )
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
    threshold = saved.get("resolved_turbulence_threshold") if cfg.use_turbulence else None
    # As in training, use_turbulence=False wins over a fixed threshold.
    fixed = overrides.get("turbulence_threshold")
    if cfg.use_turbulence and cfg.task == "trading" and fixed is not None:
        threshold = float(fixed)
    if cfg.agent == "lstm":
        result = _backtest_lstm(model_dir, cfg, saved)
    else:
        tickers = saved["tickers"]
        _, _, trade = prepare_data(
            cfg, tickers=tickers, eligibility_cutoff=saved["eligibility_cutoff"]
        )
        kwargs = env_kwargs(cfg, len(tickers))
        model, normalizer = load_model(model_dir, cfg.agent, make_env(cfg, trade, kwargs))
        out_dir = os.path.join(
            cfg.results_dir,
            "{}_backtest_{}".format(
                saved["run_name"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ),
        )
        result = _trade(model, cfg, trade, kwargs, threshold, out_dir, obs_normalizer=normalizer)
        result.update(run_name=saved["run_name"], model_dir=model_dir, results_dir=out_dir)
    effective = dataclasses.asdict(cfg)
    effective.update(
        run_name=saved["run_name"], tickers=saved["tickers"],
        eligibility_cutoff=saved["eligibility_cutoff"], execution_timing="next_open",
        resolved_turbulence_threshold=threshold, source_model_dir=os.path.abspath(model_dir),
    )
    with open(os.path.join(result["results_dir"], "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(effective, f, indent=2, default=_to_jsonable)
    return result


def run_ensemble(cfg):
    """Rolling-window A2C/PPO/DDPG ensemble over the trading period.

    ``cfg.ensemble_timesteps`` and ``cfg.ensemble_model_kwargs`` set each
    algorithm's training steps and settings; ``cfg.use_turbulence=False`` turns
    the liquidation rule off; ``cfg.standalone_accounts`` adds an account per
    algorithm to the result (``agent_account_values``) and the statistics.
    """
    from rl.model.models import DRLEnsembleAgent, MODEL_KWARGS

    cfg = dataclasses.replace(cfg, agent="ensemble", task="trading").validate()
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
        use_turbulence=cfg.use_turbulence,
        standalone_accounts=cfg.standalone_accounts,
    )
    timesteps = {name: cfg.total_timesteps for name in ENSEMBLE_AGENTS}
    timesteps.update(cfg.ensemble_timesteps or {})
    model_kwargs = {name: MODEL_KWARGS[name] for name in ENSEMBLE_AGENTS}
    model_kwargs.update(cfg.ensemble_model_kwargs or {})
    summary = ensemble.run_ensemble_strategy(
        model_kwargs["a2c"], model_kwargs["ppo"], model_kwargs["ddpg"], timesteps
    )
    if ensemble.account_value is None:
        raise ValueError("The trading period must contain at least two dates")
    os.makedirs(out_dir, exist_ok=True)
    run_config = dataclasses.asdict(cfg)
    run_config.update(
        run_name=run_name, tickers=tickers, resolved_ensemble_timesteps=timesteps,
        eligibility_cutoff=processed.attrs["eligibility_cutoff"], execution_timing="next_open",
    )
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=_to_jsonable)
    extra = {"ensemble_summary.csv": summary, "executions.csv": ensemble.executions}
    agent_accounts = ensemble.agent_account_values
    if agent_accounts is not None:
        accounts = ensemble.account_value[["date"]].assign(
            ENSEMBLE=ensemble.account_value["account_value"].to_numpy()
        )
        for name in ENSEMBLE_AGENTS:
            accounts[name.upper()] = agent_accounts[name]["account_value"].to_numpy()
        extra["agent_accounts.csv"] = accounts
    first_trade_date = ensemble.account_value["date"].iloc[0]
    stats = evaluate_account(
        ensemble.account_value,
        trade,
        cfg,
        out_dir,
        actions=ensemble.actions,
        title="Ensemble vs baselines",
        extra=extra,
        history=processed[processed["date"] < first_trade_date],
        agent_accounts=agent_accounts,
    )
    logger.info("Ensemble model choices:\n%s", summary.to_string(index=False))
    return {
        "run_name": run_name,
        "results_dir": out_dir,
        "account_value": ensemble.account_value,
        "actions": ensemble.actions,
        "executions": ensemble.executions,
        "summary": summary,
        "stats": stats,
        "agent_account_values": agent_accounts,
    }


# ----------------------------------------------------------------------
# LSTM forecasting strategy
# ----------------------------------------------------------------------
def _lstm_inputs(cfg, processed, stats):
    """Scaled windows for every row of ``processed`` with enough history."""
    from rl.forecast import lstm

    ordered = processed.sort_values(["tic", "date"]).reset_index(drop=True)
    features = lstm.apply_scaler(lstm.raw_features(ordered, cfg.tech_indicator_list), stats)
    X, y, rows = lstm.make_windows(
        features, ordered["close"], cfg.lstm_window, lstm.feature_columns(cfg.tech_indicator_list)
    )
    return ordered, X, y, rows


def _lstm_trade(cfg, model, processed, trade, stats, out_dir, extra=None):
    """Forecast the trading period, trade the signals and save the results."""
    from rl.forecast import lstm
    from rl.trade.baselines import target_weights_strategy

    ordered, X, _, rows = _lstm_inputs(cfg, processed, stats)
    in_trade = ordered["date"].iloc[rows].isin(set(trade["date"])).to_numpy()
    X, rows = X[in_trade], rows[in_trade]
    picked = ordered.iloc[rows]
    ratio = lstm.predict_ratio(model, X)
    next_close = ordered.groupby("tic")["close"].shift(-1).iloc[rows]
    predictions = pd.DataFrame(
        {
            "date": picked["date"].to_numpy(),
            "tic": picked["tic"].to_numpy(),
            "close": picked["close"].to_numpy(),
            "predicted_close": picked["close"].to_numpy() * ratio,
            "next_close": next_close.to_numpy(),
        }
    ).sort_values(["date", "tic"], ignore_index=True)
    predictions["signal"] = (predictions["predicted_close"] >= predictions["close"]).astype(float)
    weights = lstm.signal_weights(predictions, cfg.leverage)
    traded = trade[trade["date"].isin(weights.index)]
    if traded["date"].nunique() < 2:
        raise ValueError("Not enough history before the trading period for the LSTM window")
    weights = weights.iloc[:-1]
    executions = weights.rename_axis("signal_date").reset_index()
    executions.insert(1, "fill_date", sorted(traded["date"].unique())[1:])
    account_value = target_weights_strategy(
        traded, weights, cfg.initial_amount, (cfg.buy_cost_pct + cfg.sell_cost_pct) / 2
    )
    metrics = lstm.forecast_metrics(predictions)
    logger.info("Forecast quality over the trading period:\n%s", metrics.to_string())
    files = {
        "predictions.csv": predictions,
        "forecast_metrics.csv": metrics.rename_axis("metric").reset_index(name="value"),
        "executions.csv": executions,
    }
    files.update(extra or {})
    stats_table = evaluate_account(
        account_value,
        trade,
        cfg,
        out_dir,
        actions=weights,
        title="LSTM forecast strategy vs baselines",
        extra=files,
    )
    return {
        "account_value": account_value,
        "actions": weights,
        "executions": executions,
        "predictions": predictions,
        "forecast_metrics": metrics,
        "stats": stats_table,
    }


def run_lstm(cfg):
    """Train the LSTM forecaster on the training period and trade its signals."""
    import torch

    from rl.forecast import lstm

    cfg = dataclasses.replace(cfg, use_turbulence=False).validate()
    _seed_everything(cfg.seed)
    run_name = _run_name(cfg)
    model_dir = os.path.join(cfg.trained_model_dir, run_name)
    out_dir = os.path.join(cfg.results_dir, run_name)

    logger.info("==============Loading data and engineering features===========")
    processed, train, trade = prepare_data(cfg)
    tickers = processed.attrs["eligibility_tickers"]
    cutoff = processed.attrs["eligibility_cutoff"]
    fitting = train[train["date"] < cutoff]
    stats = lstm.fit_scaler(
        lstm.raw_features(fitting, cfg.tech_indicator_list), cfg.tech_indicator_list
    )
    ordered, X, y, rows = _lstm_inputs(cfg, train, stats)
    dates = ordered["date"].to_numpy()[rows]
    outcome_dates = ordered.groupby("tic")["date"].shift(-1).fillna(cutoff).to_numpy()[rows]
    labelled = np.isfinite(y)
    # A final fitting-day signal observes a held-out next close: purge it,
    # rather than letting its validation outcome enter the training labels.
    is_fit = labelled & (dates < cutoff) & (outcome_dates < cutoff)
    X_fit, y_fit = X[is_fit], y[is_fit]
    if len(X_fit) == 0:
        raise ValueError("The fitting period is shorter than lstm_window plus one outcome day")
    if processed.attrs["validation_start_date"] is not None:
        is_val = labelled & (dates >= cutoff)
        X_val, y_val = X[is_val], y[is_val]
        if len(X_val) == 0:
            raise ValueError("The LSTM validation period has no labelled windows")
    else:
        X_val, y_val = None, None
    logger.info(
        "%d tickers, %d training and %d validation windows of %d days, %d trading days",
        len(tickers), len(X_fit), 0 if X_val is None else len(X_val), cfg.lstm_window,
        trade.index.nunique(),
    )

    logger.info("==============Training LSTM forecaster===========")
    model, history = lstm.train_forecaster(
        X_fit,
        y_fit,
        X_val,
        y_val,
        hidden_size=cfg.lstm_hidden_size,
        num_layers=cfg.lstm_num_layers,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch_size,
        learning_rate=cfg.lstm_learning_rate,
        loss=cfg.lstm_loss,
    )

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    run_config = dataclasses.asdict(cfg)
    run_config.update(
        run_name=run_name,
        tickers=tickers,
        eligibility_cutoff=cutoff,
        execution_timing="next_open",
        feature_scaler=stats,
    )
    with open(os.path.join(model_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=_to_jsonable)

    logger.info("==============Trading===========")
    result = _lstm_trade(
        cfg, model, processed, trade, stats, out_dir, extra={"training_history.csv": history}
    )
    result.update(run_name=run_name, model_dir=model_dir, results_dir=out_dir, history=history)
    return result


def _backtest_lstm(model_dir, cfg, saved):
    import torch

    from rl.forecast import lstm

    tickers = saved["tickers"]
    processed, _, trade = prepare_data(
        cfg, tickers=tickers, eligibility_cutoff=saved["eligibility_cutoff"]
    )
    stats = saved["feature_scaler"]
    model = lstm.LSTMForecaster(
        len(lstm.feature_columns(cfg.tech_indicator_list)),
        cfg.lstm_hidden_size,
        cfg.lstm_num_layers,
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), weights_only=True))
    model.eval()
    out_dir = os.path.join(
        cfg.results_dir,
        "{}_backtest_{}".format(saved["run_name"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
    )
    result = _lstm_trade(cfg, model, processed, trade, stats, out_dir)
    result.update(run_name=saved["run_name"], model_dir=model_dir, results_dir=out_dir)
    return result
