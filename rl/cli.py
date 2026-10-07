"""Command-line interface: ``python main.py --mode <mode> [options]``."""

import datetime
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from rl.config import config

MODES = ("train", "backtest", "ensemble", "download_data")


def build_parser():
    parser = ArgumentParser(
        description="Train and backtest deep reinforcement learning stock-trading agents.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=MODES,
        default="train",
        help="train: train an agent and trade the test period; backtest: trade with a "
        "saved model; ensemble: rolling A2C/PPO/DDPG ensemble; download_data: save "
        "Yahoo Finance prices to CSV",
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-source",
        choices=("yahoo", "csv", "synthetic"),
        default=None,
        help="where prices come from (default: csv when --data-file is given, else yahoo)",
    )
    data.add_argument("--data-file", help="CSV of date,tic,open,high,low,close,volume rows")
    data.add_argument("--tickers", nargs="+", help="tickers to trade (default: Dow 30)")
    data.add_argument("--start-date", default=config.START_DATE)
    data.add_argument(
        "--start-trade-date",
        default=config.START_TRADE_DATE,
        help="first day of the trading (test) period; earlier days are for training",
    )
    data.add_argument("--end-date", default=config.END_DATE, help="exclusive")
    data.add_argument("--output", help="download_data: output CSV path")

    agent = parser.add_argument_group("agent")
    agent.add_argument(
        "--task",
        choices=("trading", "portfolio"),
        default="trading",
        help="trading: buy and sell share counts; portfolio: choose daily portfolio weights",
    )
    agent.add_argument("--agent", choices=("a2c", "ppo", "ddpg", "td3", "sac"), default="sac")
    agent.add_argument(
        "--timesteps", type=int, default=80_000, help="training steps (per model for ensemble)"
    )
    agent.add_argument("--seed", type=int, help="random seed for reproducible runs")
    agent.add_argument(
        "--no-normalize",
        action="store_true",
        help="feed raw observations instead of running-mean/std normalised ones",
    )
    agent.add_argument(
        "--validation-days",
        type=int,
        default=0,
        help="hold out the last N training days and keep the checkpoint that trades them "
        "best (0: train on everything and keep the final model)",
    )
    agent.add_argument(
        "--eval-freq", type=int, default=10_000, help="training steps between validation runs"
    )
    agent.add_argument("--rebalance-window", type=int, default=63, help="ensemble only")
    agent.add_argument("--validation-window", type=int, default=63, help="ensemble only")

    env = parser.add_argument_group("environment")
    env.add_argument("--hmax", type=int, default=100, help="max shares traded per stock per day")
    env.add_argument("--initial-amount", type=float, default=1_000_000)
    env.add_argument(
        "--transaction-cost", type=float, default=0.001, help="fraction paid on each buy and sell"
    )
    env.add_argument("--reward-type", choices=("asset_change", "log_return"), default="asset_change")
    env.add_argument("--reward-scaling", type=float, help="default: 1e-4 (asset_change), 1 (log_return)")
    env.add_argument(
        "--cov-lookback",
        type=int,
        default=252,
        help="portfolio task: days of returns in each covariance matrix",
    )
    env.add_argument("--no-turbulence", action="store_true", help="disable turbulence liquidation")
    env.add_argument(
        "--turbulence-threshold", type=float, help="fixed liquidation threshold (overrides quantile)"
    )
    env.add_argument(
        "--turbulence-quantile",
        type=float,
        default=0.99,
        help="liquidate when turbulence exceeds this quantile of in-sample turbulence",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--benchmark", help="Yahoo ticker to compare against, e.g. ^DJI or SPY")
    out.add_argument("--run-name", help="output sub-directory name (default: timestamp_agent)")
    out.add_argument("--model-dir", help="backtest: directory of a saved run")
    out.add_argument("--results-dir", default=config.RESULTS_DIR)
    out.add_argument("--trained-model-dir", default=config.TRAINED_MODEL_DIR)
    out.add_argument("-v", "--verbose", action="store_true", help="show stable-baselines3 training logs")
    return parser


def _train_config(options):
    from rl.autotrain.training import TrainConfig

    data_source = options.data_source or ("csv" if options.data_file else "yahoo")
    reward_scaling = options.reward_scaling
    if reward_scaling is None:
        reward_scaling = 1.0 if options.reward_type == "log_return" else 1e-4
    return TrainConfig(
        task=options.task,
        agent="ensemble" if options.mode == "ensemble" else options.agent,
        total_timesteps=options.timesteps,
        data_source=data_source,
        data_file=options.data_file,
        ticker_list=options.tickers,
        start_date=options.start_date,
        start_trade_date=options.start_trade_date,
        end_date=options.end_date,
        use_turbulence=not options.no_turbulence,
        turbulence_threshold=options.turbulence_threshold,
        turbulence_quantile=options.turbulence_quantile,
        hmax=options.hmax,
        initial_amount=options.initial_amount,
        buy_cost_pct=options.transaction_cost,
        sell_cost_pct=options.transaction_cost,
        reward_scaling=reward_scaling,
        reward_type=options.reward_type,
        cov_lookback=options.cov_lookback,
        normalize_observations=not options.no_normalize,
        validation_days=options.validation_days,
        eval_freq=options.eval_freq,
        seed=options.seed,
        benchmark_ticker=options.benchmark,
        run_name=options.run_name,
        results_dir=options.results_dir,
        trained_model_dir=options.trained_model_dir,
        rebalance_window=options.rebalance_window,
        validation_window=options.validation_window,
        verbose=1 if options.verbose else 0,
    )


def _explicit(parser, options, names):
    """Options from ``names`` whose value differs from the parser default."""
    return {
        name: getattr(options, name)
        for name in names
        if getattr(options, name) != parser.get_default(name)
    }


def main(argv=None):
    parser = build_parser()
    options = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("rl")

    if options.mode == "download_data":
        from rl.marketdata.yahoodownloader import YahooDownloader

        df = YahooDownloader(
            start_date=options.start_date,
            end_date=options.end_date,
            ticker_list=options.tickers or config.DOW_30_TICKER,
        ).fetch_data()
        path = options.output
        if path is None:
            now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
            path = os.path.join(config.DATA_SAVE_DIR, now + ".csv")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        log.info("Saved %d rows to %s", len(df), path)
        return path

    if options.mode == "ensemble" and options.task != "trading":
        parser.error("--mode ensemble supports only --task trading")

    if options.mode == "backtest":
        from rl.autotrain.training import run_backtest

        if not options.model_dir:
            parser.error("--model-dir is required for --mode backtest")
        overrides = _explicit(
            parser,
            options,
            ["data_file", "start_trade_date", "end_date", "benchmark", "turbulence_threshold",
             "results_dir"],
        )
        if options.data_source or options.data_file:
            overrides["data_source"] = options.data_source or "csv"
        if "benchmark" in overrides:
            overrides["benchmark_ticker"] = overrides.pop("benchmark")
        result = run_backtest(options.model_dir, **overrides)
    else:
        from rl.autotrain.training import run_training

        result = run_training(_train_config(options))
    log.info("Results written to %s", result["results_dir"])
    return result
