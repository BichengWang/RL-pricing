"""End-to-end runs on synthetic data with tiny training budgets."""

import json
import os

import pandas as pd
import pytest

from conftest import TECH
from rl import cli
from rl.autotrain.training import (
    TrainConfig,
    resolve_turbulence_threshold,
    run_backtest,
    run_training,
)
from rl.config import config
from rl.env.env_stocktrading import StockTradingEnv
from rl.model.models import MODEL_KWARGS, build_model, make_vec_env
from rl.preprocessing.data import data_split

TICKERS = ["AAA", "BBB", "CCC"]


def small_config(tmp_path, **overrides):
    settings = dict(
        agent="a2c",
        total_timesteps=300,
        data_source="synthetic",
        ticker_list=TICKERS,
        start_date="2016-01-01",
        start_trade_date="2017-06-01",
        end_date="2017-12-01",
        tech_indicator_list=TECH,
        seed=0,
        run_name="test_run",
        results_dir=str(tmp_path / "results"),
        trained_model_dir=str(tmp_path / "models"),
    )
    settings.update(overrides)
    return TrainConfig(**settings)


def test_train_then_backtest_reproduces_the_run(tmp_path):
    result = run_training(small_config(tmp_path))
    model_dir = result["model_dir"]
    for name in ("model.zip", "vecnormalize.pkl", "run_config.json"):
        assert os.path.exists(os.path.join(model_dir, name))
    for name in ("account_value.csv", "actions.csv", "perf_stats.csv", "backtest.png"):
        assert os.path.exists(os.path.join(result["results_dir"], name))

    stats = result["stats"]
    assert list(stats.columns) == [
        "A2C",
        "Equal-weight buy & hold",
        "Equal-weight monthly rebalance",
    ]
    account = result["account_value"]
    assert account.date.iloc[0] >= "2017-06-01" and account.date.iloc[-1] < "2017-12-01"
    with open(os.path.join(model_dir, "run_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["tickers"] == TICKERS
    assert saved["resolved_turbulence_threshold"] > 0

    # Reloading the saved model and normaliser gives the same trades.
    replay = run_backtest(model_dir, results_dir=str(tmp_path / "replay"))
    pd.testing.assert_frame_equal(replay["account_value"], account)
    pd.testing.assert_frame_equal(replay["actions"], result["actions"])


def test_cli_trains_from_csv_without_turbulence(tmp_path, raw_prices):
    data_file = tmp_path / "prices.csv"
    raw_prices.to_csv(data_file, index=False)
    result = cli.main(
        [
            "--mode", "train",
            "--data-file", str(data_file),
            "--agent", "ppo",
            "--timesteps", "64",
            "--start-date", "2016-01-01",
            "--start-trade-date", "2018-01-01",
            "--end-date", "2018-07-01",
            "--no-turbulence",
            "--reward-type", "log_return",
            "--seed", "1",
            "--run-name", "cli_run",
            "--results-dir", str(tmp_path / "results"),
            "--trained-model-dir", str(tmp_path / "models"),
        ]
    )
    with open(os.path.join(result["model_dir"], "run_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["data_source"] == "csv"
    assert saved["resolved_turbulence_threshold"] is None
    assert saved["reward_scaling"] == 1.0
    assert sorted(saved["tickers"]) == sorted(raw_prices.tic.unique())


def test_ensemble_strategy_runs_walk_forward(tmp_path):
    cfg = small_config(
        tmp_path,
        agent="ensemble",
        total_timesteps=200,
        start_trade_date="2017-06-01",
        end_date="2017-09-15",
        rebalance_window=20,
        validation_window=20,
        model_kwargs=None,
    )
    result = run_training(cfg)
    summary = result["summary"]
    assert list(summary.columns) == [
        "Iter", "Val Start", "Val End", "Model Used", "A2C Sharpe", "PPO Sharpe", "DDPG Sharpe",
    ]
    assert len(summary) >= 2
    assert set(summary["Model Used"]) <= {"A2C", "PPO", "DDPG"}
    account = result["account_value"]
    # Windows are stitched together without gaps or repeated dates.
    assert account.date.is_unique and account.date.is_monotonic_increasing
    assert account.date.iloc[0] == summary["Val End"].iloc[0]
    assert "ENSEMBLE" in result["stats"].columns


def test_build_model_does_not_mutate_default_kwargs(processed):
    data = data_split(processed, "2017-01-01", "2017-03-01")
    n = 3
    env = StockTradingEnv(data, n, 10, 1e5, 0, 0, 1e-4, 1 + 2 * n + len(TECH) * n, n, TECH)
    venv = make_vec_env(env)
    kwargs = {"buffer_size": 1000, "learning_starts": 10, "action_noise": "normal"}
    before = dict(kwargs)
    build_model("td3", venv, model_kwargs=kwargs, verbose=0)
    build_model("td3", venv, model_kwargs=kwargs, verbose=0)
    assert kwargs == before
    assert MODEL_KWARGS["sac"] == config.SAC_PARAMS
    with pytest.raises(NotImplementedError):
        build_model("dqn", venv)


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(data_source="csv").validate()
    with pytest.raises(ValueError):
        TrainConfig(start_trade_date="1999-01-01").validate()
    with pytest.raises(ValueError):
        TrainConfig(agent="dqn").validate()
    assert TrainConfig().tickers() == config.DOW_30_TICKER
    assert TrainConfig(data_source="csv", data_file="x.csv").tickers() is None


def test_turbulence_threshold_resolution(processed):
    train = data_split(processed, "2016-01-01", "2018-01-01")
    cfg = TrainConfig(turbulence_quantile=0.9)
    daily = train.drop_duplicates("date").turbulence
    assert resolve_turbulence_threshold(cfg, train) == pytest.approx(daily.quantile(0.9))
    assert resolve_turbulence_threshold(TrainConfig(turbulence_threshold=5.0), train) == 5.0
    assert resolve_turbulence_threshold(TrainConfig(use_turbulence=False), train) is None
    # Less than a year of history means turbulence is all zeros; a zero
    # threshold would liquidate every day, so it is disabled instead.
    short = data_split(processed, "2016-01-01", "2016-10-01")
    assert resolve_turbulence_threshold(cfg, short) is None


def test_saving_a_plot_keeps_the_matplotlib_backend(tmp_path):
    import matplotlib

    from rl.trade.backtest import plot_account_values

    backend = matplotlib.get_backend()
    account = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "account_value": [1.0, 1.1]})
    plot_account_values({"A": account}, save_path=str(tmp_path / "plot.png"))
    assert (tmp_path / "plot.png").exists()
    assert matplotlib.get_backend() == backend
