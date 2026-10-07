"""Selecting the training checkpoint on a held-out validation period."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from test_pipeline import small_config

from rl import cli
from rl.autotrain.training import (
    TrainConfig,
    run_backtest,
    run_training,
    split_validation,
)
from rl.preprocessing.data import data_split


def test_split_validation_holds_out_the_last_days(processed):
    train = data_split(processed, "2016-01-01", "2017-06-01")
    dates = sorted(train.date.unique())
    fit, validation = split_validation(train, 40)
    assert sorted(validation.date.unique()) == dates[-40:]
    assert sorted(fit.date.unique()) == dates[:-40]
    assert list(validation.index.unique()) == list(range(40))
    for bad in (0, 1, len(dates) - 1):
        with pytest.raises(ValueError, match="validation_days"):
            split_validation(train, bad)


def test_training_keeps_the_best_validation_checkpoint(tmp_path):
    cfg = small_config(tmp_path, total_timesteps=1000, validation_days=60, eval_freq=200)
    result = run_training(cfg)

    history = result["validation"]
    assert list(history.columns) == ["timesteps", "sharpe", "total_return"]
    assert list(history.timesteps) == [200, 400, 600, 800, 1000]
    pd.testing.assert_frame_equal(
        pd.read_csv(os.path.join(result["results_dir"], "validation.csv")), history
    )
    with open(os.path.join(result["model_dir"], "run_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    best = history.loc[history.sharpe.fillna(-np.inf).idxmax()]
    assert saved["best_validation_timesteps"] == best.timesteps
    assert saved["best_validation_sharpe"] == pytest.approx(best.sharpe)
    assert saved["validation_days"] == 60

    # The saved files are the checkpoint that was used for trading.
    replay = run_backtest(result["model_dir"], results_dir=str(tmp_path / "replay"))
    pd.testing.assert_frame_equal(replay["account_value"], result["account_value"])
    # Trading still covers the whole test period.
    assert result["account_value"].date.iloc[0] >= "2017-06-01"


def test_portfolio_task_with_validation(tmp_path):
    cfg = small_config(
        tmp_path,
        task="portfolio",
        start_date="2015-01-01",
        cov_lookback=60,
        total_timesteps=400,
        validation_days=40,
        eval_freq=200,
    )
    result = run_training(cfg)
    assert len(result["validation"]) == 2
    assert os.path.exists(os.path.join(result["model_dir"], "vecnormalize.pkl"))


def test_validation_settings_are_checked():
    with pytest.raises(ValueError):
        TrainConfig(validation_days=-1).validate()
    with pytest.raises(ValueError):
        TrainConfig(eval_freq=0).validate()
    with pytest.raises(ValueError, match="validation_window"):
        TrainConfig(agent="ensemble", validation_days=20).validate()


def test_cli_passes_validation_options():
    parser = cli.build_parser()
    options = parser.parse_args(["--validation-days", "126", "--eval-freq", "5000"])
    cfg = cli._train_config(options)
    assert (cfg.validation_days, cfg.eval_freq) == (126, 5000)
    assert cli._train_config(parser.parse_args([])).validation_days == 0
