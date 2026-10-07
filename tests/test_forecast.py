import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_panel
from rl.forecast import lstm
from rl.trade.baselines import equal_weight_rebalanced, target_weights_strategy


def test_raw_features_are_scale_free():
    df = make_panel(
        [[10, 100], [11, 110], [12.1, 121]],
        tickers=["A", "B"],
        macd=[[1, 10], [1, 10], [1, 10]],
        close_20_sma=[[5, 50], [11, 110], [0, 0]],
        rsi_10=[[30, 70], [40, 60], [50, 50]],
    )
    features = lstm.raw_features(df, ["macd", "close_20_sma", "rsi_10"])
    a = features[features.tic == "A"].reset_index(drop=True)
    b = features[features.tic == "B"].reset_index(drop=True)
    # Ten times the price gives the same scale-free features.
    for name in ("log_return", "macd", "close_20_sma"):
        np.testing.assert_allclose(a[name], b[name])
    np.testing.assert_allclose(a["log_return"], [0, np.log(1.1), np.log(1.1)])
    # Zero-filled warm-up values give a neutral 0 instead of -inf.
    np.testing.assert_allclose(a["close_20_sma"], [np.log(0.5), 0, 0])
    np.testing.assert_allclose(a["macd"], [0.1, 1 / 11, 1 / 12.1])
    stats = lstm.fit_scaler(features, ["macd", "close_20_sma", "rsi_10"])
    assert set(stats) == {"macd", "rsi_10"}
    scaled = lstm.apply_scaler(features, stats)
    assert scaled["rsi_10"].mean() == pytest.approx(0)


def test_windows_stay_within_a_ticker():
    features = pd.DataFrame(
        {"tic": ["A"] * 4 + ["B"] * 3, "date": list("1234") + list("123"), "x": np.arange(7.0)}
    )
    closes = [1, 2, 4, 8, 10, 20, 30]
    X, y, rows = lstm.make_windows(features, closes, window=2, columns=["x"])
    np.testing.assert_array_equal(rows, [1, 2, 3, 5, 6])
    np.testing.assert_array_equal(X[:, :, 0], [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6]])
    np.testing.assert_allclose(y, [2, 2, np.nan, 1.5, np.nan])


def test_mape_loss_is_the_price_mape():
    pred_ratio = torch.tensor([1.02, 0.97])
    target_ratio = torch.tensor([1.0, 1.0])
    loss = lstm.forecast_loss(torch.log(pred_ratio), target_ratio, "mape")
    assert loss.item() == pytest.approx(100 * (0.02 + 0.03) / 2)
    with pytest.raises(ValueError):
        lstm.forecast_loss(pred_ratio, target_ratio, "huber")


def test_forecaster_learns_a_simple_pattern():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 5, 2)).astype(np.float32)
    # Prices rise 2% after a positive last feature and fall 2% otherwise.
    y = np.where(X[:, -1, 0] > 0, 1.02, 0.98)
    model, history = lstm.train_forecaster(
        X[:400], y[:400], X[400:], y[400:], hidden_size=16, epochs=40, learning_rate=1e-2
    )
    assert len(history) == 40
    predicted = lstm.predict_ratio(model, X[400:])
    assert np.mean((predicted >= 1) == (y[400:] > 1)) > 0.9


def test_forecast_metrics_and_signal_weights():
    predictions = pd.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d2"],
            "tic": ["A", "B", "A", "B"],
            "close": [10.0, 10.0, 11.0, 9.0],
            "predicted_close": [11.0, 9.5, 11.0, 9.0],
            "next_close": [11.0, 9.0, np.nan, np.nan],
        }
    )
    metrics = lstm.forecast_metrics(predictions)
    assert metrics["Forecasts"] == 2
    assert metrics["Directional accuracy"] == 1.0
    assert metrics["MAPE (%)"] == pytest.approx(100 * (0 + 0.5 / 9) / 2)
    predictions["signal"] = (predictions.predicted_close >= predictions.close).astype(float)
    weights = lstm.signal_weights(predictions, leverage=2.0)
    np.testing.assert_allclose(weights.to_numpy(), [[1.0, 0.0], [1.0, 1.0]])


def test_target_weights_strategy():
    df = make_panel([[10, 20], [11, 18], [12.1, 18], [11, 19.8]])
    dates = sorted(df.date.unique())
    equal = pd.DataFrame(0.5, index=dates, columns=["T0", "T1"])
    pd.testing.assert_frame_equal(
        target_weights_strategy(df, equal, 1000.0, 0.001),
        equal_weight_rebalanced(df, 1000.0, 0.001, rebalance_every=1),
        check_exact=False,
    )
    cash = target_weights_strategy(df, equal * 0, 1000.0, 0.01)
    np.testing.assert_allclose(cash.account_value, 1000.0)
    # Twice the capital in T0 doubles its daily returns.
    levered = pd.DataFrame({"T0": 2.0, "T1": 0.0}, index=dates)
    values = target_weights_strategy(df, levered, 1000.0).account_value
    np.testing.assert_allclose(values, [1000, 1200, 1440, 1440 * (1 + 2 * (11 / 12.1 - 1))])
    with pytest.raises(ValueError):
        target_weights_strategy(df, equal.iloc[:2], 1000.0)
