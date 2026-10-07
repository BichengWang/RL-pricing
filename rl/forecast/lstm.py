"""LSTM price forecasting and the forecast-driven trading rule.

This is the "indirect" approach of Wang & Zhang (2021): a two-layer LSTM
followed by a dense layer forecasts the next close of every ticker, trained
on all tickers together with a mean absolute percentage error (MAPE) loss,
and a simple rule holds a stock while its forecast is at or above the
current close and holds cash otherwise.

The model sees a window of daily features per ticker:

- the log return of the close,
- price-level indicators (Bollinger bands, moving averages) as the log of
  their ratio to the close,
- other indicators standardised with statistics of the training period
  (``macd`` is divided by the close first, so it is scale free).

It predicts the log of the next close relative to the current one, so the
forecast does not depend on the price level and the MAPE of the forecast
price equals the MAPE of the predicted gross return.
"""

import copy
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

logger = logging.getLogger(__name__)

LOSSES = ("mape", "mse")


def is_price_level(indicator):
    """Whether an indicator is measured in price units (bands, averages)."""
    return indicator.startswith("boll") or indicator.endswith(("_sma", "_ema"))


def _safe_log_ratio(numerator, denominator):
    ratio = np.divide(
        numerator, denominator, out=np.ones_like(numerator, dtype=float), where=denominator > 0
    )
    return np.log(np.where(ratio > 0, ratio, 1.0))


def raw_features(df, tech_indicator_list):
    """Scale-free per-row features of a long (date, tic) frame, before scaling.

    Returns a frame aligned with ``df`` sorted by (tic, date), with ``date``
    and ``tic`` columns first.
    """
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    close = df["close"].to_numpy(dtype=float)
    out = pd.DataFrame({"date": df["date"], "tic": df["tic"]})
    prev = df.groupby("tic")["close"].shift(1).to_numpy(dtype=float)
    out["log_return"] = np.where(np.isnan(prev), 0.0, _safe_log_ratio(close, np.nan_to_num(prev)))
    for name in tech_indicator_list:
        values = df[name].to_numpy(dtype=float)
        if is_price_level(name):
            out[name] = _safe_log_ratio(values, close)
        elif name == "macd":
            out[name] = values / close
        else:
            out[name] = values
    return out


def fit_scaler(features, tech_indicator_list):
    """Mean and std of the indicators that are standardised."""
    stats = {}
    for name in tech_indicator_list:
        if is_price_level(name):
            continue
        std = float(features[name].std())
        stats[name] = [float(features[name].mean()), std if std > 0 else 1.0]
    return stats


def apply_scaler(features, stats):
    features = features.copy()
    for name, (mean, std) in stats.items():
        features[name] = (features[name] - mean) / std
    return features


def feature_columns(tech_indicator_list):
    return ["log_return"] + list(tech_indicator_list)


def make_windows(features, closes, window, columns):
    """Sliding windows of ``window`` days per ticker.

    ``features`` and ``closes`` are aligned and sorted by (tic, date). Returns
    ``(X, y, index)``: ``X[i]`` holds the features of the ``window`` days
    ending at row ``index[i]``, and ``y[i]`` the next close divided by that
    row's close (NaN on a ticker's last day).
    """
    values = features[columns].to_numpy(dtype=np.float32)
    close = np.asarray(closes, dtype=float)
    tics = features["tic"].to_numpy()
    xs, ys, rows = [], [], []
    start = 0
    n = len(features)
    while start < n:
        end = start
        while end < n and tics[end] == tics[start]:
            end += 1
        for row in range(start + window - 1, end):
            xs.append(values[row - window + 1 : row + 1])
            ys.append(close[row + 1] / close[row] if row + 1 < end else np.nan)
            rows.append(row)
        start = end
    if not xs:
        return (
            np.empty((0, window, len(columns)), dtype=np.float32),
            np.empty(0),
            np.empty(0, dtype=int),
        )
    return np.stack(xs), np.asarray(ys, dtype=float), np.asarray(rows)


class LSTMForecaster(nn.Module):
    """Two stacked LSTM layers and a dense layer predicting log(p[t+1] / p[t])."""

    def __init__(self, n_features, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def forecast_loss(pred_log_ratio, target_ratio, loss="mape"):
    """Loss between predicted and actual next closes, both relative to today."""
    pred = torch.exp(pred_log_ratio)
    if loss == "mape":
        return 100 * torch.mean(torch.abs(pred - target_ratio) / target_ratio)
    if loss == "mse":
        return torch.mean((pred - target_ratio) ** 2)
    raise ValueError("loss must be one of {}".format(LOSSES))


def train_forecaster(
    X,
    y,
    X_val=None,
    y_val=None,
    hidden_size=32,
    num_layers=2,
    epochs=30,
    batch_size=256,
    learning_rate=1e-3,
    loss="mape",
):
    """Train an :class:`LSTMForecaster`; keep the epoch with the best validation loss.

    Returns ``(model, history)`` where ``history`` lists the per-epoch
    training and validation losses.
    """
    if loss not in LOSSES:
        raise ValueError("loss must be one of {}".format(LOSSES))
    model = LSTMForecaster(X.shape[-1], hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_t, y_t = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_v, y_v = torch.as_tensor(X_val), torch.as_tensor(y_val, dtype=torch.float32)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X_t))
        total = 0.0
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            optimizer.zero_grad()
            batch_loss = forecast_loss(model(X_t[batch]), y_t[batch], loss)
            batch_loss.backward()
            optimizer.step()
            total += batch_loss.item() * len(batch)
        train_loss = total / len(order)
        val_loss = None
        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = forecast_loss(model(X_v), y_v, loss).item()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("epoch %d: train %s %.5f, validation %s", epoch + 1, loss, train_loss,
                    "n/a" if val_loss is None else "{:.5f}".format(val_loss))
        score = val_loss if has_val else train_loss
        if score < best_loss:
            best_loss, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def predict_ratio(model, X, batch_size=4096):
    """Predicted next close divided by the current close."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(torch.exp(model(torch.as_tensor(X[i : i + batch_size]))).numpy())
    return np.concatenate(preds) if preds else np.empty(0)


def forecast_metrics(predictions):
    """MAPE and directional accuracy of forecasts whose outcome is known.

    ``predictions`` has ``close``, ``predicted_close`` and ``next_close``
    columns (``next_close`` NaN where unknown).
    """
    known = predictions.dropna(subset=["next_close"])
    if known.empty:
        return pd.Series({"MAPE (%)": np.nan, "Directional accuracy": np.nan, "Forecasts": 0})
    actual, pred = known["next_close"], known["predicted_close"]
    up_actual = actual >= known["close"]
    up_pred = pred >= known["close"]
    return pd.Series(
        {
            "MAPE (%)": float(100 * np.mean(np.abs(pred - actual) / actual)),
            "Directional accuracy": float(np.mean(up_actual == up_pred)),
            "Forecasts": int(len(known)),
        }
    )


def signal_weights(predictions, leverage=1.0):
    """Daily target weights: ``leverage / n_tickers`` in each stock forecast to rise.

    The rest of the capital is cash (negative with leverage above 1, i.e.
    borrowed at no interest).
    """
    signals = predictions.pivot_table(index="date", columns="tic", values="signal").sort_index()
    return leverage * signals / signals.shape[1]
