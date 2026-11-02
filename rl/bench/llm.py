"""Per-asset LLM score features carried by :mod:`rl.bench.info` items.

Two model-produced fields are read: ``llm_sentiment`` (a 1-5 recommendation)
and ``llm_risk`` (a 1-5 risk score). At a decision close only items published
strictly before that close count, so :func:`score_panel` and
:func:`update_scores` apply the same rule during training and at decision time.
A missing sentiment is 0 and a missing risk is 3 (neutral), as the authors'
environments read them.

The two uses follow FinRL-DeepSeek (Benhenda, arXiv 2502.07393, §3):
:func:`apply_sentiment` scales the change of a target weight by the score and
the direction of the trade, and :func:`risk_factors` returns the
holdings-weighted risk multiplier used by CVaR-PPO.
"""

from __future__ import annotations

import numpy as np

from rl.bench.info import InfoPanel, InfoView, session_close

SENTIMENT_FIELD = "llm_sentiment"
RISK_FIELD = "llm_risk"
FIELDS = (SENTIMENT_FIELD, RISK_FIELD)
MISSING_SCORES = (0.0, 3.0)
#: Weakest effect below which a score no longer shifts the target.
_NEUTRAL = 3.0


def empty_scores(n_assets: int) -> np.ndarray:
    """The missing values for ``n_assets``; shape ``(n_assets, 2)``."""
    scores = np.empty((n_assets, 2), dtype=float)
    scores[:, 0], scores[:, 1] = MISSING_SCORES
    return scores


def _value(field):
    return float(field.value if hasattr(field, "value") else field)


def update_scores(scores: np.ndarray, info: InfoView, tickers, start: int = 0) -> None:
    """Apply items from position ``start`` on to a live per-asset score array.

    ``info`` is already point-in-time; ``start`` lets a policy that acts once
    per session read only the items published since its last decision.
    """
    positions = {ticker: i for i, ticker in enumerate(tickers)}
    for item in info.items(start=start):
        for ticker in item.tickers:
            column = positions.get(ticker)
            if column is None:
                continue
            for k, name in enumerate(FIELDS):
                if name in item.fields:
                    scores[column, k] = _value(item.fields[name])


def score_panel(panel: InfoPanel, dates, tickers) -> np.ndarray:
    """Per-session scores visible at each decision close, shape ``(T, N, 2)``.

    ``dates`` are the sessions the caller will decide on; row ``t`` holds each
    asset's most recent item published strictly before the close of ``dates[t]``.
    """
    dates = tuple(dates)
    tickers = tuple(tickers)
    positions = {ticker: column for column, ticker in enumerate(tickers)}
    closes = np.array([session_close(day).timestamp() for day in dates], dtype=float)
    series = {}
    for item in panel.items:
        moment = item.published_at.timestamp()
        for ticker in item.tickers:
            column = positions.get(ticker)
            if column is None:
                continue
            for k, name in enumerate(FIELDS):
                if name in item.fields:
                    times, values = series.setdefault((column, k), ([], []))
                    times.append(moment)
                    values.append(_value(item.fields[name]))
    scores = np.empty((len(dates), len(tickers), 2), dtype=float)
    scores[..., 0], scores[..., 1] = MISSING_SCORES
    for (column, k), (times, values) in series.items():
        index = np.searchsorted(np.array(times), closes, side="left") - 1
        seen = index >= 0
        scores[seen, column, k] = np.array(values)[index[seen]]
    return scores


def apply_sentiment(target, current, sentiment, strength: float) -> np.ndarray:
    """Scale each asset's weight change by its score, then project to long-only weights.

    A change matched by the score (5 buying, 1 selling) is multiplied by
    ``1 + strength``, a half-matched one (4 buying, 2 selling) by
    ``1 + strength / 2``, a half-opposed one by ``1 - strength / 2`` and an
    opposed one by ``1 - strength``. A neutral or missing score leaves the
    change unchanged. Negative targets are clipped to zero and a total above
    one is scaled back to one, the long-only budget the engine accepts.
    """
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)
    sentiment = np.asarray(sentiment, dtype=float)
    change = target - current
    factor = np.ones(np.broadcast_shapes(change.shape, sentiment.shape), dtype=float)
    for score in (5, 4, 2, 1):
        match = np.isclose(sentiment, score)
        if not match.any():
            continue
        if score >= 4:
            signed = np.where(score == 5, strength, strength / 2) * np.sign(change)
        else:
            signed = -np.where(score == 2, strength / 2, strength) * np.sign(change)
        factor = np.where(match, 1.0 + signed, factor)
    target = np.maximum(current + change * factor, 0.0)
    total = target.sum(axis=-1, keepdims=True)
    return np.where(total > 1.0, target / np.where(total > 1.0, total, 1.0), target)


def risk_factors(weights, risk, strength: float) -> np.ndarray:
    """Holdings-weighted mean of ``1 + strength * (risk - 3) / 2`` over invested assets.

    ``weights`` and ``risk`` have the same shape ``(..., n_assets)``; a
    portfolio with no invested assets has multiplier 1.
    """
    weights = np.asarray(weights, dtype=float)
    risk = np.asarray(risk, dtype=float)
    invested = weights > 0
    total = np.where(invested, weights, 0.0).sum(axis=-1)
    weighted = np.where(invested, weights * (1.0 + strength * (risk - _NEUTRAL) / 2.0), 0.0).sum(axis=-1)
    return np.where(total > 0, weighted / np.where(total > 0, total, 1.0), 1.0)
