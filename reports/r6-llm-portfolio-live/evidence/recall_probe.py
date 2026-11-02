#!/usr/bin/env python3
"""Measure whether the pinned R6 model recalls recent sector-ETF closes.

The probe asks the model, without tools, for the daily closes of the nine
``us-sectors`` ETFs on the five sessions before ``--start`` and for notable
events in those days. The answers are scored as absolute percentage error per
ETF-session; coverage is the share of sessions the model is willing to state.
There is no pass threshold: the pre-registration uses the measurement to
document where the model's knowledge ends relative to the live window.

No key is stored here. Set ``ANTHROPIC_API_KEY`` to run it; the script exits 2
without one. One call with at most 2,048 output tokens costs well under $1 at
the published prices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent
API_URL = "https://api.anthropic.com/v1/messages"
KEY_ENV = "ANTHROPIC_API_KEY"
MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8192
THINKING = {"type": "between_tools"}
MAX_SPEND_USD = 1.0
PRICES_USD_PER_MTOK = {"input": 2.00, "output": 10.00}

PROMPT = """Without using tools or looking anything up, recall the daily closing prices for {tickers} for each trading session on {dates}, the five trading sessions immediately before {start}. Also name notable sector or market events during those dates. Return only JSON with this shape: {{"closes": {{{example}}}, "events": {{{events}}}}}. Use numbers for prices; use null if you do not recall a price. Do not infer prices from current quotes."""


def parse_prices(response_text: str, tickers) -> dict[str, dict[str, float | None]]:
    """Parse the pinned JSON response (fences or prose around it are ignored)."""
    start, end = response_text.find("{"), response_text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the response")
    payload = json.loads(response_text[start : end + 1])
    closes = payload.get("closes")
    if not isinstance(closes, dict):
        raise ValueError("response has no 'closes' object")
    parsed: dict[str, dict[str, float | None]] = {}
    by_upper = {str(key).upper(): value for key, value in closes.items()}
    for ticker in tickers:
        rows = by_upper.get(ticker.upper())
        if not isinstance(rows, dict):
            continue
        parsed[ticker] = {}
        for day, value in rows.items():
            try:
                date.fromisoformat(day)
            except (TypeError, ValueError):
                continue
            if value is None:
                parsed[ticker][day] = None
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                parsed[ticker][day] = float(value)
    return parsed


def score_prices(
    actual: dict[str, dict[str, float]], predictions: dict[str, dict[str, float | None]]
) -> dict:
    """Absolute percentage error per ETF-session and coverage; no verdict."""
    rows, errors = [], []
    for ticker, daily in actual.items():
        for day, close in daily.items():
            predicted = predictions.get(ticker, {}).get(day)
            error = None if predicted is None or close <= 0 else abs(predicted - close) / close
            rows.append({
                "ticker": ticker, "date": day, "actual_close": close,
                "predicted_close": predicted, "absolute_pct_error": error,
            })
            if error is not None:
                errors.append(error)
    return {
        "metric": "absolute percentage error per ETF-session: abs(predicted - actual) / actual",
        "scored_sessions": len(errors),
        "total_sessions": len(rows),
        "coverage": len(errors) / len(rows) if rows else 0.0,
        "mean_absolute_pct_error": sum(errors) / len(errors) if errors else None,
        "days": rows,
        "verdict": "no pass threshold in the pre-registration; a measurement of recalled knowledge",
    }


def _response(model: str, prompt: str, api_key: str) -> dict:
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def _output_text(payload: dict) -> str:
    return "\n".join(
        item.get("text", "") for item in payload.get("content", []) if item.get("type") == "text"
    )


def _unadjusted_closes(ticker: str, sessions: list[str], start: str) -> dict[str, float]:
    """The quoted (unadjusted) close on each session, from the Yahoo cache or a download."""
    from rl.bench.data import DATA_ENV, DEFAULT_CACHE
    from rl.marketdata.yahoodownloader import YahooDownloader

    def closes(frame):
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        return dict(zip(frame["date"], frame["close"]))

    cache = Path(os.environ.get(DATA_ENV) or DEFAULT_CACHE) / "yahoo" / "{}.csv".format(ticker)
    values = closes(pd.read_csv(cache)) if cache.exists() else {}
    if not all(day in values for day in sessions):
        first = (date.fromisoformat(sessions[0]) - timedelta(days=14)).isoformat()
        values = closes(YahooDownloader(first, start, [ticker]).fetch_unadjusted())
    return {day: float(values[day]) for day in sessions if day in values}


def _actual_closes(start: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    from rl.bench.data import load_task_data
    from rl.bench.tasks import get_task

    task = get_task("us-sectors")
    data = load_task_data(task, end=start)  # sessions only; its closes are dividend-adjusted
    if len(data) < 5:
        raise RuntimeError("fewer than five sessions before {}".format(start))
    sessions = [str(day) for day in data.dates[-5:]]
    actual = {ticker: _unadjusted_closes(ticker, sessions, start) for ticker in task.tickers}
    if any(len(values) != len(sessions) for values in actual.values()):
        raise RuntimeError("the nine ETFs do not share five priced sessions before {}".format(start))
    return sessions, actual


def _cost(usage: dict) -> float:
    return (
        int(usage.get("input_tokens") or 0) * PRICES_USD_PER_MTOK["input"]
        + int(usage.get("output_tokens") or 0) * PRICES_USD_PER_MTOK["output"]
    ) / 1e6


def run(start: str, output: Path) -> int:
    model = MODEL
    api_key = os.environ.get(KEY_ENV, "").strip()
    if not api_key:
        print("{} is unset; the recall probe is pending".format(KEY_ENV), file=sys.stderr)
        return 2
    try:
        start_day = date.fromisoformat(start)
    except ValueError:
        print("--start must be a date in YYYY-MM-DD form", file=sys.stderr)
        return 2

    sessions, actual = _actual_closes(start_day.isoformat())
    tickers = list(actual)
    example = ", ".join('"{}": {{"YYYY-MM-DD": 123.45}}'.format(ticker) for ticker in tickers)
    events = ", ".join('"{}": "..."'.format(ticker) for ticker in tickers)
    prompt = PROMPT.format(
        tickers=", ".join(tickers), dates=", ".join(sessions), start=start_day.isoformat(),
        example=example, events=events,
    )
    prompt_tokens = len(prompt) // 4 + 1
    if (prompt_tokens * PRICES_USD_PER_MTOK["input"] + MAX_TOKENS * PRICES_USD_PER_MTOK["output"]) / 1e6 > MAX_SPEND_USD:
        print("the probe could exceed ${:.2f}; refusing to run".format(MAX_SPEND_USD), file=sys.stderr)
        return 2

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": start_day.isoformat(),
        "sessions": sessions,
        "tickers": tickers,
        "actual_close_source": "Yahoo unadjusted close from datasets/bench/yahoo/<ticker>.csv or "
                               "YahooDownloader.fetch_unadjusted; rl.bench.data is used only for the sessions",
        "provider": "anthropic",
        "requested_model": model,
        "thinking": THINKING,
        "max_tokens": MAX_TOKENS,
        "prompt": prompt,
        "max_spend_usd": MAX_SPEND_USD,
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        response = _response(model, prompt, api_key)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        result["error"] = "{}: {}".format(type(error).__name__, error)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("request failed: {}".format(error), file=sys.stderr)
        return 1
    raw = _output_text(response)
    usage = response.get("usage") or {}
    result.update(
        timestamp_utc=timestamp,
        returned_model=response.get("model"),
        response_id=response.get("id"),
        token_usage=usage,
        cost_usd=_cost(usage),
        raw_response=response,
        response_text=raw,
    )
    try:
        result["score"] = score_prices(actual, parse_prices(raw, tickers))
    except (ValueError, json.JSONDecodeError) as error:
        result["score"] = {"parse_error": str(error)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("could not score the response: {}".format(error), file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "{}: coverage {}/{}, mean absolute pct error {}, cost ${:.4f}; wrote {}".format(
            model, result["score"]["scored_sessions"], result["score"]["total_sessions"],
            result["score"]["mean_absolute_pct_error"], result["cost_usd"], output,
        )
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first live decision date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, help="default: evidence/recall_probe_<start>.json")
    args = parser.parse_args(argv)
    return run(args.start, args.output or OUTPUT_DIR / "recall_probe_{}.json".format(args.start))


if __name__ == "__main__":
    raise SystemExit(main())
