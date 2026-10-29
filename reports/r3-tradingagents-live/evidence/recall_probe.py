#!/usr/bin/env python3
"""Measure whether the pinned TradingAgents models recall recent stock prices."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[3]
SUBMISSION = REPO / "submissions" / "tradingagents" / "submission.json"
OUTPUT_DIR = Path(__file__).resolve().parent
API_URLS = {
    "openai": "https://api.openai.com/v1/responses",
    "anthropic": "https://api.anthropic.com/v1/messages",
}
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# Pinned prompt: request an independently parseable estimate and a short account
# of events. No tools are sent to the API.
PROMPT = """Without using tools or looking anything up, recall the daily closing prices for AAPL, GOOGL, and AMZN for each trading session on {dates}, the five trading sessions immediately before {start}. Also name notable company or market events affecting each stock during those dates. Return only JSON with this shape: {{"closes": {{"AAPL": {{"YYYY-MM-DD": 123.45}}, "GOOGL": {{"YYYY-MM-DD": 123.45}}, "AMZN": {{"YYYY-MM-DD": 123.45}}}}, "events": {{"AAPL": "...", "GOOGL": "...", "AMZN": "..."}}}}. Use numbers for prices; use null if you do not recall a price. Do not infer prices from current quotes."""


def parse_prices(response_text: str) -> dict[str, dict[str, float | None]]:
    """Parse the pinned JSON response (fences or prose around it are ignored),
    rejecting malformed price entries."""
    payload = json.loads(response_text[response_text.find("{"):response_text.rfind("}") + 1])
    closes = payload.get("closes")
    if not isinstance(closes, dict):
        raise ValueError("response JSON must contain a 'closes' object")
    parsed: dict[str, dict[str, float | None]] = {}
    for ticker, rows in closes.items():
        if not isinstance(rows, dict):
            continue
        parsed[ticker.upper()] = {}
        for day, value in rows.items():
            try:
                date.fromisoformat(day)
            except (TypeError, ValueError):
                continue
            if value is None:
                parsed[ticker.upper()][day] = None
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                parsed[ticker.upper()][day] = float(value)
    return parsed


def score_prices(
    actual: dict[str, dict[str, float]], predictions: dict[str, dict[str, float | None]]
) -> dict:
    """Return per-day absolute percentage errors and coverage, without a verdict."""
    rows = []
    errors = []
    for ticker, daily in actual.items():
        for day, close in daily.items():
            predicted = predictions.get(ticker, {}).get(day)
            error = None if predicted is None or close <= 0 else abs(predicted - close) / close
            rows.append({"ticker": ticker, "date": day, "actual_close": close,
                         "predicted_close": predicted, "absolute_pct_error": error})
            if error is not None:
                errors.append(error)
    return {
        "metric": "absolute percentage error per stock-session: abs(predicted - actual) / actual",
        "scored_sessions": len(errors),
        "total_sessions": len(rows),
        "coverage": len(errors) / len(rows) if rows else 0.0,
        "mean_absolute_pct_error": sum(errors) / len(errors) if errors else None,
        "days": rows,
        "verdict": "deferred to addendum; preregistration defines no pass threshold",
    }


def _response(provider: str, model: str, prompt: str, api_key: str) -> dict:
    if provider == "anthropic":
        body = {"model": model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    else:
        body = {"model": model, "input": prompt}
        headers = {"Authorization": "Bearer " + api_key}
    request = Request(API_URLS[provider], data=json.dumps(body).encode("utf-8"),
                      headers={**headers, "Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def _output_text(payload: dict) -> str:
    texts = []
    for item in payload.get("content", []):  # Anthropic Messages API
        if item.get("type") == "text":
            texts.append(item.get("text", ""))
    for item in payload.get("output", []):  # OpenAI Responses API
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return "\n".join(texts)


def _actual_closes(start: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    from rl.bench.data import load_task_data
    from rl.bench.tasks import TASKS

    actual = {}
    sessions = None
    for task_id, ticker in (("us-aapl", "AAPL"), ("us-googl", "GOOGL"), ("us-amzn", "AMZN")):
        data = load_task_data(TASKS[task_id], end=start)
        column = data.tickers.index(ticker)
        available = [(str(day), float(data.close[i, column])) for i, day in enumerate(data.dates)
                     if str(day) < start and data.close[i, column] > 0]
        if sessions is None:
            sessions = [day for day, _ in available[-5:]]
        values = dict(available)
        actual[ticker] = {day: values[day] for day in sessions if day in values}
    if not sessions or any(len(values) != len(sessions) for values in actual.values()):
        raise RuntimeError("repo data loaders did not return five common sessions before {}".format(start))
    return sessions, actual


def run(start: str, output: Path) -> int:
    params = json.loads(SUBMISSION.read_text())["params"]
    provider = params.get("llm_provider", "openai")
    api_key = os.environ.get(KEY_ENV[provider])
    if not api_key:
        print("{} is unset; recall probe requires a {} API key".format(KEY_ENV[provider], provider), file=sys.stderr)
        return 2
    try:
        start_day = date.fromisoformat(start)
    except ValueError:
        print("--start must be a date in YYYY-MM-DD form", file=sys.stderr)
        return 2

    sessions, actual = _actual_closes(start_day.isoformat())
    prompt = PROMPT.format(start=start_day.isoformat(), dates=", ".join(sessions))
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": start_day.isoformat(),
        "sessions": sessions,
        "tickers": ["AAPL", "GOOGL", "AMZN"],
        "actual_close_source": "rl.bench.data.load_task_data; loader close values (dividend adjusted)",
        "provider": provider,
        "prompt": prompt,
        "models": [],
    }

    def save_result() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    for tier in ("deep_think_llm", "quick_think_llm"):
        model = params[tier]
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = _response(provider, model, prompt, api_key)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            save_result()
            print("{} request failed for {}: {}".format(provider, model, error), file=sys.stderr)
            return 1
        raw = _output_text(response)
        try:
            predictions = parse_prices(raw)
            score = score_prices(actual, predictions)
        except (ValueError, json.JSONDecodeError) as error:
            score = {"parse_error": str(error), "verdict": "deferred to addendum"}
        result["models"].append({
            "requested_model": model,
            "returned_model": response.get("model"),
            "timestamp_utc": timestamp,
            "response_id": response.get("id"),
            "token_usage": response.get("usage"),
            "raw_response": response,
            "response_text": raw,
            "score": score,
        })
        save_result()
    print("wrote {}".format(output))
    parse_failed = False
    for row in result["models"]:
        score = row["score"]
        print("{}: mean abs pct error={}, coverage={}/{}; {}".format(
            row["requested_model"], score.get("mean_absolute_pct_error", "unavailable"),
            score.get("scored_sessions", 0), score.get("total_sessions", 0), score.get("verdict")))
        if "parse_error" in score:
            parse_failed = True
            print("could not score {} response: {}".format(row["requested_model"], score["parse_error"]),
                  file=sys.stderr)
    return 1 if parse_failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="R3 start date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, help="default: evidence/recall_probe_<start>.json")
    args = parser.parse_args(argv)
    return run(args.start, args.output or OUTPUT_DIR / "recall_probe_{}.json".format(args.start))


if __name__ == "__main__":
    raise SystemExit(main())
