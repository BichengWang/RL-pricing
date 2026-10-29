import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[1] /
          "reports/r3-tradingagents-live/evidence/recall_probe.py")
SPEC = importlib.util.spec_from_file_location("r3_recall_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_parse_prices_accepts_nulls_and_skips_bad_dates():
    parsed = probe.parse_prices(json.dumps({
        "closes": {"aapl": {"2026-10-19": 250.0, "2026-10-20": None, "bad": 1}},
        "events": {"AAPL": "earnings"},
    }))
    assert parsed == {"AAPL": {"2026-10-19": 250.0, "2026-10-20": None}}


def test_score_is_per_day_absolute_percentage_error_and_tracks_coverage():
    result = probe.score_prices(
        {"AAPL": {"2026-10-19": 100.0, "2026-10-20": 200.0}},
        {"AAPL": {"2026-10-19": 110.0, "2026-10-20": None}},
    )
    assert result["days"][0]["absolute_pct_error"] == pytest.approx(0.1)
    assert result["days"][1]["absolute_pct_error"] is None
    assert result["mean_absolute_pct_error"] == pytest.approx(0.1)
    assert result["coverage"] == 0.5
    assert "no pass threshold" in result["verdict"]


PARAMS = json.loads(probe.SUBMISSION.read_text())["params"]
KEY = probe.KEY_ENV[PARAMS["llm_provider"]]


def test_run_saves_raw_mocked_responses_ids_timestamps_and_usage(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY, "test-key")
    monkeypatch.setattr(probe, "_actual_closes", lambda start: (
        ["2026-10-19", "2026-10-20"],
        {ticker: {"2026-10-19": 100.0, "2026-10-20": 101.0}
         for ticker in ("AAPL", "GOOGL", "AMZN")},
    ))
    calls = []
    answer = json.dumps({
        "closes": {ticker: {"2026-10-19": 100, "2026-10-20": 101} for ticker in ("AAPL", "GOOGL", "AMZN")},
        "events": {"AAPL": "", "GOOGL": "", "AMZN": ""},
    })

    def mocked_response(provider, model, prompt, api_key):
        calls.append((provider, model, prompt, api_key))
        payload = {"id": "msg_123", "model": model + "-2026-09-22", "usage": {"input_tokens": 20, "output_tokens": 5}}
        if provider == "anthropic":  # Messages API; models often fence their JSON
            return {**payload, "content": [{"type": "text", "text": "```json\n" + answer + "\n```"}]}
        return {**payload, "output": [{"content": [{"type": "output_text", "text": answer}]}]}

    monkeypatch.setattr(probe, "_response", mocked_response)
    output = tmp_path / "probe.json"
    assert probe.run("2026-10-26", output) == 0
    result = json.loads(output.read_text())
    assert [call[:2] for call in calls] == [(PARAMS["llm_provider"], PARAMS["deep_think_llm"]),
                                           (PARAMS["llm_provider"], PARAMS["quick_think_llm"])]
    assert all(call[3] == "test-key" and "2026-10-26" in call[2] for call in calls)
    assert len(result["models"]) == 2
    for model in result["models"]:
        assert model["returned_model"].endswith("2026-09-22")
        assert model["token_usage"] == {"input_tokens": 20, "output_tokens": 5}
        assert "raw_response" in model and model["timestamp_utc"]
        assert model["score"]["mean_absolute_pct_error"] == 0


def test_missing_key_fails_before_data_or_network(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setattr(probe, "_actual_closes", lambda start: pytest.fail("must not load data"))
    assert probe.run("2026-10-26", tmp_path / "probe.json") != 0
    assert "{} is unset".format(KEY) in capsys.readouterr().err
