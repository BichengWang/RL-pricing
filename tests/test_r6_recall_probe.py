"""The R6 recall probe's parsing, scoring and spend accounting (offline)."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "reports/r6-llm-portfolio-live/evidence/recall_probe.py"
SPEC = importlib.util.spec_from_file_location("r6_recall_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

TICKERS = ["XLB", "XLE"]


def test_parse_prices_accepts_nulls_and_skips_bad_dates():
    parsed = probe.parse_prices(json.dumps({
        "closes": {"xlb": {"2026-10-19": 88.0, "2026-10-20": None, "bad": 1}},
        "events": {"XLB": "energy"},
    }), TICKERS)
    assert parsed == {"XLB": {"2026-10-19": 88.0, "2026-10-20": None}}


def test_score_is_per_day_absolute_percentage_error_and_tracks_coverage():
    result = probe.score_prices(
        {"XLB": {"2026-10-19": 100.0, "2026-10-20": 200.0}},
        {"XLB": {"2026-10-19": 110.0, "2026-10-20": None}},
    )
    assert result["days"][0]["absolute_pct_error"] == 0.1
    assert result["days"][1]["absolute_pct_error"] is None
    assert result["mean_absolute_pct_error"] == 0.1
    assert result["coverage"] == 0.5
    assert "no pass threshold" in result["verdict"]


def test_run_writes_the_response_usage_and_cost(tmp_path, monkeypatch):
    monkeypatch.setenv(probe.KEY_ENV, "test-key")
    monkeypatch.setattr(probe, "_actual_closes", lambda start: (
        ["2026-10-19", "2026-10-20"],
        {ticker: {"2026-10-19": 100.0, "2026-10-20": 101.0} for ticker in TICKERS},
    ))
    answer = json.dumps({
        "closes": {ticker: {"2026-10-19": 100, "2026-10-20": 101} for ticker in TICKERS},
        "events": {ticker: "" for ticker in TICKERS},
    })
    monkeypatch.setattr(probe, "_response", lambda model, prompt, key: {
        "id": "msg_1", "model": model, "usage": {"input_tokens": 100, "output_tokens": 50},
        "content": [{"type": "text", "text": answer}],
    })
    output = tmp_path / "probe.json"

    assert probe.run("2026-10-21", output) == 0
    result = json.loads(output.read_text())
    assert result["requested_model"] == probe.MODEL
    assert result["thinking"] == {"type": "between_tools"}
    assert result["max_tokens"] == 8192
    assert result["score"]["coverage"] == 1.0
    assert result["cost_usd"] == (100 * 2.0 + 50 * 10.0) / 1e6
    assert result["cost_usd"] < probe.MAX_SPEND_USD


def test_request_body_omits_temperature_and_pins_thinking(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"model": probe.MODEL, "usage": {}, "content": []}).encode()

    def capture(request, timeout):
        seen.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(probe, "urlopen", capture)
    probe._response(probe.MODEL, "prompt", "key")

    assert "temperature" not in seen
    assert seen["thinking"] == {"type": "between_tools"}
    assert seen["max_tokens"] == probe.MAX_TOKENS


def test_unadjusted_closes_read_the_yahoo_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_PRICING_DATA", str(tmp_path))
    (tmp_path / "yahoo").mkdir()
    (tmp_path / "yahoo" / "XLB.csv").write_text(
        "date,open,high,low,close,volume,dividends\n"
        "2026-10-19,87.0,89.0,86.0,88.0,1000,0.0\n"
        "2026-10-20,88.0,90.0,87.0,89.5,1000,0.0\n"
    )
    assert probe._unadjusted_closes("XLB", ["2026-10-19", "2026-10-20"], "2026-10-21") == {
        "2026-10-19": 88.0, "2026-10-20": 89.5,
    }


def test_run_without_a_key_is_pending(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(probe.KEY_ENV, raising=False)
    assert probe.run("2026-10-21", tmp_path / "probe.json") == 2
    assert "pending" in capsys.readouterr().err
    assert not (tmp_path / "probe.json").exists()
