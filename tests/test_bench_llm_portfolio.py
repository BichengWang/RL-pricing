"""The LLM portfolio policy and its ledger entries (no network, injected client)."""

import datetime as dt
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rl.bench import cli, ledger, runner
from rl.bench.info import InfoItem, InfoPanel, InfoView, session_close
from rl.bench.market import MarketData, MarketView
from rl.bench.policies import llm_portfolio as lp
from rl.bench.policies.llm_portfolio import InvalidDecision, LLMPortfolioPolicy
from rl.bench.submission import Submission
from rl.bench.tasks import get_task

TICKERS = get_task("us-sectors").tickers
NOW = dt.datetime(2026, 11, 2, 12, 0, tzinfo=dt.timezone.utc)
RUN = {"repository": "owner/repo", "run_id": "9000", "run_attempt": "1", "started_at": "2026-11-02T11:00:00+00:00"}


def panel(n=260, tickers=TICKERS, seed=0, start="2025-01-02"):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (n, len(tickers))), axis=0))
    dates = pd.bdate_range(start, periods=n).to_numpy(dtype="datetime64[D]")
    return MarketData(
        dates=dates, tickers=tuple(tickers), open=closes, high=closes, low=closes, close=closes,
        volume=np.ones_like(closes),
    )


def valid_payload(tickers=TICKERS, weight=None):
    weight = 1.0 / len(tickers) if weight is None else weight
    return {**{ticker: weight for ticker in tickers}, "cash": max(0.0, 1.0 - weight * len(tickers)), "rationale": "test"}


class Client:
    """The smallest Anthropic Messages client the policy uses; responses are scripted."""

    def __init__(self, responses):
        self.messages = self
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload, usage = self.responses[len(self.calls) - 1]
        if isinstance(payload, str):
            content = [SimpleNamespace(type="text", text=payload)]
        else:
            content = [SimpleNamespace(type="tool_use", name=lp._TOOL_NAME, input=payload)]
        return SimpleNamespace(
            content=content,
            usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
            model=lp.PINNED_MODEL,
        )


def view(history=None, weights=None, info=None):
    history = panel() if history is None else history
    weights = np.full(len(history.tickers), 1.0 / len(history.tickers)) if weights is None else weights
    return MarketView(history, weights, info=info or InfoView())


def policy(*payloads, usage=(1000, 100)):
    responses = [(payloads[i], usage if len(payloads) == 1 else (usage[0] * (i + 1), usage[1] * (i + 1)))
                 for i in range(len(payloads))]
    return LLMPortfolioPolicy(client=Client(responses))


def test_policy_is_pinned_and_declares_its_worst_case():
    assert lp.PINNED_MODEL == "claude-sonnet-5-5"
    assert lp.PROMPT_SHA256 == hashlib.sha256(
        (lp.SYSTEM_PROMPT + "\n\n" + lp.USER_TEMPLATE).encode()
    ).hexdigest()
    expected = 2 * (lp.MAX_INPUT_TOKENS * 2.0 + lp.MAX_OUTPUT_TOKENS * 10.0) / 1e6
    assert LLMPortfolioPolicy.max_cost_per_decision_usd == pytest.approx(math.ceil(expected * 100) / 100)
    with pytest.raises(ValueError, match="pinned"):
        LLMPortfolioPolicy(model="claude-sonnet-5")
    with pytest.raises(ValueError, match="Anthropic"):
        LLMPortfolioPolicy(llm_provider="openai")


def test_valid_tool_output_becomes_weights_and_records_provenance():
    llm = policy(valid_payload(), usage=(2000, 500))
    weights = llm.act(view())

    assert weights == pytest.approx(np.full(len(TICKERS), 1 / len(TICKERS)))
    assert llm.last_cost_usd == pytest.approx((2000 * 2.0 + 500 * 10.0) / 1e6)
    sent = llm._client.calls[0]
    assert sent["model"] == lp.PINNED_MODEL
    assert "temperature" not in sent
    assert sent["thinking"] == {"type": "between_tools"}
    assert sent["max_tokens"] == lp.MAX_OUTPUT_TOKENS == 8192
    assert sent["tool_choice"] == {"type": "auto"}
    assert sent["tools"][0]["strict"] is True
    assert set(sent["tools"][0]["input_schema"]["properties"]) == {*TICKERS, "cash", "rationale"}
    provenance = llm.ledger_inputs
    assert provenance["prompt_sha256"] == lp.PROMPT_SHA256
    assert provenance["input_sha256"] == hashlib.sha256(
        sent["messages"][0]["content"].encode()
    ).hexdigest()
    assert provenance["token_usage"] == {
        "input_tokens": 2000, "output_tokens": 500,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }
    assert provenance["response_json"] == valid_payload()
    assert provenance["target_weights"]["XLK"] == pytest.approx(1 / len(TICKERS))


def test_invalid_output_twice_raises_after_two_calls_and_bills_both():
    llm = policy("not json", "still not json")
    with pytest.raises(InvalidDecision, match="after 2 attempt"):
        llm.act(view())
    assert len(llm._client.calls) == 2
    assert llm.last_cost_usd == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1e6
                                              + (2000 * 2.0 + 200 * 10.0) / 1e6)


def test_one_retry_then_a_valid_answer_is_used():
    llm = policy({"XLK": 2.0, "cash": 0.0}, valid_payload(), usage=(1000, 100))
    weights = llm.act(view())
    assert weights.sum() == pytest.approx(1.0)
    assert len(llm._client.calls) == 2
    assert llm.ledger_inputs["attempts"] == 2


@pytest.mark.parametrize("payload", [
    {**{t: 0.1 for t in TICKERS[:-1]}, "cash": 0.1},                    # missing ticker
    {**{t: 0.1 for t in TICKERS}, "SPY": 0.0, "cash": 0.1},            # unknown ticker
    {**{t: 0.1 for t in TICKERS}, "XLK": -0.1, "cash": 0.1},          # negative
    {**{t: 0.2 for t in TICKERS}, "cash": 0.1},                        # sums above one
    {**{t: 0.1 for t in TICKERS}, "cash": 1.5},                        # bad cash
    {**{t: 0.1 for t in TICKERS}, "XLB": True, "cash": 0.1},          # boolean
    {**{t: 0.1 for t in TICKERS}, "XLE": float("nan"), "cash": 0.1},  # not finite
])
def test_invalid_weights_are_rejected(payload):
    llm = policy(payload, payload)
    with pytest.raises(InvalidDecision, match="invalid output after 2 attempt"):
        llm.act(view())


def test_table_and_prompt_use_no_data_after_the_decision_close():
    full = panel()
    cutoff = 250  # the decision is made at this close
    history = full.rows(0, cutoff + 1)
    clean = policy(valid_payload())
    clean.act(view(history))
    clean_prompt = clean._client.calls[0]["messages"][0]["content"]

    arrays = {name: getattr(full, name).copy() for name in ("open", "high", "low", "close", "volume")}
    for name in ("open", "high", "low", "close"):
        arrays[name][cutoff + 1 :] = 1e9
    poisoned = MarketData(dates=full.dates.copy(), tickers=full.tickers, **arrays)
    later = policy(valid_payload())
    later.act(view(poisoned.rows(0, cutoff + 1)))
    assert later._client.calls[0]["messages"][0]["content"] == clean_prompt
    assert later.ledger_inputs["input_sha256"] == clean.ledger_inputs["input_sha256"]


def test_information_items_after_the_decision_close_cannot_reach_the_prompt():
    history = panel()
    day = str(history.dates[-1])
    close = session_close(day)

    def item(name, published):
        text = name
        return InfoItem(
            id=name, source="test", tickers=("XLK",), published_at=published, title=name,
            text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    before = item("before", close - dt.timedelta(hours=1))
    after = item("after", close + dt.timedelta(hours=1))
    with_before = policy(valid_payload())
    with_before.act(view(history, info=InfoPanel([before]).at_session(day)))
    with_both = policy(valid_payload())
    with_both.act(view(history, info=InfoPanel([before, after]).at_session(day)))
    assert with_both._client.calls[0]["messages"][0]["content"] == with_before._client.calls[0]["messages"][0]["content"]
    assert "[XLK] after" not in with_both._client.calls[0]["messages"][0]["content"]
    assert with_both.ledger_inputs["input_sha256"] == with_before.ledger_inputs["input_sha256"]


def make_submission(root, info_sources=None, **params):
    directory = root / "llm-portfolio"
    directory.mkdir(parents=True)
    spec = {
        "name": "LLM portfolio",
        "kind": "llm",
        "policy": "rl.bench.policies.llm_portfolio:LLMPortfolioPolicy",
        "params": {"model": lp.PINNED_MODEL, "llm_provider": "anthropic", **params},
        "ledger": ["us-sectors"],
        "model_cutoffs": [{
            "model": lp.PINNED_MODEL, "training_cutoff": lp.MODEL_CUTOFF,
            "source_url": "https://platform.claude.com/docs/en/about-claude/models/overview",
        }],
    }
    if info_sources:
        spec["info_sources"] = info_sources
    (directory / "submission.json").write_text(json.dumps(spec))
    return Submission.load(directory)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "FRED_API_KEY", "SEC_EDGAR_USER_AGENT", ledger.BUDGET_ENV):
        monkeypatch.delenv(name, raising=False)


def test_ledger_records_a_valid_us_sectors_decision_without_fred_or_edgar(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    sub = make_submission(tmp_path, monthly_budget_usd=15.0)
    llm = policy(valid_payload(), usage=(2000, 500))
    monkeypatch.setattr(runner, "build_policy", lambda *_: llm)

    path = ledger.record_decision(get_task("us-sectors"), sub, data=panel(), now=NOW, run=RUN)

    entry = json.loads(path.read_text())
    assert entry["schema"] == 2 and entry["task"] == "us-sectors"
    assert entry["hold"] is False and set(entry["weights"]) == set(TICKERS)
    assert sum(entry["weights"].values()) == pytest.approx(1.0)
    assert entry["cost_usd"] == pytest.approx((2000 * 2.0 + 500 * 10.0) / 1e6)
    assert entry["reserved_cost_usd"] == lp.LLMPortfolioPolicy.max_cost_per_decision_usd
    assert entry["inputs"]["model"] == lp.PINNED_MODEL
    assert entry["inputs"]["prompt_sha256"] == lp.PROMPT_SHA256
    assert len(entry["inputs"]["input_sha256"]) == 64


def test_ledger_records_an_invalid_twice_response_as_a_hold_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    sub = make_submission(tmp_path, monthly_budget_usd=15.0)
    llm = policy("bad", "worse")
    monkeypatch.setattr(runner, "build_policy", lambda *_: llm)

    path = ledger.record_decision(get_task("us-sectors"), sub, data=panel(), now=NOW, run=RUN)

    entry = json.loads(path.read_text())
    assert entry["hold"] is True and entry["weights"] is None
    assert entry["reason"].startswith("error: InvalidDecision: invalid output after 2 attempt")
    assert entry["cost_usd"] > 0
    assert entry["reserved_cost_usd"] == lp.LLMPortfolioPolicy.max_cost_per_decision_usd
    assert entry["inputs"]["model"] == lp.PINNED_MODEL
    assert entry["inputs"]["prompt_sha256"] == lp.PROMPT_SHA256
    assert len(entry["inputs"]["input_sha256"]) == 64
    assert "no JSON object" in entry["inputs"]["error"]


def test_prompt_budget_truncates_information_items():
    history = panel()
    day = str(history.dates[-1])
    close = session_close(day)
    items = [
        InfoItem(
            id="item-{}".format(n), source="test", tickers=("XLK",),
            published_at=close - dt.timedelta(minutes=n + 1), title="long item",
            text="x" * 5000, text_sha256=hashlib.sha256(("x" * 5000).encode()).hexdigest(),
        )
        for n in range(100)
    ]
    llm = policy(valid_payload())
    llm.act(view(history, info=InfoPanel(items).at_session(day)))

    prompt = llm._client.calls[0]["messages"][0]["content"]
    assert len(prompt) <= lp.MAX_INPUT_CHARS
    assert llm.ledger_inputs["input_chars"] == len(prompt)
    assert "more items omitted" in prompt


def test_ledger_loads_only_information_published_before_the_decision_close(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.setenv("RL_PRICING_DATA", str(tmp_path / "data"))
    sub = make_submission(
        tmp_path, monthly_budget_usd=15.0,
        info_sources=[{"source": "jsonl", "path": "items.jsonl"}],
    )
    history = panel()
    day = str(history.dates[-1])
    close = session_close(day)

    def item(name, published):
        text = name
        return InfoItem(
            id=name, source="test", tickers=("XLK",), published_at=published, title=name,
            text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    before, after = item("before", close - dt.timedelta(hours=1)), item("after", close + dt.timedelta(hours=1))
    (sub.directory / "items.jsonl").write_text(
        json.dumps(before.to_dict()) + "\n" + json.dumps(after.to_dict()) + "\n"
    )
    llm = policy(valid_payload())
    monkeypatch.setattr(runner, "build_policy", lambda *_: llm)

    path = ledger.record_decision(get_task("us-sectors"), sub, data=history, now=NOW, run=RUN)
    entry = json.loads(path.read_text())
    direct = policy(valid_payload())
    direct.act(view(history, weights=np.zeros(len(TICKERS)), info=InfoPanel([before]).at_session(day)))

    assert entry["inputs"]["input_sha256"] == direct.ledger_inputs["input_sha256"]
    assert "[XLK] after" not in direct._client.calls[0]["messages"][0]["content"]


def test_a_pre_call_stale_hold_costs_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    sub = make_submission(tmp_path, monthly_budget_usd=15.0, max_age_days=0)
    llm = LLMPortfolioPolicy(client=Client([(valid_payload(), (1000, 100))]), max_age_days=0)
    monkeypatch.setattr(runner, "build_policy", lambda *_: llm)

    path = ledger.record_decision(get_task("us-sectors"), sub, data=panel(), now=NOW, run=RUN)

    entry = json.loads(path.read_text())
    assert entry["hold"] is True and entry["reason"].startswith("error: RuntimeError")
    assert entry["cost_usd"] == 0.0
    assert entry["inputs"]["cost_basis"] == "no_call"
    assert entry["reserved_cost_usd"] == lp.LLMPortfolioPolicy.max_cost_per_decision_usd


def test_ledger_entry_is_unaffected_by_rows_after_the_decision_close(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    sub = make_submission(tmp_path, monthly_budget_usd=15.0)
    full = panel()
    cutoff = 250
    history = full.rows(0, cutoff + 1)
    llm = policy(valid_payload())
    monkeypatch.setattr(runner, "build_policy", lambda *_: llm)

    path = ledger.record_decision(get_task("us-sectors"), sub, data=history, now=NOW, run=RUN)
    entry = json.loads(path.read_text())

    arrays = {name: getattr(full, name).copy() for name in ("open", "high", "low", "close", "volume")}
    for name in ("open", "high", "low", "close"):
        arrays[name][cutoff + 1 :] = 1e9
    poisoned = MarketData(dates=full.dates.copy(), tickers=full.tickers, **arrays)
    direct = policy(valid_payload())
    direct.act(view(poisoned.rows(0, cutoff + 1), weights=np.zeros(len(TICKERS))))

    assert entry["data_sha256"] == history.sha256()
    assert entry["inputs"]["input_sha256"] == direct.ledger_inputs["input_sha256"]


def test_ledger_record_command_writes_a_us_sectors_entry_with_the_fake_client(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    sub = make_submission(tmp_path, monthly_budget_usd=15.0)
    llm = policy(valid_payload(), usage=(2000, 500))
    monkeypatch.setattr(runner, "_data_for", lambda task, data, **_: panel())
    monkeypatch.setattr(runner, "build_policy", lambda *_: llm)

    assert cli.main(["ledger", "record", "--submissions-root", str(tmp_path), "--budget", "15"]) == 0

    [path] = list((sub.directory / "ledger" / "us-sectors").glob("*.json"))
    entry = json.loads(path.read_text())
    assert entry["schema"] == 2 and entry["task"] == "us-sectors" and entry["submission"] == "llm-portfolio"
    assert entry["hold"] is False and set(entry["weights"]) == set(TICKERS)
    assert "Recorded" in capsys.readouterr().out


def test_per_entrant_monthly_budget_is_hard_and_applies_without_the_environment_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    sub = make_submission(tmp_path, monthly_budget_usd=0.255)
    llm = policy(valid_payload(), usage=(2000, 500))  # $0.009
    monkeypatch.setattr(runner, "build_policy", lambda *_: llm)
    data = panel()

    ledger.record_decision(
        get_task("us-sectors"), sub, data=data.rows(0, len(data) - 1), now=NOW, run=RUN
    )
    # $0.009 spent + the $0.25 reservation does not fit the $0.255 cap: the second call is refused.
    with pytest.raises(ValueError, match="budget cap"):
        ledger.record_decision(get_task("us-sectors"), sub, data=data, now=NOW, run=RUN)
    assert len(llm._client.calls) == 1
