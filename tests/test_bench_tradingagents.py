"""TradingAgents ledger policy: rating mapping, cost, provenance and construction.

No test calls a model: graphs are fakes, and the optional ``tradingagents``
package is replaced by stand-in modules where graph construction matters.
"""

import hashlib
import json
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl.bench.engine import simulate
from rl.bench.market import MarketData, MarketView
from rl.bench.policies import tradingagents as ta
from rl.bench.policies.tradingagents import (
    DEFAULT_PRICES_USD_PER_MTOK,
    TradingAgentsPolicy,
    UsageMeter,
    price_of,
    usage_cost,
)
from rl.bench.submission import Submission
from rl.bench.tasks import TASKS

SUBMISSION = Path(__file__).resolve().parents[1] / "submissions" / "tradingagents"


def panel(closes, tickers=("AAPL",), dates=None, start="2024-01-02"):
    closes = np.asarray(closes, dtype=float).reshape(len(closes), -1)
    if dates is None:
        dates = pd.bdate_range(start, periods=len(closes)).to_numpy(dtype="datetime64[D]")
    return MarketData(
        dates=dates,
        tickers=tuple(tickers),
        open=closes,
        high=closes,
        low=closes,
        close=closes,
        volume=np.ones_like(closes),
    )


class ScriptedGraph:
    """Returns the scripted ratings in order and records every call."""

    def __init__(self, ratings, decision="**Rating**: Buy\nBecause."):
        self.ratings = list(ratings)
        self.decision = decision
        self.calls = []

    def propagate(self, ticker, trade_date):
        self.calls.append((ticker, trade_date))
        return {"final_trade_decision": self.decision}, self.ratings[len(self.calls) - 1]


def test_ratings_map_to_long_flat_or_hold_through_the_engine():
    ratings = ["Buy", "Hold", "Sell", "Hold", "Overweight", "Underweight", "Buy", "REVIEW"]
    data = panel(100 * 1.001 ** np.arange(len(ratings) + 2))
    graph = ScriptedGraph(ratings)
    policy = TradingAgentsPolicy(graph=graph)

    result = simulate(data, policy.act, start=1, stop=len(ratings) + 1, transaction_cost=0.0005)

    # simulate decides at session t and the fill is at t + 1: weights[i] follows ratings[i].
    held = (result.weights[:, 0] > 0.99).tolist()
    assert held == [True, True, False, False, True, False, True, True]
    assert [call[0] for call in graph.calls] == ["AAPL"] * len(ratings)
    assert [call[1] for call in graph.calls] == [str(d) for d in data.dates[: len(ratings)]]


def test_the_decision_is_made_as_of_the_last_close_in_the_view():
    data = panel([100, 101, 102, 103])
    graph = ScriptedGraph(["Hold"])
    TradingAgentsPolicy(graph=graph).act(MarketView(data.rows(0, 3), np.zeros(1)))
    assert graph.calls == [("AAPL", str(data.dates[2]))]


def test_weights_are_arrays_or_none():
    data = panel([100, 101])
    view = MarketView(data, np.zeros(1))
    policy = TradingAgentsPolicy(graph=ScriptedGraph(["Buy", "Sell", "Hold"]))
    assert policy.act(view).tolist() == [1.0]
    assert policy.act(view).tolist() == [0.0]
    assert policy.act(view) is None


def test_custom_rating_weights_and_validation():
    view = MarketView(panel([100, 101]), np.zeros(1))
    policy = TradingAgentsPolicy(graph=ScriptedGraph(["Overweight"]), rating_weights={"Overweight": 0.5})
    assert policy.act(view).tolist() == [0.5]
    with pytest.raises(ValueError, match="must be None or in"):
        TradingAgentsPolicy(rating_weights={"buy": 2.0})


def test_an_unmapped_rating_raises():
    view = MarketView(panel([100, 101]), np.zeros(1))
    with pytest.raises(ValueError, match="unmapped rating"):
        TradingAgentsPolicy(graph=ScriptedGraph(["Strong Buy"])).act(view)


def test_only_single_asset_tasks_are_supported():
    data = panel([[100, 50], [101, 51]], tickers=("AAPL", "MSFT"))
    graph = ScriptedGraph(["Buy"])
    with pytest.raises(ValueError, match="trades one asset"):
        TradingAgentsPolicy(graph=graph).act(MarketView(data, np.zeros(2)))
    assert graph.calls == []


def test_cost_and_ledger_inputs_are_populated():
    view = MarketView(panel([100, 101]), np.zeros(1))
    decision = "**Rating**: Sell\nThe stock is extended."

    class Graph:
        def propagate(self, ticker, trade_date):
            # A dated snapshot name must match the listed model id.
            policy.usage.add("gpt-6-luna-2026-09-22", 1_000_000, 200_000, cached_input_tokens=400_000)
            policy.usage.add("gpt-6-sol", 100_000, 50_000)
            return {"final_trade_decision": decision}, "Sell"

    policy = TradingAgentsPolicy(graph=Graph(), deep_think_llm="gpt-6-sol", quick_think_llm="gpt-6-luna")
    assert policy.act(view).tolist() == [0.0]

    luna = 600_000 * 0.10 / 1e6 + 400_000 * 0.01 / 1e6 + 200_000 * 0.50 / 1e6
    sol = 100_000 * 2.0 / 1e6 + 50_000 * 10.0 / 1e6
    assert policy.last_cost_usd == pytest.approx(luna + sol)

    inputs = policy.ledger_inputs
    assert inputs["signal"] == "Sell" and inputs["weight"] == 0.0
    assert inputs["decision_sha256"] == hashlib.sha256(decision.encode()).hexdigest()
    assert inputs["decision_head"] == decision
    assert (inputs["ticker"], inputs["trade_date"]) == ("AAPL", str(view.date))
    assert (inputs["deep_think_llm"], inputs["quick_think_llm"]) == ("gpt-6-sol", "gpt-6-luna")
    assert inputs["expected_commit"] == ta.PINNED_COMMIT
    assert inputs["token_usage"]["gpt-6-sol"] == {"input": 100_000, "cached_input": 0, "output": 50_000, "calls": 1}
    assert inputs["unpriced_models"] == [] and inputs["usage_reported"] is True
    json.dumps(inputs)  # the ledger stores it as JSON


def test_usage_is_reset_per_decision():
    view = MarketView(panel([100, 101]), np.zeros(1))

    class Graph:
        def propagate(self, ticker, trade_date):
            policy.usage.add("gpt-6-luna", 1_000_000, 0)
            return {}, "Hold"

    policy = TradingAgentsPolicy(graph=Graph())
    policy.act(view)
    policy.act(view)
    assert policy.last_cost_usd == pytest.approx(0.10)


def test_failed_runs_still_report_what_they_spent():
    view = MarketView(panel([100, 101]), np.zeros(1))

    class Graph:
        def propagate(self, ticker, trade_date):
            policy.usage.add("gpt-6-sol", 1_000_000, 0)
            raise TimeoutError("provider stalled")

    policy = TradingAgentsPolicy(graph=Graph())
    with pytest.raises(TimeoutError):
        policy.act(view)
    assert policy.last_cost_usd == pytest.approx(2.0)


def test_unlisted_models_bill_at_the_most_expensive_listed_rate():
    usage = {"mystery-model": {"input": 1_000_000, "cached_input": 0, "output": 1_000_000, "calls": 1}}
    cost, unpriced = usage_cost(usage, DEFAULT_PRICES_USD_PER_MTOK)
    top = max(DEFAULT_PRICES_USD_PER_MTOK.values(), key=lambda row: row["output"])
    assert cost == pytest.approx(top["input"] + top["output"])
    assert unpriced == ["mystery-model"]
    row, listed = price_of("gpt-6.1-sol-2026-09-29", DEFAULT_PRICES_USD_PER_MTOK)
    assert listed and row["cached_input"] == 0.10  # not gpt-6-sol's 0.20


def test_usage_meter_is_additive():
    meter = UsageMeter()
    meter.add("m", 10, 5, 2)
    meter.add("m", 1, 1)
    assert meter.snapshot() == {"m": {"input": 11, "cached_input": 2, "output": 6, "calls": 2}}
    meter.reset()
    assert meter.snapshot() == {}


def test_a_stale_decision_date_is_refused_without_calling_the_agent():
    graph = ScriptedGraph(["Buy"])
    policy = TradingAgentsPolicy(graph=graph, max_age_days=7)
    with pytest.raises(RuntimeError, match="only runs live"):
        policy.act(MarketView(panel([100, 101]), np.zeros(1)))  # dated January 2024
    assert graph.calls == []

    today = pd.Timestamp(date.today())
    recent = pd.bdate_range(end=today, periods=3).to_numpy(dtype="datetime64[D]")
    assert policy.act(MarketView(panel([100, 101, 102], dates=recent), np.zeros(1))).tolist() == [1.0]


def test_a_missing_agent_package_fails_at_construction_with_an_install_hint(monkeypatch):
    for name in ("tradingagents", "tradingagents.default_config", "tradingagents.graph.trading_graph"):
        monkeypatch.setitem(sys.modules, name, None)  # makes the import raise ImportError
    monkeypatch.setattr(ta, "installed_commit", lambda: ta.PINNED_COMMIT)
    with pytest.raises(ImportError, match=ta.PINNED_COMMIT):
        TradingAgentsPolicy()


def test_the_submission_validates_without_the_extra_or_keys(monkeypatch, capsys):
    from rl.bench import cli

    monkeypatch.setitem(sys.modules, "tradingagents", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    submission = Submission.load(SUBMISSION)
    assert submission.kind == "llm" and submission.ledger == ("us-aapl", "us-googl", "us-amzn")
    monkeypatch.setattr(cli, "discover", lambda root: {submission.id: submission})
    assert cli.main(["validate"]) == 0
    assert "0 failures" in capsys.readouterr().out

    broken = Submission(id="broken", name="Broken", kind="llm", policy=submission.policy,
                        params={"typo": True}, ledger=submission.ledger)
    monkeypatch.setattr(cli, "discover", lambda root: {broken.id: broken})
    assert cli.main(["validate"]) == 1
    assert "typo" in capsys.readouterr().err


def test_the_paper_window_tasks():
    for task_id, ticker in (("us-aapl", "AAPL"), ("us-googl", "GOOGL"), ("us-amzn", "AMZN")):
        task = TASKS[task_id]
        assert task.tickers == (ticker,) and task.source == "yahoo" and not task.live
        assert (task.start, task.validation_start) == ("2015-01-01", "2023-07-01")
        assert (task.test_start, task.test_end) == ("2024-01-01", "2024-03-30")
        assert task.transaction_cost == 0.0005 and task.periods_per_year == 252


def test_the_submission_pins_the_commit_the_module_installs():
    assert ta.INSTALL_REQUIREMENT.endswith("@" + ta.PINNED_COMMIT)
    assert json.loads((SUBMISSION / "submission.json").read_text())["params"]["expected_commit"] == ta.PINNED_COMMIT


def test_config_layers_defaults_then_params_then_overrides(tmp_path):
    defaults = {
        "llm_provider": "x",
        "max_tool_rounds": 20,
        "data_vendors": {"news_data": "yfinance", "macro_data": "fred"},
        "temperature": None,
    }
    policy = TradingAgentsPolicy(
        llm_provider="openai",
        max_debate_rounds=2,
        max_tool_rounds=6,
        data_vendors={"news_data": "alpha_vantage"},
        config_overrides={"temperature": 0.0, "llm_provider": "anthropic"},
        workdir=str(tmp_path),
        graph=ScriptedGraph([]),
    )
    config = policy._config(defaults)
    assert config["llm_provider"] == "anthropic"  # overrides win
    assert config["max_debate_rounds"] == 2 and config["max_tool_rounds"] == 6 and config["temperature"] == 0.0
    assert config["data_vendors"] == {"news_data": "alpha_vantage", "macro_data": "fred"}
    assert config["memory_log_path"] == "{}/memory/trading_memory.md".format(tmp_path)
    assert defaults["data_vendors"]["news_data"] == "yfinance"  # the release defaults are not mutated
    ephemeral = TradingAgentsPolicy(graph=ScriptedGraph([]))._config(defaults)
    assert ephemeral["results_dir"].startswith(tempfile_root()) and "tradingagents-" in ephemeral["results_dir"]


def tempfile_root():
    import tempfile

    return tempfile.gettempdir()


def _fake_agent_modules(monkeypatch, *, commit, record, rejected=()):
    """Stand-ins for tradingagents, openai and langchain_core, enough to build a graph.

    The fake OpenAI client rejects the model ids in ``rejected`` as the API does
    for a revoked key or a missing model.
    """

    class Models:
        def retrieve(self, model):
            if model in rejected:
                raise PermissionError("model {} rejected".format(model))
            return types.SimpleNamespace(id=model)

    class OpenAI:
        def __init__(self):
            self.models = Models()

    class Graph:
        def __init__(self, selected_analysts, debug, config, callbacks):
            record.update(analysts=selected_analysts, debug=debug, config=config, callbacks=callbacks)

        def propagate(self, ticker, trade_date):
            message = types.SimpleNamespace(
                usage_metadata={"input_tokens": 1_000_000, "output_tokens": 0, "input_token_details": {"cache_read": 0}},
                response_metadata={"model_name": "gpt-6-luna-2026-09-22"},
            )
            response = types.SimpleNamespace(generations=[[types.SimpleNamespace(message=message)]])
            for callback in record["callbacks"]:
                callback.on_llm_end(response)
            return {"final_trade_decision": "Rating: Buy"}, "Buy"

    modules = {
        "openai": types.ModuleType("openai"),
        "tradingagents": types.ModuleType("tradingagents"),
        "tradingagents.default_config": types.ModuleType("tradingagents.default_config"),
        "tradingagents.graph": types.ModuleType("tradingagents.graph"),
        "tradingagents.graph.trading_graph": types.ModuleType("tradingagents.graph.trading_graph"),
        "langchain_core": types.ModuleType("langchain_core"),
        "langchain_core.callbacks": types.ModuleType("langchain_core.callbacks"),
    }
    modules["openai"].OpenAI = OpenAI
    modules["tradingagents"].__version__ = "0.6.0"
    modules["tradingagents.default_config"].DEFAULT_CONFIG = {"data_vendors": {}, "max_tool_rounds": 20}
    modules["tradingagents.graph.trading_graph"].TradingAgentsGraph = Graph
    modules["langchain_core.callbacks"].BaseCallbackHandler = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ta, "installed_commit", lambda: commit)


def test_the_real_graph_is_built_at_construction_with_a_usage_callback(monkeypatch, tmp_path):
    record = {}
    _fake_agent_modules(monkeypatch, commit=ta.PINNED_COMMIT, record=record)
    policy = TradingAgentsPolicy(workdir=str(tmp_path), analysts=["market", "news"])
    assert record["analysts"] == ["market", "news"]  # built before act

    assert policy.act(MarketView(panel([100, 101]), np.zeros(1))).tolist() == [1.0]
    assert record["analysts"] == ["market", "news"] and record["debug"] is False
    assert record["config"]["deep_think_llm"] == "gpt-6-sol"
    assert policy.last_cost_usd == pytest.approx(0.10)
    assert policy.ledger_inputs["agent_version"] == "0.6.0"
    assert policy.ledger_inputs["agent_commit"] == ta.PINNED_COMMIT


@pytest.mark.parametrize("commit", ["0" * 40, None])
def test_a_different_or_unknown_installed_commit_is_refused_at_construction(monkeypatch, tmp_path, commit):
    record = {}
    _fake_agent_modules(monkeypatch, commit=commit, record=record)
    with pytest.raises(RuntimeError, match="pins {}".format(ta.PINNED_COMMIT)):
        TradingAgentsPolicy(workdir=str(tmp_path))
    assert record == {}


@pytest.mark.parametrize("rejected", ["gpt-6-sol", "gpt-6-luna"])
def test_a_model_or_key_the_api_rejects_fails_construction_before_the_graph(monkeypatch, tmp_path, rejected):
    record = {}
    _fake_agent_modules(monkeypatch, commit=ta.PINNED_COMMIT, record=record, rejected={rejected})
    with pytest.raises(RuntimeError, match="OpenAI preflight failed for model '{}'".format(rejected)):
        TradingAgentsPolicy(workdir=str(tmp_path))
    assert record == {}


def test_graph_construction_failure_is_raised_before_act(monkeypatch):
    _fake_agent_modules(monkeypatch, commit=ta.PINNED_COMMIT, record={})

    def missing_key(**kwargs):
        raise ValueError("API key for provider 'openai' is not set")

    monkeypatch.setattr(sys.modules["tradingagents.graph.trading_graph"], "TradingAgentsGraph", missing_key)
    with pytest.raises(ValueError, match="API key"):
        TradingAgentsPolicy()


def test_installed_commit_reads_direct_url_metadata(monkeypatch):
    payload = json.dumps({"url": ta.REPOSITORY, "vcs_info": {"vcs": "git", "commit_id": "abc"}})
    monkeypatch.setattr(
        ta.metadata, "distribution", lambda name: types.SimpleNamespace(read_text=lambda file: payload)
    )
    assert ta.installed_commit() == "abc"

    def missing(name):
        raise ta.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ta.metadata, "distribution", missing)
    assert ta.installed_commit() is None
