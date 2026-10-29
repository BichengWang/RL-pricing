"""TradingAgents (Xiao, Sun, Luo & Wang, arXiv 2412.20138) as a ledger participant.

The multi-agent LLM framework is installed separately with the pinned
``INSTALL_REQUIREMENT`` (the ledger workflow does this; package metadata stays
free of direct URLs so ``rl-pricing`` can be published to PyPI) and imported
only when :class:`TradingAgentsPolicy` is constructed, so ``rl.bench`` can be
imported and submissions validated without it. Construction checks the installed
commit, asks the provider's API for both pinned models (a rejected key or a
missing model fails here, before any decision can be recorded; listing a
model is free) and builds the graph.

Single-asset convention. ``act`` calls ``TradingAgentsGraph.propagate(ticker,
date)`` for ``view.tickers[0]`` with ``date`` = the decision session (the last
close in the data), and maps its rating to a long/flat target weight:

======================  ==============================================
rating                  target weight
======================  ==============================================
Buy, Overweight         ``[1.0]`` fully long
Sell, Underweight       ``[0.0]`` flat (cash)
Hold, REVIEW            ``None``: keep current holdings
======================  ==============================================

Release 0.6.0 returns the five-tier scale (Buy / Overweight / Hold /
Underweight / Sell, or ``REVIEW`` when the decision text has no parsable
rating); the paper trades buy / sell / hold. The engine is long-only, so a
sell moves to cash rather than short. ``rating_weights`` overrides the table.

Cost. ``last_cost_usd`` is token usage, collected with a LangChain callback on
every model call, times ``prices_usd_per_mtok`` (USD per million tokens). The
built-in table is copied from the OpenAI and Anthropic pricing pages (see
``DEFAULT_PRICES_USD_PER_MTOK``); a submission pins the prices it was run
with in its params. Cached input tokens bill at the cached rate (Anthropic
cache writes, which TradingAgents does not request, would bill at the input
rate), all other input tokens at the input rate, and reasoning tokens are part of the
output count. A model missing from the table is billed at the most expensive
listed rate and named in ``ledger_inputs["unpriced_models"]``.

State. Each policy instance works in ``workdir`` (a fresh temporary directory
by default), so TradingAgents' memory log starts empty and a run does not
depend on what an earlier run left on the machine. Pass a persistent
``workdir`` to let it accumulate lessons across days.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import tempfile
import threading
from collections.abc import Mapping
from datetime import date as _date
from importlib import metadata
from typing import Any, ClassVar

import numpy as np

from rl.bench.engine import Decision
from rl.bench.market import MarketView
from rl.bench.policy import Policy

log = logging.getLogger(__name__)

#: Release v0.6.0 (2026-10-03). TradingAgents is not on PyPI, so it is
#: installed from GitHub at this commit; the ledger workflow reads
#: ``INSTALL_REQUIREMENT`` from here, keeping the pin in one place.
PINNED_COMMIT = "1394a3f72aa4393e1a98f51b382434c4b4c2d972"
PINNED_VERSION = "0.6.0"
REPOSITORY = "https://github.com/TauricResearch/TradingAgents"
INSTALL_REQUIREMENT = "tradingagents @ git+{}.git@{}".format(REPOSITORY, PINNED_COMMIT)

INSTALL_HINT = (
    "TradingAgentsPolicy needs the TradingAgents package (Python >= 3.11): "
    "pip install '{}'".format(INSTALL_REQUIREMENT)
)

DEFAULT_RATING_WEIGHTS: dict[str, float | None] = {
    "buy": 1.0,
    "overweight": 1.0,
    "hold": None,
    "underweight": 0.0,
    "sell": 0.0,
    "review": None,
}

#: The key each supported provider reads from the environment (the names
#: TradingAgents' ``llm_clients/api_key_env.py`` uses).
API_KEY_ENV: dict[str, str] = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

#: USD per million tokens, standard tier, short context: OpenAI
#: (https://developers.openai.com/api/docs/pricing, read 2026-10-14) and Anthropic
#: (https://platform.claude.com/docs/en/about-claude/pricing, read 2026-10-29;
#: ``cached_input`` is the cache-hit rate).
DEFAULT_PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-6-sol": {"input": 2.00, "cached_input": 0.20, "output": 10.00},
    "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
    "gpt-5.6-sol": {"input": 4.00, "cached_input": 0.40, "output": 20.00},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "claude-opus-5-5": {"input": 4.00, "cached_input": 0.20, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
    "claude-sonnet-5": {"input": 2.00, "cached_input": 0.20, "output": 10.00},
}


class UsageMeter:
    """Token counts per model, accumulated across the LLM calls of one decision."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, dict[str, int]] = {}

    def reset(self) -> None:
        with self._lock:
            self._counts = {}

    def add(self, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        with self._lock:
            row = self._counts.setdefault(model, {"input": 0, "cached_input": 0, "output": 0, "calls": 0})
            row["input"] += int(input_tokens)
            row["cached_input"] += int(cached_input_tokens)
            row["output"] += int(output_tokens)
            row["calls"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {model: dict(row) for model, row in self._counts.items()}


def price_of(model: str, prices: Mapping[str, Mapping[str, float]]) -> tuple[Mapping[str, float], bool]:
    """The price row for a provider-reported model name and whether it was listed.

    Providers report dated snapshots (``gpt-6-luna-2026-09-22``), so the longest
    listed id that equals the name or prefixes it at a ``-`` boundary matches.
    An unlisted model gets the most expensive listed output rate.
    """
    matches = [key for key in prices if model == key or model.startswith(key + "-")]
    if matches:
        return prices[max(matches, key=len)], True
    return max(prices.values(), key=lambda row: (row["output"], row["input"])), False


def usage_cost(usage: Mapping[str, Mapping[str, int]], prices: Mapping[str, Mapping[str, float]]) -> tuple[float, list[str]]:
    """Dollar cost of a usage snapshot and the models that were not in ``prices``."""
    total, unpriced = 0.0, []
    for model, row in usage.items():
        price, listed = price_of(model, prices)
        if not listed:
            unpriced.append(model)
        fresh = max(row["input"] - row["cached_input"], 0)
        total += (
            fresh * price["input"]
            + row["cached_input"] * price.get("cached_input", price["input"])
            + row["output"] * price["output"]
        ) / 1e6
    return total, sorted(unpriced)


def _usage_callback(meter: UsageMeter):
    """A LangChain callback feeding ``meter`` from every chat-model response."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _MeterCallback(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None)
                    if not usage:
                        continue
                    metadata_ = getattr(message, "response_metadata", None) or {}
                    model = metadata_.get("model_name") or metadata_.get("model") or "unknown"
                    details = usage.get("input_token_details") or {}
                    meter.add(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                              details.get("cache_read", 0))

    return _MeterCallback()


def installed_commit() -> str | None:
    """The git commit TradingAgents was installed from, or ``None`` if unknown
    (not installed from a VCS URL, for example an editable local checkout)."""
    try:
        raw = metadata.distribution("tradingagents").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id")


def check_models(provider: str, models) -> None:
    """Retrieve each model from the provider's API with the environment's key.

    Raises ``RuntimeError`` naming the model if the key is rejected or the
    model is unavailable. Providers without a check here pass unchecked.
    """
    if provider == "openai":
        import openai

        client = openai.OpenAI()  # OPENAI_API_KEY, OPENAI_BASE_URL, as TradingAgents
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
    else:
        return
    for model in dict.fromkeys(models):
        try:
            client.models.retrieve(model)
        except Exception as error:
            raise RuntimeError(
                "{} preflight failed for model {!r}: {}: {}".format(provider, model, type(error).__name__, error)
            ) from error


class TradingAgentsPolicy(Policy):
    """Ask TradingAgents for a rating on the decision session and trade it long/flat.

    Parameters mirror the TradingAgents config keys of the same name;
    ``config_overrides`` sets any other key (for example
    ``{"openai_reasoning_effort": "medium", "temperature": 0.0}``) and is applied
    last. ``data_vendors`` is merged over the release's defaults.

    ``max_age_days`` makes the policy refuse decisions dated more than that many
    days before today, so a live-only submission cannot be replayed over
    history by the historical or live-replay runners (model cutoffs make such a
    replay contaminated and it would spend real money). ``None`` disables the
    guard.

    ``graph`` injects an object with ``propagate(ticker, date) -> (state,
    signal)`` instead of building ``TradingAgentsGraph``; tests use it so no
    model is ever called. A real graph's token usage reaches
    :attr:`usage` through a callback; an injected one may call
    ``policy.usage.add``.
    """

    #: ``fit`` does not depend on the seed; the agent's own sampling noise is
    #: not seedable, which is why the ledger runs it once per session.
    stochastic: ClassVar[bool] = False

    def __init__(
        self,
        llm_provider: str = "openai",
        deep_think_llm: str = "gpt-6-sol",
        quick_think_llm: str = "gpt-6-luna",
        max_debate_rounds: int = 1,
        max_risk_discuss_rounds: int = 1,
        max_tool_rounds: int | None = None,
        analysts: tuple[str, ...] | list[str] = ("market", "social", "news", "fundamentals"),
        data_vendors: dict[str, str] | None = None,
        config_overrides: dict[str, Any] | None = None,
        rating_weights: dict[str, float | None] | None = None,
        prices_usd_per_mtok: dict[str, dict[str, float]] | None = None,
        workdir: str | None = None,
        max_age_days: int | None = None,
        expected_commit: str | None = PINNED_COMMIT,
        graph: Any = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.deep_think_llm = deep_think_llm
        self.quick_think_llm = quick_think_llm
        self.max_debate_rounds = int(max_debate_rounds)
        self.max_risk_discuss_rounds = int(max_risk_discuss_rounds)
        self.max_tool_rounds = max_tool_rounds
        self.analysts = tuple(analysts)
        self.data_vendors = dict(data_vendors or {})
        self.config_overrides = dict(config_overrides or {})
        self.rating_weights = {k.lower(): v for k, v in {**DEFAULT_RATING_WEIGHTS, **(rating_weights or {})}.items()}
        for rating, weight in self.rating_weights.items():
            if weight is not None and not 0.0 <= float(weight) <= 1.0:
                raise ValueError("rating_weights[{!r}] must be None or in [0, 1], got {}".format(rating, weight))
        self.prices_usd_per_mtok = {**DEFAULT_PRICES_USD_PER_MTOK, **(prices_usd_per_mtok or {})}
        self.workdir = workdir
        self.max_age_days = max_age_days
        self.expected_commit = expected_commit
        self.usage = UsageMeter()
        self.last_cost_usd = 0.0
        self.ledger_inputs: dict[str, Any] = {}
        self._graph = graph
        self._injected = graph is not None
        self._tempdir: tempfile.TemporaryDirectory | None = None
        self._agent_version: str | None = None
        if self._graph is None:
            self._graph = self._build_graph()

    # -- construction of the real graph --------------------------------------------------------

    def _config(self, default_config: dict) -> dict:
        config = copy.deepcopy(default_config)
        if self.workdir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="tradingagents-")
            root = self._tempdir.name
        else:
            root = self.workdir
        config.update(
            llm_provider=self.llm_provider,
            deep_think_llm=self.deep_think_llm,
            quick_think_llm=self.quick_think_llm,
            max_debate_rounds=self.max_debate_rounds,
            max_risk_discuss_rounds=self.max_risk_discuss_rounds,
            results_dir="{}/logs".format(root),
            data_cache_dir="{}/cache".format(root),
            memory_log_path="{}/memory/trading_memory.md".format(root),
        )
        if self.max_tool_rounds is not None:
            config["max_tool_rounds"] = int(self.max_tool_rounds)
        config["data_vendors"] = {**config.get("data_vendors", {}), **self.data_vendors}
        config.update(self.config_overrides)
        return config

    def _build_graph(self):
        commit = installed_commit()
        if self.expected_commit and commit != self.expected_commit:
            raise RuntimeError(
                "TradingAgents is installed from commit {} but this submission pins {}; "
                "reinstall with pip install '{}'".format(commit, self.expected_commit, INSTALL_REQUIREMENT)
            )
        try:
            import tradingagents
            from tradingagents.default_config import DEFAULT_CONFIG
            from tradingagents.graph.trading_graph import TradingAgentsGraph
        except ImportError as error:
            raise ImportError("{} (import failed: {})".format(INSTALL_HINT, error)) from error
        self._agent_version = getattr(tradingagents, "__version__", None)
        check_models(self.llm_provider, (self.deep_think_llm, self.quick_think_llm))
        return TradingAgentsGraph(
            selected_analysts=list(self.analysts),
            debug=False,
            config=self._config(DEFAULT_CONFIG),
            callbacks=[_usage_callback(self.usage)],
        )

    # -- decisions ------------------------------------------------------------------------------

    def reset(self) -> None:
        self.last_cost_usd = 0.0
        self.ledger_inputs = {}

    def act(self, view: MarketView) -> Decision:
        if len(view.tickers) != 1:
            raise ValueError(
                "TradingAgentsPolicy trades one asset; the task has {} ({})".format(
                    len(view.tickers), ", ".join(view.tickers)
                )
            )
        ticker, day = view.tickers[0], str(view.date)
        if self.max_age_days is not None:
            age = (_date.today() - _date.fromisoformat(day)).days
            if age > self.max_age_days:
                raise RuntimeError(
                    "decision date {} is {} days old (max_age_days={}): this policy only runs live".format(
                        day, age, self.max_age_days
                    )
                )
        self.usage.reset()
        self.last_cost_usd, self.ledger_inputs = 0.0, {}
        try:
            state, signal = self._graph.propagate(ticker, day)
        finally:
            # Failed runs still spent tokens.
            cost, unpriced = usage_cost(self.usage.snapshot(), self.prices_usd_per_mtok)
            self.last_cost_usd = cost
        decision_text = str((state or {}).get("final_trade_decision") or "")
        rating = str(signal).strip().lower()
        if rating not in self.rating_weights:
            raise ValueError("TradingAgents returned unmapped rating {!r}".format(signal))
        weight = self.rating_weights[rating]
        usage = self.usage.snapshot()
        if not usage and not self._injected:
            log.warning("No token usage was reported for %s on %s; cost_usd is 0", ticker, day)
        self.ledger_inputs = {
            "agent": "TauricResearch/TradingAgents",
            "agent_version": self._agent_version if not self._injected else "injected",
            "agent_commit": installed_commit() if not self._injected else None,
            "expected_commit": self.expected_commit,
            "llm_provider": self.llm_provider,
            "deep_think_llm": self.deep_think_llm,
            "quick_think_llm": self.quick_think_llm,
            "max_debate_rounds": self.max_debate_rounds,
            "max_risk_discuss_rounds": self.max_risk_discuss_rounds,
            "analysts": list(self.analysts),
            "data_vendors": self.data_vendors,
            "config_overrides": self.config_overrides,
            "ticker": ticker,
            "trade_date": day,
            "signal": str(signal),
            "weight": weight,
            "decision_sha256": hashlib.sha256(decision_text.encode("utf-8")).hexdigest(),
            "decision_head": decision_text[:400],
            "token_usage": usage,
            "unpriced_models": unpriced,
            "usage_reported": bool(usage),
        }
        if weight is None:
            return None
        return np.array([float(weight)])
