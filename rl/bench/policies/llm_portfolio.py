"""A frozen API LLM that allocates across a task's assets in the decision ledger.

The model is Anthropic's Claude Sonnet 5.5 (``claude-sonnet-5-5``). Anthropic's
model-id documentation says that from the 4.6 generation the dateless id is the
canonical, fixed snapshot, not an alias that follows later updates
(https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions,
read 2026-11-02), so the id is the pin. The prompt is a fixed template in this
module; :data:`PROMPT_SHA256` is its hash and every entry records it.

Each decision reads one table computed from ``view.history``: 1/5/21/63-session
returns, 21/63-session volatility, distance to the 50/200-session averages and
the account's current weights. Items from ``view.info`` are added when the task
or submission declares information sources. Nothing after the decision close
can reach the table: the ledger gives the policy a history ending at that close
and an :class:`~rl.bench.info.InfoView` holding only earlier items.

The response is a tool call whose input is one number per ticker plus ``cash``
and an optional rationale. Weights must be finite, nonnegative, sum to at most
one (the remainder is cash), and name exactly the task's tickers. An invalid
response is retried once; a second invalid response raises, which the ledger
records as a hold with the error as its reason.

Cost is the provider's token usage times the published standard-tier prices
($2 per million input tokens, $10 per million output tokens,
https://platform.claude.com/docs/en/about-claude/pricing, read 2026-11-02).
:attr:`LLMPortfolioPolicy.max_cost_per_decision_usd` is the worst case of two
attempts at the bounded prompt and output sizes, rounded up; the ledger reserves
it against the entrant's monthly cap before the model is called.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import date as _date
from typing import Any, ClassVar

import numpy as np

from rl.bench.engine import Decision, check_weights
from rl.bench.market import MarketView
from rl.bench.policy import Policy

log = logging.getLogger(__name__)

#: The pinned model. Current Claude Sonnet id, read 2026-11-02 from
#: https://platform.claude.com/docs/en/about-claude/models/overview.
PINNED_MODEL = "claude-sonnet-5-5"
#: Anthropic publishes the training cutoff as "Jun 2026"; the end of the month
#: is the conservative reading for the date-valued ``model_cutoffs`` field.
MODEL_CUTOFF = "2026-06-30"
PRICES_USD_PER_MTOK = {"input": 2.00, "output": 10.00}
#: One retry is allowed, so the cap covers two attempts.
MAX_INPUT_TOKENS = 20_000
#: Two characters per token is a conservative bound for the rendered prompt.
MAX_INPUT_CHARS = 2 * MAX_INPUT_TOKENS
MAX_OUTPUT_TOKENS = 8_192
MAX_ATTEMPTS = 2
#: The model rejects non-default sampling parameters; this is its lowest thinking setting.
THINKING = {"type": "between_tools"}
_TOOL_NAME = "submit_target_weights"
_INFO_MAX_CHARS = 15_000
_INFO_TEXT_LIMIT = 500

SYSTEM_PROMPT = """\
You allocate a long-only portfolio of ETFs. You see a table computed from prices
up to and including the decision session's close, the account's current weights,
and any point-in-time information items supplied. Use only that data: do not
assume later prices, news or outcomes. Return target weights for the next
session through the submit_target_weights tool. Each weight is a fraction of
equity between 0 and 1. The weights plus cash must not exceed 1; any remainder
is cash. Keep the portfolio diversified unless the evidence is strong, and give
a short rationale."""

USER_TEMPLATE = """\
Decision session: {date}

Table columns: ret1, ret5, ret21, ret63 are close-to-close returns over that
many sessions; vol21, vol63 are annualised standard deviations of daily log
returns; dist50, dist200 are the close's distance from its 50- and 200-session
simple average; weight is the account's current asset weight. Missing values
are "n/a".

{table}

Current cash after the listed weights: {cash}

Point-in-time information published before the decision close:
{info}

Choose the target weights for the session after {date} and submit them with the
tool."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PROMPT_VERSION = "llm-portfolio-1"
PROMPT_SHA256 = _sha256(SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE)


class InvalidDecision(ValueError):
    """The model's response could not be turned into tradable weights."""


def _tool(tickers: tuple[str, ...]) -> dict:
    properties = {ticker: {"type": "number"} for ticker in tickers}
    properties["cash"] = {"type": "number"}
    properties["rationale"] = {"type": "string"}
    return {
        "name": _TOOL_NAME,
        "description": "Submit long-only target weights, cash and a short rationale.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": [*tickers, "cash"],
            "additionalProperties": False,
        },
    }


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidDecision("weights must be finite numbers, got {!r}".format(value))
    return float(value)


def _targets(payload, tickers: tuple[str, ...]) -> tuple[np.ndarray, float, str]:
    """Tradable weights, the stated cash, and the rationale from a tool input."""
    if not isinstance(payload, dict):
        raise InvalidDecision("output must be an object, got {}".format(type(payload).__name__))
    allowed = set(tickers) | {"cash", "rationale"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise InvalidDecision("unknown keys {}".format(unknown))
    missing = sorted((set(tickers) | {"cash"}) - set(payload))
    if missing:
        raise InvalidDecision("missing keys {}".format(missing))
    values = []
    for ticker in tickers:
        weight = _number(payload[ticker])
        if weight < 0 or weight > 1:
            raise InvalidDecision("{} weight {:.6g} is outside [0, 1]".format(ticker, weight))
        values.append(weight)
    cash = _number(payload["cash"])
    if cash < 0 or cash > 1:
        raise InvalidDecision("cash {:.6g} is outside [0, 1]".format(cash))
    total = sum(values) + cash
    if total > 1 + 1e-9:
        raise InvalidDecision("weights plus cash sum to {:.9g} > 1".format(total))
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise InvalidDecision("rationale must be a string")
    weights = check_weights(np.array(values), len(tickers))
    return weights, cash, rationale


def _json_text(text: str) -> dict:
    """The first JSON object in ``text``, accepting a surrounding code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(stripped[start : end + 1])


def _usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    return {
        name: getattr(usage, name)
        for name in (
            "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
        )
        if getattr(usage, name, None) is not None
    }


def _tokens(usage: dict) -> tuple[int, int]:
    """Billed input and output tokens; cache reads and writes bill as input here."""
    input_tokens = sum(
        int(usage.get(name) or 0)
        for name in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    return input_tokens, int(usage.get("output_tokens") or 0)


def _cost(usage: dict) -> float:
    input_tokens, output_tokens = _tokens(usage)
    return (input_tokens * PRICES_USD_PER_MTOK["input"] + output_tokens * PRICES_USD_PER_MTOK["output"]) / 1e6


def _response_text(response) -> str:
    parts = []
    for block in getattr(response, "content", ()) or ():
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(getattr(block, "text", ""))
        elif kind == "tool_use":
            parts.append(json.dumps(getattr(block, "input", None), sort_keys=True, default=str))
    return "\n".join(part for part in parts if part)


def _payload(response) -> dict:
    for block in getattr(response, "content", ()) or ():
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _TOOL_NAME:
            return dict(getattr(block, "input", None) or {})
    text = "\n".join(
        getattr(block, "text", "") for block in getattr(response, "content", ()) or ()
        if getattr(block, "type", None) == "text"
    )
    try:
        return _json_text(text)
    except (ValueError, json.JSONDecodeError) as error:
        raise InvalidDecision("no tool call and no JSON text: {}".format(error)) from error


def _value(value) -> str:
    return "n/a" if value is None or not np.isfinite(value) else "{:.2%}".format(float(value))


def _row(close: np.ndarray, weight: float) -> list[str]:
    last = close[-1] if close.size else np.nan

    def ret(n: int):
        if close.size <= n:
            return None
        start = close[-1 - n]
        if not (np.isfinite(start) and np.isfinite(last) and start > 0 and last > 0):
            return None
        return last / start - 1

    def vol(n: int):
        if close.size <= n:
            return None
        window = close[-1 - n :]
        if not np.isfinite(window).all() or (window <= 0).any():
            return None
        return float(np.std(np.diff(np.log(window)), ddof=1) * math.sqrt(252))

    def dist(n: int):
        if close.size < n:
            return None
        window = close[-n:]
        if not np.isfinite(window).all() or (window <= 0).any():
            return None
        return last / float(window.mean()) - 1

    return [
        _value(ret(1)), _value(ret(5)), _value(ret(21)), _value(ret(63)),
        _value(vol(21)), _value(vol(63)), _value(dist(50)), _value(dist(200)),
        _value(weight),
    ]


def input_table(view: MarketView) -> str:
    """The point-in-time table in the prompt, one row per asset."""
    headers = ["ticker", "ret1", "ret5", "ret21", "ret63", "vol21", "vol63", "dist50", "dist200", "weight"]
    rows = [
        [ticker, *_row(view.history.close[:, column], float(view.weights[column]))]
        for column, ticker in enumerate(view.tickers)
    ]
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(cell.ljust(width) for cell, width in zip(headers, widths)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines += [
        "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def info_lines(view: MarketView, budget: int = _INFO_MAX_CHARS) -> str:
    """The visible information items, compactly, within a character budget."""
    items = view.info.items()
    lines, used = [], 0
    for item in items:
        line = "- {} {} [{}] {}: {}".format(
            item.published_at.isoformat(), item.source, ",".join(item.tickers), item.title,
            item.text[:_INFO_TEXT_LIMIT],
        )
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    omitted = len(items) - len(lines)
    if omitted:
        lines.append("- ({} more items omitted to fit the input budget)".format(omitted))
    return "\n".join(lines) if lines else "(none)"


class LLMPortfolioPolicy(Policy):
    """Ask the pinned Claude Sonnet snapshot for target weights on the decision close.

    ``client`` injects an object with ``messages.create(**kwargs)``; tests use it
    so no model is called. Without it the official ``anthropic`` SDK is imported
    and its models endpoint is asked for :data:`PINNED_MODEL`, which costs nothing
    and fails before a decision can be spent if the key or model is unavailable.

    ``max_age_days`` refuses decisions dated more than that many days before
    today, so a ledger-only entrant cannot be replayed over the past (its model
    may have seen those prices and a replay would spend real money). ``None``
    disables the guard.
    """

    #: ``fit`` is not used and the model is not seeded; the ledger runs it once per session.
    stochastic: ClassVar[bool] = False
    #: Environment this policy needs beyond its provider key (the ledger checks it before acting).
    ledger_env: ClassVar[tuple[str, ...]] = ()
    #: Two attempts at the bounded input and output, at the published prices, rounded up.
    max_cost_per_decision_usd: ClassVar[float] = math.ceil(
        MAX_ATTEMPTS
        * (MAX_INPUT_TOKENS * PRICES_USD_PER_MTOK["input"] + MAX_OUTPUT_TOKENS * PRICES_USD_PER_MTOK["output"])
        / 1e6
        * 100
    ) / 100

    def __init__(
        self,
        model: str = PINNED_MODEL,
        llm_provider: str = "anthropic",
        monthly_budget_usd: float | None = None,
        client: Any = None,
        max_age_days: int | None = None,
    ) -> None:
        if model != PINNED_MODEL:
            raise ValueError("This policy is pinned to model {!r}, got {!r}".format(PINNED_MODEL, model))
        if llm_provider != "anthropic":
            raise ValueError("LLMPortfolioPolicy uses the Anthropic SDK, got llm_provider={!r}".format(llm_provider))
        if monthly_budget_usd is not None and (
            isinstance(monthly_budget_usd, bool)
            or not isinstance(monthly_budget_usd, (int, float))
            or not math.isfinite(monthly_budget_usd)
            or monthly_budget_usd < 0
        ):
            raise ValueError("monthly_budget_usd must be a non-negative finite number")
        self.model = model
        self.llm_provider = llm_provider
        self.monthly_budget_usd = monthly_budget_usd
        self.max_tokens = MAX_OUTPUT_TOKENS
        self.max_age_days = max_age_days
        self.last_cost_usd: float | None = 0.0
        self.ledger_inputs: dict[str, Any] = {}
        self._client = client if client is not None else self._build_client()

    def _build_client(self):
        try:
            import anthropic
        except ImportError as error:
            raise ImportError(
                "LLMPortfolioPolicy needs the official Anthropic SDK: pip install 'rl-pricing[agents]'"
            ) from error
        client = anthropic.Anthropic()  # ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
        try:
            client.models.retrieve(self.model)
        except Exception as error:
            raise RuntimeError(
                "Anthropic preflight failed for model {!r}: {}: {}".format(self.model, type(error).__name__, error)
            ) from error
        return client

    def reset(self) -> None:
        self.last_cost_usd = None  # no provider call yet
        self.ledger_inputs = {}

    def _prompt(self, view: MarketView) -> tuple[str, str]:
        table = input_table(view)
        cash = 1.0 - float(np.nansum(view.weights))
        info = info_lines(view)
        user = USER_TEMPLATE.format(
            date=str(view.date), table=table, cash="{:.2%}".format(max(0.0, cash)), info=info
        )
        if len(user) > MAX_INPUT_CHARS:
            excess = len(user) - MAX_INPUT_CHARS
            info = info[: max(0, len(info) - excess - 64)] + "\n- (truncated to the input budget)"
            user = USER_TEMPLATE.format(
                date=str(view.date), table=table, cash="{:.2%}".format(max(0.0, cash)), info=info
            )
        if len(user) > MAX_INPUT_CHARS:
            raise ValueError(
                "the rendered prompt is {} characters, above the {}-character input budget".format(
                    len(user), MAX_INPUT_CHARS
                )
            )
        return user, _sha256(user)

    def act(self, view: MarketView) -> Decision:
        self.last_cost_usd = None  # no provider call yet; the ledger bills nothing for this hold
        if self.max_age_days is not None:
            age = (_date.today() - _date.fromisoformat(str(view.date))).days
            if age > self.max_age_days:
                raise RuntimeError(
                    "decision date {} is {} days old (max_age_days={}): this policy only runs live".format(
                        str(view.date), age, self.max_age_days
                    )
                )
        self.reset()
        user, input_sha256 = self._prompt(view)
        tickers = view.tickers
        usage_total: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        response_texts, returned_models, attempts = [], [], 0
        error: Exception | None = None
        payload: dict | None = None
        decision = None

        def provenance(**extra):
            record = {
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "input_sha256": input_sha256,
                "input_chars": len(user),
                "input_budget_chars": MAX_INPUT_CHARS,
                "decision_date": str(view.date),
                "attempts": attempts,
                "response_text": "\n".join(response_texts),
                "returned_models": returned_models,
                "token_usage": usage_total,
                "cost_usd": _cost(usage_total),
            }
            record.update(extra)
            return record

        try:
            for attempts in range(1, MAX_ATTEMPTS + 1):
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    thinking=THINKING,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user}],
                    tools=[_tool(tickers)],
                    tool_choice={"type": "auto"},
                )
                usage = _usage(response)
                for name in usage_total:
                    usage_total[name] += int(usage.get(name) or 0)
                response_texts.append(_response_text(response))
                returned_models.append(getattr(response, "model", None))
                try:
                    payload = _payload(response)
                    decision = _targets(payload, tickers)
                except (InvalidDecision, ValueError, TypeError, KeyError) as invalid:
                    error = invalid
                    log.warning("LLM portfolio output on %s was invalid: %s", view.date, invalid)
                    continue
                break
            if decision is None:
                self.ledger_inputs = provenance(error=str(error))
                raise InvalidDecision(
                    "invalid output after {} attempt(s): {}".format(attempts, error)
                )
            weights, cash, rationale = decision
            self.ledger_inputs = provenance(
                response_json=payload,
                target_weights={ticker: float(weight) for ticker, weight in zip(tickers, weights)},
                cash=1.0 - float(weights.sum()),
                stated_cash=cash,
                rationale=rationale,
            )
            return weights
        except Exception:
            if not self.ledger_inputs:  # a provider failure before any response
                self.ledger_inputs = provenance(error="provider call failed")
            raise
        finally:
            self.last_cost_usd = _cost(usage_total)
