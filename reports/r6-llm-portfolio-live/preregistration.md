# R6 pre-registration: a frozen LLM allocates the US sector ETFs, live

Written 2026-11-02, before the first ledger entry. This report was merged to
`main` in its own pull request (the pre-registration-only PR) before the code it
governs, as `.plan/lab-protocol.md` requires; that merge commit and this file's
git commit time are the evidence that the plan came first. The first entry under
`submissions/llm-portfolio/ledger/us-sectors/` must be committed after it. The
report follows `.plan/lab-protocol.md` and `reports/README.md` and is level
**L3**: the entrant runs prospectively in the decision ledger. No historical
backtest of the model is run or reported; its training data may contain the
past, so only decisions committed before an unknown open count.

## 1. Question

Can a frozen frontier LLM, shown only a compact point-in-time table of price
features for the nine Select Sector SPDR ETFs, allocate them better than equal
weight? This is not a reproduction of a paper's numbers. It is a live test of
the claim behind current "LLM portfolio manager" products: that a general model
has cross-sectional skill once information limits, execution, fees and
contamination are controlled.

## 2. Entrant

| Item | Value |
| --- | --- |
| Code | `rl/bench/policies/llm_portfolio.py:LLMPortfolioPolicy` |
| Submission | `submissions/llm-portfolio/submission.json` (`kind: "llm"`, `trials: 1`) |
| Provider | Anthropic, official `anthropic` SDK (the `agents` extra) |
| Model | `claude-sonnet-5-5`; the model rejects non-default `temperature`, so no sampling parameters are sent and each decision is one sample at the provider's defaults; `thinking: {"type": "between_tools"}`; `max_tokens` 8,192 |
| Model cutoff | 2026-06-30 (Anthropic publishes "Jun 2026"; the end of the month is the conservative date in `model_cutoffs`) |
| Sources | <https://platform.claude.com/docs/en/about-claude/models/overview>, <https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions>, <https://platform.claude.com/docs/en/about-claude/pricing> (read 2026-11-02) |
| Prompt | `llm-portfolio-1`, a fixed template in the module; SHA-256 `db5fa0a7816f6f1cb0d670240d36a38abe5dc2816c22097f3aae0cddecaf37a2` |
| Inputs | one table per asset: 1/5/21/63-session close-to-close returns, 21/63-session annualised volatility, distance to the 50/200-session averages, current weight; plus the account's cash and any `view.info` items published strictly before the decision close |
| Output | one strict tool call (`submit_target_weights`, `strict: true`) with one weight per ETF, `cash` and an optional rationale; `tool_choice: {"type": "auto"}` because this model rejects forced tool use; a JSON object in text is accepted as a fallback; long-only, each in [0, 1], weights plus cash at most 1, exactly the nine tickers |
| Retry | one retry after an invalid response; a second invalid response is a hold with the error as its reason |
| Account | one account over the nine ETFs, target weights filled by the engine at the next open, 0.05% per side |

The model id is the pin. Anthropic's model-id documentation states that from the
4.6 generation the dateless id is the canonical id and maps to a fixed snapshot;
it is not an alias that follows updates (there is no dated variant to choose).
The models list API could not be queried from this environment because no API
key was available; the id and cutoff were read from the official model pages on
2026-11-02. Each entry stores the model name the provider reports, so an
unexpected served model is visible. A change of model id, prompt, input table or
code is a new entrant and a deviation (§11).

No information sources are declared: the table is price-only. If a source is
added later, it is a deviation and its items enter only through `MarketView.info`
with publication times before the close.

## 3. Start rule and window

The entrant is in `.github/workflows/ledger.yml` but off by default:
`LLM_PORTFOLIO_ENABLED` must equal `true` before `rl-pricing bench ledger record`
stops excluding `llm-portfolio`. The start is the **fill session after the first
verified entry** committed on `main` after that switch and after this
pre-registration and its recall probe are merged. Entries count only under the
ledger's normal rule: the workflow's push step finished before the fill
session's open.

The window is **63 sessions**, three months at 252 sessions a year, matching R3.
The report records the actual first and last fill sessions. A session with no
verified entry is a hold and is listed; nothing is back-filled. If the first
verified entry arrives after the owner intends, the window still starts at that
fill and ends 63 sessions later.

## 4. Baselines

- **Equal weight (1/N) buy and hold** on the same nine ETFs over exactly the
  same sessions: the ledger record's benchmark, bought once and executed by the
  same engine with no rebalancing.
- **Cash**: 0% return, no fees, for context only. It is a weak baseline for a
  long-only sector portfolio; a positive return alone is not skill.
- The lab's live-track rows on `us-sectors` are shown for context; they start at
  their own freeze dates and are not compared on the same window.

## 5. Metrics

From the ledger record's daily returns, paired with equal weight on the same
dates: cumulative return, annualised Sharpe (mean/std with ddof=1, times
sqrt(252), risk-free 0), maximum drawdown, turnover, average cash, number of
sessions held and per-decision cost. Annualised return is reported for reference
only; annualising three months is not meaningful.

## 6. Verdict thresholds

Three quantities, each with null value 0 and "larger is better" after the sign
shown:

| Claim | Quantity $x$ | Better when |
| --- | --- | --- |
| R6-C1 | cumulative return of the entrant minus equal weight, points | $x > 0$ |
| R6-C2 | annualised Sharpe of the entrant minus equal weight | $x > 0$ |
| R6-C3 | maximum drawdown of equal weight minus the entrant's, points | $x > 0$ |

Let $\hat{x}$ be the estimate and $[l, u]$ its paired stationary-bootstrap 95%
interval (Politis and Romano 1994, mean block $\mathrm{round}(63^{1/3}) = 4$,
10,000 resamples, seed 0), resampling the same dates for the entrant and the
baseline. Apply in this order:

1. **? Inconclusive** if $l \le 0 \le u$: the window cannot tell no effect from
   a real effect of the size this window can resolve (§7).
2. **✗ Does not hold** if $u < 0$: the entrant is worse than equal weight on the
   quantity.
3. **✅ Holds** if $l > 0$: better than equal weight. With a 95% two-sided
   interval, $l > 0$ is a **2.5% one-sided** test (equivalently 5% two-sided).

There is no "partly holds" row: the claims have no published magnitude to
shrink from. This is the protocol table's direction-only form, fixed before any
data exists. Per-day weights, costs and response hashes are published, but only
the three pooled quantities get verdicts. The three claims share one return
series and are not independent; the report says so.

## 7. Smallest effect this test can detect (estimate)

These are assumptions to be replaced by the bootstrap on the realised path.

- **Excess return.** A long-only nine-sector portfolio has roughly 1.0% to 1.2%
  daily volatility. A plausible active allocation has 0.3% to 0.8% daily
  tracking error against equal weight. Over 63 sessions the cumulative excess
  return then has a standard error of about $0.003\sqrt{63}$ to
  $0.008\sqrt{63}$, i.e. **2.4 to 6.3 points**, so the 95% interval is about
  $\pm 4.7$ to $\pm 12.4$ points. Only a large reallocation edge is detectable;
  a two-point annual edge will be inconclusive.
- **Sharpe difference.** Lo (2002): the standard error of an annualised Sharpe
  ratio is about $\sqrt{(1+SR^2/2)/\text{years}}$. With 63 sessions = 0.25
  years, it is about 2.0 at SR 0 and $\sqrt{(1+2^2/2)/0.25} = \sqrt{12} = 3.46$
  at SR 2. The difference of two such estimates has a standard error of about
  2.8 to 4.9 and a 95% half-width of about **6 to 10**, so R6-C2 will very
  likely be inconclusive unless the entrant is much worse (or much better) than
  equal weight. We say this now so an inconclusive Sharpe is not read as a
  finding.
- **Drawdown.** Three months of a long-only sector portfolio typically contains
  a 4% to 12% drawdown. The bootstrap standard error of a drawdown difference
  is of the order of several points; only a clear risk reduction is detectable.

## 8. Cost and cap

Published standard-tier prices are $2 per million input tokens and $10 per
million output tokens. The rendered prompt is capped at a conservative
`MAX_INPUT_CHARS = 40,000` (two characters per token, so at most the module's
`MAX_INPUT_TOKENS = 20,000`); information items are truncated or omitted to fit,
and a prompt that would still exceed the budget is not sent. Output is bounded
by `max_tokens = 8,192`, which includes the `between_tools` thinking that the
model bills as output; one retry is allowed. The declared worst case per
decision is therefore

$$2\left(\frac{20{,}000 \times 2}{10^6} + \frac{8{,}192 \times 10}{10^6}\right) = \$0.24384,$$

rounded up to **$0.25** (`LLMPortfolioPolicy.max_cost_per_decision_usd`).

The submission declares `monthly_budget_usd: 15.0`, a per-entrant cap. The
ledger takes the smaller of that and `LEDGER_MONTHLY_BUDGET_USD` and checks,
immediately before the call, that this month's recorded spend plus $0.25 fits;
otherwise it writes nothing and the session is a hold. With 21 to 22 sessions a
month the reservation is at most **$5.50/month**; 63 sessions is at most
**$15.75** in total. The expected actual cost, **labeled an estimate, not
measured**, is about $0.01 per decision, or about $0.6 for the window. The owner may set the real cap before the switch; changing it later is a
deviation. A cap-induced hold is a missed session and is reported.

## 9. Recall probe

The probe asks `claude-sonnet-5-5`, without tools, for the closes of the nine
ETFs on the five sessions before the first live decision, and for notable sector
events in those days. It scores absolute percentage error and coverage; the
pre-registration sets no pass threshold. The measurement documents where the
model's knowledge ends relative to the live window.

- Script: `evidence/recall_probe.py` (one call, at most 8,192 output tokens,
  `thinking: {"type": "between_tools"}`, capped at $1; it refuses to run above
  that). It scores the **unadjusted** Yahoo closes from the data cache or
  `YahooDownloader.fetch_unadjusted`, not the dividend-adjusted levels the
  benchmark loader uses.
- Run: `ANTHROPIC_API_KEY=<key> python reports/r6-llm-portfolio-live/evidence/recall_probe.py --start <first decision date>`.
- Status on 2026-11-02: **pending**. No usable key is available in this
  environment: `ANTHROPIC_API_KEY` is the placeholder `dummy-key` in the shell
  and in `~/my-code/TradingAgents/.env`, and the models API returns 401 for
  both. `evidence/recall_probe_status.json` records the check.
- The probe must be run and its JSON output committed **before** the owner sets
  `LLM_PORTFOLIO_ENABLED=true`. The output stores the exact prompt, response,
  usage and cost.

## 10. Contamination

The live window starts in November 2026, after the model's June 2026 training
cutoff, so a decision cannot recall its own outcome: **uncontaminated by
construction**. The entrant never sees a historical replay. The remaining leak
is a served model different from the pinned id; each entry stores the reported
model name and the token usage. The recall probe measures how well the model can
state the five pre-window closes; if it can, the report says so and treats the
contamination claim as unsupported for that model.

## 11. Deviations and contingencies

1. **Model.** If the provider serves a different model or retires
   `claude-sonnet-5-5`, the window continues only if the report records the
   change; a new model starts a new entrant, not a silent continuation.
2. **Prompt or code.** Any change to the template, input table, tool schema or
   decision rule is a new entrant. The prompt hash and `code_sha256` in each
   entry show when it happened.
3. **Information sources.** Adding `info_sources` after the first decision is a
   deviation and is reported; only items before the close reach the prompt.
4. **Cap.** A cap change after the first decision is a deviation.
5. **Missed sessions.** Provider outage, rate limit, invalid output twice, a
   cap hold or a late workflow run is a hold. The report lists missed,
   unverified and hold sessions and reports their returns as the engine does.
6. **Fees.** The task's 0.05% per side is fixed; a fee-sensitivity table at 0,
   5 and 10 bp is secondary and never used for a verdict.

## 12. Run prerequisites

| Name | Kind | Needed for |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | secret | every model call (already passed by `ledger.yml`) |
| `LLM_PORTFOLIO_ENABLED` | repository variable | `true` starts the entrant; unset or any other value excludes it |
| `LEDGER_MONTHLY_BUDGET_USD` | repository variable | optional global cap; the effective cap is the smaller of it and the submission's $15 |
| `monthly_budget_usd` | submission param | the entrant's hard monthly cap ($15 default; the owner sets the real value) |

The workflow installs `rl-pricing[agents]` (which brings the official Anthropic
SDK) only on the enabled branch; the R3 install path is unchanged while the
entrant is off. `rl-pricing bench validate` checks the submission without calling
a model; the ledger's preflight checks the key and constructs the client before
any entry is written.

## 13. Authors and publication

This is the lab's own entrant, not a third-party claim, so there is no author
reply step. The report is published with its per-decision ledger, token usage,
costs and the recall probe.
