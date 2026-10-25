# R3 pre-registration addendum — 2026-10-25

Complete and commit this addendum before the first TradingAgents ledger entry.
The recall probe compares the models' recalled closes with the repo loader's
dividend-adjusted close values. The pre-registration sets no pass threshold;
interpretation and the start decision remain for the owner.

## Recall probe

Command run:

```bash
python reports/r3-tradingagents-live/evidence/recall_probe.py --start 2026-10-26 --output reports/r3-tradingagents-live/evidence/recall_probe_2026-10-25.json
```

Evidence file: `evidence/recall_probe_2026-10-25.json`

| Requested model | Returned model ID | Mean absolute percentage error | Coverage | Input / output tokens |
| --- | --- | ---: | ---: | ---: |
| `gpt-6-sol` | [fill] | [fill] | [fill] | [fill] |
| `gpt-6-luna` | [fill] | [fill] | [fill] | [fill] |

Verdict: [owner's interpretation; no threshold was pre-registered]

## Three decision dry run

Command run (provide the four local values from the owner's secret manager;
do not put the values in shell history or this repository):

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
read -s FRED_API_KEY
export FRED_API_KEY
read SEC_EDGAR_USER_AGENT
export SEC_EDGAR_USER_AGENT
export LEDGER_MONTHLY_BUDGET_USD=40
python reports/r3-tradingagents-live/evidence/dry_run.py
unset OPENAI_API_KEY FRED_API_KEY SEC_EDGAR_USER_AGENT LEDGER_MONTHLY_BUDGET_USD
```

Decision date: [fill]

| Task | Signal | Input / cached / output tokens | Cost (USD) |
| --- | --- | ---: | ---: |
| `us-aapl` | [fill] | [fill] | [fill] |
| `us-googl` | [fill] | [fill] | [fill] |
| `us-amzn` | [fill] | [fill] | [fill] |

All scratch ledger entries and downloaded data were discarded: [confirm]

## Start gate

Dry run outcome and owner decision: [fill]

The owner sets the Actions secrets and variables only after this addendum is
committed to `main`.
