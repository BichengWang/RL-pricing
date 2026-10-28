# Phase 0: foundations and launch (2026-10-15 → 2026-11-05)

Goal: the lab exists, R0 is published, the TradingAgents live clock is running,
and the launch carries the first live numbers.

Critical path: P0.11–P0.13 ledger fixes → P0.15 R3 start gate → first
TradingAgents fill, target **2026-10-23**, pre-registered deadline
**2026-10-30**. Each session lost shortens R3: 67 sessions from 10-23, 62
from 10-30.

| ID | Task | Owner | Depends on | Acceptance |
| --- | --- | --- | --- | --- |
| P0.1 | Set the GitHub description, topics and social preview (command in the launch kit) | owner | – | `gh repo view` shows description and 10 topics; preview image set |
| P0.2 | Lab scaffolding: `reports/` with `TEMPLATE.md`, `claims.json` schema, generator for the verdict table into `reports/README.md` and a README block; move `leaderboard/reproductions/` into `reports/r0-wang-zhang-2026/evidence/` | agent | – | `rl-pricing bench reports` renders the table from every `claims.json`; tests cover the verdict rendering and a malformed `claims.json` |
| P0.3 | R0 report: `preregistration.md` (marked post hoc), `claims.json` (return, Sharpe per agent, LSTM drawdown), `report.md` from the existing 75 runs | agent; owner reviews | P0.2 | Every number traces to an evidence file; draft sent to Xinyi Zhang by **2026-10-27** |
| P0.4 | Decision ledger MVP: `rl/bench/ledger.py`, `.github/workflows/ledger.yml` (before the open on trading days; schedule in P0.12), entries `submissions/<id>/ledger/<task>/YYYY-MM-DD.json` holding weights, inputs' hash, model ids, the workflow run id and its start time; scorer that replays entries through the engine; a missed day is recorded as hold | agent | – | A baseline runs through the ledger for 5 sessions on `main` (the ledger only runs there; canary, P0.14), and its record equals the engine's direct simulation; PRs editing ledger files are rejected by CI |
| P0.5 | TradingAgents adapter and live start: pin the TradingAgents commit and model ids, map BUY/SELL/HOLD to long-only weights (one ticker per sleeve, a third of the equity each), store API keys as Actions secrets, log cost per run, stop the job at a monthly cap | agent; owner supplies keys and the cap | P0.4 | R3 `preregistration.md` committed **before** the first ledger entry; first entry committed before the open on or before **2026-10-30**; cost per day measured |
| P0.6 | Licensed data decision for survivorship-free US equities (candidates: Sharadar via Nasdaq Data Link, Norgate, Tiingo, Databento); confirm the licence allows publishing derived results | agent researches; owner buys | – | Decision and licence excerpt in `decisions.md` by **2026-10-31** |
| P0.7 | Declared hyperparameter sweep for the reference RL agents: about 8 configurations each for PPO and SAC, selected on validation only, `trials` set accordingly | agent | – | New historical records with `trials` > 1 and DSR shown; README findings updated with whatever they show |
| P0.8 | Monthly live report generator (`rl/bench/monthly.py`) wired into the monthly job: live table, change since last month, figure | agent | – | `leaderboard/reports/2026-10.md` (October; the job writes the last completed month) committed by CI on **2026-11-02** |
| P0.9 | README restructured lab-first (verdict table, then findings, then leaderboard); launch kit rewritten for the lab identity | agent; owner approves | P0.2, P0.3, P0.8 | README opens with the verdict table; launch kit updated |
| P0.10 | Launch: outreach to R1 and R2 authors one day before; Show HN, X, Reddit per launch kit | owner | P0.9 | Posts live **2026-11-04**; emails sent 2026-11-03 |

## Status (2026-10-25)

PRs #97–#99 merged on 2026-10-25. The ledger failed on every scheduled run
from 10-16 to 10-25 until #98; a `workflow_dispatch` run on `main` after it
(run 40218696858) committed the canary's first entry (`a1e399f`) and failed
only on TradingAgents' preflight, as designed. R3's 10-23 target was missed.

| ID | State | Evidence and gap |
| --- | --- | --- |
| P0.1 | ◐ preview missing | Description, homepage and 10 topics set on 10-25 (`gh repo view`). The social preview image has no API; set it by hand |
| P0.2 | ✅ done | `rl-pricing bench reports` renders `reports/README.md` and the README block |
| P0.3 | ◐ notice missing | `reply.md` still says "pending". A Gmail draft to Xinyi Zhang (`xyzh@stanford.edu`, the paper's address) is ready, and says the draft has been public since 10-16 |
| P0.4 | ◐ first entry | Records on GitHub since #98. Closes at 5 verified canary sessions with `ledger score` equal to the engine (P0.14) |
| P0.5 | ◐ owner keys | Preflight shipped (P0.13). Waits for the P0.15 gate, then the secrets |
| P0.6 | ◐ owner | Sharadar chosen; written confirmation and purchase not recorded |
| P0.7 | ✅ done | README limitation corrected on 10-25 (P0.20) |
| P0.8 | ◐ dry run passed | P0.17. First CI run 11-02 06:00 UTC |
| P0.9 | ◐ | README lab-first. Launch kit (outside the repository) now cites `2026-10.md` and the applied metadata command |
| P0.10 | ⏳ | Due 11-03 and 11-04 |
| P0.11 | ✅ done | #98: `':(glob)submissions/*/ledger/**'`, staged only when `git status` shows entries. Dispatch run 40218696858 committed `a1e399f` |
| P0.12 | ◐ shipped | #98: cron `17 4,7,10 * * 1-5`, checkout `ref: main`. Start delays of 5 sessions still to list |
| P0.13 | ✅ done | #98: an `llm` participant needs the three credentials and `LEDGER_MONTHLY_BUDGET_USD`; the graph and pin check run at construction. Run 40218696858: "LLM ledger preflight requires OPENAI_API_KEY, FRED_API_KEY, SEC_EDGAR_USER_AGENT, LEDGER_MONTHLY_BUDGET_USD", no TradingAgents entry, canary committed |
| P0.14 | ◐ running | `submissions/ledger-canary` (1/N, `us-aapl`, `kind: baseline`, shown in ledger and monthly tables). First fill 10-26 |
| P0.15 | ◐ tooling | #99: `evidence/recall_probe.py`, `evidence/dry_run.py`, `addendum-template.md`. Runs need the owner's keys; the prereg sets no recall threshold, so the addendum states the reading |
| P0.16 | ◐ owner sends | Draft in Gmail; send by 10-27, then record the date in `reply.md` |
| P0.17 | ✅ done | All five `live` steps ran in a scratch clone on 10-25; `2026-10.md` rendered (45 strategy-task pairs, 3 tasks). Nothing needed fixing. Local canary entries without a run identity scored as holds, as `ledger.py` documents |
| P0.18 | ✅ done | `rl-pricing` 0.5.0 on PyPI (release `v0.5.0`, 10-23). On 10-25 a clean Python 3.13 venv ran `pip install rl-pricing` and `rl-pricing bench tasks`. 1.0.0 on 10-27: metadata in `pyproject.toml`, semantic versioning from [`CHANGELOG.md`](../CHANGELOG.md), import package still `rl` (decision 10-27) |
| P0.19 | ✅ done | #97: hashes of the authors' repository at `5dbb44df` (184 files) and Yahoo `^DJI`/DIA; L1 probe (Python 3.7, TF 1.15.4, Stable-Baselines 2.10.1) trained A2C for 5 steps, run 40219347746. **L1 go**, decided 10-25. Sharadar files wait for P0.6 |
| P0.20 | ✅ done | README limitation and launch kit corrected |

R3 timing: Monday's 04:17 run finds 10-23 recorded, so the canary fills
10-26, 10-27 and 10-28 (decisions 10-23, 10-26, 10-27). The gate passes after
the 10-28 close at the earliest; secrets set that evening give a first
TradingAgents fill on **10-29** (63 sessions), a day before the deadline.

## Next steps (2026-10-19 → 2026-11-05)

| ID | Task | Owner | Depends on | Acceptance | Due |
| --- | --- | --- | --- | --- | --- |
| P0.11 | Ledger commit step: `git add -- ':(glob)submissions/*/ledger/**'` | agent | – | A `workflow_dispatch` run on `main` commits its entries | 10-19 |
| P0.12 | Ledger schedule: cron Mon–Fri at 04:17, 07:17 and 10:17 UTC, off the hour. The first run records; later runs find the entry and stop. `actions/checkout` gets `ref: main`, because a run otherwise checks out the SHA it was triggered on and can't see an earlier run's push | agent | P0.11 | Each weekday's first run starts before 12:00 UTC; start delays for 5 sessions listed in the PR. If a first run starts after 13:00 UTC, trigger from an outside scheduler with `gh workflow run ledger.yml` (`EVENTS` already accepts `workflow_dispatch`) | 10-19 |
| P0.13 | Paid-agent preflight: a `kind: llm` participant needs `OPENAI_API_KEY`, `FRED_API_KEY`, `SEC_EDGAR_USER_AGENT` and a cap, and builds its graph when it is constructed. If anything is missing, nothing is written and the job fails before the open, so no key can show up mid-window and change R3's pinned inputs. The job stays red until the owner sets the secrets (P0.15) | agent | – | Tests: key unset, cap unset, or pinned commit mismatched → no entry, non-zero exit | 10-19 |
| P0.14 | Ledger canary: `submissions/ledger-canary`, 1/N on `us-aapl`, zero cost | agent | P0.11, P0.12 | 3 verified sessions by 10-22 (R3 gate); 5 by 10-26, with `ledger score` equal to the engine's direct simulation (P0.4's acceptance, in production) | 10-26 |
| P0.15 | R3 start gate. (a) Recall probe: both models, no tools, prices and events of the three stocks for the week before the start; script and output in `reports/r3-tradingagents-live/evidence/`. (b) One dry run of the three decisions in a scratch clone, entries discarded, tokens and cost measured. (c) A dated addendum to the R3 pre-registration with (a) and (b), committed before the first entry | agent; owner lends the key locally | P0.13 | Addendum on `main`; then the owner sets the Actions secrets and variables (checklist below). The next scheduled run makes R3's first entry | 10-22 |
| P0.16 | R0 notice: email Xinyi Zhang the draft, evidence and reproduce command. `reply.md` records the send date and that the report was on `main` from 2026-10-16 | owner sends; agent drafts | – | Sent by 10-21; hard limit 10-27 (P0.3), 7 days before the launch with a day to spare | 10-21 |
| P0.17 | Monthly dry run in a scratch clone: the `live` job's steps (`bench live --refresh`, `ledger score` with a token, `bench monthly --month 2026-10`, `leaderboard`, `reports`); fix what breaks | agent | P0.14 | A partial October report renders with the live table and the canary's ledger row; nothing committed | 10-28 |
| P0.18 | Claim PyPI before the launch (moved from P1.4): distribution `rl` → `rl-pricing` in `setup.py`, version 0.5.0. The import package stays `rl`, so the layout decision holds; it can't share an environment with the unrelated `rl` (GNU Readline) distribution, which the PyPI page says. Releases publish through `.github/workflows/publish.yml` with PyPI trusted publishing, so no token is stored | agent; owner adds the PyPI publisher and publishes the release | – | In a clean venv, `pip install rl-pricing` then `rl-pricing bench tasks` works | 11-02 |
| P0.19 | R1 groundwork while the ledger runs: download and hash the authors' files and the Yahoo data (R1 §14); L1 build probe, TensorFlow 1.15 on Python 3.7 in a `python:3.7` container on a runner | agent | – | Hashes in `reports/r1-finrl-ensemble/evidence/`. L1 go/no-go decided by 11-05 instead of 11-17; the report notes the schedule change | 11-05 |
| P0.20 | Stale text: README "Reference RL agents are untuned"; the launch kit's `2026-11.md` → `2026-10.md` | agent | – | Both corrected | 10-28 |

Owner checklist, in order (updated 2026-10-25; ledger PR, P0.1 metadata and
P0.18 PyPI are done):

1. P0.16: send the Gmail draft to Xinyi Zhang by **10-27**, then record the
   send date in `reply.md`.
2. P0.1: set the social preview image by hand (launch kit § Repository
   metadata).
3. P0.15, 10-26 → 10-28: with the keys in a local shell, run the two commands
   in `reports/r3-tradingagents-live/addendum-template.md`, fill it in and
   merge it as its own PR. After the 10-28 close, if the canary has 3
   verified sessions, set the secrets `OPENAI_API_KEY` and `FRED_API_KEY`
   and the variables `SEC_EDGAR_USER_AGENT` and
   `LEDGER_MONTHLY_BUDGET_USD=40`. Setting them starts R3, so not before.
4. P0.6: ask Sharadar for written confirmation; due 10-31.
5. 11-02: paste the October report's numbers into the launch kit. 11-03:
   author emails. 11-04: posts.

| Date | Step |
| --- | --- |
| 10-25 Sun | Ledger fixed (#98); canary's first entry committed; R1 L1 go |
| 10-26 Mon | Canary's first fill. The 04:17 run finds 10-23 recorded and stops |
| 10-26 Mon – 10-28 Wed | Recall probe, dry run, addendum PR; agent lists the first runs' start delays (P0.12) |
| 10-27 Tue | Last day for the R0 notice |
| 10-28 Wed | After the close: gate (3 verified canary fills and the addendum), then the owner sets the secrets |
| 10-29 Thu | Target for the first TradingAgents fill (R3: 63 sessions) |
| 10-30 Fri | R3 start deadline (62 sessions); canary at 5 fills, P0.4 closed |
| 11-02 Mon | CI writes `leaderboard/reports/2026-10.md` |
| 11-03 Tue | R1 and R2 author emails |
| 11-04 Wed | Launch |
| 11-05 Thu | Reddit posts begin |

## Exit criteria

- R0 published with verdicts; co-author notified at least 7 days earlier.
- TradingAgents has at least 3 live ledger entries.
- October live report committed by CI on 2026-11-02.
- Launch done.
