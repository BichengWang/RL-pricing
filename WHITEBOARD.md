# Whiteboard

Open questions for the owner from the repository audit of 2026-11-21. Each
was reproduced against the code; none is patched, because the fix would change
a pre-registered entrant, a registered rule, published numbers or a documented
design choice. Remove a row once it is decided (and recorded in
[`.plan/decisions.md`](.plan/decisions.md) where it changes direction).

## Needs a decision soon

| # | Question | What was found | Options |
| --- | --- | --- | --- |
| 1 | **Ledger trigger.** Move to an outside `workflow_dispatch` scheduler? | The cron asks for 04:17, 07:17 and 10:17 UTC; since 11-02 GitHub has started one run per weekday, between 11:50 and 13:18 UTC. On 11-17 it started at 13:18 and pushed at 13:30:35. The reopen condition in `decisions.md` (2026-10-18: first run after 13:00 UTC) has been met. From 2027-03-15 the open is 13:30 UTC, so a day like 11-17 would score as holds for every entrant; R3 ends before that, R6's 63 sessions may not | (a) outside scheduler calling `gh workflow run ledger.yml`, as the decision already says; (b) more cron slots as a stopgap. Run time also decides what news the agents see, so (b) is not neutral for R3 |
| 2 | **R6 prompt keeps the oldest information and drops the newest** (`rl/bench/policies/llm_portfolio.py`, `info_lines` and `_prompt`). Fix before R6's first entry? | Items come oldest first and the 15,000-character budget is filled from the front, so once a source fills it the prompt is pinned to its first ~30 items; an item from an hour before the close is the one dropped. `llm-portfolio` declares no information sources today, so its prompt bytes would not change, but its `code_sha256` would, which R6 §11.2 counts as a deviation | Fix now, while R6 has no ledger entries (fill newest first, print in date order, say older items were omitted), or record it as a known limit |
| 3 | **Schema 1 ledger entries skip the commit checks** ([#152](https://github.com/BichengWang/RL-pricing/pull/152), open without the `auto-merge` label) | An edit to a schema 1 entry after its commit is scored as the original decision. The fix is stricter than R3 addendum 2026-11-02 §1 ("keep the registered run-start verification"). All 15 committed schema 1 entries pass the stricter check and re-score identically | Merge #152 with a dated R3 addendum, or close it |

## Cost accounting for R3 (TradingAgents)

| # | Question | What was found |
| --- | --- | --- |
| 4 | Raise `max_cost_per_decision_usd` above $0.70? | Three committed entries cost $0.72 to $0.745 (us-googl, 11-02 to 11-04), so the reservation is not a worst case and the $60 monthly cap can be overshot by the difference. The figure is named in R3 addendum 2026-11-02 §2. Spend is about $41 a month |
| 5 | Bill nothing when the staleness guard refuses before any call? | `TradingAgentsPolicy.act` leaves `last_cost_usd` at 0.0 when `max_age_days` raises before `propagate`, and the ledger bills the $0.70 reservation. `LLMPortfolioPolicy` reports `None` for the same guard and is billed nothing. Decisions are unaffected; R3's code identity would change |

## Method choices

| # | Question | What was found |
| --- | --- | --- |
| 6 | Should the monthly report score multi-seed rows as the leaderboard does? | Monthly scores the seed-averaged return series; the leaderboard prints the mean of per-seed values. On the committed `live/us-sectors/td3` record that is a Sharpe of 6.41 against 3.98. Both now say so (#146), but the first monthly report with 20 or more sessions will print the higher figure next to a link to the board |
| 7 | How should a seed with an undefined Sharpe count? | `leaderboard._mean_sd` uses a plain mean, so one seed that never trades blanks the row's Sharpe and ranks it last; `figures.py` uses `nanmean` for the same quantity. Options: average the defined seeds and show how many, or count a cash seed as 0. No committed record has mixed seeds |
| 8 | Share-trading env: end of data as a time limit, not a terminal state? | `rl/env/env_stocktrading.py` returns `terminated=True` at the end of the data, so SB3 does not bootstrap there; `rl/bench/gym_env.py` returns `truncated=True`. Changing it would change R0 setups A and B and R1 L2/L3 numbers. Suggested: leave it until R1's runs are done, then change it as a declared deviation |
| 9 | Score a retired submission's last partial month? | `ledger.score_all` (and the live track) skip retired submissions, so verified decisions between the last monthly scoring and the retirement date never reach the record. The skip is documented in CHANGELOG 2.0.0; this consequence may not be intended. Nothing is retired yet |

## Integrity design, before R5 records are committed

| # | Question | What was found |
| --- | --- | --- |
| 10 | Should a submission's fingerprint cover information files outside its directory? | `fingerprint()` hashes only files inside the submission directory. The `r5-*` arms declare jsonl sources under `datasets/`, so the file can change without the record or freeze going stale, and `update_live` would replay a frozen policy on back-dated items. Latent: no live task uses an outside source. Hashing the outside file makes the fingerprint differ between the laptop that has `datasets/` and CI, which does not, so CI would see R5 records as stale and could not recompute them. Options: require sources inside the submission directory, or record the source's hash in the record and compare that instead |
| 11 | Make `bench.yml`'s `validate` a required status check on `main`? | The "Reject edits to bot-owned results" guard now also covers committed freezes and renames (#150), but `main` has no required checks, so a labelled PR merges whether the guard passes or not. Auto Merge also merges before `tests` finishes |
| 12 | Is a refreshed EDGAR list acceptable on the live track? | `--refresh` now reaches the information sources (#151). `update_live` replays every session since publication against the panel loaded now, and EDGAR's lists are not strictly point-in-time (reused tickers, removed or reclassified filings, late listings), so a refresh can change past live decisions. No entrant uses sec-edgar on a live task yet |

## Smaller notes

- The portfolio task (`rl/env/env_portfolio.py`) maps `Box[0, 1]` actions through a softmax, so with 30 stocks a weight stays between about 1.25% and 8.6% and the agent cannot hold cash. Inherited from FinRL, undocumented; widening it breaks old portfolio checkpoints.
- On `--data-source synthetic`, `--seed` also picks the market, so a seed sweep mixes agent and data randomness.
- `requirements.txt` pins the versions the suite was last run with; a fresh `pip install -e .` resolves newer ones (pandas 3.1.1, SB3 2.9.2, torch 2.15.0). R1 §5.4 records SB3 2.9.0, so make the L2 runs from the lock.
- `stockstats` 0.6.x calls `set_index(inplace=True)`, which pandas 3.1 warns about (all of the suite's warnings) and a later pandas will reject.
- A manual run of `bench.yml`'s `live` job before 2026-12-01 would re-render `leaderboard/reports/2026-10.md` with the new footnote from #146.
- After a sweep, CI's `run-all --stale` recomputes the tuned record once, because `sweep.json` and the progress file are fingerprinted and change after the record is written.
