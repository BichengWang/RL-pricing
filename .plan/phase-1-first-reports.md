# Phase 1: first external reports (2026-11-06 → 2026-12-31)

Goal: one external report published on schedule with the authors' reply, the
TradingAgents replication under way, and the integrity work third-party
submissions need.

| ID | Task | Owner | Depends on | Acceptance |
| --- | --- | --- | --- | --- |
| P1.1 | **R1 FinRL ensemble**: pre-registration and its §12 code first committed 2026-10-15 UTC, on `main` since 2026-10-16; data hashes and the L1 build probe moved to P0.19. L1: authors' code (ICAIF 2020 repository, pinned) on 2016-01-04 → 2020-05-08, compared with Sharpe 1.30 and max drawdown −9.7%. L2: toolkit ensemble mode with next-open fills, Dow 30 as of 2016-01-01 from licensed survivorship-free data, at least 5 seeds, turbulence rule on and off. L3: same method 2020-05 → 2025-12 | agent | P0.2, P0.6 | Draft to authors **2026-11-24**; published **2026-12-01** with `reply.md` |
| P1.2 | **R2 TradingAgents replication and contamination**: pre-register by 2026-12-01. L1: authors' code on 2024-01-01 → 2024-03-29 for AAPL, GOOGL, AMZN with the paper's models where available. Recall probe for every model used. Contrast with a model whose documented cutoff predates 2024 | agent | P0.5 (shared adapter) | Draft to authors **2026-12-29**; publication 2027-01-05 (Phase 2) |
| P1.3 | November live report | CI | P0.8 | `leaderboard/reports/2026-11.md` committed 2026-12-02 |
| P1.4 | Moved to P0.18: claim `rl-pricing` on PyPI before the launch | – | – | – |
| P1.5 | ✅ done 2026-10-29 (#110, #111). CI computes historical records for new or changed submissions on merge; contributors no longer submit them. Records carry `submission_sha256`; the freeze job runs `run-all --stale` and also fires when Auto Merge lands a PR that changed a submission (its `GITHUB_TOKEN` merges start no `push` runs; none had run since 10-16). Dispatch run 40919119099 on `main`: nothing stale, nothing to freeze | agent | – | A PR that adds `leaderboard/historical/**` is rejected; merge produces the records |
| P1.6 | `protocol` version on every record, and a rule that retired submissions stay on the board, marked retired | agent | – | Records carry `protocol: 1`; deleting a submission keeps its live records rendered as retired |
| P1.7 | R4 gate: AlphaMix+ versus RD-Agent(Q). Check data licence and do a compute probe for each | agent; owner decides | – | Decision in `decisions.md` by **2026-12-15** |
| P1.8 | Issue templates: "Nominate a claim" (shipped in #87), "Submit a policy" | agent | – | Templates live; first nomination scored with the rubric |

## Status (2026-11-07)

| ID | State | Evidence and gap |
| --- | --- | --- |
| P1.1 | ◐ L1 running; L2/L3 wait for data | L1 choices fixed in [`addendum-2026-11-07.md`](../reports/r1-finrl-ensemble/addendum-2026-11-07.md) (#131) before any run; workflow mode `full` (#130); scorer `evidence/l1.py` (#132). Five runs of the authors' unmodified `run_DRL.py` started 22:31 UTC (Actions run 42420982142, commit `25e03bc`). §12 code is merged, so L2 needs only the Sharadar extract (P0.6, owner): about 4.5 hours for arms A and B at seeds 0–4 on the laptop, 11–12 hours for all 25 runs. **Data must be local by 11-19** to draft on 11-24 |
| P1.2 | not started | Due 12-01. The paper's models are OpenAI's and no OpenAI key is available (decision 2026-10-29); the pre-registration needs the owner's choice of models and an API budget (about $0.65 per ticker-session on Claude, so about $120 for 61 sessions × 3 tickers per model) |
| P1.3 | CI | 12-02 06:00 UTC. First month with the five M3 entrants, published on all three live tasks |
| P1.5 | ✅ | – |
| P1.6 | ✅ done by M1 (#117) | Records carry `protocol_version: "2"` (schema 1 still read); a submission with a `retired` date leaves the ranking and its records are listed apart (`rl/bench/leaderboard.py`) |
| P1.7 | not started | Due 12-15 |
| P1.8 | ◐ | "Submit a policy" template added 11-07; labels `nomination` and `submission` created 11-07 (the nomination template's label had not existed, so earlier issues carry none). No nomination yet |

Carried over from Phase 0, owner: P0.6 Sharadar confirmation and purchase
(now on R1's critical path); P0.1 social preview; P0.10 Show HN, Reddit and the
R1/R2 author heads-up emails; R0's seven days end 11-08, then `reply.md`
records the reply or "no reply by 2026-11-08". From v2: M5 waits for the R6
recall probe with a real key and `LLM_PORTFOLIO_ENABLED`; M4's L1 go/no-go
(11-24) needs a 128 GB host.

## Exit criteria

- R1 published on 2026-12-01 with the authors' reply or a dated "no reply".
- R2 draft with the authors.
- `pip install rl-pricing` works.
- Historical records are CI-computed.
