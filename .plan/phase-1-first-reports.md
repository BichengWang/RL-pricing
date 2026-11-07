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

Updated 23:30 UTC.

| ID | State | Evidence and gap |
| --- | --- | --- |
| P1.1 | ◐ L1 running; L2/L3 ready, waiting for data | **L1**: analysis fixed in [`addendum-2026-11-07.md`](../reports/r1-finrl-ensemble/addendum-2026-11-07.md) (#131) before any run; workflow mode `full` (#130); scorer `evidence/l1.py` (#132) and [`claims.json`](../reports/r1-finrl-ensemble/claims.json) with all 19 claims (#135). The first full run, 42420982142, failed in all five jobs before training (the API zipball answered HTTP 504 to five parallel downloads); the workflow now checks out the pinned commit (#134), and run 42421725185 has trained since 22:45 UTC. Next: download its artifacts, `l1.py --claims`, evidence PR; the failed run is reported (addendum §5). **L2/L3**: analysis fixed in [`addendum-2026-11-07-l2-l3.md`](../reports/r1-finrl-ensemble/addendum-2026-11-07-l2-l3.md) (#136) before any run; `evidence/l2_runs.py` and `evidence/l2.py` merged (#137) and smoke-tested on synthetic prices (prefix check exact over 1,095 sessions; all 14 L2/L3 rows produced). Once the Sharadar extract is local: `python -m rl.marketdata.sharadar` with the ticker map, then `l2_runs.py --data-file <csv> --workers 8` (11–12 hours for all 25 runs), then `l2.py --data-file <csv> --claims reports/r1-finrl-ensemble/claims.json`. **Data must be local by 11-19** to draft on 11-24 |
| P1.2 | ◐ draft, waiting for the owner's model decision | Draft pre-registration on branch `r2-prereg-draft` (`23d6039`), not merged; it becomes the pre-registration once §10 records the owner's choice. Findings below. Due 12-01 |
| P1.3 | CI | 12-02 06:00 UTC. First month with the five M3 entrants, published on all three live tasks |
| P1.5 | ✅ | – |
| P1.6 | ✅ done by M1 (#117) | Records carry `protocol_version: "2"` (schema 1 still read); a submission with a `retired` date leaves the ranking and its records are listed apart (`rl/bench/leaderboard.py`) |
| P1.7 | not started | Due 12-15 |
| P1.8 | ◐ | "Submit a policy" template added 11-07; labels `nomination` and `submission` created 11-07 (the nomination template's label had not existed, so earlier issues carry none). No nomination yet |

**Protocol correction (#138).** The rule of thumb for the standard error of
a Sharpe ratio in [`lab-protocol.md`](lab-protocol.md) put the annual Sharpe
where Lo (2002) uses the per-period one, which overstates the error at high
Sharpe ratios (0.65 instead of 0.48 at R1's claimed 1.30). The R0, R1, R3 and
R6 pre-registrations quote the old form in their smallest-effect sections
only; no verdict rule uses it. R0's published "Lo's formula: 0.90" is 0.71,
recorded under its Corrections; R1, R3 and R6 state the corrected figure in
their reports.

### R2 findings (draft of 2026-11-07)

- **Models.** `o1-preview` (the paper's deep tier) was shut down 2025-07-28;
  `o1` and `o3-mini` on 2026-10-23. `gpt-4o` and `gpt-4o-mini` (cutoff
  October 2023) are the only closed models still served with a cutoff before
  the 2024 window, and only **until 2027-01-14**. No Claude or Gemini model
  with a pre-2024 cutoff is served; Llama 3.1/3.3 ("December 2023") is the
  open-weights option. Facts marked † in the draft were read through search
  summaries and are checked before the draft is committed as a
  pre-registration.
- **Paper.** Table I is the same in all seven arXiv versions: cumulative
  return and Sharpe 26.62 and 8.21 (AAPL), 24.36 and 6.39 (GOOGL), 23.21 and
  5.60 (AMZN). In 11 of 17 cells "CR" equals log(1 + "ARR"), so CR looks like
  a log return; the AMZN figure implies 22.21, not 23.21; the buy-and-hold row
  does not reproduce (AAPL about −8% against −5.23%).
- **Code.** No code existed while the paper was revised: v0.1.0 came
  2025-06-05, two days after v7, and the cached dataset of the backtest was
  never released. v0.6.0 (R3's pin) runs historical dates point-in-time, but
  its news and social sources serve only recent items, so a 2024 run sees
  prices, indicators, SEC filings and macro data without news. L1 therefore
  replicates the released code, not the paper's pipeline.
- **Detectable effect.** With the corrected formula a single stock's Sharpe
  over the window has a standard error of about 2.0 to 2.2; on the realised
  2024-Q1 returns the pooled gap over buy and hold needs about 5.3 to be
  detected, against the paper's 5.54. Most per-ticker claims will likely be
  inconclusive.
- **Prior re-tests.** None reproduces Table I. TradingAgents issue #168
  re-ran AAPL on the paper's window: −25.4%.

## Owner decisions

| # | Decision | Needed by | Why then |
| --- | --- | --- | --- |
| 1 | P0.6: Sharadar written confirmation and purchase, extract on the laptop | **11-19** | L2/L3 take 11–12 hours, then the R1 draft goes to the authors on 11-24. If it slips: draft on 11-24 with L1 only and add L2/L3 later, or move R1 by the delay |
| 2 | P1.2 models (D1), with runs per arm (D2: 1, 3 or 5), news source (D3) and the capability control (D4), as in §10 of the R2 draft | **11-24** | The pre-registration is due 12-01. Option (a) needs an OpenAI key and every run finished before 2027-01-14 |
| 3 | R0: did Xinyi Zhang reply? | 11-08 | `reply.md` then prints the reply or "no reply by 2026-11-08" |

R2 options (cost per run of 183 decisions, from R3's measured tokens; draft
§10):

| Option | Arms | Cost at 1 / 3 / 5 runs | Can earn ✅ | Limits |
| --- | --- | --- | --- | --- |
| (a) OpenAI key: `gpt-4o` + `gpt-4o-mini`, GPT-6 as the newer contrast | P and N | $39 / $117 / $195 | yes | no o1-preview; must finish before 2027-01-14 |
| (b) Anthropic only | A | $122 / $366 / $609 | no: "possibly contaminated", at most ◐ | no pre-2024 contrast, so P1.2's contrast is not met |
| (c) Anthropic plus Llama 3.3 70B | A and L | $126 / $379 / $632 | no for A | needs a Llama host with tool calling (DeepInfra) |

Still open from Phase 0 and v2, owner: P0.1 social preview; P0.10 Show HN,
Reddit and the R1/R2 author heads-up emails; M5 waits for the R6 recall probe
with a real key and `LLM_PORTFOLIO_ENABLED`; M4's L1 go/no-go (11-24) needs a
128 GB host.

## Exit criteria

- R1 published on 2026-12-01 with the authors' reply or a dated "no reply".
- R2 draft with the authors.
- `pip install rl-pricing` works.
- Historical records are CI-computed.
