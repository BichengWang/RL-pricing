# P2.6: arXiv v2 of the paper

Status: proposal of 2026-11-22. Nothing in it is decided. The open decisions
are rows 13 to 23 of [`WHITEBOARD.md`](../WHITEBOARD.md), each ending with its
recommended option. The owner's choices go into one
[`decisions.md`](decisions.md) entry, drafted by 2026-12-06 and merged on
2026-12-21, after the co-author notice period. That defers the rule of
[`README.md`](README.md), "change `decisions.md` first when direction
changes", by three weeks; the rows stay on the whiteboard until the entry
merges. Owner: Bicheng Wang.

## What P2.6 is, and what it is not

The plan's line, in [`phase-2-live-verdicts.md`](phase-2-live-verdicts.md):
"arXiv v2 of Wang & Zhang 2609.31870 with the R0 findings", owned by the owner,
depending on P0.3 (the R0 report), with acceptance "v2 posted". P2.6 replaces
the lab's own paper on arXiv, under the same identifier, inside Phase 2
(2027-01-01 → 2027-03-31). Its preparation (the notice, the registrations, A0
and the code PRs) falls in December 2026, inside Phase 1, and
[`phase-1-first-reports.md`](phase-1-first-reports.md) does not list it yet.
The lab paper P3.1 ([`phase-3-authority.md`](phase-3-authority.md)) is a new
paper, "Do trading-AI results survive? Pre-registered reproductions of N
published claims", across R0–R4 and the protocol. v2 and v3 are about one
paper's claims; P3.1's abstract would not lead with v1.

| Topic | In v2 and v3 | In P3.1 |
| --- | --- | --- |
| v1's claims and the new runs on them | owns them; the new runs are the lab's record under a new report id | cites v3 and the new report in one row and one paragraph; reprints none of their tables or figures |
| R0 | restated as post hoc evidence about v1 | one of N re-tests |
| Protocol, verdict machinery, live track, ledger | used as instruments, in one paragraph | owns them |
| The turbulence rule: R0's setup A against B, and any ablation of it | one sentence; no new run uses the rule | owns it |
| Survivorship-free ETF and crypto tasks, offline RL on crypto, benchmark-wide multiple-comparison and overfitting statistics | at most one paragraph of supporting evidence | owns them |
| Share environment against weights environment (R0's open question) | named as open | owns it |
| A point-in-time universe on licensed data | named as open (row 19) | if data are bought |
| A prospective continuation of the retraining test (E8-P) | registered only | scored by the live track |

One home per result set: if either paper goes to an archival venue, the other
does not reuse its results. Proposed: v2 and v3 stay on arXiv only, and the
owner settles the venue question before P3.1's drafting opens with Phase 3 on
2027-04-01.

## What exists today

From the repository at `e965b20`:

- **R0's verdicts.** [R0](../reports/r0-wang-zhang-2026/report.md) re-tested
  eight claims at L2, five seeds per agent in three setups, under a post hoc
  pre-registration: ✗ for DDPG's 55.5% annual return; ◐ for the Sharpe ratios
  of DDPG, TD3, SAC, PPO and A2C; ? for the LSTM's maximum drawdown and Sharpe
  ratio. No agent beat 1/N with p < 0.05.
- **What R0 did not test**, by its own Deviations: no L1 replication (2); 1/N
  instead of the S&P 500, so no alpha or beta against the index (3); its LSTM
  rows ran the lab's forecaster, not the paper's pipeline (5); rolling
  retraining, leveraged LSTM, cumulative return, and the return and drawdown
  figures of TD3, SAC, PPO and A2C (8).
- **The R0 notice.** [`reply.md`](../reports/r0-wang-zhang-2026/reply.md)
  records it as sent on 2026-11-01 23:19 UTC, the announcement about seven
  hours later as a deviation from the seven-day rule, and no reply yet.
- **The 2021 notebooks** left `main` in `0feaad7` (#84, 2026-10-13) and remain
  in git history ([`docs/toolkit.md`](../docs/toolkit.md), "Changes from the
  2021 version").
- **What the toolkit cannot run yet** ([`docs/toolkit.md`](../docs/toolkit.md)
  and the code). No single-agent walk-forward mode: the only retraining loop is
  the rolling ensemble, whose standalone accounts retrain A2C, PPO and DDPG but
  not TD3 or SAC (`DRLEnsembleAgent`). No same-close execution: every run fills
  at the next open, and a saved run not recorded as `next_open` is refused
  (`rl/autotrain/training.py`). No S&P 500 series is committed in the
  repository, and R0 used none (its Deviation 3); SPY is one of the ten assets
  of the `us-multi-asset` task. `--benchmark SPY` adds an unhashed Yahoo
  buy-and-hold column, and the toolkit's metrics (`rl/trade/metrics.py`)
  include no alpha or beta; `rl/bench/scoring.py` computes them against a
  series it is given. Per-agent hyperparameters can be set only from Python
  (`TrainConfig.model_kwargs`), and evaluation is always deterministic.
- **Data.** No licensed equity data is recorded as bought: P0.6 (Sharadar
  Prices) has no written confirmation or purchase recorded, and it covers
  prices only. No source in the repository or its data plan supplies historical
  index membership or market capitalisation.

## The research round of 2026-11-21 and 2026-11-22

A research round prepared this proposal. It made a claim-by-claim inventory of
v1 (95 claims), recovered the 2021 setup from git history, recomputed R0's and
the README's numbers from committed records, and built a bibliography of 223
entries, each checked against a primary record. Each of its eleven reports on
the repository and on v1 was checked a second time. A design round followed:
three rival designs, three referee-style judgements, seven experiment
specifications and three critical reviews. Nothing was trained or backtested,
and nothing in the repository was changed. The notes, the draft registrations
and the draft corrections are kept outside the repository until the co-author
notice period ends.

## Proposed shape: two replacements under one title

Title proposed for both (row 15): *Deep Reinforcement Learning for Equity
Trading: Corrections and a Multi-Seed Re-Examination*. It keeps v1's opening
words for readers who reach the paper by its bare identifier, names no method
that stage 1 does not test, and stays true if P2.6 closes on stage 1.

| Stage | Version | Contents, by category | Needs | Gates | Posts |
| --- | --- | --- | --- | --- | --- |
| 1 | arXiv v2 | Corrections to v1, each traced to committed records; R0 restated as post hoc evidence, read by the rules of an analysis-only registration (A0); reference points from committed records; a claim table giving every v1 claim its v2 status, with untested claims marked untested; the stage-2 registration cited by commit, with no promise that it will run | no runs and no new data; A0 (under 0.2 CPU-h) | notice of 2026-12-07 and its 14 days; arXiv's written answer; A0 run from its merged registration; corrections merged (gates 1 to 4) | 2027-01-04 (fallback 2027-01-11) |
| 2 | arXiv v3 | The registered runs on v1's window and on a later window; alpha and beta against SPY (E4); the L1 re-run of the 2021 code (E2) if its container works; the new report under its own id; the updated claim table | the core runs (97.27 CPU-h) from 2027-01-16 | stage-2 registration merged before its code; E3-0 passes; run cut-off 2027-02-07; second notice from 2027-02-22; new report published by 2027-03-22 | 2027-03-08 (latest 2027-03-29) |

The owner holds both stages: the decisions, both notices, every merge and both
postings.

**Why two rather than one.** The bare identifier resolves to the latest
version, so v1, with the 55.5% headline that R0 marks ✗, stays the version of
record until it is replaced. Stage 1 needs no laptop time, which R1's runs and
then R5's L2 hold until 2027-01-15. The stage-2 registration, merged and cited
by stage 1, commits the design in public before any result exists. The risks
(the 2021 container, the calendar, the notice) do not fall on one date. One
replacement in March would be cleaner for citations, but would leave v1's
abstract up about nine more weeks (2027-01-04 against 2027-03-08). Two
replacements cost two notices, two arXiv Comments fields and citations to an
interim version; that cost is bounded because stage 1 is complete on its own
and the title does not change.

**Cut-off.** Stage 1 is P2.6 as worded and is written to stand alone as the
final version. If the new report is not published by 2027-03-22, a
`decisions.md` line closes P2.6 on stage 1 and makes stage 2 its own plan
item. The latest stage-2 posting inside P2.6's window is 2027-03-29.

## Gates before any v2 draft is public

1. **Co-author notice.** One notice on 2026-12-07 carries every draft with the
   evidence and the reproduction commands, as the protocol asks
   ([`lab-protocol.md`](lab-protocol.md), Authors): the stage-1 text with a
   claim-by-claim change list, the A0 and stage-2 registrations, the
   corrections, and the draft `decisions.md` entry. No v2 draft, registration
   or correction is merged, pushed or opened as a pull request before
   2026-12-21, 14 days later, twice the protocol's seven (row 16): a pull
   request with the `auto-merge` label merges within a minute, and the
   repository is public. A change to a verdict rule, family or arm in those
   days restarts the 14 days for the registration; any other change goes to
   the co-author as a dated supplement. The new report and v3 get their own
   14-day notice from 2027-02-22.
2. **arXiv's written answer.** Questions on the replacement procedure and on
   the cross-lists (row 23) go to arXiv support in writing by 2026-12-04.
   Stage 1 posts only after the written answer arrives, and follows it; the
   answer is archived with the new report.
3. **Registration before code.** On 2026-12-21, each in its own docs-only PR,
   in this order: the `decisions.md` entry; A0's registration; the
   corrections; the stage-2 registration with a `claims.json` of null
   verdicts. A0's code and run follow its registration, and stage 1 does not
   post without them. The stage-2 code follows the stage-2 registration
   ([`lab-protocol.md`](lab-protocol.md), Pre-registration). After that, only
   a dated addendum merged before the first fit may settle an implementation
   detail; it names the code commit and data hashes and changes no claim,
   threshold, verdict rule, family, arm, seed, window or cut-off.
4. **Corrections to the R0 report and the README.** The research round lists
   26. They are drafted before the notice and sent with it. All but two merge
   in one docs-only PR from 2026-12-21, before stage 1 posts, the R0 ones under
   R0's dated Corrections section; one merges when stage 1 posts, and one with
   the new report. R0's `preregistration.md` and `claims.json` are not edited.

**Wording.** Both windows have been seen: v1 and R0 scored 2019–2020, and the
people running the tests lived through 2021–2025. v2, v3 and the new report
therefore say "registered before the runs", never "pre-registered", and mark
each result registered (commit, date) or not. The one exception is E8-P, whose
sessions lie after its freeze. The new report's registration keeps the
protocol's file name, `preregistration.md`.

## Experiment programme

Ids are working ids; the registration fixes them. Compute is at R1's probe
rates on the M5 laptop (A2C 74 s, PPO 65 s, DDPG 650 s per 100,000-step fit,
TD3 and SAC assumed at DDPG's rate), 2–4× wall time under contention.
Predictions and verdict rules belong to the registration.

| Id | Question | Fits | CPU-h | Depends on | Core or optional |
| --- | --- | --- | --- | --- | --- |
| A0 | What can be said about v1 from committed records alone? Every reading post hoc | 0 | under 0.2 | its own registration | stage 1, required |
| E3-0 | Does R0 setup B reproduce at HEAD on a fresh hashed download? | 27 (+25) | 2.48 (+2.32) | hashed downloads; bridge scorer | core; gate for stage 2 |
| E1 | On 2019–2020, does one refit before 2020 change each agent's result against training once? | 205 | 23.79 | E3-0; driver and switches; throughput probe | core |
| E1-H | At a second off-policy configuration, fixed in the registration, where do v1's trained-once DDPG, TD3 and SAC values fall in the seed distribution? | 60 | 10.83 | E1's code | core |
| R-S1 | Holding each trained-once policy fixed, how much do same-close fills move its result? | 0 | minutes | E1, E1-H; same-close transform | outside the core, zero fits |
| E8 | From 2021-07-01 to 2025-12-31, after v1's window, does yearly forward retraining change each agent's result against training once? | 502 (fallback 252) | 58.23 (29.21) | E3-0; E1's code; data gates; hashed SPY and T-bill series | core |
| E8-B | Does E8's retraining contrast depend on the price-adjustment basis? | 100 | 1.93 | E8 | core |
| E4 | What are alpha and beta against the S&P 500 for R0's runs and the new arms, and how do 1/N and random portfolios of the 20 stocks compare with SPY? | 0 | seconds | hashed SPY, `^GSPC` and T-bill series | outside the core, zero fits |
| E2 | Where do v1's Table 4 values fall among replicates of the 2021 code, re-run in a pinned container? | 150 + 10 | 19.22 at R1's rates | the container | outside the core, on GitHub Actions |
| E1-S (training arm) | How much does training under same-close fills add to R-S1's effect? | 50 | 5.80 | R-S1; date rule | optional |
| E1-L | Does the download's price-adjustment basis move the trained-once result on 2019–2020? | 52 | 6.00 | E1; date rule | optional |
| E11 | Does a 400,000-step budget change trained-once DDPG and SAC? | 10 | 7.22 | E1's code; date rule | optional |
| E8-H | On E8's window, how do trained-once DDPG, TD3 and SAC do at E1-H's configuration? | 30 | 5.42 | E8; date rule | optional |
| E8-D | Does E8's retraining contrast change at the library's default hyperparameters? | 250 | 29.01 (a lower bound) | E8; date rule | optional |
| E3-0P | Follow-up of E3-0, run only if E3-0 calls for it | 25 | 2.32 | E3-0 | conditional |
| E8-P | Does the retrained arm hold on sessions after a 2027 freeze? | 100 | 11.6 | E8; a frozen-policy replay for the share environment | registration only; scored later by the live track |

**Totals**, from the unrounded per-item figures. The proposed core (E3-0, E1,
E1-H, E8, E8-B) is 894 fits and 97.27 CPU-h: 12.2 h on 8 workers, 24.3–48.6 h
at 2–4× contention, or 68.25 CPU-h with E8 at its 10-seed fallback. The
optional items add 392 fits and 53.46 CPU-h, for 1,286 fits and 150.73 CPU-h
on the laptop. E3-0P (25 fits, 2.32 CPU-h, only if triggered), E2 (160 fits,
on GitHub Actions) and E8-P (registered only) are not in these totals. Replays
and controls cost no fits.

**Order.** E3-0 → E1 → E1-H (with R-S1) → E8 block 1 → E8 block 2 → E8-B →
the optional items (E1-S, E1-L, E11, E8-H, E8-D), each started only if every
item ahead of it has finished and the date is no later than 2027-02-01. E4 and
E2 run in parallel. At the cut-off proposed for the registration, 2027-02-07,
an unfinished block is dropped by date. The scorer runs once, after every
planned run or the cut-off. Not proposed for v2 or v3: E3-B (a one-switch
ladder, 22.7 CPU-h), E6 (a re-run of v1's Table 3 forecasters), E7 (a
comparison of DDPG's and TD3's sensitivity to hyperparameters), E10 (the share
environment against the weights environment, to P3.1) and E1-Q (a quarterly
retraining arm, 700 fits, 81.2 CPU-h). E5 (the forecast strategy) and E9 (a
point-in-time universe) wait on rows 20 and 19.

## Code and data to prepare

Each item is its own PR. Code merges after the registration it serves; the
paper's source merges after the notice period.

| Item | Needed by |
| --- | --- |
| A0's scripts in the new report's `evidence/a0/`, with a CI job | A0 |
| Bridge scorer, and a runner that refuses to start without a passing bridge check on `main` | E3-0 |
| Single-agent walk-forward driver for all five agents, split into a fit-only and a trade-only step, with slice settings and boundary rules | E1, E8 and their variants |
| Switches: hyperparameters from the command line, sampled or deterministic evaluation, and `--no-run-stats` (a run writes its account, action and execution files but no performance table or plot) | every stage-2 run |
| Collector and paired scorer, built on R1's [`l2_runs.py`](../reports/r1-finrl-ensemble/evidence/l2_runs.py) and [`l2.py`](../reports/r1-finrl-ensemble/evidence/l2.py), with SciPy pinned in `requirements.txt` in the same PR | E1, E1-H, E8, E8-B |
| Price-file builders with their checks, the same-close transform, and E8's data gates | E1, R-S1, E1-S, E1-L, E8, E8-B |
| Zero-fit replays and learning controls: an untrained-network policy, an equal-dollar policy through the environment, divergence and support diagnostics | E1, E1-H, E8 |
| Hashed downloads of the 20 stocks' prices, SPY and `^GSPC`, on the pattern of R1's [`fetch_inputs.py`](../reports/r1-finrl-ensemble/evidence/fetch_inputs.py), and of FRED's DTB3 through R1's [`l1.py`](../reports/r1-finrl-ensemble/evidence/l1.py); each file's hash and download date are recorded, and the file itself is never committed | E3-0, E1, E4, E8 |
| Tests for prefix identity and same-seed determinism of `run_training` (untested today), and for the gradient-update count | every stage-2 run |
| A container for the 2021 code: notebook export, the pinned 2021 software stack, a replacement downloader declared as a deviation, a rescaler and a collector, run on GitHub Actions like R1's L1 | E2 |
| The paper's source under `paper/`, outside the folders the site builds from (`mkdocs.yml`), after a PR that adds `.gitignore` rules for `.aux`, `.bbl`, `.out` and `.tar.gz` | stage 1, from 2026-12-21 |

## Timeline

All dates are proposals.

| Dates | Work | Beside it |
| --- | --- | --- |
| 2026-11-23 → 11-30 | Owner reads the design in the private notes, decides rows 13 to 16 and, under row 14, names the new report's id; no v2 work in the repository | R1 draft to its authors (P1.1), the R2 decisions (Phase 1 owner decision 2) and R5's L1 go/no-go (R5 pre-registration §14) on 11-24; R1 published and the R2 pre-registration due 12-01 |
| 12-02 → 12-06 | Rows 17 to 23 decided, row 23 before the arXiv support questions go out in writing by 12-04; A0's registration, the stage-1 draft with its change list, the stage-2 registration draft, the 26 corrections, the draft `decisions.md` entry, and the evidence and reproduction commands for the notice. By 12-04 the owner confirms the 12-07 notice or moves to the fallback | drafts on unpushed local branches |
| **12-07 (Mon)** | **Co-author notice**, every draft attached | 14 days before any v2 merge |
| 12-08 → 12-20 | Private revisions only; no v2 work on 12-15 | R4 gate and R5 L1 on 12-15 |
| **12-21 (Mon)** | Merges, each its own docs-only PR: `decisions.md` entry; A0 registration; corrections; stage-2 registration | before any code |
| 12-22 → 12-23 | A0 code PR, run and outputs PR; outputs sent to the co-author as a supplement | – |
| 12-22 → 2027-01-15 | Code PRs; the E2 container; hashed downloads; E8's data gates; the addendum naming code and hashes | laptop held by R5 L2 |
| 12-28 → 01-02 | Stage-1 text final, with A0's registered values; a PR prepared that adds "v1" to the repository's citations of 2609.31870, because the bare identifier will then resolve to v2 (the citation PR) | – |
| **2027-01-04 (Mon)** | **Post stage 1 (arXiv v2)** after arXiv's written answer; the citation PR merged | R2 published 01-05 |
| 01-16 → 01-18 | E3-0 and the throughput probe | laptop free after 01-15 |
| 01-18 → 02-07 | The queue; optional items only if started by 02-01; E4; E2 on Actions. **Cut-off 02-07** | core 24.3–48.6 h wall |
| 02-08 → 02-21 | Score once; write the new report and the stage-2 text | R3 draft to its authors 02-08, published 02-15 |
| **02-22 (Mon)** | **Co-author notice** for the new report and stage 2 | 14 days; R6 due 03-01 |
| **03-08 (Mon)** | **Publish the new report; post stage 2 (arXiv v3)** | R4 draft to its authors the same day, published 03-15 (P2.3); needs the March waiver (row 18) |
| 03-22 | Cut-off: if the report is not published, P2.6 closes on stage 1 and stage 2 becomes its own item | – |
| 03-29 (Mon) | Latest stage-2 posting inside P2.6 | Phase 2 ends 03-31 |

**Fallback**, if the first week of December cannot carry the drafting: notice
on 2026-12-14, merges on 2026-12-28, code PRs 12-29 → 2027-01-15, A0 run
12-29 → 12-30, stage 1 on 2027-01-11. The stage-2 dates do not move.

## Compute and calendar

Every L2 run goes on the M5 laptop, so the bridge, E1 and E8 share one
platform; GitHub Actions runs only E2's container. No API spend, licensed data
or large-memory host is needed. R1's runs and then R5's L2 hold the laptop
until 2027-01-15; the report cap, the March waiver and the December A0
exception are in row 18. Stage 1 is a paper version, not a report. The first
week of December carries 7–8 days of owner review in 5 calendar days; if the
owner cannot confirm the 12-07 notice by 2026-12-04, the fallback applies.

## Risks

| Risk | Mitigation |
| --- | --- |
| A v2 draft becomes public before the notice period ends | Drafts stay on unpushed local branches until 2026-12-21; no pull request carries a draft before then |
| arXiv's answer is late, or asks for more than the plan covers | Stage 1 waits for the written answer and follows it |
| The laptop is not free on 2027-01-16 (R1's data or R5's L2 slips) | Stage 1 needs no runs; E8's 10-seed fallback (core 68.25 CPU-h); optional items start only by 02-01; the 02-07 cut-off drops unfinished blocks; the 03-22 cut-off closes P2.6 on stage 1 |
| Off-policy fits run slower than R1's probe | A throughput probe on synthetic prices before the runs; it changes the wall-time forecast only |
| The 2021 container does not build | E2 is not a gate; v3 says that no L1 re-execution exists, and why |
| The bridge or a data gate fails | The stage-2 runs stop, with no data substitution; stage 1 is unaffected |
| A result is seen before the scorer runs | `--no-run-stats` on every job, exit codes only from the runner, one scoring pass |
| The co-author points to an error | Fixed, recorded under Corrections with the date and credited before the stage posts ([`lab-protocol.md`](lab-protocol.md), Authors) |
