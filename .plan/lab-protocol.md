# Lab protocol

How every reproduction is run, judged and published. A report that skips a
step says so in its first paragraph.

## Three levels

| Level | Question | Setup |
| --- | --- | --- |
| L1 replication | Do the authors' own code and setup produce their numbers? | Authors' code at a pinned commit, their data window, their metric definitions. Deviations (missing data, dead APIs) are listed. |
| L2 robustness | Does the effect survive an honest protocol? | This repository's engine: signal at the close, fill at the next open, proportional fees, at least 5 seeds, declared trials, survivorship-free or explicitly labeled universe, seed-averaged metrics against 1/N and the paper's own baseline. A policy sees a dated item (filing, news) at the decision for session *t* only if it was published strictly before that session's close (16:00 New York; 00:00 UTC for crypto); filings use EDGAR's acceptance time. `rl-pricing bench check-causal` tests this for every policy. |
| L3 generalization | Does it hold outside the original window? | Later or earlier windows, other universes, and for anything that can be run daily, the live decision ledger. |

A claim can hold at L1 and fail at L2. The verdict table shows the level each
verdict rests on.

## Pre-registration

Before the first run, commit `reports/<id>/preregistration.md` containing:

1. The claims, quoted verbatim with table or page references.
2. For each claim, the metric, window, baseline and the threshold that decides
   the verdict (below).
3. Seeds, costs, data sources with hashes once downloaded, software commits.
4. The smallest effect the test can detect. Rule of thumb: the standard error of
   an annualised Sharpe ratio SR is about √((1 + SR²/(2q)) / years) for q
   periods a year (Lo, 2002; q = 252 for daily returns, so the SR term barely
   matters). That is roughly 0.7 over two years and 2.0 over three months.
   Reports compute it by bootstrap. Until 2026-11-07 this line read
   √((1 + SR²/2) / years), which puts the annual Sharpe where Lo has the
   per-period one and overstates the error at high Sharpe ratios (0.65 instead
   of 0.48 at SR 1.3 over 4.3 years; 8.3 instead of 2.1 at SR 5.6 over three
   months). The R0, R1, R3 and R6 pre-registrations quote the old form in
   their smallest-effect sections; no verdict rule uses it, and each report
   states the corrected figure.
5. Planned analyses. Anything added later is reported under "Deviations",
   with the reason.

The git commit time of the pre-registration is the evidence that the plan came
first. Merge it to `main` in its own pull request, before the code or runs it
governs. A squash merge restamps commits: the R3 and R1 pre-registrations were
first committed at 00:58 and 01:24 UTC on 2026-10-15, but they carry 01:01 UTC
on 2026-10-16 on `main`. The original commits survive only on the branches
`feat/lab-foundations` (R3) and `feat/phase0-followups` (R1 and R3), so those
branches must not be deleted. R0 predates this protocol and is labeled "post hoc".

## Verdicts

One verdict per claim, against the threshold pre-registered for it:

| Verdict | Rule |
| --- | --- |
| ✅ Holds | Same direction, the claimed value lies inside our 95% interval, and (for performance claims) the seed-averaged result beats the stated baseline with bootstrap p < 0.05 |
| ◐ Partly holds | Same direction, but the magnitude shrinks by more than half or significance is lost |
| ✗ Does not hold | Opposite direction, or no seed reaches the claimed value |
| ? Inconclusive | The test cannot detect the claimed effect (window too short, data unavailable); the report says what would decide it |

Wording stays neutral: "the claim does not hold under X". We don't say
"the authors were wrong".

## LLM agents and contamination

- A historical test of an LLM agent counts only if the model's documented
  training cutoff is before the test window. Submissions declare it in
  `model_cutoffs` (also required for any model-produced news score); the
  runner refuses a window that starts earlier unless the submission is
  labelled `contamination: "possible"`, and the leaderboard shows the label.
  Such a result is "possibly contaminated" and cannot earn ✅.
- Every LLM report includes a recall probe: can the model name the test
  period's prices or events without tools? The method follows ai-hedge-fund
  issue #720.
- The authoritative LLM test is the decision ledger. A scheduled job runs the
  agent before the US open (12:00 UTC), commits its target weights, and the
  engine fills them at that open. A missed day is recorded as "hold", never
  back-filled. Model identifiers, prompts and agent commit are pinned for the
  whole window.

## Authors

1. Seven days before publication, email the corresponding author the draft
   report, the evidence and the reproduction command.
2. Print their reply verbatim in `reply.md`, unedited, or note "no reply by
   <date>".
3. If they find an error, fix it, record it under "Corrections" with the date,
   and give them credit.
4. Co-authors of our own work (R0: Xinyi Zhang) get the same notice.

## Statistics

- Seed-averaged returns for significance; per-seed results always published.
- Paired stationary bootstrap for Sharpe differences; deflated Sharpe with the
  number of configurations the original authors and we tried, when known.
- Every report states its smallest detectable effect. Results from many
  configurations get a multiple-comparison note.

## Data and licences

- No redistribution of vendor prices. Reports publish derived metrics,
  equity curves and hashes of the inputs.
- Yahoo data: personal-research use; Binance Vision: CC BY-NC-SA 4.0,
  attributed; licensed vendors: only if the licence allows publishing derived
  results (checked before purchase).

## Report files

```text
reports/<id>/
├── preregistration.md   # committed first
├── claims.json          # [{id, quote, source, metric, claimed, ours, ci95, level, verdict}]
├── report.md            # summary, method, results, deviations, corrections
├── reply.md             # authors' reply, verbatim
└── evidence/            # per-seed results, manifests, reproduce command
```

`reports/README.md` and the main README's verdict table are generated from
every `claims.json`.
