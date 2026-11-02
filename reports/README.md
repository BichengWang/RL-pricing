# Reports

This repository re-tests published trading-AI claims under one pre-registered
protocol and publishes the verdicts, with the authors' replies. Each folder
here is one report. The leaderboard and the live track are the instruments
that produce the evidence; the reports are the product.

Protocol in one paragraph: the claims and the rule that decides each one are
committed before the first run (`preregistration.md`); the test uses this
repository's engine (signal at the close, fill at the next open, fees,
at least 5 seeds); per-seed results are published; the authors get the draft
seven days before publication and their reply is printed unedited
(`reply.md`). Wording stays neutral: a claim "does not hold under X"; we do
not say anyone was wrong.

## Verdicts

<!-- reports:start -->
| Report | Claim | Level | Claimed | Ours (95% interval) | Verdict |
| --- | --- | --- | --- | --- | --- |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `ddpg-annual-return` (annual_return) | L2 | 55.5% | 24.6% [17.3%, 31.9%] | ✗ does not hold |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `ddpg-sharpe` (sharpe) | L2 | 1.38 | 1.16 [0.93, 1.39] | ◐ partly holds |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `td3-sharpe` (sharpe) | L2 | 1.37 | 1.10 [0.69, 1.52] | ◐ partly holds |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `sac-sharpe` (sharpe) | L2 | 1.33 | 1.23 [1.07, 1.39] | ◐ partly holds |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `ppo-sharpe` (sharpe) | L2 | 1.09 | 1.17 [0.90, 1.44] | ◐ partly holds |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `a2c-sharpe` (sharpe) | L2 | 0.82 | 1.09 [0.91, 1.27] | ◐ partly holds |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `lstm-max-drawdown` (max_drawdown) | L2 | -9.6% | -26.5% [-29.4%, -23.7%] | ? inconclusive |
| [r0-wang-zhang-2026](r0-wang-zhang-2026/report.md) | `lstm-sharpe` (sharpe) | L2 | 1.19 | 0.75 [0.39, 1.12] | ? inconclusive |

Verdicts: ✅ holds, ◐ partly holds, ✗ does not hold, ? inconclusive. [How a verdict is decided](README.md#how-a-verdict-is-decided).
<!-- reports:end -->

### How a verdict is decided

One verdict per claim, against the threshold pre-registered for it. The
level says what the verdict rests on: **L1** replication (the authors' code
and setup), **L2** robustness (this repository's engine, at least 5 seeds, 1/N
and the paper's own baseline), **L3** generalization (other windows and
universes, the live decision ledger).

| Verdict | Rule |
| --- | --- |
| ✅ Holds | Same direction, the claimed value lies inside our 95% interval, and (for performance claims) the seed-averaged result beats the stated baseline with bootstrap p < 0.05 |
| ◐ Partly holds | Same direction, but the magnitude shrinks by more than half or significance is lost |
| ✗ Does not hold | Opposite direction, or no seed reaches the claimed value |
| ? Inconclusive | The test cannot detect the claimed effect (window too short, data unavailable); the report says what would decide it |

A claim can hold at L1 and fail at L2. LLM agents tested on a window their
model could have seen in training are labeled "possibly contaminated" and
cannot earn ✅; the authoritative test for them is the decision ledger.

## Nominate a claim

Open an issue titled `Claim: <paper or product> – <one-line claim>` with:

1. **The claim, verbatim**, with the table, page or URL it comes from.
2. **The metric and window** the claim is about, and the baseline it beats.
3. **What can be run**: code at a pinned commit, a data source, or only the
   paper's description (then the report can reach L2/L3 but not L1).
4. **Who to notify**: the corresponding author's contact, if known.

We pick claims that are specific (a number, a window, a baseline) and testable
with public data. Reports follow `reports/TEMPLATE.md`; the pre-registration
is committed first and its commit time is the evidence that the plan came
before the results.

## Files of a report

```text
reports/<id>/
├── preregistration.md   # committed first
├── claims.json          # one entry per claim; validated by `rl-pricing bench reports`
├── report.md            # summary, method, results, deviations, corrections
├── reply.md             # authors' reply, verbatim (or "no reply by <date>")
└── evidence/            # per-seed results, manifests, reproduce command
```

`claims.json` is a list of `{id, quote, source, metric, claimed, ours, ci95,
level, verdict, note}`; `verdict` is one of `holds`, `partly`,
`does-not-hold`, `inconclusive`, or `null` while the claim is pre-registered and
untested (such claims are left out of the tables). The verdict table above and
the one in the main README are generated from every `claims.json`:

```bash
rl-pricing bench reports        # validate, then refresh both tables
```

Vendor prices are never redistributed: reports publish derived metrics,
equity curves and hashes of the inputs.
