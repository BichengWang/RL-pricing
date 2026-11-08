# R0 — Re-testing Wang & Zhang (2026), "Deep Reinforcement Learning for Equity Trading"

**Read first.** This is a re-test of our own paper
([arXiv:2609.31870](https://arxiv.org/abs/2609.31870) v1; PDF of an earlier
version in `docs/Wang_Zhang_2021_Deep_Learning_Stock_Trading_arxiv.pdf`, see
[Claims](#claims)) on its own 20 stocks and 2019–2020 window. Under this repository's execution model (signal
at the close, fill at the next open, 0.1% fee per side), the reported 55.51%
annual return for DDPG does not hold: no run of any agent in any setup
reaches it (best of 75 runs: 46.0%). The reported Sharpe ratios of the five
agents hold only partly: the claimed values lie inside, or at the best seed
of, the range of our five seeds, but the seed means are below the claim for
DDPG, TD3 and SAC and above it for PPO and A2C, the agents' order differs, and none of the five beats 1/N buy and hold
with bootstrap p < 0.05 (seed-averaged returns; best p is 0.133, SAC). The two LSTM claims are inconclusive,
because the LSTM we could run is not the paper's pipeline. Steps of the
protocol this report skipped or changed: the pre-registration is **post hoc**
([`preregistration.md`](preregistration.md)); no L1 replication (the paper's
PDF gives no fill timing or fees, and our runs fix both); the S&P 500 baseline was
replaced by 1/N; the co-author notice was sent on 2026-11-01 23:19 UTC, about
seven hours before the announcement instead of seven days ([`reply.md`](reply.md)). Nothing below says anyone was wrong; it says
which claims hold under which setup.

Numbers are cited as **[S]** `evidence/paper-2021-share-env.json` (per-run
metrics), **[E]** `evidence/paper-2021-share-env-equity.json` (daily account
values of the same 50 runs and of 1/N),
**[B]** `leaderboard/historical/paper-2021/<submission>.json` and
**[sum]** `evidence/summary.json`; every value in [sum] and in
`evidence/tables.md` is computed from [S], [E] and [B] by
`evidence/analyze.py` (`rl.bench.scoring`, `rl.bench.leaderboard`).

## Summary

Generated from [`claims.json`](claims.json); level L2 throughout. "Ours" is
setup A (share env + turbulence rule), the seed mean with its 95% Student-t
interval over 5 seeds; the LSTM rows come from setup C.

| Claim | Level | Claimed | Ours (95% interval) | Verdict |
| --- | --- | --- | --- | --- |
| `ddpg-annual-return` | L2 | 55.5% | 24.6% [17.3%, 31.9%] | ✗ does not hold |
| `ddpg-sharpe` | L2 | 1.38 | 1.16 [0.93, 1.39] | ◐ partly holds |
| `td3-sharpe` | L2 | 1.37 | 1.10 [0.69, 1.52] | ◐ partly holds |
| `sac-sharpe` | L2 | 1.33 | 1.23 [1.07, 1.39] | ◐ partly holds |
| `ppo-sharpe` | L2 | 1.09 | 1.17 [0.90, 1.44] | ◐ partly holds |
| `a2c-sharpe` | L2 | 0.82 | 1.09 [0.91, 1.27] | ◐ partly holds |
| `lstm-max-drawdown` | L2 | -9.6% | -26.5% [-29.4%, -23.7%] | ? inconclusive |
| `lstm-sharpe` | L2 | 1.19 | 0.75 [0.39, 1.12] | ? inconclusive |

Verdicts by setup, same rule everywhere ([sum] `verdicts_by_setup`):

| Claim | A: share env + rule | B: share env, no rule | C: benchmark task |
| --- | --- | --- | --- |
| `ddpg-annual-return` | ✗ | ✗ | ✗ |
| `ddpg-sharpe` | ◐ | ✗ | ✗ |
| `td3-sharpe` | ◐ | ◐ | ✗ |
| `sac-sharpe` | ◐ | ◐ | ✗ |
| `ppo-sharpe` | ◐ | ◐ | ✗ |
| `a2c-sharpe` | ◐ | ◐ | ◐ |

What would change a verdict: a run of the paper's own LSTM pipeline under this
engine decides the two `?` rows. For the five Sharpe claims, ✅ needs
p < 0.05 against 1/N; the best seed-averaged result is SAC's (A: p = 0.218;
B: p = 0.133), so a longer window or other universes would be needed for that.

## Claims

Quoted verbatim, with table and page, in
[`preregistration.md`](preregistration.md#claims) (and `quote`/`source` in
`claims.json`). All concern the 2019–2020 test set. Metrics: annual return
(CAGR), Sharpe ratio (mean/std × √252, risk-free rate 0), maximum drawdown.
The quotes and page numbers come from the PDF in `docs/`, an earlier version
titled *Applying Deep Learning to Stock Trading: LSTM Forecasting and Deep
Reinforcement Learning*; arXiv v1 has the same Table 4 numbers and the same
quoted p. 6 text, and differs in the abstract's wording (its LSTM drawdown
reads "(9.6%)" for "(−9.6%)").

## Method

- **Level**: L2 (this repository's engine, 5 seeds). No L1 run.
- **Setup A** — share env + turbulence rule: `python main.py --mode train`,
  the paper's share-trading environment (the paper's formulation: action = shares to trade per stock, reward =
  portfolio value change), now with next-open fills and 0.1% fees. The
  environment's default rule liquidates everything when a turbulence index
  exceeds the 0.99 quantile of its in-sample values (`--turbulence-quantile`,
  `rl/cli.py`). 20 stocks, train 2000–2018, trade 2019–2020, A2C / PPO / DDPG /
  TD3 / SAC, 80,000 timesteps, default hyperparameters, seeds 0–4. 25 runs. [S]
  `runs` with `turbulence_rule: true`.
- **Setup B** — the same with `--no-turbulence`. 25 runs. [S] `turbulence_rule: false`.
- **Setup C** — benchmark task `paper-2021`: this repository's weights
  environment (long-only weights with cash, decisions at the close, fills at
  the next open), same stocks and window, validation 2017–2018 to choose the
  checkpoint, 5 seeds per learned policy, plus `lstm-forecast` and six
  baselines. [B], commit `68c03c7f69c6e1ccb0eecaf6ba6d3c25cde6e2c0`,
  created 2026-10-13. Agent settings are in `submissions/<id>/submission.json`
  (100,000 timesteps for DDPG/TD3/SAC, 200,000 for PPO/A2C).
- **Data**: Yahoo Finance daily prices, 20 stocks chosen in 2021 by market
  capitalisation (survivorship-biased by construction, as the paper itself
  says in its Limitations). SHA-256 of the share-env input
  `1d65373352576fa8f30db279f13dd06201bb1c2105054e9adfcbc1cbd96f544f`
  (105,680 rows, [S] `data`); of the benchmark input
  `d16bbbd2b2cf0835a66d51029d449b1e62fe01c1221a27bae652dc8ba0df09e8`
  ([B] `data.sha256`). No prices are redistributed.
- **Baseline**: 1/N buy and hold of the same 20 stocks, because no S&P 500
  series is in our evidence. A/B: Sharpe 1.1049, annual return 29.66% ([S]
  `equal_weight_buy_and_hold`). C: Sharpe 1.11, annual return 29.9%, max
  drawdown −29.4% ([sum] `benchmark.one_over_n`).
- **Seeds and trials**: 5 seeds, seed mean ± sd, 95% Student-t interval
  (t = 2.776, 4 degrees of freedom). One configuration per agent and setup
  (default hyperparameters, no tuning); the authors' trial count is unknown.
- **Significance**: for each setup, Sharpe of the seed-averaged daily returns
  against 1/N, one-sided paired stationary bootstrap
  (`rl.bench.scoring.sharpe_difference`, 2,000 resamples). Setups A and B use the
  daily account values in [E]; setup C uses [B].
- **Smallest detectable effect**: the bootstrap standard error of the
  annualised Sharpe ratio over these 505 sessions is 0.69 (Lo's formula:
  0.71; 0.90 before the correction of 2026-11-07) ([sum] `detectability`). A one-sided 5% test needs about 1.13. The
  claimed edge of the best agent over 1/N (1.38 vs 1.10, a difference of 0.28)
  is far below this: the window cannot confirm or reject a Sharpe advantage of
  that size, so rule 1 (no seed reaches the claimed value) is the only way a
  claim is rejected here.

## Results

All tables are from `evidence/tables.md` ([sum] `per_setup`), computed from [S]
and [B]. Reported = the paper's Table 4.

### Sharpe ratio, mean ± sd over 5 seeds [min, max]

| Agent | Reported | A: share env + rule | B: share env, no rule | C: benchmark |
| --- | --- | --- | --- | --- |
| DDPG | 1.38 | 1.16 ± 0.19 [0.92, 1.43] | 1.06 ± 0.18 [0.80, 1.24] | 0.70 ± 0.17 [0.57, 0.97] |
| TD3 | 1.37 | 1.10 ± 0.34 [0.57, 1.37] | 1.00 ± 0.30 [0.53, 1.37] | 0.70 ± 0.21 [0.43, 1.01] |
| SAC | 1.33 | 1.23 ± 0.13 [1.07, 1.42] | 1.23 ± 0.18 [1.01, 1.49] | 0.66 ± 0.26 [0.19, 0.82] |
| PPO | 1.09 | 1.17 ± 0.21 [0.80, 1.32] | 0.95 ± 0.21 [0.74, 1.21] | 0.98 ± 0.04 [0.93, 1.02] |
| A2C | 0.82 | 1.09 ± 0.15 [0.91, 1.27] | 0.97 ± 0.15 [0.80, 1.16] | 1.03 ± 0.01 [1.01, 1.05] |
| LSTM forecast | 1.19 | – | – | 0.75 ± 0.29 [0.45, 1.08] |
| *1/N buy and hold* | – | *1.10* | *1.10* | *1.11* |

### Annual return, mean ± sd over 5 seeds [min, max]

| Agent | Reported | A: share env + rule | B: share env, no rule | C: benchmark |
| --- | --- | --- | --- | --- |
| DDPG | 55.51% | 24.6% ± 5.9 [15.9%, 30.4%] | 30.6% ± 8.3 [18.5%, 40.5%] | 16.2% ± 5.4 [12.0%, 25.0%] |
| TD3 | 42.00% | 23.3% ± 9.4 [8.8%, 31.3%] | 28.7% ± 12.3 [10.8%, 44.9%] | 15.7% ± 5.8 [8.5%, 24.4%] |
| SAC | 38.95% | 25.1% ± 2.5 [21.1%, 27.8%] | 38.5% ± 7.0 [27.8%, 46.0%] | 15.1% ± 7.6 [1.6%, 19.7%] |
| PPO | 27.89% | 16.0% ± 4.7 [10.6%, 22.3%] | 25.3% ± 7.9 [15.6%, 33.3%] | 23.9% ± 1.2 [22.3%, 25.4%] |
| A2C | 20.47% | 21.5% ± 5.1 [16.5%, 28.2%] | 26.0% ± 6.5 [19.0%, 35.4%] | 25.1% ± 0.4 [24.7%, 25.8%] |
| LSTM forecast | 16.99% | – | – | 17.9% ± 9.5 [9.1%, 28.3%] |
| *1/N buy and hold* | – | *29.7%* | *29.7%* | *29.9%* |

### Maximum drawdown, mean over 5 seeds [worst, best]

| Agent | Reported | A: share env + rule | B: share env, no rule | C: benchmark |
| --- | --- | --- | --- | --- |
| DDPG | −33.52% | −18.0% [−18.6%, −17.4%] | −27.8% [−30.8%, −25.7%] | −29.3% [−31.7%, −27.4%] |
| TD3 | −25.24% | −19.0% [−20.2%, −17.5%] | −28.2% [−32.7%, −25.0%] | −29.3% [−33.7%, −26.5%] |
| SAC | −24.19% | −17.7% [−21.8%, −14.8%] | −27.0% [−32.2%, −22.5%] | −30.2% [−33.7%, −28.1%] |
| PPO | −25.33% | −9.2% [−11.7%, −6.7%] | −27.4% [−32.0%, −25.3%] | −28.1% [−28.6%, −27.6%] |
| A2C | −31.11% | −17.2% [−18.9%, −14.5%] | −29.0% [−31.2%, −26.5%] | −28.2% [−28.3%, −28.0%] |
| LSTM forecast | −9.57% | – | – | −26.5% [−28.8%, −23.8%] |
| *1/N buy and hold* | – | – | – | *−29.4%* |

### Sharpe against 1/N, seed-averaged returns (one-sided paired stationary bootstrap)

Seed-averaged daily returns of the 5 seeds against 1/N. A and B: [E]
`runs[].equity` and `equal_weight_buy_and_hold`, with alpha, beta and the
Newey-West *t* of the seed-averaged series. C: [B] `runs[].equity` and
`benchmark.equity`; alpha, beta and *t* as in `leaderboard/LEADERBOARD.md`
(per-seed values averaged). Source for all rows: `evidence/tables.md`, [sum]
`per_setup`.

| Setup | Agent | Sharpe of seed-avg returns | ΔSharpe vs 1/N | p | Alpha | Alpha t | Beta | Seeds above 1/N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | DDPG | 1.22 | +0.12 | 0.320 | 7.4% | 0.9 | 0.56 | 3/5 |
| A | TD3 | 1.20 | +0.09 | 0.328 | 6.4% | 0.9 | 0.55 | 3/5 |
| A | SAC | 1.33 | +0.22 | 0.218 | 8.9% | 1.2 | 0.52 | 4/5 |
| A | PPO | 1.26 | +0.15 | 0.339 | 6.7% | 1.1 | 0.30 | 4/5 |
| A | A2C | 1.14 | +0.04 | 0.398 | 5.4% | 0.7 | 0.54 | 2/5 |
| B | DDPG | 1.11 | +0.00 | 0.496 | 1.2% | 0.3 | 1.00 | 3/5 |
| B | TD3 | 1.06 | −0.04 | 0.654 | −0.2% | −0.0 | 0.99 | 1/5 |
| B | SAC | 1.32 | +0.22 | 0.133 | 7.5% | 1.6 | 1.00 | 4/5 |
| B | PPO | 1.00 | −0.11 | 0.713 | −2.0% | −0.4 | 0.95 | 1/5 |
| B | A2C | 0.99 | −0.11 | 0.792 | −2.3% | −0.6 | 0.98 | 1/5 |
| C | DDPG | 0.71 | −0.40 | 0.985 | −9.9% | −2.2 | 0.95 | 0/5 |
| C | TD3 | 0.71 | −0.40 | 0.977 | −9.4% | −1.7 | 0.91 | 0/5 |
| C | SAC | 0.68 | −0.43 | 0.998 | −10.8% | −2.2 | 0.95 | 0/5 |
| C | PPO | 0.98 | −0.13 | 0.898 | −2.9% | −1.2 | 0.92 | 0/5 |
| C | A2C | 1.03 | −0.08 | 0.825 | −1.8% | −0.8 | 0.92 | 0/5 |
| C | LSTM forecast | 0.84 | −0.27 | 0.831 | −4.4% | −0.5 | 0.81 | 0/5 |

### Sessions with unchanged equity (the rule's cash days)

Sessions on which a run's equity does not change (|daily return| < 1e-9);
[E] `runs[].equity`, [sum] `share_env_equity.cash_days`.

| Setting | Runs | Runs with no such session | Distinct sets of such sessions | Sessions |
| --- | --- | --- | --- | --- |
| A: share env + rule | 25 | 0 | 1 | 2020-03-17, 2020-03-20, 2020-03-23, 2020-03-24, 2020-03-25 |
| B: share env, no rule | 25 | 25 | 1 (empty) | – |

### Per-seed results

Per-seed Sharpe ratios and annual returns for every setup and agent are in
[`evidence/tables.md`](evidence/tables.md) (from [S] `runs[].sharpe`,
`runs[].annual_return` and from [B] `runs[].equity`). Seeds above the 1/N
Sharpe of 1.10 in setups A / B ([sum] `share_env_pooled.sharpe_beating_one_over_n`):
DDPG 3/5 and 3/5, TD3 3/5 and 1/5, SAC 4/5 and 4/5, PPO 4/5 and 1/5, A2C 2/5 and 1/5.

### Pooled share-env runs

Across the 50 runs of A and B the best annual return is 46.0% (SAC, no rule,
seed 0) and the median 26.1% ([sum] `share_env_pooled`); 1/N earned 29.7%.

## Verdicts

Rule: [`preregistration.md`](preregistration.md#verdict-rule-as-applied).
The verdict in `claims.json` is setup A's (the default of the authors' own
training script, and for every agent the setup with the higher or equal seed-mean
Sharpe, so the one most favourable to the claims).

**`ddpg-annual-return` — ✗ does not hold.** Claimed 55.51%. The best of the
15 DDPG runs is 40.5% (B, seed 3) and the best of all 75 DRL runs is 46.0%
(SAC, B, seed 0; [sum] `pooled_75`), so no seed reaches the claimed value in any setup. DDPG's
seed-mean annual return (A 24.6%, B 30.6%, C 16.2%) is below 1/N (29.7% / 29.7% /
29.9%) in A and C and 0.9 points above it in B, and DDPG is not the agent with
the highest seed-mean return in any setup (A: SAC 25.1%; B: SAC 38.5%; C: A2C
25.1%).

**`ddpg-sharpe` — ◐ partly holds.** Claimed 1.38; A: mean 1.16 [0.93, 1.39],
best seed 1.43, so the claim sits at the top of the interval and one seed in
five reaches it. Seed-averaged returns: Sharpe 1.22, ΔSharpe against 1/N +0.12,
p = 0.320: not significant. In B the best seed is 1.24: ✗. In C the mean is 0.70
and no seed beats 1/N (p = 0.985): ✗.

**`td3-sharpe` — ◐ partly holds.** Claimed 1.37; A: mean 1.10 [0.69, 1.52], best
seed 1.37 (seed 0, 1.3726); the spread across seeds is large (0.57 to 1.37).
Seed-averaged: Sharpe 1.20, ΔSharpe +0.09, p = 0.328. B: the same best seed,
1.37 (ΔSharpe −0.04, p = 0.654). C: ✗ (mean 0.70, best 1.01).

**`sac-sharpe` — ◐ partly holds.** Claimed 1.33; A: mean 1.23 [1.07, 1.39], 4 of 5
seeds above 1/N, best seed 1.42; seed-averaged Sharpe 1.33, ΔSharpe +0.22,
p = 0.218. B: mean 1.23, best 1.49; seed-averaged Sharpe 1.32, ΔSharpe +0.22,
p = 0.133, alpha 7.5% a year (HAC t = 1.6). The closest case to ✅ and still
short of p < 0.05. C: ✗ (mean 0.66, best 0.82; alpha −10.8% a year, HAC t = −2.2).

**`ppo-sharpe` — ◐ partly holds.** Claimed 1.09; A: mean 1.17 [0.90, 1.44]
(above the claim), 4 of 5 seeds above 1/N; seed-averaged ΔSharpe +0.15,
p = 0.339. B: mean 0.95 (ΔSharpe −0.11, p = 0.713). C: mean 0.98, no seed
above 1/N, so the claimed value is not reached by any seed in the benchmark's
range (best 1.02): ✗.

**`a2c-sharpe` — ◐ partly holds.** Claimed 0.82; A: mean 1.09 [0.91, 1.27]
excludes the claim from above, i.e. our A2C is better than the paper's;
seed-averaged ΔSharpe against 1/N +0.04, p = 0.398. The
paper's order, DDPG > TD3 > SAC > PPO > A2C, does not appear in any setup (A:
SAC > PPO > DDPG > TD3 > A2C; B: SAC > DDPG > TD3 > A2C > PPO; C: A2C > PPO >
DDPG ≈ TD3 > SAC); differences between neighbouring agents are smaller than the
seed standard deviations.

**`lstm-max-drawdown`, `lstm-sharpe` — ? inconclusive.** Setup C's
`lstm-forecast`: mean Sharpe 0.75 [0.39, 1.12] against the claimed 1.19, best
seed 1.08; mean maximum drawdown −26.5% (best seed −23.8%) against −9.57%
(1/N: −29.4%). No seed reaches either claimed value, but that submission is not
the paper's pipeline (Deviations), so the numbers speak about our
reimplementation, not about the paper's strategy. Deciding it needs a run of
`rl/forecast/lstm.py` (the paper's price-forecast pipeline: MAPE loss, long or
cash per stock) through this engine.

**Multiple comparisons.** 8 claims, 3 setups. No cell earned ✅, so a
correction could only lower a verdict that no one is relying on; the verdicts
above are the uncorrected ones.

### Observations that are not verdicts

- *Setup A against B.* The rule sells everything on five sessions, 2020-03-17,
  -20, -23, -24 and -25, in all 25 runs with the rule on; no run without the
  rule has such a session (table above, [E]). That one difference moves the
  measured exposure: beta against 1/N is 0.30–0.56 with the rule and 0.95–1.00
  without it, and the mean maximum drawdown is −17.2% to −19.0% (PPO −9.2%) with
  it and −27.0% to −29.0% without ([sum] `per_setup`). The paper's reported
  drawdowns (−24.19% to −33.52%) are closer in size to B's than to A's. The rule
  lifts the seed-mean Sharpe of DDPG, TD3, PPO and A2C by 0.10, 0.11, 0.22 and
  0.13 and leaves SAC's unchanged (1.233 vs 1.230). The paper does not mention a
  turbulence rule. [INFERENCE] Setup B may be the closer copy of the setup the
  paper describes; under B, `ddpg-sharpe` is ✗ rather than ◐.
- *Alpha.* Against 1/N no agent has a significant positive alpha in A or B (HAC
  t between −0.6 and 1.6; SAC without the rule: 7.5% a year, t = 1.6, beta 1.00).
- *Setup C.* In the benchmark environment every learned agent trails 1/N (0/5
  seeds above it for each agent, p between 0.825 and 0.998). Which difference
  between the share environment and the weights environment drives that (shares
  versus weights, features, per-trade limits, checkpoint selection) is open; this
  report does not test it.

## Deviations

1. **Post hoc pre-registration**; see the banner of
   [`preregistration.md`](preregistration.md). The primary-setup choice and the
   verdict rule were written with the numbers in view.
2. **No L1.** Our runs fix next-open fills and 0.1% fees; the PDF specifies
   neither fill timing nor transaction costs. Whether the paper's numbers
   reproduce under its own execution is untested.
3. **Baseline**: 1/N instead of the S&P 500. Alpha and beta against the index
   are not tested.
4. **Hyperparameters**: defaults (A, B: 80,000 timesteps; C: 100,000 or 200,000).
   The paper calls its non-rolling DDPG "individually tuned" (Section 5.2); we did
   not tune, so a DDPG gap could come from tuning rather than from the
   execution model. The paper's hyperparameters are not in the PDF.
5. **LSTM**: tested only through the `lstm-forecast` submission of setup C,
   which forecasts each stock's next open-to-open return from 30 sessions of
   log-return, range and close/open features and holds the stocks expected to
   rise with equal weights (`submissions/lstm-forecast/submission.json`); the
   paper forecasts prices with MAPE loss and holds or sells per stock.
6. **Intervals** are 95% Student-t intervals over 5 seeds; they are wide
   (TD3: 0.69 to 1.52) and assume roughly normal seed variation.
7. **Share-env series.** Setups A and B first had per-run summary metrics only
   ([S]); the daily account values ([E]) were added afterwards and now give the
   significance and alpha/beta figures. They were exported from the toolkit's
   `account_value.csv` files and reproduce every per-run Sharpe, annual return
   and maximum drawdown in [S] to within 5e-5 (checked by `analyze.py`). The 1/N
   curve in [E] (Sharpe 1.1049, annual return 29.66%) matches [S]. No verdict
   changed when [E] was added.
8. **Not tested**: rolling retraining, leveraged LSTM, cumulative return, the
   paper's alpha and beta (ours are against 1/N, not the S&P 500), and the paper's
   return and drawdown figures for TD3, SAC, PPO and A2C (shown for context in
   the tables only).

## Limitations

- Two years (505 sessions) and one regime (a bull market with the 2020 crash):
  the Sharpe standard error of 0.69 is larger than every effect discussed.
- A universe picked in 2021 from the largest companies: survivorship bias,
  which the paper discloses and we did not remove (the task is a re-test of the
  paper's setup, not a ranking).
- Five seeds and one configuration per agent; no hyperparameter search by
  either side that we can account for.
- The setups differ from the paper in execution (next-open, fees) and, for C,
  in environment design, so a ◐ or ✗ says "does not reproduce under this
  setup", not "cannot be produced by any setup".

## Corrections

- **2026-11-07, smallest detectable effect.** "Lo's formula: 0.90" was
  computed with the annual Sharpe ratio of 1/N (1.11) in the term where Lo
  (2002) uses the per-session one; the correct value is 0.71 (`analyze.py`
  and `evidence/summary.json` fixed). The same form is in this report's
  pre-registration and in `.plan/lab-protocol.md`, now corrected there. The
  bootstrap standard error (0.69), the 1.13 threshold derived from it and
  every verdict are unchanged. Found while drafting the R2 pre-registration.

## Reproduce

From a clean checkout with the package installed (`python -m pip install -e .`).
Data and commands for A/B are in [S] (`data.download`, `command`).

```bash
# Setup A and B: 5 agents x seeds 0-4, with and without the rule (50 runs)
python main.py --mode download_data \
  --tickers AAPL MSFT AMZN BRK-B JPM JNJ UNH HD PG NVDA DIS BAC CMCSA XOM VZ T ADBE PFE CSCO INTC \
  --start-date 2000-01-01 --end-date 2021-01-01 --output datasets/paper2021.csv
for agent in a2c ppo ddpg td3 sac; do for seed in 0 1 2 3 4; do
  python main.py --mode train --data-file datasets/paper2021.csv --start-date 2000-01-01 \
    --start-trade-date 2019-01-01 --end-date 2021-01-01 --agent $agent --seed $seed [--no-turbulence]
done; done

# Setup C: benchmark task, then the leaderboard
rl-pricing bench data paper-2021
rl-pricing bench run-all --task paper-2021
rl-pricing bench leaderboard

# Every number in this report, and claims.json
python reports/r0-wang-zhang-2026/evidence/analyze.py
rl-pricing bench reports
```

The evidence files [S] and [E] hold the per-run metrics and the daily account
values of those 50 runs (exported from each run's `account_value.csv`); no
script in the repository assembles them from the run outputs, so rerunning A and
B reproduces the runs but not the file layout. `analyze.py` checks that [E]
reproduces [S]. Downloads depend on Yahoo
Finance and can differ from the hashed inputs; compare the SHA-256 values above.
