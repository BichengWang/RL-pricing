# Fee sensitivity

How much of the historical track's verdict is the fee? Every historical record is re-priced at fees from 0 to 50 bp per side, and so is 1/N buy-and-hold. Written by `rl-pricing bench fees` from [`historical/`](historical/); every number is also in [`fees.json`](fees.json).

**Method.** Each record keeps, per seed and session, equity net of the charged fee *c* and turnover *τ* (traded notional over pre-trade equity). A fill leaves post-fee equity *B*(1 − *cτ*), so at fee *c′* a session's return becomes 1 + *r′* = (1 + *r*)(1 − *c′τ*)/(1 − *cτ*). Decisions are held fixed, which is what these policies would do at another fee: they decide from prices and their current weights, and after a trade the weights are the targets whatever the fee. The replay keeps *τ* at its charged-fee value, so it is exact at 0 bp up to forced delisting sales (a holding sold on a session the policy holds pays the fee on the sold part only, which leaves the later weights slightly fee-dependent; largest observed error 5e-6 Sharpe, 1/N on `crypto-majors` after EOSUSDT's delisting). The error grows with turnover and fee; `tests/test_bench_fees.py` holds it within 0.01 Sharpe of exact engine runs up to 50 bp.

This is not retraining: the learned policies were trained and selected on returns net of the charged fee, and a policy trained at another fee could trade differently. The LSTM forecaster's buy hurdle is also pinned at 10 bp by its submission (`rl/bench/policies/forecast.py`), so its decision rule is itself fee-shaped.

*Sharpe* is that of the seed-averaged daily returns (the basis of the leaderboard's ΔSharpe, so it can differ from the leaderboard's mean per-seed Sharpe). *ΔSharpe* is that minus 1/N's, with the leaderboard's one-sided paired stationary-bootstrap p-value (uncorrected for the many comparisons). *Break-even*: the fee at which the entry's Sharpe equals 1/N's, assuming a single crossing in 0–50 bp (no entry's ΔSharpe changes sign more than once on the fee grid); *never* means it trails 1/N even at 0 bp, *> 50 bp* that it leads at every fee shown.

## Entry-task pairs ahead of 1/N

Pairs with a higher Sharpe ratio than 1/N at each fee, out of all pairs in the group; in brackets, how many of them have p<0.05. `paper-2021` is a survivorship-biased re-evaluation, so it is counted apart.

**Ranked tasks**: `crypto-majors`, `us-multi-asset`, `us-sectors`

| Fee per side | SB3 RL and LSTM | Other learned | Rule-based |
| --- | --- | --- | --- |
| as charged (5 bp or 10 bp) | 0 of 27 (0) | 4 of 12 (2) | 4 of 15 (1) |
| 0 bp | 17 of 27 (3) | 8 of 12 (3) | 5 of 15 (1) |
| 1 bp | 11 of 27 (1) | 8 of 12 (2) | 5 of 15 (1) |
| 2 bp | 6 of 27 (0) | 6 of 12 (2) | 4 of 15 (1) |
| 5 bp | 4 of 27 (0) | 4 of 12 (2) | 4 of 15 (1) |
| 10 bp | 0 of 27 (0) | 3 of 12 (2) | 4 of 15 (1) |
| 25 bp | 0 of 27 (0) | 3 of 12 (2) | 4 of 15 (1) |
| 50 bp | 0 of 27 (0) | 2 of 12 (1) | 3 of 15 (1) |

**Re-evaluations (survivorship-biased, not ranked)**: `paper-2021`

| Fee per side | SB3 RL and LSTM | Rule-based |
| --- | --- | --- |
| as charged (10 bp) | 0 of 6 (0) | 1 of 5 (0) |
| 0 bp | 3 of 6 (0) | 1 of 5 (0) |
| 1 bp | 1 of 6 (0) | 1 of 5 (0) |
| 2 bp | 0 of 6 (0) | 1 of 5 (0) |
| 5 bp | 0 of 6 (0) | 1 of 5 (0) |
| 10 bp | 0 of 6 (0) | 1 of 5 (0) |
| 25 bp | 0 of 6 (0) | 1 of 5 (0) |
| 50 bp | 0 of 6 (0) | 1 of 5 (0) |

Groups: *SB3 RL and LSTM*: A2C, CVaR-PPO, DDPG, LSTM forecast, PPO, PPO (tuned), SAC, SAC (tuned), TD3; *Other learned*: Behaviour cloning, CQL, Decision Transformer, IQL; *Rule-based*: 1/N monthly rebalance, Inverse volatility, Minimum variance, Momentum (top 3), Trend following (200d SMA). Constant-return entries (cash) have no Sharpe ratio and are left out.

## By task

Fee columns give ΔSharpe vs 1/N (one-sided p) at that fee; the ◆ row gives 1/N's own Sharpe, so an entry's Sharpe is the ◆ value plus its ΔSharpe. Rows follow the Sharpe ratio at the charged fee.

### Crypto majors — `crypto-majors`

2023-01-01 to 2025-12-31 (1096 sessions) · fee charged 10 bp per side

| Strategy | Kind | Turnover/yr | 0 bp | 1 bp | 2 bp | 5 bp | 10 bp (charged) | 25 bp | 50 bp | Break-even |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1/N buy and hold** ◆ | baseline | 0.3× | 0.78 | 0.78 | 0.78 | 0.78 | 0.78 | 0.77 | 0.77 | – |
| Minimum variance | baseline | 2.6× | +0.87 (p<0.001) | +0.87 (p<0.001) | +0.87 (p<0.001) | +0.87 (p<0.001) | +0.86 (p<0.001) | +0.86 (p<0.001) | +0.84 (p<0.001) | > 50 bp |
| Decision Transformer | rl | 9.1× | +0.61 (p=0.001) | +0.61 (p=0.001) | +0.61 (p=0.001) | +0.60 (p=0.001) | +0.59 (p=0.002) | +0.57 (p=0.004) | +0.52 (p=0.007) | > 50 bp |
| CQL | rl | 11.2× | +0.46 (p=0.017) | +0.46 (p=0.017) | +0.46 (p=0.018) | +0.45 (p=0.020) | +0.44 (p=0.022) | +0.40 (p=0.030) | +0.35 (p=0.053) | > 50 bp |
| Inverse volatility | baseline | 2.5× | +0.11 (p=0.081) | +0.11 (p=0.083) | +0.11 (p=0.084) | +0.11 (p=0.085) | +0.10 (p=0.089) | +0.10 (p=0.102) | +0.09 (p=0.128) | > 50 bp |
| Behaviour cloning | rl | 28.7× | +0.15 (p=0.044) | +0.14 (p=0.051) | +0.14 (p=0.060) | +0.12 (p=0.084) | +0.09 (p=0.140) | +0.02 (p=0.441) | -0.12 (p=0.904) | 28.0 bp |
| Momentum (top 3) | baseline | 5.3× | +0.07 (p=0.388) | +0.07 (p=0.390) | +0.07 (p=0.391) | +0.06 (p=0.394) | +0.06 (p=0.401) | +0.05 (p=0.429) | +0.03 (p=0.467) | > 50 bp |
| PPO (tuned) | rl | 146.1× | +0.24 (p=0.020) | +0.22 (p=0.036) | +0.19 (p=0.056) | +0.10 (p=0.196) | -0.04 (p=0.621) | -0.47 (p=1.000) | -1.19 (p=1.000) | 8.6 bp |
| 1/N monthly rebalance | baseline | 1.5× | -0.06 (p=0.745) | -0.06 (p=0.745) | -0.06 (p=0.747) | -0.06 (p=0.750) | -0.06 (p=0.753) | -0.06 (p=0.767) | -0.07 (p=0.784) | never |
| IQL | rl | 16.1× | -0.04 (p=0.719) | -0.04 (p=0.736) | -0.05 (p=0.750) | -0.06 (p=0.783) | -0.07 (p=0.841) | -0.11 (p=0.951) | -0.18 (p=0.998) | never |
| A2C | rl | 15.9× | -0.04 (p=0.704) | -0.05 (p=0.720) | -0.05 (p=0.732) | -0.06 (p=0.767) | -0.07 (p=0.822) | -0.11 (p=0.942) | -0.19 (p=0.994) | never |
| TD3 | rl | 216.5× | +0.32 (p=0.041) | +0.28 (p=0.062) | +0.24 (p=0.092) | +0.12 (p=0.256) | -0.08 (p=0.657) | -0.67 (p=1.000) | -1.66 (p=1.000) | 8.1 bp |
| PPO | rl | 145.5× | +0.16 (p=0.085) | +0.14 (p=0.124) | +0.11 (p=0.172) | +0.03 (p=0.388) | -0.10 (p=0.778) | -0.50 (p=1.000) | -1.16 (p=1.000) | 6.2 bp |
| DDPG | rl | 214.7× | +0.20 (p=0.086) | +0.15 (p=0.144) | +0.11 (p=0.209) | -0.01 (p=0.521) | -0.22 (p=0.920) | -0.84 (p=1.000) | -1.89 (p=1.000) | 4.7 bp |
| CVaR-PPO | rl | 76.8× | -0.08 (p=0.836) | -0.09 (p=0.879) | -0.11 (p=0.910) | -0.15 (p=0.969) | -0.22 (p=0.998) | -0.43 (p=1.000) | -0.78 (p=1.000) | never |
| SAC | rl | 326.4× | +0.34 (p=0.027) | +0.28 (p=0.057) | +0.22 (p=0.110) | +0.04 (p=0.418) | -0.27 (p=0.899) | -1.20 (p=1.000) | -2.75 (p=1.000) | 5.6 bp |
| Trend following (200d SMA) | baseline | 3.3× | -0.67 (p=0.969) | -0.67 (p=0.970) | -0.67 (p=0.970) | -0.68 (p=0.970) | -0.68 (p=0.971) | -0.69 (p=0.973) | -0.71 (p=0.976) | never |
| SAC (tuned) | rl | 322.8× | -0.08 (p=0.650) | -0.14 (p=0.768) | -0.21 (p=0.858) | -0.40 (p=0.980) | -0.72 (p=1.000) | -1.69 (p=1.000) | -3.31 (p=1.000) | never |
| LSTM forecast | forecast | 246.7× | -0.63 (p=0.930) | -0.68 (p=0.947) | -0.73 (p=0.956) | -0.88 (p=0.982) | -1.13 (p=0.996) | -1.88 (p=1.000) | -3.13 (p=1.000) | never |
| Cash | baseline | 0.0× | – | – | – | – | – | – | – | – |

### US multi-asset allocation — `us-multi-asset`

2018-01-02 to 2025-12-31 (2011 sessions) · fee charged 5 bp per side

| Strategy | Kind | Turnover/yr | 0 bp | 1 bp | 2 bp | 5 bp (charged) | 10 bp | 25 bp | 50 bp | Break-even |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1/N buy and hold** ◆ | baseline | 0.1× | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.72 | – |
| Momentum (top 3) | baseline | 5.0× | +0.00 (p=0.483) | +0.00 (p=0.488) | -0.00 (p=0.495) | -0.01 (p=0.506) | -0.03 (p=0.526) | -0.08 (p=0.606) | -0.16 (p=0.725) | 1.4 bp |
| Behaviour cloning | rl | 15.3× | +0.05 (p=0.262) | +0.04 (p=0.322) | +0.02 (p=0.388) | -0.02 (p=0.600) | -0.10 (p=0.859) | -0.32 (p=1.000) | -0.70 (p=1.000) | 3.5 bp |
| IQL | rl | 12.3× | +0.02 (p=0.404) | +0.01 (p=0.468) | -0.00 (p=0.532) | -0.04 (p=0.714) | -0.10 (p=0.920) | -0.27 (p=1.000) | -0.56 (p=1.000) | 1.9 bp |
| 1/N monthly rebalance | baseline | 0.4× | -0.04 (p=0.821) | -0.04 (p=0.821) | -0.04 (p=0.824) | -0.04 (p=0.827) | -0.04 (p=0.833) | -0.05 (p=0.860) | -0.05 (p=0.892) | never |
| A2C | rl | 7.2× | -0.03 (p=0.791) | -0.04 (p=0.822) | -0.05 (p=0.858) | -0.06 (p=0.940) | -0.10 (p=0.988) | -0.19 (p=1.000) | -0.35 (p=1.000) | never |
| Decision Transformer | rl | 6.3× | -0.04 (p=0.698) | -0.05 (p=0.721) | -0.05 (p=0.748) | -0.07 (p=0.803) | -0.10 (p=0.887) | -0.18 (p=0.990) | -0.33 (p=1.000) | never |
| Inverse volatility | baseline | 1.1× | -0.07 (p=0.758) | -0.07 (p=0.762) | -0.07 (p=0.766) | -0.07 (p=0.777) | -0.08 (p=0.788) | -0.09 (p=0.836) | -0.12 (p=0.890) | never |
| LSTM forecast | forecast | 34.9× | +0.02 (p=0.454) | -0.01 (p=0.549) | -0.03 (p=0.633) | -0.10 (p=0.868) | -0.22 (p=0.990) | -0.58 (p=1.000) | -1.17 (p=1.000) | 0.7 bp |
| PPO | rl | 24.1× | -0.02 (p=0.683) | -0.04 (p=0.822) | -0.06 (p=0.920) | -0.12 (p=0.999) | -0.23 (p=1.000) | -0.54 (p=1.000) | -1.07 (p=1.000) | never |
| SAC (tuned) | rl | 22.7× | -0.04 (p=0.759) | -0.06 (p=0.859) | -0.08 (p=0.916) | -0.14 (p=0.995) | -0.24 (p=1.000) | -0.53 (p=1.000) | -1.03 (p=1.000) | never |
| CVaR-PPO | rl | 23.8× | -0.03 (p=0.794) | -0.06 (p=0.904) | -0.08 (p=0.964) | -0.14 (p=0.999) | -0.24 (p=1.000) | -0.56 (p=1.000) | -1.09 (p=1.000) | never |
| PPO (tuned) | rl | 23.6× | -0.05 (p=0.864) | -0.07 (p=0.941) | -0.09 (p=0.979) | -0.15 (p=1.000) | -0.25 (p=1.000) | -0.56 (p=1.000) | -1.08 (p=1.000) | never |
| Trend following (200d SMA) | baseline | 2.4× | -0.14 (p=0.750) | -0.14 (p=0.755) | -0.15 (p=0.759) | -0.16 (p=0.774) | -0.17 (p=0.790) | -0.22 (p=0.848) | -0.29 (p=0.910) | never |
| CQL | rl | 6.9× | -0.13 (p=0.709) | -0.14 (p=0.723) | -0.15 (p=0.735) | -0.18 (p=0.773) | -0.23 (p=0.832) | -0.37 (p=0.939) | -0.61 (p=0.992) | never |
| Minimum variance | baseline | 1.0× | -0.19 (p=0.762) | -0.19 (p=0.764) | -0.19 (p=0.766) | -0.19 (p=0.771) | -0.20 (p=0.779) | -0.22 (p=0.811) | -0.26 (p=0.844) | never |
| SAC | rl | 45.9× | +0.00 (p=0.508) | -0.04 (p=0.803) | -0.08 (p=0.948) | -0.20 (p=1.000) | -0.40 (p=1.000) | -1.00 (p=1.000) | -2.00 (p=1.000) | 0.1 bp |
| TD3 | rl | 148.9× | +0.17 (p=0.150) | +0.05 (p=0.409) | -0.07 (p=0.756) | -0.43 (p=1.000) | -1.03 (p=1.000) | -2.84 (p=1.000) | -5.82 (p=1.000) | 1.4 bp |
| DDPG | rl | 155.4× | +0.12 (p=0.189) | -0.01 (p=0.551) | -0.13 (p=0.870) | -0.50 (p=1.000) | -1.11 (p=1.000) | -2.95 (p=1.000) | -5.98 (p=1.000) | 1.0 bp |
| Cash | baseline | 0.0× | – | – | – | – | – | – | – | – |

### US sector rotation — `us-sectors`

2018-01-02 to 2025-12-31 (2011 sessions) · fee charged 5 bp per side

| Strategy | Kind | Turnover/yr | 0 bp | 1 bp | 2 bp | 5 bp (charged) | 10 bp | 25 bp | 50 bp | Break-even |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1/N buy and hold** ◆ | baseline | 0.1× | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | – |
| Decision Transformer | rl | 3.7× | +0.02 (p=0.398) | +0.02 (p=0.406) | +0.02 (p=0.412) | +0.01 (p=0.435) | -0.00 (p=0.476) | -0.03 (p=0.573) | -0.09 (p=0.743) | 9.3 bp |
| 1/N monthly rebalance | baseline | 0.4× | +0.01 (p=0.440) | +0.01 (p=0.441) | +0.01 (p=0.443) | +0.01 (p=0.449) | +0.01 (p=0.458) | +0.00 (p=0.483) | -0.00 (p=0.524) | 40.4 bp |
| Behaviour cloning | rl | 15.4× | +0.04 (p=0.326) | +0.03 (p=0.373) | +0.02 (p=0.415) | -0.01 (p=0.545) | -0.05 (p=0.742) | -0.20 (p=0.989) | -0.43 (p=1.000) | 4.3 bp |
| A2C | rl | 10.0× | +0.02 (p=0.343) | +0.01 (p=0.407) | +0.00 (p=0.470) | -0.01 (p=0.650) | -0.04 (p=0.910) | -0.13 (p=1.000) | -0.28 (p=1.000) | 2.8 bp |
| Inverse volatility | baseline | 0.8× | -0.01 (p=0.645) | -0.01 (p=0.649) | -0.01 (p=0.653) | -0.02 (p=0.664) | -0.02 (p=0.682) | -0.02 (p=0.734) | -0.03 (p=0.802) | never |
| Momentum (top 3) | baseline | 5.7× | -0.00 (p=0.485) | -0.01 (p=0.493) | -0.01 (p=0.500) | -0.02 (p=0.522) | -0.03 (p=0.559) | -0.07 (p=0.667) | -0.15 (p=0.808) | never |
| IQL | rl | 12.8× | +0.01 (p=0.391) | +0.00 (p=0.461) | -0.00 (p=0.527) | -0.03 (p=0.701) | -0.07 (p=0.900) | -0.18 (p=1.000) | -0.38 (p=1.000) | 1.6 bp |
| SAC (tuned) | rl | 15.0× | +0.01 (p=0.430) | -0.00 (p=0.511) | -0.01 (p=0.604) | -0.04 (p=0.823) | -0.08 (p=0.983) | -0.22 (p=1.000) | -0.44 (p=1.000) | 1.0 bp |
| CVaR-PPO | rl | 23.1× | +0.02 (p=0.308) | +0.01 (p=0.441) | -0.01 (p=0.588) | -0.05 (p=0.904) | -0.12 (p=0.998) | -0.32 (p=1.000) | -0.67 (p=1.000) | 1.5 bp |
| PPO (tuned) | rl | 28.1× | +0.02 (p=0.339) | +0.00 (p=0.500) | -0.02 (p=0.656) | -0.07 (p=0.958) | -0.15 (p=0.999) | -0.40 (p=1.000) | -0.82 (p=1.000) | 1.1 bp |
| CQL | rl | 7.0× | -0.05 (p=0.728) | -0.05 (p=0.740) | -0.06 (p=0.754) | -0.07 (p=0.788) | -0.09 (p=0.849) | -0.16 (p=0.948) | -0.27 (p=0.996) | never |
| LSTM forecast | forecast | 41.2× | +0.04 (p=0.349) | +0.02 (p=0.434) | -0.00 (p=0.542) | -0.07 (p=0.827) | -0.18 (p=0.995) | -0.53 (p=1.000) | -1.09 (p=1.000) | 1.9 bp |
| PPO | rl | 28.8× | +0.01 (p=0.404) | -0.01 (p=0.598) | -0.02 (p=0.751) | -0.08 (p=0.979) | -0.16 (p=1.000) | -0.42 (p=1.000) | -0.85 (p=1.000) | 0.6 bp |
| Trend following (200d SMA) | baseline | 2.5× | -0.09 (p=0.672) | -0.09 (p=0.678) | -0.09 (p=0.682) | -0.10 (p=0.695) | -0.11 (p=0.718) | -0.14 (p=0.771) | -0.20 (p=0.841) | never |
| Minimum variance | baseline | 1.6× | -0.13 (p=0.814) | -0.13 (p=0.814) | -0.13 (p=0.817) | -0.13 (p=0.820) | -0.14 (p=0.828) | -0.15 (p=0.850) | -0.17 (p=0.889) | never |
| SAC | rl | 74.3× | +0.07 (p=0.108) | +0.03 (p=0.308) | -0.01 (p=0.610) | -0.15 (p=0.987) | -0.37 (p=1.000) | -1.03 (p=1.000) | -2.14 (p=1.000) | 1.7 bp |
| DDPG | rl | 134.6× | +0.01 (p=0.480) | -0.06 (p=0.807) | -0.14 (p=0.962) | -0.36 (p=1.000) | -0.73 (p=1.000) | -1.83 (p=1.000) | -3.67 (p=1.000) | 0.1 bp |
| TD3 | rl | 120.9× | -0.02 (p=0.654) | -0.09 (p=0.911) | -0.16 (p=0.987) | -0.37 (p=1.000) | -0.71 (p=1.000) | -1.73 (p=1.000) | -3.43 (p=1.000) | never |
| Cash | baseline | 0.0× | – | – | – | – | – | – | – | – |

### Re-evaluation: Wang & Zhang (2021) setup — `paper-2021`

2019-01-02 to 2020-12-31 (505 sessions) · fee charged 10 bp per side

*Survivorship-biased universe: a re-evaluation of a published setup, not a ranking.*

| Strategy | Kind | Turnover/yr | 0 bp | 1 bp | 2 bp | 5 bp | 10 bp (charged) | 25 bp | 50 bp | Break-even |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1/N buy and hold** ◆ | baseline | 0.5× | 1.11 | 1.11 | 1.11 | 1.11 | 1.11 | 1.11 | 1.10 | – |
| Momentum (top 3) | baseline | 6.7× | +0.21 (p=0.221) | +0.21 (p=0.222) | +0.21 (p=0.224) | +0.21 (p=0.231) | +0.20 (p=0.240) | +0.17 (p=0.270) | +0.13 (p=0.319) | > 50 bp |
| 1/N monthly rebalance | baseline | 1.0× | -0.07 (p=0.772) | -0.07 (p=0.772) | -0.07 (p=0.773) | -0.07 (p=0.775) | -0.07 (p=0.777) | -0.07 (p=0.789) | -0.08 (p=0.807) | never |
| A2C | rl | 9.2× | -0.05 (p=0.723) | -0.05 (p=0.731) | -0.06 (p=0.745) | -0.07 (p=0.782) | -0.08 (p=0.825) | -0.14 (p=0.920) | -0.23 (p=0.974) | never |
| Inverse volatility | baseline | 1.7× | -0.10 (p=0.832) | -0.10 (p=0.832) | -0.10 (p=0.833) | -0.10 (p=0.837) | -0.10 (p=0.843) | -0.11 (p=0.862) | -0.12 (p=0.882) | never |
| PPO | rl | 22.5× | -0.04 (p=0.671) | -0.05 (p=0.701) | -0.06 (p=0.732) | -0.08 (p=0.814) | -0.13 (p=0.898) | -0.26 (p=0.976) | -0.48 (p=0.997) | never |
| LSTM forecast | forecast | 79.1× | +0.06 (p=0.365) | +0.03 (p=0.420) | -0.01 (p=0.469) | -0.11 (p=0.643) | -0.27 (p=0.831) | -0.78 (p=0.984) | -1.62 (p=1.000) | 1.9 bp |
| Minimum variance | baseline | 2.7× | -0.29 (p=0.846) | -0.30 (p=0.847) | -0.30 (p=0.848) | -0.30 (p=0.849) | -0.31 (p=0.850) | -0.32 (p=0.863) | -0.35 (p=0.875) | never |
| Trend following (200d SMA) | baseline | 2.6× | -0.36 (p=0.956) | -0.36 (p=0.956) | -0.36 (p=0.956) | -0.37 (p=0.958) | -0.38 (p=0.960) | -0.40 (p=0.966) | -0.43 (p=0.977) | never |
| DDPG | rl | 98.7× | -0.01 (p=0.551) | -0.05 (p=0.675) | -0.09 (p=0.777) | -0.20 (p=0.927) | -0.40 (p=0.985) | -0.97 (p=1.000) | -1.93 (p=1.000) | never |
| TD3 | rl | 102.4× | +0.00 (p=0.513) | -0.04 (p=0.621) | -0.08 (p=0.715) | -0.20 (p=0.893) | -0.40 (p=0.977) | -1.01 (p=1.000) | -2.03 (p=1.000) | 0.1 bp |
| SAC | rl | 122.8× | +0.04 (p=0.353) | -0.00 (p=0.541) | -0.05 (p=0.742) | -0.20 (p=0.967) | -0.43 (p=0.998) | -1.15 (p=1.000) | -2.34 (p=1.000) | 0.9 bp |
| Cash | baseline | 0.0× | – | – | – | – | – | – | – | – |
