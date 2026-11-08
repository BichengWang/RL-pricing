# Leaderboard

Can deep RL beat 1/N? Every row is traded by the same causal execution model (signal at close *t*, fill at open *t+1*, proportional fees, long-only, no leverage). Negative results stay in.

## Historical track

Fixed public split: fit before the test window, then trade it.

### Crypto majors — `crypto-majors`

2023-01-01 to 2025-12-31 (1096 sessions) · fee 0.1% per side · BTCUSDT, ETHUSDT, BNBUSDT, EOSUSDT, BCCUSDT, NEOUSDT, ADAUSDT, LTCUSDT, XRPUSDT, ONTUSDT

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Minimum variance](../submissions/minimum-variance/) | baseline | 1.64 | 597.2% | 90.9% | -33.3% | 2.6× | 0.70 | 43.1% (t=3.8) | +0.86 (p<0.001) | 1.00 | 1 |
| 2 | [Decision Transformer](../submissions/decision-transformer/) | rl | 1.25 ± 0.20 | 410.4% | 68.9% | -33.9% | 9.1× | 0.73 | 30.9% (t=2.1) | +0.59 (p=0.002) | 0.89 | 5 |
| 3 | [CQL](../submissions/cql/) | rl | 1.21 ± 0.09 | 297.8% | 58.1% | -34.1% | 11.2× | 0.73 | 23.3% (t=2.2) | +0.44 (p=0.022) | 0.82 | 5 |
| 4 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 0.88 | 178.7% | 40.7% | -46.8% | 2.5× | 0.97 | 6.7% (t=1.4) | +0.10 (p=0.089) | 0.94 | 1 |
| 5 | [Behaviour cloning](../submissions/bc/) | rl | 0.85 ± 0.20 | 173.0% | 38.2% | -45.2% | 28.7× | 0.91 | 5.7% (t=0.5) | +0.09 (p=0.140) | 0.62 | 5 |
| 6 | [Momentum (top 3)](../submissions/momentum/) | baseline | 0.84 | 162.0% | 37.8% | -47.9% | 5.3× | 0.89 | 7.8% (t=0.6) | +0.06 (p=0.401) | 0.93 | 1 |
| 7 | **1/N buy and hold** ◆ | baseline | 0.78 | 134.0% | 32.7% | -44.5% | 0.3× | 1.00 | 0.0% | – | 0.91 | – |
| 8 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 0.72 | 111.7% | 28.4% | -51.7% | 1.5× | 1.03 | -3.0% (t=-0.6) | -0.06 (p=0.753) | 0.89 | 1 |
| 9 | [PPO (tuned)](../submissions/ppo-tuned/) | rl | 0.71 ± 0.22 | 110.3% | 26.9% | -49.3% | 146.1× | 0.83 | -0.5% (t=-0.2) | -0.04 (p=0.621) | 0.43 | 5 |
| 10 | [IQL](../submissions/iql/) | rl | 0.71 ± 0.05 | 105.3% | 27.0% | -46.5% | 16.1× | 0.96 | -3.4% (t=-0.7) | -0.07 (p=0.841) | 0.51 | 5 |
| 11 | [A2C](../submissions/a2c/) | rl | 0.70 ± 0.03 | 102.4% | 26.4% | -45.6% | 15.9× | 0.91 | -3.3% (t=-0.7) | -0.07 (p=0.822) | 0.89 | 5 |
| 12 | [TD3](../submissions/td3/) | rl | 0.66 ± 0.25 | 105.7% | 25.1% | -54.4% | 216.5× | 0.89 | -2.1% (t=-0.2) | -0.08 (p=0.657) | 0.89 | 5 |
| 13 | [PPO](../submissions/ppo/) | rl | 0.66 ± 0.06 | 89.2% | 23.6% | -54.1% | 145.5× | 0.91 | -4.3% (t=-0.6) | -0.10 (p=0.778) | 0.88 | 5 |
| 14 | [CVaR-PPO](../submissions/cvar-ppo/) | rl | 0.55 ± 0.13 | 60.7% | 16.7% | -48.8% | 76.8× | 0.91 | -11.3% (t=-2.0) | -0.22 (p=0.998) | 0.41 | 5 |
| 15 | [DDPG](../submissions/ddpg/) | rl | 0.53 ± 0.17 | 53.6% | 14.6% | -53.8% | 214.7× | 0.85 | -9.9% (t=-0.8) | -0.22 (p=0.920) | 0.83 | 5 |
| 16 | [SAC](../submissions/sac/) | rl | 0.44 ± 0.20 | 37.7% | 10.0% | -59.1% | 326.4× | 0.84 | -11.6% (t=-0.7) | -0.27 (p=0.899) | 0.81 | 5 |
| 17 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.10 | -13.5% | -4.7% | -50.6% | 3.3× | 0.61 | -23.6% (t=-1.9) | -0.68 (p=0.971) | 0.57 | 1 |
| 18 | [SAC (tuned)](../submissions/sac-tuned/) | rl | 0.07 ± 0.49 | -11.5% | -9.6% | -67.6% | 322.8× | 0.80 | -33.5% (t=-1.8) | -0.72 (p=1.000) | 0.09 | 5 |
| 19 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | -0.32 ± 0.25 | -58.2% | -26.6% | -73.3% | 246.7× | 0.58 | -44.1% (t=-1.9) | -1.13 (p=0.996) | 0.27 | 5 |
| 20 | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

### US multi-asset allocation — `us-multi-asset`

2018-01-02 to 2025-12-31 (2011 sessions) · fee 0.05% per side · SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, VNQ

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **1/N buy and hold** ◆ | baseline | 0.73 | 96.8% | 8.9% | -26.6% | 0.1× | 1.00 | 0.0% | – | 0.98 | – |
| 2 | [Momentum (top 3)](../submissions/momentum/) | baseline | 0.72 | 112.1% | 9.9% | -23.7% | 5.0× | 0.87 | 2.4% (t=0.7) | -0.01 (p=0.506) | 0.98 | 1 |
| 3 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 0.69 | 84.7% | 8.0% | -25.8% | 0.4× | 0.95 | -0.4% (t=-0.8) | -0.04 (p=0.827) | 0.97 | 1 |
| 4 | [IQL](../submissions/iql/) | rl | 0.68 ± 0.07 | 69.8% | 6.8% | -23.3% | 12.3× | 0.79 | -0.2% (t=-0.2) | -0.04 (p=0.714) | 0.77 | 5 |
| 5 | [A2C](../submissions/a2c/) | rl | 0.66 ± 0.02 | 72.8% | 7.1% | -23.8% | 7.2× | 0.87 | -0.6% (t=-1.2) | -0.06 (p=0.940) | 0.97 | 5 |
| 6 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 0.66 | 59.4% | 6.0% | -24.2% | 1.1× | 0.72 | -0.4% (t=-0.4) | -0.07 (p=0.777) | 0.97 | 1 |
| 7 | [Behaviour cloning](../submissions/bc/) | rl | 0.61 ± 0.18 | 71.3% | 6.7% | -24.9% | 15.3× | 0.77 | 0.0% (t=0.0) | -0.02 (p=0.600) | 0.79 | 5 |
| 8 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | 0.60 ± 0.08 | 90.4% | 8.4% | -30.7% | 34.9× | 1.09 | -1.0% (t=-0.6) | -0.10 (p=0.868) | 0.96 | 5 |
| 9 | [PPO](../submissions/ppo/) | rl | 0.60 ± 0.01 | 64.9% | 6.5% | -24.2% | 24.1× | 0.89 | -1.3% (t=-2.3) | -0.12 (p=0.999) | 0.95 | 5 |
| 10 | [CVaR-PPO](../submissions/cvar-ppo/) | rl | 0.59 ± 0.02 | 61.3% | 6.2% | -24.0% | 23.8× | 0.87 | -1.5% (t=-2.7) | -0.14 (p=0.999) | 0.68 | 5 |
| 11 | [SAC (tuned)](../submissions/sac-tuned/) | rl | 0.58 ± 0.06 | 62.5% | 6.3% | -25.2% | 22.7× | 0.88 | -1.4% (t=-1.9) | -0.14 (p=0.995) | 0.58 | 5 |
| 12 | [PPO (tuned)](../submissions/ppo-tuned/) | rl | 0.58 ± 0.01 | 60.5% | 6.1% | -24.4% | 23.6× | 0.88 | -1.6% (t=-2.9) | -0.15 (p=1.000) | 0.57 | 5 |
| 13 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.57 | 39.1% | 4.2% | -15.6% | 2.4× | 0.49 | -0.1% (t=-0.1) | -0.16 (p=0.774) | 0.94 | 1 |
| 14 | [Minimum variance](../submissions/minimum-variance/) | baseline | 0.54 | 29.9% | 3.3% | -21.4% | 1.0× | 0.34 | 0.3% (t=0.2) | -0.19 (p=0.771) | 0.93 | 1 |
| 15 | [Decision Transformer](../submissions/decision-transformer/) | rl | 0.52 ± 0.32 | 75.8% | 6.3% | -29.9% | 6.3× | 0.81 | -0.4% (t=-0.2) | -0.07 (p=0.803) | 0.75 | 5 |
| 16 | [SAC](../submissions/sac/) | rl | 0.52 ± 0.15 | 54.2% | 5.5% | -27.2% | 45.9× | 0.88 | -2.1% (t=-2.1) | -0.20 (p=1.000) | 0.93 | 5 |
| 17 | [CQL](../submissions/cql/) | rl | 0.48 ± 0.19 | 35.3% | 3.7% | -22.1% | 6.9× | 0.40 | 0.2% (t=0.1) | -0.18 (p=0.773) | 0.64 | 5 |
| 18 | [TD3](../submissions/td3/) | rl | 0.27 ± 0.06 | 24.5% | 2.8% | -30.2% | 148.9× | 0.89 | -4.6% (t=-2.0) | -0.43 (p=1.000) | 0.80 | 5 |
| 19 | [DDPG](../submissions/ddpg/) | rl | 0.21 ± 0.19 | 19.3% | 2.0% | -32.1% | 155.4× | 0.92 | -5.6% (t=-2.8) | -0.50 (p=1.000) | 0.74 | 5 |
| 20 | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

### US sector rotation — `us-sectors`

2018-01-02 to 2025-12-31 (2011 sessions) · fee 0.05% per side · XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 0.69 | 140.9% | 11.6% | -36.7% | 0.4× | 1.00 | 0.2% (t=0.3) | +0.01 (p=0.449) | 0.97 | 1 |
| 2 | **1/N buy and hold** ◆ | baseline | 0.68 | 138.5% | 11.5% | -35.5% | 0.1× | 1.00 | 0.0% | – | 0.97 | – |
| 3 | [A2C](../submissions/a2c/) | rl | 0.67 ± 0.01 | 119.0% | 10.3% | -34.3% | 10.0× | 0.91 | -0.2% (t=-0.3) | -0.01 (p=0.650) | 0.97 | 5 |
| 4 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 0.67 | 126.9% | 10.8% | -35.7% | 0.8× | 0.96 | -0.2% (t=-0.3) | -0.02 (p=0.664) | 0.97 | 1 |
| 5 | [Momentum (top 3)](../submissions/momentum/) | baseline | 0.67 | 142.2% | 11.7% | -31.6% | 5.7× | 0.97 | 0.8% (t=0.3) | -0.02 (p=0.522) | 0.97 | 1 |
| 6 | [IQL](../submissions/iql/) | rl | 0.65 ± 0.02 | 110.5% | 9.8% | -32.0% | 12.8× | 0.87 | -0.3% (t=-0.3) | -0.03 (p=0.701) | 0.74 | 5 |
| 7 | [SAC (tuned)](../submissions/sac-tuned/) | rl | 0.65 ± 0.03 | 110.8% | 9.8% | -33.6% | 15.0× | 0.89 | -0.5% (t=-0.7) | -0.04 (p=0.823) | 0.64 | 5 |
| 8 | [Behaviour cloning](../submissions/bc/) | rl | 0.64 ± 0.18 | 117.2% | 9.9% | -32.8% | 15.4× | 0.85 | 0.2% (t=0.2) | -0.01 (p=0.545) | 0.76 | 5 |
| 9 | [CVaR-PPO](../submissions/cvar-ppo/) | rl | 0.64 ± 0.03 | 108.7% | 9.6% | -33.9% | 23.1× | 0.90 | -0.7% (t=-1.1) | -0.05 (p=0.904) | 0.72 | 5 |
| 10 | [PPO (tuned)](../submissions/ppo-tuned/) | rl | 0.62 ± 0.03 | 104.1% | 9.3% | -34.2% | 28.1× | 0.90 | -1.0% (t=-1.4) | -0.07 (p=0.958) | 0.61 | 5 |
| 11 | [PPO](../submissions/ppo/) | rl | 0.61 ± 0.06 | 101.4% | 9.1% | -33.7% | 28.8× | 0.90 | -1.2% (t=-1.5) | -0.08 (p=0.979) | 0.95 | 5 |
| 12 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | 0.60 ± 0.04 | 111.4% | 9.8% | -38.9% | 41.2× | 0.95 | -0.9% (t=-0.5) | -0.07 (p=0.827) | 0.96 | 5 |
| 13 | [CQL](../submissions/cql/) | rl | 0.60 ± 0.04 | 94.8% | 8.7% | -31.9% | 7.0× | 0.82 | -0.7% (t=-0.4) | -0.07 (p=0.788) | 0.70 | 5 |
| 14 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.59 | 59.0% | 6.0% | -17.4% | 2.5× | 0.51 | -0.0% (t=-0.0) | -0.10 (p=0.695) | 0.95 | 1 |
| 15 | [Decision Transformer](../submissions/decision-transformer/) | rl | 0.55 ± 0.19 | 131.0% | 9.9% | -36.2% | 3.7× | 0.87 | 0.8% (t=0.2) | +0.01 (p=0.435) | 0.78 | 5 |
| 16 | [Minimum variance](../submissions/minimum-variance/) | baseline | 0.55 | 80.2% | 7.7% | -32.9% | 1.6× | 0.76 | -1.0% (t=-0.4) | -0.13 (p=0.820) | 0.94 | 1 |
| 17 | [SAC](../submissions/sac/) | rl | 0.52 ± 0.12 | 83.5% | 7.8% | -33.2% | 74.3× | 0.90 | -2.3% (t=-1.3) | -0.15 (p=0.987) | 0.93 | 5 |
| 18 | [DDPG](../submissions/ddpg/) | rl | 0.31 ± 0.14 | 41.7% | 4.2% | -39.5% | 134.6× | 0.97 | -6.2% (t=-2.7) | -0.36 (p=1.000) | 0.82 | 5 |
| 19 | [TD3](../submissions/td3/) | rl | 0.31 ± 0.16 | 39.7% | 4.0% | -40.4% | 120.9× | 0.94 | -6.2% (t=-2.7) | -0.37 (p=1.000) | 0.81 | 5 |
| 20 | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

### Re-evaluation: Wang & Zhang (2021) setup — `paper-2021`

2019-01-02 to 2020-12-31 (505 sessions) · fee 0.1% per side · AAPL, MSFT, AMZN, BRK-B, JPM, JNJ, UNH, HD, PG, NVDA, DIS, BAC, CMCSA, XOM, VZ, T, ADBE, PFE, CSCO, INTC

*Survivorship-biased universe (chosen in hindsight): this re-tests a published setup with this benchmark's execution model and is not a ranking.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| – | [Momentum (top 3)](../submissions/momentum/) | baseline | 1.31 | 121.1% | 48.6% | -30.4% | 6.7× | 1.19 | 10.5% (t=1.0) | +0.20 (p=0.240) | 0.96 | 1 |
| – | **1/N buy and hold** ◆ | baseline | 1.11 | 68.9% | 29.9% | -29.4% | 0.5× | 1.00 | 0.0% | – | 0.94 | – |
| – | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 1.04 | 60.6% | 26.7% | -29.5% | 1.0× | 0.96 | -1.5% (t=-0.6) | -0.07 (p=0.777) | 0.93 | 1 |
| – | [A2C](../submissions/a2c/) | rl | 1.03 ± 0.01 | 56.7% | 25.1% | -28.2% | 9.2× | 0.92 | -1.8% (t=-0.8) | -0.08 (p=0.825) | 0.92 | 5 |
| – | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 1.01 | 55.1% | 24.5% | -28.9% | 1.7× | 0.91 | -2.1% (t=-0.8) | -0.10 (p=0.843) | 0.92 | 1 |
| – | [PPO](../submissions/ppo/) | rl | 0.98 ± 0.04 | 53.6% | 23.9% | -28.1% | 22.5× | 0.92 | -2.9% (t=-1.2) | -0.13 (p=0.898) | 0.91 | 5 |
| – | [Minimum variance](../submissions/minimum-variance/) | baseline | 0.81 | 33.7% | 15.6% | -25.8% | 2.7× | 0.70 | -4.2% (t=-0.7) | -0.31 (p=0.850) | 0.87 | 1 |
| – | [LSTM forecast](../submissions/lstm-forecast/) | forecast | 0.75 ± 0.29 | 39.7% | 17.9% | -26.5% | 79.1× | 0.81 | -4.4% (t=-0.5) | -0.27 (p=0.831) | 0.88 | 5 |
| – | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.74 | 23.9% | 11.3% | -20.1% | 2.6× | 0.58 | -5.2% (t=-1.5) | -0.38 (p=0.960) | 0.85 | 1 |
| – | [DDPG](../submissions/ddpg/) | rl | 0.70 ± 0.17 | 35.3% | 16.2% | -29.3% | 98.7× | 0.95 | -9.9% (t=-2.2) | -0.40 (p=0.985) | 0.84 | 5 |
| – | [TD3](../submissions/td3/) | rl | 0.70 ± 0.21 | 34.2% | 15.7% | -29.3% | 102.4× | 0.91 | -9.4% (t=-1.7) | -0.40 (p=0.977) | 0.84 | 5 |
| – | [SAC](../submissions/sac/) | rl | 0.66 ± 0.26 | 33.0% | 15.1% | -30.2% | 122.8× | 0.95 | -10.8% (t=-2.2) | -0.43 (p=0.998) | 0.83 | 5 |
| – | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

## Live track

Policies frozen, then published at the date shown and scored on the sessions after it; they cannot have been tuned on the results.

### Crypto majors (published 2026-10-14) — `crypto-majors`

2026-10-15 to 2026-11-01 (18 sessions) · fee 0.1% per side · BTCUSDT, ETHUSDT, BNBUSDT, EOSUSDT, BCCUSDT, NEOUSDT, ADAUSDT, LTCUSDT, XRPUSDT, ONTUSDT

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Cash](../submissions/cash/) | baseline | – | 0.0% | – | 0.0% | – | 0.00 | – | – | – | 1 |
| 2 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | – | -2.8% | – | -5.4% | – | 0.09 | – | – | – | 3 |
| 3 | [Minimum variance](../submissions/minimum-variance/) | baseline | – | -3.6% | – | -7.2% | – | 0.88 | – | – | – | 1 |
| 4 | [Momentum (top 3)](../submissions/momentum/) | baseline | – | -4.3% | – | -7.6% | – | 0.77 | – | – | – | 1 |
| 5 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | – | -5.2% | – | -7.3% | – | 0.71 | – | – | – | 1 |
| 6 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | – | -7.6% | – | -9.9% | – | 0.96 | – | – | – | 1 |
| 7 | [PPO](../submissions/ppo/) | rl | – | -7.8% | – | -9.3% | – | 0.84 | – | – | – | 3 |
| 8 | [DDPG](../submissions/ddpg/) | rl | – | -7.8% | – | -9.2% | – | 0.69 | – | – | – | 3 |
| 9 | [A2C](../submissions/a2c/) | rl | – | -8.0% | – | -9.7% | – | 0.88 | – | – | – | 3 |
| 10 | [SAC](../submissions/sac/) | rl | – | -8.1% | – | -10.0% | – | 0.90 | – | – | – | 3 |
| 11 | [TD3](../submissions/td3/) | rl | – | -8.5% | – | -10.2% | – | 0.89 | – | – | – | 3 |
| 12 | **1/N buy and hold** ◆ | baseline | – | -9.0% | – | -10.9% | – | 1.00 | – | – | – | – |
| 13 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | – | -9.0% | – | -10.9% | – | 1.00 | – | – | – | 1 |

### Crypto majors (published 2026-10-16) — `crypto-majors`

2026-10-17 to 2026-11-01 (16 sessions) · fee 0.1% per side · BTCUSDT, ETHUSDT, BNBUSDT, EOSUSDT, BCCUSDT, NEOUSDT, ADAUSDT, LTCUSDT, XRPUSDT, ONTUSDT

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [PPO (tuned)](../submissions/ppo-tuned/) | rl | – | -6.7% | – | -8.8% | – | 0.78 | – | – | – | 3 |
| 2 | [SAC (tuned)](../submissions/sac-tuned/) | rl | – | -8.2% | – | -10.4% | – | 0.87 | – | – | – | 3 |
| 3 | **1/N buy and hold** ◆ | benchmark | – | -8.2% | – | -10.9% | – | – | – | – | – | – |

### US multi-asset allocation (published 2026-10-14) — `us-multi-asset`

2026-10-15 to 2026-10-30 (12 sessions) · fee 0.05% per side · SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, VNQ

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Minimum variance](../submissions/minimum-variance/) | baseline | – | 1.1% | – | -0.6% | – | -0.28 | – | – | – | 1 |
| 2 | [TD3](../submissions/td3/) | rl | – | 0.4% | – | -0.9% | – | 0.73 | – | – | – | 3 |
| 3 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | – | 0.2% | – | -0.7% | – | 0.58 | – | – | – | 1 |
| 4 | [Cash](../submissions/cash/) | baseline | – | 0.0% | – | 0.0% | – | 0.00 | – | – | – | 1 |
| 5 | [DDPG](../submissions/ddpg/) | rl | – | -0.2% | – | -1.5% | – | 0.84 | – | – | – | 3 |
| 6 | **1/N buy and hold** ◆ | baseline | – | -0.6% | – | -1.3% | – | 1.00 | – | – | – | – |
| 7 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | – | -0.6% | – | -1.3% | – | 1.00 | – | – | – | 1 |
| 8 | [SAC](../submissions/sac/) | rl | – | -0.7% | – | -1.3% | – | 0.97 | – | – | – | 3 |
| 9 | [PPO](../submissions/ppo/) | rl | – | -0.8% | – | -1.2% | – | 0.92 | – | – | – | 3 |
| 10 | [A2C](../submissions/a2c/) | rl | – | -0.8% | – | -1.3% | – | 0.98 | – | – | – | 3 |
| 11 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | – | -0.8% | – | -1.2% | – | 0.71 | – | – | – | 1 |
| 12 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | – | -1.0% | – | -2.0% | – | 1.43 | – | – | – | 3 |
| 13 | [Momentum (top 3)](../submissions/momentum/) | baseline | – | -1.5% | – | -3.1% | – | 2.16 | – | – | – | 1 |

### US multi-asset allocation (published 2026-10-16) — `us-multi-asset`

2026-10-19 to 2026-10-30 (10 sessions) · fee 0.05% per side · SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, VNQ

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **1/N buy and hold** ◆ | benchmark | – | -0.7% | – | -1.3% | – | – | – | – | – | – |
| 2 | [SAC (tuned)](../submissions/sac-tuned/) | rl | – | -0.7% | – | -1.2% | – | 0.92 | – | – | – | 3 |
| 3 | [PPO (tuned)](../submissions/ppo-tuned/) | rl | – | -0.7% | – | -1.2% | – | 0.93 | – | – | – | 3 |

### US sector rotation (published 2026-10-14) — `us-sectors`

2026-10-15 to 2026-10-30 (12 sessions) · fee 0.05% per side · XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [DDPG](../submissions/ddpg/) | rl | – | 2.9% | – | -0.9% | – | 0.80 | – | – | – | 3 |
| 2 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | – | 2.5% | – | -0.8% | – | 1.12 | – | – | – | 1 |
| 3 | **1/N buy and hold** ◆ | baseline | – | 2.2% | – | -0.6% | – | 1.00 | – | – | – | – |
| 4 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | – | 2.2% | – | -0.6% | – | 1.00 | – | – | – | 1 |
| 5 | [SAC](../submissions/sac/) | rl | – | 2.1% | – | -0.8% | – | 0.72 | – | – | – | 3 |
| 6 | [A2C](../submissions/a2c/) | rl | – | 2.1% | – | -0.6% | – | 0.90 | – | – | – | 3 |
| 7 | [PPO](../submissions/ppo/) | rl | – | 2.0% | – | -0.6% | – | 0.88 | – | – | – | 3 |
| 8 | [Minimum variance](../submissions/minimum-variance/) | baseline | – | 1.9% | – | -0.5% | – | 0.96 | – | – | – | 1 |
| 9 | [TD3](../submissions/td3/) | rl | – | 1.6% | – | -0.8% | – | 0.61 | – | – | – | 3 |
| 10 | [Momentum (top 3)](../submissions/momentum/) | baseline | – | 1.4% | – | -0.9% | – | 0.40 | – | – | – | 1 |
| 11 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | – | 1.0% | – | -0.2% | – | 0.48 | – | – | – | 1 |
| 12 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | – | 0.9% | – | -1.1% | – | 0.66 | – | – | – | 3 |
| 13 | [Cash](../submissions/cash/) | baseline | – | 0.0% | – | 0.0% | – | 0.00 | – | – | – | 1 |

### US sector rotation (published 2026-10-16) — `us-sectors`

2026-10-19 to 2026-10-30 (10 sessions) · fee 0.05% per side · XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **1/N buy and hold** ◆ | benchmark | – | 1.9% | – | -0.6% | – | – | – | – | – | – |
| 2 | [SAC (tuned)](../submissions/sac-tuned/) | rl | – | 1.7% | – | -0.5% | – | 0.87 | – | – | – | 3 |
| 3 | [PPO (tuned)](../submissions/ppo-tuned/) | rl | – | 1.7% | – | -0.5% | – | 0.84 | – | – | – | 3 |

## Ledger track: decisions committed before each open

Each decision is committed to the repository by a scheduled workflow before the session it trades opens; entries whose workflow run cannot be verified, and sessions without an entry, count as holds.

### AAPL (TradingAgents paper window) (decisions from 2026-10-23) — `us-aapl`

2026-10-26 to 2026-10-30 (5 sessions) · fee 0.05% per side · AAPL

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Ledger canary (1/N)](../submissions/ledger-canary/) | baseline | – | 0.2% | – | -1.2% | – | 1.00 | – | – | – | 1 |
| 2 | **1/N buy and hold** ◆ | benchmark | – | 0.2% | – | -1.2% | – | – | – | – | – | – |

### AAPL (TradingAgents paper window) (decisions from 2026-10-28) — `us-aapl`

2026-10-29 to 2026-10-30 (2 sessions) · fee 0.05% per side · AAPL

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [TradingAgents](../submissions/tradingagents/) | llm | – | -1.0% | – | -1.2% | – | 1.00 | – | – | – | 1 |
| 2 | **1/N buy and hold** ◆ | benchmark | – | -1.0% | – | -1.2% | – | – | – | – | – | – |

### AMZN (TradingAgents paper window) (decisions from 2026-10-28) — `us-amzn`

2026-10-29 to 2026-10-30 (2 sessions) · fee 0.05% per side · AMZN

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [TradingAgents](../submissions/tradingagents/) | llm | – | -2.7% | – | -2.7% | – | 1.00 | – | – | – | 1 |
| 2 | **1/N buy and hold** ◆ | benchmark | – | -2.7% | – | -2.7% | – | – | – | – | – | – |

### GOOGL (TradingAgents paper window) (decisions from 2026-10-28) — `us-googl`

2026-10-29 to 2026-10-30 (2 sessions) · fee 0.05% per side · GOOGL

*Fewer than 20 sessions: ranked by return; annualised and inferential statistics are not shown yet.*

| # | Strategy | Kind | Sharpe | Return | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [TradingAgents](../submissions/tradingagents/) | llm | – | -2.5% | – | -2.5% | – | 1.00 | – | – | – | 1 |
| 2 | **1/N buy and hold** ◆ | benchmark | – | -2.5% | – | -2.5% | – | – | – | – | – | – |

◆ marks the 1/N buy-and-hold benchmark (equal weights at the start of the period, never rebalanced); every ΔSharpe is measured against it over the same sessions.

**Metrics.** Computed from daily simple returns with a zero risk-free rate. *Sharpe*: mean/std × √(periods per year); with several seeds, mean ± standard deviation of per-seed Sharpe ratios. *Return* (total over the window, net of fees), *CAGR*, *Max DD* (peak-to-trough, including the starting value) and *Beta*/*Alpha* (OLS of daily returns on the benchmark's; alpha annualised, *t* is Newey-West) are means across seeds. *Turnover/yr*: annualised one-way traded fraction of equity. *ΔSharpe*: annualised Sharpe of the seed-averaged returns minus the benchmark's, with a one-sided paired stationary-bootstrap p-value (H1: better than 1/N). *DSR*: deflated Sharpe ratio of the seed-averaged returns, i.e. the probability the true Sharpe is positive after accounting for the submission's `trials` and for the dispersion of Sharpe ratios across this table (Bailey & López de Prado, 2014).
