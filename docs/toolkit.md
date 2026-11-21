# Training toolkit (`--mode` CLI)

The benchmark in the [README](../README.md) evaluates any policy under one
execution model. This page documents the separate training pipeline that
predates it: FinRL-style share trading, portfolio allocation, the LSTM
forecaster and the rolling ensemble, driven by `python main.py --mode ...`
(or the installed `rl-pricing` command).

## Quick start (offline)

The synthetic data source generates correlated random-walk prices, so this
training-and-backtest run needs no network access. The small training budget
is for learning the workflow, not tuning a strategy:

```bash
python main.py --mode train --data-source synthetic --tickers AAA BBB CCC DDD \
    --start-date 2015-01-01 --start-trade-date 2019-01-01 --end-date 2020-01-01 \
    --agent ppo --timesteps 2000 --seed 0 --run-name offline-ppo
```

Training uses dates before 2019; the saved policy then trades the held-out
2019 period using close signals and next-open fills. Inspect
`results/offline-ppo/perf_stats.csv`, `results/offline-ppo/executions.csv` and
`results/offline-ppo/backtest.png`, and keep `trained_models/offline-ppo/`
for a later backtest:

```bash
python main.py --mode backtest --model-dir trained_models/offline-ppo
```

Replay uses the saved data source, dates and frozen tickers, and writes to a
new `results/offline-ppo_backtest_<timestamp>/` directory. Use a different
`--run-name` for each training experiment to avoid overwriting its artifacts.
A seed controls randomness, but does not guarantee identical results across
dependency versions or hardware.

## Real market data

```bash
# Train SAC on the Dow 30 (2000-2018) and trade 2019-2020, compared with the
# Dow Jones index as well as the built-in baselines
python main.py --mode train --agent sac --timesteps 80000 --benchmark ^DJI

# Save prices once, then train from the file (repeatable, no re-download)
python main.py --mode download_data --tickers AAPL MSFT JPM --output datasets/prices.csv
python main.py --mode train --data-file datasets/prices.csv --agent a2c
```

Yahoo downloads (including `--benchmark`) require network access and available
symbols. For your own CSV, use one row per ticker and date with columns
`date,tic,open,high,low,close,volume`; dates must be `YYYY-MM-DD` or
`YYYYMMDD`. Supplying `--data-file` selects the CSV source automatically, and without `--tickers`
the CSV pipeline considers all tickers in the file for fitting-history
eligibility. Ensure the file covers both training and trading dates; data
outside `[--start-date, --end-date)` are ignored. Open and close prices must
be finite, positive and on the same adjustment basis.

The default Dow 30 list is the January 2019 membership under current tickers,
so backtests on it that start earlier are survivorship-biased. The benchmark
tasks avoid this; see the README.

## Choosing the checkpoint on validation data

RL training is noisy, and the model at the end of training is often not the
best one. With `--validation-days N` the last N days of the training period
are held out: every `--eval-freq` steps the agent trades them, and the
checkpoint with the best Sharpe ratio there is kept and used for trading:

```bash
python main.py --mode train --agent ppo --timesteps 200000 \
    --validation-days 252 --eval-freq 10000
```

The test period is never used for this choice. Each evaluation is listed in
`results/<run>/validation.csv`, and `run_config.json` records which checkpoint
was chosen.

Ticker eligibility is decided from fitting history only, before validation.
The sorted universe is then frozen for validation, trading and saved-model
replay. Missing or invalid open/close prices in these held-out panels raise
a ticker/date error instead of silently changing the universe or dropping
sessions.

## Backtesting a saved model

Single-agent training saves its model, settings and, when enabled,
observation-normalisation statistics, so it can be re-run on the same or a
different period:

```bash
python main.py --mode backtest --model-dir trained_models/<run-name> \
    --start-trade-date 2020-01-01 --end-date 2021-01-01
```

Artifacts must record `execution_timing: "next_open"`; older same-close models
must be retrained. Omitting `--transaction-cost` preserves the checkpoint's
saved buy/sell fees; an explicit value (including `0`) overrides both. Fees
must be finite fractions in `[0, 1)`. Backtests write their effective settings
to `results/<run>/run_config.json`; the saved checkpoint settings remain
unchanged. Explicitly supplied options override saved settings even when they
match the CLI defaults; for example, `--leverage 1.0` uses unleveraged targets
for an LSTM checkpoint.

## Portfolio allocation

`--task portfolio` swaps share trading for portfolio allocation: every day the
agent picks long-only weights at the close, rebalances at the next session's
open, and values its positions at that session's close. Existing holdings keep
their overnight returns, and transaction costs are paid on actual turnover.
Its observation is the return covariance matrix of the last `--cov-lookback`
days stacked on technical indicators.

```bash
python main.py --mode train --task portfolio --agent ppo --timesteps 50000
```

`actions.csv` holds the target weights chosen at each signal close;
`executions.csv` pairs each signal with its next-session fill. `--hmax` and
the turbulence options do not apply to this task, and the ensemble supports
only `--task trading`.

## LSTM forecasting strategy

A two-layer LSTM forecasts every ticker's next close. At each signal close,
the rule targets `1/N` of the account, times `--leverage`, for each stock whose
forecast is at or above that close, and keeps the rest in cash. Targets fill
at the next open:

```bash
python main.py --mode lstm --lstm-epochs 30
python main.py --mode backtest --model-dir trained_models/<run-name> --leverage 2
```

One model is trained on all tickers with a MAPE loss (`--lstm-loss mse` is
the alternative). Inputs are the last `--lstm-window` days of scale-free log
returns and indicators. The most recent 10% of training sessions are held out
to pick the epoch; scaling and fitting labels exclude them. The run writes
`predictions.csv`, `forecast_metrics.csv`, `training_history.csv`,
`actions.csv`, `executions.csv` and `model.pt`.

## Ensemble strategy

Every `--rebalance-window` days the ensemble trains A2C, PPO and DDPG, picks
the one with the best Sharpe ratio on the preceding `--validation-window`
days, retrains it and trades the next window, carrying positions over:

```bash
python main.py --mode ensemble --timesteps 20000 --rebalance-window 63 --validation-window 63
```

`ensemble_summary.csv` records validation dates, signal/valuation endpoints and
transition counts.

Ensemble-specific settings:

- `--ensemble-timesteps a2c=30000 ppo=100000 ddpg=10000` sets the training
  steps per algorithm; algorithms not listed use `--timesteps`.
  `TrainConfig.ensemble_model_kwargs` (Python API) replaces the default model
  settings per algorithm, e.g. `{"ddpg": {"action_noise": {"type":
  "ornstein_uhlenbeck", "sigma": 0.5}}}`; an `action_noise` given by name alone
  uses sigma 0.1.
- `--no-turbulence` turns the liquidation rule off for the ensemble as for the
  other modes (no turbulence index is computed and none is needed). The
  ensemble sets its own threshold every window, so it does not accept
  `--turbulence-threshold` or `--turbulence-quantile`.
- `--standalone-agents` also trades A2C, PPO and DDPG each with its own
  account over the same windows: positions carry over, and every window the
  algorithm is refit on all data before it, exactly as the winner is (the
  winner's model is shared). The accounts are in `agent_accounts.csv`
  (`date,ENSEMBLE,A2C,PPO,DDPG`) and in `perf_stats.csv`.

`rl/autotrain/r1.py` (`python -m rl.autotrain.r1 --data-file prices.csv --seed 0`)
runs the ensemble with the FinRL authors' recipe for the R1 report: indicators
`macd, rsi_30, cci_30, dx_30`, A2C 30,000 / PPO 100,000 / DDPG 10,000 steps, PPO
with `ent_coef` 0.005 and 128-step rollouts in minibatches of 16, DDPG with
Ornstein-Uhlenbeck noise of sigma 0.5 (`--ddpg-noise-sigma`), standalone accounts
on, prices from a CSV.

### Sharadar prices

`rl/marketdata/sharadar.py` converts a Sharadar SEP extract (columns `ticker,
date, open, high, low, close, volume, closeadj`) into the CSV above on one
adjustment basis: `close` becomes `closeadj` and `open`, `high` and `low` are
scaled by `closeadj / close`. A JSON ticker map renames or stitches series,
e.g. `{"UTX": "RTX", "DD": [{"ticker": "DD", "end": "2017-08-31"}, {"ticker":
"DWDP", "start": "2017-09-01"}]}`:

```bash
python -m rl.marketdata.sharadar SEP.csv datasets/sharadar.csv --tickers AAPL UTX DD \
    --ticker-map map.json --start-date 2009-01-01 --end-date 2026-01-01
```

The output is derived from licensed data: keep it out of version control.

## Options

Run `python main.py --help` for the full list. The most useful ones:

| Option | Default | Meaning |
| --- | --- | --- |
| `--task` | `trading` | `trading` (buy and sell shares) or `portfolio` (daily portfolio weights) |
| `--agent` | `sac` | `a2c`, `ppo`, `ddpg`, `td3` or `sac` |
| `--timesteps` | 80000 | training steps (per model for the ensemble) |
| `--ensemble-timesteps` | none | ensemble: `AGENT=STEPS` pairs overriding `--timesteps` per algorithm |
| `--standalone-agents` | off | ensemble: also trade A2C, PPO and DDPG each with its own account |
| `--data-source` / `--data-file` | `yahoo` | `yahoo`, `csv` or `synthetic` |
| `--tickers` | Dow 30 | tickers to trade |
| `--start-date`, `--start-trade-date`, `--end-date` | 2000-01-01, 2019-01-01, 2021-01-01 | training period is `[start, start-trade)`, trading period `[start-trade, end)` |
| `--seed` | none | controls random seeds |
| `--transaction-cost` | training: 0.001; backtest: saved setting | fraction paid on every buy and sell |
| `--hmax` | 100 | maximum shares traded per stock per day |
| `--reward-type` | `asset_change` | or `log_return` |
| `--cov-lookback` | 252 | portfolio task: days of returns in each covariance matrix |
| `--turbulence-quantile` / `--turbulence-threshold` / `--no-turbulence` | 0.99 quantile | when to liquidate during market turmoil |
| `--no-normalize` | off | feed raw observations to the agent |
| `--validation-days` / `--eval-freq` | 0 / 10000 | hold out the last N training days and keep the best checkpoint on them |
| `--lstm-window`, `--lstm-epochs`, `--lstm-hidden-size`, `--lstm-loss` | 20, 30, 32, `mape` | LSTM strategy settings |
| `--leverage` | 1 | LSTM strategy: fraction of the account invested when every stock is forecast to rise |
| `--benchmark` | none | Yahoo ticker to add to the comparison, e.g. `^DJI` or `SPY` |

## Outputs

| Path | Contents |
| --- | --- |
| `trained_models/<run>/model.zip` | the trained stable-baselines3 model (the best validation checkpoint with `--validation-days`) |
| `trained_models/<run>/vecnormalize.pkl` | observation-normalisation statistics, when normalisation is enabled |
| `trained_models/<run>/run_config.json` | every setting, frozen tickers, eligibility cutoff, execution timing and turbulence threshold |
| `results/<run>/account_value.csv` | daily account value over the trading period |
| `results/<run>/actions.csv` | signal-date-indexed signed executed shares; target weights for portfolio/LSTM |
| `results/<run>/executions.csv` | signal date, next-session fill date and signed shares or target weights |
| `results/<run>/perf_stats.csv` | agent, baselines and benchmark side by side |
| `results/<run>/backtest.png` | cumulative return and drawdown chart |
| `results/<run>/validation.csv` | with `--validation-days`: validation Sharpe and return at each evaluation |

The agent is always compared with an equal-weight buy-and-hold portfolio and
an equal-weight portfolio rebalanced monthly, both starting in cash and buying
at the next open with transaction costs. The ensemble mode also compares with a
long-only minimum-variance portfolio (weights from the sample covariance of the
trailing 252 returns, re-estimated every 63 sessions, filled at the next open),
computed from the sessions before the trading period; it is left out, with a
warning, when fewer than 252 sessions precede it
(`rl.trade.baselines.min_variance_portfolio`).

## Python API

```python
from rl.autotrain.training import TrainConfig, run_training

result = run_training(TrainConfig(agent="ppo", total_timesteps=50_000, data_source="synthetic",
                                  ticker_list=["AAA", "BBB", "CCC"], seed=0))
print(result["stats"])
```

## The trading environment

`rl.env.env_stocktrading.StockTradingEnv` holds cash and whole shares. Each day
the agent observes the close and indicators, then outputs a value in [-1, 1]
per stock, scaled by `hmax` into shares to sell (negative) or buy (positive).
Orders fill at the next session's open. Sells execute first, then buys from
the largest order down, limited by cash including transaction costs.
Signal-day turbulence at or above the threshold schedules liquidation at the
next open and blocks buying. The observation layout is
`[cash, prices, holdings, indicators]`, standardised by running statistics.

`rl.env.env_portfolio.StockPortfolioEnv` (used by `--task portfolio`) holds a
fractional, long-only portfolio. A softmax turns one score per stock into
weights; fees and post-fee targets are solved together so the account remains
self-financing.

## Changes from the 2021 version

- **Causal evaluation.** Eligibility is fitted before validation and frozen
  thereafter; LSTM scaling and fitting labels exclude validation outcomes.
- **Next-open execution.** Agents, the forecasting strategy, baselines and the
  optional benchmark use close signals and next-open fills on one adjustment basis.
- **Complete rolling accounting.** The ensemble includes its partial final
  window and preserves every return and fill at window boundaries.
- **Modern dependencies.** gymnasium and stable-baselines3 2.x; pyfolio is
  replaced by `rl/trade/metrics.py`.
- **Accounting fixes.** Buys could spend more cash than available, forced
  turbulence sales recorded no cost, failed buys were counted as trades,
  indicators were back-filled from the future, and evaluation sampled random
  actions instead of deterministic ones.
- **Removed.** The 2021 notebooks, TensorFlow `dl/` code and the cash-penalty
  and stop-loss environments; they remain in Git history.
