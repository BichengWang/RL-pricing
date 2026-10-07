# Reinforcement Learning on Pricing Market

## Full document:

We beat 90% portfolio managers.   
https://drive.google.com/file/d/1-ogSAAA5L0sdy1iLt8HQKwV2kcN1UuCc/view?usp=sharing

Paper Copy: [Applying Deep Learning to Stock Trading](docs/Wang_Zhang_2021_Deep_Learning_Stock_Trading_arxiv.pdf)

## YouTube detail explanation:
https://www.youtube.com/watch?v=bE8MFq4sB2k

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Command-line entry point (`--mode train`, `backtest`, `ensemble`, `download_data`) |
| `rl/` | Trading library adapted from [FinRL](https://github.com/AI4Finance-Foundation/FinRL) |
| `rl/marketdata/` | Yahoo Finance downloader and a synthetic price generator for offline runs |
| `rl/preprocessing/` | Panel cleaning, technical indicators, turbulence index, covariance features |
| `rl/env/` | Trading environments (gymnasium API) |
| `rl/model/models.py` | DRL agents (A2C, PPO, DDPG, SAC, TD3) and the rolling ensemble |
| `rl/trade/` | Performance metrics, rule-based baselines and backtest plots |
| `rl/autotrain/training.py` | End-to-end train / backtest / ensemble pipelines (`TrainConfig`) |
| `rl/config/config.py` | Default dates, ticker lists, technical indicators and agent hyperparameters |
| `tests/` | Offline test suite (synthetic data, no network needed) |
| `dl/` | Deep-learning models and preprocessing (TensorFlow) |
| `rl_portfolio_trading.ipynb` | Running results (the notebook to start with) |
| `rl_portfolio_trading_phase2.ipynb` | Phase 2 notebook |
| `notebooks/`, `*.ipynb` | Other experiments |
| `img/` | Backtest and performance figures |
| `docs/` | Paper copy |

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/BichengWang/RL-pricing.git
cd RL-pricing
python -m venv .venv && source .venv/bin/activate
# Optional: CPU-only PyTorch is a much smaller download
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

`requirements.in` lists the supported version ranges (installed by `setup.py`);
`requirements.txt` pins the versions the test suite was last run with.

## Usage

### Quick start (offline)

The synthetic data source generates correlated random-walk prices, so the whole
pipeline runs without network access:

```bash
python main.py --mode train --data-source synthetic --tickers AAA BBB CCC DDD \
    --start-date 2015-01-01 --start-trade-date 2019-01-01 --end-date 2020-01-01 \
    --agent ppo --timesteps 20000 --seed 0
```

### Real market data

```bash
# Train SAC on the Dow 30 (2000-2018) and trade 2019-2020, compared with the
# Dow Jones index as well as the built-in baselines
python main.py --mode train --agent sac --timesteps 80000 --benchmark ^DJI

# Save prices once, then train from the file (repeatable, no re-download)
python main.py --mode download_data --tickers AAPL MSFT JPM --output datasets/prices.csv
python main.py --mode train --data-file datasets/prices.csv --agent a2c
```

### Backtesting a saved model

Each training run saves its model, observation-normalisation statistics and
settings, so it can be re-run on the same or a different period:

```bash
python main.py --mode backtest --model-dir trained_models/<run-name> \
    --start-trade-date 2020-01-01 --end-date 2021-01-01
```

### Ensemble strategy

The ensemble walks forward through the trading period. Every
`--rebalance-window` days it trains A2C, PPO and DDPG, picks the one with the
best Sharpe ratio on the preceding `--validation-window` days, retrains it and
trades the next window, carrying positions over:

```bash
python main.py --mode ensemble --timesteps 20000 --rebalance-window 63 --validation-window 63
```

### Options

Run `python main.py --help` for the full list. The most useful ones:

| Option | Default | Meaning |
| --- | --- | --- |
| `--agent` | `sac` | `a2c`, `ppo`, `ddpg`, `td3` or `sac` |
| `--timesteps` | 80000 | training steps (per model for the ensemble) |
| `--data-source` / `--data-file` | `yahoo` | `yahoo`, `csv` or `synthetic` |
| `--tickers` | Dow 30 | tickers to trade |
| `--start-date`, `--start-trade-date`, `--end-date` | 2000-01-01, 2019-01-01, 2021-01-01 | training period is `[start, start-trade)`, trading period `[start-trade, end)` |
| `--seed` | none | makes runs reproducible |
| `--transaction-cost` | 0.001 | fraction paid on every buy and sell |
| `--hmax` | 100 | maximum shares traded per stock per day |
| `--reward-type` | `asset_change` | or `log_return` |
| `--turbulence-quantile` / `--turbulence-threshold` / `--no-turbulence` | 0.99 quantile | when to liquidate during market turmoil |
| `--no-normalize` | off | feed raw observations to the agent |
| `--benchmark` | none | Yahoo ticker to add to the comparison, e.g. `^DJI` or `SPY` |

### Outputs

A run named `<run>` (default: timestamp and agent) writes:

| Path | Contents |
| --- | --- |
| `trained_models/<run>/model.zip` | the trained stable-baselines3 model |
| `trained_models/<run>/vecnormalize.pkl` | observation-normalisation statistics |
| `trained_models/<run>/run_config.json` | every setting, the tickers and the turbulence threshold used |
| `results/<run>/account_value.csv` | daily account value over the trading period |
| `results/<run>/actions.csv` | shares bought (+) or sold (-) per ticker per day |
| `results/<run>/perf_stats.csv` | agent, baselines and benchmark side by side |
| `results/<run>/backtest.png` | cumulative return and drawdown chart |

The statistics are those of pyfolio's `perf_stats` (annual return, Sharpe,
Sortino, Calmar, max drawdown, ...) plus the final account value. The agent is
always compared with an equal-weight buy-and-hold portfolio and an equal-weight
portfolio rebalanced monthly, both paying the same transaction costs.

### Python API

```python
from rl.autotrain.training import TrainConfig, run_training, run_backtest

result = run_training(TrainConfig(agent="ppo", total_timesteps=50_000, data_source="synthetic",
                                  ticker_list=["AAA", "BBB", "CCC"], seed=0))
print(result["stats"])
```

The building blocks (`YahooDownloader`, `FeatureEngineer`, `data_split`,
`StockTradingEnv`, `DRLAgent`, `backtest_stats`, `backtest_plot`) keep their
names and arguments, so the notebooks' workflow still applies.

## The trading environment

`rl.env.env_stocktrading.StockTradingEnv` holds cash and whole shares. Each day
the agent outputs a value in [-1, 1] per stock, scaled by `hmax` into shares to
sell (negative) or buy (positive) at the close. Sells execute first, then buys
from the largest order down, limited by the cash available **including**
transaction costs. When the turbulence index of the day is at or above the
threshold, all positions are sold (paying costs) and buying is blocked. The
observation is `[cash, prices, holdings, indicators]`; by default the pipeline
standardises it with running statistics, because these differ by several
orders of magnitude.

## Changes from the 2021 version

- **Modern dependencies.** Runs on current Python, pandas, NumPy, gymnasium and
  stable-baselines3 2.x. Environments use the gymnasium API: `reset()` returns
  `(obs, info)` and `step()` returns `(obs, reward, terminated, truncated, info)`.
  pyfolio is no longer needed; `rl/trade/metrics.py` computes the same
  statistics.
- **Fixed train mode**, which crashed calling `DRL_prediction` with the wrong
  arguments.
- **Environment accounting fixes.** Buys could spend more cash than available
  (costs were ignored when sizing orders); forced turbulence sales recorded no
  transaction cost; failed buys were counted as trades.
- **No look-ahead or cross-ticker leakage in features.** Missing values used to
  be back-filled across the whole frame, pulling values from later rows and
  other tickers. Each ticker is now forward-filled from its own past only, and
  tickers that are missing for much of the period (such as stocks listed
  part-way through) are dropped instead of crashing the environment.
- **Deterministic evaluation.** Trading uses the policy's deterministic action
  (it previously sampled random actions).
- **Hyperparameter dicts are not modified** when building a model (reusing a
  dict with `"action_noise": "normal"` for a second model used to fail).
- **Ensemble.** The turbulence lookback is 63 trading days rather than 63 data
  rows (about two days with 30 tickers), validation Sharpe ratios are
  annualised with sqrt(252), and the stitched account values are returned for
  benchmarking.
- **Portfolio environment** charges its transaction costs and rewards the
  change in portfolio value instead of its level.
- **New:** command-line options, backtest and ensemble modes, CSV and synthetic
  data sources, rule-based baselines, saved run configurations, and an offline
  test suite with CI.

Trained models from the old version cannot be loaded (they were saved with
stable-baselines3 0.11). The notebooks were run with the 2021 stack; their
saved outputs are kept as a record.

## Tests

The tests use synthetic data and offline fixtures, so they need no network:

```bash
pip install -e .
python -m pytest
```

They cover the environments' accounting, feature engineering (including a
check that no future data leaks into earlier rows), the metrics and baselines,
the Yahoo downloader's column handling, and short end-to-end runs of the
train, backtest, CLI and ensemble pipelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
