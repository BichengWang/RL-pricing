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
| `main.py` | Command-line entry point (`--mode train` / `--mode download_data`) |
| `rl/` | Trading library adapted from [FinRL](https://github.com/AI4Finance-Foundation/FinRL): market data download, feature engineering, gym environments, DRL agents (A2C, PPO, DDPG, SAC, TD3) and backtesting |
| `rl/config/config.py` | Dates, ticker lists, technical indicators and agent hyperparameters |
| `dl/` | Deep-learning models and preprocessing |
| `rl_portfolio_trading.ipynb` | Running results (the notebook to start with) |
| `rl_portfolio_trading_phase2.ipynb` | Phase 2 notebook |
| `notebooks/`, `*.ipynb` | Other experiments |
| `img/` | Backtest and performance figures |
| `docs/` | Paper copy |

## Setup

Needs network access (market data is downloaded from Yahoo Finance). `requirements.txt` pins 2021-era versions (for example `stable-baselines3==0.11.0a4`), so use a Python version from that period, such as 3.8.

```bash
git clone https://github.com/BichengWang/RL-pricing.git
cd RL-pricing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Download Dow 30 daily prices (START_DATE to END_DATE in rl/config/config.py)
# and save them to datasets/<timestamp>.csv
python main.py --mode download_data

# Download data, add technical indicators, train a SAC agent on the pre-2019
# split, trade on the remainder, and write account values, actions and
# backtest stats to results/
python main.py --mode train
```

Trained models, TensorBoard logs, datasets and results are written to `trained_models/`, `tensorboard_log/`, `datasets/` and `results/` in the working directory (created on first run).

To change the agent, date range, tickers or hyperparameters, edit `rl/config/config.py` and `rl/autotrain/training.py`. To explore the results interactively, open `rl_portfolio_trading.ipynb` in Jupyter.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
