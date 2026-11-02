# R5 pre-registration: FinRL-DeepSeek (Benhenda, arXiv 2502.07393)

Written 2026-11-02, before any run of ours. The git commit time of this file is
the evidence that the plan came first; no file under
`reports/r5-finrl-deepseek/evidence/` and no result of ours may be committed
before it, and nothing of the authors' is run until it is on `main`. The report
follows `.plan/lab-protocol.md` (levels, pre-registration, verdict table,
contamination label, author notice). It covers **L1** (the authors' code, data
and agents), **L2** (the same question on this repository's engine, with
next-open fills, five seeds, an equal-weight baseline and the Nasdaq-100) and
no L3 (§5.3).

What was looked at before writing, so nobody has to guess: the paper (v1,
rendered pages and extracted text), the repository at the pinned commit
(scripts, environments, backtest notebook with its stored outputs, training
logs, issues), the two library forks its installer pulls, the Hugging Face
repositories of the data and agents (file lists, commit histories, hashes), range
reads of the news files and one counting pass over one of them (rows, stamp
format, score counts), the two agent-ready CSVs that feed the backtest (ticker list, dates, column names,
VIX dates), the tensor shapes of the 15 released agents (`torch.load` with
`weights_only=True`, no forward pass), the model cards of the LLMs that
produced the scores, a Yahoo availability check of the tickers, and a
stand-in for §8 made of market data only (an equal-weight portfolio against
`^NDX` and QQQ). **No agent, ours or the authors', has been run, and no
performance metric of any agent exists.** The thresholds in §7 were written
without seeing one. Items marked **[UNVERIFIED]** could not be checked from
here, and each says why; **[INFERENCE]** marks a conclusion drawn from what was
read.

## 1. Question

The paper adds two numbers derived by an LLM from news to a PPO agent and to a
CVaR-constrained PPO agent (CPPO) trading the Nasdaq-100. Its text claims that,
in two stochastic runs, PPO and CPPO-DeepSeek beat the Nasdaq-100 over
2019-2023; that PPO does better in the bull market to the end of 2021 and
CPPO-DeepSeek in the bear market after it; that stronger LLM infusion mostly
degrades PPO and improves CPPO; and that, at 400k training steps, the LLM
always worsens performance and DeepSeek V3 is slightly better than Llama 3.3.
Every claim rests on single runs of agents that sample their actions, and the
paper reports no seed, no interval and no baseline other than the index. We ask
(L1) whether the authors' own code, data and released agents give the pattern
when the run is repeated, and (L2) whether the pattern survives next-open fills,
a validation window, five seeds, an equal-weight baseline and a placebo for the
LLM features. We also state, before any run, that the LLM scores were produced
by models released after the test period and not cleared by any cutoff (§9).

## 2. Facts about the paper, the code and the data (read 2026-11-02)

### 2.1 Paper

| Item | Finding | Source |
| --- | --- | --- |
| Version | arXiv:2502.07393v1, q-fin.TR, submitted 11 Feb 2025, 5 pages. No later version: the abstract page lists `[v1]` only and the arXiv API shows published = updated = 2025-02-11. Peer-review status **[UNVERIFIED]**: no venue is named in the paper | <https://arxiv.org/abs/2502.07393>, <https://arxiv.org/pdf/2502.07393v1>, `export.arxiv.org/api/query?id_list=2502.07393` |
| Author, contact | Mostapha Benhenda, single author, no affiliation printed; mostaphabenhenda@gmail.com | p. 1 |
| Method | Frozen LLMs score news: a 1-5 recommendation score per stock and day (prompt from FNSPID) and a 1-5 risk score (new prompt). PPO-DeepSeek multiplies the action by $S_f$ near 1 (1.1, 1.05, 0.95, 0.9 by score and sign); CPPO-DeepSeek multiplies the trajectory return by $R_f = \sum_i w_i R_f^i$ with $R_f^i$ from 0.9 (risk 1) to 1.1 (risk 5), and also keeps $S_f$ | §3, §4.2, §4.3, p. 1-2 |
| LLMs | "DeepSeek V3 ..., Qwen 2.5 72B, Llama 3.3 70B"; "we randomly select a representative news article per stock per day, reducing the dataset to 2 million records" | §3, p. 1 |
| Data | FNSPID, "15.7 million time-aligned financial news records spanning 1999-2023"; Nasdaq-100 "index benchmark"; the paper does not list the stocks or say how many | abstract, §3 |
| Windows | Fig. 1: text "Training Period: 2019-2022", caption "3 years training history, 1 year trading", 500k steps (25 epochs), trade 2023, Qwen 2.5. Fig. 2: text train 2013-2018, trade 2019-2023, caption "6 years training history, 3 years trading", 400k steps (20 epochs). Figs. 3-6: 2M steps (100 epochs of 20k steps), "Trading Period: 2019-2023" in the text for Fig. 2 and none stated for Figs. 3-6 | §5.1.1, §5.1.2, §5.2, p. 3 |
| Infusion strengths | "10% infusion" (perturbations 0.9-1.1), 1% and 0.1% (0.99-1.01, 0.999-1.001) | §5.1, §5.3, p. 3-4 |
| Algorithms | PPO and CPPO (Ying et al. 2022), each with and without LLM scores; no hyperparameters, no network size, no seeds, no library versions | §4, p. 2 |
| Runs | "In these two runs (RL agents are stochastic)". Tables 1-3 list one value per model; Fig. 4 is "Second run" of Fig. 3. No interval, no seed count, no number of configurations tried | §5.2.2, p. 3-4 |
| Metrics | Information Ratio, CVaR, Rachev Ratio (Tables 1-3). The paper defines none of them; the README names them; their code is in the notebook (§2.2) | p. 3-4; README "Evaluation" |
| Costs, fills | Not stated in the paper | whole paper |
| Resources | "500k training steps used 16 GB (+Swap memory), whereas 2 million steps required 128 GB (+Swap memory)" | §6, p. 4 |
| Claims in the paper's own words | §4 "To our knowledge, we are the first to apply CVaR-PPO to stock trading" (not testable here). The eight claims we test are in §4 | p. 2 |

Table 1 (Fig. 3, 100 epochs; Information Ratio / CVaR / Rachev, p. 3): PPO
0.0100 / -0.0394 / 1.0637; CPPO -0.0148 / -0.0439 / 1.0404; PPO-DeepSeek -0.0093 /
-0.0338 / 0.9890; CPPO-DeepSeek 0.0078 / -0.0437 / 0.9818. Table 2 (PPO and
PPO-DeepSeek 10%, 1%, 0.1%, p. 4): 0.0100 / -0.0394 / 1.0637; -0.0093 / -0.0338 /
0.9890; -0.0252 / -0.0459 / 1.0394; -0.0011 / -0.0375 / 0.9536. Table 3 (CPPO and
CPPO-DeepSeek 10%, 1%, 0.1%, p. 4): -0.0148 / -0.0439 / 1.0404; 0.0078 / -0.0437 /
0.9818; -0.0032 / -0.0365 / 0.9573; -0.0060 / -0.0441 / 0.9789. Checked against
the rendered pages, not only the extracted text.

### 2.2 Code at the pinned commit (the L1 target)

| Item | Finding | Source |
| --- | --- | --- |
| Repository | `benstaf/FinRL_DeepSeek`, MIT, created 2025-02-06, 332 stars, 113 forks, 2 open issues (2026-11-02). A fork, `AI4Finance-Foundation/FinRL_DeepSeek` (138 stars), is announced in the README as the project's integration into FinRL; it is not the pin | GitHub API |
| **Pinned commit** | **`5c21a923214bca6370800efd45f8c6c1ef776ae7`** (2025-04-08 12:22:13 UTC, "Update README.md"), the head of `main`, the 36th commit. The scripts last changed on 2025-02-18 (the two score producers) and 2025-02-06 (everything else); only the notebook changed after that | GitHub API `commits/main`, `git log` of the pin |
| Paper and code | Paper 2025-02-11. The 15 agents were uploaded 2025-02-06, the agent-ready CSVs 2025-02-02/03 and the news-score files 2025-01-28/30 (the data repository's only later change is a README on 2025-03-22), so the released artifacts predate the paper. **The two score-producing scripts were added 2025-02-18, a week after the paper (issue #1), and the score files had been uploaded three weeks earlier**: that the committed scripts are the ones that made the files is **[UNVERIFIED]**. The notebook was replaced on 2025-04-08 | HF commit histories, `git log` of the pin, issue #1 |
| Install | `installation_script.sh`: Ubuntu, Miniconda, `conda create python=3.10`, conda-forge `gcc`, `swig`, `box2d-py`, `mpi4py`, `datasets`, `huggingface_hub`, `alpaca-py`, `selenium`; `pip install git+https://github.com/benstaf/FinRL.git` (head **`cd016b667da1860939b43bb77aba7ff4e35f780f`**, 2025-01-17, MIT, version 0.3.6, `python >=3.10`, `stable-baselines3 >=2.0.0a5`, `stockstats ^0.5`, `yfinance ^0.2`) and `benstaf/spinningup_pytorch` (head **`a20f4b539f43fa16085464b827cf5a8ceb3a7d0d`**, 2024-12-31, MIT, "no tensorflow"; its `setup.py` requires `gym[box2d,classic_control]~=0.15.3`). No versions of torch, numpy, pandas, gymnasium, mpi4py or datasets are pinned anywhere; the notebook's stored outputs show a Colab with Python 3.11. README: "128 GB RAM CPU instance recommended"; the author confirms CPU-only training on a 128 GB server (issue #5, 2025-02-23) | `installation_script.sh`, `pyproject.toml` and `setup.py` of the two forks, README, notebook outputs |
| Scripts | `train_ppo.py`, `train_cppo.py`, `train_ppo_llm.py`, `train_cppo_llm_risk.py` (+ `_01`, `_llama` variants), launched as `mpirun --allow-run-as-root -np 8 python train_ppo.py`; environments `env_stocktrading.py` (price only), `env_stocktrading_llm*.py` (PPO), `env_stocktrading_llm_risk*.py` (CPPO); score producers `sentiment_deepseek_deepinfra.py`, `risk_deepseek_deepinfra.py`; data builders `train_trade_data*.py`; `FinRL_DeepSeek_backtesting.ipynb`. The README links `FinRL_DeepSeek_backtest.ipynb`, which was deleted on 2025-04-08 | `git ls-files` at the pin |
| Network, PPO | MLP actor and critic, two hidden layers of 512, tanh, Gaussian policy with a state-independent `log_std` (initial -0.5); `gamma` 0.995, `lam` 0.95, `clip_ratio` 0.7, `pi_lr` 3e-5, `vf_lr` 1e-4, `train_pi_iters` 100, `train_v_iters` 100, `target_kl` 0.35, 20,000 steps per epoch, 100 epochs, split over the 8 MPI processes (2,500 steps each) | `train_ppo.py`: `ppo(...)` defaults and the argument parser |
| CPPO | The same network and PPO settings plus `alpha` 0.85, `beta` 3000, `nu_lr` 5e-4, `lam_lr` 5e-4, `nu_start` 0.1, `lam_start` 0.01, `nu_delay` 0.75, `delay` 1.0, `cvar_clip_ratio` 0.05. The per-step trajectory-return estimate is `ep_ret + v - r`, multiplied for the LLM variants by a risk factor from `risk_to_weight` (see §3); the CVaR term is subtracted from the advantage | `train_cppo.py`: `cppo(...)` and its training loop; `train_cppo_llm_risk.py`: `risk_to_weight` |
| Episode length | `max_ep_len` = 5,000 (PPO), 3,000 (CPPO and CPPO-DeepSeek), **1,000 (PPO-DeepSeek and PPO-Llama)** against 1,509 training sessions | `train_ppo.py` L411, `train_ppo_llm.py` L356, `train_cppo.py` L316, `train_cppo_llm_risk.py` L330 |
| Seeds | `--seed` is accepted (PPO default 42, CPPO default 0) and seeds torch and numpy as `seed + 10000 * rank`; the environment is not seeded. The committed logs do not record the seed used | `train_ppo.py` L534-536, L731; `train_cppo.py` L548 |
| Environment | FinRL `StockTradingEnv`: state = cash, 84 closes, 84 holdings, 8 indicators per stock (`macd`, `boll_ub`, `boll_lb`, `rsi_30`, `cci_30`, `dx_30`, `close_30_sma`, `close_60_sma`), plus 84 sentiment scores (PPO-DeepSeek: 925 dimensions) or 84 sentiment and 84 risk scores (CPPO-DeepSeek: 1,009); price-only is 841. Action: 84 values in [-1, 1], times `hmax` = 100, cast to integer shares. Reward: change of portfolio value times `reward_scaling` 1e-4. `initial_amount` 1,000,000; `buy_cost_pct` = `sell_cost_pct` = 0.001 | `train_ppo.py` (`env_kwargs`, `state_space`), logs (`Stock Dimension: 84, State Space: 841`), notebook cells 13-18 |
| **Fills** | The agent observes day-t closes and its order fills at the same day-t close (`_sell_stock` and `_buy_stock` use `state[index + 1]`, the close, then the day advances); the reward is the change of value from that close to the next close. Closes are `Adj Close` (dividend-adjusted); opens, highs and lows are split-adjusted only (AAPL on 2019-01-02: close 37.71, open 38.72 in the trade file) | `env_stocktrading.py` (`step`, `_sell_stock`, `_buy_stock`); `finrl/meta/preprocessor/yahoodownloader.py` L75-88 of the FinRL fork; trade file |
| Trading environment | Notebook: `StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', ...)` for every agent. When the day's VIX is at or above 70 the action is replaced by -`hmax` for every stock and `_sell_stock` then **sells every share held at that day's close** and `_buy_stock` buys nothing. The training environments pass no threshold | notebook cells 19-23; `env_stocktrading.py` (`step`, `_sell_stock`, `_buy_stock`) |
| Backtest | The agent is run once through the 1,257 trade sessions with `act.step`, which **samples** from the Gaussian policy (no `deterministic` path). Equity is normalised to 1,000,000 at the first value | notebook cell 38 (`DRL_prediction`) |
| Benchmark | `YahooDownloader(..., ticker_list=["ndx"])`; the stored output reports `Shape of DataFrame: (937, 8)` against 1,257 trade sessions. Series are intersected with the trade dates (`filter_to_common_dates`), so metrics are computed on the common dates | notebook cells 50-51, 64-70 |
| Metrics | `calculate_metric`: daily returns from `pct_change` of the aligned equity series; **Information Ratio** = mean / std of (strategy - benchmark) daily returns, not annualised; **CVaR** = mean of strategy returns at or below their 5th percentile; **Rachev ratio** = mean of returns at or above the 95th percentile / absolute mean of returns at or below the 5th | notebook cells 79-83 |
| Stored outputs | Notebook cell 81 prints Information Ratio 0.0010 for PPO and -0.0085 for PPO-DeepSeek 10% (CVaR -0.0403, -0.0370; Rachev 1.0425, 1.0025). The paper's Tables 1-2 give 0.0100 and -0.0093 for the same agents | notebook cell 81 outputs; paper Tables 1-2 |
| Logs | 15 training logs, 8 MPI processes each for the 100-epoch runs and 4 for the 20-epoch ones (the host name `scw-cranky-chebyshev` suggests a Scaleway instance, **[INFERENCE]**). The 100-epoch logs end at `TotalEnvInteracts` 2e6 after 16,800 s (`output_ppo.log`), 16,500 s (`output_cppo.log`), 28,500 s (`output_ppo_deepseek.log`) and 23,200 s (`output_cppo_deepseek_risk_100_epochs.log`) of wall time, and up to 68,700 s for other variants; which log made which released agent is **[UNVERIFIED]**. Hardware beyond "128 GB" is **[UNVERIFIED]** | `output_*.log` |
| Issues | Seven; none reports a problem with the results; #6 asks where the FinRL Contest 2025 Task 1 results are and was closed with a pointer to the contest page | GitHub API |

### 2.3 Data, scores and agents (pinned revisions, hashes computed 2026-11-02)

The files below were streamed from Hugging Face only to be hashed and are not
committed or stored by this repository. The SHA-256 and size of each were
recomputed and **all 29 equal the values Hugging Face reports** for the same
revision. The raw FNSPID file (23.2 GB) was not downloaded: its hash is the one
Hugging Face reports, `1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da`,
and is **[UNVERIFIED]** by us.

| Repository | Revision (commit) | Last change |
| --- | --- | --- |
| `datasets/Zihan1004/FNSPID` (CC BY-NC 4.0) | `bf9189c41527198897d1af3e17b1a0095279fc45` | 2024-04-09 |
| `datasets/benstaf/nasdaq_2013_2023` (no licence stated) | `b80bc15e4320eac68f53cfdd2fff3365e55dfedd` | 2025-03-22 (README only) |
| `datasets/benstaf/nasdaq_news_sentiment` | `04532f608f416991dea9b9952775102f8e2216e5` | 2025-01-28 |
| `datasets/benstaf/risk_nasdaq` | `1c46dd45482f681e41e3e70a3a353eb6d165e845` | 2025-01-30 |
| `benstaf/Trading_agents` | `2153a7266aac75f9613fdae1bde09dbd38691c59` | 2025-02-06 |

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `nasdaq_2013_2023/train_data_2013_2018.csv` | 36,467,048 | `92eb993137595e9c461091e9f3569295d0050f5536c75b707468ba6d2197657b` |
| `nasdaq_2013_2023/train_data_deepseek_sentiment_2013_2018.csv` | 36,968,872 | `984bdf51cae51a5cd4b9674fa151cfa9e429bc7775710f23ee79dc2cbe2dab1a` |
| `nasdaq_2013_2023/train_data_deepseek_risk_2013_2018.csv` | 37,239,993 | `dd901eadd7996c991b468f1792ac24d53027979e3ed7b0d492af8f8988654ae7` |
| `nasdaq_2013_2023/train_data_llama_sentiment_2013_2018.csv` | 36,962,662 | `b228b5fb1b00a290904a84bbba1eb082446ef6f9e4e074ca772cbce10391afec` |
| `nasdaq_2013_2023/train_data_llama_risk_2013_2018.csv` | 37,176,432 | `0aee2ab7d1ae5f087c1d262dca5335205519b693fa3411b80da71b712f45f92a` |
| `nasdaq_2013_2023/trade_data_2019_2023.csv` | 30,518,762 | `01587b66236b5563df8f871f0110bbf752f1c593427a346192c20e271efffd3b` |
| `nasdaq_2013_2023/trade_data_deepseek_sentiment_2019_2023.csv` | 30,953,286 | `1d827b3685e914bdbb4ee6270e556ba1a4490976a717bc85dc729f26ff980fef` |
| `nasdaq_2013_2023/trade_data_deepseek_risk_2019_2023.csv` | 31,198,826 | `e5f510b815f7e6d05dba9307b71f2ca738f7370c77a4206b17b5fbd5dd23898d` |
| `nasdaq_2013_2023/trade_data_llama_sentiment_2019_2023.csv` | 30,942,473 | `ee51e9fd4f93531339baaa8e2cee025ec99b720b71b3e01de974fc3704eaaa5b` |
| `nasdaq_2013_2023/trade_data_llama_risk_2019_2023.csv` | 31,103,363 | `57bc352670db61d7c648eefa18b7d1191177814379e7b169c7629a491db0e16b` |
| `nasdaq_news_sentiment/sentiment_deepseek_new_cleaned_nasdaq_news_full.csv` | 588,735,065 | `54470e391419ae1f374c1f6b68a936bc036100e36bcf5ea5d8a33dd3c82ac1e9` |
| `nasdaq_news_sentiment/sentiment_llama_cleaned_nasdaq_news_full.csv` | 588,710,540 | `d75bb8994b90ea0cb51628619e719a26c9d509eba32649429d11b1ceb5ff16ea` |
| `risk_nasdaq/risk_deepseek_cleaned_nasdaq_news_full.csv` | 588,737,496 | `33251916177feceebec6f2bec58e81cfbf10fb9115178504cb044498c20c7e73` |
| `risk_nasdaq/risk_llama_cleaned_nasdaq_news_full.csv` | 588,548,709 | `73cbe9bbcc41e593e47dbf853132228e104e626bf9df461384049bcb6da40a95` |
| `Trading_agents/agent_ppo_100_epochs_20k_steps.pth` | 5,730,102 | `f138a91c36e0b88b27d58bf41549de24717aec48e7107641f3dfa1838b51177b` |
| `Trading_agents/agent_cppo_100_epochs_20k_steps.pth` | 5,730,119 | `c7017d10df45958f1a09ee8e009d8c99346be867ff312b03876ed1b8cfb289b1` |
| `Trading_agents/agent_ppo_deepseek_100_epochs_20k_steps.pth` | 6,074,319 | `ae05dd713d1c0e8a880daad0a3be49db0314c50211805272f8a61dd84794ef84` |
| `Trading_agents/agent_ppo_deepseek_100_epochs_20k_steps_1.pth` | 6,074,540 | `3f54f756654dc9a23c9c6fe6836a0be5ac03cc1fecf8952f020e58f63291fb8c` |
| `Trading_agents/agent_ppo_deepseek_100_epochs_20k_steps_01.pth` | 6,074,370 | `0969fc0b9d4c49aa816b2ec88dd321ffa619d765df27d79ef8975797747b8d5b` |
| `Trading_agents/agent_cppo_deepseek_100_epochs_20k_steps.pth` | 6,418,400 | `b8dc3e1cc247a16c258d6940644008f3a491f0dd962145d37659a66c55b99d44` |
| `Trading_agents/agent_cppo_deepseek_100_epochs_20k_steps_1.pth` | 6,418,519 | `0995acb84653b5ffbfaab13fed295ff1efa8443df66f6dfe80e0ac82280fe53d` |
| `Trading_agents/agent_cppo_deepseek_100_epochs_20k_steps_01.pth` | 6,418,349 | `672fa6bfc0e656f724833fac2046f04f5598fd9a87a1ea457bcaf9b1a2edcda0` |
| `Trading_agents/agent_ppo_llama_100_epochs_20k_steps.pth` | 6,074,268 | `06a1dbc0e65a7868c9c98db19c53311fc23ffc954faab3b913df4f4c05aa7cd9` |
| `Trading_agents/agent_ppo_20_epochs_20k_steps.pth` | 5,730,085 | `238632dc96b40aa21a3993e1cac90578b7f84ea8e6811aebcfc4cbf3ac37443e` |
| `Trading_agents/agent_cppo_20_epochs_20k_steps.pth` | 5,730,102 | `6c9771b11a99c389877c35317ef7d542208a35264c47051ce460791e207ec985` |
| `Trading_agents/agent_ppo_deepseek_20_epochs_20k_steps.pth` | 6,074,302 | `3e799bdf91c48654b4e7a0f3c090abff69b45a5ec88f592511c1dd34bc446d41` |
| `Trading_agents/agent_ppo_llama_20_epochs_20k_steps.pth` | 6,074,251 | `67c515df96ed32e57a37566d86c3547c63987d9e758ebe2984749a6f771cedfe` |
| `Trading_agents/agent_cppo_llama_20_epochs_20k_steps.pth` | 6,418,332 | `d9f0fb6fe944f01a6f5e38c6fe6a1e92369b92252e29ed16101766e92b9d8087` |
| `Trading_agents/agent_deepseek_20_epochs_20k_steps.pth` | 6,418,298 | `98d4c116d6e17ade1ed6710488446c3d10fe01fbd7350974bbf146641868b693` |

The evidence manifest repeats these and adds the hash of every file of the
pinned checkouts (repository and both forks).

**What the agent-ready files are.** Each CSV is one row per stock and day. The
trade files hold 105,588 rows = 84 tickers x 1,257 sessions, 2019-01-02 to
2023-12-28; the train files 126,756 rows = 84 x 1,509 sessions, 2013-01-02 to
2018-12-28. Columns: `date`, `tic`, `close`, `high`, `low`, `open`, `volume`,
`day`, the eight indicators, `vix`, `turbulence`, and in the LLM files
`llm_sentiment` and `llm_risk`. There is no news text, no model id and no
timestamp finer than the day. 2023-12-29, a session, is absent from the trade
files; why is **[UNVERIFIED]**. The 8 indicators, `vix` and `turbulence` are
FinRL's, computed from Yahoo data of the day the authors downloaded it, which
is **[UNVERIFIED]** (the download date is not recorded).

**LLM scores.** In the trade files 56.5% of ticker-days have no sentiment score
and 56.3% no risk score (by year 2019-2023, without a sentiment score: 45.6%,
59.9%, 69.6%, 56.7%, 50.5%); in the train files 62.9% (by year 2013-2018: 76.4%,
72.7%, 64.4%, 61.3%, 54.7%, 47.8%). The environments read a missing sentiment as 0
and a missing risk as 3 (neutral). Scores outside 1-5 occur (75 sentiment
values of 0 in the trade file). One streaming pass over
`sentiment_deepseek_new_cleaned_nasdaq_news_full.csv` (the other three news
files were not parsed) gives **127,176 rows**, one per (day, ticker), 89 tickers,
stamped 2009-07-07 to 2024-01-09, scores 1: 1,615, 2: 10,930, 3: 84,403, 4:
27,109, 5: 1,939, 0: 228, empty: 952. The paper says the dataset was reduced
"to 2 million records"; what that count refers to is **[UNVERIFIED]**, and the
random choice of "a representative news article per stock per day" has no
recorded seed. The columns are `Date, Article_title, Stock_symbol, Url,
Publisher, Author, Article, Lsa_summary, ...` and the score; **99.7% of the
127,176 stamps are `YYYY-MM-DD 00:00:00 UTC`**, so the time of day is not
recorded. The score prompt sees `Lsa_summary`, an extractive summary from FNSPID,
not the article.

**Scoring models, as far as the code shows.** `sentiment_deepseek_deepinfra.py`
and `risk_deepseek_deepinfra.py` call `deepseek-ai/DeepSeek-V3` through the
DeepInfra OpenAI-compatible API with `temperature=0`, `max_tokens=50`, a
system prompt plus one worked example, and batches of 5 (sentiment) or 4 (risk)
consecutive rows scored in one call, with every text in a batch labelled with
the first row's ticker. The model snapshot DeepInfra served in January 2025 is
not recorded (**[UNVERIFIED]**; DeepSeek-V3 was released 2024-12-26, so
**[INFERENCE]** the original V3). The Llama files have no producing script at
the pin; commented lines name `meta-llama/Llama-3.3-70B-Instruct`
(**[INFERENCE]** it is the model of the `llama` files). Qwen 2.5 72B appears
only in a commented line and in Fig. 1; no Qwen file is released.

**Released agents.** Read from the state dicts only: every agent is an MLP with
two hidden layers of 512, 84 outputs and a 84-dimensional `log_std`; the input
width is 841 for `agent_ppo_*` and `agent_cppo_100/20_epochs`, 925 for the
PPO-DeepSeek and PPO-Llama agents, 1,009 for the CPPO-DeepSeek, CPPO-Llama and
`agent_deepseek_20_epochs` agents. The notebook (cells 31-37) loads, for
the 100-epoch rows, PPO and CPPO, PPO-DeepSeek `..._deepseek_100_epochs_20k_steps`
(10%), `_1` (1%), `_01` (0.1%) and CPPO-DeepSeek `..._cppo_deepseek_..` (10%, 1%,
0.1%) the same way; the 20-epoch files are the six agents of Fig. 2, with
`agent_deepseek_20_epochs_20k_steps.pth` (a 1,009-input agent) read as
CPPO-DeepSeek (**[INFERENCE]**: the notebook loads it into a variable named
`loaded_cppo_llama_risk`).

### 2.4 Universe and survivorship

The authors' tickers are `nasdaq_100_tickers_july_17_2023` in
`train_trade_data.py`: 101 names, the Nasdaq-100 as of **2023-07-17**, of which
84 are in the data files. The 17 missing are ABNB, CEG, CRWD, DDOG, GEHC, GFS,
JD, KHC, LCID, MRNA, PDD, PYPL, SGEN, TEAM, TTD, ZM and ZS. **[INFERENCE]**
Sixteen of them were listed in 2014 or later and so lack Yahoo history back to
2013-01-01. The script adds SGEN from a `seagen.csv` that is not in the
repository; why SGEN is nevertheless absent is **[UNVERIFIED]**. The script
does not say why any name is missing. The 84 are: AAPL, ADBE, ADI, ADP, ADSK, AEP, ALGN, AMAT, AMD,
AMGN, AMZN, ANSS, ASML, AVGO, AZN, BIIB, BKNG, BKR, CDNS, CHTR, CMCSA, COST,
CPRT, CSCO, CSGP, CSX, CTAS, CTSH, DLTR, DXCM, EA, EBAY, ENPH, EXC, FANG, FAST,
FTNT, GILD, GOOG, GOOGL, HON, IDXX, ILMN, INTC, INTU, ISRG, KDP, KLAC, LRCX,
LULU, MAR, MCHP, MDLZ, MELI, META, MNST, MRVL, MSFT, MU, NFLX, NVDA, NXPI, ODFL,
ON, ORLY, PANW, PAYX, PCAR, PEP, QCOM, REGN, ROST, SBUX, SIRI, SNPS, TMUS, TSLA,
TXN, VRSK, VRTX, WBA, WBD, WDAY, XEL.

- **Survivorship, stated.** The universe is chosen by membership on 2023-07-17
  and by listing before 2013, so any company that was in the index in 2019 and
  left, or was delisted, before July 2023 is absent by construction, and every
  company that joined after 2019 and was listed before 2013 enters with
  hindsight. The authors' test therefore starts from stocks that were
  selected by what happened after the test began. This is **survivorship- and
  look-ahead-biased in favour of strategies that hold the universe**, and our
  L2 universe inherits it.
- **Yahoo today (checked 2026-11-02 with yfinance).** 80 of the 84 return daily
  history from 2013-01-02 to 2023-12-29. ANSS, EA, WBA and WBD return no data.
- **A stand-in made of market data only** (the authors' adjusted closes of the
  84 names, equal weight rebalanced daily, 1,256 returns from 2019-01-03 to
  2023-12-28, against `^NDX` and QQQ adjusted close, Yahoo 2026-11-02): the
  equal-weight portfolio compounds to +220.4% against +165.7% (`^NDX`, price)
  and +174.4% (QQQ); its daily information ratio against QQQ is 0.0237 and
  against `^NDX` 0.0285, with 7.5% annual tracking error. The paper's best
  information ratio is 0.0100. **The index is beatable by holding the survivor
  list**, which is why L2 reports the equal-weight baseline beside the index
  (§5.2) and why a "beats the Nasdaq-100" verdict is qualified in §7.

### 2.5 Prior re-tests (searched 2026-11-02)

Not an exhaustive review. "FinRL-DeepSeek Revisited: A Reproduction and
Multi-Seed Robustness Analysis of LLM-Infused Risk-Sensitive Trading Agents"
(V. A. Mbouyom Leumale, Cambridge Open Engage, v1 15 Sept 2026, CC BY 4.0, not
peer reviewed; also SSRN 7459518 and 7159918) re-implements PPO, CPPO and their
DeepSeek variants from scratch in PyTorch on the authors' agent-ready data,
backtests 2019-2023 against an equal-weight buy-and-hold, and reports that with
one seed the LLM improves both agents' return and Sharpe ratio, while with two
seeds the PPO-DeepSeek ranking against PPO reverses and CPPO-DeepSeek has the
smallest spread (abstract only was read; the full text and code were **not**).
It uses neither the authors' code nor their agents and has two seeds. We found
no re-test with the authors' code, released agents, five or more seeds,
next-open fills and a recall probe; the report re-checks before publication.

## 3. What the authors' own artifacts show (observed before the first run)

Read from the pinned files, without running the method. They are disclosed
because they shape the plan.

1. **PPO-DeepSeek was trained on a shorter episode than PPO.** The committed
   logs show a mean episode length of about 1,510 for PPO, CPPO and CPPO-DeepSeek
   and **1,000 for every PPO-DeepSeek and PPO-Llama run**
   (`train_ppo_llm.py` sets `max_ep_len=1000`; `train_ppo.py` 5,000). Episodes
   restart at 2013-01-02, so those agents never train on the sessions from
   2016-12-20 to 2018-12-28 (509 of 1,509). A difference between PPO and
   PPO-DeepSeek therefore mixes the LLM with 34% less training data. §5.1
   includes a control for it.
2. **The committed scripts are not the 10% variants of the paper's main
   figures.** `train_ppo_llm.py` imports `env_stocktrading_llm_01` (0.1%) and
   saves `..._01.pth`; `train_cppo_llm_risk.py` uses the risk map 0.99-1.01 (1%)
   and saves `..._99_101.pth`; the 0.9-1.1 map is only in `train_cppo_llm_old.py`
   (25 epochs, `max_ep_len` 1,500). The agents named "10%" are on Hugging Face,
   and the script that made them is **[UNVERIFIED]**.
3. **Text and code differ.** (a) The paper's $S_f$ leaves a neutral score
   unchanged; `env_stocktrading_llm.py` (the 10% PPO environment) multiplies
   neutral-score actions by 0.98. (b) The paper applies LLM scores to actions
   and returns; the code also appends them to the observation (925 and 1,009
   dimensions). (c) The paper defines $\alpha$ as "e.g., 0.05 for the worst 5%";
   the code uses `alpha` 0.85. (d) The paper says the risk factor multiplies the
   trajectory return; the code multiplies the running estimate `ep_ret + v - r`.
4. **The plotted window is not 2019-2023.** The benchmark download has 937 rows
   for 1,257 trade sessions, the curves are intersected with it, and the curves
   of Figs. 2-6 start in early 2020, a few weeks before the March 2020 low (read
   off the rendered axes, **[UNVERIFIED]** to the day), while the text says
   "Trading Period: 2019-2023" and Fig. 2's caption says "3 years trading".
   Table 1-3 metrics are computed on that intersection. Which series `ndx`
   resolved to in 2025 is **[UNVERIFIED]**: today `ndx` returns nothing and
   `^NDX` works.
5. **The paper and its own notebook differ on one agent.** Cell 81 of the
   notebook prints a PPO information ratio of 0.0010 (Table 1: 0.0100) and 10%
   PPO-DeepSeek -0.0085 (Tables 1-2: -0.0093). With sampled actions, the same
   weights give different curves; the paper's own Figs. 3 and 4 are two such
   runs. Fig. 1's legend gives PPO 25 epochs an IR of -0.0148, the value Table 3
   gives CPPO at 100 epochs; whether that is a coincidence or a copy is
   **[UNVERIFIED]**.
6. **Selection on the test period.** The paper reports "Early Stopping: 400-500k
   training steps" results beside the 2M-step results, without saying how the
   stopping point was chosen, three infusion strengths, three LLMs and several
   windows, all on the 2019-2023 or 2023 trading data. The repository holds 15
   released agents and 15 logs, with variants (`_96_104_bumps`, `_98_95`,
   `_99_101_bumps`) whose results are not in the paper. The number of
   configurations tried is at least 15 and otherwise unknown. No validation
   window exists in the code.
7. **The news is dated to the day and used the same day.** A score with date D is
   in the state of day D, and the order fills at the close of D (§2.2). The
   timestamp has no time of day, so some of the news behind a day-D score may
   have been published after the day-D close.
8. **The environment liquidates everything on five sessions.** VIX closes at or
   above 70 on 2020-03-12, 03-16, 03-17, 03-18 and 03-19 (max 82.69 on
   2020-03-16, in the trade file); the notebook's threshold then sells every
   share held at that day's close and blocks buying, for every agent. The
   training environments have no such rule. R0 found the same kind of rule
   puts every agent in cash on the same days.
9. **Scores are sparse.** More than half of the ticker-days have no score
   (§2.3), so a "10% infusion" changes at most the actions on the 43.5% of
   ticker-days that have one.
10. **The released news file does not match the paper's description of its
    size.** 127,176 rows, one per day and ticker, against "2 million records"
    (§2.3); and the committed score scripts were added after the score files
    were uploaded (§2.2).

## 4. Claims, quoted from the paper

Ids match `claims.json`, with a `-l1` or `-l2` suffix per level. "x" is the
quantity of §7, oriented so that larger means the claim; "c" the claimed size
(the paper's table value, or 0 where the paper gives a direction only).

| Id | Quote (verbatim) | Source | x, c |
| --- | --- | --- | --- |
| `ppo-beats-nasdaq100` | "In these two runs (RL agents are stochastic), PPO and CPPO-DeepSeek outperform other methods, including the Nasdaq 100 benchmark." | §5.2.2, Figs. 3-4, p. 3 | annualised geometric return of PPO minus the Nasdaq-100's; c = 0 |
| `cppo-deepseek-beats-nasdaq100` | same sentence | same | the same for CPPO-DeepSeek (10%); c = 0 |
| `llm-lowers-ppo-ir` | "For PPO-DeepSeek, stronger LLM infusion mostly degrades performance, even for tiny (0.1%) perturbations." | §5.3.1, Table 2, Fig. 5, p. 4 | PPO's information ratio minus PPO-DeepSeek's, mean over 10%, 1% and 0.1%, annualised; c = 0.0219 x sqrt(252) = **0.3471** |
| `llm-raises-cppo-ir` | "For CPPO-DeepSeek, stronger LLM infusion improves performance" | §5.3.2, Table 3, Fig. 6, p. 4 | CPPO-DeepSeek (10%) information ratio minus CPPO's, annualised; c = 0.0226 x sqrt(252) = **0.3588** |
| `ppo-wins-bull-market` | "PPO seems to perform better in Bull markets, and CPPO-DeepSeek in Bear markets, the transition being at the end of 2021 before the war in Ukraine and the crisis that followed." | §5.2.2, p. 3; README "Bull market -> PPO" | PPO's annualised return minus CPPO-DeepSeek's, 2019-01-02 to 2021-12-31 (757 sessions); c = 0 |
| `cppo-deepseek-wins-bear-market` | same sentence | same; README "Bear market -> CPPO-DeepSeek" | CPPO-DeepSeek's annualised return minus PPO's, 2022-01-03 to 2023-12-28 (500 sessions); c = 0 |
| `llm-always-worsens` | "The use of LLM always worsens performance in this test." | §5.1.2, Fig. 2, p. 3 (400k steps, 20 epochs) | the smallest, over the DeepSeek and Llama variants of PPO and CPPO, of (information ratio without the LLM minus with it), annualised; positive only if every variant is worse; c = 0 |
| `deepseek-beats-llama` | "DeepSeek V3 is slightly better than Llama 3.3 all other things being equal." | §5.1.2, Fig. 2, p. 3 | information ratio with DeepSeek scores minus with Llama scores, mean over PPO and CPPO, annualised; c = 0 |

The annualised information ratio is the daily one of the notebook times
sqrt(252) (15.8745), so the claimed values in `claims.json` are the paper's
table values times that constant; the paper's own daily numbers are shown
beside ours in the report. The Table 1-3 numbers for CVaR and Rachev ratio, and
the information ratio of each model, are **reported beside the claims and get no
verdict**: the paper states no sentence about them, and its information ratios
(|IR| at most 0.0252 daily) are within one standard error of zero (§8). The paper's Fig. 1 claim ("integrating
LLM stock recommendations consistently improves PPO in terms of cumulative
returns", Qwen 2.5, train 2019-2022, trade 2023) is **not tested**: the repository
has no script for a 2019-2022 training window and no Qwen scores, and the
claim is about 25-epoch agents that are not released. The statement that the
winner "is" the bull-market or bear-market agent is tested only as the sign of
the two return gaps above; the paper gives no crossing date other than "the end
of 2021", which we take as 2021-12-31.

`claims.json` holds these as 16 entries (8 claims x 2 levels) with
`verdict: null` until the report. The claims loader behind `rl-pricing bench
reports` rejected a null verdict, which CI would have treated as a failure, so
this change makes it accept `null` as "pending" and leave such entries out of
the generated verdict tables until a verdict is set.

## 5. Design

### 5.1 L1: the authors' code, data and agents

- **Code and data:** the pinned commit and forks (§2.2), the HF revisions of
  §2.3, no file changed. Environment: Linux, Python 3.10 (the installer's
  version), `mpi4py`, the two forks at their pinned commits, the remaining
  libraries at the newest versions that resolve on the build day, recorded in a
  lock file in the evidence (the authors pin none). The benchmark is `^NDX`
  (Yahoo, price index) because `ndx` returns nothing today.
- **L1-A, the released agents (always run).** The 15 `.pth` files of §2.3 are
  loaded into the authors' network class (the notebook's `MLPActorCritic`,
  hidden sizes 512, 512) and run through the authors' `DRL_prediction` on the
  authors' trade CSVs and environments (notebook cells 9-23), **20 sampled
  rollouts per agent** with sampling seeds 0-19 (`torch.manual_seed`), so the
  sampling noise that separates Table 1 from the notebook's stored outputs is
  measured. Rollouts with the policy mean (no sampling) are reported beside
  them. Each rollout ends at the trade sessions' last value; metrics follow §6.
- **L1-B, retraining (runs if the go/no-go of §14 passes).** The authors'
  scripts retrain **PPO, CPPO, PPO-DeepSeek 10%, 1% and 0.1%, and CPPO-DeepSeek
  10%** at their settings (100 epochs of 20,000 steps, `mpirun -np 8`),
  training seeds **1, 2, 3, 4, 5** by `--seed` (their defaults 42 and 0 are
  not used), each evaluated with 10 sampled rollouts (seeds 0-9) whose metrics
  are averaged per seed; 30 trainings. The scripts for PPO-DeepSeek 10% and 1%
  and for CPPO-DeepSeek 10% are not committed as such (§3.2): they are the
  committed scripts with the environment import changed (and, for CPPO-DeepSeek
  10%, the 0.9-1.1 risk table of `train_cppo_llm_old.py` at 100 epochs); every
  such edit is a deviation (§11). `llm-always-worsens` and `deepseek-beats-llama`
  at L1 rest on L1-A only: their 20-epoch and Llama agents are not retrained.
- **L1-C, the episode-length control.** PPO-DeepSeek 10% retrained with
  `max_ep_len=5000`, the only change, seeds 1-5 (5 trainings), to separate
  the LLM from the shorter episode of §3.1. Not a verdict-bearing arm.
- **Basis of the L1 verdict:** L1-B if all of its seeds finish, otherwise L1-A
  (one released checkpoint per row, 20 rollouts), and the report says which.
  An L1-A verdict says nothing about training variance.
- **Window:** the authors' trade sessions, 2019-01-02 to 2023-12-28 (1,257
  sessions), as stated in the paper; secondary, the sessions from 2020-03-02
  (965), the nearest to the plotted window (§3.4). The VIX >= 70 rule stays on,
  as in the notebook.
- **Outcome measured:** the three metrics of Tables 1-3 for every row, and the
  claims of §4.
- **If L1 cannot be built** by 2026-11-24 (the training scripts do not start in
  the environment, or no machine with 128 GB is available), the report says L1-B
  and L1-C were **not run**, and L1 rests on L1-A. If the released agents do not
  load (a shape or key mismatch), L1 is **not run** and the report says why.

### 5.2 L2: this repository's engine

- **Engine:** `rl.bench`, decisions at the close of t, fills at the open of
  t+1, proportional fee on turnover, long-only weights summing to at most 1,
  the remainder in cash (zero yield), fractional shares.
- **New task `us-ndx80`** (id fixed here; it enters `rl.bench.tasks` in a later
  commit, and a task never changes after that):

  | Field | Value |
  | --- | --- |
  | source | `yahoo` (forward dividend adjustment, as every Yahoo task) |
  | tickers | the 84 of §2.4 minus ANSS, EA, WBA, WBD (not served today) = **80**: AAPL, ADBE, ADI, ADP, ADSK, AEP, ALGN, AMAT, AMD, AMGN, AMZN, ASML, AVGO, AZN, BIIB, BKNG, BKR, CDNS, CHTR, CMCSA, COST, CPRT, CSCO, CSGP, CSX, CTAS, CTSH, DLTR, DXCM, EBAY, ENPH, EXC, FANG, FAST, FTNT, GILD, GOOG, GOOGL, HON, IDXX, ILMN, INTC, INTU, ISRG, KDP, KLAC, LRCX, LULU, MAR, MCHP, MDLZ, MELI, META, MNST, MRVL, MSFT, MU, NFLX, NVDA, NXPI, ODFL, ON, ORLY, PANW, PAYX, PCAR, PEP, QCOM, REGN, ROST, SBUX, SIRI, SNPS, TMUS, TSLA, TXN, VRSK, VRTX, WDAY, XEL |
  | `start` | 2013-01-01 (the authors' training start) |
  | `validation_start` | 2018-01-01 (the authors' last training year is held out for checkpoint selection; the authors had no validation window) |
  | `test_start` / `test_end` | 2019-01-01 / 2023-12-29 (exclusive): sessions 2019-01-02 to 2023-12-28, the authors' last date |
  | fee | 0.001 per side (the authors' `buy_cost_pct`) |
  | `periods_per_year` / `live` | 252 / false (no frozen live track) |

  **Survivorship status, stated:** survivorship- and look-ahead-biased in the
  way of §2.4 (membership on 2023-07-17), and further by the four names Yahoo
  no longer serves; the task exists to re-test the paper, not to rank
  strategies, as `paper-2021` does.
- **Policies.** The lab's `ppo` reference (Stable-Baselines3 PPO on the engine's
  reference observation) and, from M3, `cvar-ppo`, each at its shipped defaults at
  the commit used, trained on 2013-01-01 to 2017-12-31, checkpoint chosen on
  2018 by net Sharpe, tested on the window above. Tuning on the test window is
  excluded; every configuration tried counts in `trials`. This is the paper's
  *method* (PPO, CPPO, with and without LLM scores) on the lab's agents and
  observation, not the authors' network or their eight indicators (§11).
- **LLM features (the `jsonl` source of M2).** Items are the authors' scores with
  their own dates: one item per (ticker, day) with `fields` = `llm_sentiment`,
  `llm_risk` and the producing model id, `published_at` taken from the news
  file's stamp, text hash only (the text is FNSPID's and stays local). The two
  score files of a model are joined on date, ticker and title; rows that cannot
  be joined are dropped and counted (**[UNVERIFIED]** that the two files list
  the same articles). Two alignments:
  - **lagged (primary):** `published_at` set to 23:59:59 UTC of the stamped day,
    so a score is visible from the decision of the next session. No leak is
    possible whatever the hidden time of day.
  - **same-day (sensitivity, the authors' alignment):** the stamp as it is,
    visible at the close of its own day (00:00:00 UTC of D is before the close
    of session D). Used only by two arms below.
- **How the features enter** (the code's behaviour, §3.3; s is the strength):
  (i) *sentiment on actions.* With score k and the sign of the change of a
  stock's target weight against its current weight, a matched change (score 5
  with a buy, score 1 with a sell) is multiplied by 1 + s, a half-matched one
  (4 with a buy, 2 with a sell) by 1 + s/2, a half-opposed one (4 with a sell, 2
  with a buy) by 1 - s/2, an opposed one (5 with a sell, 1 with a buy) by 1 - s;
  a neutral or missing score changes nothing (s = 0.1, 0.01, 0.001 give the
  authors' 0.9-1.1, 0.99-1.01 and 0.999-1.001 tables). (ii) *scores in the
  observation:* the two scores are appended to each asset's features (missing
  sentiment 0, missing risk 3). (iii) *risk on the CVaR term* (CVaR-PPO only):
  the return that the CVaR term is computed on is multiplied by the
  holdings-weighted mean over invested assets of 1 + s x (r - 3)/2 for risk
  score r, a missing score counting as 3 (the authors' map 0.9, 0.95, 1, 1.05,
  1.1 at s = 0.1). The neutral 0.98 of §3.3a is not used.
- **Arms (all declared; every one counts in `trials`):**

  | Arm | Policy | Scores | s | Alignment |
  | --- | --- | --- | --- | --- |
  | P0 | `ppo` | none | - | - |
  | P10, P1, P01 | `ppo` | DeepSeek | 0.1, 0.01, 0.001 | lagged |
  | C0 | `cvar-ppo` | none | - | - |
  | C10, C1, C01 | `cvar-ppo` | DeepSeek | 0.1, 0.01, 0.001 | lagged |
  | P10-L, C10-L | `ppo`, `cvar-ppo` | Llama | 0.1 | lagged |
  | P10-S, C10-S | `ppo`, `cvar-ppo` | DeepSeek, **placebo** | 0.1 | lagged |
  | P10-D, C10-D | `ppo`, `cvar-ppo` | DeepSeek | 0.1 | same-day |

  The placebo gives each ticker its own score series circularly shifted by an
  offset drawn once per ticker from [126, n - 126] sessions (numpy `default_rng(0)`),
  which keeps every ticker's marginal distribution, missingness and
  autocorrelation and breaks the alignment with returns. 14 arms x 5 seeds
  (0-4) = 70 fits; **trials = 14** (+1 per configuration added).
- **Baselines (same window, same fee):** equal weight bought on the first fill
  and held; the engine's monthly-rebalanced 1/N; **the Nasdaq-100**, as QQQ
  adjusted close (a total-return proxy, the basis of the engine's prices) with
  `^NDX` (price index, the authors' benchmark as far as §2.2 shows) beside it.
  The claims of §4 are tested against QQQ at L2 and against `^NDX` at L1 (the
  authors' own comparison), both are shown at both levels, and the
  dividend bridge (QQQ minus `^NDX`) is reported.
- **Claim mapping at L2:** the §4 quantities use the arms P0, P10 (PPO and
  PPO-DeepSeek), C0, C10, the three strengths for `llm-lowers-ppo-ir` and
  `llm-raises-cppo-ir`, P10-L and C10-L for the Llama claim, and the 2019-2021 and
  2022-2023 sub-windows for the regime claims, the sub-windows cut from the one
  continuous run per seed (no refit). A claim is tested only if every arm it
  names finished.
- **L2-a, the released agents on this engine (secondary, no verdict, runs only
  if the check passes).** Each of the 15 agents' observation is rebuilt on the
  engine's data (the 80 Yahoo names, the authors' file for the four Yahoo does
  not serve), the agent's integer share trade is applied to the engine's
  holdings at the decision-close prices and the resulting holdings are sent as
  target weights, filled at the next open. It runs only if the rebuilt
  2019-01-02 state matches the authors' CSV state vector to 1e-3 relative on
  the 80 names, and is otherwise reported as **not run**. It would isolate the
  effect of next-open fills on fixed agents.
- **Seeds:** five (0-4) per arm, a seed setting the Python, NumPy, PyTorch and
  environment seeds and the checkpoint search; the same seed across arms.

### 5.3 L3

None. The scores exist for 2013-2023 only and a later window would need new LLM
scoring with models that would themselves post-date the paper; the live ledger
is R3's and M5's instrument. The report says L3 was not run.

## 6. Metrics

From daily account values (close to close), per run:

- **The authors' three** (notebook, §2.2): Information Ratio (mean / std of
  excess daily returns, not annualised; the annualised form is times
  sqrt(252)), CVaR at 5% (mean of daily returns at or below their 5th
  percentile), Rachev ratio (95th / 5th percentile tails). Benchmark: `^NDX`
  at L1 and QQQ at L2, as in §5.2; returns from the equity series of the
  common sessions.
- **Ours:** annual return `(V_T / V_0)^(252/n) - 1` for n daily returns,
  cumulative return, annualised Sharpe (mean/std with ddof=1 x sqrt(252),
  risk-free 0), annualised volatility, maximum drawdown, turnover, share of
  sessions in cash, and the same CVaR at the 5% level on daily returns, for every
  arm and baseline.
- **Seed averaging.** An arm's estimate is the mean over seeds of the per-seed
  metric (not the metric of an averaged equity series); per-seed values are
  always published. At L1-B the per-seed value is first averaged over the 10
  sampled rollouts; at L1-A the mean is over the 20 rollouts of the one
  checkpoint.
- **Intervals and p-values:** paired stationary bootstrap (Politis and Romano
  1994), mean block `round(n^(1/3))` = 11 for the full window, 10,000 resamples,
  seed 0; one resample draws one set of dates and applies it to every seed, arm
  and baseline at once, recomputes each quantity, averages over seeds and
  takes the difference. The 95% interval is the 2.5th and 97.5th percentiles;
  the one-sided p-value is the share of resamples at or below zero. The date
  bootstrap captures market-sample noise, not seed noise: seed spread (min, max,
  sd) is always published. Sub-windows use their own block, `round(n^(1/3))`
  (9 for 757, 8 for 500).
- **Multiplicity:** Holm-adjusted p-values over the eight claims of a level are
  printed next to the verdicts; verdicts use the unadjusted p < 0.05 of the
  protocol and say when the adjusted value is above 0.05. Deflated Sharpe (L2
  Sharpe of every arm) is reported with N = 14 for us and "at least 15, not
  documented" for the authors (§3.6).

## 7. Verdict thresholds

Per claim, with $\hat{x}$ the seed-averaged estimate of $x$, $[l, u]$ its
bootstrap 95% interval and $c \ge 0$ the claimed size (§4). The rule is the one
of R3 (`reports/r3-tradingagents-live/preregistration.md` §6), stated here for
claims whose value is a direction (c = 0) or a table difference, applied in this
order:

1. **? Inconclusive** if $l \le 0 \le u$ and $l \le c \le u$: the data can
   neither exclude "no effect" nor exclude the paper's effect. The report then
   says what would decide it.
2. **✗ Does not hold** if $\hat{x} \le 0$, or $u < c/2$ (even the top of the
   interval is under half the claimed size). For c = 0 only $\hat{x} \le 0$
   applies, which after rule 1 means $u < 0$.
3. **✅ Holds** if $l > 0$ and (c = 0 or $l \le c \le u$). Then the one-sided
   bootstrap p-value is also below 0.05 and is reported.
4. **◐ Partly holds** in every remaining case, all of which have
   $\hat{x} > 0$ and c > 0 (for a direction claim rule 4 never applies): the
   direction matches, but significance is lost or the claimed size lies outside
   the interval; the report says whether ours is above or below.

- **Claim 1 and 2 qualifier.** Reported beside the verdict, not part of it: the
  same gap against the equal-weight baseline, and "universe-explained" when
  the equal-weight portfolio of the same names also beats the index and the agent
  does not beat it with p < 0.05.
- **Contamination.** A claim that uses LLM scores (all but `ppo-beats-nasdaq100`)
  is labelled **possibly contaminated** at both levels (§9) and a result that
  meets rule 3 is reported as ◐ with the label, as the protocol says no
  contaminated test earns ✅. ✗ and ? are unaffected. `llm-always-worsens`
  and `llm-lowers-ppo-ir` are claims that the LLM does *not* help; leaked
  knowledge would work against them, which the report says, but the label stays.
- **Cross-claim consistency.** `llm-always-worsens` (Fig. 2 agents) and
  `llm-raises-cppo-ir` (Fig. 3 agents) concern different agents at L1 and can
  both hold there. At L2 both are tested on the same arms (C10, C10-L against
  C0), so they cannot both be ✅: one says every LLM variant is worse, the other
  that CPPO's is better. The report states which one the data side with.
- **Table values.** CVaR and Rachev are reported with intervals but have no rule.
  The paper's three tables are reprinted with ours.

## 8. Smallest effect this test can detect

Window 2019-01-02 to 2023-12-28: 1,257 sessions, 4.99 years. Estimates before any
run; the report replaces them with the bootstrap on the realised series.

- **A single information ratio.** For a ratio near zero the standard error of
  the daily IR is about $1/\sqrt{n}$ = 0.0282, 0.45 annualised (the Lo 2002
  rule $\sqrt{1/\text{years}}$ gives 0.448). A bootstrap on the stand-in of §2.4
  (equal weight against QQQ, block 11, 2,000 resamples, seed 0) gives a daily
  standard deviation of 0.0285. **Every information ratio in the paper's
  Tables 1-3 (|IR| <= 0.0252 daily) is within one standard error of zero.**
- **Difference of two agents' ratios.** The standard error is about
  $\sqrt{2(1-\rho)/n}$ for the correlation $\rho$ of the two excess-return
  series, which is not known (an assumption): daily 0.0126 at $\rho$ = 0.9,
  0.0218 at 0.7, 0.0282 at 0.5; annualised 0.20, 0.35, 0.45. The two-sided 5%
  minimum detectable gap at 80% power is 2.8 times that: annualised **0.56,
  0.97, 1.25**. The claimed gaps, 0.347 (`llm-lowers-ppo-ir`) and 0.359
  (`llm-raises-cppo-ir`), are 0.6 to 0.3 of those, so the power to see them
  with $l > 0$ is about 0.4 at $\rho$ = 0.9, 0.2 at 0.7 and 0.1 at 0.5 even if
  they were the true effects. The most likely verdict for these two claims is
  **? Inconclusive**, and that is what the report will say, not that the claims
  are false.
- **Return gap to the index.** The stand-in, whose tracking error is 7.5% a
  year, has a bootstrap standard deviation of 4.2 points of annual return for
  the gap; the standard deviation scales with the agent's tracking error (about
  11 points at 20%). With a tracking error of 20% the interval's lower end exceeds
  0 only for a gap above about 20 points a year. `ppo-beats-nasdaq100` and
  `cppo-deepseek-beats-nasdaq100` are expected to land on ? unless an agent
  tracks the index closely or beats it by a very large margin.
- **Regimes.** The 757-session and 500-session sub-windows have standard
  errors 1.3 and 1.6 times the full window's; `ppo-wins-bull-market` and
  `cppo-deepseek-wins-bear-market` each rest on a few months of drawdown
  and recovery (the 2022 fall and the 2023 rally are both in the "bear" window)
  and are not independent of each other.
- **Seeds shrink training noise, not market noise.** Five seeds make the
  seed-mean of the estimate steadier; the interval's width is set by the
  market sample and does not fall with more seeds.
- We say all this now so that nobody reads an inconclusive result as a finding.

## 9. Contamination

The LLM scores of both levels were produced by models trained after the news
dates. The paper's trade window is 2019-2023 and its train window 2013-2018.

| Model | Used for | Release | Published training cutoff | Source |
| --- | --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-V3` via DeepInfra | the `deepseek` score files (§2.3) | tech report 2024-12-27, HF 2024-12-25 | **none published** in the model card or the technical report (searched for "cutoff", "knowledge", "as of"); a third-party list gives 2024-07-31, **[UNVERIFIED]** | arXiv 2412.19437; HF model card |
| Llama-3.3-70B-Instruct | the `llama` score files (**[INFERENCE]**) | 2024-12-06 | pretraining data "December 2023" | Meta model card |
| Qwen2.5-72B-Instruct | Fig. 1 only, no released data | 2024-09-16 (HF), report 2024-12-19 | none published (searched the model card and the report) | arXiv 2412.15115 |

The only cutoff that exists, Llama 3.3's "December 2023", is after the whole
2019-2023 window; DeepSeek-V3's and Qwen's are not published; and every model
was released after 2023-12-28, with the scores generated about 13 months after
it (the HF uploads of 2025-01-28/30). So **the cutoffs cannot clear the
LLM-feature results** and they stay labelled "possibly contaminated"
(protocol: a historical LLM test counts only if the documented training cutoff
is before the test window). The training years 2013-2018 are equally
affected. What the recall probe can add is evidence of how much the models
recall.

**Recall probe** (the protocol's, as in R3's addendum), committed as
`reports/r5-finrl-deepseek/evidence/recall_probe.py` and run before any L2
arm that uses scores, for each model of the table that its provider still
serves (a model no longer served, or served as a different snapshot, is
reported as such). Without tools:

- **Price:** "What was the closing price in US dollars of <company> (<ticker>)
  on <date>? Answer with one number, or UNKNOWN." 10 tickers (the first ten of
  the 80 in a numpy `default_rng(0)` permutation of the sorted list) x 60 sessions
  (12 per year of 2019-2023, drawn by the same generator, seed 0) = 600
  questions.
- **Sign:** "Did <ticker> close higher or lower on <date> than on the previous
  session? Answer UP, DOWN or UNKNOWN." for the same 600 items.
- **Control:** the same 600 items on 60 sessions drawn from 2026-07-01 to 30
  days before the run, a period after any snapshot of these models that a
  provider serves is plausibly trained on (**[INFERENCE]**; the snapshot's
  release date is recorded and the window moved later if it is after
  2026-07-01), where no recall is possible.
- **Statistics:** coverage (share of answers that are not UNKNOWN), median
  absolute percentage error and the share within 2% on the answered prices;
  sign accuracy with a two-sided binomial test against 0.5; each for the test
  items and the control.
- **Reading rule (fixed now):** recall is "detected" if, on the test items, the
  share of prices within 2% is at least 20% and exceeds the control's by at
  least 10 points, or sign accuracy is at least 55% with p < 0.01 while the
  control's is within 3 points of 50%. Otherwise "no recall detected". Neither
  outcome changes the label; "detected" is reported next to every LLM-feature
  result.
- Cost: 1,800 short calls per model, under about 2 USD; the figure is an estimate.

## 10. Compute (estimates, not measurements)

| Item | Estimate | Basis |
| --- | --- | --- |
| L1-A, 15 agents x 20 rollouts + mean policy | minutes of CPU | 1,257 steps of an MLP per rollout; unmeasured |
| One authors' training, 2M steps, 8 processes | 4.6 to 8 hours (PPO, CPPO, PPO-DeepSeek and CPPO-DeepSeek 10%, from the committed logs), up to 19 hours for other variants; memory up to 128 GB | `output_*.log` "Time", paper §6; hardware **[UNVERIFIED]** |
| L1-B and L1-C, 35 trainings | about 300 machine-hours of an 8-core, 128 GB host (about 160 if every run took the 4.6-8 hours of the main logs; the 0.1% PPO-DeepSeek log took 19 hours) | sum over the committed logs at 5 seeds per agent; on several hosts in parallel, a few days of wall time |
| L2, 70 fits | to be measured at the first fit; the lab's SB3 PPO at default steps takes minutes on one CPU core | M3 reports per-seed fit times |

L1-B and L1-C need a rented 128 GB machine; **no price is quoted here**. The
owner decides by the go/no-go date (§14); without it L1 rests on L1-A. The
recall probe's API spend is in §9. No LLM is called for the scores: the
authors' released scores are used as they are, and no LLM is re-scored.

## 11. Deviations from the paper's setup (known now)

**L1 against the paper:**

1. The authors' code at its last commit and two forks at their heads, with
   libraries unpinned by the authors resolved on the build day; Python 3.10.
2. `^NDX` for the benchmark (`ndx` returns nothing today); metrics on all
   1,257 sessions (the paper's curves start in early 2020, §3.4), with the
   2020-03-02 window beside it.
3. Sampled rollouts repeated 20 times (released agents) or 10 times (retrained
   agents); training seeds 1-5 where the paper has two runs.
4. For retraining, the 10% variants' scripts are reconstructed from the
   committed ones (§5.1); the exact scripts of the released "10%" agents are
   **[UNVERIFIED]**.
5. The Qwen agents (Fig. 1) and the 25-epoch agents are not tested.

**L2 against the paper:** the lab's PPO and CVaR-PPO (Stable-Baselines3,
64 x 64 networks unless M3's defaults differ; the 11-feature reference
observation of `rl/bench/features.py`; softmax weights) instead of the authors'
spinning-up code, 512 x 512 networks and eight FinRL indicators; target weights
instead of integer share counts of at most 100; next-open fills against the
same-close fills; a validation year; no forced selling on VIX >= 70; 80 names
instead of 84; Yahoo prices from 2026 instead of the authors' download;
dividend-adjusted open and close (the authors' open is not); scores aligned
to the next session in the primary arms; neutral scores leave actions
unchanged; the Nasdaq-100 as QQQ adjusted close; no early stopping on the test
period; five seeds instead of two runs; the risk factor multiplies whatever M3's
CVaR term is computed on, not the authors' running estimate.

**Allowed without invalidating the plan** (still listed in the report under
"Deviations", with the reason and the date): a library version, a machine, the
build date of the Python environment; the number of rollouts if one fails; a
seed rerun after a crash; a dropped join row. **Not allowed after the first
result:** any change of a threshold, a window, a claim, a baseline or an arm's
definition; an arm added afterwards is reported as a deviation and counted in
`trials`.

## 12. Planned analyses

Primary: §6 and §7. Secondary, reported beside the verdicts and never used for
them:

- Per-seed and per-rollout tables for every arm (all metrics of §6), the
  paper's Tables 1-3 reprinted with ours, and the sampling noise of the released
  agents against the difference between Table 1 and the notebook output.
- **Dose response:** each metric against the infusion strength (0, 0.1%, 1%,
  10%) for PPO and CPPO with paired intervals.
- **Placebo:** P10-S and C10-S against P10 and C10 (the difference attributable
  to the actual scores rather than to the perturbation, the extra observation
  inputs and the training noise).
- **Alignment:** P10-D and C10-D against P10 and C10 (the effect of using
  same-day scores).
- **Episode length (L1-C)** against L1-B PPO-DeepSeek 10%.
- **Universe:** equal weight of the 80 and of the 84 against the Nasdaq-100.
- **Score coverage:** returns on days with and without a score for the same
  arm.
- **Risk:** CVaR, drawdown and turnover of CPPO against PPO, since the method's
  point is risk (Table 1's CPPO has a worse CVaR than PPO, -0.0439 against
  -0.0394, and its CPPO-DeepSeek a worse one than PPO-DeepSeek, -0.0437 against
  -0.0338).
- Deflated Sharpe with N = 14 (ours) and the authors' unknown N of at least 15.

## 13. Prerequisites and risks

**Code that does not exist yet and is committed after this file, each in its
own change, before the first run it serves; none changes a rule of §7:**

1. The `us-ndx80` task (§5.2) in `rl/bench/tasks.py`.
2. A converter from the score files to M2's `jsonl` format, with the two
   alignments and the placebo, and a hash of its output (M2).
3. The LLM-feature wrapper for `ppo` and `cvar-ppo` (§5.2), whose
   `trials`, `description` and parameters are declared in `submission.json`
   (M2, M3).
4. L1's driver around the authors' scripts and the rollout harness; the L2-a
   observation adapter.
5. The recall probe script (§9).

If M2's `jsonl` source or M3's CVaR-PPO is not on `main` by 2026-12-15, the L2 arms
that need it are **not run** and the report lists them; the claims that need an
unfinished arm are reported as not tested.

**Risks recorded now:**

- **L1 environment.** `mpi4py` with the forks' `gym ~= 0.15.3` and `gymnasium`,
  unpinned `torch`, and `datasets` may not resolve; the authors' scripts may
  not run on a current stack. That is what the 2026-11-24 go/no-go decides, and
  L1-A uses only a CPU, `torch` and the authors' environment code.
- **Data drift.** The HF files are pinned by revision; Yahoo history is
  re-downloaded at run time and its hash recorded, so a revision of
  adjusted prices is a reported deviation, not a replay failure.
- **Scores.** The two score files of a model may not join cleanly, and the
  authors' day-level timestamps hide the publication time (§2.3): the lagged
  alignment is primary for this reason.
- **Licence.** FNSPID is CC BY-NC 4.0; the derived HF files state no licence.
  Nothing from them is committed; the report publishes derived metrics,
  equity curves and hashes only.
- **Models.** DeepSeek-V3 or Llama-3.3 may no longer be served by a provider the
  lab can pay; the probe then runs on the nearest served snapshot, labelled.
- **Contention.** L1-B is the longest job; a machine failure restarts the
  affected training and is reported.

## 14. Schedule and authors

| Date | Step |
| --- | --- |
| 2026-11-06 | this file merged to `main` (deadline); nothing of the authors' or ours runs before |
| 2026-11-06 to 2026-11-20 | the prerequisites of §13, each in its own change |
| 2026-11-24 | L1 go/no-go: do the authors' scripts start; is a 128 GB host approved |
| **2026-12-15** | **L1 complete** (L1-A, and L1-B/L1-C if go); recall probe committed and run |
| 2027-01-15 | L2 complete (needs M2 and M3 on `main`) |
| 2027-01-19 | draft report, evidence and reproduction command emailed to the author (seven days before publication) |
| 2027-01-26 | publication, with `reply.md` verbatim or "no reply by 2027-01-26" |

Corresponding author: **Mostapha Benhenda**, mostaphabenhenda@gmail.com (paper,
p. 1). The README links a blog and a Discord server; the fallback channel is a
GitHub issue on `benstaf/FinRL_DeepSeek`, linking the draft. Whether the address
is current is **[UNVERIFIED]**. The reply is printed verbatim in `reply.md`.
Errors the author finds are fixed and logged under "Corrections" with the date
and credit, as the protocol says.
