# R2 pre-registration: TradingAgents replication and contamination test (Xiao, Sun, Luo, Wang, arXiv 2412.20138)

> **DRAFT, pending the owner's model decision (§10). Not yet committed as a
> pre-registration.** It becomes one when the owner has made the choices in
> §10, they are written into §5 and §10, and the file is merged to `main` in
> its own pull request before any R2 model call (`.plan/lab-protocol.md`,
> "Pre-registration"). Due **2026-12-01** (`.plan/phase-1-first-reports.md`, P1.2).

Drafted 2026-11-07, before any R2 model call. The report follows
`.plan/lab-protocol.md` (levels, pre-registration, verdict table, LLM
contamination rules, author notice). It has two parts:

- **L1**: the authors' released code on the paper's window, tickers and metric
  definitions, with the paper's models where they are still served. The code
  that produced Table 1 has never been published (§2.2), so L1 runs the
  released framework at the commit R3 pins. Every difference from the paper is
  listed in §11.
- **Contamination test**: a recall probe for every model used (§5.5), and a
  contrast between a model whose documented training cutoff is before 2024 and
  one whose cutoff is later (§5.6).

There are no L2 or L3 verdicts here. The live test of the same agent is R3
([`../r3-tradingagents-live/preregistration.md`](../r3-tradingagents-live/preregistration.md)
and its two addenda), which this file cites instead of re-deriving.

What was looked at before writing: the paper (all seven arXiv versions), the
authors' repository history through the GitHub API, the release code at v0.1.0
and v0.6.0, Yahoo prices for the window (no model was called), the token usage
in R3's committed ledger entries, and provider documentation. §3 lists what
the paper's own table and figures show; the scripts behind §3 and §8 are
[`prereg-observations/table1_checks.py`](prereg-observations/table1_checks.py)
and [`prereg-observations/mde.py`](prereg-observations/mde.py). The thresholds in §7 were
written without any R2 result. Items marked **[UNVERIFIED]** could not be
checked, with the reason; **[INFERENCE]** marks a conclusion drawn from
evidence rather than stated by a source. Rows marked † were read on 2026-11-07
by a separate search pass through a summarising web reader and were not
re-opened while drafting.

## 1. Question

The paper's Table 1 reports, for AAPL, GOOGL and AMZN over "January 1st to
March 29th, 2024", cumulative returns of 23.21% to 26.62%, Sharpe ratios of
5.60 to 8.21 and maximum drawdowns of 0.91% to 2.11%, against buy-and-hold
returns of -5.23% to 17.1%. We ask two things.

1. **L1.** Does the released code, run on that window with the paper's models
   where they are still served, reproduce these numbers?
2. **Contamination.** The window lies inside the training data of every
   current frontier model. Can the models recall the window's prices without
   tools, and does a model that may have read about 2024-Q1 do better on it
   than one whose documented cutoff predates it?

## 2. Facts about the paper, the code, the data and the models (read 2026-11-07)

### 2.1 Paper

| Item | Finding | Source |
| --- | --- | --- |
| Versions | v1 2024-12-28, v2 2025-01-09, v3 2025-01-10, v4 2025-02-23, v5 2025-03-02, v6 2025-04-15, **v7 2025-06-03 (latest, quoted here)**. Table 1's rows are identical in all seven (same SHA-256 of the extracted rows). SHA-256: v7 PDF `431d0c39365b4c46b43162371fa15b3dcf8d142b377d642b3e5925dc81f3487b`, v1 `ed4729448e5a288ce1ed7b86181448a8f5e1b271b742cf0e39b836bc382667c2` | <https://arxiv.org/abs/2412.20138>; <https://arxiv.org/pdf/2412.20138v1> … `v7` |
| Authors and contact | Yijia Xiao, Edward Sun, Wei Wang (UCLA, Tauric Research), Di Luo (UCLA, MIT). v1 p. 1 lists yijia.xiao@cs.ucla.edu, edwardsun12895@g.ucla.edu, diluo@mit.edu, weiwang@cs.ucla.edu; v7 p. 1 lists no addresses. The arXiv submitter is Yijia Xiao, who also maintains the repository (comment on issue #805, 2026) | v1 and v7 p. 1; arXiv abstract page |
| Window | v7 §5.1, p. 10: "a comprehensive backtesting simulation from January 1st to March 29th, 2024". v1 said instead: "We simulate the trading environment for the period from June 19, 2024, to November 19, 2024", while its data paragraph and every figure axis already showed January to March 2024; v2 changed the sentence to "January 1st, 2024, to March 29th, 2024". 2024-03-29 was Good Friday: the window has **61 sessions, 2024-01-02 to 2024-03-28** (Yahoo, `table1_checks.py`) | v1 and v2 "Simulation Setup"; v7 p. 10 |
| Assets | Text: "Apple, Nvidia, Microsoft, Meta, and Google"; Table 1 reports AAPL, GOOGL, AMZN only | v7 §5.1, p. 10; Table 1, p. 11 |
| Models | §4.3, p. 9: "Quick-thinking models, such as gpt-4o-mini and gpt-4o, efficiently handle fast, low-depth tasks like summarization, data retrieval, and converting tabular data to text ... deep-thinking models like o1-preview excel in reasoning-intensive tasks"; "Specifically, all analyst nodes rely on deep-thinking models to ensure robust analysis, while quick-thinking models handle data retrieval from APIs and tools for efficiency. Researchers and traders use deep-thinking models". The first public README: "We utilize `o1-preview` and `gpt-4o` as our deep thinking and fast thinking LLMs for our experiments" | v7 p. 9; `README.md` at v0.1.0 (`cc97cb6`) |
| Decisions | §5.1: "Agents make decisions based solely on data available up to each trading day, ensuring no future data is used (eliminating look-ahead bias). Based on their analysis, TradingAgents generates trading signals to buy, sell, or hold assets, which are then executed." Fill timing, position size, initial capital, transaction costs and slippage are **not stated** (also R3 §2) | v7 p. 10 |
| Short positions | Captions of Figs. 6, S1, S3: "Green / Red Arrows indicate Long / Short Positions respectively" | v7 pp. 11, 20, 21 |
| Data | §5.2: prices ("Open, high, low, close, volume, and adjusted close prices from January 1st, 2024, to March 29th, 2024"); news "from diverse sources such as Bloomberg, Yahoo, EODHD, FinnHub, and Reddit"; social posts "from Reddit, X/Twitter, and other platforms along with sentiment scores of posts calculated by auxiliary language models"; insider data "including transactions from SEDI and relevant company filings"; statements; company profiles "reported by third parties"; "Sixty standard technical analysis indicators". Whether any of it is point-in-time is not stated beyond the §5.1 sentence above | v7 p. 10 |
| Metrics | (S1) CR = (V_end − V_start)/V_start × 100%; (S2) AR = ((V_end/V_start)^(1/N) − 1) × 100%, "N is the number of years in the simulation"; (S3) SR = (R̄ − R_f)/σ, "R_f is the risk-free rate (e.g., yield of 3-month Treasury bills)", with no return frequency, annualisation or R_f value; (S4) MDD = max over t of (Peak_t − Trough_t)/Peak_t × 100% | v7 §S1.2, pp. 19-20 |
| Baselines | §S1.1: Buy and Hold, "Investing equal amounts in all selected stocks and holding them throughout the simulation period"; MACD, KDJ and RSI, ZMR, SMA, one sentence each, no parameters. No baseline code in any release (R3 §2; the v0.1.0 tree has none) | v7 p. 19 |
| Authors' caveat (v4 onward) | "We benchmarked TradingAgents over 3 months due to intensive LLM and tool use (11 LLM calls & 20+ tool calls/prediction). The highest Sharpe Ratio exceeds our expected empirical range (SR above 2 – very good, above 3 – excellent). We exported TradingAgents's decision sequences and examined them to ensure calculation correctness. We believe the exceptionally high SR resulted from the phenomenon that there were few pullbacks in TradingAgents during that period. We report results as they are in our experiments faithfully." | v7, footnote to §6.1.2, p. 12 |
| Sharpe text, v1 to v3 | "consistently outperforming all baseline models across AAPL, GOOGL, and AMZN with Sharpe Ratios of at least 5.60—surpassing the next best models by a significant margin of at least 2.07 points". Removed in v4, the version that added the footnote above | v1 p. 7 |
| Backtest harness | Figs. 6, S1, S3 show panels "Broker", "Trades - Net Profit/Loss" and "BuySell (True, 0.015)"; Figs. 7, S2, S4 name the strategies `BuyAndHoldStrategy`, `MACDStrategy`, `KDJRSIStrategy`, `ZMRStrategy`, `SMAStrategy`. **[INFERENCE]** These are backtrader plot panels and strategy classes; `backtrader` is in v0.1.0's `requirements.txt` | v7 pp. 11-12, 20-22; `requirements.txt` at `cc97cb6` |

### 2.2 The authors' code

| Item | Finding | Source |
| --- | --- | --- |
| Repository | `TauricResearch/TradingAgents`, Apache-2.0, 121,177 stars, created 2024-12-28T03:31Z (the day of v1) | GitHub API, 2026-11-07 |
| Code at the paper's versions | **None.** The commits from 2024-12-28 to 2025-02-02 hold only `README.md`, `index.html` and `index_complete.html` (tree at `635e91a`, 2025-02-02). The first code is **v0.1.0, `cc97cb6d5deb10eac370db0c6678e2796a62eba8`, 2025-06-05** ("initial public release"), two days after v7 | `GET /repos/TauricResearch/TradingAgents/commits?until=2025-06-10`; `git/trees/635e91a` |
| v0.1.0 | OpenAI only (`ChatOpenAI`); defaults `deep_think_llm` "o4-mini", `quick_think_llm` "gpt-4o-mini"; analysts, researchers, trader and risk debaters on the **quick** tier, Research Manager and Risk Manager on the deep tier, unlike §4.3; `data_dir` "/Users/yluo/Documents/Code/ScAI/FR1-data". Offline tools (FinnHub news and insider data, SimFin statements filtered by publish date, Reddit) read the authors' cached data: "The agents' offline tools rely on cached data from our **Tauric TradingDB**, a curated dataset we use for backtesting. We're currently in the process of refining this dataset, and we plan to release it soon". Online tools ask `gpt-4.1-mini` with `web_search_preview` for posts "from 7 days before {curr_date} to {curr_date}". A quick-tier call extracts "SELL, BUY, or HOLD". No backtest script, no baseline code, no per-day decisions | `default_config.py`, `graph/setup.py`, `graph/trading_graph.py`, `dataflows/interface.py`, `graph/signal_processing.py`, `README.md` at `cc97cb6` |
| Tauric TradingDB | Not mentioned in the v0.7.1 README; no release found. **[UNVERIFIED]** whether it was published elsewhere | `README.md` at `c454491` (search for "TradingDB") |
| Look-ahead fixes | Commit `e1113880` (2026-03-29, #475): "fix: prevent look-ahead bias in backtesting data fetchers", closing #203 ("Clear look ahead bias", 2025-08-17). Commit `0c1231a` (2026-06-13): "keep future/undated news out of historical windows". Historical runs of releases before v0.2.3 (2026-03-29) could therefore read data published after the decision date | GitHub API (commits, issue #203) |
| Backtest module | v0.5.0 (2026-09-18) added `tradingagents/backtest.py`: "this evaluates decision quality. It is not a portfolio simulator, and must not grow one" | `backtest.py` at `1394a3f` |
| Release used | **v0.6.0, `1394a3f72aa4393e1a98f51b382434c4b4c2d972`** (2026-10-03), R3's pin, so R3's adapter, measurements and API facts apply (R3 §2). Latest release: v0.7.1, `c4544913a14ed3df5f1ed7b9e8483270ce20682a`, 2026-10-27 | GitHub releases API |
| Authors on reproducibility | "Live data moves. News, StockTwits, and Reddit return different content as time passes ... the social and news sources still reflect 'now'." "Backtest results are not guaranteed to match any published figure." On issue #805 the maintainer calls model-level leakage "a separate, fundamental limitation a code change can't fully eliminate" | `README.md` at `c454491`; issue #805 |

### 2.3 What v0.6.0 serves for a decision dated in 2024

From `tradingagents/dataflows/` at `1394a3f`. **[INFERENCE]** from the code;
the dry run (§13) checks it.

| Input | Behaviour for a past date |
| --- | --- |
| Prices, indicators | yfinance with `auto_adjust=True`, so dividend-adjusted as of the run day (`yahoo/ohlcv.py`); windows clamped to the trade date (`as_of_window`) |
| Statements | SEC EDGAR as filed, filing date on or before the trade date, then Yahoo |
| Profile, valuation multiples | Withheld: "Profile fundamentals are withheld for this date" (`withhold_live_profile`) |
| Insider trades | Withheld when the vendor dates a trade by when it happened (`withhold_undisclosed_trades`) |
| News (yfinance), Reddit, StockTwits | These feeds "return their latest items whatever window is asked for"; a window they did not observe is reported as "`<... unavailable for <start>..<end>: it only serves recent items, so this is not an absence of ...>`" (`coverage_gap`). For 2024 dates the News and Social analysts therefore receive **no articles or posts** |
| News (Alpha Vantage, optional) | `NEWS_SENTIMENT` with `time_from` / `time_to` (`alpha_vantage/news.py`); needs `ALPHA_VANTAGE_API_KEY`. Free tier "25 API requests per day"; cheapest plan "75 requests/min + premium support: $49.99/month" (<https://www.alphavantage.co/premium/>). History depth and when its sentiment scores were computed: **[UNVERIFIED]**, the documentation page was too long to read through |
| Macro | FRED with the vintage pinned to the as-of date (`fred.py`: "pins the data vintage") |
| Prediction markets | Withheld for past dates (R3 §10, item 8) |

So under the release's own point-in-time rules a 2024-Q1 run is an agent of
prices, indicators, filings and macro data. The paper's news, social and
insider inputs are absent unless a historical news vendor is configured
(§10, D3).

### 2.4 Models: served today and documented cutoffs

| Model | Paper role | Served? | Documented training cutoff | Price per M tokens (in / cached / out) | Source |
| --- | --- | --- | --- | --- | --- |
| `o1-preview` | deep | **No**, shut down 2025-07-28 (announced 2025-04-28) | "Oct 01, 2023" † | – | <https://developers.openai.com/api/docs/deprecations> |
| `o1` (`o1-2024-12-17`), `o3-mini` | – | **No**, shut down 2026-10-23 | "Oct 01, 2023" | – | deprecations; <https://developers.openai.com/api/docs/models/o1>, `/o3-mini` |
| `gpt-4o` (`gpt-4o-2024-08-06`, `gpt-4o-2024-11-20`) | quick | **Yes, until 2027-01-14** (deprecation announced 2026-10-14); `gpt-4o-2024-05-13` shut down 2026-10-23 | "Oct 01, 2023" † | 2.50 / 1.25 / 10.00 † | deprecations; `/models/gpt-4o`; `/pricing` |
| `gpt-4o-mini-2024-07-18` | quick | **Yes, until 2027-01-14** | "Oct 01, 2023" † | 0.15 / 0.075 / 0.60 † | deprecations; `/models/gpt-4o-mini` |
| `gpt-4o` and `gpt-4o-mini` on Azure | – | Retire 2027-04-14; "Deprecated" means "No longer available to new customers"; only `gpt-4o` 2024-11-20 is open to new subscriptions † | as above | – | <https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-retirement-schedule> † |
| `gpt-6-sol`, `gpt-6-luna` | – | Yes | "Apr 20, 2026", "May 18, 2026" | 2 / 0.2 / 10; 0.1 / 0.01 / 0.5 | <https://developers.openai.com/api/docs/models/gpt-6-sol>, `/gpt-6-luna` |
| `claude-opus-5-5` | – (R3 deep) | Yes, retirement "Not sooner than September 22, 2027" | "Jun 2026" (reliable knowledge and training data) | 4 / 0.20 / 20 | <https://platform.claude.com/docs/en/about-claude/models/overview> |
| `claude-sonnet-5` | – (R3 quick) | Yes (legacy), "Not sooner than June 30, 2027" | "Jan 2026" (both) | 2 / 0.20 / 10 | <https://platform.claude.com/docs/en/models/sonnet-5/overview> |
| Oldest served Claude models | – | Haiku 4.5, Opus 4.5 (legacy) | Haiku 4.5 "Feb 2025" reliable, "Jul 2025" training; Opus 4.5 "May 2025", "Aug 2025" | – | `/models/haiku-4-5/overview`, `/models/opus-4-5/overview` |
| Claude 3 Haiku, Opus, Sonnet | – | **No**, retired 2026-04-20, 2026-01-05, 2025-07-21 † | "August 2023" (Claude 3 model card) † | – | <https://platform.claude.com/docs/en/about-claude/model-deprecations> † |
| Gemini 1.0 and 1.5 | – | **No**, retired 2025-04-21 to 2025-09-24 † | not checked | – | Vertex AI model versions page † |
| Llama 3.3 70B Instruct; Llama 3.1 8B/70B/405B | – | Weights downloadable (licence-gated); DeepInfra serves FP8 "Turbo" builds (3.3 70B at 0.10 / – / 0.32; 3.1 8B at 0.02 / – / 0.04); Bedrock lists them as Legacy ("New customers can't use Legacy models"), end of life 2027-03-30 † | "December 2023" (pretraining) † | as listed † | Meta model cards for Llama 3.1 and 3.3 on GitHub; DeepInfra pricing; Bedrock model lifecycle † |

Tool calling: `gpt-4o`, `gpt-4o-mini` and Llama 3.1/3.3 support function
calling †; Meta warns that "Llama 8B-Instruct can not reliably maintain a
conversation alongside tool calling definitions" †. Llama 3 (8B "March 2023",
70B "December 2023") and Llama 2 have no official tool-calling format †.

**Reading.** The only closed models open to a new account whose documented
cutoff predates the window are `gpt-4o` and `gpt-4o-mini`, and only until
**2027-01-14**. The paper's quick tier is therefore runnable; its deep tier
(o1-preview) is not, and no pre-2024 reasoning model is served by anyone. No
Anthropic or Google model with a pre-2024 cutoff is served. Llama 3.1/3.3
("December 2023", before the window by days) is the open-weights option.

## 3. What the paper's own table and figures show (observed before the first run)

Computed 2026-11-07 from the v7 PDF and Yahoo prices, with no model call
([`prereg-observations/table1_checks.py`](prereg-observations/table1_checks.py)).
They are disclosed because they shape the plan.

1. **The column labelled CR% behaves like a log return and the column labelled
   ARR% like the simple return.** In 11 of Table 1's 17 filled cells, CR equals
   100·ln(1 + ARR/100) to rounding, including the TradingAgents cells for AAPL
   (26.62 and 30.5) and GOOGL (24.36 and 27.58). The broker "value" printed at
   the last bar of Fig. 6 (AAPL) is 130,501.44 and of Fig. S3 (GOOGL)
   127,586.29: from 100,000 these are simple returns of 30.50% and 27.59% (the
   ARR column) and log returns of 26.62% and 24.36% (the CR column). The
   initial capital of 100,000 is **[INFERENCE]**; the paper does not state it.
   Neither column is annualised as eq. (S2) defines: over 61 sessions, (S2)
   turns a 26.62% return into about 165%. For AMZN, Fig. S1's value 124,872.71
   gives log 22.21% and simple 24.87%, against the table's 23.21 and 24.90, so
   the AMZN CR cell, which the text's "at least a 23.21% cumulative return"
   rests on, may be a typo for 22.21 (**[INFERENCE]**). Consequence: §6 uses
   eq. (S1) as defined, compares it with the printed values, and reports the
   log reading beside it (§7, rule note).
2. **Other inconsistencies in Table 1.** The GOOGL ZMR row has CR -0.58 but
   ARR +0.58, and its Sharpe and drawdown (2.12, 2.34) equal the SMA row's. The
   AMZN MACD row is empty. The GOOGL Sharpe "Improvement" of 4.26 does not
   equal 6.39 minus the best baseline's 2.31 (MACD), which is 4.08; it is
   close to 6.39 − 2.12 = 4.27.
3. **The buy-and-hold row does not reproduce.** Table 1: CR -5.23 / 7.78 /
   17.1, Sharpe -1.29 / 1.35 / 3.53, MDD 11.90 / 13.04 / 3.80 (AAPL / GOOGL /
   AMZN). Yahoo, dividend adjusted, to the 2024-03-28 close: from the
   2023-12-29 close, log returns -11.45 / 7.74 / 17.16; from the 2024-01-02
   open, simple returns -8.26 / 8.94 / 19.03; Sharpe (closes from 2023-12-29)
   -2.14 / 1.28 / 2.79; MDD 13.30 / 14.40 / 4.85. GOOGL and AMZN come within
   0.1 point of the table under the log return from the 2023-12-29 close; AAPL
   differs by 1.8 to 6.2 points (log return against the CR cell) under every
   convention tried. Two GitHub issues report the same AAPL gap (#137, comment
   of 2025-07-13: "approximately -7.5%"; #168: "Buy & Hold Return [%]
   -7.62767").
4. **The authors' prices were downloaded 7 to 8 months after the window.**
   Fig. S1's last AMZN bar (O 180.17, H 181.70, L 179.26, C 180.38) is Yahoo's
   raw 2024-03-28 bar exactly (AMZN pays no dividend). Fig. 6's last AAPL close,
   170.86, is the raw 171.48 adjusted for the dividends through the
   2024-11-08 ex-date and not the 2025-02-10 one; Fig. S3's GOOGL close, 150.56,
   is the raw 150.93 adjusted through the 2024-09-09 ex-date and not the
   2024-12-09 one. **[INFERENCE]** The price file was downloaded between
   2024-11-08 and 2024-12-08 (the only `o1-preview` snapshot is dated
   2024-09-12 †, and the appendix's demonstration date is 2024-11-19). If news and
   social data for 2024-Q1 were fetched at the same time, they could contain
   later information; the paper does not say whether its cached data
   ("Tauric TradingDB") was point-in-time.
5. **The AAPL run ends short.** Fig. 6's broker panel shows "cash 246516.57"
   above "value 130501.44" at the last bar. **[INFERENCE]** Cash above total
   value means short-sale proceeds are held, i.e. the AAPL position at the end
   is short. With the captions' "Long / Short", L1's primary mapping trades a
   Sell as a short (§5.2).
6. **Code and paper assign the tiers differently.** Every release, from v0.1.0
   to v0.6.0, puts the analysts on the quick tier; §4.3 says "all analyst
   nodes rely on deep-thinking models" (§2.2; R3 §2).

## 4. Claims, quoted from the paper

Table 1 (v7 p. 11; identical in v1-v6), TradingAgents and buy-and-hold rows
(CR, ARR, MDD in %):

| | AAPL CR / ARR / SR / MDD | GOOGL CR / ARR / SR / MDD | AMZN CR / ARR / SR / MDD |
| --- | --- | --- | --- |
| TradingAgents | 26.62 / 30.5 / 8.21 / 0.91 | 24.36 / 27.58 / 6.39 / 1.69 | 23.21 / 24.90 / 5.60 / 2.11 |
| B&H | -5.23 / -5.09 / -1.29 / 11.90 | 7.78 / 8.09 / 1.35 / 13.04 | 17.1 / 17.6 / 3.53 / 3.80 |

Text claims: §6.1.1, p. 11: "TradingAgents achieves at least a 23.21%
cumulative return and a 24.90% annual return on the three sampled stocks,
surpassing the best-performing baselines by a margin of 6.1%." §6.1.2, p. 12:
"TradingAgents outperforms market benchmarks such as Buy-and-Hold and
rule-based strategies consistently". §6.1.3, pp. 12-13: "TradingAgents
maintained a relatively low maximum drawdown compared to many baselines ...
ensured that the maximum drawdown remained within a manageable limit, not
exceeding 2." Abstract: "notable improvements in cumulative returns, Sharpe
ratio, and maximum drawdown".

Per-ticker claims have a level $c_m$ (the TradingAgents cell) and a gap $c_d$
to buy and hold (TradingAgents cell minus the B&H cell, oriented so larger is
better; for drawdown, B&H depth minus TradingAgents depth). Pooled claims are
the mean of the three per-ticker gaps, the form R3 uses (R3 §3, §5).

| Id | Quote (verbatim) | Source | Metric | $c_m$ | $c_d$ |
| --- | --- | --- | --- | --- | --- |
| `ta-cr-aapl` | "26.62" (CR%, AAPL) | Table 1, p. 11 | cumulative return, % | 26.62 | +31.85 |
| `ta-cr-googl` | "24.36" (CR%, GOOGL) | Table 1 | cumulative return, % | 24.36 | +16.58 |
| `ta-cr-amzn` | "23.21" (CR%, AMZN) | Table 1; §6.1.1 "at least a 23.21% cumulative return" | cumulative return, % | 23.21 | +6.11 |
| `ta-sharpe-aapl` | "8.21" (SR, AAPL) | Table 1 | annualised Sharpe | 8.21 | +9.50 |
| `ta-sharpe-googl` | "6.39" (SR, GOOGL) | Table 1 | annualised Sharpe | 6.39 | +5.04 |
| `ta-sharpe-amzn` | "5.60" (SR, AMZN) | Table 1 | annualised Sharpe | 5.60 | +2.07 |
| `ta-mdd-aapl` | "0.91" (MDD%, AAPL) | Table 1; §6.1.3 "not exceeding 2" | max drawdown depth, % | 0.91 | +10.99 |
| `ta-mdd-googl` | "1.69" (MDD%, GOOGL) | Table 1; §6.1.3 | max drawdown depth, % | 1.69 | +11.35 |
| `ta-mdd-amzn` | "2.11" (MDD%, AMZN) | Table 1; §6.1.3 | max drawdown depth, % | 2.11 | +1.69 |
| `ta-cr-vs-bh` | "surpassing the best-performing baselines by a margin of 6.1%" | §6.1.1, p. 11 | mean CR gap to B&H, points | – | +18.18 |
| `ta-sharpe-vs-bh` | "TradingAgents outperforms market benchmarks such as Buy-and-Hold ... consistently" | §6.1.2, p. 12 | mean Sharpe gap to B&H | – | +5.54 |
| `ta-mdd-vs-bh` | "maintained a relatively low maximum drawdown compared to many baselines" | §6.1.3, pp. 12-13 | mean drawdown gap to B&H, points | – | +8.01 |

The quote's "6.1%" is the smallest of the three margins over each ticker's
best baseline (AMZN, where the best baseline is buy and hold); against buy and
hold alone, Table 1's mean gap is +18.18 points, the value R3 also tests (R3
§3). Not given a verdict: the ARR column (not annualised as (S2) defines, §3.1;
our (S2) value is reported for reference); the "Improvement" row and the
claims against MACD, KDJ and RSI, ZMR and SMA, which have no parameters or code,
so any version we wrote would be our invention under the paper's name (as in
R3 §4). Only buy and hold is implemented. Ids get an arm suffix in
`claims.json` (for example `ta-sharpe-aapl-l1-p`).

## 5. Design

### 5.1 Arms

The owner chooses which arms run (§10). The design is the same for every arm.

| Arm | Provider | Quick tier (analysts, researchers, debaters, trader) | Deep tier (Research and Portfolio Managers) | Documented cutoffs | Label under the protocol |
| --- | --- | --- | --- | --- | --- |
| **P**, paper models where served | OpenAI | `gpt-4o-mini-2024-07-18` | `gpt-4o-2024-08-06` (o1-preview is not served) | Oct 2023, Oct 2023 | cutoff before the window |
| **P4**, optional variant | OpenAI | `gpt-4o-2024-08-06` | `gpt-4o-2024-08-06` | Oct 2023 | cutoff before the window |
| **N**, newer OpenAI | OpenAI | `gpt-6-luna` | `gpt-6-sol` | 2026-05-18, 2026-04-20 | possibly contaminated |
| **A**, R3's models | Anthropic | `claude-sonnet-5` | `claude-opus-5-5` | Jan 2026, Jun 2026 | possibly contaminated |
| **L**, open weights | OpenAI-compatible host or self-hosted | `Llama-3.3-70B-Instruct` | `Llama-3.3-70B-Instruct` | Dec 2023 | cutoff before the window |

- P keeps the paper's quick model and replaces o1-preview with the strongest
  served model of the same cutoff. P4 follows §4.3's statement that analysts
  use the deeper model, as far as that is possible with the release's tiers.
- Dated snapshot ids are pinned where the provider has them (the `gpt-4o`
  alias points to `gpt-4o-2024-08-06` today †); the provider-reported model
  name of every call is stored (`inputs.token_usage`, as in R3).
- Fixed for every arm: TradingAgents v0.6.0 at `1394a3f`, the four analysts,
  one debate round per team, `max_tool_rounds` default, the data vendors of
  §2.3 (or D3's choice, the same for all arms), memory off (a fresh working
  directory per decision, as in R3 §10 item 9), the provider's default
  temperature, and `medium` effort for reasoning models (R3's setting).
  **[INFERENCE]** Memory must be off in a replay: the release settles a
  decision against prices after it, and in 2026 every 2024 holding window has
  already traded, so a lesson from day t could reach day t+1 with returns from
  after t+1.
- Headline verdicts come from arm P when it runs, otherwise from arm A. Every
  arm that runs gets its own verdict column.

### 5.2 Decisions and execution

- **Agent.** `rl.bench.policies.tradingagents:TradingAgentsPolicy`, unchanged,
  with the arm's provider and models, `max_age_days: null` (the guard that
  stops R3's submission from being replayed) and `expected_commit` = `1394a3f…`.
- **Dates.** The engine's convention: "decision is at session `start - 1` and
  fills at `start`'s open" (`rl/bench/engine.py`, `simulate`). Each ticker gets
  **61 decisions, dated 2023-12-29 to 2024-03-27**, each a call of
  `propagate(ticker, date)`; v0.6.0 treats every such date as historical (§2.3).
  3 tickers × 61 = **183 decisions per run per arm**.
- **Fills.** At the next session's open (2024-01-02 to 2024-03-28), marked at
  each close. The account starts at 1.0 in cash at the 2024-01-02 open; 61
  daily returns.
- **Mapping, primary (L1): long/short.** Buy and Overweight → +1 (100% of
  equity long); Sell and Underweight → −1 (100% short); Hold and `REVIEW` keep
  the position; flat until the first Buy or Sell. Short-sale proceeds earn
  nothing and pay no borrow fee (the paper states neither). Chosen because the
  paper's captions say "Long / Short" and §3.5 shows a short at the end.
- **Mapping, secondary: long/flat**, R3's mapping (Sell and Underweight → cash).
- **Costs.** Primary 0 (the paper states none); 5 bp per side (the
  `us-aapl`/`us-googl`/`us-amzn` tasks' fee) and 10 bp as secondary.
- **Benchmark.** Each stock bought at the 2024-01-02 open and held to the
  2024-03-28 close, with the same fee.
- **Where it runs.** The engine is long-only ("Weights must be nonnegative
  (long-only)", `rl/bench/engine.py`). The long/flat 5 bp path runs on the engine
  through the three tasks; the long/short path is computed offline from each
  decision's logged rating and Yahoo opens and closes, like R3's "short shadow"
  (R3 §11). Before any metric is reported, the offline scorer must reproduce
  the engine's account values for the long/flat 5 bp path to 1e-9 (§13).

### 5.3 Runs

The agent cannot be seeded (R3 §4). Each arm is run **R** times, R chosen by
the owner from 1, 3 or 5 (§10, D2). A run is all 183 decisions; runs are
independent, none is selected or discarded, and all are published. With memory
off, decisions are independent, so they can run in any order and in parallel.
A decision that raises an error is retried at most twice and otherwise
recorded as `REVIEW` (keep the position) and listed.

### 5.4 Data

Scoring prices: Yahoo through the lab's loader, SHA-256 recorded, not
redistributed (protocol, "Data and licences"). The agent's inputs are fetched
by its own tools at run time (§2.3) and not archived. Each decision stores its
rating, the provider-reported model names and token usage, the Portfolio
Manager's text hash and first 400 characters (as R3), the number of
coverage-gap notices per analyst, and the full final state, which is kept
locally with its SHA-256 committed. Excerpts appear in the report only where an
analysis needs them (§12).

### 5.5 Recall probe (every model of every arm that runs)

Run before that arm's first full run and committed in an addendum. The script
extends R3's (`../r3-tradingagents-live/evidence/recall_probe.py`, its pinned
JSON prompt, no tools) to fixed session lists, as
`reports/r2-tradingagents-replication/evidence/recall_probe.py`.

- **Test items.** Unadjusted closes of AAPL, GOOGL and AMZN on the 61 sessions
  2024-01-02 to 2024-03-28, asked in chunks of at most 13 sessions; the five
  sessions before the window (2023-12-22 to 2023-12-29), as R3 asked them;
  each stock's return from the 2023-12-29 close to the 2024-03-28 close and in
  each of January, February and March 2024 (12 numbers); and the main events of
  each stock in 2024-Q1 (free text, reported, not scored).
- **Positive control.** The same questions for the first 61 sessions on or
  after 2023-04-03, before every arm's documented cutoff: it shows whether the
  probe detects recall where the model was trained.
- **Negative control.** The same questions for the first 61 sessions on or
  after 2026-07-01, after every arm's documented cutoff (the latest is Opus
  5.5's "Jun 2026"). R3 found no recall after the cutoff for Opus 5.5 and
  Sonnet 5 (R3 addendum 2026-10-29 §2: 0 of 15 prices given).
- **Scores.** Coverage, median absolute percentage error, share within 2%;
  sign accuracy of consecutive predicted closes with a two-sided binomial test
  against 0.5; sign and absolute error of the 12 return items.
- **Reading rule** (R5 §9's, fixed now): recall is **detected** if, on the test
  items, the share of closes within 2% is at least 20% and at least 10 points
  above the negative control's, or sign accuracy is at least 55% with p < 0.01
  while the negative control's is within 3 points of 50%. Otherwise **no recall
  detected**.
- **Consequence, fixed now.** The documented cutoff sets the label (protocol).
  In addition, an arm with a pre-2024 cutoff (P, P4, L) whose model shows
  recall detected on the test items is relabelled possibly contaminated: the
  evidence then contradicts the documented cutoff. "No recall detected" never
  removes a label.
- Cost: about 20 short calls per model, under $2 per model (estimate).

### 5.6 Contamination contrast (no verdict: not a claim of the paper)

Needs one arm with a pre-2024 cutoff (P or L) and one possibly contaminated
arm (N or A). With the primary mapping and fee:

$$\Delta_k = \bar{x}_k(\text{contaminated arm}) - \bar{x}_k(\text{pre-2024 arm})$$

for $k$ = pooled cumulative return and pooled Sharpe, from the paired stationary
bootstrap of §6 (the same dates for both arms, all tickers and all runs), with
a 95% interval and a one-sided p-value for $\Delta_k > 0$. The pairs are N − P
(option a, one provider), A − L or A − P (option c). Reading, fixed now:
$\Delta_k > 0$ with $p < 0.05$ is "consistent with contamination, and
confounded with the difference in model generation"; anything else is "no
contamination premium detected at this window's resolution" (§8 gives that
resolution). Neither reading is a finding about the paper's authors, whose
models had pre-window cutoffs.

**Capability control (optional, §10 D4).** The same pair run on the first 61
sessions on or after 2026-07-01, after every arm's documented cutoff, with the
same release and vendors. The difference in differences
$\text{DiD}_k = \Delta_k(\text{2024-Q1}) - \Delta_k(\text{2026-Q3})$ separates a
contamination premium from a capability gap; without it the two cannot be
told apart. Arm P can only take part before 2027-01-14.

### 5.7 Software record

This repository at the commit of the final record; TradingAgents at
`1394a3f` (the adapter refuses another commit); Python and the lock file of the
run environment; per-call model names from the providers.

## 6. Metrics

Per run, per ticker, from the daily account values (the 2024-01-02 opening
value, then the 61 closes):

- **Cumulative return** (S1), simple, primary. Log return $100\ln(V_T/V_0)$ is
  reported beside it (§3.1).
- **Annual return** (S2) with N = 61/252 years, reported only (§4).
- **Sharpe**: $\sqrt{252}\cdot\text{mean}(r)/\text{std}(r, \text{ddof}=1)$ of
  daily returns, risk-free 0, primary (R3 §5; the paper gives no value).
  Secondary: risk-free = FRED DTB3, averaged over the window.
- **Maximum drawdown** (S4) on the account values, as a depth in %.
- Also: turnover, share of sessions long, short and flat, rating counts,
  `REVIEW` count, coverage-gap notices per analyst, token usage and cost.
- **Pooled** = mean of the three per-ticker statistics (R3 §5).
- **Runs.** The estimate is the mean over runs of the per-run statistic.
- **Intervals and p-values.** Paired stationary bootstrap (Politis and Romano
  1994), mean block round(61^(1/3)) = 4, 10,000 resamples, seed 0. Each
  resample draws one set of dates and applies it to every run, ticker, arm and
  the benchmark, then recomputes each statistic. The 95% interval is the 2.5th
  and 97.5th percentiles; the one-sided p-value is the share of resamples with
  a gap at or below zero. The date bootstrap measures market-sample noise, not
  run-to-run noise; the run spread (min, max, sd) is published beside it.
- **Multiplicity.** Holm-adjusted p-values over the 12 claims of each arm are
  printed next to the verdicts; verdicts use the unadjusted p < 0.05 of the
  protocol and say when the adjusted value is above 0.05. The 12 claims share
  three return series and are not independent.

## 7. Verdict thresholds

For each claim and arm: the gap $d$ (agent minus B&H on the same dates and
basis, larger is better; for drawdown, B&H depth minus agent depth) with
estimate $\hat{d}$, interval $[l_d, u_d]$, one-sided p-value $p_d$ and claimed
value $c_d$ (§4); for per-ticker claims also the level $m$ (the agent's own
statistic; for drawdown, its depth with "smaller is better") with interval
$[l_m, u_m]$ and claimed value $c_m$. The rule is R3's (§6) and R5's (§7),
with the level check of R1 (§7) for per-ticker claims, and the protocol's ✅
row read literally: same direction, the claimed value inside our interval, and
the baseline beaten with p < 0.05. Apply in this order:

1. **? Inconclusive** if $l_d \le 0 \le u_d$ and $l_d \le c_d \le u_d$: the
   window cannot tell "no advantage over buy and hold" from the paper's
   advantage. The report says what would decide it.
2. **✗ Does not hold** if $\hat{d} \le 0$; or $u_d < c_d/2$ (even the top of
   the interval is under half the claimed gap); or, when R ≥ 3 and the claim
   is per ticker, no run reaches $c_m$ and the interval lies wholly on the
   wrong side of it ($u_m < c_m$ for return and Sharpe, $l_m > c_m$ for
   drawdown depth).
3. **✅ Holds** if $l_d > 0$ (so $p_d < 0.05$), and for per-ticker claims
   $c_m \in [l_m, u_m]$, and for pooled claims $c_d \in [l_d, u_d]$, and the
   arm is not labelled possibly contaminated.
4. **◐ Partly holds** in every remaining case, all of which have
   $\hat{d} > 0$: significance is lost, or the claimed value lies outside the
   interval (the report says "exceeds" or "below"), or rule 3 was met by a
   possibly contaminated arm, which the protocol does not allow to earn ✅
   (R5 §7 practice).

Notes fixed now:

- **CR reading.** The verdict on `ta-cr-*` uses eq. (S1) against the printed
  value. The report also states the verdict that the log reading of §3.1 would
  give (our log return against the printed value). If the two differ, both are
  shown and the one that rests on the paper's stated definition, (S1), is the
  verdict.
- **R = 1** reduces rule 2 to its first two parts, R3's single-path form.
- **Level.** Every verdict is L1 and is shown with the arm and mapping it rests
  on (primary: long/short, 0 fees). The long/flat and fee variants are
  reported beside the verdicts and never change them.
- Per-ticker claims can rarely be resolved by one ticker over 61 sessions
  (§8); we say so now so that a "?" is not read as a finding.

## 8. Smallest effect this test can detect

Window: 61 daily returns, 0.242 years. Estimates before any run, from the
realised 2024-Q1 returns
([`prereg-observations/mde.py`](prereg-observations/mde.py)); the report
replaces them with the bootstrap on the agent's own series.

- **Lo (2002), done properly.** Lo's formula
  $SE = \sqrt{(1 + SR^2/2)/T}$ holds per sampling period: with daily data the
  $SR^2/2$ term takes the daily Sharpe, so the annualised standard error is
  $\sqrt{(1 + SR_{ann}^2/504)/\text{years}}$. That gives **2.03 at SR 0, 2.09 at
  5.60 and 2.16 at 8.21**. Putting the annualised Sharpe into the $SR^2/2$
  term, as R3 §7 and the protocol's rule of thumb did (the protocol was
  corrected on 2026-11-07, #138), gives the same 2.0 at
  SR 0 but 8.30 at 5.60 and 11.97 at 8.21, which overstates the uncertainty of
  a high Sharpe from daily data four- to fivefold.
- **Bootstrap of buy and hold** (block 4, 4,000 resamples, seed 0): standard
  deviation of the Sharpe 2.04 / 2.19 / 1.43 (AAPL / GOOGL / AMZN), 1.54 for
  the three-stock mean; of the cumulative return 9.2 / 14.7 / 12.1 points, 9.3
  pooled.
- **No-skill agents** (150 per mapping, a rating redrawn with probability 0.2
  per session, 800 resamples each), bootstrap standard deviation per ticker and
  for the three-ticker mean:

| Mapping | Agent Sharpe | Sharpe gap to B&H | CR gap to B&H, points |
| --- | --- | --- | --- |
| long/short (primary) | 2.09 / 2.01 / 1.82; mean 1.14 | 2.80 / 2.81 / 2.26; mean 1.89 | 13.4 / 18.4 / 16.0; mean 11.3 |
| long/flat | 2.04 / 2.04 / 1.61; mean 1.34 | 1.60 / 1.61 / 1.31; mean 0.95 | 6.6 / 9.6 / 8.8; mean 5.9 |

- **Minimum detectable effect** (two-sided 5%, 80% power, 2.80 standard
  deviations):

| Quantity | long/short | long/flat | Paper's value |
| --- | --- | --- | --- |
| Per-ticker Sharpe level against 0 | 5.1 to 5.9 | 4.5 to 5.7 | 8.21 / 6.39 / 5.60 |
| Per-ticker Sharpe gap | 6.3 to 7.9 | 3.7 to 4.5 | 9.50 / 5.04 / 2.07 |
| Pooled Sharpe gap (`ta-sharpe-vs-bh`) | 5.3 | 2.7 | 5.54 |
| Per-ticker CR gap, points | 38 to 51 | 18 to 27 | 31.85 / 16.58 / 6.11 |
| Pooled CR gap (`ta-cr-vs-bh`), points | 32 | 17 | 18.18 |

- **Consequences.** If the paper's pooled Sharpe gap were the true effect, the
  long/short test would exclude zero with probability about 0.83
  (5.54/1.89 = 2.9 standard deviations); for the pooled CR gap the probability
  is about 0.36 (18.18/11.3 = 1.6). Per ticker, AAPL's Sharpe gap (9.50, 3.4
  standard deviations) is detectable, GOOGL's (5.04, 1.8) has power near 0.43
  and AMZN's (2.07, 0.9) about 0.15; of the CR gaps, AAPL's (31.85, 2.4) has
  power near 0.66, GOOGL's and AMZN's much less. Those claims will most likely
  be "?" unless the agent does far better or far worse than the paper says.
- **Drawdown.** Buy and hold fell 13.3%, 14.4% and 4.2% to 4.9% at its worst
  inside the window (§3.3). The claimed 0.91% to 2.11% needs an almost
  monotone account, so `ta-mdd-*` can fail or hold clearly.
- **Contrast between arms** (§5.6). For two agents trading the same stock with
  independent positions, the standard deviation of their pooled Sharpe
  difference is about √2 × 1.14 ≈ 1.6 (long/short), a minimum detectable
  contamination premium of about **4.5 Sharpe points** (**[INFERENCE]**:
  independence is assumed; agreeing ratings make it smaller). Only a large
  premium can be seen in one quarter. "Profit Mirage" (arXiv 2510.07920)
  reports a 55.68% Sharpe decay for TradingAgents on GPT-4o from a window
  before its cutoff to one after it † (§14).
- **Runs.** More runs (R) steady the run mean but do not narrow the interval,
  whose width comes from the market sample.

## 9. Contamination labels

| Arm | Documented cutoffs | Label before the probe | Can earn ✅ |
| --- | --- | --- | --- |
| P, P4 | `gpt-4o` and `gpt-4o-mini`: Oct 01, 2023 † | cutoff before the window | yes, unless §5.5 detects recall |
| L | Llama 3.3: December 2023 † | cutoff before the window, by days | yes, unless §5.5 detects recall |
| N | 2026-04-20, 2026-05-18 | possibly contaminated | no |
| A | Jun 2026, Jan 2026 | possibly contaminated | no |

Each arm's submission record declares `model_cutoffs` (model,
`training_cutoff`, `source_url`) and, for N and A, `contamination:
"possible"`; the runner refuses a window that starts before a declared cutoff
without the label (`rl/bench/info.py`, `historical_contamination`). Two leaks
that no label covers are checked and reported: the agent is told the ticker and
the date of every decision, so a model can recall what followed; and the
release's inputs could carry later information (adjusted prices, §2.3; anything
the future-mention audit of §12 finds).

## 10. Open decision for the owner: models and budget

**This draft does not choose.** Whatever is chosen, §5.1 is cut to the arms
that run, this section records the choice and its date, and the file then
becomes the pre-registration. The plan's budget note: "about $0.65 per
ticker-session on Claude, so about $120 for 61 sessions × 3 tickers per model"
(`.plan/phase-1-first-reports.md`, P1.2 status).

**Cost per arm and run** (183 decisions). Tokens per decision are R3's
measurement over its 18 ledger entries from 2026-10-28 to 2026-11-05
(`submissions/tradingagents/ledger/`): deep tier 14,028 input and 4,461 output
tokens (2 calls), quick tier 133,620 input and 25,360 output tokens (14.7
calls), no cached input; mean cost $0.666 a decision on Opus 5.5 and Sonnet 5
(min 0.624, max 0.745). Other arms apply those token counts to their prices,
which is an **estimate**: models differ in length of output, and reasoning
models (N) produce more. R3's own pre-run estimate for N was $0.17 a decision
(R3 §9), against $0.099 here.

| Arm | Prices used (per M tokens, in / out) | Per decision | R = 1 | R = 3 | R = 5 |
| --- | --- | ---: | ---: | ---: | ---: |
| P | deep `gpt-4o` 2.50 / 10; quick `gpt-4o-mini` 0.15 / 0.60 | $0.115 | $21 | $63 | $105 |
| P4 | `gpt-4o` on both tiers | $0.667 | $122 | $366 | $611 |
| N | `gpt-6-sol` 2 / 10; `gpt-6-luna` 0.10 / 0.50 | $0.099 (R3 §9: $0.17) | $18 ($31) | $54 ($93) | $90 ($156) |
| A | measured | $0.666 | $122 | $366 | $609 |
| L | Llama 3.3 70B, DeepInfra FP8, 0.10 / 0.32 † | $0.024 | $4.5 | $13 | $22 |

Add for every arm a dry run (3 decisions) and the recall probe (§5.5): under
$5 per arm (estimate). Wall time: R3's dry run took 648 s for three decisions
(R3 addendum 2026-10-29 §3), so a run is about 11 hours one decision at a time
and about 4 hours with the three tickers in parallel (estimate, Anthropic
models; rate limits **[UNVERIFIED]**).

**D1. Models.**

| Option | Arms | What it can conclude | Cannot | Cost at R = 1 / 3 / 5 |
| --- | --- | --- | --- | --- |
| **(a) OpenAI key, the paper's models where served** | P (+ P4 optional) and N | L1 with the paper's quick model and a deep-tier substitute of the same cutoff, eligible for ✅; contamination contrast N − P within one provider | o1-preview (not served); must finish before **2027-01-14** | P + N: $39 / $117 / $195 (with N at R3's estimate: $52 / $156 / $261); P4 adds $122 a run |
| **(b) Anthropic only** | A | L1 on R3's models, directly comparable with R3's live result | Every verdict is "possibly contaminated", so at most ◐; no pre-2024 contrast, so P1.2's "contrast with a model whose documented cutoff predates 2024" is not met and contamination rests on the recall probe alone | $122 / $366 / $609 |
| **(c) Anthropic plus a pre-2024 contrast** | A and L, or A and P | As (b), plus the contrast A − L (or A − P) | Separating contamination from capability needs D4 | A + L: $126 / $379 / $632; A + P: $143 / $429 / $715 |

Arm L needs a key for a host that serves Llama 3.3 70B with tool calling
(DeepInfra †) or a GPU host: a 70B model does not fit this laptop's 32 GiB at
a usable precision (**[INFERENCE]**; machine as in R1 §9). Bedrock's Llama
builds are closed to new customers †.

**D2. Runs per arm**, R = 1, 3 or 5 (§5.3). R1's L1 used 5 unseeded runs;
R = 1 is R3's single-path form.

**D3. News source.** (i) The release's defaults, keyless apart from FRED:
no news or social posts for 2024 dates (§2.3), the same for every arm. (ii)
Alpha Vantage historical news for every arm: closer to the paper's inputs,
needs a key and probably the $49.99/month plan (the free tier's 25 requests a
day cannot serve 183 decisions with several news calls each), and its history
depth and sentiment provenance are **[UNVERIFIED]** until checked.

**D4. Capability control** (§5.6): the contrast pair also on 2026-Q3. It adds
one run per arm of the pair (for example $39 for P + N or $126 for A + L at
R = 1).

Budget context: the owner approved $30-300 a month for LLM APIs and data
(R3 addendum 2026-10-29 §4), and R3 itself spends up to $60 a month through
January 2027.

## 11. Deviations from the paper's setup (known now)

1. **Code.** The released framework v0.6.0 (2026-10-03), not the unpublished
   pipeline behind Table 1; the first public code postdates v7 (§2.2). The
   release adds five-tier ratings, a Portfolio Manager, SEC EDGAR statements
   and point-in-time rules (R3 §10 item 2).
2. **Models.** o1-preview is not served; the deep tier is a substitute in every
   arm. The release's analysts use the quick tier, the paper's the deep tier
   (§3.6); P4 is the closest approximation.
3. **Data.** The release's vendors under its point-in-time rules, not the
   Tauric TradingDB (unreleased) or the paper's Bloomberg, EODHD, FinnHub, X and
   SEDI sources: no 2024 news, social or insider data unless D3 (ii) (§2.3).
   Prices are adjusted as of the run day, not as of late 2024 (§3.4).
4. **Execution.** Next-open fills, 100% long or short, flat until the first
   Buy or Sell, no borrow cost or interest. The paper states none of these;
   its harness is inferred to be backtrader (§2.1).
5. **Costs** 0 in the primary analysis, as the paper states none.
6. **Ratings.** Five tiers mapped to three positions (§5.2).
7. **Memory off** (our §5.1); the paper credits "a reflective agent" (its
   §6.2, p. 13).
8. **Runs.** R runs per arm; the paper reports one.
9. **Baselines.** Buy and hold only; the four rule baselines are not run (§4).
10. **Sharpe** with risk-free 0 and √252; the paper names 3-month T-bills
    without a value and states no annualisation.
11. **Tickers.** The three of Table 1, not the five named in §5.1.
12. **Dates.** 61 decisions dated 2023-12-29 to 2024-03-27, filled 2024-01-02
    to 2024-03-28; the paper's own decision dates are not stated.

## 12. Planned analyses

Primary: §6 and §7. Secondary, reported beside the verdicts and never used for
them:

- The long/flat mapping, and fees of 0, 5 and 10 bp per side, for every claim.
- The log reading of cumulative return (§3.1, §7 note) and Sharpe with
  risk-free = DTB3.
- Per-run, per-ticker tables of every statistic; run-to-run agreement of the
  ratings (Fleiss' kappa for R ≥ 3) and agreement between arms on the same
  decision.
- The contamination contrast $\Delta_k$ and, if run, $\text{DiD}_k$ (§5.6).
- Recall-probe tables per model (§5.5).
- **Future-mention audit.** Every decision's full text is scanned for calendar
  dates later than its decision date and for stated price levels of the stock
  that it first reached after that date; counts per arm, with examples. A hit
  is evidence of a leak from the model or a tool and is reported as such.
- Coverage-gap notices per analyst and arm: how often the agent ran without a
  data source.
- Cost and tokens per decision against §10.
- Table 1 reprinted beside ours, and R3's live per-ticker results beside the
  2024 ones (pattern only; different windows, no test).
- Deflated Sharpe with N = the number of arms × 2 mappings for us and
  "unknown" for the authors.

Anything added after the first run is listed in the report under
"Deviations", with the date and the reason.

## 13. Prerequisites and risks

**Code to commit before the first run**, each as a separate commit after this
file; none changes a verdict rule, and none touches
`rl/bench/policies/tradingagents.py`, so R3's recorded code identity does not
change:

1. A driver, `evidence/run_l1.py`: builds the adapter for an arm with the §5.1
   settings, makes the 61 decisions per ticker, writes one JSON per decision
   (§5.4) and keeps the full text locally.
2. A scorer, `evidence/score.py`: long/short and long/flat account values,
   fees, the metrics of §6, the bootstrap and the verdicts of §7; it must
   reproduce the engine's long/flat 5 bp account values to 1e-9 (§5.2).
3. The recall probe, `evidence/recall_probe.py` (§5.5).
4. One submission record per arm with `model_cutoffs` and, for N and A,
   `contamination: "possible"` (§9).

**Keys and variables** the owner provides for the chosen option:
`OPENAI_API_KEY` (P, P4, N), `ANTHROPIC_API_KEY` (A), a key for the Llama host
(L), `FRED_API_KEY` and `SEC_EDGAR_USER_AGENT` (every arm, as in R3 §12),
`ALPHA_VANTAGE_API_KEY` (D3 (ii) only).

**Start gate**, per arm, in an addendum committed before its first full run (the
R3 addendum pattern): a dry run of three decisions (one per ticker, dated
2023-12-29) that replaces §10's cost estimate with a measurement and checks
§2.3's behaviour; and the recall probe.

**Risks recorded now:**

- **`gpt-4o` and `gpt-4o-mini` shut down on 2027-01-14.** Arm P (and P4, and
  D4 with P) must finish before then. The fallback is Azure's `gpt-4o`
  2024-11-20 until 2027-04-14, a different snapshot and a reported deviation.
- **Compatibility [UNVERIFIED].** v0.6.0's model catalog lists neither
  `gpt-4o` nor Llama. Its validator accepts any model for OpenAI-compatible
  hosts and, **[INFERENCE]**, warns but proceeds for an unlisted OpenAI model;
  whether every analyst's tool calls work on `gpt-4o`, `gpt-4o-mini` and Llama
  3.3 70B is checked by the dry run. An arm that cannot complete a decision is
  reported as not run, with the reason.
- **The agent tested is not the paper's.** With default vendors the News and
  Social analysts have nothing to read (§2.3). A verdict then reads "the claim
  does not hold (or holds) under the released code with point-in-time data
  and no historical news", and the report says so in its first paragraph.
- **Model-level leakage** cannot be closed in this release: every prompt
  names the ticker and the date (§9).
- **Vendor limits.** SEC EDGAR fair-access limits and Yahoo throttling may slow
  runs; a failed tool call degrades the agent's inputs silently, which the
  coverage-gap counts partly show.
- **Cost overrun.** Each arm stops at 1.5 times its §10 estimate (or the dry
  run's measurement, once there is one); continuing needs an addendum.
- **One quarter, three large caps, one regime.** Most per-ticker claims cannot
  be resolved (§8).

## 14. Prior re-tests (searched 2026-11-07)

Not an exhaustive review. None reproduces Table 1. The only re-run of the
paper's own window found is issue #168, and the only contrast of windows before
and after a model's cutoff found is "Profit Mirage", on another universe and
other windows.

| Source | What was run | Finding | Read |
| --- | --- | --- | --- |
| TradingAgents issue #168 (fperdigon, 2025-07-24; closed 2026-05-31) | AAPL 2024-01-02 to 2024-03-28, default `o4-mini` / `gpt-4o-mini`, backtesting.py | "Return [%] -25.4382", "Sharpe Ratio -12.26803", "Buy & Hold Return [%] -7.62767", "Max. Drawdown [%] -26.24381", "# Trades 53". The maintainer closed it: returns "aren't guaranteed to match any published figure" | issue and comments via `gh` |
| Issue #137 (2025-07-12; closed 2026-09-18) | Request for the baseline and metric code; a comment of 2025-07-13 finds AAPL buy and hold "approximately -7.5%" against the paper's -5.23% | Closed "This is addressed in the current version", no code linked. Issue #119 asks the same † | `gh` |
| Issues #203 (2025-08-17, closed 2026-03-29) and #805 (2026-05-11, open); #33 † | Look-ahead in historical runs; model-level leakage | Data leaks fixed in `e1113880` (2026-03-29) and `0c1231a` (2026-06-13); model leakage left open (§2.2) | `gh` |
| Koviazin, Mudarisov, Polyachenko, State, "Reproducibility in the TradingAgents Framework", ACM, 2026-01-09, doi 10.1145/3800973.3801029 | GOOGL, May to July 2025, GPT-4o and Qwen3:30B | Qwen 18.1%, GPT-4o 15.8%, GOOGL buy and hold 19.1% †. Only the abstract was read (ACM returned HTTP 403): **[UNVERIFIED]** beyond it | abstract † |
| Li, Zeng, Xing, Xu, Xu, "Profit Mirage", arXiv 2510.07920, 2025-10-09 | NASDAQ-100, GPT-4o, Q2-Q3 2021 (before its cutoff) against Q3-Q4 2024 (after) | TradingAgents "still suffer[s] a 55.68% Sharpe decay" † | abstract read; body † |
| Ye, Han, Hu et al., "The Alpha Illusion", arXiv 2605.16895, 2026-05-16 | 2025-01 to 2026-01, five tickers, costs including tokens | TradingAgents "$106.4K gross / $102.3K net" against buy and hold $104.8K † | † |
| Qian, Peng, Wang et al., "When Agents Trade" (Agent Market Arena), arXiv 2510.11695 | Live, 2025-08-01 to 2025-09-30, TSLA, BMRN, BTC, ETH, five model backends | TSLA: buy and hold 46.88%, TradingAgents -38.72% to 21.91% † | † |
| Zhou, Yu, Huang et al., "TradingMoE", arXiv 2608.11785 | 33 US stocks, 2021-2023 test slice | TradingAgents +10.42% against Nasdaq buy and hold +11.27% † | † |
| Xiong, Zhang, Feng, Sun, You, arXiv 2509.09995 (v1 "QuantAgent", 2025-09-12; v4 "QuantHarness", 2026-07-27), cited in `.plan/targets.md` | 1-hour and 4-hour bars; a "TradingAgent" row in Table 1 of v4 † | Not comparable with a daily replication; window and model of that row not given † | abstract and versions read |
| FINSABER (arXiv 2505.07078), LiveTradeBench (2511.03628), Look-Ahead-Bench (2601.13770) and others | Cite TradingAgents without re-running it † | – | † |

The report re-checks for new re-tests before publication.

## 15. Schedule and authors

| Date | Step |
| --- | --- |
| 2026-11-07 | this draft, branch `r2-prereg-draft`, pending §10 |
| by 2026-11-24 | owner's choices for D1-D4 (a suggestion, leaving a week to merge) |
| **2026-12-01** | pre-registration merged to `main` in its own pull request (deadline, P1.2) |
| after the merge | §13's code, one commit each; per arm the dry run and recall probe, committed as an addendum |
| 2026-12-02 to 2026-12-20 | full runs; arm P (option a) before 2027-01-14 in any case |
| **2026-12-29** | draft report, evidence and reproduction command emailed to the authors |
| **2027-01-05** | publication (Phase 2), with `reply.md` verbatim or "no reply by 2027-01-05" |

Authors: Yijia Xiao (yijia.xiao@cs.ucla.edu, arXiv submitter and repository
maintainer), with Edward Sun (edwardsun12895@g.ucla.edu), Di Luo
(diluo@mit.edu) and Wei Wang (weiwang@cs.ucla.edu) in copy, at the addresses of
v1 p. 1, which date from 2024; a GitHub issue on `TauricResearch/TradingAgents`
is the fallback channel. Their reply is printed verbatim in `reply.md`. Errors
they find are fixed and logged under "Corrections" with the date and credit, as
the protocol says.
