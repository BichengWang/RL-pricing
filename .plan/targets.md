# Targets

The owner delegated the choice ("best quality and best fit", 2026-10-14). This
is how targets are scored and which ones were chosen.

## Rubric

| Criterion | Weight | 5 means |
| --- | --- | --- |
| Claim quality | 30% | Peer-reviewed, concrete numbers, stated window and baseline, code available |
| Protocol fit | 25% | Our instruments can test it fairly with licensable data and affordable compute |
| Reach | 20% | Large existing audience (stars, citations) |
| Novelty | 15% | Nobody has published a multi-seed or out-of-sample re-test |
| Cost | 10% | Cheap in compute, API spend and maintainer time |

## Scored candidates (evidence gathered 2026-10-14)

| Candidate | Claim | Quality | Fit | Reach | Novelty | Cost | Score | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FinRL ensemble, Yang, Liu, Zhong & Walid, ICAIF 2020 | Ensemble of PPO/A2C/DDPG on Dow 30 (2016-01-04 → 2020-05-08): Sharpe 1.30, max drawdown −9.7% vs DJIA 0.47 / −37.1% | 4 | 5 | 4 | 4 | 5 | **4.35** | **R1** |
| TradingAgents, Xiao, Sun, Luo & Wang, arXiv 2412.20138 (live) | Same agent run prospectively for three months on AAPL, GOOGL, AMZN | 3 | 4 | 5 | 5 | 3 | **3.95** | **R3** |
| TradingAgents (backtest) | 2024-01-01 → 2024-03-29: cumulative return 23–27%, Sharpe 5.60–8.21 on AAPL, GOOGL, AMZN | 3 | 4 | 5 | 4 | 3 | **3.80** | **R2** |
| AlphaMix+, PRUDEX-Compass, TMLR 2023 | Risk-aware mixture-of-experts RL; best Sharpe (1.64) with lower seed variance | 4 | 3 | 3 | 4 | 4 | 3.55 | R4 candidate |
| RD-Agent(Q), Microsoft, NeurIPS 2025 | CSI 300 test 2017–2020: IC 0.0532, annual return 14.21%, IR 1.74 | 5 | 2 | 4 | 3 | 2 | 3.45 | R4 candidate |
| ai-hedge-fund (64k stars) | No formal performance claims; README says educational | 1 | 3 | 5 | 3 | 3 | 2.80 | Live ledger entrant only |
| FinRL-DeepSeek, Benhenda, arXiv 2502.07393 (contest 2025 task 1) | PPO / CPPO-DeepSeek beat Nasdaq-100 in two runs; LLM infusion hurts PPO and helps CPPO | 2 | 3 | 3 | 2 | 3 | 2.55 | **R5** (v2.0.0, decided 2026-11-01); the one prior re-test is a two-seed reimplementation, not the authors' code |

Novelty scores are our assessment from a web search on 2026-10-14, not an
exhaustive literature review. Each pre-registration re-checks for prior
re-tests.

## Why these

- **R1 FinRL ensemble.** The most-cited deep RL trading result, and it set the
  template many papers copy (Dow 30, turbulence liquidation, single run).
  Our toolkit already implements its ensemble mode. R0 showed the turbulence
  rule moves every agent to cash on the same days, so the ablation is ready.
  The Dow 30 as of 2016-01-01 includes names Yahoo no longer serves; licensed
  survivorship-free data closes that gap.
- **R2/R3 TradingAgents.** The largest audience in trading AI (111k stars) and a
  claim its authors themselves qualify. The test window predates every current
  model's training cutoff, so the historical result can't be trusted at face
  value. That splits it into a replication with a contamination probe (R2) and a
  three-month live run, the same length as the paper's window (R3). The live
  clock is the critical path: R3 starts by 2026-10-30.
- **R4 by gate.** AlphaMix+ fits our tasks and is cheap. RD-Agent(Q) is the more
  prestigious result, but needs China A-share data and a heavy LLM research
  loop. Decide by 2026-12-15 (P1.7) on data licence and a compute probe.
- **R5 FinRL-DeepSeek.** The 2025 LLM-plus-RL trading paper whose training code,
  agent-ready data and trained agents are all public (MIT code). Its claims are
  qualitative and rest on two stochastic runs, so the test is seeds, next-open
  fills and a recall probe for the LLMs that scored the news
  ([pre-registration](../reports/r5-finrl-deepseek/preregistration.md)).

## Sources

- FinRL ensemble: Yang et al., ICAIF 2020; arXiv repost 2511.12120; SSRN 3690996.
- TradingAgents: arXiv 2412.20138 v7 (2025-06-03), Table 1; QuantHarness, arXiv 2509.09995.
- ai-hedge-fund: README; issue #720 (recall measured 2026-09-24); PR #721.
- FinRL-DeepSeek: arXiv 2502.07393 v1; code `benstaf/FinRL_DeepSeek`; "FinRL-DeepSeek Revisited" (Cambridge Open Engage, 2026-09-15; two seeds, own reimplementation).
- AlphaMix+: PRUDEX-Compass, TMLR 2023, arXiv 2302.00586.
- RD-Agent(Q): NeurIPS 2025; critique in AutoScientist-Quant, arXiv 2608.28632.
- Star counts: GitHub API, 2026-10-14.

## Nominations

After launch, anyone can nominate a claim through an issue template
(paper, quoted claim, code link). Nominations are scored with the same rubric;
the top one becomes the next free monthly slot.
