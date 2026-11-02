# Changelog

Releases of the `rl-pricing` package on [PyPI](https://pypi.org/project/rl-pricing/).

From 1.0.0 the package follows [semantic versioning](https://semver.org/).
A major release is needed to break any of these:

- the policy interface: `rl.bench.policy.Policy` (`act`, `fit`, `reset`,
  `save`, `load`, `stochastic`) and the `rl.bench.market.MarketView` and
  `MarketData` it receives;
- the `submission.json` fields and the `module:Class` / `policy.py:Class`
  policy references, including the built-in policies under
  `rl.bench.policies`;
- the JSON record formats under `leaderboard/` and `submissions/<id>/frozen/`,
  and the task definitions in `rl.bench.tasks` (a task's universe, windows and
  fees never change; a changed task gets a new id);
- the `rl-pricing` and `rl-pricing bench` commands and their options.

Other modules, including the share-trading toolkit's Python functions, may
change in minor releases.

## Unreleased

- Five offline and risk-sensitive RL entrants join the ranked tasks:
  behaviour cloning, IQL, CQL, a Decision Transformer
  (`rl.bench.policies.offline`, `iql`, `cql`, `decision_transformer`) and
  CVaR-PPO (`rl.bench.policies.cvar_ppo`, PPO from Stable-Baselines3 with the
  CPPO tail-risk penalty). The offline ones train on engine-logged
  trajectories of the built-in strategies from the training sessions only;
  every one keeps its best validation-Sharpe checkpoint and replays exactly
  from its saved weights.
- Historical records store `submission_sha256`, a hash of the submission's
  `name`, `kind`, `policy`, `params`, `trials` and shipped files.
  `rl-pricing bench run-all --stale` recomputes only the records whose
  submission changed, and `run-all` without `--task` covers every task with a
  live track. CI runs it after each merge, so contributors no longer commit
  historical records.
- `TradingAgentsPolicy` asks the provider's API (OpenAI or Anthropic) for
  both pinned models when it is constructed. A revoked key or a missing model
  now fails the ledger run with no entry, so a later run before the open can
  still record; before, it wrote an `error:` hold that no later run could
  replace.
- The ledger's paid-agent preflight requires the key of the submission's
  `llm_provider` (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`); the price table
  lists Claude Opus 5.5, Sonnet 5.5 and Sonnet 5.
- `claims.json` accepts `"verdict": null` for a pre-registered claim that has
  not been tested; `rl-pricing bench reports` validates it and leaves it out of
  the verdict tables until a verdict is set.
- `MarketView.info` gives a policy dated items (filings, news, precomputed
  scores) published strictly before the decision session's close; it is empty
  unless a task or submission declares `info_sources`: `sec-edgar` (8-K, 10-Q,
  10-K by EDGAR acceptance time; needs `SEC_EDGAR_USER_AGENT`) or `jsonl`.
  Records that read information store `info_sha256`.
- `model_cutoffs` and `contamination` submission fields. `kind: llm`
  submissions and any submission with model-produced item fields must declare
  `model_cutoffs`; a historical test window starting before a declared cutoff
  is refused unless the record is labelled `contamination: "possible"`, which
  the leaderboard shows.
- `rl-pricing bench check-causal --task <t> --submission <id>` runs the policy
  on truncated windows and on poisoned future prices and items and fails if an
  earlier decision changes. CI runs it on every non-LLM submission a pull
  request adds or changes.

## 1.0.0 (2026-10-27)

First stable release.

- `rl-pricing --version`, and `rl.__version__`.
- Every historical, live and ledger record also stores the `rl-pricing`
  version, so records written outside a git checkout stay traceable.
- Packaging metadata moved from `setup.py` to `pyproject.toml`; Python 3.14
  is supported and tested.
- The PyPI page shows how to score your own policy without cloning the
  repository.
- Since 0.5.0: the decision ledger checks a paid agent's keys and spending
  cap before the open and records a zero-cost canary (#98); reports gained
  the R1 input hashes, the R1 L1 probe and the R3 recall probe (#97, #99);
  `--seeds` rejects an empty set before anything is written (#101); the
  Yahoo cache stays exact on pandas 3.1 and the code raises no pandas 4
  deprecation warnings (#102).

## 0.5.0 (2026-10-23)

First release on PyPI, as `rl-pricing`: the `rl-pricing bench` benchmark
(causal engine, tasks, multi-seed runner, scoring, live track, decision
ledger, reports) and the share-trading toolkit.
