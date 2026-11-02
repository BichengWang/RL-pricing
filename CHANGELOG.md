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

## 2.0.0 (2026-11-02)

Modern policies on stronger instruments ([`.plan/v2.md`](.plan/v2.md)):
offline and risk-aware RL entrants, a point-in-time information layer with a
lookahead check, records tied to the code that produced them, push-time proof
for ledger decisions, a hard per-entrant cost cap, the R5 pipeline and a live
LLM portfolio entrant. Breaking changes are marked below.

- `us-ndx80` joins the tasks as a re-evaluation (not ranked): the 80
  Nasdaq-100 names of FinRL-DeepSeek that Yahoo still serves, 2019-2023, with
  the unavoidable survivorship bias stated in the task description.
- PPO and CVaR-PPO take an opt-in `llm_strength` parameter (default 0, no
  change): with per-asset `llm_sentiment` and `llm_risk` items visible
  strictly before the decision close, the scores join the observation, the
  sentiment scales matching trades, and CVaR-PPO scales its tail-return
  estimate by the holdings-weighted risk. `fit` reads the same point-in-time
  view. Information items may carry `source_sha256` for source text that is
  not retained, and `InfoView.items(..., start=)` lets a sequential policy
  read only newly visible items.
- Submissions may declare `tasks`, a list of task ids restricting the
  submission to them (omitted means every task), so report-specific entrants
  are not run or frozen on unrelated tasks.
- Fourteen R5 L2 arms (`submissions/r5-*`), a hash-checked converter from
  FinRL-DeepSeek's released news scores to M2 jsonl
  (`reports/r5-finrl-deepseek/scores.py`) and a reproduce command
  (`reports/r5-finrl-deepseek/reproduce.py`). No R5 result is committed.
- New ledger entrant `llm-portfolio`: `LLMPortfolioPolicy` in
  `rl.bench.policies.llm_portfolio` asks the pinned Claude Sonnet 5.5
  (`claude-sonnet-5-5`) for long-only target weights on the nine
  `us-sectors` ETFs, through a fixed prompt and strict tool schema. The model
  rejects non-default sampling parameters and forced tool use, so the request
  sends no temperature, uses `tool_choice: "auto"`, pins
  `thinking: "between_tools"` and bounds the rendered prompt to a 20,000-token
  character budget. Each entry records the model id, prompt and rendered-input
  SHA-256s, the full response, token usage and cost; the declared worst case is
  $0.25 per decision. The official `anthropic` SDK is the new `agents` extra,
  installed only when `LLM_PORTFOLIO_ENABLED` is `true`; the workflow excludes
  the entrant otherwise and refuses to start it without a committed recall
  probe.
- The ledger's LLM preflight requires the names in the policy class's
  `ledger_env` instead of always requiring `FRED_API_KEY` and
  `SEC_EDGAR_USER_AGENT`; `TradingAgentsPolicy` keeps both, a price-only policy
  needs only its provider key.
- `submission.json` may set `monthly_budget_usd`, a per-entrant monthly cap.
  The ledger enforces the smaller of it and `LEDGER_MONTHLY_BUDGET_USD` before
  calling the policy.
- Ledger policies receive `MarketView.info` when the task or submission
  declares information sources, filtered to items published strictly before
  the decision close. A policy that sets `last_cost_usd = None` reports that no
  provider call was made (a pre-call hold); the ledger then records zero cost
  instead of the declared reservation. `ledger record` gained
  `--exclude <id>` (repeatable) so the workflow can hold an entrant until its
  switch is on.
- Ledger recording and scoring are covered on a multi-asset task: a weights
  vector over the nine `us-sectors` ETFs is recorded as a schema 2 entry and
  scored through the same engine and commit verification as single-asset
  entries.
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
- **Breaking:** historical, live and ledger records, `freeze.json` and ledger
  entries are written as schema 2: they add `protocol_version: "2"` and
  `code: {package_version, git_commit, code_sha256}`, the hash of the policy's
  source and of the `rl.bench` modules it imports. Readers (`leaderboard`,
  `monthly`, the live and ledger scorers) still load schema 1; consumers that
  accept only `schema: 1` must accept 2.
- `rl-pricing bench live` compares the current `code_sha256` with the frozen
  one. A frozen policy replayed on other code gets `code_changed: true` and
  both hashes in its record, and the leaderboard and monthly report mark the
  row. A schema 1 freeze has no hash and is recorded as
  `code_identity: "unrecorded"`.
- **Breaking:** `rl-pricing bench freeze --force` and the new `--stale` start a
  new cohort instead of overwriting: `freeze.json` gets `cohort: <n>`, the
  earlier cohorts move to `frozen/<task>/cohorts/<n>/`, and each cohort keeps
  its own publication (`publications.json` key `submission/task/<n>`, cohort 1
  keeps `submission/task`) and live record (`<id>.json` for cohort 1,
  `<id>.cohort-<n>.json` after). The leaderboard labels the cohorts of a
  submission that has several. An interrupted re-freeze is completed by the
  next `freeze` and never deletes an archived cohort. CI now runs
  `freeze --stale`, so changing a submission after its freeze no longer
  strands its live track.
- Ledger entries (schema 2) store `decision_sha256`, `policy_sha256` (the
  policy's own module only) and `code`. A ledger record keeps `code` for the
  code that scored it; the code behind the decisions is in `decision_code` and
  `code_hashes`, and the row is marked only when decisions span several
  `policy_sha256` values (`policy_versions`), not when the harness changed.
- Ledger scoring (schema 2 entries) requires that the Actions jobs API (the
  run's own attempt) shows the run's `Commit and push ledger entries` step
  succeeded before the open, and that the repository history shows the commit
  that added the entry file was authored by `github-actions[bot]` with the
  workflow's message, added exactly the entry's bytes and descends from the
  run's `head_sha`; `verified_by` names the rule (`push-time`, or `run-start`
  for the schema 1 entries written under the old one). This ties an entry to a
  commit and the commit to a run, but git authorship is not a signature:
  someone with push rights can forge it, and the pushed commit SHA the push
  step writes to its step summary is not readable through the API, so it is
  not checked. Jobs that cannot be read (404) or a shallow clone make scoring
  fail without rewriting any record; the `live` job now fetches full history.
  Finished runs and their jobs are cached on disk.
- **Breaking:** the ledger's monthly cap is hard. A ledger participant that
  declares `max_cost_per_decision_usd` (a param, or a class attribute: 0.70 for
  `TradingAgentsPolicy`, from the largest decision of the R3 dry run) has it
  validated, and an LLM participant must declare it; if the month's spend plus
  that reservation would exceed `LEDGER_MONTHLY_BUDGET_USD`, checked just before
  the policy runs, nothing is written and the command fails, where it used to
  write a `budget cap` hold. A decision that reports no usage is billed the
  reservation.
- `submission.json` accepts `retired: "YYYY-MM-DD"`. After that date `run-all`,
  `freeze`, `live`, `ledger record` and `ledger score` skip the submission, and
  the leaderboard lists its records under "Retired" instead of ranking them;
  the monthly report leaves it out.
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
