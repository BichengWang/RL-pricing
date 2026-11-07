---
name: Submit a policy
about: Enter a trading policy on the leaderboard and the live track
title: "Policy: <name>"
labels: submission
assignees: ''
---

<!--
Policies enter by pull request: add `submissions/<id>/` and nothing else, as
README § "Submit a policy" describes. CI writes the records, checks for
lookahead and freezes the policy for the live track on merge. Use this issue
to announce a submission, ask for help before opening the pull request, or
propose an LLM entrant for the daily ledger. Please do not paste vendor price
data.
-->

## The policy

- Name and one-line description:
- Kind: <!-- baseline / rl / forecast / llm / other -->
- Pull request or branch (if ready):
- Paper or write-up it follows (link, optional):

## Declarations

<!-- These go into submission.json. They are what CI cannot check for you. -->

- `trials`, every configuration you tried, abandoned ones included:
- Data the policy reads besides the task's prices (for example dated filings or news through `view.info`):
- Tasks it belongs to (all, or a list):

## Checks run locally

- [ ] `rl-pricing bench validate`
- [ ] `rl-pricing bench check-causal --task synthetic --submission <id>`
- [ ] `rl-pricing bench run --task <task> --submission <id>`

## LLM entrants only

<!--
Historical results count only for models whose documented training cutoff is
before the test window (`model_cutoffs`); otherwise the record is labelled
"possibly contaminated". Live LLM entrants run in the daily ledger under a
monthly cost cap.
-->

- Provider, pinned model ids and their documented cutoffs:
- Estimated cost per decision and per month:
- Who pays for the API calls:

## Your relation to the work

- [ ] I wrote the policy
- [ ] It reimplements someone else's published method (link above)
- [ ] Other:
