"""Decision ledger: recording, idempotency, budget, verification and scoring (offline)."""

import dataclasses
import datetime as dt
import io
import json
import os
import subprocess
import urllib.error
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from rl.bench import cli, ledger, leaderboard, monthly, runner
from rl.bench.data import DATA_ENV, load_task_data
from rl.bench.engine import simulate
from rl.bench.market import FIELDS
from rl.bench.policies import tradingagents as ta
from rl.bench.policy import Policy
from rl.bench.submission import Submission
from rl.bench.tasks import get_task

TASK = get_task("synthetic")
BASE = 1000
REPO = "owner/repo"
NOW = dt.datetime(2019, 6, 1, 12, 0, tzinfo=dt.timezone.utc)
CALLS = []


class Dependent(Policy):
    """Half the current weights plus half equal weight: needs the replayed account state."""

    def fit(self, history, validation_start, seed):
        raise AssertionError("ledger policies are never fitted")

    def act(self, view):
        CALLS.append(str(view.date))
        return 0.5 * view.weights + 0.5 / len(view.tickers)


class Costly(Policy):
    last_cost_usd = 0.0

    def act(self, view):
        CALLS.append(str(view.date))
        self.last_cost_usd = 3.0
        self.ledger_inputs = {"model": "m-1"}
        return np.full(len(view.tickers), 0.1)


class Boom(Policy):
    def act(self, view):
        CALLS.append(str(view.date))
        self.last_cost_usd = 0.5
        raise RuntimeError("no network")


class Oversized(Policy):
    def act(self, view):
        return np.full(len(view.tickers), 0.5)


class Idle(Policy):
    def act(self, view):
        return None


RACE = {}


class Racing(Policy):
    """Another run records the same session while this one is deciding."""

    def act(self, view):
        RACE["path"].write_text('{"first": true}\n')
        return np.full(len(view.tickers), 0.1)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    CALLS.clear()
    for name in (
        "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_REPOSITORY", "GITHUB_WORKFLOW_REF",
        "GITHUB_TOKEN", ledger.BUDGET_ENV, "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FRED_API_KEY",
        "SEC_EDGAR_USER_AGENT",
    ):
        monkeypatch.delenv(name, raising=False)


HEAD = {}


def git(root, *args, author=("tester", "initial commit")):
    env = {**os.environ, "GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": "a@example.com",
           "GIT_COMMITTER_NAME": author[0], "GIT_COMMITTER_EMAIL": "a@example.com"}
    done = subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True, env=env
    )
    return done.stdout.strip()


def push_entries(root, author=(ledger.COMMIT_AUTHOR, ledger.COMMIT_SUBJECT)):
    """Commit whatever changed under ``root`` the way the ledger workflow does."""
    git(root, "add", "-A")
    if git(root, "status", "--porcelain"):
        git(root, "commit", "-q", "-m", author[1], author=author)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture(autouse=True)
def repo(tmp_path):
    """A repository at ``tmp_path`` whose first commit is the head every fake run started from."""
    git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("base\n")
    HEAD["sha"] = push_entries(tmp_path, author=("tester", "initial commit"))
    return tmp_path


@pytest.fixture(scope="module")
def data():
    return load_task_data(TASK, end="2020-01-01")


def make_submission(root, sub_id="alpha", policy="test_bench_ledger:Dependent", ledger_tasks=("synthetic",), **extra):
    directory = root / sub_id
    directory.mkdir(parents=True)
    spec = {"name": sub_id.title(), "kind": "other", "policy": policy, "ledger": list(ledger_tasks), **extra}
    (directory / "submission.json").write_text(json.dumps(spec))
    return Submission.load(directory)


@pytest.fixture
def sub(tmp_path):
    return make_submission(tmp_path / "submissions")


def identity(k, repository=REPO):
    return {
        "repository": repository,
        "run_id": str(9000 + k),
        "run_attempt": "1",
        "workflow": "{}/.github/workflows/ledger.yml@refs/heads/main".format(REPO),
        "started_at": "2019-06-01T12:00:00+00:00",
    }


def stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def good_run(data, index, **changes):
    """The GitHub API's view of a ledger run that started an hour before the fill's open."""
    opens = ledger._open_utc(TASK, str(data.dates[index + 1]))
    run = {
        "path": ledger.WORKFLOW_PATH,
        "head_branch": "main",
        "event": "schedule",
        "head_sha": HEAD["sha"],
        "run_started_at": stamp(opens - dt.timedelta(hours=1)),
    }
    run.update(changes)
    return run


def good_jobs(data, index, completed_at=None, **step):
    """The jobs API's view of that run: its push step succeeded half an hour before the open."""
    opens = ledger._open_utc(TASK, str(data.dates[index + 1]))
    push = {"name": ledger.PUSH_STEP, "conclusion": "success", "completed_at": stamp(opens - dt.timedelta(minutes=30))}
    if completed_at is not None:
        push["completed_at"] = completed_at
    push.update(step)
    steps = [{"name": "Record decisions", "conclusion": "success", "completed_at": "2000-01-01T00:00:00Z"}, push]
    return {"jobs": [{"name": "record", "run_attempt": 1, "status": "completed", "steps": steps}]}


class FakeGitHub:
    def __init__(self, runs, jobs=None):
        self.runs = {str(k): v for k, v in runs.items()}
        self.jobs = {str(k): v for k, v in (jobs or {}).items()}
        self.paths = []

    def __call__(self, path):
        self.paths.append(path)
        prefix = "/repos/{}/actions/runs/".format(REPO)
        assert path.startswith(prefix), path
        run_id, _, rest = path[len(prefix):].partition("/")
        if rest:
            assert rest == "attempts/1/jobs?per_page=100", path
            return self.jobs.get(run_id)
        return self.runs.get(run_id)


def fake_github(data, runs):
    """``runs`` (run id -> run document), each with a well-timed push step."""
    return FakeGitHub(runs, {run_id: good_jobs(data, run_id - 9000) for run_id in runs})


def record(sub, data, indices, identities=None, task=TASK, **kwargs):
    """Record the decision after session ``i`` for each ``i``, seeing only data through ``i``."""
    paths = []
    for i in indices:
        run = (identities or {}).get(i, identity(i))
        paths.append(
            ledger.record_decision(task, sub, data=data.rows(0, i + 1), now=NOW, run=run, **kwargs)
        )
    return paths


def honest_github(data, indices):
    return fake_github(data, {9000 + i: good_run(data, i) for i in indices})


def from_entries(sub, data, first, stop, skip=()):
    """The engine trading the weights as recorded, except for the skipped sessions' entries.

    A policy decides on the account state it was shown at record time, so an entry later found
    unverifiable changes the state under which *later* entries were decided, not their weights.
    """
    skipped = {str(data.dates[i]) for i in skip}
    entries = ledger.load_entries(sub, "synthetic")
    targets = {
        date: None if entry["hold"] else np.array([entry["weights"][t] for t in TASK.tickers])
        for date, entry in entries.items()
        if date not in skipped
    }
    return simulate(
        data.rows(0, stop),
        lambda view: targets.get(str(view.date)),
        start=first + 1,
        transaction_cost=TASK.transaction_cost,
    )


def direct(data, first, stop, skip=()):
    policy = Dependent()
    skipped = {str(data.dates[i]) for i in skip}
    return simulate(
        data.rows(0, stop),
        lambda view: None if str(view.date) in skipped else policy.act(view),
        start=first + 1,
        transaction_cost=TASK.transaction_cost,
    )


def score(sub, data, stop, github, tmp_path, push=True):
    if push:
        push_entries(tmp_path)
    return ledger.score_ledger(
        TASK, sub, data=data.rows(0, stop), out_root=tmp_path / "board", github=github, repository=REPO
    )


def load(path):
    return json.loads(path.read_text())


def assert_same_equity(record, reference):
    assert record["runs"][0]["equity"] == runner._round(reference.equity)
    assert record["runs"][0]["turnover"] == runner._round(reference.turnover)
    assert record["dates"] == [str(d) for d in reference.dates]


# --- scoring equals the engine ------------------------------------------------


def test_five_sessions_score_exactly_like_the_engine(sub, data, tmp_path):
    indices = range(BASE, BASE + 5)
    paths = record(sub, data, indices)
    entry = load(paths[0])
    assert paths[0] == sub.directory / "ledger" / "synthetic" / "{}.json".format(data.dates[BASE])
    assert list(entry) == [
        "schema", "protocol_version", "code", "policy_sha256", "submission", "task", "decision_date", "weights",
        "hold", "reason",
        "cost_usd", "reserved_cost_usd", "inputs", "data_sha256", "run", "created_at", "decision_sha256",
    ]
    assert entry["decision_date"] == str(data.dates[BASE])
    assert entry["data_sha256"] == data.rows(0, BASE + 1).sha256()
    assert entry["protocol_version"] == "2" and entry["decision_sha256"] == ledger.decision_sha256(entry)
    assert entry["code"] == runner.code_identity(sub)
    assert entry["hold"] is False and list(entry["weights"]) == list(TASK.tickers)

    github = honest_github(data, indices)
    path = score(sub, data, BASE + 6, github, tmp_path)

    assert path == tmp_path / "board" / "ledger" / "synthetic" / "alpha.json"
    record_ = load(path)
    assert record_["track"] == "ledger" and record_["schema"] == 2 and record_["code"] == entry["code"]
    assert record_["fit"]["verified_by"] == {str(data.dates[i]): "push-time" for i in indices}
    assert record_["fit"]["first_decision"] == str(data.dates[BASE])
    assert {k: record_["fit"][k] for k in ("entries", "verified", "unverified", "missed_sessions")} == {
        "entries": 5, "verified": 5, "unverified": 0, "missed_sessions": 0,
    }
    assert_same_equity(record_, direct(data, BASE, BASE + 6))
    start = BASE + 1
    benchmark = runner._benchmark(
        TASK, data.rows(0, BASE + 6), data.rows(0, BASE + 1), start, np.datetime64("2016-01-01")
    )
    assert record_["benchmark"]["equity"] == runner._round(benchmark.equity)
    assert sorted(github.paths) == sorted(
        "/repos/{}/actions/runs/{}{}".format(REPO, 9000 + i, suffix)
        for i in indices
        for suffix in ("", "/attempts/1/jobs?per_page=100")
    )


def test_policy_sees_the_replayed_account_weights(sub, data):
    record(sub, data, range(BASE, BASE + 3))
    reference = direct(data, BASE, BASE + 3)
    last = load(sub.directory / "ledger" / "synthetic" / "{}.json".format(data.dates[BASE + 2]))
    expected = 0.5 * reference.weights[-1] + 0.5 / 5
    np.testing.assert_allclose([last["weights"][t] for t in TASK.tickers], expected, rtol=1e-12)


def test_decision_whose_fill_has_not_happened_is_pending(sub, data, tmp_path):
    indices = range(BASE, BASE + 5)
    record(sub, data, indices)
    path = score(sub, data, BASE + 5, honest_github(data, indices), tmp_path)
    fit = load(path)["fit"]
    assert (fit["entries"], fit["verified"], fit["pending"]) == (5, 4, 1)
    assert_same_equity(load(path), direct(data, BASE, BASE + 5))


def test_missed_session_is_a_hold_and_is_never_backfilled(sub, data, tmp_path):
    indices = [BASE, BASE + 1, BASE + 3, BASE + 4]
    record(sub, data, indices)
    path = score(sub, data, BASE + 6, honest_github(data, indices), tmp_path)
    record_ = load(path)
    assert record_["fit"]["missed_sessions"] == 1 and record_["fit"]["verified"] == 4
    assert_same_equity(record_, direct(data, BASE, BASE + 6, skip=[BASE + 2]))

    late = record(sub, data, [BASE + 2])
    assert late == [None]
    assert not (sub.directory / "ledger" / "synthetic" / "{}.json".format(data.dates[BASE + 2])).exists()


# --- recording ----------------------------------------------------------------


def test_recording_is_idempotent(sub, data):
    [first] = record(sub, data, [BASE])
    before = first.read_bytes()
    later = dt.datetime(2019, 6, 2, tzinfo=dt.timezone.utc)
    again = ledger.record_decision(TASK, sub, data=data.rows(0, BASE + 1), now=later, run=identity(77))
    assert again == first and first.read_bytes() == before
    assert CALLS == [str(data.dates[BASE])]


def test_run_identity_and_start_come_from_actions_environment(sub, data, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_RUN_ID", "4242")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", REPO + "/.github/workflows/ledger.yml@refs/heads/main")
    path = ledger.record_decision(TASK, sub, data=data.rows(0, BASE + 1), now=NOW)
    assert load(path)["run"] == {
        "repository": REPO,
        "run_id": "4242",
        "run_attempt": "2",
        "workflow": REPO + "/.github/workflows/ledger.yml@refs/heads/main",
        "started_at": "2019-06-01T12:00:00+00:00",
    }
    assert load(path)["created_at"] == "2019-06-01T12:00:00+00:00"


def test_a_reservation_that_does_not_fit_the_cap_writes_nothing(tmp_path, data):
    costly = make_submission(
        tmp_path / "s", "costly", "test_bench_ledger:Costly", params={"max_cost_per_decision_usd": 4}
    )
    [first] = record(costly, data, [BASE], budget=5)
    entry = load(first)
    assert entry["cost_usd"] == 3.0 and entry["reserved_cost_usd"] == 4.0 and entry["inputs"]["model"] == "m-1"

    with pytest.raises(ValueError, match="budget cap"):  # 3 spent + 4 reserved > 5, though 3 < 5
        record(costly, data, [BASE + 1], budget=5)
    assert len(CALLS) == 1  # the policy was never called
    assert list((costly.directory / "ledger" / "synthetic").glob("*.json")) == [first]

    next_month = dt.datetime(2019, 7, 1, 12, tzinfo=dt.timezone.utc)
    path = ledger.record_decision(
        TASK, costly, data=data.rows(0, BASE + 2), now=next_month, run=identity(2), budget=5
    )
    assert load(path)["hold"] is False and len(CALLS) == 2


def test_budget_comes_from_the_environment(tmp_path, data, monkeypatch):
    costly = make_submission(tmp_path / "s", "costly", "test_bench_ledger:Costly")
    monkeypatch.setenv(ledger.BUDGET_ENV, "3")
    record(costly, data, [BASE])
    with pytest.raises(ValueError, match="budget cap"):  # no declaration: spending the cap stops the entrant
        record(costly, data, [BASE + 1])
    assert len(CALLS) == 1
    monkeypatch.setenv(ledger.BUDGET_ENV, "")  # an unset repository variable: no cap
    [uncapped] = record(costly, data, [BASE + 2])
    assert load(uncapped)["hold"] is False
    monkeypatch.setenv(ledger.BUDGET_ENV, "lots")
    with pytest.raises(ValueError, match=ledger.BUDGET_ENV):
        record(costly, data, [BASE + 3])


def test_a_decision_without_reported_usage_is_billed_its_reservation(tmp_path, data):
    quiet = make_submission(
        tmp_path / "s", "quiet", "test_bench_ledger:Idle", params={"max_cost_per_decision_usd": 2.5}
    )
    [path] = record(quiet, data, [BASE], budget=10)
    assert load(path)["cost_usd"] == 2.5
    assert ledger.month_cost(quiet, "2019-06") == 2.5


@pytest.mark.parametrize("declared", [0, -1, "1", True, float("inf")])
def test_llm_participants_must_declare_a_positive_worst_case_cost(tmp_path, data, monkeypatch, declared):
    params = {} if declared is None else {"max_cost_per_decision_usd": declared}
    sub = dataclasses.replace(make_submission(tmp_path, params=params), kind="llm")
    for name in ("OPENAI_API_KEY", "FRED_API_KEY", "SEC_EDGAR_USER_AGENT"):
        monkeypatch.setenv(name, "test-only")
    with pytest.raises(ValueError, match="max_cost_per_decision_usd"):
        ledger.record_decision(TASK, sub, data=data, budget=40)
    assert not (sub.directory / "ledger").exists()


def test_llm_without_a_declaration_is_refused_and_the_shipped_agent_declares_one(tmp_path, data):
    sub = dataclasses.replace(make_submission(tmp_path / "a"), kind="llm")
    with pytest.raises(ValueError, match="must declare"):
        ledger.declared_max_cost(sub)
    assert ledger.declared_max_cost(copy_participant(tmp_path / "b", "tradingagents")) == 0.7
    assert ledger.declared_max_cost(make_submission(tmp_path / "c")) is None


def test_cap_is_checked_again_after_the_account_is_read(tmp_path, data, monkeypatch):
    costly = make_submission(tmp_path / "s", "costly", "test_bench_ledger:Costly")
    record(costly, data, [BASE], budget=10)  # 3 spent
    real = ledger._simulate

    def spent_meanwhile(*args):
        other = costly.directory / "ledger" / "synthetic" / "2030-01-01.json"
        other.write_text(json.dumps({"created_at": "2019-06-01T11:00:00+00:00", "cost_usd": 9.0}))
        return real(*args)

    monkeypatch.setattr(ledger, "_simulate", spent_meanwhile)
    with pytest.raises(ValueError, match="budget cap"):
        record(costly, data, [BASE + 1], budget=10)
    assert len(CALLS) == 1


def test_an_entry_recorded_meanwhile_is_not_overwritten(tmp_path, data):
    racing = make_submission(tmp_path / "s", "racing", "test_bench_ledger:Racing")
    RACE["path"] = racing.directory / "ledger" / "synthetic" / "{}.json".format(data.dates[BASE])
    RACE["path"].parent.mkdir(parents=True)
    [path] = record(racing, data, [BASE])
    assert path == RACE["path"] and path.read_text() == '{"first": true}\n'


def test_validate_checks_every_ledger_participants_cost_declaration(tmp_path, capsys):
    root = tmp_path / "subs"
    make_submission(
        root, "alpha", "rl.bench.policies.baselines:EqualWeightBuyHold", kind="baseline",
        params={"max_cost_per_decision_usd": -1},
    )
    assert cli.main(["validate", "--submissions-root", str(root)]) == 1
    assert "max_cost_per_decision_usd" in capsys.readouterr().err


def test_policy_exception_becomes_a_hold_with_the_reason(tmp_path, data):
    boom = make_submission(tmp_path / "s", "boom", "test_bench_ledger:Boom")
    [path] = record(boom, data, [BASE])
    entry = load(path)
    assert entry["hold"] is True and entry["weights"] is None
    assert entry["reason"] == "error: RuntimeError: no network"
    assert entry["cost_usd"] == 0.5


def test_invalid_weights_become_a_hold(tmp_path, data):
    oversized = make_submission(tmp_path / "s", "oversized", "test_bench_ledger:Oversized")
    entry = load(record(oversized, data, [BASE])[0])
    assert entry["hold"] is True and entry["reason"].startswith("error: PolicyError")


def test_policy_returning_none_is_a_hold_without_reason(tmp_path, data):
    idle = make_submission(tmp_path / "s", "idle", "test_bench_ledger:Idle")
    entry = load(record(idle, data, [BASE])[0])
    assert entry["hold"] is True and entry["weights"] is None and entry["reason"] is None


def copy_participant(root, sub_id):
    source = Path(__file__).resolve().parents[1] / "submissions" / sub_id / "submission.json"
    directory = root / sub_id
    directory.mkdir(parents=True)
    (directory / "submission.json").write_bytes(source.read_bytes())
    return Submission.load(directory)


@pytest.mark.parametrize("missing,cap,commit", [
    ("ANTHROPIC_API_KEY", "40", ta.PINNED_COMMIT),  # the provider's key, from submission.json
    ("FRED_API_KEY", "40", ta.PINNED_COMMIT),
    ("SEC_EDGAR_USER_AGENT", "40", ta.PINNED_COMMIT),
    (None, None, ta.PINNED_COMMIT),
    (None, " ", ta.PINNED_COMMIT),
    (None, "nan", ta.PINNED_COMMIT),
    (None, "-1", ta.PINNED_COMMIT),
    (None, "40", "0" * 40),
    (None, "40", None),
    ("ANTHROPIC_API_KEY", "0", ta.PINNED_COMMIT),  # preflight also precedes budget holds
    (None, "0", "0" * 40),
])
def test_llm_preflight_failure_writes_no_entry_but_canary_records(
    tmp_path, data, monkeypatch, capsys, missing, cap, commit,
):
    root = tmp_path / "submissions"
    # This participant sorts before the canary, proving failure doesn't stop later pairs.
    paid = copy_participant(root, "tradingagents")
    paid.directory.rename(root / "aaa-paid")
    canary = copy_participant(root, "ledger-canary")
    for name in ("ANTHROPIC_API_KEY", "FRED_API_KEY", "SEC_EDGAR_USER_AGENT"):
        if name != missing:
            monkeypatch.setenv(name, "test-only")
    if cap is not None:
        monkeypatch.setenv(ledger.BUDGET_ENV, cap)
    monkeypatch.setattr(ta, "installed_commit", lambda: commit)

    def single_asset(task, supplied, **kwargs):
        panel = data.rows(0, BASE + 1)
        return dataclasses.replace(panel, tickers=task.tickers,
                                   **{k: getattr(panel, k)[:, :1] for k in FIELDS})

    monkeypatch.setattr(runner, "_data_for", single_asset)

    assert cli.main(["ledger", "record", "--submissions-root", str(root)]) == 1
    output = capsys.readouterr()
    expected = missing or ("pins" if commit != ta.PINNED_COMMIT else ledger.BUDGET_ENV)
    assert expected in output.err
    assert not (root / "aaa-paid" / "ledger").exists()
    [path] = list((canary.directory / "ledger").glob("*/*.json"))
    entry = load(path)
    assert entry["task"] == "us-aapl" and entry["cost_usd"] == 0
    assert entry["hold"] is False and entry["weights"] == {"AAPL": 1.0}


def test_llm_graph_construction_failure_writes_nothing(tmp_path, data, monkeypatch):
    sub = dataclasses.replace(
        make_submission(tmp_path, params={"max_cost_per_decision_usd": 1}), kind="llm"
    )
    for name in ("OPENAI_API_KEY", "FRED_API_KEY", "SEC_EDGAR_USER_AGENT"):
        monkeypatch.setenv(name, "test-only")

    def broken(*args):
        raise RuntimeError("graph setup failed")

    monkeypatch.setattr(runner, "build_policy", broken)
    with pytest.raises(RuntimeError, match="graph setup failed"):
        ledger.record_decision(TASK, sub, data=data, budget=40)
    assert not (sub.directory / "ledger").exists()


def test_canary_five_sessions_match_direct_engine_and_render_as_baseline(tmp_path, data):
    sub = copy_participant(tmp_path / "submissions", "ledger-canary")
    task = get_task("us-aapl")
    assert sub.kind == "baseline" and sub.ledger == (task.id,) and task.tickers == ("AAPL",)
    single = dataclasses.replace(data, tickers=task.tickers, **{k: getattr(data, k)[:, :1] for k in FIELDS})
    indices = range(BASE, BASE + 5)
    paths = record(sub, single, indices, task=task, budget=0)
    assert all(load(p)["cost_usd"] == 0 for p in paths)
    assert load(paths[0])["weights"] == {"AAPL": 1.0}
    assert all(load(p)["hold"] for p in paths[1:])
    before = [p.read_bytes() for p in paths]
    assert record(sub, single, indices, task=task, budget=0) == paths
    assert [p.read_bytes() for p in paths] == before

    push_entries(tmp_path)
    board = tmp_path / "board"
    path = ledger.score_ledger(task, sub, data=single.rows(0, BASE + 6), out_root=board,
                               github=honest_github(single, indices), repository=REPO)
    record_ = load(path)
    policy = runner.build_policy(sub, task)
    reference = simulate(single.rows(0, BASE + 6), policy.act, start=BASE + 1,
                         transaction_cost=task.transaction_cost)
    assert_same_equity(record_, reference)
    assert record_["fit"]["verified"] == 5
    assert "Ledger canary (1/N)" in leaderboard.render_markdown(leaderboard.build(board))
    entry = monthly._entry(record_, "2020-01-01")
    text = monthly.render_markdown("2019-12", "2019-12-31", [entry], None, None)
    assert "| Ledger canary (1/N) (ledger) | baseline |" in text


# --- verification ----------------------------------------------------------------


def opens(data, index):
    return ledger._open_utc(TASK, str(data.dates[index + 1]))


def pushed_at(offset):
    """Jobs whose push step completed ``offset`` after the open."""
    return lambda d, i: good_jobs(d, i, completed_at=stamp(opens(d, i) + offset))


late = pushed_at(dt.timedelta(minutes=1))


def other_attempt(d, i):
    return {"jobs": [{**good_jobs(d, i)["jobs"][0], "run_attempt": 2}]}


# case: (changes to the entry's run identity, run document, jobs document, expected reason); None = as honest
BAD = {
    "pushed after the open, run started before": ({}, None, late, "at or after the open"),
    "pushed at the open": ({}, None, pushed_at(dt.timedelta()), "at or after the open"),
    "push step failed": ({}, None, lambda d, i: good_jobs(d, i, conclusion="failure"), "ended 'failure'"),
    "no push step": ({}, None, lambda d, i: {"jobs": [{"run_attempt": 1, "steps": []}]}, "has 0 steps"),
    "push step of another attempt": ({}, None, other_attempt, "has 0 steps"),
    "wrong workflow": ({}, lambda d, i: good_run(d, i, path=".github/workflows/tests.yml"), None, "is not"),
    "wrong branch": ({}, lambda d, i: good_run(d, i, head_branch="feature"), None, "is not on main"),
    "wrong trigger": ({}, lambda d, i: good_run(d, i, event="push"), None, "triggered by"),
    "missing run": ({}, lambda d, i: None, None, "not found"),
    "other repository": ({"repository": "evil/fork"}, None, None, "another repository"),
    "no run identity": (None, None, None, "no run identity"),
    "reused run": ({"run_id": str(9000 + BASE)}, None, None, "already backs"),
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_unverifiable_entry_counts_as_a_hold_and_is_listed(case, sub, data, tmp_path):
    changes, response, jobs, reason = BAD[case]
    indices = range(BASE, BASE + 4)
    bad = BASE + 1
    identities = {bad: None if changes is None else {**identity(bad), **changes}}
    record(sub, data, indices, identities)
    github = honest_github(data, indices)
    if response is not None:
        github.runs[str(9000 + bad)] = response(data, bad)
        if github.runs[str(9000 + bad)] is None:
            del github.runs[str(9000 + bad)]
    if jobs is not None:
        github.jobs[str(9000 + bad)] = jobs(data, bad)
    path = score(sub, data, BASE + 5, github, tmp_path)

    record_ = load(path)
    fit = record_["fit"]
    assert (fit["entries"], fit["verified"], fit["unverified"]) == (4, 3, 1)
    [listed] = fit["unverified_entries"]
    assert listed["decision_date"] == str(data.dates[bad]) and reason in listed["reason"]
    assert_same_equity(record_, from_entries(sub, data, BASE, BASE + 5, skip=[bad]))


def test_window_starts_after_the_first_verified_entry(sub, data, tmp_path):
    indices = range(BASE, BASE + 4)
    record(sub, data, indices)
    github = honest_github(data, indices)
    github.jobs[str(9000 + BASE)] = late(data, BASE)
    path = score(sub, data, BASE + 5, github, tmp_path)
    record_ = load(path)
    assert record_["fit"]["first_decision"] == str(data.dates[BASE + 1])
    assert record_["fit"]["unverified"] == 1 and record_["fit"]["missed_sessions"] == 0
    assert_same_equity(record_, from_entries(sub, data, BASE + 1, BASE + 5))


def test_nothing_verified_writes_no_record(sub, data, tmp_path):
    record(sub, data, [BASE, BASE + 1])
    assert score(sub, data, BASE + 4, FakeGitHub({}), tmp_path) is None
    assert not (tmp_path / "board").exists()


def test_run_path_with_ref_suffix_is_accepted(sub, data, tmp_path):
    indices = range(BASE, BASE + 2)
    record(sub, data, indices)
    github = fake_github(
        data, {9000 + i: good_run(data, i, path=ledger.WORKFLOW_PATH + "@refs/heads/main") for i in indices}
    )
    assert load(score(sub, data, BASE + 3, github, tmp_path))["fit"]["verified"] == 2



def rewrite(path, **changes):
    entry = load(path)
    entry.update(changes)
    path.write_text(json.dumps(entry, indent=2) + "\n")


def downgrade(path):
    """The entry as the schema 1 recorder wrote it."""
    entry = {k: v for k, v in load(path).items() if k not in ("protocol_version", "code", "reserved_cost_usd", "decision_sha256")}
    path.write_text(json.dumps({**entry, "schema": 1}, indent=2) + "\n")


def test_schema_1_entries_keep_the_run_start_rule(sub, data, tmp_path):
    indices = range(BASE, BASE + 4)
    for path in record(sub, data, indices):
        downgrade(path)
    github = FakeGitHub({9000 + i: good_run(data, i) for i in indices})  # no jobs API at all
    github.runs[str(9000 + BASE + 1)] = good_run(
        data, BASE + 1, run_started_at=stamp(opens(data, BASE + 1) + dt.timedelta(minutes=1))
    )
    fit_record = load(score(sub, data, BASE + 5, github, tmp_path))
    assert fit_record["fit"]["verified_by"] == {
        str(data.dates[i]): "run-start" for i in indices if i != BASE + 1
    }
    assert [e["decision_date"] for e in fit_record["fit"]["unverified_entries"]] == [str(data.dates[BASE + 1])]
    assert not any("jobs" in path for path in github.paths)
    assert fit_record["code_identity"] == "unrecorded" and "decision_code" not in fit_record
    assert fit_record["code"] == runner.code_identity(sub)  # every schema 2 record says what scored it


def test_an_edited_entry_no_longer_matches_its_digest(sub, data, tmp_path):
    indices = range(BASE, BASE + 3)
    paths = record(sub, data, indices)
    rewrite(paths[1], reason="edited after the push")
    fit = load(score(sub, data, BASE + 4, honest_github(data, indices), tmp_path))["fit"]
    assert fit["verified"] == 2
    [listed] = fit["unverified_entries"]
    assert listed["decision_date"] == str(data.dates[BASE + 1]) and "decision_sha256" in listed["reason"]


def test_harness_edits_are_audit_only(sub, data, tmp_path):
    indices = range(BASE, BASE + 4)
    paths = record(sub, data, indices)
    first = load(paths[0])
    assert first["policy_sha256"] == sub.policy_sha256()
    harness = load(paths[2])
    harness["code"]["code_sha256"] = "0" * 64  # the harness around the policy was edited
    rewrite(paths[2], code=harness["code"], decision_sha256=ledger.decision_sha256(harness))
    scored = load(score(sub, data, BASE + 5, honest_github(data, indices), tmp_path))
    assert scored["code"] == runner.code_identity(sub)
    assert scored["decision_code"] == first["code"]  # the newest decision's code
    assert scored["code_hashes"] == [first["code"]["code_sha256"], "0" * 64]
    assert "code_changed" not in scored and "policy_versions" not in scored
    assert "policy versions" not in leaderboard.render_markdown(leaderboard.build(tmp_path / "board"))


def test_decisions_spanning_policy_versions_are_marked(sub, data, tmp_path):
    indices = range(BASE, BASE + 4)
    paths = record(sub, data, indices)
    changed = load(paths[3])
    changed["policy_sha256"] = "1" * 64  # the policy's own source changed
    rewrite(paths[3], policy_sha256="1" * 64, decision_sha256=ledger.decision_sha256(changed))
    scored = load(score(sub, data, BASE + 5, honest_github(data, indices), tmp_path))
    assert scored["policy_versions"] == 2 and "code_changed" not in scored
    text = leaderboard.render_markdown(leaderboard.build(tmp_path / "board"))
    assert "[Alpha](../submissions/alpha/) (decisions span 2 policy versions)" in text
    assert "(decisions span 2 policy versions)" in monthly._label(monthly._entry(scored, "2020-01-01"))


def bot_commit(root, *paths):
    git(root, "add", "--", *map(str, paths))
    git(root, "commit", "-q", "-m", ledger.COMMIT_SUBJECT, author=(ledger.COMMIT_AUTHOR, ledger.COMMIT_SUBJECT))


def middle_entry_reason(sub, data, tmp_path, indices, github=None):
    """Score without committing anything more; the middle entry must be the only unverified one."""
    fit = load(score(sub, data, BASE + 5, github or honest_github(data, indices), tmp_path, push=False))["fit"]
    [listed] = fit["unverified_entries"]
    assert listed["decision_date"] == str(data.dates[BASE + 1]) and fit["verified"] == len(indices) - 1
    return listed["reason"]


def test_entry_added_by_someone_other_than_the_workflow_is_a_hold(sub, data, tmp_path):
    indices = range(BASE, BASE + 3)
    paths = record(sub, data, indices)
    bot_commit(tmp_path, paths[0], paths[2])
    push_entries(tmp_path, author=("maintainer", "add an entry by hand"))
    assert "not by the ledger workflow" in middle_entry_reason(sub, data, tmp_path, indices)


def test_entry_that_was_never_committed_is_a_hold(sub, data, tmp_path):
    indices = range(BASE, BASE + 3)
    paths = record(sub, data, indices)
    bot_commit(tmp_path, paths[0], paths[2])
    assert "never committed" in middle_entry_reason(sub, data, tmp_path, indices)


def test_entry_edited_after_its_commit_is_a_hold(sub, data, tmp_path):
    indices = range(BASE, BASE + 3)
    paths = record(sub, data, indices)
    push_entries(tmp_path)
    edited = load(paths[1])
    edited["reason"] = "edited after the push"
    rewrite(paths[1], reason=edited["reason"], decision_sha256=ledger.decision_sha256(edited))  # digest still fits
    assert "differs from the one commit" in middle_entry_reason(sub, data, tmp_path, indices)


def test_commit_that_does_not_descend_from_the_runs_head_is_a_hold(sub, data, tmp_path):
    indices = range(BASE, BASE + 3)
    record(sub, data, indices)
    push_entries(tmp_path)
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "later")
    later = git(tmp_path, "rev-parse", "HEAD")
    github = honest_github(data, indices)
    github.runs[str(9000 + BASE + 1)]["head_sha"] = later  # a run that started after the entry was pushed
    assert "does not descend" in middle_entry_reason(sub, data, tmp_path, indices, github)
    github.runs[str(9000 + BASE + 1)]["head_sha"] = "0" * 40
    assert "not in the repository history" in middle_entry_reason(sub, data, tmp_path, indices, github)


def test_unreadable_evidence_scores_nothing_and_rewrites_nothing(sub, data, tmp_path, tmp_path_factory):
    indices = range(BASE, BASE + 3)
    record(sub, data, indices)
    push_entries(tmp_path)
    board = tmp_path / "board" / "ledger" / "synthetic" / "alpha.json"

    github = honest_github(data, indices)
    del github.jobs[str(9000 + BASE + 1)]  # the jobs API answers 404
    with pytest.raises(ledger.VerificationUnavailable, match="cannot be read"):
        score(sub, data, BASE + 4, github, tmp_path, push=False)

    shallow = tmp_path_factory.mktemp("shallow")
    subprocess.run(["git", "clone", "-q", "--depth", "1", "file://" + str(tmp_path), str(shallow / "clone")], check=True)
    clone = Submission.load(shallow / "clone" / "submissions" / "alpha")
    with pytest.raises(ledger.VerificationUnavailable, match="shallow"):
        score(clone, data, BASE + 4, honest_github(data, indices), tmp_path, push=False)

    outside = make_submission(tmp_path_factory.mktemp("plain"), "alpha")
    (outside.directory / "ledger" / "synthetic").mkdir(parents=True)
    for path in (sub.directory / "ledger" / "synthetic").glob("*.json"):
        (outside.directory / "ledger" / "synthetic" / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ledger.VerificationUnavailable, match="not in a git repository"):
        score(outside, data, BASE + 4, honest_github(data, indices), tmp_path, push=False)
    assert not board.exists()


def test_open_times_follow_the_exchange_calendar():
    assert ledger._open_utc(TASK, "2019-07-01") == dt.datetime(2019, 7, 1, 13, 30, tzinfo=dt.timezone.utc)
    assert ledger._open_utc(TASK, "2019-01-02") == dt.datetime(2019, 1, 2, 14, 30, tzinfo=dt.timezone.utc)
    crypto = dataclasses.replace(TASK, source="binance")
    assert ledger._open_utc(crypto, "2019-07-01") == dt.datetime(2019, 7, 1, 0, 0, tzinfo=dt.timezone.utc)


def test_github_client_reads_json_sends_token_and_maps_404(monkeypatch):
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        seen.append(request)
        if request.full_url.endswith("/missing"):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return Response(b'{"id": 7}')

    monkeypatch.setattr(ledger.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    assert ledger.github_get("/repos/o/r/actions/runs/7") == {"id": 7}
    assert seen[0].get_header("Authorization") == "Bearer t0ken"
    assert ledger.github_get("/missing") is None


def test_github_client_caches_only_finished_runs_and_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ENV, str(tmp_path))
    answers = {
        "/repos/o/r/actions/runs/1": {"id": 1, "status": "completed"},
        "/repos/o/r/actions/runs/1/jobs?per_page=100": {"jobs": [{"status": "completed"}]},
        "/repos/o/r/actions/runs/2": {"id": 2, "status": "in_progress"},
    }
    calls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        calls.append(request.full_url.removeprefix(ledger.GITHUB_API))
        return Response(json.dumps(answers[calls[-1]]).encode())

    monkeypatch.setattr(ledger.urllib.request, "urlopen", fake_urlopen)
    for path in answers:
        assert ledger.github_get(path) == answers[path]
    for path in answers:
        assert ledger.github_get(path) == answers[path]
    assert calls == [*answers, "/repos/o/r/actions/runs/2"]  # the unfinished run is asked again


# --- submissions, record_all, leaderboard, CLI --------------------------------------


def test_submission_ledger_field(tmp_path):
    assert make_submission(tmp_path / "a", "x", ledger_tasks=("synthetic", "us-sectors")).ledger == (
        "synthetic", "us-sectors",
    )
    directory = tmp_path / "plain"
    directory.mkdir()
    spec = {"name": "P", "kind": "baseline", "policy": "rl.bench.policies.baselines:EqualWeightBuyHold"}
    (directory / "submission.json").write_text(json.dumps(spec))
    assert Submission.load(directory).ledger == ()
    for bad in ("synthetic", ["a", "a"], [""], [3]):
        (directory / "submission.json").write_text(json.dumps({**spec, "ledger": bad}))
        with pytest.raises(ValueError, match="ledger"):
            Submission.load(directory)
    (directory / "submission.json").write_text(json.dumps({**spec, "ledgr": []}))
    with pytest.raises(ValueError, match="Unknown submission fields"):
        Submission.load(directory)


def test_record_all_and_score_all(tmp_path, data):
    root = tmp_path / "submissions"
    make_submission(root, "alpha")
    make_submission(root, "beta", ledger_tasks=("nope",))
    make_submission(root, "gamma", ledger_tasks=())
    written, failures = ledger.record_all(root, data={"synthetic": data.rows(0, BASE + 1)}, now=NOW, run=identity(0))
    assert written == [root / "alpha" / "ledger" / "synthetic" / "{}.json".format(data.dates[BASE])]
    assert len(failures) == 1 and failures[0].startswith("nope/beta") and "Unknown task" in failures[0]
    assert ledger.record_all(root, submission="alpha", data={"synthetic": data.rows(0, BASE + 1)})[0] == written
    with pytest.raises(KeyError):
        ledger.record_all(root, submission="missing")

    push_entries(tmp_path)
    scored, score_failures = ledger.score_all(
        root,
        out_root=tmp_path / "board",
        data={"synthetic": data.rows(0, BASE + 3)},
        github=FakeGitHub({9000: good_run(data, BASE)}, {9000: good_jobs(data, BASE)}),
        repository=REPO,
    )
    assert scored == [tmp_path / "board" / "ledger" / "synthetic" / "alpha.json"]
    assert len(score_failures) == 1 and score_failures[0].startswith("nope/beta")


def test_retired_submissions_are_not_recorded_after_their_date(tmp_path, data):
    root = tmp_path / "submissions"
    make_submission(root, "active", retired="2019-06-01")  # retired from the day after
    make_submission(root, "gone", retired="2019-05-31")
    written, failures = ledger.record_all(root, data={"synthetic": data.rows(0, BASE + 1)}, now=NOW, run=identity(0))
    assert failures == [] and [path.parts[-4] for path in written] == ["active"]
    assert not (root / "gone" / "ledger").exists()


def test_leaderboard_renders_a_ledger_section(sub, data, tmp_path):
    indices = range(BASE, BASE + 5)
    record(sub, data, indices)
    score(sub, data, BASE + 6, honest_github(data, indices), tmp_path)
    board = tmp_path / "board"
    summary = leaderboard.build(board)
    [section] = summary["sections"]
    assert section["track"] == "ledger" and section["decisions_from"] == str(data.dates[BASE])
    text = leaderboard.render_markdown(summary)
    assert "## Ledger track: decisions committed before each open" in text
    assert "(decisions from {})".format(data.dates[BASE]) in text
    assert "## Live track" not in text and "## Historical track" not in text


def test_cli_registration():
    parser = ArgumentParser()
    common = ArgumentParser(add_help=False)
    common.add_argument("--submissions-root", default="submissions")
    common.add_argument("--out-root", default="leaderboard")
    ledger.add_cli(parser.add_subparsers(dest="command"), common)
    options = parser.parse_args(["ledger", "record", "--task", "synthetic", "--budget", "4"])
    assert options.handler is ledger._cmd_record and options.task == "synthetic" and options.budget == 4.0
    options = parser.parse_args(["ledger", "score", "--out-root", "x"])
    assert options.handler is ledger._cmd_score and options.out_root == "x"
