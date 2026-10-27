"""Runner, live-track freeze/update and leaderboard behaviour on the synthetic task."""

import dataclasses
import datetime as dt
import json

import numpy as np
import pytest

from rl import cli
from rl.bench import leaderboard, runner
from rl.bench.data import load_task_data
from rl.bench.engine import simulate
from rl.bench.market import as_day
from rl.bench.policies.baselines import EqualWeightBuyHold
from rl.bench.policy import Policy
from rl.bench.submission import Submission
from rl.bench.tasks import get_task

TASK = get_task("synthetic")
LIVE = dataclasses.replace(TASK, id="synthetic-live", live=True)
BUILT_WITH = []


class TiltPolicy(Policy):
    """Seed-dependent fixed weights; fitted state round-trips through save/load."""

    stochastic = True

    def __init__(self, invested: float = 0.9, transaction_cost: float = 0.0):
        self.invested = invested
        self.weights = None
        BUILT_WITH.append(transaction_cost)

    def fit(self, history, validation_start, seed):
        self.fit_end = history.dates[-1]
        weights = np.random.default_rng(seed).random(len(history.tickers))
        self.weights = self.invested * weights / weights.sum()

    def act(self, view):
        return self.weights

    def save(self, directory):
        (directory / "weights.json").write_text(json.dumps(self.weights.tolist()))

    def load(self, directory):
        self.weights = np.array(json.loads((directory / "weights.json").read_text()))


class Concentrated(Policy):
    stochastic = False

    def act(self, view):
        weights = np.zeros(len(view.tickers))
        weights[0] = 1.0
        return weights


@pytest.fixture(scope="module")
def data():
    return load_task_data(TASK, end="2020-01-01")


def make_submission(root, sub_id, policy, params=None, kind="other", trials=1):
    directory = root / sub_id
    directory.mkdir(parents=True)
    spec = {"name": sub_id.title(), "kind": kind, "policy": policy, "params": params or {}, "trials": trials}
    (directory / "submission.json").write_text(json.dumps(spec))
    return Submission.load(directory)


@pytest.fixture
def tilt(tmp_path):
    BUILT_WITH.clear()
    return make_submission(tmp_path / "submissions", "tilt", "test_bench_runner:TiltPolicy", trials=4)


def direct(policy, data, start, cost=TASK.transaction_cost):
    return simulate(data, policy.act, start=start, transaction_cost=cost)


def test_evaluate_matches_direct_simulation_and_injects_fee(tilt, data, tmp_path):
    path = runner.evaluate(TASK, tilt, data=data, out_root=tmp_path / "board")

    assert path == tmp_path / "board" / "historical" / "synthetic" / "tilt.json"
    record = json.loads(path.read_text())
    assert record["schema"] == 1 and record["track"] == "historical"
    assert [run["seed"] for run in record["runs"]] == [0, 1, 2, 3, 4]
    assert record["data"]["sha256"] == data.sha256()
    assert set(BUILT_WITH) == {TASK.transaction_cost}

    start = data.index_of(TASK.test_start)
    benchmark = direct(EqualWeightBuyHold(), data, start)
    np.testing.assert_allclose(record["benchmark"]["equity"], benchmark.equity, rtol=1e-9)
    assert record["dates"] == [str(d) for d in benchmark.dates]
    assert record["dates"][1] == TASK.test_start

    expected = TiltPolicy(transaction_cost=0.0)
    expected.fit(data.between(None, TASK.test_start), as_day(TASK.validation_start), 3)
    assert expected.fit_end < as_day(TASK.test_start)
    run = record["runs"][3]
    reference = direct(expected, data, start)
    np.testing.assert_allclose(run["equity"], reference.equity, rtol=1e-9)
    np.testing.assert_allclose(run["turnover"], reference.turnover, rtol=1e-9, atol=1e-12)


def test_deterministic_submission_runs_a_single_seed(tmp_path, data):
    sub = make_submission(tmp_path / "s", "conc", "test_bench_runner:Concentrated")
    record = json.loads(runner.evaluate(TASK, sub, data=data, out_root=tmp_path / "b").read_text())
    assert [run["seed"] for run in record["runs"]] == [0]


@pytest.mark.parametrize("count", [0, -1])
def test_cli_rejects_nonpositive_seeds_without_overwriting_results(tilt, data, tmp_path, count):
    out = tmp_path / "board"
    path = runner.evaluate(TASK, tilt, data=data, seeds=[0], out_root=out)
    original = path.read_bytes()

    with pytest.raises(SystemExit, match="--seeds must be positive"):
        cli.main([
            "bench", "run", "--task", TASK.id, "--submission", tilt.id,
            "--submissions-root", str(tilt.directory.parent), "--out-root", str(out),
            "--seeds", str(count),
        ])
    assert path.read_bytes() == original


def test_empty_seeds_do_not_overwrite_results_or_frozen_artifacts(tilt, data, tmp_path):
    path = runner.evaluate(TASK, tilt, data=data, seeds=[0], out_root=tmp_path / "board")
    marker = runner.freeze(LIVE, tilt, data=data, seeds=[0])
    original = {p: p.read_bytes() for p in [path, marker, *marker.parent.rglob("weights.json")]}

    with pytest.raises(ValueError, match="At least one seed is required"):
        runner.evaluate(TASK, tilt, data=data, seeds=[], out_root=tmp_path / "board")
    with pytest.raises(ValueError, match="At least one seed is required"):
        runner.freeze(LIVE, tilt, data=data, seeds=iter(()), force=True)
    for path, content in original.items():
        assert path.read_bytes() == content


def test_evaluate_logs_when_data_changes_under_a_record(tilt, data, tmp_path, caplog):
    out = tmp_path / "board"
    runner.evaluate(TASK, tilt, data=data, seeds=[0], out_root=out)
    with caplog.at_level("WARNING"):
        runner.evaluate(TASK, tilt, data=data, seeds=[0], out_root=out)
        assert "changed" not in caplog.text
        runner.evaluate(TASK, tilt, data=data.rows(1, len(data)), seeds=[0], out_root=out)
    assert "data sha256 changed" in caplog.text


def freeze_early(tilt, data):
    early = data.between(None, "2019-01-01")
    return early, runner.freeze(LIVE, tilt, data=early)


def publish(tilt, tmp_path, day):
    when = dt.datetime.fromisoformat(str(day)).replace(hour=12, tzinfo=dt.timezone.utc)
    return runner.publish(tilt.directory.parent, out_root=tmp_path / "board", now=when)


def test_freeze_then_update_live_equals_fresh_fit_over_post_freeze_sessions(tilt, data, tmp_path):
    early, marker = freeze_early(tilt, data)
    frozen = json.loads(marker.read_text())
    assert frozen["fitted_through"] == str(early.dates[-1])
    assert frozen["seeds"] == [0, 1, 2]
    assert sorted(frozen["artifacts"]) == ["seed{}/weights.json".format(s) for s in range(3)]
    gap = as_day(frozen["validation_start"]) - (as_day(frozen["fitted_through"]) - TASK.validation_length)
    assert np.timedelta64(0, "D") <= gap < np.timedelta64(7, "D")

    assert publish(tilt, tmp_path, frozen["fitted_through"]) == ["tilt/synthetic-live"]
    path = runner.update_live(LIVE, tilt, data=data, out_root=tmp_path / "board")
    record = json.loads(path.read_text())
    assert path.parent.parent.name == "live" and record["track"] == "live"
    assert record["fit"]["frozen_at"] == frozen["frozen_at"]
    assert record["dates"][0] == frozen["fitted_through"]
    assert as_day(record["dates"][1]) > as_day(frozen["fitted_through"])
    assert record["dates"][-1] == str(data.dates[-1])

    start = len(early)
    for seed, run in enumerate(record["runs"]):
        fresh = TiltPolicy()
        fresh.fit(early, as_day(frozen["validation_start"]), seed)
        np.testing.assert_allclose(run["equity"], direct(fresh, data, start).equity, rtol=1e-9)
    np.testing.assert_allclose(
        record["benchmark"]["equity"], direct(EqualWeightBuyHold(), data, start).equity, rtol=1e-9
    )


def test_update_live_without_new_sessions_writes_nothing(tilt, data, tmp_path):
    early, _ = freeze_early(tilt, data)
    publish(tilt, tmp_path, early.dates[-1])
    assert runner.update_live(LIVE, tilt, data=early, out_root=tmp_path / "board") is None
    assert not (tmp_path / "board" / "live").exists()


def test_live_scoring_waits_for_publication_and_starts_after_it(tilt, data, tmp_path):
    early, marker = freeze_early(tilt, data)
    assert runner.update_live(LIVE, tilt, data=data, out_root=tmp_path / "board") is None
    publish(tilt, tmp_path, "2019-01-15")
    record = json.loads(runner.update_live(LIVE, tilt, data=data, out_root=tmp_path / "board").read_text())
    assert record["dates"][0] == "2019-01-15" and record["dates"][1] == "2019-01-16"
    assert record["fit"]["published_at"].startswith("2019-01-15")
    # Publishing again changes nothing; a re-freeze needs a new publication.
    assert publish(tilt, tmp_path, "2019-02-01") == []
    runner.freeze(LIVE, tilt, data=early, force=True, seeds=[7])
    assert runner.update_live(LIVE, tilt, data=data, out_root=tmp_path / "board") is None
    assert publish(tilt, tmp_path, "2019-03-01") == ["tilt/synthetic-live"]
    record = json.loads(runner.update_live(LIVE, tilt, data=data, out_root=tmp_path / "board").read_text())
    assert record["dates"][1] == "2019-03-04"


def test_tampered_artifact_is_rejected(tilt, data, tmp_path):
    _, marker = freeze_early(tilt, data)
    (marker.parent / "seed1" / "weights.json").write_text(json.dumps([1, 0, 0, 0, 0]))
    with pytest.raises(ValueError, match="changed since the freeze"):
        runner.update_live(LIVE, tilt, data=data, out_root=tmp_path / "board")


def test_refreeze_needs_force_and_changed_params_are_rejected(tilt, data, tmp_path):
    early, marker = freeze_early(tilt, data)
    with pytest.raises(FileExistsError):
        runner.freeze(LIVE, tilt, data=early)
    runner.freeze(LIVE, tilt, data=early, force=True, seeds=[7])
    assert json.loads(marker.read_text())["seeds"] == [7]
    assert [p.name for p in marker.parent.glob("seed*")] == ["seed7"]

    changed = Submission(**{**tilt.__dict__, "params": {"invested": 0.5}})
    with pytest.raises(ValueError, match="changed since it was frozen"):
        runner.update_live(LIVE, changed, data=data, out_root=tmp_path / "board")


def test_ledger_participants_are_never_replayed_on_past_data(tilt, data, tmp_path, capsys):
    ledger_only = Submission(**{**tilt.__dict__, "ledger": ("synthetic",)})
    with pytest.raises(ValueError, match="decision ledger"):
        runner.evaluate(TASK, ledger_only, data=data, out_root=tmp_path / "board")
    with pytest.raises(ValueError, match="decision ledger"):
        runner.freeze(LIVE, ledger_only, data=data)

    root = tmp_path / "subs"
    make_submission(root, "conc", "test_bench_runner:Concentrated")
    directory = root / "agent"
    directory.mkdir()
    spec = {"name": "Agent", "kind": "llm", "policy": "test_bench_runner:Concentrated", "ledger": ["synthetic"]}
    (directory / "submission.json").write_text(json.dumps(spec))
    argv = ["--submissions-root", str(root), "--out-root", str(tmp_path / "out")]
    assert cli.main(["bench", "run-all", "--task", "synthetic", *argv]) == 0
    assert sorted(p.stem for p in (tmp_path / "out" / "historical" / "synthetic").glob("*.json")) == ["conc"]


def test_tasks_without_a_live_track_cannot_be_frozen(tilt, data):
    assert not TASK.live
    with pytest.raises(ValueError, match="no live track"):
        runner.freeze(TASK, tilt, data=data)


@pytest.fixture
def board(tmp_path, data):
    root = tmp_path / "submissions"
    out = tmp_path / "board"
    subs = [
        make_submission(root, "tilt", "test_bench_runner:TiltPolicy", trials=4),
        make_submission(root, "conc", "test_bench_runner:Concentrated"),
        make_submission(root, "buyhold", "rl.bench.policies.baselines:EqualWeightBuyHold", kind="baseline"),
    ]
    for sub in subs:
        runner.evaluate(TASK, sub, data=data, seeds=[0, 1] if sub.id == "tilt" else None, out_root=out)
    return out


def test_leaderboard_orders_by_sharpe_and_marks_benchmark(board, tmp_path):
    summary = leaderboard.write_leaderboard(board, readme=None)

    (section,) = summary["sections"]
    rows = section["rows"]
    sharpes = [row["sharpe"] for row in rows]
    assert sharpes == sorted(sharpes, reverse=True)
    assert [row["rank"] for row in rows] == [1, 2, 3, 4]
    assert sum(row["benchmark"] for row in rows) == 1
    buyhold = next(row for row in rows if row["id"] == "buyhold")
    assert buyhold["delta_sharpe"] == pytest.approx(0.0, abs=1e-6)
    bench = next(row for row in rows if row["benchmark"])
    assert bench["sharpe"] == pytest.approx(buyhold["sharpe"], abs=1e-6)
    tilt = next(row for row in rows if row["id"] == "tilt")
    assert tilt["seeds"] == 2 and tilt["sharpe_sd"] > 0
    assert all(0 <= row["dsr"] <= 1 for row in rows if np.isfinite(row["dsr"]))

    text = (board / "LEADERBOARD.md").read_text()
    assert "◆" in text and "[Tilt](../submissions/tilt/)" in text
    assert json.loads((board / "summary.json").read_text())["sections"][0]["task"] == "synthetic"


def test_readme_block_replacement_is_idempotent(board, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n{}\nold\n{}\noutro\n".format(leaderboard.README_START, leaderboard.README_END))
    leaderboard.write_leaderboard(board, readme=readme)
    first = readme.read_text()
    leaderboard.write_leaderboard(board, readme=readme)
    assert readme.read_text() == first
    assert first.startswith("intro\n") and first.endswith("\noutro\n") and "old" not in first
    # Synthetic demo results stay out of the README headline.
    assert "synthetic" not in first

    plain = tmp_path / "PLAIN.md"
    plain.write_text("no markers\n")
    leaderboard.write_leaderboard(board, readme=plain)
    assert plain.read_text() == "no markers\n"


def test_cli_lists_tasks_and_runs_a_submission(tmp_path, capsys):
    submissions = tmp_path / "submissions"
    make_submission(submissions, "conc", "test_bench_runner:Concentrated")
    out = tmp_path / "board"

    cli.main(["bench", "tasks"])
    assert "us-sectors" in capsys.readouterr().out

    argv = ["--submissions-root", str(submissions), "--out-root", str(out)]
    cli.main(["bench", "run", "--task", "synthetic", "--submission", "conc", *argv])
    assert (out / "historical" / "synthetic" / "conc.json").exists()

    cli.main(["bench", "leaderboard", "--out-root", str(out), "--no-readme"])
    assert "Conc" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        cli.main(["bench", "run", "--task", "synthetic", "--submission", "missing", *argv])


def test_submission_can_ship_its_own_policy_file(tmp_path, data):
    directory = tmp_path / "submissions" / "first-asset"
    directory.mkdir(parents=True)
    (directory / "policy.py").write_text(
        "import numpy as np\n"
        "from rl.bench.policy import Policy\n\n"
        "class FirstAsset(Policy):\n"
        "    def act(self, view):\n"
        "        weights = np.zeros(len(view.tickers))\n"
        "        weights[0] = 1.0\n"
        "        return weights\n"
    )
    spec = {"name": "First", "kind": "other", "policy": "policy.py:FirstAsset"}
    (directory / "submission.json").write_text(json.dumps(spec))
    record = json.loads(runner.evaluate(TASK, Submission.load(directory), data=data, out_root=tmp_path / "b").read_text())
    reference = direct(Concentrated(), data, data.index_of(TASK.test_start))
    np.testing.assert_allclose(record["runs"][0]["equity"], reference.equity, rtol=1e-9)

    missing = Submission(**{**Submission.load(directory).__dict__, "policy": "absent.py:FirstAsset"})
    with pytest.raises(FileNotFoundError):
        missing.policy_class()
