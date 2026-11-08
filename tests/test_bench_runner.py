"""Runner, live-track freeze/update and leaderboard behaviour on the synthetic task."""

import dataclasses
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rl import cli
from rl.bench import leaderboard, monthly, runner, scoring
from rl.bench.data import load_task_data
from rl.bench.engine import simulate
from rl.bench.market import as_day
from rl.bench.policies.baselines import EqualWeightBuyHold
from rl.bench.policy import Policy
from rl.bench.submission import Submission
from rl.bench.tasks import TASKS, get_task

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
    assert record["schema"] == 2 and record["track"] == "historical" and record["protocol_version"] == "2"
    assert record["code"] == runner.code_identity(tilt) and len(record["code"]["code_sha256"]) == 64
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
    assert publish(tilt, tmp_path, "2019-03-01") == ["tilt/synthetic-live/2"]  # cohort 2 has its own key
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


def _cells(text, name):
    [line] = [line for line in text.splitlines() if name in line]
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_short_live_window_keeps_return_and_drawdown_only(tilt, data, tmp_path):
    out = tmp_path / "board"
    early, _ = freeze_early(tilt, data)
    publish(tilt, tmp_path, early.dates[-1])
    runner.update_live(LIVE, tilt, data=data.between(None, "2019-01-15"), out_root=out)
    summary = leaderboard.build(out)
    [section] = summary["sections"]
    assert section["sessions"] < scoring.MIN_SHARPE_SESSIONS and section["ranked_by"] == "total_return"
    returns = [row["total_return"] for row in section["rows"]]
    assert len(returns) == 2 and np.isfinite(returns).all() and returns == sorted(returns, reverse=True)
    tilt_row = next(row for row in section["rows"] if row["id"] == "tilt")
    assert {"turnover", "alpha", "alpha_t", "delta_sharpe", "dsr"} <= tilt_row.keys()
    for row in section["rows"]:
        assert np.isfinite(row["max_drawdown"])
        assert all(np.isnan(row[key]) for key in leaderboard._ANNUALISED if key in row)

    text = leaderboard.render_markdown(summary)
    assert "Fewer than {} sessions".format(scoring.MIN_SHARPE_SESSIONS) in text
    cells = dict(zip(leaderboard._COLUMNS, _cells(text, "[Tilt](../submissions/tilt/)")))
    assert cells["Return"].endswith("%") and cells["Max DD"].endswith("%")
    assert {cells[key] for key in ("Sharpe", "CAGR", "Turnover/yr", "Alpha (t)", "ΔSharpe vs 1/N", "DSR")} == {"–"}

    runner.update_live(LIVE, tilt, data=data, out_root=out)  # the same record, now long enough
    [section] = leaderboard.build(out)["sections"]
    assert section["ranked_by"] == "sharpe" and all(np.isfinite(row["sharpe"]) for row in section["rows"])


def test_headline_shows_the_window_of_each_section_of_a_task(tilt, data, tmp_path):
    out = tmp_path / "board"
    early, _ = freeze_early(tilt, data)
    publish(tilt, tmp_path, "2019-01-15")
    fixed = Submission.load(shipped(tmp_path))
    runner.freeze(LIVE, fixed, data=early)
    publish(fixed, tmp_path, "2019-03-01")
    for sub in (tilt, fixed):
        runner.update_live(LIVE, sub, data=data, out_root=out)
    summary = leaderboard.build(out)
    first, second = summary["sections"]
    assert first["task"] == second["task"] == LIVE.id
    assert (first["published"], first["first"]) == ("2019-01-15", "2019-01-16")
    assert (second["published"], second["first"]) == ("2019-03-01", "2019-03-04")

    text = leaderboard.render_markdown(summary)
    assert "(published 2019-01-15)" in text and "(published 2019-03-01)" in text
    benchmarks = [line for line in leaderboard.render_headline(summary).splitlines() if "◆" in line]
    assert len(benchmarks) == 2
    for section, line in zip((first, second), benchmarks):
        assert "| `{}` (from {}, {} sessions) |".format(LIVE.id, section["first"], section["sessions"]) in line


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


def test_run_all_stale_recomputes_only_records_whose_submission_changed(tmp_path, capsys):
    root, out = tmp_path / "subs", tmp_path / "out"
    make_submission(root, "conc", "test_bench_runner:Concentrated")
    shipped = root / "first"
    shipped.mkdir()
    (shipped / "policy.py").write_text(
        "from test_bench_runner import Concentrated as First\n"
    )
    spec = {"name": "First", "kind": "other", "policy": "policy.py:First", "description": "v1"}
    (shipped / "submission.json").write_text(json.dumps(spec))
    argv = ["bench", "run-all", "--task", "synthetic", "--stale", "--submissions-root", str(root), "--out-root", str(out)]

    def written():
        assert cli.main(argv) == 0
        return sorted(line.split("/")[-1] for line in capsys.readouterr().out.splitlines() if line.startswith("Wrote"))

    assert written() == ["conc.json", "first.json"]
    # Imports leave bytecode behind; descriptions don't change results.
    (shipped / "__pycache__").mkdir(exist_ok=True)
    (shipped / "__pycache__" / "policy.cpython-313.pyc").write_bytes(b"stale bytecode")
    (shipped / "submission.json").write_text(json.dumps({**spec, "description": "v2"}))
    assert written() == []
    (shipped / "policy.py").write_text("from test_bench_runner import TiltPolicy as First\n")
    assert written() == ["first.json"]
    (shipped / "submission.json").write_text(json.dumps({**spec, "trials": 3}))
    assert written() == ["first.json"]
    # Without --stale every record is recomputed.
    assert cli.main(argv[:4] + argv[5:]) == 0
    assert capsys.readouterr().out.count("Wrote") == 2


# --- code identity, cohorts, retirement --------------------------------------------

POLICY = (
    "import numpy as np\n"
    "from rl.bench.policy import Policy\n\n"
    "class Fixed(Policy):\n"
    "    def act(self, view):\n"
    "        return np.full(len(view.tickers), 0.5 / len(view.tickers))\n"
)
EDITED = POLICY.replace("0.5 /", "0.25 /")  # a different size: no stale bytecode can hide the edit


def shipped(tmp_path, source=POLICY, **extra):
    directory = tmp_path / "submissions" / "fixed"
    directory.mkdir(parents=True)
    (directory / "policy.py").write_text(source)
    spec = {"name": "Fixed", "kind": "other", "policy": "policy.py:Fixed", **extra}
    (directory / "submission.json").write_text(json.dumps(spec))
    return directory


def downgrade(path, *drop):
    """Rewrite a schema 2 record or freeze as schema 1 wrote it."""
    document = json.loads(path.read_text())
    for key in ("protocol_version", "code", *drop):
        document.pop(key, None)
    document["schema"] = 1
    path.write_text(json.dumps(document, indent=2) + "\n")


def test_code_sources_follow_rl_bench_imports_and_nothing_else(tmp_path):
    local = Submission.load(shipped(tmp_path))
    assert {"rl/bench/__init__.py", "rl/bench/policy.py", "submissions/fixed/policy.py"} <= set(local.code_sources())
    builtin = Submission(id="b", name="B", kind="baseline", policy="rl.bench.policies.baselines:EqualWeightBuyHold")
    sources = builtin.code_sources()
    assert {"rl/bench/policies/baselines.py", "rl/bench/policies/__init__.py", "rl/bench/policy.py"} <= set(sources)
    assert not any(name.endswith(("leaderboard.py", "ledger.py", "runner.py")) for name in sources)
    assert builtin.code_sha256() == builtin.code_sha256() != local.code_sha256()


def test_live_replay_on_edited_code_is_marked_and_never_silent(tmp_path, data):
    directory = shipped(tmp_path)
    sub = Submission.load(directory)
    out = tmp_path / "board"
    marker = runner.freeze(LIVE, sub, data=data.between(None, "2019-01-01"))
    frozen = json.loads(marker.read_text())
    assert frozen["schema"] == 2 and frozen["cohort"] == 1 and frozen["code"] == runner.code_identity(sub)
    publish(sub, tmp_path, frozen["fitted_through"])

    same = json.loads(runner.update_live(LIVE, sub, data=data, out_root=out).read_text())
    assert same["code_changed"] is False and same["frozen_code"] == frozen["code"] and same["cohort"] == 1

    (directory / "policy.py").write_text(EDITED)
    edited = json.loads(runner.update_live(LIVE, Submission.load(directory), data=data, out_root=out).read_text())
    assert edited["code_changed"] is True
    assert edited["frozen_code"]["code_sha256"] == frozen["code"]["code_sha256"] != edited["code"]["code_sha256"]

    summary = leaderboard.build(out)
    [row] = [r for s in summary["sections"] for r in s["rows"] if r["id"] == "fixed"]
    assert row["code_changed"] is True
    assert "[Fixed](../submissions/fixed/) (code changed)" in leaderboard.render_markdown(summary)
    assert "(code changed)" in monthly._label(monthly._entry(edited, "2019-12-31"))


def test_schema_1_freeze_and_records_replay_and_render_as_before(tmp_path, data):
    sub = Submission.load(shipped(tmp_path))
    out = tmp_path / "board"
    marker = runner.freeze(LIVE, sub, data=data.between(None, "2019-01-01"))
    downgrade(marker, "cohort", "submission_sha256")
    publish(sub, tmp_path, "2019-01-01")
    path = runner.update_live(LIVE, sub, data=data, out_root=out)
    record = json.loads(path.read_text())
    assert record["code_identity"] == "unrecorded" and "code_changed" not in record
    assert runner.freeze_is_current(LIVE, sub, out)  # no fingerprint stored: its historical record decides

    runner.evaluate(LIVE, sub, data=data, out_root=out)
    assert runner.freeze_is_current(LIVE, sub, out)
    (sub.directory / "policy.py").write_text(EDITED)
    assert not runner.freeze_is_current(LIVE, Submission.load(sub.directory), out)
    (sub.directory / "policy.py").write_text(POLICY)

    before = leaderboard.render_markdown(leaderboard.build(out))
    entry_before = monthly._public(monthly._entry(record, "2019-12-31"))
    for stored in out.glob("*/*/*.json"):
        downgrade(stored, "cohort", "frozen_code", "code_identity", "code_changed", "submission_sha256")
        assert json.loads(stored.read_text())["schema"] == 1
    assert leaderboard.render_markdown(leaderboard.build(out)) == before
    assert monthly._public(monthly._entry(json.loads(path.read_text()), "2019-12-31")) == entry_before


def test_a_changed_submission_starts_a_new_cohort_beside_the_old_one(tmp_path, data):
    directory = shipped(tmp_path)
    sub = Submission.load(directory)
    out = tmp_path / "board"
    marker = runner.freeze(LIVE, sub, data=data.between(None, "2019-01-01"))
    first = marker.read_bytes()
    assert runner.freeze_is_current(LIVE, sub, out)
    assert publish(sub, tmp_path, "2019-01-15") == ["fixed/synthetic-live"]
    old_record = runner.update_live(LIVE, sub, data=data, out_root=out)
    assert old_record.name == "fixed.json"
    old_bytes = old_record.read_bytes()

    (directory / "policy.py").write_text(EDITED)
    changed = Submission.load(directory)
    assert not runner.freeze_is_current(LIVE, changed, out)  # a changed definition, not merely changed code
    later = data.between(None, "2019-02-01")
    runner.freeze(LIVE, changed, data=later, force=True)
    second = json.loads(marker.read_text())
    assert second["cohort"] == 2 and second["fitted_through"] == str(later.dates[-1])
    assert runner.freeze_is_current(LIVE, changed, out)
    archived = marker.parent / "cohorts" / "1"
    assert (archived / "freeze.json").read_bytes() == first and runner.verify_freeze(archived)["cohort"] == 1
    assert not list(directory.glob(".freeze-*"))

    assert runner.update_live(LIVE, changed, data=data, out_root=out) is None  # cohort 2 is not yet published
    assert publish(sub, tmp_path, "2019-03-01") == ["fixed/synthetic-live/2"]
    new_record = runner.update_live(LIVE, changed, data=data, out_root=out)
    assert new_record.name == "fixed.cohort-2.json" and json.loads(new_record.read_text())["cohort"] == 2
    assert old_record.read_bytes() == old_bytes  # the first cohort's record is untouched
    publications = json.loads((out / runner.PUBLICATIONS_FILE).read_text())
    assert sorted(publications) == ["fixed/synthetic-live", "fixed/synthetic-live/2"]

    summary = leaderboard.build(out)
    assert sorted(r.get("cohort", 1) for s in summary["sections"] for r in s["rows"] if r["id"] == "fixed") == [1, 2]
    text = leaderboard.render_markdown(summary)
    assert "[Fixed](../submissions/fixed/) (cohort 1)" in text and "[Fixed](../submissions/fixed/) (cohort 2)" in text
    report = monthly.write_monthly_report(out, "2019-12", submissions_root=directory.parent).read_text()
    assert "Fixed (cohort 1)" in report and "Fixed (cohort 2)" in report


def test_freeze_stale_refreezes_only_changed_submissions_and_skips_retired(tmp_path, monkeypatch):
    monkeypatch.setitem(TASKS, LIVE.id, LIVE)
    directory = shipped(tmp_path)
    gone = directory.parent / "gone"
    gone.mkdir()
    (gone / "submission.json").write_text(json.dumps({
        "name": "Gone", "kind": "other", "policy": "test_bench_runner:Concentrated", "retired": "2019-01-01",
    }))
    argv = ["bench", "freeze", "--stale", "--task", LIVE.id, "--submissions-root", str(directory.parent),
            "--out-root", str(tmp_path / "board")]
    marker = directory / "frozen" / LIVE.id / "freeze.json"
    assert cli.main(argv) == 0
    first = marker.read_bytes()
    assert json.loads(first)["cohort"] == 1 and not (gone / "frozen").exists()
    assert cli.main(argv) == 0
    assert marker.read_bytes() == first
    (directory / "policy.py").write_text(EDITED)
    assert cli.main(argv) == 0
    assert json.loads(marker.read_text())["cohort"] == 2
    assert (marker.parent / "cohorts" / "1" / "freeze.json").read_bytes() == first


@pytest.mark.parametrize("bad", ["2019-13-01", "tomorrow", "2019-1-1", 20190101])
def test_retired_must_be_a_date(tmp_path, bad):
    with pytest.raises(ValueError, match="retired"):
        Submission.load(shipped(tmp_path, retired=bad))


def test_a_submission_is_retired_the_day_after_its_date():
    sub = Submission(id="a", name="A", kind="other", policy="m:C", retired="2019-01-01")
    assert not sub.is_retired(dt.date(2019, 1, 1)) and sub.is_retired(dt.date(2019, 1, 2))
    assert not Submission(id="b", name="B", kind="other", policy="m:C").is_retired()


def test_retired_submissions_leave_the_ranking_but_keep_their_records(board, tmp_path):
    root = board.parent / "submissions"
    spec = json.loads((root / "tilt" / "submission.json").read_text())
    (root / "tilt" / "submission.json").write_text(json.dumps({**spec, "retired": "2019-01-01"}))
    record_path = board / "historical" / "synthetic" / "tilt.json"
    kept = record_path.read_bytes()

    summary = leaderboard.write_leaderboard(board, readme=None, submissions_root=root)
    [section] = summary["sections"]
    assert sorted(r["id"] for r in section["rows"] if r["id"]) == ["buyhold", "conc"]
    assert [r["rank"] for r in section["rows"]] == [1, 2, 3]
    [retired] = summary["retired"]
    assert (retired["id"], retired["track"], retired["task"], retired["retired"]) == (
        "tilt", "historical", "synthetic", "2019-01-01",
    )
    text = (board / "LEADERBOARD.md").read_text()
    assert "## Retired" in text and "[Tilt](../submissions/tilt/)" in text.split("## Retired")[1]
    assert "[Tilt]" not in text.split("## Retired")[0]
    assert record_path.read_bytes() == kept

    out = board.parent / "again"
    argv = ["--submissions-root", str(root), "--out-root", str(out)]
    assert cli.main(["bench", "run-all", "--task", "synthetic", *argv]) == 0
    assert sorted(p.stem for p in (out / "historical" / "synthetic").glob("*.json")) == ["buyhold", "conc"]


@pytest.mark.parametrize("crash", ["archiving the second file", "moving the archive to the new freeze", "the final rename"])
def test_an_interrupted_refreeze_never_loses_an_archived_cohort(tilt, data, tmp_path, monkeypatch, crash):
    early, marker = freeze_early(tilt, data)
    first = marker.read_bytes()
    publish(tilt, tmp_path, "2019-01-15")
    changed = Submission(**{**tilt.__dict__, "params": {"invested": 0.5}})
    later = data.between(None, "2019-02-01")

    real_move, real_rename, calls = shutil.move, Path.rename, []

    def move(source, destination, *args):
        calls.append(source)
        if len(calls) == (2 if crash.startswith("archiving") else 5):
            raise OSError("job cancelled")
        return real_move(source, destination, *args)

    def rename(self, target):
        if self.name.startswith(".freeze-"):
            raise OSError("job cancelled")
        return real_rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(shutil, "move", move)
        patched.setattr(Path, "rename", rename)
        with pytest.raises(OSError, match="cancelled"):
            runner.freeze(LIVE, changed, data=later, force=True, seeds=[0, 1, 2])

    again = runner.freeze(LIVE, changed, data=later, force=True, seeds=[0, 1, 2])  # the next CI run
    assert again == marker and json.loads(marker.read_text())["cohort"] == 2
    archived = marker.parent / "cohorts" / "1"
    assert (archived / "freeze.json").read_bytes() == first
    assert runner.verify_freeze(archived)["cohort"] == 1  # every artifact of the old cohort is there
    assert not list(tilt.directory.glob(".freeze-*"))
    assert publish(tilt, tmp_path, "2019-03-01") == ["tilt/synthetic-live/2"]
    assert sorted(json.loads((tmp_path / "board" / runner.PUBLICATIONS_FILE).read_text())) == [
        "tilt/synthetic-live", "tilt/synthetic-live/2",
    ]


def test_a_staging_directory_holding_cohorts_is_never_wiped(tilt, data):
    early, marker = freeze_early(tilt, data)
    staging = tilt.directory / ".freeze-{}.tmp".format(LIVE.id)
    (staging / "cohorts" / "1").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="archived cohorts"):
        runner.freeze(LIVE, tilt, data=early, force=True)
    assert (staging / "cohorts" / "1").is_dir() and marker.exists()


def test_every_record_writer_keeps_the_code_that_scored_it(tmp_path, data):
    sub = Submission.load(shipped(tmp_path))
    out = tmp_path / "board"
    marker = runner.freeze(LIVE, sub, data=data.between(None, "2019-01-01"))
    publish(sub, tmp_path, "2019-01-01")
    records = [
        runner.evaluate(LIVE, sub, data=data, out_root=out),
        runner.update_live(LIVE, sub, data=data, out_root=out),
    ]
    for path in records:
        record = json.loads(path.read_text())
        assert record["schema"] == 2 and record["code"] == runner.code_identity(sub), path
    assert json.loads(marker.read_text())["code"] == runner.code_identity(sub)


def test_monthly_report_leaves_out_retired_submissions(tmp_path, data):
    directory = shipped(tmp_path)
    sub = Submission.load(directory)
    out = tmp_path / "board"
    runner.freeze(LIVE, sub, data=data.between(None, "2019-01-01"))
    publish(sub, tmp_path, "2019-01-01")
    runner.update_live(LIVE, sub, data=data, out_root=out)
    root = directory.parent
    assert "Fixed" in monthly.write_monthly_report(out, "2019-12", submissions_root=root).read_text()
    spec = json.loads((directory / "submission.json").read_text())
    (directory / "submission.json").write_text(json.dumps({**spec, "retired": "2019-06-01"}))
    text = monthly.write_monthly_report(out, "2019-12", submissions_root=root).read_text()
    assert "Fixed" not in text and "No sessions scored yet" in text
