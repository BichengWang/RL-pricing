"""Declared sweeps: validation-only selection, trials, outputs and failure reporting."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from rl.bench import sweep
from rl.bench.engine import simulate
from rl.bench.market import MarketData
from rl.bench.policy import Policy
from rl.bench.submission import Submission
from rl.bench.tasks import get_task

TASK = dataclasses.replace(get_task("synthetic"), tickers=("AAA", "BBB", "CCC"))
SEEN = []  # (asset, last date the policy was shown) while scoring
FITS = []


class Pick(Policy):
    """Always holds one asset; ``asset < 0`` fails to fit."""

    stochastic = True

    def __init__(self, asset: int = 0):
        self.asset = asset

    def fit(self, history, validation_start, seed):
        FITS.append((self.asset, seed))
        if self.asset < 0:
            raise RuntimeError("cannot fit")

    def act(self, view):
        SEEN.append((self.asset, view.history.dates[-1]))
        weights = np.zeros(len(view.tickers))
        weights[self.asset] = 1.0
        return weights


def drifting_data() -> MarketData:
    """Asset 0 wins on validation and loses on test; asset 1 is the reverse."""
    dates = pd.bdate_range("2010-01-01", "2019-12-31").to_numpy(dtype="datetime64[D]")
    rng = np.random.default_rng(0)
    validation = (dates >= np.datetime64(TASK.validation_start)) & (dates < np.datetime64(TASK.test_start))
    test = dates >= np.datetime64(TASK.test_start)
    drift = np.zeros((len(dates), 3))
    drift[validation, 0], drift[test, 0] = 0.002, -0.002
    drift[validation, 1], drift[test, 1] = 0.0003, 0.003
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.005, drift.shape), axis=0))
    return MarketData(
        dates=dates, tickers=("AAA", "BBB", "CCC"), open=close, high=close, low=close, close=close,
        volume=np.ones_like(close),
    )


@pytest.fixture(autouse=True)
def reset():
    SEEN.clear()
    FITS.clear()


@pytest.fixture
def data():
    return drifting_data()


def make_base(root, policy="test_bench_sweep:Pick", params=None, sub_id="pick"):
    directory = root / sub_id
    directory.mkdir(parents=True)
    spec = {"name": "Pick", "kind": "other", "policy": policy, "params": params or {}, "trials": 1,
            "authors": ["tester"]}
    (directory / "submission.json").write_text(json.dumps(spec))
    return Submission.load(directory)


def run(tmp_path, data, grid, **kwargs):
    base = make_base(tmp_path / "submissions")
    return sweep.sweep(
        base, grid, [TASK], seeds=3, out_id="pick-tuned", processes=1,
        submissions_root=tmp_path / "submissions", out_root=tmp_path / "board", data={TASK.id: data},
        **kwargs,
    )


def test_selection_uses_validation_only(tmp_path, data):
    grid = [{"asset": 2}, {"asset": 1}, {"asset": 0}]
    document = run(tmp_path, data, grid)
    assert document["selection"]["index"] == 2 and document["selection"]["overrides"] == {"asset": 0}

    # No configuration other than the selected one is ever shown a test session.
    test_start = np.datetime64(TASK.test_start)
    assert {a for a, last in SEEN if last >= test_start} == {0}
    assert {a for a, _ in SEEN} == {0, 1, 2}

    # Asset 1 is the best on the test window, asset 0 on validation.
    start = data.index_of(TASK.test_start)
    sharpes = [
        sweep._sharpe(simulate(data, Pick(a).act, start, TASK.transaction_cost).returns) for a in range(3)
    ]
    assert int(np.argmax(sharpes)) == 1


def test_trials_outputs_and_records(tmp_path, data):
    grid = [{"asset": 2}, {"asset": 1}, {"asset": 0}]
    document = run(tmp_path, data, grid, rationale="because")
    out = tmp_path / "submissions" / "pick-tuned"

    submission = Submission.load(out)
    assert submission.trials == len(grid) == 3
    assert submission.params == {"asset": 0}
    assert submission.name == "Pick (tuned)" and submission.authors == ("tester",)
    assert "3 configurations" in submission.description and "validation" in submission.description

    saved = json.loads((out / "sweep.json").read_text())
    assert saved == json.loads(json.dumps(document))
    assert [c["overrides"] for c in saved["configs"]] == grid
    assert [c["selected"] for c in saved["configs"]] == [False, False, True]
    assert saved["rationale"] == "because"
    for config in saved["configs"]:
        score = config["validation"]["synthetic"]
        assert set(score["sharpe_per_seed"]) == {"0", "1", "2"} and score["sharpe_mean"] is not None
    assert saved["started_at"] and saved["finished_at"] and saved["failures"] == []
    assert saved["selection"]["mean_rank"] == 3.0
    assert not (out / sweep.PROGRESS_FILE).exists()

    # Only the selected configuration has test records, one per task, under the new id.
    records = sorted(p.relative_to(tmp_path / "board").as_posix() for p in (tmp_path / "board").rglob("*.json"))
    assert records == ["historical/synthetic/pick-tuned.json"]
    record = json.loads((tmp_path / "board" / records[0]).read_text())
    assert record["trials"] == 3 and record["params"] == {"asset": 0}
    assert [run_["seed"] for run_ in record["runs"]] == [0, 1, 2]
    assert saved["records"] == {"synthetic": str(tmp_path / "board" / records[0])}


def test_written_submission_keeps_every_declaration_its_records_were_computed_from(tmp_path, data):
    root = tmp_path / "submissions"
    base = make_base(root)
    (tmp_path / "news.jsonl").write_text("")
    declared = {
        "tasks": ["synthetic"], "info_sources": [{"source": "jsonl", "path": "../../news.jsonl"}],
        "model_cutoffs": [{"model": "m", "training_cutoff": "2009-12-31", "source_url": "https://example.com/m"}],
        "contamination": "possible", "retired": "2099-12-31",
    }
    spec = json.loads((base.directory / "submission.json").read_text())
    (base.directory / "submission.json").write_text(json.dumps({**spec, **declared}))
    base = Submission.load(base.directory)
    sweep.sweep(base, [{"asset": 0}, {"asset": 1}], [TASK], seeds=1, out_id="pick-tuned", processes=1,
                submissions_root=root, out_root=tmp_path / "board", data={TASK.id: data})

    written = Submission.load(root / "pick-tuned")
    assert written == dataclasses.replace(
        base, id="pick-tuned", directory=root / "pick-tuned", name="Pick (tuned)", params=written.params,
        trials=2, description=written.description,
    )
    assert "info_sha256" in json.loads((tmp_path / "board" / "historical" / "synthetic" / "pick-tuned.json").read_text())


def test_failed_job_is_reported_and_config_excluded(tmp_path, data):
    document = run(tmp_path, data, [{"asset": -1}, {"asset": 1}, {"asset": 0}])

    assert document["selection"]["index"] == 2
    assert document["configs"][0]["eligible"] is False and document["configs"][0]["mean_rank"] is None
    assert {f["config"] for f in document["failures"]} == {0}
    assert len(document["failures"]) == 3 and "cannot fit" in document["failures"][0]["error"]
    assert document["configs"][0]["validation"]["synthetic"]["failed_seeds"] == [0, 1, 2]
    assert Submission.load(tmp_path / "submissions" / "pick-tuned").trials == 3


def test_all_failed_raises_and_writes_sweep_json(tmp_path, data):
    with pytest.raises(sweep.SweepError):
        run(tmp_path, data, [{"asset": -1}, {"asset": -2}])
    out = tmp_path / "submissions" / "pick-tuned"
    assert not (out / "submission.json").exists()
    saved = json.loads((out / "sweep.json").read_text())
    assert saved["selection"] is None and len(saved["failures"]) == 6
    assert not (tmp_path / "board").exists()


def test_existing_output_is_refused(tmp_path, data):
    run(tmp_path, data, [{"asset": 0}, {"asset": 1}])
    with pytest.raises(FileExistsError):
        run(tmp_path, data, [{"asset": 0}, {"asset": 1}])


def test_duplicate_or_empty_grids_are_rejected(tmp_path, data):
    base = make_base(tmp_path / "submissions")
    for grid in ([], [{"asset": 0}, {"asset": 0}], [{}]):
        with pytest.raises(ValueError):
            sweep.sweep(base, grid, [TASK], out_id="x", processes=1, data={TASK.id: data},
                        submissions_root=tmp_path / "submissions", out_root=tmp_path / "board")


def test_resumes_from_progress_file(tmp_path, data):
    grid = [{"asset": 0}, {"asset": 1}]
    out = tmp_path / "submissions" / "pick-tuned"
    out.mkdir(parents=True)
    lines = []
    for index, params in enumerate(grid):
        for seed in range(3):
            lines.append({
                "task": "synthetic", "config": index, "seed": seed, "digest": sweep._digest(params),
                "ok": True, "sharpe": 1.0 if index == 1 else -1.0, "validation_sessions": 1,
                "data_sha256": "x", "seconds": 1.0, "finished_at": "t",
            })
    (out / sweep.PROGRESS_FILE).write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    document = run(tmp_path, data, grid)

    assert document["selection"]["index"] == 1  # the stored scores, not a refit
    assert {asset for asset, _ in FITS} == {1}  # only the final evaluation fitted anything
    assert len(FITS) == 3


def test_rank_helpers():
    assert sweep._average_ranks([0.1, 0.3, 0.2]) == [1.0, 3.0, 2.0]
    assert sweep._average_ranks([1.0, 1.0, 0.0]) == [2.5, 2.5, 1.0]
    scores = {"a": [0.1, 0.5, 0.4], "b": [0.5, 0.2, 0.6]}
    best, ranking = sweep._select(scores, 3)
    assert best == 2 and ranking["mean_rank"] == {0: 1.5, 1: 2.0, 2: 2.5}
    # exact ties go to the earliest declared configuration
    assert sweep._select({"a": [1.0, 1.0], "b": [2.0, 2.0]}, 2)[0] == 0
    # an incomplete configuration is neither ranked nor selectable
    best, ranking = sweep._select({"a": [None, 0.1, 0.2], "b": [0.9, 0.1, 0.2]}, 3)
    assert best == 2 and 0 not in ranking["mean_rank"]
    assert sweep._select({"a": [None]}, 1)[0] is None


def test_declared_grids_have_at_most_eight_distinct_configs():
    for algo, grid in sweep.GRIDS.items():
        assert len(grid) <= 8, algo
        assert len({json.dumps(c, sort_keys=True) for c in grid}) == len(grid)
        assert sweep.RATIONALES[algo]


def test_process_pool_and_cli(tmp_path):
    """Spawned workers load only pre-test data themselves; the CLI wires through."""
    root = tmp_path / "submissions"
    make_base(root, "rl.bench.policies.baselines:Momentum", {"top": 2}, sub_id="mom")
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps([{"lookback": 60}, {"lookback": 120}]))

    code = sweep.main([
        "sweep", "--submissions-root", str(root), "--out-root", str(tmp_path / "board"),
        "--base", "mom", "--out-id", "mom-tuned", "--grid-file", str(grid), "--tasks", "synthetic",
        "--processes", "2",
    ])

    assert code == 0
    saved = json.loads((root / "mom-tuned" / "sweep.json").read_text())
    assert saved["seeds"] == [0] and len(saved["configs"]) == 2  # deterministic policy: one seed
    assert saved["selection"]["params"]["lookback"] in (60, 120)
    assert Submission.load(root / "mom-tuned").trials == 2
    assert (tmp_path / "board" / "historical" / "synthetic" / "mom-tuned.json").exists()
