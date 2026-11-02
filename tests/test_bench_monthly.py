"""Monthly live report on synthetic live records built through the real runner."""

import dataclasses
import datetime as dt
import json
from argparse import ArgumentParser

import numpy as np
import pytest

from rl.bench import monthly, runner
from rl.bench.data import load_task_data
from rl.bench.policy import Policy
from rl.bench.submission import Submission
from rl.bench.tasks import get_task

TASK = get_task("synthetic")
LIVE = dataclasses.replace(TASK, id="synthetic-live", live=True)


class Tilt(Policy):
    stochastic = True

    def __init__(self, invested: float = 0.9):
        self.invested = invested
        self.weights = None

    def fit(self, history, validation_start, seed):
        weights = np.random.default_rng(seed).random(len(history.tickers))
        self.weights = self.invested * weights / weights.sum()

    def act(self, view):
        return self.weights

    def save(self, directory):
        (directory / "weights.json").write_text(json.dumps(self.weights.tolist()))

    def load(self, directory):
        self.weights = np.array(json.loads((directory / "weights.json").read_text()))


def make_submission(root, sub_id, invested):
    directory = root / sub_id
    directory.mkdir(parents=True)
    spec = {"name": sub_id.title(), "kind": "other", "policy": "test_bench_monthly:Tilt",
            "params": {"invested": invested}}
    (directory / "submission.json").write_text(json.dumps(spec))
    return Submission.load(directory)


def publish(root, board, day):
    when = dt.datetime.fromisoformat(day).replace(hour=12, tzinfo=dt.timezone.utc)
    runner.publish(root, out_root=board, now=when)


@pytest.fixture(scope="module")
def data():
    return load_task_data(TASK, end="2020-01-01")


@pytest.fixture
def board(tmp_path, data):
    """Two live records of one task: one published 2019-01-15, the other 2019-02-11."""
    early = data.between(None, "2019-01-01")
    root = tmp_path / "submissions"
    board = tmp_path / "board"
    first = make_submission(root, "first", 0.9)
    second = make_submission(root, "second", 0.5)
    runner.freeze(LIVE, first, data=early)
    publish(root, board, "2019-01-15")
    runner.freeze(LIVE, second, data=early)
    publish(root, board, "2019-02-11")
    for submission in (first, second):
        assert runner.update_live(LIVE, submission, data=data, out_root=board)
    return board


def test_report_renders_per_task_tables_with_window_and_sharpe_rule(board):
    path = monthly.write_monthly_report(board, "2019-03")

    assert path == board / "reports" / "2019-03.md"
    assert (board / "reports" / "2019-03.png").stat().st_size > 0
    text = path.read_text()
    assert text.startswith("# Live report 2019-03")
    assert "[LEADERBOARD.md](../LEADERBOARD.md)" in text and "![" in text and "2019-03.png" in text
    assert "Window: 2019-01-15 " in text and "→ 2019-03-29" in text
    assert "`synthetic-live`" in text
    assert "| First |" in text and "| Second |" in text
    summary = json.loads((board / "reports" / "2019-03.json").read_text())
    entries = summary["entries"]
    assert set(entries) == {"live/synthetic-live/first", "live/synthetic-live/second"}
    first, second = entries["live/synthetic-live/first"], entries["live/synthetic-live/second"]
    assert first["start"] == "2019-01-16" and second["start"] == "2019-02-12"
    assert first["sessions"] > second["sessions"] >= monthly.MIN_SHARPE_SESSIONS
    assert first["vs_benchmark"] == pytest.approx(first["total_return"] - first["benchmark_return"], abs=1e-9)
    assert first["sharpe"] is not None and first["max_drawdown"] <= 0
    assert len(text.split("## Findings")[1].split("\n\n")[1].splitlines()) == 3


def test_numbers_match_the_record_curves(board):
    monthly.write_monthly_report(board, "2019-12")
    record = json.loads((board / "live" / "synthetic-live" / "first.json").read_text())
    summary = json.loads((board / "reports" / "2019-12.json").read_text())["entries"]["live/synthetic-live/first"]
    runs = np.array([run["equity"] for run in record["runs"]])
    returns = (runs[:, 1:] / runs[:, :-1] - 1).mean(axis=0)
    equity = np.cumprod(1 + returns)
    benchmark = np.array(record["benchmark"]["equity"])
    assert summary["total_return"] == pytest.approx(equity[-1] - 1)
    assert summary["benchmark_return"] == pytest.approx(benchmark[-1] / benchmark[0] - 1)
    assert summary["max_drawdown"] == pytest.approx(
        (np.concatenate(([1.0], equity)) / np.maximum.accumulate(np.concatenate(([1.0], equity))) - 1).min()
    )


def test_short_window_shows_session_count_instead_of_sharpe(board):
    path = monthly.write_monthly_report(board, "2019-01")
    text = path.read_text()
    summary = json.loads((board / "reports" / "2019-01.json").read_text())["entries"]
    assert set(summary) == {"live/synthetic-live/first"}  # the other one is published in February
    sessions = summary["live/synthetic-live/first"]["sessions"]
    assert sessions < monthly.MIN_SHARPE_SESSIONS and summary["live/synthetic-live/first"]["sharpe"] is None
    assert "– ({} sessions)".format(sessions) in text
    assert summary["live/synthetic-live/first"]["last"] == "2019-01-31"
    assert "Only one strategy-task pair is scored so far." in text


def test_idempotent(board):
    paths = [monthly.write_monthly_report(board, "2019-02")]
    first = {p.name: p.read_bytes() for p in (board / "reports").iterdir()}
    paths.append(monthly.write_monthly_report(board, "2019-02"))
    second = {p.name: p.read_bytes() for p in (board / "reports").iterdir()}
    assert first == second and set(first) == {"2019-02.md", "2019-02.png", "2019-02.json"}
    assert paths[0] == paths[1]


def test_month_over_month_diff_uses_previous_summary(board):
    monthly.write_monthly_report(board, "2019-01")
    assert "## Change since" not in (board / "reports" / "2019-01.md").read_text()
    path = monthly.write_monthly_report(board, "2019-03")  # 2019-02 never rendered: diff against 2019-01
    text = path.read_text()
    assert "## Change since 2019-01" in text
    old = json.loads((board / "reports" / "2019-01.json").read_text())["entries"]["live/synthetic-live/first"]
    new = json.loads((board / "reports" / "2019-03.json").read_text())["entries"]["live/synthetic-live/first"]
    change = text.split("## Change since 2019-01")[1]
    assert "{} → {}".format(old["sessions"], new["sessions"]) in change
    delta = "{:+.1f} pp".format((new["vs_benchmark"] - old["vs_benchmark"]) * 100)
    assert delta in change
    # The pair that started after January has no like-for-like predecessor.
    assert "(new)" in change.split("Second")[1].split("\n")[0]
    # Re-rendering January afterwards must not diff against the later month.
    monthly.write_monthly_report(board, "2019-01")
    assert "## Change since" not in (board / "reports" / "2019-01.md").read_text()


def test_no_live_records_writes_a_short_report(tmp_path):
    board = tmp_path / "board"
    path = monthly.write_monthly_report(board, "2026-10")
    text = path.read_text()
    assert "No sessions scored yet" in text and "LEADERBOARD.md" in text
    assert "|" not in text and not (board / "reports" / "2026-10.png").exists()
    assert json.loads((board / "reports" / "2026-10.json").read_text())["entries"] == {}


def test_month_before_any_session_is_empty(board):
    text = monthly.write_monthly_report(board, "2018-12").read_text()
    assert "No sessions scored yet" in text


def test_ledger_records_are_included_and_labelled(board):
    ledger = board / "ledger" / "synthetic-live"
    ledger.mkdir(parents=True)
    record = json.loads((board / "live" / "synthetic-live" / "first.json").read_text())
    record["track"] = "ledger"
    record["fit"] = {"first_decision": "2019-01-14", "entries": 3, "verified": 3, "unverified": 0,
                     "missed_sessions": 0}
    (ledger / "first.json").write_text(json.dumps(record))
    text = monthly.write_monthly_report(board, "2019-03").read_text()
    assert "First (ledger)" in text
    entries = json.loads((board / "reports" / "2019-03.json").read_text())["entries"]
    assert "ledger/synthetic-live/first" in entries and entries["ledger/synthetic-live/first"]["published"] == "2019-01-14"


def test_default_month_is_the_last_completed_month(tmp_path):
    now = dt.datetime(2026, 11, 2, 6, 0, tzinfo=dt.timezone.utc)
    assert monthly.write_monthly_report(tmp_path, now=now).name == "2026-10.md"
    january = dt.datetime(2027, 1, 2, tzinfo=dt.timezone.utc)
    assert monthly.write_monthly_report(tmp_path, now=january).name == "2026-12.md"
    with pytest.raises(ValueError, match="YYYY-MM"):
        monthly.write_monthly_report(tmp_path, "2026-13")


def test_add_cli_registers_monthly(board, capsys):
    parser = ArgumentParser()
    common = ArgumentParser(add_help=False)
    common.add_argument("--out-root")
    common.add_argument("--submissions-root", default="submissions")
    monthly.add_cli(parser.add_subparsers(dest="command"), common)
    options = parser.parse_args(["monthly", "--out-root", str(board), "--month", "2019-03"])
    assert options.handler(options) == 0
    assert "# Live report 2019-03" in capsys.readouterr().out


def test_formatting_never_shows_negative_zero():
    assert monthly._pp(-0.00001) == "+0.0 pp" and monthly._pct(-0.0000001) == "+0.0%"
    assert monthly._pp(0.0123) == "+1.2 pp"
