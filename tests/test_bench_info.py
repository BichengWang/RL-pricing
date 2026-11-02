"""Point-in-time information, model cutoffs and the lookahead check, offline on the synthetic task."""

import dataclasses
import hashlib
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from rl import cli
from rl.bench import info as info_module
from rl.bench import leaderboard, runner
from rl.bench.causal import check_causal
from rl.bench.data import DATA_ENV, load_task_data
from rl.bench.info import InfoItem, InfoPanel, load_task_info, session_close
from rl.bench.policy import Policy
from rl.bench.submission import Submission, discover
from rl.bench.tasks import get_task

TASK = get_task("synthetic")
NEW_YORK = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ENV, str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def data():
    return load_task_data(TASK, end="2020-01-01")


def item(item_id, published_at, tickers=("AAA",), text="", **extra):
    return InfoItem(
        id=item_id, source="test", tickers=tickers, published_at=published_at, title=item_id,
        text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(), **extra,
    )


def eastern(text):
    return datetime.fromisoformat(text).replace(tzinfo=NEW_YORK)


def visible(panel, day, **kwargs):
    return [i.id for i in panel.at_session(day, **kwargs).items()]


def make_submission(root, sub_id, policy, kind="other", **spec):
    directory = root / sub_id
    directory.mkdir(parents=True, exist_ok=True)
    spec = {"name": sub_id, "kind": kind, "policy": policy, "params": {}, "trials": 1, **spec}
    (directory / "submission.json").write_text(json.dumps(spec))
    return Submission.load(directory)


# --- visibility -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "published, decision_day, next_day",
    [
        ("2018-03-09 16:05", "2018-03-09", "2018-03-12"),  # EST Friday before the spring-forward weekend
        ("2018-03-12 16:05", "2018-03-12", "2018-03-13"),  # EDT: 20:05 UTC, an hour before a fixed-EST close
        ("2018-11-02 16:05", "2018-11-02", "2018-11-05"),  # EDT Friday before the fall-back weekend
        ("2018-11-05 16:05", "2018-11-05", "2018-11-06"),  # EST: 21:05 UTC
    ],
)
def test_filing_after_the_close_is_visible_only_from_the_next_session(published, decision_day, next_day):
    panel = InfoPanel([item("late", eastern(published))])
    assert visible(panel, decision_day) == []
    assert visible(panel, next_day) == ["late"]


def test_the_close_instant_itself_is_not_visible():
    panel = InfoPanel([
        item("before", eastern("2018-03-12 15:59:59")),
        item("at", eastern("2018-03-12 16:00:00")),
    ])
    assert visible(panel, "2018-03-12") == ["before"]
    assert visible(panel, "2018-03-13") == ["before", "at"]


def test_crypto_sessions_close_at_midnight_utc():
    panel = InfoPanel([
        item("inside", datetime.fromisoformat("2023-05-01T23:59:59+00:00")),
        item("boundary", datetime.fromisoformat("2023-05-02T00:00:00+00:00")),
    ])
    assert visible(panel, "2023-05-01", crypto=True) == ["inside"]
    assert visible(panel, "2023-05-02", crypto=True) == ["inside", "boundary"]


def test_view_filters_by_ticker_and_since_and_cannot_be_widened():
    panel = InfoPanel([
        item("a", eastern("2018-03-01 10:00"), tickers=("AAA",)),
        item("b", eastern("2018-03-02 10:00"), tickers=("BBB",)),
        item("c", eastern("2018-03-05 10:00"), tickers=("AAA", "BBB")),
    ])
    view = panel.at_session("2018-03-06")
    assert [i.id for i in view.items(ticker="AAA")] == ["a", "c"]
    assert [i.id for i in view.items(since=eastern("2018-03-02 10:00"))] == ["b", "c"]
    with pytest.raises(AttributeError):
        view._visible = ()


@pytest.mark.parametrize("value", ["2018-03-12T10:00:00", "2018-03-12T10:00:00.500000+00:00"], ids=["naive", "subsecond"])
def test_timestamps_need_a_zone_and_second_precision(value):
    with pytest.raises(ValueError, match="timezone and second precision"):
        item("bad", value)


def test_items_reject_a_text_hash_that_does_not_match():
    with pytest.raises(ValueError, match="text_sha256"):
        InfoItem(
            id="x", source="s", tickers=("AAA",), published_at="2018-01-01T00:00:00+00:00",
            title="", text="body", text_sha256="0" * 64,
        )


# --- sources ---------------------------------------------------------------------------


def news(item_id, published_at, text="headline", tickers=("AAA",), **extra):
    return {
        "id": item_id, "source": "news", "tickers": list(tickers), "published_at": published_at,
        "title": item_id, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), **extra,
    }


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_jsonl_source_loads_relative_to_the_submission_with_model_scores(tmp_path):
    submission = make_submission(
        tmp_path / "subs", "scored", "rl.bench.policies.baselines:Cash",
        info_sources=[{"source": "jsonl", "path": "news.jsonl"}],
    )
    write_jsonl(submission.directory / "news.jsonl", [
        news("n1", "2018-03-12T14:00:00Z", fields={"score": {"value": 0.7, "model": "scorer-1"}, "n": 3}),
        news("elsewhere", "2018-03-12T14:00:00Z", tickers=("ZZZ",)),
    ])
    panel = load_task_info(TASK, submission)
    (loaded,) = panel.at_session("2018-03-13").items()
    assert loaded.id == "n1" and loaded.fields["n"] == 3
    assert (loaded.fields["score"].value, loaded.fields["score"].model) == ("0.7" and 0.7, "scorer-1")
    assert panel.models == {"scorer-1"}


def edgar_cache(tmp_path):
    """Cached EDGAR responses: SEC's JSON acceptance times are wrong on purpose, headers are right."""
    cache = tmp_path / "cache" / "info" / "sec-edgar"
    (cache / "headers").mkdir(parents=True)
    (cache / "company_tickers.json").write_text(json.dumps({
        "0": {"cik_str": 1, "ticker": "AAA", "title": "A"},
        "1": {"cik_str": 2, "ticker": "OTHER", "title": "B"},
    }))
    filings = [
        ("0000000001-18-000001", "8-K", "2018-03-09", "20180309160500"),   # EST, after the close
        ("0000000001-18-000002", "10-Q", "2018-03-12", "20180312160500"),  # EDT, after the close
        ("0000000001-18-000003", "4", "2018-03-12", "20180312100000"),     # not a requested form
        ("0000000001-09-000009", "10-K", "2009-12-01", None),              # before the task: never fetched
    ]
    (cache / "CIK0000000001.json").write_text(json.dumps({"filings": {"files": [], "recent": {
        "form": [f[1] for f in filings],
        "filingDate": [f[2] for f in filings],
        "accessionNumber": [f[0] for f in filings],
        "primaryDocument": ["doc.htm"] * len(filings),
        "primaryDocDescription": ["Current report"] * len(filings),
        "acceptanceDateTime": ["2018-03-09T12:00:00.000Z"] * len(filings),
    }}}))
    for accession, _, _, stamp in filings:
        if stamp:
            (cache / "headers" / (accession + ".html")).write_text(
                "<PRE>&lt;ACCEPTANCE-DATETIME&gt;{}\nACCESSION NUMBER:\t\t{}\n</PRE>".format(stamp, accession)
            )


def test_edgar_filings_use_header_acceptance_times_offline(tmp_path, monkeypatch, data):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test test@example.com")
    monkeypatch.setattr(info_module.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    edgar_cache(tmp_path)
    task = dataclasses.replace(TASK, info_sources=({"source": "sec-edgar"},))
    panel = load_task_info(task)
    assert visible(panel, "2018-03-09") == []
    assert visible(panel, "2018-03-12") == ["0000000001-18-000001"]
    assert visible(panel, "2018-03-13") == ["0000000001-18-000001", "0000000001-18-000002"]
    first = panel.at_session("2018-03-13").items()[0]
    assert first.published_at == eastern("2018-03-09 16:05")
    assert first.title == "8-K Current report" and first.tickers == ("AAA",) and first.text == ""

    class Recorder(Policy):
        stochastic = False

        def __init__(self):
            self.seen = {}

        def act(self, view):
            self.seen[str(view.date)] = [i.id[-1] for i in view.info.items(ticker="AAA")]

    recorder = Recorder()
    runner._run(task, recorder, data, data.index_of("2018-03-09"), None, None, None, panel)
    assert recorder.seen["2018-03-08"] == []
    assert recorder.seen["2018-03-09"] == []
    assert recorder.seen["2018-03-12"] == ["1"]
    assert recorder.seen["2018-03-13"] == ["1", "2"]


def test_edgar_requires_a_contact_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    task = dataclasses.replace(TASK, info_sources=({"source": "sec-edgar"},))
    with pytest.raises(ValueError, match="SEC_EDGAR_USER_AGENT"):
        load_task_info(task)


def test_edgar_header_without_an_acceptance_time_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test test@example.com")
    edgar_cache(tmp_path)
    (tmp_path / "cache" / "info" / "sec-edgar" / "headers" / "0000000001-18-000001.html").write_text("<PRE></PRE>")
    task = dataclasses.replace(TASK, info_sources=({"source": "sec-edgar"},))
    with pytest.raises(ValueError, match="no acceptance time"):
        load_task_info(task)


# --- records and cutoffs ---------------------------------------------------------------


class Conc(Policy):
    stochastic = False

    def act(self, view):
        weights = np.zeros(len(view.tickers))
        weights[0] = 1.0
        return weights


def run_record(submission, tmp_path, data):
    path = runner.evaluate(TASK, submission, data=data, seeds=[0], out_root=tmp_path / "out")
    return json.loads(path.read_text())


def test_record_hashes_the_info_panel_only_when_sources_are_declared(tmp_path, data):
    plain = make_submission(tmp_path / "subs", "plain", "test_bench_info:Conc")
    assert {"info_sha256", "contamination"}.isdisjoint(run_record(plain, tmp_path, data))
    informed = make_submission(
        tmp_path / "subs", "informed", "test_bench_info:Conc",
        info_sources=[{"source": "jsonl", "path": "news.jsonl"}],
    )
    write_jsonl(informed.directory / "news.jsonl", [news("n1", "2018-03-12T14:00:00Z")])
    record = run_record(informed, tmp_path, data)
    assert record["info_sha256"] == load_task_info(TASK, informed).sha256()
    write_jsonl(informed.directory / "news.jsonl", [news("n1", "2018-03-12T14:00:00Z", text="edited")])
    assert run_record(informed, tmp_path, data)["info_sha256"] != record["info_sha256"]


def scored_submission(tmp_path, cutoffs, **spec):
    directory = tmp_path / "subs" / "scored"
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "news.jsonl", [
        news("n1", "2017-06-01T14:00:00Z", fields={"score": {"value": 0.7, "model": "scorer-1"}}),
    ])
    return make_submission(
        tmp_path / "subs", "scored", "test_bench_info:Conc",
        info_sources=[{"source": "jsonl", "path": "news.jsonl"}], model_cutoffs=cutoffs, **spec,
    )


def cutoff(date):
    return [{"model": "scorer-1", "training_cutoff": date, "source_url": "https://example.com/model"}]


def test_model_produced_fields_need_a_declared_cutoff(tmp_path, data):
    submission = scored_submission(tmp_path, [])
    with pytest.raises(ValueError, match="scorer-1"):
        runner.evaluate(TASK, submission, data=data, seeds=[0], out_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_llm_submissions_need_cutoffs_for_historical_records(tmp_path, data):
    submission = make_submission(tmp_path / "subs", "agent", "test_bench_info:Conc", kind="llm")
    with pytest.raises(ValueError, match="model_cutoffs"):
        runner.evaluate(TASK, submission, data=data, seeds=[0], out_root=tmp_path / "out")


def test_test_window_before_the_cutoff_is_refused_unless_labelled(tmp_path, data):
    unlabelled = scored_submission(tmp_path, cutoff("2018-06-30"))
    with pytest.raises(ValueError, match="contamination"):
        runner.evaluate(TASK, unlabelled, data=data, seeds=[0], out_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()

    record = run_record(scored_submission(tmp_path, cutoff("2018-06-30"), contamination="possible"), tmp_path, data)
    assert record["contamination"] == "possible"
    assert "contamination: possible" in leaderboard.render_markdown(leaderboard.build(tmp_path / "out"))


def test_cutoff_before_the_test_window_needs_no_label(tmp_path, data):
    assert "contamination" not in run_record(scored_submission(tmp_path, cutoff("2017-12-31")), tmp_path, data)


@pytest.mark.parametrize(
    "entry",
    [
        {"model": "m", "training_cutoff": "2018-6-30", "source_url": "https://example.com"},
        {"model": "m", "training_cutoff": "2018-06-30", "source_url": "ftp://example.com"},
        {"model": "m", "training_cutoff": "2018-06-30"},
    ],
)
def test_cutoff_declarations_are_validated(tmp_path, entry):
    with pytest.raises(ValueError):
        make_submission(tmp_path, "bad", "test_bench_info:Conc", model_cutoffs=[entry])


# --- check-causal ----------------------------------------------------------------------


class ReadsTomorrow(Policy):
    """Long the first asset when tomorrow's close is higher, read through the array base."""

    stochastic = False

    def act(self, view):
        close = view.history.close
        full = close.base if close.base is not None else close
        today = len(view.history) - 1
        weights = np.zeros(len(view.tickers))
        weights[0] = 1.0 if today + 1 < len(full) and full[today + 1, 0] > close[today, 0] else 0.0
        return weights


class ReadsNews(Policy):
    """Weight of the first asset follows the titles of every visible item, so info drives decisions."""

    stochastic = False

    def act(self, view):
        weights = np.zeros(len(view.tickers))
        weights[0] = sum(len(i.title) for i in view.info.items(ticker=view.tickers[0])) % 10 / 10
        return weights


class NoisyUnderSeed(Policy):
    """Random weights from the global generator: causal only when seeded identically."""

    stochastic = True

    def act(self, view):
        weights = np.random.random(len(view.tickers))
        return weights / weights.sum()


def daily_news(data):
    """Per session, one item an hour before the close and one five minutes after it."""
    items = []
    for day in data.dates[data.index_of("2018-01-01"):]:
        close = eastern("{} 16:00".format(day))
        items.append(item("early-{}".format(day), close - timedelta(hours=1)))
        items.append(item("late-{}".format(day), close + timedelta(minutes=5)))
    return InfoPanel(items)


def test_leaky_price_reader_fails_truncation_and_future_poisoning(tmp_path, data):
    submission = make_submission(tmp_path, "leaky", "test_bench_info:ReadsTomorrow")
    with pytest.raises(ValueError) as failure:
        check_causal(TASK, submission, data=data)
    assert "truncation" in str(failure.value) and "future poisoning" in str(failure.value)


def test_info_visible_at_the_close_changes_decisions_without_breaking_causality(tmp_path, data):
    submission = make_submission(tmp_path, "news", "test_bench_info:ReadsNews")
    assert check_causal(TASK, submission, data=data, info=daily_news(data))


@pytest.mark.parametrize(
    "lookahead, failing",
    [(timedelta(hours=1), ("truncation", "future poisoning")), (timedelta(seconds=1), ("future poisoning",))],
    ids=["an hour ahead", "inclusive close"],
)
def test_an_info_view_that_shows_later_items_fails(tmp_path, data, monkeypatch, lookahead, failing):
    submission = make_submission(tmp_path, "news", "test_bench_info:ReadsNews")
    monkeypatch.setattr(
        InfoPanel, "at_session",
        lambda self, day, crypto=False: self.before(session_close(day, crypto=crypto) + lookahead),
    )
    with pytest.raises(ValueError) as failure:
        check_causal(TASK, submission, data=data, info=daily_news(data))
    assert all(check in str(failure.value) for check in failing)


def test_the_check_compares_stochastic_policies_under_a_fixed_seed(tmp_path, data):
    submission = make_submission(tmp_path, "noisy", "test_bench_info:NoisyUnderSeed")
    assert check_causal(TASK, submission, data=data, seed=3)


def test_every_built_in_policy_is_causal_on_synthetic(tmp_path, data):
    baselines = [s for s in discover().values() if s.kind == "baseline"]
    assert len(baselines) >= 8
    for submission in baselines:
        assert check_causal(TASK, submission, data=data), submission.id
    learned = [
        make_submission(tmp_path, "ppo", "rl.bench.policies.sb3:SB3Policy", kind="rl", params={
            "algo": "ppo", "timesteps": 128, "episode_length": 100, "net_arch": [8],
            "algo_kwargs": {"n_steps": 64, "batch_size": 32},
        }),
        make_submission(tmp_path, "lstm", "rl.bench.policies.forecast:LSTMForecastPolicy", kind="forecast", params={
            "window": 10, "hidden_size": 4, "num_layers": 1, "epochs": 1,
        }),
    ]
    for submission in learned:
        assert check_causal(TASK, submission, data=data, seed=1), submission.id


def test_cli_check_causal_reports_the_verdict(tmp_path, capsys):
    make_submission(tmp_path, "tomorrow", "test_bench_info:ReadsTomorrow")
    make_submission(tmp_path, "cash", "rl.bench.policies.baselines:Cash", kind="baseline")
    argv = ["bench", "check-causal", "--task", "synthetic", "--submissions-root", str(tmp_path)]
    assert cli.main([*argv, "--submission", "cash"]) == 0
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*argv, "--submission", "tomorrow"])
    assert exit_info.value.code == 1
    assert "not causal" in capsys.readouterr().err


def test_check_needs_pre_test_history(tmp_path, data):
    submission = make_submission(tmp_path, "cash", "rl.bench.policies.baselines:Cash", kind="baseline")
    with pytest.raises(ValueError, match="pre-test history"):
        check_causal(TASK, submission, data=data.between("2018-01-01", None))
