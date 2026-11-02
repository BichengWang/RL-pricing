"""LLM score features, the R5 arm declarations and the score converter."""

import dataclasses
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from rl.bench import leaderboard, llm, runner
from rl.bench.causal import check_causal
from rl.bench.data import load_task_data
from rl.bench.gym_env import PortfolioEnv, observation
from rl.bench.info import InfoItem, InfoPanel, InfoView, session_close
from rl.bench.market import MarketView
from rl.bench.policies.cvar_ppo import cvar_penalty
from rl.bench.submission import Submission, discover
from rl.bench.tasks import TASKS, get_task

REPO = Path(__file__).resolve().parents[1]
TASK = get_task("synthetic")
NEW_YORK = ZoneInfo("America/New_York")
SCORES = REPO / "reports/r5-finrl-deepseek/scores.py"
SPEC = importlib.util.spec_from_file_location("r5_scores", SCORES)
scores = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scores  # dataclasses resolve the defining module through sys.modules
SPEC.loader.exec_module(scores)

ARM_IDS = tuple("r5-" + arm for arm in (
    "p0", "p10", "p1", "p01", "c0", "c10", "c1", "c01",
    "p10-l", "c10-l", "p10-s", "c10-s", "p10-d", "c10-d",
))


def make_item(item_id, published_at, ticker="AAA", sentiment=5.0, risk=1.0, model="m-1"):
    fields = {}
    if sentiment is not None:
        fields["llm_sentiment"] = {"value": sentiment, "model": model}
    if risk is not None:
        fields["llm_risk"] = {"value": risk, "model": model}
    return InfoItem(
        id=item_id, source="test", tickers=(ticker,), published_at=published_at, title=item_id,
        text="", text_sha256=hashlib.sha256(b"").hexdigest(), fields=fields,
    )


@pytest.fixture(scope="module")
def market():
    return load_task_data(TASK, end="2020-01-01")


def fixture_panel(data):
    """A visible score before every close and a different one after it."""
    items = []
    for index, day in enumerate(data.dates):
        close = session_close(day)
        for column, ticker in enumerate(data.tickers):
            score = 1 + (index + column) % 5
            items.append(make_item("early-{}-{}".format(day, ticker), close - timedelta(hours=2),
                                   ticker, sentiment=score, risk=6 - score))
            items.append(make_item("late-{}-{}".format(day, ticker), close + timedelta(hours=2),
                                   ticker, sentiment=6 - score, risk=score))
    return InfoPanel(items)


# --- task and arm declarations ---------------------------------------------------------


def test_us_ndx80_is_the_registered_re_evaluation_task():
    task = get_task("us-ndx80")
    assert len(task.tickers) == 80 and len(set(task.tickers)) == 80
    assert task.source == "yahoo"
    assert (task.start, task.validation_start, task.test_start, task.test_end) == (
        "2013-01-01", "2018-01-01", "2019-01-01", "2023-12-29")
    assert task.transaction_cost == 0.001 and task.periods_per_year == 252
    assert task.live is False
    assert "survivorship" in task.description and "re-evaluation" in task.description
    assert "ANSS" not in task.tickers and "WBA" not in task.tickers
    assert leaderboard._role("us-ndx80") == "reevaluation"
    assert TASKS["us-ndx80"].tickers[0] == "AAPL"


def test_the_fourteen_arms_are_declared_for_us_ndx80_only():
    found = discover(REPO / "submissions")
    assert set(ARM_IDS) <= set(found)
    for arm in ARM_IDS:
        submission = found[arm]
        assert submission.tasks == ("us-ndx80",)
        assert submission.ledger == ()
        assert submission.trials == 14
        assert submission.applies_to("us-ndx80") and not submission.applies_to("us-sectors")
        with pytest.raises(ValueError, match="declared for"):
            runner.evaluate(get_task("us-sectors"), submission, seeds=[0], out_root="/tmp/never")
    score_arms = [found[arm] for arm in ARM_IDS if found[arm].info_sources]
    assert len(score_arms) == 12
    for submission in score_arms:
        assert submission.contamination == "possible"
        assert submission.model_cutoffs
        assert submission.params["llm_strength"] in (0.1, 0.01, 0.001)
    assert found["r5-p0"].params.get("llm_strength") is None
    assert found["r5-p10-s"].info_sources[0]["path"].endswith("deepseek-lagged-placebo.jsonl")
    assert found["r5-p10-d"].info_sources[0]["path"].endswith("deepseek-same-day.jsonl")


def test_a_submission_may_only_name_existing_tasks(tmp_path):
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "submission.json").write_text(json.dumps({
        "name": "bad", "kind": "other", "policy": "rl.bench.policies.baselines:Cash",
        "tasks": ["no-such-task"],
    }))
    with pytest.raises(ValueError, match="Unknown submission tasks"):
        Submission.load(directory)


def test_existing_submissions_keep_their_fingerprints():
    # "tasks" is only written into the fingerprint when declared, so every
    # record made before this change stays current.
    assert discover(REPO / "submissions")["ppo"].fingerprint() == (
        "a58fa5856958ac5604f227e38cee169801b0476f6d6f4fcf99288600a1a2b1b8")


def test_source_hash_anchors_text_that_is_not_retained():
    item = make_item("one", "2023-12-16T23:59:59+00:00")
    anchored = dataclasses.replace(item, source_sha256="a" * 64)
    assert anchored.to_dict()["source_sha256"] == "a" * 64
    restored = InfoItem(**json.loads(json.dumps(anchored.to_dict())))
    assert restored == anchored
    with pytest.raises(ValueError, match="source_sha256"):
        dataclasses.replace(item, source_sha256="not-a-digest")


# --- point-in-time features ------------------------------------------------------------


def test_scores_are_visible_only_strictly_before_the_decision_close():
    close = session_close("2018-03-12")
    panel = InfoPanel([
        make_item("before", close - timedelta(seconds=1), sentiment=5.0, risk=1.0),
        make_item("at", close, sentiment=1.0, risk=5.0),
    ])
    scores_at = llm.score_panel(panel, ["2018-03-12"], ["AAA"])
    assert scores_at[0, 0].tolist() == [5.0, 1.0]
    view = panel.at_session("2018-03-12")
    state = llm.empty_scores(1)
    llm.update_scores(state, view, ["AAA"])
    assert state[0].tolist() == [5.0, 1.0]
    assert [item.id for item in view.items()] == ["before"]


def test_missing_scores_are_per_session_and_not_carried_forward():
    panel = InfoPanel([make_item("one", "2018-03-12T14:00:00+00:00", sentiment=4.0, risk=None)])
    values = llm.score_panel(panel, ["2018-03-12", "2018-03-13"], ["AAA", "BBB"])
    assert values[0, 0].tolist() == [4.0, 3.0]
    assert values[1, 0].tolist() == [0.0, 3.0]  # no score in (Tuesday close, Wednesday close)
    assert values[:, 1].tolist() == [[0.0, 3.0], [0.0, 3.0]]


def test_a_score_applies_only_to_the_session_the_alignment_assigns_it():
    lagged = InfoPanel([make_item("day", "2018-03-12T23:59:59+00:00", sentiment=5.0, risk=1.0)])
    values = llm.score_panel(lagged, ["2018-03-12", "2018-03-13", "2018-03-14"], ["AAA"])
    assert values[:, 0].tolist() == [[0.0, 3.0], [5.0, 1.0], [0.0, 3.0]]
    same_day = InfoPanel([make_item("day", "2018-03-12T00:00:00+00:00", sentiment=5.0, risk=1.0)])
    values = llm.score_panel(same_day, ["2018-03-12", "2018-03-13", "2018-03-14"], ["AAA"])
    assert values[:, 0].tolist() == [[5.0, 1.0], [0.0, 3.0], [0.0, 3.0]]


def test_crypto_sessions_use_their_own_close():
    panel = InfoPanel([make_item("day", "2023-05-01T23:59:59+00:00", sentiment=5.0, risk=1.0)])
    us = llm.score_panel(panel, ["2023-05-01", "2023-05-02"], ["AAA"])
    utc = llm.score_panel(panel, ["2023-05-01", "2023-05-02"], ["AAA"], crypto=True)
    assert us[:, 0].tolist() == [[0.0, 3.0], [5.0, 1.0]]
    assert utc[:, 0].tolist() == [[5.0, 1.0], [0.0, 3.0]]


def test_the_latest_visible_item_wins_and_earlier_items_are_never_reread():
    panel = InfoPanel([
        make_item("old", "2018-03-12T10:00:00+00:00", sentiment=2.0, risk=4.0),
        make_item("new", "2018-03-12T15:00:00+00:00", sentiment=5.0, risk=1.0),
    ])
    view = panel.at_session("2018-03-13")
    state = llm.empty_scores(1)
    llm.update_scores(state, view, ["AAA"], start=1)
    assert state[0].tolist() == [5.0, 1.0]
    llm.update_scores(state, view, ["AAA"], start=len(view))
    assert state[0].tolist() == [5.0, 1.0]
    with pytest.raises(ValueError, match="start"):
        view.items(start=len(view) + 1)


# --- the two feature uses --------------------------------------------------------------


def test_sentiment_scales_the_change_it_agrees_or_disagrees_with():
    current = np.array([0.4, 0.4, 0.2])
    target = np.array([0.5, 0.0, 0.2])
    matched = llm.apply_sentiment(target, current, np.array([5.0, 1.0, 3.0]), 0.1)
    np.testing.assert_allclose(matched, [0.51, 0.0, 0.2])
    opposed = llm.apply_sentiment(target, current, np.array([1.0, 5.0, 3.0]), 0.1)
    np.testing.assert_allclose(opposed, [0.49, 0.04, 0.2])
    half = llm.apply_sentiment(target, current, np.array([4.0, 2.0, 3.0]), 0.1)
    np.testing.assert_allclose(half, [0.505, 0.0, 0.2])  # 2 with a sale is half-matched
    neutral = llm.apply_sentiment(target, current, np.array([3.0, 0.0, 3.0]), 0.1)
    np.testing.assert_allclose(neutral, target)
    # A sale larger than the holding is clipped, and a total above one is scaled back.
    sold = llm.apply_sentiment(np.array([0.0]), np.array([0.5]), np.array([1.0]), 0.5)
    assert sold.tolist() == [0.0]
    full = llm.apply_sentiment(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([5.0, 5.0]), 0.2)
    assert full.sum() == pytest.approx(1.0)


def test_risk_multiplier_is_holdings_weighted_over_invested_assets():
    weights = np.array([0.5, 0.25, 0.0])
    risk = np.array([5.0, 1.0, 1.0])
    assert llm.risk_factors(weights, risk, 0.1) == pytest.approx((0.5 * 1.1 + 0.25 * 0.9) / 0.75)
    assert llm.risk_factors(np.zeros(3), risk, 0.1) == pytest.approx(1.0)
    assert llm.risk_factors(np.array([0.6]), np.array([3.0]), 0.1) == pytest.approx(1.0)


def test_cvar_penalty_multiplies_the_estimated_return_by_the_risk_factor():
    final = np.array([-0.4, 0.1, 0.2, 0.3])
    rewards = np.tile(final / 4, (4, 1))
    values = rewards * np.arange(4, 0, -1)[:, None]
    starts = np.zeros((4, 4))
    factors = np.ones((4, 4))
    factors[:, 0] = 2.0
    base, _ = cvar_penalty(rewards, values, starts, np.zeros(4), 0.5, 1.0)
    scaled, _ = cvar_penalty(rewards, values, starts, np.zeros(4), 0.5, 1.0, factors)
    assert scaled[3, 0] > base[3, 0]
    with pytest.raises(ValueError, match="factors"):
        cvar_penalty(rewards, values, starts, np.zeros(4), 0.5, 1.0, np.ones(3))


# --- policy wiring ---------------------------------------------------------------------


def test_llm_observation_appends_two_columns_per_asset(market):
    features = np.ones((3, 11))
    plain = observation(features, np.zeros(3))
    enriched = observation(features, np.zeros(3), np.ones((3, 2)))
    assert plain.size == 3 * 11 + 4 and enriched.size == 3 * 13 + 4
    np.testing.assert_array_equal(enriched[: 3 * 13].reshape(3, 13)[:, 11:], np.ones((3, 2)))


def test_the_environment_trades_the_sentiment_scaled_target(market):
    data = market.between("2015-06-01", "2016-01-01")
    extra = np.zeros((len(data), len(data.tickers), 2))
    extra[..., 0] = 5.0
    extra[..., 1] = 3.0
    env = PortfolioEnv(data, 0.001, episode_length=10, first_session=1, extra=extra, sentiment_strength=0.1)
    env.reset(seed=0)
    # Buy the first asset; the matched score scales the change by 1.1.
    action = np.array([5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    previous = env._account.weights(env._decision).copy()
    env.step(action)
    target = env._account.weights(env._decision)
    assert target[0] > previous[0] + 0.5 and target.sum() <= 1.0 + 1e-9
    with pytest.raises(ValueError, match="sentiment_strength"):
        PortfolioEnv(data, 0.001, extra=None, sentiment_strength=0.1)


def tiny(submission):
    params = dict(submission.params)
    params.update(timesteps=32, episode_length=20, net_arch=[8], eval_every=16)
    params["algo_kwargs"] = {"n_steps": 32, "batch_size": 16}
    return dataclasses.replace(submission, params=params)


def fitted_actor(submission, data, info):
    policy = runner.build_policy(submission, TASK)
    history = data.between(None, "2017-01-01")
    runner.fit_policy(policy, history, np.datetime64("2016-06-01"), 3, info)
    return policy


def test_fit_cannot_see_an_item_published_after_its_history(market, tmp_path):
    submission = tiny(discover(REPO / "submissions")["r5-p10"])
    history = market.between(None, "2017-01-01")
    close = session_close(history.dates[-1])
    panel = fixture_panel(market)
    later = InfoPanel([
        make_item("future-" + ticker, close + timedelta(seconds=1), ticker=ticker, sentiment=1.0, risk=5.0)
        for ticker in market.tickers
    ])
    first = fitted_actor(submission, market, panel)
    second = fitted_actor(submission, market, InfoPanel([*panel.items, *later.items]))
    first.save(tmp_path / "first")
    second.save(tmp_path / "second")
    (actor_a,) = [path for path in (tmp_path / "first").iterdir() if path.suffix == ".npz"]
    (actor_b,) = [path for path in (tmp_path / "second").iterdir() if path.suffix == ".npz"]
    assert np.array_equal(np.load(actor_a)["w0"], np.load(actor_b)["w0"])


def test_fit_policy_cuts_the_panel_at_the_fitted_close(market):
    from rl.bench.policy import Policy

    class RecordsInfo(Policy):
        def __init__(self):
            self.info = None
            self.crypto = None

        def fit(self, history, validation_start, seed, info=None, crypto=None):
            self.info, self.crypto = info, crypto

        def act(self, view):
            return np.zeros(len(view.tickers))

    history = market.between(None, "2017-01-01")
    close = session_close(history.dates[-1], crypto=True)
    panel = fixture_panel(market)
    assert any(item.published_at >= close for item in panel.items)
    policy = RecordsInfo()
    runner.fit_policy(policy, history, np.datetime64("2016-06-01"), 0, panel, crypto=True)
    assert policy.crypto is True
    assert all(item.published_at < close for item in policy.info.items)


def test_act_cannot_see_an_item_published_after_the_decision_close(market):
    submission = tiny(discover(REPO / "submissions")["r5-p10"])
    policy = fitted_actor(submission, market, fixture_panel(market))
    day = "2017-06-01"
    close = session_close(day)
    base = make_item("before", close - timedelta(hours=1), ticker="AAA", sentiment=5.0, risk=1.0)
    after = make_item("after", close, ticker="AAA", sentiment=1.0, risk=5.0)
    panel = InfoPanel([base, after])
    view = panel.at_session(day)
    assert [item.id for item in view.items()] == ["before"]
    day_index = market.index_of(day)
    history = market.rows(0, day_index + 1)
    policy.reset()
    with_scores = policy.act(MarketView(history, np.zeros(5), view))
    policy.reset()
    without = policy.act(MarketView(history, np.zeros(5), InfoView()))
    assert not np.array_equal(with_scores, without)  # the visible score does change the action
    assert "after" not in [item.id for item in view.items()]


@pytest.mark.parametrize("arm", ARM_IDS)
def test_every_registered_arm_passes_the_lookahead_check(arm, market):
    submission = tiny(discover(REPO / "submissions")[arm])
    info = fixture_panel(market) if submission.info_sources else InfoPanel()
    assert check_causal(TASK, submission, data=market, info=info, seed=0)


def test_cli_check_causal_passes_for_every_arm_with_a_jsonl_fixture(tmp_path, market):
    root = tmp_path / "subs"
    root.mkdir()
    fixture = tmp_path / "scores.jsonl"
    fixture.write_text("".join(json.dumps(item.to_dict()) + "\n" for item in fixture_panel(market).items))
    from rl import cli

    for arm in ARM_IDS:
        submission = tiny(discover(REPO / "submissions")[arm])
        spec = {
            "name": submission.name, "kind": submission.kind, "policy": submission.policy,
            "params": submission.params, "trials": submission.trials,
            "authors": list(submission.authors), "url": submission.url,
            "description": submission.description, "tasks": list(submission.tasks),
            "model_cutoffs": [dict(entry) for entry in submission.model_cutoffs],
        }
        if submission.info_sources:
            spec["info_sources"] = [{"source": "jsonl", "path": "scores.jsonl"}]
            spec["contamination"] = submission.contamination
        directory = root / arm
        directory.mkdir()
        (directory / "submission.json").write_text(json.dumps(spec))
        (directory / "scores.jsonl").write_bytes(fixture.read_bytes())
        argv = ["bench", "check-causal", "--task", "synthetic", "--submission", arm,
                "--submissions-root", str(root)]
        assert cli.main(argv) == 0, arm


# --- the score converter ---------------------------------------------------------------

HEADER = "Unnamed: 0.1,Unnamed: 0,Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary,{column}\n"


def write_csv(path, column, rows):
    lines = [HEADER.format(column=column)]
    for index, (day, ticker, title, score, article) in enumerate(rows):
        lines.append('{},{},"{}","{}",{},,,,"{}",,"{}"\n'.format(
            index, index, day, title, ticker, article, score))
    path.write_text("".join(lines))


def fixture_model(tmp_path, monkeypatch):
    sentiment = tmp_path / "sentiment.csv"
    risk = tmp_path / "risk.csv"
    stamp = "2023-12-16 00:00:00 UTC"
    write_csv(sentiment, "sentiment_deepseek", [
        (stamp, "AAPL", "Good news", 5, "apple article"),
        (stamp, "MSFT", "Mixed", 4, "msft article"),
        ("2023-12-17 04:00:00 UTC", "AAPL", "Only sentiment", 2, "third"),
        ("2023-12-18 00:00:00 UTC", "AAPL", "Twice", 3, "fourth"),
        ("2023-12-18 00:00:00 UTC", "AAPL", "Another", 3, "fifth"),
    ])
    write_csv(risk, "risk_deepseek", [
        (stamp, "AAPL", "Good news", 1, "apple article"),
        (stamp, "MSFT", "Mixed", 3, "msft article"),
        ("2023-12-18 00:00:00 UTC", "AAPL", "Twice", 2, "fourth"),
        ("2023-12-18 00:00:00 UTC", "AAPL", "Another", 1, "fifth"),
    ])
    monkeypatch.setattr(scores, "download", lambda remote, cache, refresh=False: sentiment
                        if remote is scores.SENTIMENT_DEEPSEEK else risk)
    return sentiment, risk


def test_the_converter_joins_on_date_ticker_and_title(tmp_path, monkeypatch):
    fixture_model(tmp_path, monkeypatch)
    rows, counts = scores.joined_rows("deepseek", tmp_path, ("AAPL", "MSFT"))
    assert counts["sentiment_rows"] == 5 and counts["risk_rows"] == 4
    assert counts["joined"] == 2
    assert counts["dropped_sentiment"] == 1  # "Only sentiment" is not in the risk file
    assert counts["dropped_ambiguous"] == 2  # 2023-12-18 has two joined articles
    joined = {(row[1], row[2]): (row[3], row[4]) for row in rows}
    assert joined[("AAPL", "Good news")] == (5.0, 1.0)
    assert joined[("MSFT", "Mixed")] == (4.0, 3.0)
    items = list(scores._items("deepseek", scores.aligned_rows(rows, "lagged")))
    apple = next(item for item in items if item.tickers == ("AAPL",))
    assert apple.published_at == datetime(2023, 12, 16, 23, 59, 59, tzinfo=scores.UTC)
    assert apple.fields["llm_sentiment"].model == "deepseek-ai/DeepSeek-V3"
    assert apple.source_sha256 == hashlib.sha256(b"apple article").hexdigest()
    assert apple.text == "" and apple.text_sha256 == hashlib.sha256(b"").hexdigest()
    same_day = list(scores._items("deepseek", scores.aligned_rows(rows, "same-day")))
    assert same_day[0].published_at == datetime(2023, 12, 16, 0, 0, tzinfo=scores.UTC)


def test_a_tampered_score_file_is_rejected(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_bytes(b"not the registered file")
    wrong = dataclasses.replace(scores.SENTIMENT_DEEPSEEK, size=path.stat().st_size, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        scores.verify(path, wrong)
    correct = dataclasses.replace(wrong, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert scores.verify(path, correct) == path
    with pytest.raises(ValueError, match="bytes"):
        scores.verify(path, dataclasses.replace(correct, size=path.stat().st_size + 1))
    with pytest.raises(FileNotFoundError):
        scores.verify(tmp_path / "missing.csv", correct)


def test_a_download_with_the_wrong_hash_is_not_kept(tmp_path, monkeypatch):
    class Response:
        def __init__(self, content):
            self.content = content
            self.offset = 0

        def read(self, size=-1):
            block = self.content[self.offset:self.offset + size]
            self.offset += len(block)
            return block

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    content = b"a,b\n1,2\n"
    remote = dataclasses.replace(scores.SENTIMENT_DEEPSEEK, size=len(content), sha256="f" * 64)
    monkeypatch.setattr(scores.urllib.request, "urlopen", lambda url, timeout: Response(content))
    with pytest.raises(ValueError, match="downloaded SHA-256"):
        scores.download(remote, tmp_path / "cache")
    assert not (tmp_path / "cache" / remote.name).exists()
    assert not (tmp_path / "cache" / (remote.name + ".part")).exists()


def test_placebo_keeps_weekend_rows_and_shifts_per_session():
    tickers = ("AAA", "BBB")
    sessions = np.array([
        day for day in np.arange(np.datetime64("2020-01-01"), np.datetime64("2021-04-01"), dtype="datetime64[D]")
        if np.is_busday(day)
    ])
    first = scores.placebo_offsets(len(sessions), tickers)
    assert first == scores.placebo_offsets(len(sessions), tickers)
    assert all(126 <= offset <= len(sessions) - 126 for offset in first.values())
    # Saturday, Sunday and Monday publications; the two weekend rows map to Monday.
    rows = [
        (datetime(2020, 6, 6, 23, 59, 59, tzinfo=scores.UTC), "AAA", "sat", 5.0, 1.0, "a"),
        (datetime(2020, 6, 7, 23, 59, 59, tzinfo=scores.UTC), "AAA", "sun", 4.0, 2.0, "b"),
        (datetime(2020, 6, 8, 23, 59, 59, tzinfo=scores.UTC), "AAA", "mon", 3.0, 3.0, "c"),
        (datetime(2020, 6, 6, 23, 59, 59, tzinfo=scores.UTC), "BBB", "sat", 2.0, 4.0, "d"),
    ]
    counts = {}
    shifted = scores.placebo_rows(rows, sessions, tickers, counts)
    assert counts["placebo_deduped"] == 1 and counts.get("placebo_dropped_after", 0) == 0
    by_key = {(row[1], row[2]): row for row in shifted}
    assert ("AAA", "sat") not in by_key  # the later Sunday row wins Monday
    expected_positions = {}
    for row in (("sun", "2020-06-08"), ("mon", "2020-06-09")):
        session_position = int(np.searchsorted(sessions, np.datetime64(row[1], "D")))
        moved = (session_position + first["AAA"]) % len(sessions)
        expected_positions[row[0]] = session_close(sessions[moved]) - timedelta(seconds=1)
    assert by_key[("AAA", "sun")][0] == expected_positions["sun"]
    assert by_key[("AAA", "mon")][0] == expected_positions["mon"]
    bbb_position = int(np.searchsorted(sessions, np.datetime64("2020-06-08", "D")))
    bbb_moved = (bbb_position + first["BBB"]) % len(sessions)
    assert by_key[("BBB", "sat")][0] == session_close(sessions[bbb_moved]) - timedelta(seconds=1)
    with pytest.raises(ValueError, match="more than"):
        scores.placebo_offsets(200, tickers)


def test_build_writes_to_the_given_output_and_forwards_refresh(tmp_path, monkeypatch):
    sentiment, risk = fixture_model(tmp_path, monkeypatch)
    calls = []

    def download(remote, cache, refresh=False):
        calls.append(refresh)
        return sentiment if remote is scores.SENTIMENT_DEEPSEEK else risk

    monkeypatch.setattr(scores, "download", download)
    output = tmp_path / "out" / "deepseek-lagged.jsonl"
    manifest = scores.build(
        "deepseek-lagged", ("AAPL", "MSFT"), [], cache=tmp_path / "raw", output=output, refresh=True)
    assert Path(manifest["path"]) == output and output.exists()
    assert manifest["items"] == 2
    assert calls and all(calls)


def test_cvar_risk_factors_clip_actions_to_the_logit_bound():
    from rl.bench.gym_env import LOGIT_BOUND
    from rl.bench.policies.cvar_ppo import CVaRPPO

    model = object.__new__(CVaRPPO)
    model.risk_strength = 0.1
    n_assets, width = 5, 13
    observations = np.zeros((2, 1, n_assets * width + n_assets + 1))
    per_asset = observations[..., : n_assets * width].reshape(2, 1, n_assets, width)
    per_asset[..., 10] = 1.0  # listed
    per_asset[..., 11] = 3.0  # sentiment
    per_asset[..., 12] = 5.0  # risk
    observations[..., -(n_assets + 1):-1] = 0.1
    actions = np.full((2, 1, n_assets + 1), 50.0)
    np.testing.assert_allclose(
        model._risk_factors(observations, actions),
        model._risk_factors(observations, np.full_like(actions, LOGIT_BOUND)),
    )


def test_the_runner_injects_the_session_close_mode():
    submission = discover(REPO / "submissions")["r5-p0"]
    assert runner.policy_params(submission, get_task("crypto-majors"))["crypto"] is True
    assert runner.policy_params(submission, get_task("us-ndx80"))["crypto"] is False
