#!/usr/bin/env python3
"""Convert FinRL-DeepSeek's released news scores to the M2 ``jsonl`` format.

The pre-registration (``reports/r5-finrl-deepseek/preregistration.md`` §2.3,
§5.2) pins four Hugging Face files by revision and SHA-256: one sentiment and
one risk file per model. A row's score is joined to the other file's score on
date, ticker and article title; unjoined, ambiguous and unscored rows are
dropped and counted. Each joined (ticker, day) becomes one
:class:`rl.bench.info.InfoItem` with ``llm_sentiment`` and ``llm_risk`` model
fields, the producing model id, and the article's text only as a SHA-256.

Two alignments are registered: ``lagged`` publishes a day's score at 23:59:59
UTC of that day (visible from the next session) and ``same-day`` keeps the
file's stamp (visible at that day's close). ``placebo`` maps each row to the
session at which that alignment first shows it, reduces the rows to one per
(ticker, session), and circularly shifts each ticker's per-session series
within the task's sessions by an offset drawn once per ticker from
``[126, n - 126]`` with ``numpy.default_rng(0)``; weekend news is therefore
kept, as it is in the real arms.

Downloads and the converted files are cached under ``datasets/``, which is
gitignored: no score data and no converted file is committed. Data are not
redistributed; only hashes and counts appear in a report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from rl.bench.info import session_close

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "datasets" / "r5-finrl-deepseek"
CHUNK = 1 << 20
UTC = timezone.utc
PLACEBO_RANGE = 126


@dataclass(frozen=True)
class Remote:
    repository: str
    revision: str
    name: str
    size: int
    sha256: str
    column: str

    @property
    def url(self) -> str:
        return "https://huggingface.co/datasets/{}/resolve/{}/{}".format(self.repository, self.revision, self.name)


# §2.3 of the pre-registration: pinned revisions and hashes, recomputed 2026-11-02.
SENTIMENT_DEEPSEEK = Remote(
    "benstaf/nasdaq_news_sentiment", "04532f608f416991dea9b9952775102f8e2216e5",
    "sentiment_deepseek_new_cleaned_nasdaq_news_full.csv", 588_735_065,
    "54470e391419ae1f374c1f6b68a936bc036100e36bcf5ea5d8a33dd3c82ac1e9", "sentiment_deepseek",
)
RISK_DEEPSEEK = Remote(
    "benstaf/risk_nasdaq", "1c46dd45482f681e41e3e70a3a353eb6d165e845",
    "risk_deepseek_cleaned_nasdaq_news_full.csv", 588_737_496,
    "33251916177feceebec6f2bec58e81cfbf10fb9115178504cb044498c20c7e73", "risk_deepseek",
)
SENTIMENT_LLAMA = Remote(
    "benstaf/nasdaq_news_sentiment", "04532f608f416991dea9b9952775102f8e2216e5",
    "sentiment_llama_cleaned_nasdaq_news_full.csv", 588_710_540,
    "d75bb8994b90ea0cb51628619e719a26c9d509eba32649429d11b1ceb5ff16ea", "sentiment_llama",
)
RISK_LLAMA = Remote(
    "benstaf/risk_nasdaq", "1c46dd45482f681e41e3e70a3a353eb6d165e845",
    "risk_llama_cleaned_nasdaq_news_full.csv", 588_548_709,
    "73cbe9bbcc41e593e47dbf853132228e104e626bf9df461384049bcb6da40a95", "risk_llama",
)

#: The producing model of each score file. Llama is an inference from the
#: commented model id in the authors' scripts (§2.3); no file names its model.
MODELS = {
    "deepseek": ("deepseek-ai/DeepSeek-V3", SENTIMENT_DEEPSEEK, RISK_DEEPSEEK),
    "llama": ("meta-llama/Llama-3.3-70B-Instruct", SENTIMENT_LLAMA, RISK_LLAMA),
}

#: Registered alignments; ``placebo`` shifts each ticker's series before writing.
VARIANTS = {
    "deepseek-lagged": ("deepseek", "lagged", False),
    "deepseek-same-day": ("deepseek", "same-day", False),
    "llama-lagged": ("llama", "lagged", False),
    "deepseek-lagged-placebo": ("deepseek", "lagged", True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, remote: Remote) -> Path:
    """``path`` if its size and SHA-256 are ``remote``'s; raise otherwise."""
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != remote.size:
        raise ValueError("{} has {} bytes, expected {}".format(path, size, remote.size))
    actual = sha256_file(path)
    if actual != remote.sha256:
        raise ValueError("{} has SHA-256 {}, expected {}".format(path, actual, remote.sha256))
    return path


def download(remote: Remote, cache: Path, *, refresh: bool = False) -> Path:
    """The verified local copy of ``remote``; downloads it once into ``cache``.

    An existing file is verified, never trusted by name: a tampered cache is
    rejected. A download is written beside it and renamed only after its hash
    matches, so an interrupted transfer cannot be mistaken for the file.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / remote.name
    if path.exists() and not refresh:
        return verify(path, remote)
    partial = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(remote.url, timeout=120) as response, open(partial, "wb") as handle:
            for block in iter(lambda: response.read(CHUNK), b""):
                digest.update(block)
                size += len(block)
                handle.write(block)
        if size != remote.size:
            raise ValueError("{} downloaded {} bytes, expected {}".format(remote.name, size, remote.size))
        if digest.hexdigest() != remote.sha256:
            raise ValueError(
                "{} downloaded SHA-256 {}, expected {}".format(remote.name, digest.hexdigest(), remote.sha256)
            )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def _rows(path: Path, remote: Remote, tickers: set[str]):
    """``(stamp, ticker, title, score, article)`` for the task's tickers."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Date", "Article_title", "Stock_symbol", remote.column, "Article"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("{} lacks columns {}".format(path.name, sorted(missing)))
        for row in reader:
            ticker = (row["Stock_symbol"] or "").strip()
            if ticker not in tickers:
                continue
            yield (
                (row["Date"] or "").strip(), ticker, row["Article_title"] or "",
                (row[remote.column] or "").strip(), row["Article"] or "",
            )


def _score(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _stamp(value: str) -> datetime:
    """The file's ``YYYY-MM-DD HH:MM:SS UTC`` stamp as an aware UTC datetime."""
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=UTC)


def _published(day: str, alignment: str) -> datetime:
    date = datetime.strptime(day[:10], "%Y-%m-%d").date()
    if alignment == "lagged":
        return datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=UTC)
    return _stamp(day)


def joined_rows(model_key: str, cache: Path, tickers, *, refresh: bool = False):
    """Joined ``(day, ticker, title, sentiment, risk, article)`` rows and drop counts.

    The two files are joined on ``(date, ticker, title)``. A key repeated within
    one file keeps its first row; a (ticker, day) with more than one joined
    article is ambiguous and dropped, because the registration defines one item
    per (ticker, day).
    """
    _, sentiment_file, risk_file = MODELS[model_key]
    tickers = set(tickers)
    sentiment_path = download(sentiment_file, cache, refresh=refresh)
    risk_path = download(risk_file, cache, refresh=refresh)
    counts = {
        "sentiment_rows": 0, "risk_rows": 0, "joined": 0,
        "dropped_sentiment": 0, "dropped_risk": 0, "dropped_ambiguous": 0,
        "dropped_unscored": 0, "text_mismatch": 0,
    }

    def index(path, remote, count_key):
        rows = {}
        for day, ticker, title, score, article in _rows(path, remote, tickers):
            counts[count_key] += 1
            # The registration joins on the day, not its (hidden) time of day.
            rows.setdefault((day[:10], ticker, title), (day, score, article))
        return rows

    sentiment = index(sentiment_path, sentiment_file, "sentiment_rows")
    risk = index(risk_path, risk_file, "risk_rows")
    grouped = {}
    for key, (day, sentiment_score, article) in sentiment.items():
        if key not in risk:
            counts["dropped_sentiment"] += 1
            continue
        risk_score, risk_article = risk[key][1:]
        if risk_article != article:
            counts["text_mismatch"] += 1
        day_key, ticker, title = key
        grouped.setdefault((day_key, ticker), []).append((day, title, sentiment_score, risk_score, article))
    for key in risk:
        if key not in sentiment:
            counts["dropped_risk"] += 1
    joined = []
    for (day_key, ticker), rows in grouped.items():
        if len(rows) > 1:
            counts["dropped_ambiguous"] += len(rows)
            continue
        day, title, sentiment_score, risk_score, article = rows[0]
        sentiment_value, risk_value = _score(sentiment_score), _score(risk_score)
        if sentiment_value is None and risk_value is None:
            counts["dropped_unscored"] += 1
            continue
        joined.append((day, ticker, title, sentiment_value, risk_value, article))
    counts["joined"] = len(joined)
    joined.sort(key=lambda row: (row[0], row[1]))
    return joined, counts


def aligned_rows(rows, alignment: str):
    """Rows with their publication instant under ``alignment`` first."""
    return [(_published(row[0], alignment), *row[1:]) for row in rows]


def _items(model_key: str, rows):
    from rl.bench.info import InfoItem

    model, sentiment_file, risk_file = MODELS[model_key]
    source = "fnspid-{}".format(model_key)
    empty = hashlib.sha256(b"").hexdigest()
    for published_at, ticker, title, sentiment, risk, article in rows:
        fields = {}
        if sentiment is not None:
            fields["llm_sentiment"] = {"value": sentiment, "model": model}
        if risk is not None:
            fields["llm_risk"] = {"value": risk, "model": model}
        yield InfoItem(
            id="{}-{}-{}".format(model_key, published_at.date().isoformat(), ticker),
            source=source,
            tickers=(ticker,),
            published_at=published_at,
            title=title,
            text="",
            text_sha256=empty,
            fields=fields,
            source_sha256=hashlib.sha256(article.encode("utf-8")).hexdigest(),
        )


def placebo_offsets(n_sessions: int, tickers) -> dict[str, int]:
    """One offset per ticker from ``[126, n - 126]``, drawn in ticker order with ``default_rng(0)``."""
    if n_sessions <= 2 * PLACEBO_RANGE:
        raise ValueError("The placebo needs more than {} sessions, got {}".format(2 * PLACEBO_RANGE, n_sessions))
    rng = np.random.default_rng(0)
    return {
        ticker: int(rng.integers(PLACEBO_RANGE, n_sessions - PLACEBO_RANGE + 1))
        for ticker in tickers
    }


def placebo_rows(rows, sessions, tickers, counts: dict):
    """Shift each ticker's per-session score series by its circular offset.

    ``rows`` carry their publication instant. A row belongs to the first
    session whose close is after that instant (the session at which the real
    arm sees it), so weekend news is kept and lands on the next session. If a
    ticker has several rows for one session, the latest publication wins.
    The shifted value is published one second before its new session's close,
    so the same per-session window sees it.
    """
    sessions = [np.datetime64(day, "D") for day in sessions]
    closes = np.array([session_close(day).timestamp() for day in sessions])
    offsets = placebo_offsets(len(sessions), tickers)
    latest = {}
    for row in rows:
        position = int(np.searchsorted(closes, row[0].timestamp(), side="right"))
        if position >= len(sessions):
            counts["placebo_dropped_after"] = counts.get("placebo_dropped_after", 0) + 1
            continue
        key = (row[1], position)
        if key in latest:
            counts["placebo_deduped"] = counts.get("placebo_deduped", 0) + 1
            if latest[key][0] > row[0]:
                continue
        latest[key] = row
    shifted = []
    for (ticker, position), row in latest.items():
        moved = (position + offsets[ticker]) % len(sessions)
        published = session_close(sessions[moved]) - timedelta(seconds=1)
        shifted.append((published, ticker, *row[2:]))
    shifted.sort(key=lambda row: (row[0], row[1]))
    return shifted


def build(variant: str, tickers, sessions, cache=DEFAULT_CACHE, *, output=None, refresh: bool = False) -> dict:
    """Write the jsonl to ``output`` (default ``<cache>/<variant>.jsonl``) and return its manifest."""
    if variant not in VARIANTS:
        raise ValueError("Unknown score variant {!r}; choose from {}".format(variant, sorted(VARIANTS)))
    model_key, alignment, placebo = VARIANTS[variant]
    tickers = tuple(tickers)
    rows, counts = joined_rows(model_key, Path(cache), tickers, refresh=refresh)
    rows = aligned_rows(rows, alignment)
    if placebo:
        rows = placebo_rows(rows, sessions, tickers, counts)
    items = sorted(_items(model_key, rows), key=lambda item: (item.published_at, item.id))
    output = Path(cache) / (variant + ".jsonl") if output is None else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    with open(partial, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
    os.replace(partial, output)
    return {
        "variant": variant,
        "model": MODELS[model_key][0],
        "alignment": alignment,
        "placebo": placebo,
        "path": str(output),
        "items": len(items),
        "sha256": sha256_file(output),
        **counts,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--task", default="us-ndx80")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--refresh", action="store_true", help="download the pinned files again")
    options = parser.parse_args(argv)

    from rl.bench.data import load_task_data
    from rl.bench.tasks import get_task

    task = get_task(options.task)
    data = load_task_data(task, end=task.test_end)
    manifest = build(options.variant, task.tickers, data.dates, options.cache, refresh=options.refresh)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
