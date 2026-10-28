"""Daily market data for benchmark tasks.

:func:`load_task_data` returns a :class:`~rl.bench.market.MarketData` panel
for a :class:`~rl.bench.tasks.Task`:

* ``yahoo``: daily prices and dividends from Yahoo Finance via
  :class:`~rl.marketdata.yahoodownloader.YahooDownloader`, cached per ticker
  unadjusted and adjusted for dividends forward from the task's first
  session. A session's prices therefore depend only on earlier sessions:
  loading more data never changes them, and the same window always hashes
  the same. Yahoo's terms forbid redistribution, so the cache stays local
  (the default ``datasets/`` directory is gitignored) and is never published.
* ``binance``: spot daily klines from the public https://data.binance.vision
  archive. Every downloaded zip is verified against Binance's published
  SHA-256 checksum and cached together with it. Missing files (a symbol
  before its listing or after its delisting) mean the asset is not tradable
  those days.
* ``synthetic``: deterministic random walks, no network.

Dates are calendar days (UTC for crypto). Sessions are the union of every
asset's dates inside ``[task.start, end)``; an asset without a row for a
session is NaN there.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rl.bench.market import FIELDS, MarketData, as_day
from rl.bench.tasks import Task
from rl.marketdata.synthetic import generate_synthetic_data
from rl.marketdata.yahoodownloader import PRICE_COLUMNS, YahooDownloader, total_return_factors

logger = logging.getLogger(__name__)

DATA_ENV = "RL_PRICING_DATA"
DEFAULT_CACHE = "datasets/bench"
BINANCE_BASE = "https://data.binance.vision/data/spot"
# A monthly archive appears a few days after the month ends, and a day's file
# the day after; inside this window a missing file may just not be published.
RECENT = timedelta(days=45)
RETRIES = 5
TIMEOUT = 60
WORKERS = 8
DAY_MS = 86_400_000
# Spot timestamps switched from milliseconds to microseconds on 2025-01-01.
MICROSECONDS_FROM = 10**14
KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
)
FRAME_COLUMNS = ["date", *FIELDS]
# Yahoo cache files hold unadjusted prices and dividends; caches without this
# format number hold Yahoo's adjusted prices and are downloaded again.
YAHOO_CACHE_FORMAT = 2
YAHOO_COLUMNS = [*FRAME_COLUMNS, "dividends"]
# Cached prices must parse to exactly the floats that were written. pandas 3.1
# parses every float that way and deprecates the option that asks for it.
EXACT_FLOATS = (
    {} if tuple(int(part) for part in pd.__version__.split(".")[:2]) >= (3, 1)
    else {"float_precision": "round_trip"}
)


def load_task_data(task: Task, *, end=None, cache_dir=None, refresh: bool = False) -> MarketData:
    """Load the panel for ``task`` covering sessions in ``[task.start, end)``.

    ``end=None`` means everything available: through yesterday (UTC) for the
    downloaded sources and up to ``task.test_end`` for the synthetic one.
    Columns follow ``task.tickers``. ``refresh`` ignores the local cache.
    """
    cache = Path(cache_dir or os.environ.get(DATA_ENV) or DEFAULT_CACHE)
    start = as_day(task.start)
    if task.source == "synthetic":
        stop = as_day(end or task.test_end)
        frames = _synthetic(task, start, stop)
    else:
        today = np.datetime64(datetime.now(timezone.utc).date(), "D")
        stop = min(as_day(end), today) if end is not None else today
        if task.source == "yahoo":
            frames = _yahoo(task, start, stop, cache / "yahoo", refresh)
        elif task.source == "binance":
            frames = _binance(task, start, stop, cache / "binance", refresh)
        else:
            raise ValueError("Unknown data source {!r}".format(task.source))
    data = _assemble(task.tickers, frames, start, stop)
    _log_coverage(task, data)
    return data


# --------------------------------------------------------------------------
# Assembly


def _clean(frame: pd.DataFrame, ticker: str, start, stop) -> pd.DataFrame:
    """One ticker's rows in ``[start, stop)`` with valid prices, indexed by date."""
    frame = frame[FRAME_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    for name in FIELDS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame[(frame["date"] >= pd.Timestamp(str(start))) & (frame["date"] < pd.Timestamp(str(stop)))]
    if frame["date"].duplicated().any():
        raise ValueError("{} has duplicate dates".format(ticker))
    prices = frame[["open", "high", "low", "close"]]
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    if not valid.all():
        logger.warning(
            "%s: dropped %d rows with missing or non-positive prices (first %s)",
            ticker, int((~valid).sum()), frame.loc[~valid, "date"].iloc[0].date(),
        )
    return frame[valid].sort_values("date").set_index("date")


def _assemble(tickers, frames: dict[str, pd.DataFrame], start, stop) -> MarketData:
    """A panel with columns ``tickers`` from per-ticker OHLCV frames."""
    cleaned = {t: _clean(frames[t], t, start, stop) for t in tickers if t in frames}
    if not any(len(f) for f in cleaned.values()):
        raise ValueError("No data between {} and {} for {}".format(start, stop, list(tickers)))
    index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in cleaned.values()))))
    arrays = {name: np.full((len(index), len(tickers)), np.nan) for name in FIELDS}
    for column, ticker in enumerate(tickers):
        frame = cleaned.get(ticker)
        if frame is None or frame.empty:
            continue
        rows = index.get_indexer(frame.index)
        for name in FIELDS:
            arrays[name][rows, column] = frame[name].to_numpy()
    return MarketData(dates=index.to_numpy(dtype="datetime64[D]"), tickers=tuple(tickers), **arrays)


def _log_coverage(task: Task, data: MarketData):
    listed = np.isfinite(data.close)
    first_session, last_session = data.dates[0], data.dates[-1]
    logger.info(
        "%s: %d sessions %s to %s from %s", task.id, len(data), first_session, last_session, task.source
    )
    for column, ticker in enumerate(data.tickers):
        rows = np.flatnonzero(listed[:, column])
        if rows.size == 0:
            logger.warning("%s: %s has no data", task.id, ticker)
            continue
        gaps = int(rows[-1] - rows[0] + 1 - rows.size)
        late = data.dates[rows[0]] != first_session
        early = data.dates[rows[-1]] != last_session
        if gaps or late or early:
            logger.warning(
                "%s: %s listed %s to %s (%d of %d sessions), %d gap sessions inside its span",
                task.id, ticker, data.dates[rows[0]], data.dates[rows[-1]], rows.size, len(data), gaps,
            )


# --------------------------------------------------------------------------
# Synthetic


def _synthetic(task: Task, start, stop) -> dict[str, pd.DataFrame]:
    raw = generate_synthetic_data(list(task.tickers), str(start), str(stop), seed=0)
    if "high" not in raw:
        raw["high"] = raw[["open", "close"]].max(axis=1)
    if "low" not in raw:
        raw["low"] = raw[["open", "close"]].min(axis=1)
    return {tic: frame for tic, frame in raw.groupby("tic")}


# --------------------------------------------------------------------------
# Yahoo


def _write_atomic(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    temp.write_bytes(payload)
    os.replace(temp, path)


def _download_yahoo(ticker: str, start, stop) -> pd.DataFrame:
    """Unadjusted daily OHLCV and dividends for ``[start, stop)`` from ``YahooDownloader``."""
    return YahooDownloader(str(start), str(stop), [ticker]).fetch_unadjusted()


def _adjust_forward(frame: pd.DataFrame, start, stop) -> pd.DataFrame:
    """Rows in ``[start, stop)`` adjusted for dividends, scaled to the first row's prices."""
    dates = pd.to_datetime(frame["date"])
    frame = frame[(dates >= pd.Timestamp(str(start))) & (dates < pd.Timestamp(str(stop)))]
    frame = frame.sort_values("date")
    factors = total_return_factors(frame["close"], frame["dividends"])
    adjusted = frame[FRAME_COLUMNS].copy()
    adjusted[PRICE_COLUMNS] = frame[PRICE_COLUMNS].mul(factors, axis=0)
    return adjusted


def _yahoo(task: Task, start, stop, cache: Path, refresh: bool) -> dict[str, pd.DataFrame]:
    frames = {}
    for ticker in task.tickers:
        data_path, meta_path = cache / "{}.csv".format(ticker), cache / "{}.json".format(ticker)
        frame = None
        if not refresh and data_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("format") == YAHOO_CACHE_FORMAT
                and as_day(meta["start"]) <= start
                and as_day(meta["end"]) >= stop
            ):
                frame = pd.read_csv(data_path, **EXACT_FLOATS)
        if frame is None:
            logger.info("Downloading %s from Yahoo Finance", ticker)
            frame = _download_yahoo(ticker, start, stop)
            if frame.empty:
                raise ValueError("Yahoo Finance returned no data for {} in [{}, {})".format(ticker, start, stop))
            frame = frame[YAHOO_COLUMNS]
            _write_atomic(data_path, frame.to_csv(index=False).encode())
            _write_atomic(
                meta_path,
                json.dumps({
                    "format": YAHOO_CACHE_FORMAT,
                    "start": str(start),
                    "end": str(stop),
                    "fetched": datetime.now(timezone.utc).isoformat(),
                }).encode(),
            )
        frames[ticker] = _adjust_forward(frame, start, stop)
    return frames


# --------------------------------------------------------------------------
# Binance


def _http_get(url: str) -> bytes | None:
    """Body of ``url``; ``None`` on 404. Retries transient failures with backoff."""
    request = urllib.request.Request(url, headers={"User-Agent": "rl-bench"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code != 429 and error.code < 500 or attempt == RETRIES - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == RETRIES - 1:
                raise
        time.sleep(2**attempt)


def _checksum_of(text: bytes, name: str) -> str:
    """The SHA-256 hex digest in a ``.CHECKSUM`` file (``<hex>  <file name>``)."""
    parts = text.decode().split()
    if len(parts) < 1 or len(parts[0]) != 64 or (len(parts) > 1 and parts[-1].lstrip("*") != name):
        raise ValueError("Malformed checksum file for {}: {!r}".format(name, text[:200]))
    return parts[0].lower()


def _verify(payload: bytes, checksum: bytes, name: str):
    expected = _checksum_of(checksum, name)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("Checksum mismatch for {}: expected {}, got {}".format(name, expected, actual))


def _archive(url: str, directory: Path, refresh: bool, recent: bool) -> bytes | None:
    """A verified zip from the cache or the archive; ``None`` if it does not exist.

    Confirmed absences of settled (not recent) files are remembered in a
    ``.missing`` marker so reruns stay offline.
    """
    name = url.rsplit("/", 1)[1]
    zip_path = directory / name
    checksum_path = directory / (name + ".CHECKSUM")
    missing_path = directory / (name + ".missing")
    if not refresh:
        if zip_path.exists() and checksum_path.exists():
            payload = zip_path.read_bytes()
            _verify(payload, checksum_path.read_bytes(), name)
            return payload
        if missing_path.exists():
            return None
    payload = _http_get(url)
    if payload is None:
        if not recent:
            _write_atomic(missing_path, b"")
        return None
    checksum = _http_get(url + ".CHECKSUM")
    if checksum is None:
        raise ValueError("No checksum published for {}".format(url))
    _verify(payload, checksum, name)
    _write_atomic(zip_path, payload)
    _write_atomic(checksum_path, checksum)
    missing_path.unlink(missing_ok=True)
    return payload


def parse_klines(payload: bytes, source: str = "klines") -> pd.DataFrame:
    """Daily rows (date, OHLC, quote volume as ``volume``) from a kline zip.

    Handles files with or without a header row and millisecond or microsecond
    open times; the date is the UTC day of the open time.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("{} should hold exactly one CSV, found {}".format(source, archive.namelist()))
        raw = pd.read_csv(archive.open(members[0]), header=None, dtype=str)
    if raw.shape[1] != len(KLINE_COLUMNS):
        raise ValueError("{} has {} columns, expected {}".format(source, raw.shape[1], len(KLINE_COLUMNS)))
    raw.columns = KLINE_COLUMNS
    if not raw.empty and not raw["open_time"].iloc[0].strip().isdigit():
        raw = raw.iloc[1:]
    opened = raw["open_time"].astype("int64").to_numpy()
    opened = np.where(opened >= MICROSECONDS_FROM, opened // 1000, opened)
    if (opened % DAY_MS).any():
        raise ValueError("{} has klines not opening at 00:00 UTC; is it a daily file?".format(source))
    return pd.DataFrame(
        {
            "date": (opened // DAY_MS).astype("datetime64[D]"),
            "open": raw["open"].astype(float).to_numpy(),
            "high": raw["high"].astype(float).to_numpy(),
            "low": raw["low"].astype(float).to_numpy(),
            "close": raw["close"].astype(float).to_numpy(),
            "volume": raw["quote_volume"].astype(float).to_numpy(),
        }
    )


def _binance_symbol_month(symbol, month_start: date, start, stop, today: date, directory, refresh):
    """Klines of one calendar month, from the monthly zip or else daily zips."""
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    pieces = []
    base = "{}/{{}}/klines/{}/1d/{}-1d-{{}}.zip".format(BINANCE_BASE, symbol, symbol)
    if next_month <= today.replace(day=1):
        name = month_start.strftime("%Y-%m")
        payload = _archive(base.format("monthly", name), directory, refresh, next_month + RECENT > today)
        if payload is not None:
            return [parse_klines(payload, "{} {}".format(symbol, name))]
        if next_month + RECENT <= today:
            return []
    first = max(month_start, np.datetime64(start, "D").astype(date))
    last = min(next_month, np.datetime64(stop, "D").astype(date))
    day = first
    while day < last:
        payload = _archive(base.format("daily", day.isoformat()), directory, refresh, True)
        if payload is not None:
            pieces.append(parse_klines(payload, "{} {}".format(symbol, day)))
        day += timedelta(days=1)
    return pieces


def _binance(task: Task, start, stop, cache: Path, refresh: bool) -> dict[str, pd.DataFrame]:
    today = datetime.now(timezone.utc).date()
    first = np.datetime64(start, "D").astype(date).replace(day=1)
    last = np.datetime64(stop - np.timedelta64(1, "D"), "D").astype(date)
    months = []
    month = first
    while month <= last:
        months.append(month)
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    jobs = [(symbol, month) for symbol in task.tickers for month in months]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(
            pool.map(
                lambda job: _binance_symbol_month(
                    job[0], job[1], start, stop, today, cache / job[0], refresh
                ),
                jobs,
            )
        )
    frames = {symbol: [] for symbol in task.tickers}
    for (symbol, _), pieces in zip(jobs, results):
        frames[symbol].extend(pieces)
    return {
        symbol: pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=FRAME_COLUMNS)
        for symbol, pieces in frames.items()
    }
