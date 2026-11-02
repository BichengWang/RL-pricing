"""Point-in-time daily market panels.

:class:`MarketData` stores one row per session and one column per asset.
A NaN price marks an asset that cannot trade that session (not yet listed,
halted or delisted); open and close prices are either both present or both
missing. Arrays are read-only so a policy cannot alter the data it is given.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rl.bench.info import InfoView

FIELDS = ("open", "high", "low", "close", "volume")


def as_day(value) -> np.datetime64:
    """A calendar day as ``numpy.datetime64[D]``."""
    return np.datetime64(pd.Timestamp(value).date(), "D")


@dataclass(frozen=True, eq=False)
class MarketData:
    dates: np.ndarray
    tickers: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self):
        dates = np.array(self.dates, dtype="datetime64[D]")
        tickers = tuple(str(t) for t in self.tickers)
        if len(set(tickers)) != len(tickers):
            raise ValueError("Tickers must be unique")
        if dates.ndim != 1 or (dates.size > 1 and not (np.diff(dates) > np.timedelta64(0, "D")).all()):
            raise ValueError("Dates must be strictly increasing")
        shape = (dates.size, len(tickers))
        arrays = {}
        for name in FIELDS:
            values = np.array(getattr(self, name), dtype=float)
            if values.shape != shape:
                raise ValueError("{} has shape {}, expected {}".format(name, values.shape, shape))
            arrays[name] = values
        listed = np.isfinite(arrays["close"])
        if not np.array_equal(listed, np.isfinite(arrays["open"])):
            raise ValueError("Open and close prices must be missing on the same sessions")
        for name in ("open", "high", "low", "close"):
            present = arrays[name][listed]
            if not (np.isfinite(present).all() and (present > 0).all()):
                raise ValueError("Listed sessions need finite positive {} prices".format(name))
        object.__setattr__(self, "dates", _frozen(dates))
        object.__setattr__(self, "tickers", tickers)
        for name, values in arrays.items():
            object.__setattr__(self, name, _frozen(values))

    @classmethod
    def _unchecked(cls, dates, tickers, arrays):
        """Wrap already-validated read-only arrays (views of a validated panel)."""
        data = object.__new__(cls)
        object.__setattr__(data, "dates", dates)
        object.__setattr__(data, "tickers", tickers)
        for name in FIELDS:
            object.__setattr__(data, name, arrays[name])
        return data

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> MarketData:
        """Build from a long frame with ``date, tic`` and the OHLCV columns."""
        missing = [c for c in ("date", "tic", *FIELDS) if c not in frame.columns]
        if missing:
            raise ValueError("Data is missing columns {}".format(missing))
        if frame.duplicated(["date", "tic"]).any():
            raise ValueError("Data has duplicate (date, tic) rows")
        frame = frame.assign(date=pd.to_datetime(frame["date"]).dt.normalize())
        wide = {
            name: frame.pivot(index="date", columns="tic", values=name).sort_index().sort_index(axis=1)
            for name in FIELDS
        }
        close = wide["close"]
        return cls(
            dates=close.index.to_numpy(dtype="datetime64[D]"),
            tickers=tuple(close.columns),
            **{name: wide[name].reindex_like(close).to_numpy(dtype=float) for name in FIELDS},
        )

    def to_frame(self) -> pd.DataFrame:
        """The long ``date, tic, open, high, low, close, volume`` frame of listed sessions."""
        n_dates, n_tickers = self.close.shape
        frame = pd.DataFrame(
            {
                "date": np.repeat(self.dates, n_tickers),
                "tic": np.tile(np.array(self.tickers, dtype=object), n_dates),
                **{name: getattr(self, name).ravel() for name in FIELDS},
            }
        )
        return frame[np.isfinite(frame["close"])].reset_index(drop=True)

    def __len__(self) -> int:
        return self.dates.size

    def index_of(self, date) -> int:
        """Index of the first session on or after ``date``."""
        return int(np.searchsorted(self.dates, as_day(date), side="left"))

    def rows(self, start: int, stop: int) -> MarketData:
        """Sessions ``[start, stop)`` as zero-copy views."""
        if not 0 <= start <= stop <= len(self):
            raise IndexError("Rows [{}, {}) outside 0..{}".format(start, stop, len(self)))
        return MarketData._unchecked(
            self.dates[start:stop],
            self.tickers,
            {name: getattr(self, name)[start:stop] for name in FIELDS},
        )

    def between(self, start=None, end=None) -> MarketData:
        """Sessions with ``start <= date < end``; ``None`` leaves a side open."""
        first = 0 if start is None else self.index_of(start)
        last = len(self) if end is None else self.index_of(end)
        return self.rows(first, max(first, last))

    def window(self, n: int) -> MarketData:
        """The most recent ``n`` sessions (fewer if the history is shorter)."""
        if n < 1:
            raise ValueError("Window length must be positive")
        return self.rows(max(0, len(self) - n), len(self))

    def select(self, tickers) -> MarketData:
        """A panel restricted to ``tickers`` in the given order."""
        positions = [self.tickers.index(t) for t in tickers]
        return MarketData(
            self.dates, tuple(tickers), **{name: getattr(self, name)[:, positions] for name in FIELDS}
        )

    def sha256(self) -> str:
        """Content hash of dates, tickers and prices, independent of file formatting."""
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.dates.astype("int64")).tobytes())
        digest.update("\x1f".join(self.tickers).encode())
        for name in FIELDS:
            digest.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class MarketView:
    """What a policy may see when it decides at a session's close.

    ``history`` ends at the decision session. ``weights`` are the current
    asset weights of the account at that close after price drift, as
    fractions of equity; the remainder is cash.
    """

    history: MarketData
    weights: np.ndarray
    info: InfoView = field(default_factory=InfoView)

    @property
    def date(self) -> np.datetime64:
        return self.history.dates[-1]

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.history.tickers

    @property
    def listed(self) -> np.ndarray:
        """Assets with a price at the decision session."""
        return np.isfinite(self.history.close[-1])


def _frozen(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values)
    values.setflags(write=False)
    return values
