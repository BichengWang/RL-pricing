"""Convert Sharadar SEP price files into the toolkit's CSV format.

Sharadar's Equity Prices table (SEP, https://sharadar.com/docs/stocks) has the
columns ``ticker, date, open, high, low, close, volume, closeadj, closeunadj``
(plus ``lastupdated``). ``open``, ``high``, ``low`` and ``close`` are adjusted
for splits only; ``closeadj`` is also adjusted for cash dividends and
spin-offs. The toolkit uses one adjustment basis, so every price is scaled by
``closeadj / close``: the output ``close`` is ``closeadj`` and ``open``,
``high`` and ``low`` are multiplied by the same factor, which keeps every
bar's shape (``low <= open, close <= high`` and the open-to-close ratio).
``volume`` is copied. The result has the columns
``date, tic, open, high, low, close, volume``; it is derived from vendor data
and must stay local, never committed.

A ticker map renames or stitches series: ``{"UTX": "RTX"}`` writes Sharadar's
``RTX`` rows under the ticker ``UTX``; a list stitches several source tickers,
each optionally limited to inclusive dates, e.g.
``{"DD": [{"ticker": "DD", "end": "2017-08-31"}, {"ticker": "DWDP", "start":
"2017-09-01", "end": "2019-03-31"}]}``. Segments are adjusted independently
(each by its own ``closeadj`` base), so a jump at a seam is possible; the
conversion warns about close-to-close moves above 25% there.

``python -m rl.marketdata.sharadar --help`` converts a file from the command line.
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["date", "tic", "open", "high", "low", "close", "volume"]
SOURCE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "closeadj"]
SEAM_JUMP = 0.25


def load_ticker_map(path):
    """A ticker map from a JSON file (see the module docstring)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _segments(ticker_map, tickers):
    """``[(output_tic, source_ticker, start, end)]`` for the requested tickers."""
    ticker_map = ticker_map or {}
    segments = []
    for tic in tickers:
        spec = ticker_map.get(tic, tic)
        for part in [spec] if isinstance(spec, (str, dict)) else list(spec):
            if isinstance(part, str):
                part = {"ticker": part}
            unknown = set(part) - {"ticker", "start", "end"}
            if "ticker" not in part or unknown:
                raise ValueError(
                    "Ticker map entry for {} must have 'ticker' and optionally 'start'/'end': "
                    "{!r}".format(tic, part)
                )
            segments.append((
                tic,
                part["ticker"],
                None if part.get("start") is None else pd.Timestamp(part["start"]),
                None if part.get("end") is None else pd.Timestamp(part["end"]),
            ))
    return segments


def _adjusted(frame, tic, source):
    """One output row per source row with every price on the ``closeadj`` basis."""
    prices = frame[["open", "high", "low", "close", "closeadj"]].apply(
        pd.to_numeric, errors="coerce"
    )
    factor = prices["closeadj"] / prices["close"]
    out = pd.DataFrame({
        "date": frame["date"].dt.strftime("%Y-%m-%d"),
        "tic": tic,
        "open": prices["open"] * factor,
        "high": prices["high"] * factor,
        "low": prices["low"] * factor,
        "close": prices["closeadj"],
        "volume": pd.to_numeric(frame["volume"], errors="coerce"),
    })
    prices_ok = np.isfinite(out[["open", "high", "low", "close"]].to_numpy()).all(axis=1)
    prices_ok &= (out[["open", "high", "low", "close"]].to_numpy() > 0).all(axis=1)
    dropped = int((~prices_ok).sum())
    if dropped:
        logger.warning(
            "%s (from %s): dropped %d rows without finite positive prices", tic, source, dropped
        )
    return out[prices_ok]


def convert_sharadar(
    source, tickers, ticker_map=None, start_date=None, end_date=None, chunksize=1_000_000
):
    """Toolkit-format prices for ``tickers`` from a Sharadar SEP CSV file.

    ``source`` is a path or file object (a bulk SEP export works; ``.zip`` and
    ``.gz`` are read directly and the file is streamed in chunks).
    ``start_date`` is inclusive and ``end_date`` exclusive, like the pipeline's
    own dates. Returns a frame sorted by ``date, tic``.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("tickers must not be empty")
    segments = _segments(ticker_map, tickers)
    wanted = {source_ticker for _, source_ticker, _, _ in segments}
    start = None if start_date is None else pd.Timestamp(start_date)
    end = None if end_date is None else pd.Timestamp(end_date)

    chunks = []
    with pd.read_csv(
        source, usecols=lambda name: name in SOURCE_COLUMNS, chunksize=chunksize,
        dtype={"ticker": str},
    ) as reader:
        for chunk in reader:
            missing = set(SOURCE_COLUMNS) - set(chunk.columns)
            if missing:
                raise ValueError("Sharadar file is missing columns {}".format(sorted(missing)))
            chunk = chunk[chunk["ticker"].isin(wanted)].copy()
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunks.append(chunk)
    raw = pd.concat(chunks, ignore_index=True)
    if raw.empty:
        raise ValueError("None of {} found in the Sharadar file".format(sorted(wanted)))

    parts = []
    for tic, source_ticker, seg_start, seg_end in segments:
        rows = raw[raw["ticker"] == source_ticker]
        if seg_start is not None:
            rows = rows[rows["date"] >= seg_start]
        if seg_end is not None:
            rows = rows[rows["date"] <= seg_end]
        if start is not None:
            rows = rows[rows["date"] >= start]
        if end is not None:
            rows = rows[rows["date"] < end]
        if rows.empty:
            logger.warning(
                "No Sharadar rows for %s (source %s) in the requested dates", tic, source_ticker
            )
            continue
        parts.append(_adjusted(rows, tic, source_ticker))
    if not parts:
        raise ValueError("No Sharadar rows for the requested tickers and dates")
    out = pd.concat(parts, ignore_index=True).sort_values(["tic", "date"], ignore_index=True)
    duplicated = out.duplicated(["date", "tic"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Overlapping segments or duplicate rows for {} on {} dates; "
            "limit the ticker map segments with 'start'/'end'".format(
                sorted(out.loc[duplicated, "tic"].unique()),
                out.loc[duplicated, "date"].nunique(),
            )
        )
    _warn_seam_jumps(out, segments)
    absent = sorted(set(tickers) - set(out["tic"]))
    if absent:
        logger.warning("No data for: %s", ", ".join(absent))
    return out.sort_values(["date", "tic"], ignore_index=True)[OUTPUT_COLUMNS]


def _warn_seam_jumps(out, segments):
    """Warn about large close-to-close moves where stitched segments meet."""
    for tic in {tic for tic, *_ in segments}:
        if sum(1 for s in segments if s[0] == tic) < 2:
            continue
        close = out.loc[out["tic"] == tic].set_index("date")["close"]
        moves = close.pct_change().abs()
        for date in moves.index[moves > SEAM_JUMP]:
            logger.warning(
                "%s moves %.0f%% into %s; check the ticker map segments there",
                tic, 100 * moves[date], date,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a Sharadar SEP CSV into the toolkit's date,tic,open,high,low,"
        "close,volume CSV (prices on the closeadj basis)."
    )
    parser.add_argument("input", help="Sharadar SEP CSV (or .zip/.gz) with a ticker column")
    parser.add_argument("output", help="CSV to write; keep it out of version control")
    parser.add_argument("--tickers", nargs="+", required=True, help="output tickers")
    parser.add_argument(
        "--ticker-map", help="JSON file: output ticker -> source ticker or list of segments"
    )
    parser.add_argument("--start-date", help="first date, inclusive")
    parser.add_argument("--end-date", help="last date, exclusive")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ticker_map = load_ticker_map(options.ticker_map) if options.ticker_map else None
    df = convert_sharadar(
        options.input, options.tickers, ticker_map, options.start_date, options.end_date
    )
    df.to_csv(options.output, index=False)
    logger.info("Wrote %d rows for %d tickers to %s", len(df), df["tic"].nunique(), options.output)
    return df


if __name__ == "__main__":
    main()
