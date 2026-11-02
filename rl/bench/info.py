"""Immutable dated information and locally cached JSONL / EDGAR sources.

Only an :class:`InfoView` containing already-visible records reaches a policy.
Model-produced fields use ``{"value": 0.7, "model": "model-revision"}``;
ordinary numeric fields may be bare numbers. JSONL text hashes cover UTF-8 text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.request
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo

UTC = timezone.utc
FORMS = {"8-K", "10-Q", "10-K"}
TEXT_LIMIT = 1_000_000


def timestamp(value) -> datetime:
    """Parse an explicit timezone and reject lossy subsecond timestamps."""
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond:
        raise ValueError("Information timestamps need a timezone and second precision")
    return parsed.astimezone(UTC)


def session_close(day, *, crypto: bool = False) -> datetime:
    """Regular US close, or the end of a UTC crypto day (DST aware)."""
    date = datetime.fromisoformat(str(day)[:10])
    if crypto:
        return (date + timedelta(days=1)).replace(tzinfo=UTC)
    return date.replace(hour=16, tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)


@dataclass(frozen=True)
class ModelField:
    value: float
    model: str

    def __post_init__(self):
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError("Information fields must be finite numbers")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model-produced fields need a model id")


@dataclass(frozen=True)
class InfoItem:
    id: str
    source: str
    tickers: tuple[str, ...]
    published_at: datetime
    title: str
    text_sha256: str
    text: str
    fields: Mapping[str, float | ModelField] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("id", "source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("Information {} must be nonempty".format(name))
        if isinstance(self.tickers, str) or not all(isinstance(t, str) and t for t in self.tickers):
            raise ValueError("Information tickers must be a list of ticker strings")
        if not isinstance(self.title, str) or not isinstance(self.text, str):
            raise ValueError("Information title and text must be strings")
        if self.text_sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("Information text_sha256 does not match text")
        fields = {}
        for key, value in self.fields.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Information field names must be nonempty strings")
            if isinstance(value, dict):
                value = ModelField(**value)
            if not isinstance(value, ModelField) and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ValueError("Information fields must be finite numbers or model/value records")
            fields[key] = value
        object.__setattr__(self, "tickers", tuple(sorted(set(self.tickers))))
        object.__setattr__(self, "published_at", timestamp(self.published_at))
        object.__setattr__(self, "fields", MappingProxyType(fields))

    def to_dict(self) -> dict:
        result = {name: getattr(self, name) for name in ("id", "source", "title", "text_sha256", "text")}
        result["tickers"] = list(self.tickers)
        result["published_at"] = self.published_at.isoformat()
        if self.fields:
            result["fields"] = {
                key: {"value": value.value, "model": value.model} if isinstance(value, ModelField) else value
                for key, value in self.fields.items()
            }
        return result


@dataclass(frozen=True)
class InfoView:
    """A snapshot containing only records strictly before a decision close."""

    _visible: tuple[InfoItem, ...] = ()

    def items(self, ticker=None, since=None) -> tuple[InfoItem, ...]:
        since = None if since is None else timestamp(since)
        return tuple(
            item for item in self._visible
            if (ticker is None or ticker in item.tickers)
            and (since is None or item.published_at >= since)
        )


class InfoPanel:
    """Run-owned source snapshot; never passed to a policy."""

    def __init__(self, items=()):
        unique = {}
        for item in items:
            key = (item.source, item.id)
            if key in unique and unique[key].to_dict() != item.to_dict():
                raise ValueError("Conflicting information item {}".format(key))
            unique[key] = item
        self._items = tuple(sorted(unique.values(), key=lambda i: (i.published_at, i.source, i.id)))
        self._times = tuple(i.published_at for i in self._items)

    def before(self, close) -> InfoView:
        return InfoView(self._items[:bisect_left(self._times, timestamp(close))])

    def at_session(self, day, *, crypto=False) -> InfoView:
        return self.before(session_close(day, crypto=crypto))

    def truncated(self, close) -> InfoPanel:
        """Only the items visible at ``close``, as if later ones had never been published."""
        return InfoPanel(self._items[:bisect_left(self._times, timestamp(close))])

    def poisoned(self, close, tickers) -> InfoPanel:
        """Same visible items; every item from ``close`` on is replaced, and two are injected.

        The injected items are published exactly at ``close`` and one second
        later, so an inclusive comparison also changes what a policy sees.
        """
        junk = {"title": "poison", "text": "poison", "text_sha256": hashlib.sha256(b"poison").hexdigest()}
        close = timestamp(close)
        kept = self._items[:bisect_left(self._times, close)]
        replaced = [replace(item, fields={}, **junk) for item in self._items[len(kept):]]
        injected = [
            InfoItem(id="poison-{}".format(n), source="check-causal", tickers=tuple(tickers),
                     published_at=close + timedelta(seconds=n), **junk)
            for n in (0, 1)
        ]
        return InfoPanel([*kept, *replaced, *injected])

    def sha256(self) -> str:
        content = [item.to_dict() for item in self._items]
        return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @property
    def models(self) -> set[str]:
        return {v.model for item in self._items for v in item.fields.values() if isinstance(v, ModelField)}


def validate_sources(sources) -> tuple[dict, ...]:
    if not isinstance(sources, (list, tuple)):
        raise ValueError("info_sources must be a list")
    result = []
    for spec in sources:
        spec = {"source": spec} if isinstance(spec, str) else dict(spec)
        source = spec.get("source")
        allowed = {"source", "path"} if source == "jsonl" else {"source", "download_text", "max_text_bytes"}
        if source not in ("jsonl", "sec-edgar") or set(spec) - allowed:
            raise ValueError("Unknown information source or options: {}".format(spec))
        if source == "jsonl" and (not isinstance(spec.get("path"), str) or not spec["path"]):
            raise ValueError("jsonl sources need a path")
        if source == "sec-edgar":
            if not isinstance(spec.get("download_text", False), bool):
                raise ValueError("download_text must be boolean")
            limit = spec.get("max_text_bytes", TEXT_LIMIT)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= TEXT_LIMIT:
                raise ValueError("max_text_bytes must be between 1 and {}".format(TEXT_LIMIT))
        result.append(spec)
    return tuple(result)


def load_task_info(task, submission=None, *, cache_dir=None, refresh=False) -> InfoPanel:
    """Combine declared sources once, with submission paths relative to its directory.

    Cache files stay under the price cache's ``info/`` directory. EDGAR
    metadata-only records hash empty text; downloaded text hashes the retained
    UTF-8 text, at most ``max_text_bytes`` source bytes (never a whole large filing).
    """
    from rl.bench.data import DATA_ENV, DEFAULT_CACHE

    cache = Path(cache_dir or os.environ.get(DATA_ENV) or DEFAULT_CACHE) / "info"
    items = []
    declarations = [(getattr(task, "info_sources", ()), Path.cwd())]
    if submission is not None:
        declarations.append((submission.info_sources, submission.directory or Path.cwd()))
    for sources, root in declarations:
        for spec in validate_sources(sources):
            if spec["source"] == "jsonl":
                content = (root / spec["path"]).read_bytes()
                parsed = [InfoItem(**json.loads(line)) for line in content.splitlines() if line.strip()]
                _cache_bytes(cache / "jsonl" / (hashlib.sha256(content).hexdigest() + ".jsonl"), content)
                items.extend(parsed)
            else:
                items.extend(_edgar(task, spec, cache / "sec-edgar", refresh))
    return InfoPanel(item for item in items if not item.tickers or set(item.tickers).intersection(task.tickers))


def _cache_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_bytes(content)
    temporary.replace(path)


def _sec_get(url: str, path: Path, user_agent: str, *, refresh=False, limit=None) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes() if limit is None else path.read_bytes()[:limit]
    # Sequential requests stay below SEC's ten requests/second fair-access limit.
    time.sleep(0.11)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read() if limit is None else response.read(limit)
    _cache_bytes(path, content)
    return content


def _accepted(header: str, accession: str) -> datetime:
    """EDGAR's acceptance instant from a filing's SGML header, a New York wall clock.

    Filings from before EDGAR stamped acceptance carry only ``<FILING-DATE>``;
    they are placed at the end of that day in New York, after its close, so a
    policy sees them from the next session.
    """
    if accession not in header:
        raise ValueError("EDGAR header for {} names another filing".format(accession))
    stamp = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", header)
    if stamp:
        wall = datetime.strptime(stamp.group(1), "%Y%m%d%H%M%S")
    else:
        filed = re.search(r"<FILING-DATE>(\d{8})", header)
        if filed is None:
            raise ValueError("EDGAR header for {} has neither acceptance nor filing date".format(accession))
        wall = datetime.strptime(filed.group(1), "%Y%m%d").replace(hour=23, minute=59, second=59)
    return timestamp(wall.replace(tzinfo=ZoneInfo("America/New_York")))


def _edgar(task, spec: dict, cache: Path, refresh: bool) -> list[InfoItem]:
    """8-K, 10-Q and 10-K filings of the task's currently listed tickers.

    Filing lists come from the submissions API (``refresh`` re-downloads them).
    Their ``acceptanceDateTime`` is NOT used: SEC's JSON mixes UTC and New York
    clocks that both end in ``Z``, so each filing's time is read from its
    immutable ``<accession>.hdr.sgml`` instead (served for every year, unlike the
    ``-index-headers.html`` page), cached permanently.
    """
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not user_agent:
        raise ValueError("sec-edgar requires SEC_EDGAR_USER_AGENT (name and contact email)")

    def get_json(name, url):
        return json.loads(_sec_get(url, cache / name, user_agent, refresh=refresh))

    companies = get_json("company_tickers.json", "https://www.sec.gov/files/company_tickers.json")
    ciks = {}
    for company in companies.values():
        ticker = company["ticker"].replace(".", "-")
        if ticker in task.tickers:
            ciks.setdefault(int(company["cik_str"]), []).append(ticker)
    items = []
    for cik, tickers in ciks.items():
        name = "CIK{:010d}.json".format(cik)
        company = get_json(name, "https://data.sec.gov/submissions/" + name)
        batches = [company["filings"]["recent"]]
        for older in company["filings"].get("files", []):
            name = older["name"]
            if not re.fullmatch(r"CIK\d+-submissions-\d+\.json", name):
                raise ValueError("Unexpected EDGAR history filename")
            batches.append(get_json(name, "https://data.sec.gov/submissions/" + name))
        for batch in batches:
            for index, form in enumerate(batch["form"]):
                if form not in FORMS or batch["filingDate"][index] < str(task.start):
                    continue
                accession = batch["accessionNumber"][index]
                document = batch["primaryDocument"][index]
                if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or not re.fullmatch(r"[\w.\-]*", document):
                    raise ValueError("Unexpected EDGAR accession or document")
                folder = "https://www.sec.gov/Archives/edgar/data/{}/{}/".format(cik, accession.replace("-", ""))
                header = _sec_get(
                    folder + accession + ".hdr.sgml", cache / "headers" / (accession + ".sgml"), user_agent
                ).decode("utf-8", errors="replace")
                text = ""
                if spec.get("download_text", False) and document:
                    limit = spec.get("max_text_bytes", TEXT_LIMIT)
                    raw = _sec_get(
                        folder + document, cache / "documents" / "{}-{}-{}".format(accession, limit, document),
                        user_agent, limit=limit,
                    )
                    text = raw.decode("utf-8", errors="replace")
                description = batch["primaryDocDescription"][index]
                items.append(InfoItem(
                    id=accession, source="sec-edgar", tickers=tuple(tickers), published_at=_accepted(header, accession),
                    title="{} {}".format(form, description).strip(),
                    text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                ))
    return items


def historical_contamination(submission, task, panel: InfoPanel) -> str | None:
    """Validate model provenance before any historical policy calls or writes."""
    cutoffs = {entry["model"]: entry for entry in submission.model_cutoffs}
    if submission.kind == "llm" and not cutoffs:
        raise ValueError("LLM historical submissions require model_cutoffs")
    missing = panel.models - cutoffs.keys()
    if missing:
        raise ValueError("Model-produced information requires model_cutoffs for {}".format(sorted(missing)))
    if any(str(task.test_start) < entry["training_cutoff"] for entry in cutoffs.values()):
        if submission.contamination != "possible":
            raise ValueError('Test window precedes a model training cutoff; declare contamination: "possible"')
        return "possible"
    return submission.contamination
