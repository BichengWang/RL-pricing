#!/usr/bin/env python3
"""Fetch the pinned FinRL source and Yahoo price inputs and write their hashes.

Raw downloads stay under .cache/r1-finrl-ensemble and are not committed.
The manifest is reproducible from the same source URLs; retrieval timestamps
record when each run of this script was performed.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PIN = "5dbb44dfcb8a01a94762dc44fca2bcb2c626d1dc"
REPOSITORY = "https://github.com/AI4Finance-Foundation/FinRL-Trading"
ARCHIVE_URL = f"{REPOSITORY}/archive/{PIN}.zip"
YAHOO_PERIOD1 = 1230768000  # 2009-01-01 00:00:00 UTC
YAHOO_PERIOD2 = 1767312000  # 2026-01-02 00:00:00 UTC (exclusive)
YAHOO_SYMBOLS = ("^DJI", "DIA")
CACHE = ROOT / ".cache" / "r1-finrl-ensemble"
MANIFEST = ROOT / "reports" / "r1-finrl-ensemble" / "evidence" / "hashes.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RL-pricing-reproducibility-lab/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def record_file(path: str, data: bytes, url: str, version: str) -> dict[str, object]:
    return {
        "path": path,
        "url": url,
        "version": version,
        "size_bytes": len(data),
        "sha256": sha256(data),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)

    archive = get(ARCHIVE_URL)
    archive_path = CACHE / f"FinRL-Trading-{PIN}.zip"
    archive_path.write_bytes(archive)
    files = [
        record_file(archive_path.name, archive, ARCHIVE_URL, PIN),
    ]

    source_files: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as bundle:
        members = sorted(
            (item for item in bundle.infolist() if not item.is_dir()),
            key=lambda item: item.filename,
        )
        prefix = members[0].filename.split("/", 1)[0] + "/"
        checkout = CACHE / f"FinRL-Trading-{PIN}"
        for item in members:
            relative = item.filename.removeprefix(prefix)
            if not relative:
                continue
            data = bundle.read(item)
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            source_files.append(record_file(relative, data, ARCHIVE_URL, PIN))
    files.append(
        {
            "path": f"FinRL-Trading@{PIN}",
            "url": ARCHIVE_URL,
            "version": PIN,
            "file_count": len(source_files),
            "files": source_files,
        }
    )

    yahoo_records = []
    for symbol in YAHOO_SYMBOLS:
        query = urllib.parse.urlencode(
            {
                "period1": YAHOO_PERIOD1,
                "period2": YAHOO_PERIOD2,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol, safe='')}?{query}"
        data = get(url)
        filename = f"yahoo-{symbol.replace('^', '')}.json"
        (CACHE / filename).write_bytes(data)
        yahoo_records.append(
            record_file(filename, data, url, "Yahoo chart API, daily, 2009-01-01 through 2025-12-31")
        )
        time.sleep(1)

    manifest = {
        "schema_version": 1,
        "description": "R1 FinRL ensemble source and Yahoo baseline input hashes. Raw downloads are excluded from git.",
        "repository": REPOSITORY,
        "commit": PIN,
        "archive_url": ARCHIVE_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_directory": ".cache/r1-finrl-ensemble (local, untracked)",
        "authors_repository": files,
        "yahoo_data": yahoo_records,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    print(f"Pinned source files hashed: {len(source_files)}")
    for record in yahoo_records:
        print(f"{record['path']}: {record['size_bytes']} bytes sha256={record['sha256']}")


if __name__ == "__main__":
    main()
