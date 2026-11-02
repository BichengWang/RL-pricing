"""Verdict table of the reproducibility lab.

Every report lives in ``reports/<id>/`` and states its claims in
``reports/<id>/claims.json``: a list of objects

``id``       unique within the file (kebab-case, used as the claim's name)
``quote``    the claim, verbatim
``source``   where it is made (paper table, page, README line)
``metric``   metric name; names containing ``return``, ``drawdown``, ``cagr`` or
             ``volatility`` are shown as percentages, others with two decimals
``claimed``  the claimed value (number)
``ours``     our seed-averaged value (number) or ``null`` when not measured
``ci95``     ``[low, high]`` 95% interval of ``ours`` or ``null``
``level``    ``"L1"`` replication, ``"L2"`` robustness, ``"L3"`` generalization
``verdict``  ``"holds"``, ``"partly"``, ``"does-not-hold"`` or ``"inconclusive"``;
             ``null`` while the claim is pre-registered and not yet tested (it is
             validated but has no row in the table)
``note``     one-sentence reason (free text)

:func:`load_reports` validates all of them (unknown fields are rejected, errors
name the file and the claim) and :func:`write_reports` renders the table into
``reports/README.md`` and the main README between the
``<!-- reports:start -->`` / ``<!-- reports:end -->`` markers.  The rules that
decide a verdict are in the lab protocol; this module only displays them.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
README_START = "<!-- reports:start -->"
README_END = "<!-- reports:end -->"

LEVELS = ("L1", "L2", "L3")
#: verdict -> table symbol, in the order the protocol lists them.
VERDICTS = {
    "holds": "✅",
    "partly": "◐",
    "does-not-hold": "✗",
    "inconclusive": "?",
}
_LABELS = {
    "holds": "holds",
    "partly": "partly holds",
    "does-not-hold": "does not hold",
    "inconclusive": "inconclusive",
}
_FIELDS = ("id", "quote", "source", "metric", "claimed", "ours", "ci95", "level", "verdict", "note")
_PERCENT_METRICS = ("return", "drawdown", "cagr", "volatility")


class ClaimsError(ValueError):
    """A ``claims.json`` file does not follow the schema."""


@dataclass(frozen=True)
class Claim:
    id: str
    quote: str
    source: str
    metric: str
    claimed: float
    ours: float | None
    ci95: tuple[float, float] | None
    level: str
    verdict: str | None
    note: str


@dataclass(frozen=True)
class Report:
    id: str
    claims: tuple[Claim, ...]


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_claim(entry, index: int, where: str) -> Claim:
    label = "{}: claim #{}".format(where, index + 1)
    if not isinstance(entry, dict):
        raise ClaimsError("{} must be an object, got {}".format(label, type(entry).__name__))
    if isinstance(entry.get("id"), str) and entry["id"]:
        label += " ({!r})".format(entry["id"])
    missing = [field for field in _FIELDS if field not in entry]
    if missing:
        raise ClaimsError("{} is missing field(s): {}".format(label, ", ".join(missing)))
    unknown = sorted(set(entry) - set(_FIELDS))
    if unknown:
        raise ClaimsError("{} has unknown field(s): {}".format(label, ", ".join(unknown)))
    for field in ("id", "quote", "source", "metric", "note"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise ClaimsError("{}: {!r} must be a non-empty string".format(label, field))
    if not _number(entry["claimed"]):
        raise ClaimsError("{}: 'claimed' must be a finite number, got {!r}".format(label, entry["claimed"]))
    if entry["ours"] is not None and not _number(entry["ours"]):
        raise ClaimsError("{}: 'ours' must be a finite number or null, got {!r}".format(label, entry["ours"]))
    interval = entry["ci95"]
    if interval is not None:
        if not (isinstance(interval, (list, tuple)) and len(interval) == 2 and all(_number(x) for x in interval)):
            raise ClaimsError("{}: 'ci95' must be [low, high] or null, got {!r}".format(label, interval))
        if interval[0] > interval[1]:
            raise ClaimsError("{}: 'ci95' low {} exceeds high {}".format(label, interval[0], interval[1]))
        interval = (float(interval[0]), float(interval[1]))
    if entry["level"] not in LEVELS:
        raise ClaimsError("{}: level {!r} is not one of {}".format(label, entry["level"], ", ".join(LEVELS)))
    if entry["verdict"] is not None and entry["verdict"] not in VERDICTS:
        raise ClaimsError(
            "{}: verdict {!r} is not one of {} or null".format(label, entry["verdict"], ", ".join(VERDICTS))
        )
    return Claim(
        id=entry["id"],
        quote=entry["quote"],
        source=entry["source"],
        metric=entry["metric"],
        claimed=float(entry["claimed"]),
        ours=None if entry["ours"] is None else float(entry["ours"]),
        ci95=interval,
        level=entry["level"],
        verdict=entry["verdict"],
        note=entry["note"],
    )


def load_claims(path) -> tuple[Claim, ...]:
    """Validated claims of one ``claims.json``."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ClaimsError("{}: invalid JSON ({})".format(path, error)) from error
    if not isinstance(raw, list) or not raw:
        raise ClaimsError("{}: expected a non-empty list of claims".format(path))
    claims = tuple(_parse_claim(entry, i, str(path)) for i, entry in enumerate(raw))
    seen: set[str] = set()
    for claim in claims:
        if claim.id in seen:
            raise ClaimsError("{}: duplicate claim id {!r}".format(path, claim.id))
        seen.add(claim.id)
    return claims


def load_reports(root=REPORTS_DIR) -> list[Report]:
    """Every ``root/<id>/claims.json``, validated, ordered by report id."""
    reports = []
    for path in sorted(Path(root).glob("*/claims.json")):
        reports.append(Report(id=path.parent.name, claims=load_claims(path)))
    return reports


# --- rendering -------------------------------------------------------------


def _format(metric: str, value: float | None) -> str:
    if value is None:
        return "–"
    if any(key in metric.lower() for key in _PERCENT_METRICS):
        return "{:.1f}%".format(value * 100)
    return "{:.2f}".format(value)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _ours(claim: Claim) -> str:
    text = _format(claim.metric, claim.ours)
    if claim.ours is not None and claim.ci95 is not None:
        text += " [{}, {}]".format(_format(claim.metric, claim.ci95[0]), _format(claim.metric, claim.ci95[1]))
    return text


def render_table(reports: list[Report], link_prefix: str = "") -> str:
    """The verdict table: ``Report | Claim | Level | Claimed | Ours | Verdict``.

    Rows follow report-id order, then the order of each ``claims.json``; claims
    without a verdict yet are left out.  ``link_prefix`` is the path from the
    document to ``reports/``.
    """
    rows = []
    for report in reports:
        link = "[{0}]({1}{0}/report.md)".format(report.id, link_prefix)
        for claim in report.claims:
            if claim.verdict is None:
                continue
            rows.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    link,
                    _cell("`{}` ({})".format(claim.id, claim.metric)),
                    claim.level,
                    _format(claim.metric, claim.claimed),
                    _cell(_ours(claim)),
                    "{} {}".format(VERDICTS[claim.verdict], _LABELS[claim.verdict]),
                )
            )
    if not rows:
        return "No reports yet."
    lines = [
        "| Report | Claim | Level | Claimed | Ours (95% interval) | Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
    ]
    legend = "{}.".format(", ".join("{} {}".format(symbol, _LABELS[name]) for name, symbol in VERDICTS.items()))
    rules = "[How a verdict is decided]({}README.md#how-a-verdict-is-decided)".format(link_prefix)
    return "\n".join([*lines, "", "Verdicts: {} {}.".format(legend, rules)])


def update_block(path, text: str, start: str = README_START, end: str = README_END) -> bool:
    """Replace the text between ``start`` and ``end`` in ``path``; ``False`` if absent."""
    path = Path(path)
    if not path.exists():
        return False
    content = path.read_text()
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(content):
        return False
    block = "{}\n{}\n{}".format(start, text.strip("\n"), end)
    updated = pattern.sub(lambda _: block, content, count=1)
    if updated != content:
        path.write_text(updated)
    return True


def write_reports(root=REPORTS_DIR, *, readme: str | Path | None = "README.md") -> str:
    """Validate every claims file and refresh the generated tables.

    ``<root>/README.md`` gets the table with links relative to it; ``readme``
    (the main README, skipped when ``None``) gets the same table with links
    relative to the repository root.  Both are untouched when they lack the
    markers.  Returns the table rendered for ``<root>/README.md``.
    """
    root = Path(root)
    reports = load_reports(root)
    table = render_table(reports)
    if not update_block(root / "README.md", table):
        log.warning("%s has no %s ... %s markers", root / "README.md", README_START, README_END)
    if readme is not None:
        main = render_table(reports, link_prefix="{}/".format(root.as_posix()))
        if not update_block(readme, main):
            log.info("%s has no reports markers; skipped", readme)
    return table


def add_cli(subparsers, common=None):
    """Register ``rl-pricing bench reports``; its parser carries ``handler=run_cli``."""
    parser = subparsers.add_parser(
        "reports",
        help="validate reports/*/claims.json and render the verdict tables",
    )
    parser.add_argument("--root", default=str(REPORTS_DIR), help="directory of report folders")
    parser.add_argument("--readme", default="README.md", help="README whose reports markers are refreshed")
    parser.add_argument("--no-readme", action="store_true")
    parser.set_defaults(handler=run_cli)
    return parser


def run_cli(options) -> int:
    """Handler of ``bench reports``: exit 1 with the validation error on a bad claims file."""
    try:
        table = write_reports(options.root, readme=None if options.no_readme else options.readme)
    except ClaimsError as error:
        log.error("%s", error)
        return 1
    print(table)
    return 0
