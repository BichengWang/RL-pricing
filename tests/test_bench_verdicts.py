"""Verdict table: claims.json validation, rendering and README block replacement."""

import json
from argparse import ArgumentParser
from pathlib import Path

import pytest

from rl.bench import verdicts
from rl.bench.verdicts import ClaimsError

REPO = Path(__file__).resolve().parents[1]


def _claim(**overrides):
    claim = {
        "id": "sharpe",
        "quote": "Sharpe 1.5",
        "source": "Table 2",
        "metric": "sharpe",
        "claimed": 1.5,
        "ours": 0.8,
        "ci95": [0.4, 1.2],
        "level": "L2",
        "verdict": "does-not-hold",
        "note": "no seed reaches 1.5",
    }
    claim.update(overrides)
    return claim


def _write(root, report_id, claims):
    folder = Path(root) / report_id
    folder.mkdir(parents=True)
    (folder / "claims.json").write_text(json.dumps(claims))
    return folder / "claims.json"


# --- validation ------------------------------------------------------------


def test_valid_claims_load(tmp_path):
    path = _write(tmp_path, "r1", [_claim(), _claim(id="drawdown", metric="max_drawdown", ours=None, ci95=None)])
    claims = verdicts.load_claims(path)
    assert [c.id for c in claims] == ["sharpe", "drawdown"]
    assert claims[0].ci95 == (0.4, 1.2) and claims[1].ours is None and claims[1].ci95 is None


def test_bad_verdict_is_a_clear_error(tmp_path):
    path = _write(tmp_path, "r1", [_claim(), _claim(id="second", verdict="maybe")])
    with pytest.raises(ClaimsError) as error:
        verdicts.load_claims(path)
    message = str(error.value)
    assert str(path) in message and "claim #2" in message and "'second'" in message
    assert "'maybe'" in message and "does-not-hold" in message


def test_missing_field_names_the_field(tmp_path):
    claim = _claim()
    del claim["quote"], claim["note"]
    path = _write(tmp_path, "r1", [claim])
    with pytest.raises(ClaimsError, match=r"claim #1 \('sharpe'\) is missing field\(s\): quote, note"):
        verdicts.load_claims(path)


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"level": "L4"}, "level 'L4'"),
        ({"claimed": "1.5"}, "'claimed' must be a finite number"),
        ({"claimed": float("nan")}, "'claimed' must be a finite number"),
        ({"claimed": True}, "'claimed' must be a finite number"),
        ({"ours": "n/a"}, "'ours' must be a finite number or null"),
        ({"ci95": [1.0]}, "'ci95' must be [low, high] or null"),
        ({"ci95": [2.0, 1.0]}, "low 2.0 exceeds high 1.0"),
        ({"quote": " "}, "'quote' must be a non-empty string"),
        ({"extra": 1}, "unknown field(s): extra"),
    ],
)
def test_field_errors(tmp_path, overrides, fragment):
    path = _write(tmp_path, "r1", [_claim(**overrides)])
    with pytest.raises(ClaimsError) as error:
        verdicts.load_claims(path)
    assert fragment in str(error.value)


def test_file_level_errors(tmp_path):
    duplicate = _write(tmp_path, "dup", [_claim(), _claim()])
    with pytest.raises(ClaimsError, match="duplicate claim id 'sharpe'"):
        verdicts.load_claims(duplicate)
    empty = _write(tmp_path, "empty", [])
    with pytest.raises(ClaimsError, match="non-empty list"):
        verdicts.load_claims(empty)
    broken = tmp_path / "broken" / "claims.json"
    broken.parent.mkdir()
    broken.write_text("{not json")
    with pytest.raises(ClaimsError, match="invalid JSON"):
        verdicts.load_claims(broken)
    with pytest.raises(ClaimsError, match="must be an object"):
        verdicts.load_claims(_write(tmp_path, "scalar", [3]))


# --- rendering -------------------------------------------------------------


def test_table_order_symbols_and_formats(tmp_path):
    _write(tmp_path, "r2-second", [_claim(id="only", verdict="inconclusive")])
    _write(
        tmp_path,
        "r1-first",
        [
            _claim(id="z-holds", verdict="holds"),
            _claim(id="a-partly", verdict="partly"),
            _claim(id="ret", metric="annual_return", claimed=0.5551, ours=0.246, ci95=[0.17, 0.32], verdict="does-not-hold"),
            _claim(id="dd", metric="max_drawdown", claimed=-0.0957, ours=None, ci95=None, verdict="inconclusive"),
        ],
    )
    table = verdicts.render_table(verdicts.load_reports(tmp_path))
    rows = [line for line in table.splitlines() if line.startswith("| [")]
    assert [row.split(" | ")[1] for row in rows] == [
        "`z-holds` (sharpe)",
        "`a-partly` (sharpe)",
        "`ret` (annual_return)",
        "`dd` (max_drawdown)",
        "`only` (sharpe)",
    ]
    assert rows[0].startswith("| [r1-first](r1-first/report.md) | ")
    assert rows[4].startswith("| [r2-second](r2-second/report.md) | ")
    assert rows[0].endswith("| ✅ holds |") and rows[1].endswith("| ◐ partly holds |")
    assert rows[2].endswith("| ✗ does not hold |") and rows[3].endswith("| ? inconclusive |")
    assert "| L2 | 1.50 | 0.80 [0.40, 1.20] |" in rows[0]
    assert "| L2 | 55.5% | 24.6% [17.0%, 32.0%] |" in rows[2]
    assert "| L2 | -9.6% | – |" in rows[3]
    assert table.splitlines()[0] == "| Report | Claim | Level | Claimed | Ours (95% interval) | Verdict |"
    assert "✅ holds, ◐ partly holds, ✗ does not hold, ? inconclusive" in table


def test_table_link_prefix_and_pipe_escape(tmp_path):
    _write(tmp_path, "r1", [_claim(metric="a|b")])
    table = verdicts.render_table(verdicts.load_reports(tmp_path), link_prefix="reports/")
    assert "[r1](reports/r1/report.md)" in table
    assert "(a\\|b)" in table
    assert "(reports/README.md#how-a-verdict-is-decided)" in table


def test_pending_claims_validate_but_are_not_rendered(tmp_path):
    _write(tmp_path, "r1", [_claim(id="decided"), _claim(id="pending", ours=None, ci95=None, verdict=None)])
    _write(tmp_path, "r2", [_claim(id="all-pending", ours=None, ci95=None, verdict=None)])
    reports = verdicts.load_reports(tmp_path)
    assert [c.verdict for c in reports[0].claims] == ["does-not-hold", None]
    table = verdicts.render_table(reports)
    assert "`decided`" in table and "pending" not in table and "[r2]" not in table
    assert verdicts.render_table(reports[1:]) == "No reports yet."


def test_no_reports():
    assert verdicts.render_table([]) == "No reports yet."


# --- README blocks ---------------------------------------------------------

_DOC = "intro\n{}\nold\n{}\noutro\n".format(verdicts.README_START, verdicts.README_END)


def test_update_block_replaces_and_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(_DOC)
    assert verdicts.update_block(readme, "table\n")
    once = readme.read_text()
    assert once == "intro\n{}\ntable\n{}\noutro\n".format(verdicts.README_START, verdicts.README_END)
    assert verdicts.update_block(readme, "table\n")
    assert readme.read_text() == once
    assert verdicts.update_block(readme, "other")
    assert "table" not in readme.read_text() and readme.read_text().count(verdicts.README_START) == 1


def test_update_block_without_markers_or_file(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")
    assert not verdicts.update_block(readme, "x")
    assert readme.read_text() == "no markers\n"
    assert not verdicts.update_block(tmp_path / "missing.md", "x")


def test_write_reports_updates_both_readmes_idempotently(tmp_path):
    root = tmp_path / "reports"
    _write(root, "r1", [_claim()])
    (root / "README.md").write_text(_DOC)
    main = tmp_path / "README.md"
    main.write_text(_DOC)
    table = verdicts.write_reports(root, readme=main)
    inner, outer = (root / "README.md").read_text(), main.read_text()
    assert "[r1](r1/report.md)" in inner and table in inner
    assert "[r1]({}/r1/report.md)".format(root.as_posix()) in outer
    verdicts.write_reports(root, readme=main)
    assert (root / "README.md").read_text() == inner and main.read_text() == outer


def test_write_reports_tolerates_readme_without_markers(tmp_path):
    root = tmp_path / "reports"
    _write(root, "r1", [_claim()])
    (root / "README.md").write_text(_DOC)
    main = tmp_path / "README.md"
    main.write_text("no markers\n")
    verdicts.write_reports(root, readme=main)
    assert main.read_text() == "no markers\n"


# --- CLI -------------------------------------------------------------------


def _parser():
    parser = ArgumentParser()
    verdicts.add_cli(parser.add_subparsers(dest="command"), None)
    return parser


def test_cli_renders_and_fails_on_bad_claims(tmp_path, capsys):
    root = tmp_path / "reports"
    path = _write(root, "r1", [_claim()])
    options = _parser().parse_args(["reports", "--root", str(root), "--no-readme"])
    assert options.handler(options) == 0
    assert "`sharpe` (sharpe)" in capsys.readouterr().out
    path.write_text(json.dumps([_claim(verdict="nope")]))
    assert options.handler(options) == 1


# --- the repository's own reports -----------------------------------------


def test_r0_is_in_the_generated_readme():
    reports = verdicts.load_reports(REPO / "reports")
    assert "r0-wang-zhang-2026" in [report.id for report in reports]
    r0 = next(report for report in reports if report.id == "r0-wang-zhang-2026")
    assert [c.id for c in r0.claims] == [
        "ddpg-annual-return", "ddpg-sharpe", "td3-sharpe", "sac-sharpe", "ppo-sharpe", "a2c-sharpe",
        "lstm-max-drawdown", "lstm-sharpe",
    ]
    text = (REPO / "reports" / "README.md").read_text()
    block = text.split(verdicts.README_START)[1].split(verdicts.README_END)[0].strip("\n")
    assert block == verdicts.render_table(reports)
    assert (REPO / "reports" / "r0-wang-zhang-2026" / "report.md").exists()
