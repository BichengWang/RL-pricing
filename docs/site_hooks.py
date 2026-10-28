"""MkDocs hooks that build the GitHub Pages site straight from the repository.

The site publishes the README (as the home page), every report under
``reports/``, the leaderboard with its monthly live reports, and ``docs/``.
``on_config`` writes the navigation from what is on disk, so a new report or
monthly report appears without editing ``mkdocs.yml``. ``on_files`` swaps the
docs-directory listing for those repository files, keeping their repository
paths, so the links between them work unchanged. ``on_page_markdown`` points
every other relative link (submissions, code, JSON evidence) at GitHub.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File, InclusionLevel

REPOSITORY = "https://github.com/BichengWang/RL-pricing"
BRANCH = "main"
# Repository files published under another site path, for cleaner URLs.
SITE_PATHS = {"README.md": "index.md", "leaderboard/LEADERBOARD.md": "leaderboard/index.md"}

# Published files, as repository-relative globs; everything else links to GitHub.
SOURCES = (
    "docs/*.md",
    "docs/*.pdf",
    "docs/*.svg",
    "reports/README.md",
    "reports/*/*.md",
    "reports/*/evidence/*.md",
    "reports/**/*.png",
    "reports/**/*.svg",
    "leaderboard/*.md",
    "leaderboard/*.png",
    "leaderboard/reports/*.md",
    "leaderboard/reports/*.png",
)

# Pages of a report folder, in navigation order; other pages follow by name.
REPORT_PAGES = {
    "report.md": "Report",
    "preregistration.md": "Pre-registration",
    "reply.md": "Authors' reply",
    "evidence/tables.md": "Evidence tables",
}

_published: dict[str, Path] = {}  # site path -> source file
_LINK = re.compile(r"(\]\()(<[^>]*>|[^)\s]+)")
_REFERENCE = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*)(\S+)", re.MULTILINE)
_FENCE = re.compile(r"^\s*(```|~~~)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_HEADING = re.compile(r"^# (.+?)\s*$", re.MULTILINE)


def _site_path(repository_path: str) -> str:
    return SITE_PATHS.get(repository_path, repository_path)


def _collect(root: Path) -> dict[str, Path]:
    found = {"README.md": root / "README.md"}
    for pattern in SOURCES:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                found[path.relative_to(root).as_posix()] = path
    return {_site_path(name): path for name, path in found.items()}


def _nav(published: dict[str, Path]) -> list:
    nav: list = [{"Home": "index.md"}]
    if "reports/README.md" in published:
        nav.append({"Verdicts": "reports/README.md"})
    reports = []
    for folder in sorted({name.split("/")[1] for name in published if name.count("/") >= 2 and name.startswith("reports/")}):
        prefix = "reports/{}/".format(folder)
        pages = [name for name in published if name.startswith(prefix) and name.endswith(".md")]
        order = list(REPORT_PAGES)
        pages.sort(key=lambda name: (order.index(name[len(prefix):]) if name[len(prefix):] in REPORT_PAGES else len(order), name))
        entries = [{REPORT_PAGES.get(name[len(prefix):], Path(name).stem.replace("-", " ").capitalize()): name} for name in pages]
        if entries:
            reports.append({folder: entries})
    if reports:
        nav.append({"Reports": reports})
    if "leaderboard/index.md" in published:
        nav.append({"Leaderboard": "leaderboard/index.md"})
    monthly = sorted((name for name in published if name.startswith("leaderboard/reports/") and name.endswith(".md")), reverse=True)
    if monthly:
        nav.append({"Live track": [{Path(name).stem: name} for name in monthly]})
    if "docs/toolkit.md" in published:
        nav.append({"Training toolkit": "docs/toolkit.md"})
    return nav


def on_config(config):
    root = Path(config.config_file_path).parent
    _published.clear()
    _published.update(_collect(root))
    config.nav = _nav(_published)
    return config


def on_files(files, config):
    docs_dir = Path(config.docs_dir).resolve()
    for file in list(files):
        if file.abs_src_path and Path(file.abs_src_path).resolve().is_relative_to(docs_dir):
            files.remove(file)
    for name, path in _published.items():
        files.append(File.generated(config, name, abs_src_path=str(path), inclusion=InclusionLevel.INCLUDED))
    return files


def _rewrite(target: str, page_dir: str, root: Path) -> str:
    bracketed = target.startswith("<") and target.endswith(">")
    link = target[1:-1] if bracketed else target
    if not link or link.startswith(("#", "/")) or _SCHEME.match(link):
        return target
    path, hash_, anchor = link.partition("#")
    resolved = posixpath.normpath(posixpath.join(page_dir, path))
    if resolved.startswith(".."):
        return target
    site = _site_path(resolved)
    if site in _published:
        rewritten = posixpath.relpath(site, page_dir or ".")
    elif (root / resolved).exists():
        kind = "tree" if (root / resolved).is_dir() else "blob"
        rewritten = "{}/{}/{}/{}".format(REPOSITORY, kind, BRANCH, "" if resolved == "." else resolved).rstrip("/")
    else:
        return target
    rewritten += hash_ + anchor
    return "<{}>".format(rewritten) if bracketed else rewritten


def on_page_markdown(markdown, page, config, files):
    # Navigation labels stay short; the page title, link previews and social
    # card use the page's own heading.
    heading = _HEADING.search(markdown)
    if heading and "title" not in page.meta:
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.group(1)).replace("`", "").replace("**", "")
        # The social plugin writes the title into meta attributes unescaped.
        page.meta["title"] = re.sub(r'"([^"]*)"', "\u201c\\1\u201d", title).replace('"', "\u201d")
    root = Path(config.config_file_path).parent
    page_dir = posixpath.dirname(page.file.src_uri)
    lines, fenced = [], False
    for line in markdown.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            line = _LINK.sub(lambda m: m.group(1) + _rewrite(m.group(2), page_dir, root), line)
            line = _REFERENCE.sub(lambda m: m.group(1) + _rewrite(m.group(2), page_dir, root), line)
        lines.append(line)
    return "".join(lines)
