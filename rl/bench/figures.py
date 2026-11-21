"""Leaderboard figures.

``sharpe.png`` shows, for every historical task, each strategy's per-seed
Sharpe ratios against the 1/N benchmark, so seed dispersion is visible next
to the gap (or lack of one) to doing nothing clever. Published single-run
results listed in ``reported.json`` are drawn as stars on their task.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl.bench import scoring
from rl.bench.tasks import TASKS

REPORTED_FILE = "reported.json"
_KIND_COLORS = {"rl": "#d62728", "forecast": "#9467bd", "baseline": "#1f77b4"}


def _equity_sharpe(equity, periods_per_year) -> float:
    equity = np.asarray(equity, dtype=float)
    returns = equity[1:] / equity[:-1] - 1
    if returns.std() == 0:
        return float("nan")
    return scoring.annualised_sharpe(returns, periods_per_year)


def _panels(records: list[dict]) -> list[dict]:
    by_task: dict[str, list[dict]] = {}
    for record in records:
        if record["track"] == "historical" and TASKS.get(record["task"]) and TASKS[record["task"]].source != "synthetic":
            by_task.setdefault(record["task"], []).append(record)
    panels = []
    for task_id in sorted(by_task, key=lambda t: (not TASKS[t].live, t)):
        rows = []
        benchmark = None
        for record in by_task[task_id]:
            ppy = record["periods_per_year"]
            sharpes = [_equity_sharpe(run["equity"], ppy) for run in record["runs"]]
            if not np.isfinite(sharpes).any():
                continue
            rows.append({"id": record["submission"], "name": record["name"], "kind": record["kind"], "sharpes": sharpes})
            benchmark = _equity_sharpe(record["benchmark"]["equity"], ppy)
        rows.sort(key=lambda row: np.nanmean(row["sharpes"]))
        last = max(record["dates"][-1] for record in by_task[task_id])
        panels.append({"task": task_id, "rows": rows, "benchmark": benchmark, "last": last})
    return panels


def render_sharpe_figure(records: list[dict], path, reported: dict | None = None) -> Path | None:
    """Draw per-seed Sharpe ratios per strategy and task; ``None`` if nothing to draw."""
    panels = [panel for panel in _panels(records) if panel["rows"]]
    if not panels:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reported = reported or {}
    columns = min(2, len(panels))
    grid_rows = -(-len(panels) // columns)
    height = max(len(panel["rows"]) for panel in panels) * 0.3 + 1.2
    fig, axes = plt.subplots(grid_rows, columns, figsize=(6.2 * columns, height * grid_rows), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, panel in zip(axes.flat, panels):
        stars = reported.get(panel["task"], {}).get("sharpe", {})
        for y, row in enumerate(panel["rows"]):
            color = _KIND_COLORS.get(row["kind"], "#7f7f7f")
            ax.scatter(row["sharpes"], [y] * len(row["sharpes"]), s=22, color=color, alpha=0.75, zorder=3)
            if row["id"] in stars:
                ax.scatter([stars[row["id"]]], [y], marker="*", s=140, color="#ff7f0e", zorder=4)
        if panel["benchmark"] is not None and np.isfinite(panel["benchmark"]):
            ax.axvline(panel["benchmark"], color="black", linestyle="--", linewidth=1, label="1/N buy and hold")
        ax.set_yticks(range(len(panel["rows"])))
        ax.set_yticklabels([row["name"] for row in panel["rows"]], fontsize=8)
        task = TASKS[panel["task"]]
        years = "{} to {}".format(task.test_start[:4], panel["last"][:4])
        title = task.title if task.live else "{}, survivorship-biased".format(task.title)
        ax.set_title("{}\n{}".format(title, years), fontsize=9)
        ax.set_xlabel("Sharpe ratio (each dot is a seed)", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=kind)
        for kind, color in _KIND_COLORS.items()
    ]
    handles.append(plt.Line2D([], [], color="black", linestyle="--", label="1/N buy and hold"))
    if any(reported.get(panel["task"]) for panel in panels):
        handles.append(plt.Line2D([], [], marker="*", linestyle="", markersize=11, color="#ff7f0e", label="reported in paper (1 run)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_reported(root) -> dict:
    path = Path(root) / REPORTED_FILE
    return json.loads(path.read_text()) if path.exists() else {}
