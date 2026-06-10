# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "matplotlib>=3.7",
#   "slack-sdk>=3.27",
# ]
# ///
"""Biweekly metrics report for the ``kernels`` ecosystem.

Produces two graphs and posts them to a Slack channel:

1. Monthly download counts of the ``kernels`` PyPI package (via pypistats).
2. Total downloads of the kernels hosted on the Hugging Face Hub
   (the ``kernels-community`` org), tracked over time.

The PyPI graph is historical out of the box (pypistats serves ~180 days of
daily data). The Hub total-downloads graph is a time series that accrues one
data point per run; each run appends a snapshot to ``HISTORY_CSV`` which the
CI job commits back to the repository.

Run locally without posting to Slack:

    uv run scripts/report_kernels_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import requests  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration (overridable via environment variables)
# --------------------------------------------------------------------------- #
PYPI_PACKAGE = os.environ.get("PYPI_PACKAGE", "kernels")
# NB: there is no `huggingface.co/kernels` org; the kernels live under
# `kernels-community`. Override with HF_ORG if that ever changes.
HF_ORG = os.environ.get("HF_ORG", "kernels-community")

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_CSV = Path(
    os.environ.get(
        "HISTORY_CSV", REPO_ROOT / "scripts" / "metrics" / "hf_downloads_history.csv"
    )
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "scripts" / "metrics"))

USER_AGENT = "kernels-metrics-report (+https://github.com/huggingface/kernels)"
TIMEOUT = 30

# Dark, simple theme with solid, high-contrast colors.
BG = "#0d1117"  # near-black background
FG = "#e6edf3"  # primary text
GRID = "#30363d"  # subtle gridlines
MUTED = "#8b949e"  # secondary text
PYPI_COLOR = "#58a6ff"  # solid bright blue
HF_COLOR = "#ff9d00"  # solid HF amber
PARTIAL_COLOR = "#6e7681"  # dimmed marker for the in-progress month


# --------------------------------------------------------------------------- #
# Data fetching
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict | None = None) -> object:
    resp = requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def fetch_pypi_daily(package: str) -> "OrderedDict[str, int]":
    """Return an ordered ``{YYYY-MM-DD: downloads}`` mapping (no mirrors)."""
    payload = _get_json(
        f"https://pypistats.org/api/packages/{package}/overall",
        params={"mirrors": "false"},
    )
    daily: dict[str, int] = {}
    for row in payload["data"]:  # type: ignore[index]
        daily[row["date"]] = daily.get(row["date"], 0) + int(row["downloads"])
    return OrderedDict(sorted(daily.items()))


def aggregate_monthly(daily: "OrderedDict[str, int]") -> list[tuple[str, int, bool]]:
    """Aggregate daily counts into ``(YYYY-MM, downloads, is_partial)`` rows.

    The leading month is dropped when its data does not start on the 1st (it
    would understate the bar). The current month is flagged ``is_partial``.
    """
    if not daily:
        return []

    dates = list(daily.keys())
    first, last = dates[0], dates[-1]

    monthly: "OrderedDict[str, int]" = OrderedDict()
    for ds, dl in daily.items():
        ym = ds[:7]
        monthly[ym] = monthly.get(ym, 0) + dl

    months = list(monthly.items())
    # Drop the leading partial month (data didn't start on the 1st).
    if not first.endswith("-01") and len(months) > 1:
        months = months[1:]

    today = dt.date.fromisoformat(last)
    last_day_of_month = (today.replace(day=28) + dt.timedelta(days=4)).replace(
        day=1
    ) - dt.timedelta(days=1)
    current_partial = today < last_day_of_month
    current_ym = last[:7]

    return [(ym, dl, (ym == current_ym and current_partial)) for ym, dl in months]


def fetch_hf_downloads(org: str) -> tuple[int, int, int, list[tuple[str, int]]]:
    """Return ``(num_repos, total_30d, total_all_time, top_repos)`` for an org."""
    models = _get_json(
        "https://huggingface.co/api/models",
        params={
            "author": org,
            "limit": 1000,
            "expand[]": ["downloads", "downloadsAllTime"],
        },
    )
    total_30d = total_all = 0
    repos: list[tuple[str, int]] = []
    for m in models:  # type: ignore[union-attr]
        d30 = int(m.get("downloads") or 0)
        dall = int(m.get("downloadsAllTime") or 0)
        total_30d += d30
        total_all += dall
        repos.append((m["id"], d30))
    repos.sort(key=lambda r: r[1], reverse=True)
    return len(models), total_30d, total_all, repos[:5]  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# History (committed time series for the Hub totals)
# --------------------------------------------------------------------------- #
HISTORY_FIELDS = ["date", "hf_org", "num_repos", "downloads_30d", "downloads_all_time"]


def update_history(
    today: str, org: str, num_repos: int, d30: int, dall: int
) -> list[dict]:
    """Append (or replace) today's snapshot and return the full history."""
    rows: "OrderedDict[str, dict]" = OrderedDict()
    if HISTORY_CSV.exists():
        with HISTORY_CSV.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["date"]] = r
    rows[today] = {
        "date": today,
        "hf_org": org,
        "num_repos": str(num_repos),
        "downloads_30d": str(d30),
        "downloads_all_time": str(dall),
    }
    ordered = [rows[k] for k in sorted(rows)]
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    return ordered


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _new_dark_fig() -> tuple[plt.Figure, plt.Axes]:
    """Create a figure/axes pair styled with the dark, simple theme."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, which="both")
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    return fig, ax


def _save_dark(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_pypi_monthly(
    months: list[tuple[str, int, bool]], package: str, out: Path
) -> None:
    fig, ax = _new_dark_fig()
    labels = [ym for ym, _, _ in months]
    values = [v for _, v, _ in months]
    x = list(range(len(months)))

    ax.plot(x, values, color=PYPI_COLOR, linewidth=2.5, zorder=3)
    ax.fill_between(x, values, color=PYPI_COLOR, alpha=0.18, zorder=2)
    # Solid markers; the in-progress month gets a dimmed marker.
    marker_colors = [PARTIAL_COLOR if p else PYPI_COLOR for _, _, p in months]
    ax.scatter(
        x, values, color=marker_colors, edgecolor=BG, linewidth=1.2, s=55, zorder=4
    )
    for xi, (_, v, partial) in zip(x, months):
        ax.annotate(
            f"{v:,}" + ("*" if partial else ""),
            (xi, v),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=8,
            color=FG,
        )

    ax.set_title(
        f"PyPI monthly downloads — {package}",
        fontsize=14,
        fontweight="bold",
        color=FG,
        pad=12,
    )
    ax.set_ylabel("downloads")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.04)
    ax.set_ylim(bottom=0)
    if any(p for _, _, p in months):
        ax.text(
            0.99,
            -0.32,
            "* current month (partial)",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color=MUTED,
        )
    _save_dark(fig, out)


def plot_hf_history(history: list[dict], org: str, out: Path) -> None:
    dates = [dt.date.fromisoformat(r["date"]) for r in history]
    all_time = [int(r["downloads_all_time"]) for r in history]

    fig, ax = _new_dark_fig()
    if len(dates) == 1:
        ax.scatter(dates, all_time, color=HF_COLOR, edgecolor=BG, s=70, zorder=4)
        ax.set_xlim(dates[0] - dt.timedelta(days=21), dates[0] + dt.timedelta(days=21))
        ax.set_ylim(0, all_time[0] * 1.25)
    else:
        ax.plot(dates, all_time, color=HF_COLOR, linewidth=2.5, zorder=3)
        ax.scatter(
            dates, all_time, color=HF_COLOR, edgecolor=BG, linewidth=1.2, s=45, zorder=4
        )
        ax.fill_between(dates, all_time, color=HF_COLOR, alpha=0.15, zorder=2)
        ax.set_ylim(bottom=0)
    ax.annotate(
        f"{all_time[-1]:,}",
        (dates[-1], all_time[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="right",
        fontsize=9,
        fontweight="bold",
        color=FG,
    )

    ax.set_title(
        f"Total Hub downloads (all-time) — hf.co/{org}",
        fontsize=14,
        fontweight="bold",
        color=FG,
        pad=12,
    )
    ax.set_ylabel("cumulative downloads")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    if len(dates) == 1:
        ax.text(
            0.5,
            0.5,
            "history accrues per run —\nthe line fills in over time",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=MUTED,
        )
    _save_dark(fig, out)


# --------------------------------------------------------------------------- #
# Slack
# --------------------------------------------------------------------------- #
def post_to_slack(pypi_png: Path, hf_png: Path, comment: str) -> None:
    from slack_sdk import WebClient

    token = os.environ["SLACK_BOT_TOKEN"]
    channel = os.environ["SLACK_CHANNEL_ID"]
    client = WebClient(token=token)
    client.files_upload_v2(
        channel=channel,
        initial_comment=comment,
        file_uploads=[
            {
                "file": str(pypi_png),
                "title": "PyPI monthly downloads",
                "filename": pypi_png.name,
            },
            {
                "file": str(hf_png),
                "title": "Hub total downloads",
                "filename": hf_png.name,
            },
        ],
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write PNGs locally; do not post to Slack.",
    )
    args = parser.parse_args()

    today = os.environ.get("REPORT_DATE") or dt.date.today().isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching PyPI stats for '{PYPI_PACKAGE}'...")
    months = aggregate_monthly(fetch_pypi_daily(PYPI_PACKAGE))
    if not months:
        print("No PyPI data returned; aborting.", file=sys.stderr)
        return 1

    print(f"Fetching Hub downloads for org '{HF_ORG}'...")
    num_repos, d30, dall, top_repos = fetch_hf_downloads(HF_ORG)
    history = update_history(today, HF_ORG, num_repos, d30, dall)

    pypi_png = OUTPUT_DIR / "pypi_monthly_downloads.png"
    hf_png = OUTPUT_DIR / "hf_total_downloads.png"
    plot_pypi_monthly(months, PYPI_PACKAGE, pypi_png)
    plot_hf_history(history, HF_ORG, hf_png)
    print(f"Wrote {pypi_png} and {hf_png}")

    pypi_last = months[-1]
    top_lines = "\n".join(f"  • `{name}` — {dl:,} (30d)" for name, dl in top_repos)
    comment = (
        f":bar_chart: *kernels biweekly metrics* — {today}\n"
        f"*PyPI `{PYPI_PACKAGE}`*: {pypi_last[1]:,} downloads in {pypi_last[0]}"
        f"{' (partial)' if pypi_last[2] else ''}\n"
        f"*hf.co/{HF_ORG}*: {dall:,} all-time · {d30:,} in last 30d · {num_repos} repos\n"
        f"Top repos (30d):\n{top_lines}"
    )

    if args.dry_run:
        print("\n--- DRY RUN (not posting) ---")
        print(comment)
        return 0

    print("Posting to Slack...")
    post_to_slack(pypi_png, hf_png, comment)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
