"""Build Figure 2 from the daily comment counts.

The co-mention share is defined as co-mention
comments divided by all climate-related comments on the same day, where all
climate-related comments equal climate-only plus co-mention comments. The
Pearson correlation is saved with the descriptive summary.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_DIR
    / "Data"
    / "daily_comment_volume_and_covid_deaths.csv"
)
OUTPUT_DIR = PROJECT_DIR / "Figures"
SOURCE_OUTPUT = PROJECT_DIR / "Data" / "Figure_2_source_data.csv"
AUDIT_OUTPUT = PROJECT_DIR / "QA" / "Figure_2_audit.json"
OUTPUT = OUTPUT_DIR / "Figure_2"

GREEN = "#008B72"
ORANGE = "#D55E00"


def load_and_prepare() -> pd.DataFrame:
    frame = pd.read_csv(DATA_FILE)
    required = {
        "date",
        "climate_only_comments",
        "covid_only_comments",
        "co_mention_comments",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) != 711 or frame["date"].duplicated().any():
        raise ValueError("Expected 711 unique daily observations")
    if (frame["climate_only_comments"] <= 0).any():
        raise ValueError("Climate-only volume must be positive on every date")

    frame["co_mention_share_among_climate_comments"] = (
        frame["co_mention_comments"]
        / (frame["climate_only_comments"] + frame["co_mention_comments"])
    )
    frame["climate_only_comments_ma7"] = frame[
        "climate_only_comments"
    ].rolling(window=7, min_periods=1).mean()
    frame["co_mention_share_among_climate_comments_ma7"] = frame[
        "co_mention_share_among_climate_comments"
    ].rolling(window=7, min_periods=1).mean()
    return frame


def save_source_and_audit(frame: pd.DataFrame) -> None:
    SOURCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "date",
        "climate_only_comments",
        "covid_only_comments",
        "co_mention_comments",
        "co_mention_share_among_climate_comments",
        "climate_only_comments_ma7",
        "co_mention_share_among_climate_comments_ma7",
    ]
    frame[columns].to_csv(SOURCE_OUTPUT, index=False)

    correlation, pvalue = pearsonr(
        frame["climate_only_comments"],
        frame["co_mention_share_among_climate_comments"],
    )
    audit = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(DATA_FILE),
        "source_data": str(SOURCE_OUTPUT),
        "analysis_window": [
            frame["date"].min().strftime("%Y-%m-%d"),
            frame["date"].max().strftime("%Y-%m-%d"),
        ],
        "daily_rows": len(frame),
        "climate_only_comments": int(frame["climate_only_comments"].sum()),
        "co_mention_comments": int(frame["co_mention_comments"].sum()),
        "share_definition": (
            "daily co-mention comments / (daily climate-only comments + "
            "daily co-mention comments)"
        ),
        "smoothing": "7-day trailing moving average with min_periods=1",
        "descriptive_pearson_r_raw_daily_series": float(correlation),
        "descriptive_pearson_pvalue": float(pvalue),
    }
    AUDIT_OUTPUT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def draw(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12.5,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, left = plt.subplots(figsize=(10, 6))
    right = left.twinx()
    right.spines["top"].set_visible(False)

    climate_line = left.plot(
        frame["date"],
        frame["climate_only_comments_ma7"],
        color=GREEN,
        linewidth=2.4,
        label="Climate-only Comments (7-day MA)",
    )[0]
    ratio_line = right.plot(
        frame["date"],
        frame["co_mention_share_among_climate_comments_ma7"],
        color=ORANGE,
        linewidth=2.4,
        alpha=0.9,
        label="Co-mention Share (7-day MA)",
    )[0]

    left.set_ylabel(
        "Climate-only Comment Volume",
        color=GREEN,
        fontsize=13.5,
        fontweight="bold",
    )
    right.set_ylabel(
        "Co-mention Share of Climate-related Comments",
        color=ORANGE,
        fontsize=13.5,
        fontweight="bold",
    )
    left.tick_params(axis="y", colors=GREEN)
    right.tick_params(axis="y", colors=ORANGE)
    right.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    right.set_ylim(bottom=0)

    left.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    left.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    left.set_xlim(frame["date"].min(), frame["date"].max())
    left.legend(
        [climate_line, ratio_line],
        [climate_line.get_label(), ratio_line.get_label()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        fontsize=11.8,
    )

    fig.subplots_adjust(left=0.11, right=0.88, top=0.97, bottom=0.18)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        OUTPUT.with_suffix(".jpeg"),
        format="jpeg",
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    frame = load_and_prepare()
    save_source_and_audit(frame)
    draw(frame)
    print(f"Figure written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
