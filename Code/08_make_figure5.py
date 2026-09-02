"""Build the two-panel qualitative Figure 5 from the coded counts.

Panel a shows the frequency of the five interpretive patterns. Panel b retains
the same category colours and uses colour depth to encode quarterly counts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data" / "Figure_5_source_data.csv"
OUT_DIR = ROOT / "Figures"
OUT_STEM = OUT_DIR / "Figure_5"

INK = "#20262D"
MUTED = "#5B6878"
GRID = "#E4E9EC"

THEME_ORDER = ["C1", "C2", "C3", "C4", "C5"]
THEME_COLORS = {
    "C1": "#4477AA",
    "C2": "#D99A2B",
    "C3": "#168A74",
    "C4": "#C45C43",
    "C5": "#83699A",
}
THEME_LABELS = {
    "C1": "Crisis comparison and\nattention competition",
    "C2": "Causal links and\nshared drivers",
    "C3": "Science, misinformation\nand politicization",
    "C4": "Governance, collective action\nand policy capacity",
    "C5": "Social consequences, inequality\nand future transition",
}
QUARTERS = [
    "2020-Q1",
    "2020-Q2",
    "2020-Q3",
    "2020-Q4",
    "2021-Q1",
    "2021-Q2",
    "2021-Q3",
    "2021-Q4",
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_source() -> tuple[pd.Series, pd.DataFrame]:
    source = pd.read_csv(SOURCE)
    totals = (
        source.loc[source["record_type"].eq("theme_total")]
        .set_index("code")["count"]
        .reindex(THEME_ORDER)
        .astype(int)
    )
    quarter = source.loc[source["record_type"].eq("quarter_theme_count")].copy()
    matrix = (
        quarter.pivot(index="code", columns="quarter", values="count")
        .reindex(index=THEME_ORDER, columns=QUARTERS)
        .astype(int)
    )

    expected = {"C1": 31, "C2": 15, "C3": 70, "C4": 46, "C5": 40}
    if totals.to_dict() != expected:
        raise ValueError(f"Unexpected theme totals: {totals.to_dict()}")
    if int(totals.sum()) != 202:
        raise ValueError("Substantive category counts must sum to 202")
    if matrix.shape != (5, 8) or (matrix <= 0).any().any():
        raise ValueError("Expected positive counts for five categories across eight quarters")
    return totals, matrix


def draw_panel_a(ax: plt.Axes, totals: pd.Series) -> None:
    y = np.arange(len(THEME_ORDER))
    colors = [THEME_COLORS[code] for code in THEME_ORDER]
    ax.barh(
        y,
        totals.values,
        color=colors,
        height=0.62,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{code}  {THEME_LABELS[code]}" for code in THEME_ORDER], fontsize=7.2
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 78)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Coded comments in the formal sample", fontsize=7.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7C858C")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.set_title(
        "Five recurrent interpretive patterns",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        pad=7,
    )
    ax.text(
        -0.15,
        1.09,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    for yy, value in zip(y, totals.values):
        ax.text(
            value + 1.2,
            yy,
            f"{value}",
            ha="left",
            va="center",
            fontsize=7.5,
            color=INK,
        )


def draw_panel_b(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    max_count = float(matrix.to_numpy().max())
    rgba = np.ones((matrix.shape[0], matrix.shape[1], 4), dtype=float)
    for row, code in enumerate(THEME_ORDER):
        base = np.asarray(to_rgb(THEME_COLORS[code]))
        for column in range(matrix.shape[1]):
            value = float(matrix.iat[row, column])
            strength = 0.14 + 0.78 * (value / max_count) ** 0.72
            rgba[row, column, :3] = 1.0 - strength * (1.0 - base)

    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [label.replace("-", " ") for label in matrix.columns],
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.0,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7.4)
    for tick, code in zip(ax.get_yticklabels(), THEME_ORDER):
        tick.set_color(THEME_COLORS[code])
        tick.set_fontweight("bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix.iat[row, column])
            red, green, blue = rgba[row, column, :3]
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if luminance < 0.56 else INK,
                fontweight="bold" if value >= 10 else "normal",
            )

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(3.5, color="white", lw=2.4)
    ax.set_title(
        "Temporal coverage of the coded sample",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        pad=7,
    )
    ax.text(
        -0.10,
        1.09,
        "b",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )


def save_figure(fig: plt.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "facecolor": "white"}
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, **common)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), **common)
    fig.savefig(OUT_STEM.with_suffix(".svg"), **common)
    fig.savefig(
        OUT_STEM.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )


def build_composite(totals: pd.Series, matrix: pd.DataFrame) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.4, 4.1),
        gridspec_kw={"width_ratios": [0.47, 0.53], "wspace": 0.40},
    )
    draw_panel_a(ax_a, totals)
    draw_panel_b(ax_b, matrix)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.86, bottom=0.17)
    save_figure(fig)
    plt.close(fig)


def write_audit(totals: pd.Series, matrix: pd.DataFrame) -> None:
    audit = ROOT / "QA" / "Figure5_data_and_export_audit.txt"
    audit.parent.mkdir(parents=True, exist_ok=True)
    outputs = [OUT_STEM.with_suffix(ext) for ext in (".png", ".pdf", ".svg", ".tiff")]
    lines = [
        "Figure 5 data and export audit",
        "================================",
        f"Source: {SOURCE}",
        f"Theme totals: {totals.to_dict()}",
        f"Substantive total: {int(totals.sum())}",
        f"Quarter matrix shape: {matrix.shape}",
        f"Every category present in every quarter: {bool((matrix > 0).all().all())}",
        f"Theme palette: {THEME_COLORS}",
        "Panel a: theme-specific bars and exact coded-comment counts",
        "Panel b: theme hue by row; within-row colour depth encodes count",
        "Multiple-comparison or significance encoding: not applicable",
        "",
        "Exports:",
    ]
    for path in outputs:
        lines.append(f"- {path.name}: {path.stat().st_size} bytes")
    audit.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    totals, matrix = load_source()
    build_composite(totals, matrix)
    write_audit(totals, matrix)
    print(OUT_STEM)


if __name__ == "__main__":
    main()
