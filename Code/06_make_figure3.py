"""Build Figure 3 from the daily analysis dataset.

The figure compares three discourse groups across ten NRC measures, five
LIWC-22 measures, and two VADER
sentiment measures. Each violin contains 711 date-level means. Friedman tests
pair observations by calendar day, and exact two-sided p values are reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare


ROOT = Path(__file__).resolve().parents[1]
DAILY_FILE = ROOT / "Data" / "daily_emotion_scores.csv"
VOLUME_FILE = ROOT / "Data" / "daily_comment_volume_and_covid_deaths.csv"
OUTPUT_DIR = ROOT / "Figures"
OUTPUT_STEM = OUTPUT_DIR / "Figure_3"
SOURCE_OUTPUT = ROOT / "Data" / "Figure_3_source_data.csv"
SUPPLEMENT_SOURCE_DIR = ROOT / "Data" / "Supplementary_Table_Sources"
FRIEDMAN_OUTPUT = SUPPLEMENT_SOURCE_DIR / "Supplementary_Table_4_Friedman_Tests_source.csv"
MEANS_OUTPUT = SUPPLEMENT_SOURCE_DIR / "Supplementary_Table_5_Weighted_Descriptive_Means_source.csv"
AUDIT_OUTPUT = ROOT / "QA" / "Figure_3_audit.json"

INK = "#20262E"
MUTED = "#5F6B76"
GRID = "#D8DEE6"
GROUP_ORDER = ("Climate-only", "COVID-only", "Co-mention")
GROUP_COLORS = {
    "Climate-only": "#00897B",
    "COVID-only": "#D97706",
    "Co-mention": "#2F6F9F",
}
GROUP_KEYS = {
    "climate": "Climate-only",
    "covid": "COVID-only",
    "both": "Co-mention",
}
COUNT_COLUMNS = {
    "climate": "climate_only_comments",
    "covid": "covid_only_comments",
    "both": "co_mention_comments",
}

# panel label, title, measurement family, displayed measure, column suffix
PANELS = (
    (
        "a",
        "NRC negative emotion",
        "NRC",
        (
            ("Sadness", "sadness_score_freq"),
            ("Anger", "anger_score_freq"),
            ("Fear", "fear_score_freq"),
            ("Disgust", "disgust_score_freq"),
            ("Negative", "negative_score_freq"),
        ),
    ),
    (
        "b",
        "NRC positive emotion",
        "NRC",
        (
            ("Joy", "joy_score_freq"),
            ("Anticipation", "anticipation_score_freq"),
            ("Trust", "trust_score_freq"),
            ("Surprise", "surprise_score_freq"),
            ("Positive", "positive_score_freq"),
        ),
    ),
    (
        "c",
        "LIWC",
        "LIWC-22",
        (
            ("Anger", "liwc_emo_anger_pct"),
            ("Sadness", "liwc_emo_sad_pct"),
            ("Anxiety", "liwc_emo_anx_pct"),
            ("Negative", "liwc_emo_neg_pct"),
            ("Positive", "liwc_emo_pos_pct"),
        ),
    ),
    (
        "d",
        "VADER",
        "VADER",
        (
            ("Negative", "vader_negative_pct"),
            ("Positive", "vader_positive_pct"),
        ),
    ),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.75,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 8.2,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    volume = pd.read_csv(VOLUME_FILE, parse_dates=["date"])
    merged = daily.merge(
        volume[["date", *COUNT_COLUMNS.values()]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 711 or merged["date"].nunique() != 711:
        raise ValueError(f"Expected 711 paired dates, found {len(merged):,}")
    if merged["date"].min() != pd.Timestamp("2020-01-21"):
        raise ValueError("Unexpected analysis start date")
    if merged["date"].max() != pd.Timestamp("2021-12-31"):
        raise ValueError("Unexpected analysis end date")

    records: list[pd.DataFrame] = []
    tests: list[dict[str, float | str | int]] = []
    for _, _, family, measures in PANELS:
        for measure, suffix in measures:
            arrays = []
            for group_key, group_label in GROUP_KEYS.items():
                column = f"{group_key}_{suffix}"
                if column not in merged:
                    raise KeyError(f"Missing required column: {column}")
                arrays.append(merged[column])
                records.append(
                    pd.DataFrame(
                        {
                            "date": merged["date"],
                            "family": family,
                            "measure": measure,
                            "group": group_label,
                            "group_key": group_key,
                            "score_percent": merged[column],
                            "n_comments": merged[COUNT_COLUMNS[group_key]],
                        }
                    )
                )
            statistic, p_value = friedmanchisquare(*arrays)
            tests.append(
                {
                    "family": family,
                    "measure": measure,
                    "n_paired_dates": len(merged),
                    "friedman_chi_square": float(statistic),
                    "degrees_of_freedom": 2,
                    "p_value": float(p_value),
                }
            )

    long_data = pd.concat(records, ignore_index=True)
    test_data = pd.DataFrame(tests)
    if len(long_data) != 17 * 3 * 711:
        raise ValueError(f"Unexpected long-data row count: {len(long_data):,}")
    if long_data["score_percent"].isna().any():
        raise ValueError("Figure source data contain missing scores")
    return long_data, test_data


def weighted_means(data: pd.DataFrame) -> pd.DataFrame:
    values = data.assign(weighted=data["score_percent"] * data["n_comments"])
    summary = (
        values.groupby(["family", "measure", "group"], as_index=False)
        .agg(weighted_sum=("weighted", "sum"), n_comments=("n_comments", "sum"))
    )
    summary["weighted_mean_percent"] = summary["weighted_sum"] / summary["n_comments"]
    return summary


def add_weighted_means(
    axis: plt.Axes,
    summary: pd.DataFrame,
    family: str,
    measures: tuple[tuple[str, str], ...],
) -> None:
    offsets = {"Climate-only": -0.27, "COVID-only": 0.0, "Co-mention": 0.27}
    for measure_index, (measure, _) in enumerate(measures):
        for group in GROUP_ORDER:
            row = summary.loc[
                (summary["family"] == family)
                & (summary["measure"] == measure)
                & (summary["group"] == group)
            ].iloc[0]
            axis.scatter(
                measure_index + offsets[group],
                float(row["weighted_mean_percent"]),
                marker="D",
                s=13,
                facecolor=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.55,
                zorder=5,
            )


def save_bundle(figure: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), **common)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), **common)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, **common)
    figure.savefig(
        OUTPUT_STEM.with_suffix(".jpeg"),
        dpi=300,
        pil_kwargs={"quality": 95},
        **common,
    )
    figure.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )


def main() -> None:
    data, tests = load_data()
    summary = weighted_means(data)
    configure_style()
    mpl.rcParams["svg.hashsalt"] = "cee-figure3-fasttext-vader"

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.5))
    legend_handles = None
    legend_labels = None
    for axis, (panel, title, family, measures) in zip(axes.flat, PANELS):
        order = [measure for measure, _ in measures]
        subset = data.loc[(data["family"] == family) & data["measure"].isin(order)]
        sns.violinplot(
            data=subset,
            x="measure",
            y="score_percent",
            hue="group",
            order=order,
            hue_order=GROUP_ORDER,
            palette=GROUP_COLORS,
            inner="quartile",
            cut=0,
            density_norm="width",
            common_norm=False,
            linewidth=0.6,
            saturation=0.84,
            ax=axis,
        )
        add_weighted_means(axis, summary, family, measures)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.9)
        axis.set_title(f"{panel}  {title}", loc="left", pad=6)
        axis.set_xlabel("")
        if family == "NRC":
            axis.set_ylabel("NRC category frequency\n(% of words per comment)")
        elif family == "LIWC-22":
            axis.set_ylabel("LIWC-22 category frequency\n(% of words per comment)")
        else:
            axis.set_ylabel("VADER component score (%)")
        axis.tick_params(axis="x", length=0)
        handles, labels = axis.get_legend_handles_labels()
        if legend_handles is None:
            legend_handles, legend_labels = handles, labels
        if axis.get_legend() is not None:
            axis.get_legend().remove()

    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        columnspacing=1.8,
        handletextpad=0.45,
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.965, bottom=0.09, wspace=0.24, hspace=0.31)

    save_bundle(figure)
    plt.close(figure)

    SOURCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FRIEDMAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(SOURCE_OUTPUT, index=False)
    tests.to_csv(FRIEDMAN_OUTPUT, index=False)
    summary.to_csv(MEANS_OUTPUT, index=False)
    audit = {
        "status": "complete",
        "daily_input": str(DAILY_FILE),
        "volume_input": str(VOLUME_FILE),
        "analysis_window": [str(data["date"].min().date()), str(data["date"].max().date())],
        "paired_dates": int(data["date"].nunique()),
        "indicators": {"NRC": 10, "LIWC-22": 5, "VADER": 2},
        "groups": list(GROUP_ORDER),
        "test": "paired Friedman test across the three discourse groups by calendar day",
        "inference": "exact two-sided p values are reported",
        "all_p_below_0_001": bool((tests["p_value"] < 0.001).all()),
    }
    AUDIT_OUTPUT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(OUTPUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
