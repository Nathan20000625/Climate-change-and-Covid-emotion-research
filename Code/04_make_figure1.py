"""Reproduce Figure 1 from the daily analysis files.

Panel a shows seven-day moving averages of climate-only comment volume and
U.S. COVID-19 deaths with descriptive event-window annotations.
Panel b estimates the log-log attention models with conventional OLS
inference. Panel c estimates standardized NRC, LIWC-22, and VADER
language-indicator models with the same controls and conventional OLS inference. VADER
is displayed as a separate sentiment-measurement block.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data"
OUTPUT_DIR = PROJECT_DIR / "Figures"
SOURCE_DIR = PROJECT_DIR / "Data"
OUTPUT = OUTPUT_DIR / "Figure_1"

VOLUME_FILE = DATA_DIR / "daily_comment_volume_and_covid_deaths.csv"
EMOTION_FILE = DATA_DIR / "daily_emotion_scores.csv"

GREEN = "#008B72"
ORANGE = "#D55E00"
INK = "#2B2B2B"
GRID = "#D9D9D9"
EVENT_FILL = "#8C8C8C"
BASELINE = "#888888"
EVENT_WINDOW_DAYS = 5

WEATHER_CONTROLS = [
    "WinterStorm", "Wildfire", "TropicalCyclone",
    "SevereStorm", "Flood", "Drought",
]
MODEL_CONTROLS = [
    "debates", "climatenews", *WEATHER_CONTROLS,
    "GovernmentResponseIndex_Average",
]
CONTINUOUS_EMOTION_CONTROLS = [
    "US_daily_covid_death", "climatenews",
    "GovernmentResponseIndex_Average",
]
OTHER_EMOTION_CONTROLS = ["debates", *WEATHER_CONTROLS]
EMOTION_ORDER = [
    "sadness", "anger", "fear", "disgust",
    "surprise", "joy", "trust", "anticipation",
    "negative", "positive",
]
NRC_LABELS = {
    "negative": "Negative emotion",
    "positive": "Positive emotion",
}
LIWC_ORDER = [
    ("emo_anger", "Anger"),
    ("emo_sad", "Sadness"),
    ("emo_anx", "Anxiety"),
    ("emo_neg", "Negative emotion"),
    ("emo_pos", "Positive emotion"),
]
VADER_ORDER = [
    ("negative", "Negative sentiment"),
    ("positive", "Positive sentiment"),
]

# Selected event windows displayed in panel a.
EVENTS = [
    ("2020-09-10", "Western US\nwildfires", 0.73),
    ("2021-02-19", "US rejoins\nParis Agreement", 0.57),
    ("2021-06-28", "Pacific NW\nheatwave", 0.76),
    ("2021-08-09", "IPCC AR6 WG I", 0.94),
    ("2021-11-01", "COP26 World Leaders\nSummit", 0.68),
]


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a numeric series using its sample standard deviation."""
    values = pd.to_numeric(series, errors="raise").astype(float)
    scale = values.std(ddof=1)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"Cannot standardize {series.name}: invalid SD")
    return (values - values.mean()) / scale


def significance_stars(pvalue: float) -> str:
    """Format conventional significance symbols from a supplied p or q value."""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def weekday_dummies(dates: pd.Series) -> pd.DataFrame:
    """Create Tuesday-Sunday indicators, with Monday as the reference."""
    return pd.get_dummies(
        pd.to_datetime(dates).dt.dayofweek,
        prefix="weekday",
        drop_first=True,
        dtype=float,
    )


def load_daily_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate the two public daily input files."""
    volume = pd.read_csv(VOLUME_FILE)
    volume["date"] = pd.to_datetime(volume["date"], errors="raise")
    for column in [
        "climate_only_comments", "covid_only_comments",
        "co_mention_comments", "US_daily_covid_death",
    ]:
        volume[column] = pd.to_numeric(volume[column], errors="raise")

    daily = pd.read_csv(EMOTION_FILE)
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")

    required = {
        "date", "US_daily_covid_confirm", "US_daily_covid_death",
        *MODEL_CONTROLS,
        *(f"climate_{emotion}_score_freq" for emotion in EMOTION_ORDER),
        *(f"climate_liwc_{metric}_pct" for metric, _ in LIWC_ORDER),
        *(f"climate_vader_{metric}_pct" for metric, _ in VADER_ORDER),
    }
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"Missing required columns in {EMOTION_FILE.name}: {missing}")
    if len(volume) != 711 or len(daily) != 711:
        raise ValueError(
            f"Expected 711 rows in each input, found {len(volume)} and {len(daily)}"
        )
    if volume["date"].duplicated().any() or daily["date"].duplicated().any():
        raise ValueError("Duplicate dates found in an input file")

    frame = daily.merge(
        volume[["date", "climate_only_comments"]], on="date", how="inner",
        validate="one_to_one",
    )
    if len(frame) != 711 or frame[list(required)].isna().any().any():
        raise ValueError("Inputs do not form a complete 711-day analysis frame")

    death_difference = (
        daily.set_index("date")["US_daily_covid_death"].astype(float)
        - volume.set_index("date")["US_daily_covid_death"].astype(float)
    ).abs().max()
    if death_difference != 0:
        raise ValueError("COVID-19 death values differ between the two input files")

    volume = volume.sort_values("date").reset_index(drop=True)
    frame = frame.sort_values("date").reset_index(drop=True)
    return volume, frame


def fit_attention_model(
    frame: pd.DataFrame, exposure: str, label: str,
) -> dict[str, float | int | str]:
    """Fit one log-log climate-volume model with conventional OLS inference."""
    weekdays = weekday_dummies(frame["date"])
    design = pd.DataFrame(
        {exposure: np.log1p(frame[exposure].astype(float))},
        index=frame.index,
    )
    design = pd.concat(
        [design, frame[MODEL_CONTROLS].astype(float), weekdays], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    outcome = np.log1p(frame["climate_only_comments"].astype(float))
    fitted = sm.OLS(outcome, design).fit()
    ci = fitted.conf_int().loc[exposure]
    return {
        "label": label,
        "independent_variable": exposure,
        "n": int(fitted.nobs),
        "coefficient": float(fitted.params[exposure]),
        "standard_error": float(fitted.bse[exposure]),
        "pvalue": float(fitted.pvalues[exposure]),
        "CI95_low": float(ci[0]),
        "CI95_high": float(ci[1]),
        "r_squared": float(fitted.rsquared),
    }


def fit_emotion_model(
    frame: pd.DataFrame,
    outcome_column: str,
    label: str,
    measurement: str,
    metric: str,
) -> dict[str, float | int | str]:
    """Fit one standardized mortality model with conventional OLS inference."""
    outcome = zscore(frame[outcome_column])

    controls = frame[
        [*CONTINUOUS_EMOTION_CONTROLS, *OTHER_EMOTION_CONTROLS]
    ].astype(float).copy()
    for column in CONTINUOUS_EMOTION_CONTROLS:
        controls[column] = zscore(controls[column])
    controls = pd.concat(
        [controls, weekday_dummies(frame["date"])], axis=1
    )
    design = sm.add_constant(controls, has_constant="add")
    fitted = sm.OLS(outcome, design).fit()
    focal = "US_daily_covid_death"
    ci = fitted.conf_int().loc[focal]
    return {
        "label": label,
        "measurement": measurement,
        "metric": metric,
        "outcome": outcome_column,
        "n": int(fitted.nobs),
        "standardized_beta": float(fitted.params[focal]),
        "standard_error": float(fitted.bse[focal]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "pvalue": float(fitted.pvalues[focal]),
    }


def prepare_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create panel-a series and re-estimate panels b and c."""
    volume, frame = load_daily_data()
    volume["climate_ma7"] = (
        volume["climate_only_comments"].rolling(7, min_periods=1).mean()
    )
    volume["deaths_ma7"] = (
        volume["US_daily_covid_death"].rolling(7, min_periods=1).mean()
    )

    attention = pd.DataFrame(
        [
            fit_attention_model(frame, "US_daily_covid_confirm", "Confirmed cases"),
            fit_attention_model(frame, "US_daily_covid_death", "COVID-19 deaths"),
        ]
    )
    nrc = pd.DataFrame(
        [
            fit_emotion_model(
                frame,
                f"climate_{name}_score_freq",
                NRC_LABELS.get(name, name.title()),
                "NRC Emotion Lexicon",
                name,
            )
            for name in EMOTION_ORDER
        ]
    )
    liwc = pd.DataFrame(
        [
            fit_emotion_model(
                frame,
                f"climate_liwc_{metric}_pct",
                label,
                "LIWC-22",
                metric,
            )
            for metric, label in LIWC_ORDER
        ]
    )
    vader = pd.DataFrame(
        [
            fit_emotion_model(
                frame,
                f"climate_vader_{metric}_pct",
                label,
                "VADER",
                metric,
            )
            for metric, label in VADER_ORDER
        ]
    )
    emotion = pd.concat([nrc, liwc, vader], ignore_index=True)
    emotion["p_lt_0_05"] = emotion["pvalue"] < 0.05
    return volume, attention, emotion


def save_source_data(
    volume: pd.DataFrame,
    attention: pd.DataFrame,
    emotion: pd.DataFrame,
) -> None:
    """Save all plotted series and newly estimated coefficients."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    volume.to_csv(SOURCE_DIR / "Figure_1a_daily_source.csv", index=False)
    attention.to_csv(SOURCE_DIR / "Figure_1b_OLS_source.csv", index=False)
    emotion.to_csv(SOURCE_DIR / "Figure_1c_OLS_source.csv", index=False)
    events = pd.DataFrame(
        EVENTS, columns=["event_date", "label", "label_y_fraction"]
    )
    events["window_end_date"] = (
        pd.to_datetime(events["event_date"])
        + pd.Timedelta(days=EVENT_WINDOW_DAYS)
    ).dt.strftime("%Y-%m-%d")
    events.to_csv(SOURCE_DIR / "Figure_1a_event_windows.csv", index=False)


def draw_figure(
    volume: pd.DataFrame,
    attention: pd.DataFrame,
    emotion: pd.DataFrame,
) -> None:
    """Draw and export the complete three-panel Figure 1."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    # Use a spacious 1+2 panel composition.
    fig = plt.figure(figsize=(14, 12.5))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[0.85, 1.35], width_ratios=[1.0, 1.25],
        hspace=0.34, wspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, :])
    ax_a2 = ax_a.twinx()
    line = ax_a.plot(
        volume["date"], volume["climate_ma7"], color=GREEN, lw=2.0
    )[0]
    area = ax_a2.fill_between(
        volume["date"], 0, volume["deaths_ma7"],
        color="#E69F5B", alpha=0.35, linewidth=0,
    )
    ax_a.set_ylabel(
        "Climate-only Comment Volume", color=GREEN,
        fontsize=14, fontweight="bold",
    )
    ax_a2.set_ylabel(
        "COVID-19 Daily Deaths", color=ORANGE,
        fontsize=14, fontweight="bold",
    )
    ax_a.tick_params(axis="y", colors=GREEN)
    ax_a2.tick_params(axis="y", colors="#B55D14")
    ax_a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_a.set_xlim(volume["date"].min(), volume["date"].max())

    ymin, ymax = ax_a.get_ylim()
    for date_text, label, fraction in EVENTS:
        start = pd.Timestamp(date_text)
        end = start + pd.Timedelta(days=EVENT_WINDOW_DAYS)
        midpoint = start + (end - start) / 2
        ax_a.axvspan(
            start, end, color=EVENT_FILL, alpha=0.16,
            linewidth=0, zorder=0.4,
        )
        ax_a.text(
            midpoint, ymin + fraction * (ymax - ymin), label,
            ha="center", va="bottom", fontsize=9.5,
            color="#333333", linespacing=0.95,
            bbox={
                "boxstyle": "square,pad=0.10", "facecolor": "white",
                "edgecolor": "none", "alpha": 0.80,
            },
            zorder=5,
        )
    ax_a.legend(
        [line, area],
        ["Climate-only Comments (7-day MA)", "COVID-19 Deaths (7-day MA)"],
        frameon=False, loc="upper left", fontsize=13,
    )
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=18, x=-0.05)

    ax_b = fig.add_subplot(grid[1, 0])
    # Tightly spaced positions keep the sparse panel visually balanced.
    y_b = np.array([0.7, 0.3])
    ax_b.set_ylim(0.2, 0.8)
    for row, y in zip(attention.itertuples(index=False), y_b):
        ax_b.errorbar(
            float(row.coefficient), y,
            xerr=[
                [float(row.coefficient) - float(row.CI95_low)],
                [float(row.CI95_high) - float(row.coefficient)],
            ],
            fmt="o", mfc=INK, mec=INK, color=INK,
            capsize=5, elinewidth=1.5, markersize=8,
        )
        stars = significance_stars(float(row.pvalue))
        if stars:
            ax_b.text(
                float(row.CI95_high) + 0.002, y, stars,
                va="center", ha="left", fontsize=13.5,
                fontweight="bold", color=INK,
            )
    ax_b.axvline(0, color=BASELINE, lw=1.0, ls="--")
    ax_b.set_yticks(y_b, attention["label"])
    ax_b.tick_params(axis="y", labelsize=12.5)
    ax_b.set_xlabel("OLS Coefficient (log-log)", fontsize=13)
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=18, x=-0.1)

    ax_c = fig.add_subplot(grid[1, 1])
    # Keep the three non-equivalent measurement systems visually separate.
    section_names = ["NRC Emotion Lexicon", "LIWC-22", "VADER"]
    section_labels = {
        "NRC Emotion Lexicon": "NRC",
        "LIWC-22": "LIWC-22",
        "VADER": "VADER",
    }
    y_positions: list[float] = []
    section_first_y: dict[str, float] = {}
    section_last_y: dict[str, float] = {}
    cursor = float(len(emotion) + len(section_names) - 1)
    for section in section_names:
        section_rows = int((emotion["measurement"] == section).sum())
        section_first_y[section] = cursor
        for _ in range(section_rows):
            y_positions.append(cursor)
            cursor -= 1.0
        section_last_y[section] = cursor + 1.0
        cursor -= 1.0
    y_c = np.asarray(y_positions, dtype=float)
    for row, y in zip(emotion.itertuples(index=False), y_c):
        value = float(row.standardized_beta)
        color = ORANGE if value < 0 else GREEN
        marker = {
            "NRC Emotion Lexicon": "o",
            "LIWC-22": "s",
            "VADER": "^",
        }[row.measurement]
        ax_c.errorbar(
            value, y,
            xerr=[
                [value - float(row.ci95_low)],
                [float(row.ci95_high) - value],
            ],
            fmt=marker, mfc=color, mec=color, color=color,
            capsize=5, elinewidth=1.5, markersize=8,
        )
        stars = significance_stars(float(row.pvalue))
        if stars:
            ax_c.text(
                float(row.ci95_high) + 0.01, y, stars,
                va="center", ha="left", color=color,
                fontsize=13.5, fontweight="bold",
            )
    ax_c.axvline(0, color=BASELINE, lw=1.0, ls="--")
    ax_c.set_yticks(y_c, emotion["label"])
    ax_c.tick_params(axis="y", labelsize=11)
    for upper, lower in zip(section_names[:-1], section_names[1:]):
        separator_y = (section_last_y[upper] + section_first_y[lower]) / 2
        ax_c.axhline(separator_y, color="#B0B0B0", lw=0.9)
    for section in section_names:
        ax_c.text(
            -0.02, section_first_y[section] + 1.0, section_labels[section],
            transform=ax_c.get_yaxis_transform(),
            ha="right", va="center", fontsize=13,
            fontweight="bold", color="#555555",
        )
    ax_c.set_ylim(y_c[-1] - 0.5, y_c[0] + 1.15)
    ax_c.set_xlabel("Standardized OLS Coefficient", fontsize=13)
    ax_c.set_title("c", loc="left", fontweight="bold", fontsize=18, x=-0.1)

    # Explicit margins are stable with the twin y-axis in panel a.
    fig.subplots_adjust(left=0.10, right=0.91, top=0.95, bottom=0.09)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        OUTPUT.with_suffix(".jpeg"), format="jpeg", dpi=300,
        bbox_inches="tight", pil_kwargs={"quality": 95},
    )
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    volume, attention, emotion = prepare_panels()
    save_source_data(volume, attention, emotion)
    draw_figure(volume, attention, emotion)
    print(attention.to_string(index=False))
    print(emotion[[
        "measurement", "label", "standardized_beta", "ci95_low", "ci95_high",
        "pvalue", "p_lt_0_05",
    ]].to_string(index=False))
    print(f"Figure written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
