"""Export source data for Supplementary Tables 1-3.

The script imports the final Figure 1 module so that data validation,
variable definitions, transformations, and covariates remain identical to
the plotted models. It derives the descriptive statistics in Supplementary
Table 1 and exports the complete OLS coefficients for Supplementary Tables 2
and 3.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURE_MODULE_PATH = Path(__file__).with_name("04_make_figure1.py")
EXPORT_DIR = PROJECT_DIR / "Data" / "Supplementary_Table_Sources"
TABLE_1_CSV = EXPORT_DIR / "Supplementary_Table_1_Descriptive_Statistics_source.csv"
TABLE_2_CSV = EXPORT_DIR / "Supplementary_Table_2_Full_OLS_Volume_Results_source.csv"
TABLE_3_CSV = EXPORT_DIR / "Supplementary_Table_3_Full_OLS_Emotion_Results_source.csv"


def load_figure_module():
    """Load the numbered Figure 1 script as a Python module."""
    spec = importlib.util.spec_from_file_location("figure1_model", FIGURE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {FIGURE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def coefficient_rows(
    fitted,
    *,
    panel: str,
    model_id: str,
    measurement: str,
    outcome_label: str,
    outcome_column: str,
    focal_predictor: str,
    outcome_scale: str,
    predictor_scale: str,
) -> list[dict[str, object]]:
    """Convert a fitted statsmodels OLS object to tidy coefficient rows."""
    intervals = fitted.conf_int()
    rows: list[dict[str, object]] = []
    for term in fitted.params.index:
        pvalue = float(fitted.pvalues[term])
        rows.append(
            {
                "panel": panel,
                "model_id": model_id,
                "measurement_system": measurement,
                "outcome_label": outcome_label,
                "outcome_column": outcome_column,
                "focal_predictor": focal_predictor,
                "term": str(term),
                "is_focal_term": str(term) == focal_predictor,
                "estimate": float(fitted.params[term]),
                "standard_error": float(fitted.bse[term]),
                "t_statistic": float(fitted.tvalues[term]),
                "pvalue": pvalue,
                "ci95_low": float(intervals.loc[term, 0]),
                "ci95_high": float(intervals.loc[term, 1]),
                "significance": stars(pvalue),
                "n": int(fitted.nobs),
                "df_resid": float(fitted.df_resid),
                "r_squared": float(fitted.rsquared),
                "adjusted_r_squared": float(fitted.rsquared_adj),
                "outcome_scale": outcome_scale,
                "predictor_scale": predictor_scale,
                "inference": "Conventional OLS; exact two-sided p values",
            }
        )
    return rows


def fit_attention_models(module, frame: pd.DataFrame):
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    definitions = [
        ("B1", "Confirmed cases", "US_daily_covid_confirm"),
        ("B2", "COVID-19 deaths", "US_daily_covid_death"),
    ]
    for model_id, label, exposure in definitions:
        weekdays = module.weekday_dummies(frame["date"])
        design = pd.DataFrame(
            {exposure: np.log1p(frame[exposure].astype(float))},
            index=frame.index,
        )
        design = pd.concat(
            [design, frame[module.MODEL_CONTROLS].astype(float), weekdays], axis=1
        )
        design = sm.add_constant(design, has_constant="add")
        outcome = np.log1p(frame["climate_only_comments"].astype(float))
        fitted = sm.OLS(outcome, design).fit()
        rows.extend(
            coefficient_rows(
                fitted,
                panel="Figure 1b",
                model_id=model_id,
                measurement="Daily attention",
                outcome_label="Climate-only comment volume",
                outcome_column="climate_only_comments",
                focal_predictor=exposure,
                outcome_scale="log(1 + daily climate-only comments)",
                predictor_scale=f"log(1 + {label.lower()})",
            )
        )
        metadata.append(
            {
                "panel": "Figure 1b",
                "model_id": model_id,
                "measurement_system": "Daily attention",
                "outcome": "log(1 + daily climate-only comments)",
                "focal_predictor": f"log(1 + {label.lower()})",
                "continuous_controls": ", ".join(module.MODEL_CONTROLS),
                "indicator_controls": "Tuesday-Sunday indicators; Monday reference",
                "n": int(fitted.nobs),
                "inference": "Conventional OLS; 95% CI; exact two-sided p values",
            }
        )
    return rows, metadata


def emotion_definitions(module):
    for name in module.EMOTION_ORDER:
        yield (
            f"C_NRC_{name}",
            "NRC Emotion Lexicon",
            module.NRC_LABELS.get(name, name.title()),
            f"climate_{name}_score_freq",
        )
    for metric, label in module.LIWC_ORDER:
        yield (
            f"C_LIWC_{metric}",
            "LIWC-22",
            label,
            f"climate_liwc_{metric}_pct",
        )
    for metric, label in module.VADER_ORDER:
        yield (
            f"C_VADER_{metric}",
            "VADER",
            label,
            f"climate_vader_{metric}_pct",
        )


def build_descriptive_statistics(
    volume: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    """Derive the corpus summary and daily statistics in Supplementary Table 1."""
    panel_a = [
        {
            "panel": "A. Corpus construction",
            "row_order": 1,
            "variable": "Final analytic corpus",
            "definition": (
                "Comments dated 21 January 2020 to 31 December 2021"
            ),
            "count": int(volume["total_comments"].sum()),
        },
        {
            "panel": "A. Corpus construction",
            "row_order": 2,
            "variable": "Climate-only comments",
            "definition": "Comments assigned only to the climate discourse group",
            "count": int(volume["climate_only_comments"].sum()),
        },
        {
            "panel": "A. Corpus construction",
            "row_order": 3,
            "variable": "COVID-only comments",
            "definition": "Comments assigned only to the COVID-19 discourse group",
            "count": int(volume["covid_only_comments"].sum()),
        },
        {
            "panel": "A. Corpus construction",
            "row_order": 4,
            "variable": "Co-mention comments",
            "definition": "Comments mentioning both climate change and COVID-19",
            "count": int(volume["co_mention_comments"].sum()),
        },
        {
            "panel": "A. Corpus construction",
            "row_order": 5,
            "variable": "Calendar days",
            "definition": "Days included in the daily analysis",
            "count": int(len(volume)),
        },
        {
            "panel": "A. Corpus construction",
            "row_order": 6,
            "variable": "Start date",
            "definition": "First included date",
            "date": volume["date"].min().strftime("%Y-%m-%d"),
        },
        {
            "panel": "A. Corpus construction",
            "row_order": 7,
            "variable": "End date",
            "definition": "Last included date",
            "date": volume["date"].max().strftime("%Y-%m-%d"),
        },
    ]

    daily_variables = [
        ("Climate-only comments", "climate_only_comments"),
        ("COVID-only comments", "covid_only_comments"),
        ("Co-mention comments", "co_mention_comments"),
        ("Total retained comments", "total_comments"),
        (
            "Co-mention share of climate-related comments (%)",
            "co_mention_share_pct",
        ),
        ("US daily COVID-19 deaths", "US_daily_covid_death"),
    ]
    nrc_metrics = [
        ("Joy", "joy"),
        ("Sadness", "sadness"),
        ("Anger", "anger"),
        ("Fear", "fear"),
        ("Surprise", "surprise"),
        ("Disgust", "disgust"),
        ("Trust", "trust"),
        ("Anticipation", "anticipation"),
        ("Positive", "positive"),
        ("Negative", "negative"),
    ]
    groups = [
        ("climate-only", "climate"),
        ("COVID-only", "covid"),
        ("co-mention", "both"),
    ]
    for label, metric in nrc_metrics:
        for group_label, prefix in groups:
            daily_variables.append(
                (f"NRC {label}, {group_label}", f"{prefix}_{metric}_score_freq")
            )

    liwc_metrics = [
        ("positive emotion", "emo_pos"),
        ("negative emotion", "emo_neg"),
        ("anxiety", "emo_anx"),
        ("anger", "emo_anger"),
        ("sadness", "emo_sad"),
    ]
    for label, metric in liwc_metrics:
        for group_label, prefix in groups:
            daily_variables.append(
                (f"LIWC-22 {label}, {group_label}", f"{prefix}_liwc_{metric}_pct")
            )

    for label, metric in [("positive", "positive"), ("negative", "negative")]:
        for group_label, prefix in groups:
            daily_variables.append(
                (f"VADER {label}, {group_label}", f"{prefix}_vader_{metric}_pct")
            )

    daily_variables.extend(
        [
            ("US daily confirmed COVID-19 cases", "US_daily_covid_confirm"),
            ("Presidential-debate indicator", "debates"),
            ("Climate-news volume", "climatenews"),
            ("Winter-storm indicator", "WinterStorm"),
            ("Wildfire indicator", "Wildfire"),
            ("Tropical-cyclone indicator", "TropicalCyclone"),
            ("Severe-storm indicator", "SevereStorm"),
            ("Flood indicator", "Flood"),
            ("Drought indicator", "Drought"),
            ("Government Response Index", "GovernmentResponseIndex_Average"),
        ]
    )

    volume_columns = set(volume.columns)
    rows = panel_a.copy()
    for row_order, (label, column) in enumerate(daily_variables, start=1):
        source = volume if column in volume_columns else frame
        values = pd.to_numeric(source[column], errors="raise")
        rows.append(
            {
                "panel": "B. Daily descriptive statistics",
                "row_order": row_order,
                "variable": label,
                "source_column": column,
                "n": int(values.count()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )

    table = pd.DataFrame(rows)
    expected_counts = {
        "Final analytic corpus": 25_911_526,
        "Climate-only comments": 1_351_032,
        "COVID-only comments": 24_438_852,
        "Co-mention comments": 121_642,
        "Calendar days": 711,
    }
    observed = table.set_index("variable")["count"].dropna().astype(int).to_dict()
    if any(observed.get(key) != value for key, value in expected_counts.items()):
        raise ValueError(f"Unexpected corpus counts: {observed}")
    if not (table.loc[table["panel"].str.startswith("B"), "n"] == 711).all():
        raise ValueError("At least one descriptive series does not contain 711 days")
    return table


def fit_emotion_models(module, frame: pd.DataFrame):
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    focal = "US_daily_covid_death"
    for model_id, measurement, label, outcome_column in emotion_definitions(module):
        outcome = module.zscore(frame[outcome_column])
        controls = frame[
            [*module.CONTINUOUS_EMOTION_CONTROLS, *module.OTHER_EMOTION_CONTROLS]
        ].astype(float).copy()
        for column in module.CONTINUOUS_EMOTION_CONTROLS:
            controls[column] = module.zscore(controls[column])
        controls = pd.concat(
            [controls, module.weekday_dummies(frame["date"])], axis=1
        )
        design = sm.add_constant(controls, has_constant="add")
        fitted = sm.OLS(outcome, design).fit()
        rows.extend(
            coefficient_rows(
                fitted,
                panel="Figure 1c",
                model_id=model_id,
                measurement=measurement,
                outcome_label=label,
                outcome_column=outcome_column,
                focal_predictor=focal,
                outcome_scale="Standardized daily climate-only language score",
                predictor_scale="Standardized daily COVID-19 deaths",
            )
        )
        metadata.append(
            {
                "panel": "Figure 1c",
                "model_id": model_id,
                "measurement_system": measurement,
                "outcome": f"Standardized {label} score in climate-only comments",
                "focal_predictor": "Standardized daily COVID-19 deaths",
                "continuous_controls": (
                    "Standardized climate-news volume and Government Response Index"
                ),
                "indicator_controls": (
                    "Presidential debates; six weather/disaster indicators; "
                    "Tuesday-Sunday indicators (Monday reference)"
                ),
                "n": int(fitted.nobs),
                "inference": "Conventional OLS; 95% CI; exact two-sided p values",
            }
        )
    return rows, metadata


def main() -> None:
    module = load_figure_module()
    volume, frame = module.load_daily_data()
    descriptive = build_descriptive_statistics(volume, frame)
    attention_rows, _ = fit_attention_models(module, frame)
    emotion_rows, _ = fit_emotion_models(module, frame)

    full = pd.DataFrame(attention_rows + emotion_rows)
    focal = full.loc[full["is_focal_term"]].copy()

    if len(focal) != 19:
        raise ValueError(f"Expected 19 focal estimates, found {len(focal)}")
    if not (focal["n"] == 711).all():
        raise ValueError("At least one Figure 1 model does not use 711 daily observations")

    s2_full = full.loc[full["panel"].eq("Figure 1b")].copy()
    s3_full = full.loc[full["panel"].eq("Figure 1c")].copy()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    descriptive.to_csv(TABLE_1_CSV, index=False, float_format="%.10g")
    s2_full.to_csv(TABLE_2_CSV, index=False, float_format="%.10g")
    s3_full.to_csv(TABLE_3_CSV, index=False, float_format="%.10g")
    print(
        f"Exported {len(descriptive)} descriptive rows, "
        f"{len(s2_full)} Table 2 coefficients, and "
        f"{len(s3_full)} Table 3 coefficients"
    )
    print(f"Supplementary table sources written to: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
