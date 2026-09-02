"""Export complete Figure 1 OLS results for the supplementary table.

The script imports the final Figure 1 module so that data validation,
variable definitions, transformations, and covariates remain identical to
the plotted models. It exports focal estimates, all fitted coefficients, and
one auditable model-metadata table.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURE_MODULE_PATH = Path(__file__).with_name("04_make_figure1.py")
EXPORT_DIR = PROJECT_DIR / "QA" / "Figure1_table_inputs"
EXPORT_JSON = EXPORT_DIR / "Figure1_OLS_tables.json"


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
    _, frame = module.load_daily_data()
    attention_rows, attention_metadata = fit_attention_models(module, frame)
    emotion_rows, emotion_metadata = fit_emotion_models(module, frame)

    full = pd.DataFrame(attention_rows + emotion_rows)
    focal = full.loc[full["is_focal_term"]].copy()
    metadata = pd.DataFrame(attention_metadata + emotion_metadata)

    if len(focal) != 19:
        raise ValueError(f"Expected 19 focal estimates, found {len(focal)}")
    if not (focal["n"] == 711).all():
        raise ValueError("At least one Figure 1 model does not use 711 daily observations")

    focal_table = focal[
        [
            "panel", "model_id", "measurement_system", "outcome_label",
            "focal_predictor", "n", "estimate", "standard_error",
            "ci95_low", "ci95_high", "pvalue", "significance",
            "r_squared", "adjusted_r_squared", "outcome_scale",
            "predictor_scale", "inference",
        ]
    ].copy()
    focal_table["focal_predictor"] = focal_table["focal_predictor"].replace(
        {
            "US_daily_covid_confirm": "Daily confirmed COVID-19 cases",
            "US_daily_covid_death": "Daily COVID-19 deaths",
        }
    )
    focal_table.insert(
        5,
        "estimate_type",
        np.where(
            focal_table["panel"].eq("Figure 1b"),
            "OLS coefficient (log-log)",
            "Standardized OLS coefficient",
        ),
    )

    s2_focal = focal_table.loc[focal_table["panel"].eq("Figure 1b")].copy()
    s3_focal = focal_table.loc[focal_table["panel"].eq("Figure 1c")].copy()
    s2_full = full.loc[full["panel"].eq("Figure 1b")].copy()
    s3_full = full.loc[full["panel"].eq("Figure 1c")].copy()
    readme = pd.DataFrame(
        {
            "Item": [
                "Purpose",
                "Table S2",
                "Table S3",
                "Inference",
                "Sample",
            ],
            "Description": [
                "Complete ordinary least squares results underlying Figure 1b-c.",
                "Associations of confirmed cases or deaths with climate-only comment volume.",
                "Associations of daily deaths with NRC, LIWC-22 and VADER indicators.",
                "Exact two-sided p values from conventional OLS standard errors are reported.",
                "All models use 711 daily observations from 21 January 2020 to 31 December 2021.",
            ],
        }
    )

    def payload(table: pd.DataFrame) -> dict[str, object]:
        return {
            "columns": table.columns.tolist(),
            "rows": json.loads(table.to_json(orient="values", date_format="iso")),
        }

    export = {
        "Read me": payload(readme),
        "Table S2 focal": payload(s2_focal),
        "Table S2 full": payload(s2_full),
        "Table S3 focal": payload(s3_focal),
        "Table S3 full": payload(s3_full),
        "Model metadata": payload(metadata),
    }
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_JSON.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported {len(focal)} focal estimates and {len(full)} coefficients")
    print(f"Structured table input written to: {EXPORT_JSON}")


if __name__ == "__main__":
    main()
