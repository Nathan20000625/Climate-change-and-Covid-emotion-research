"""Generate ECM diagnostic and sensitivity tables.

The script reads the 17-indicator daily dataset and the 34 baseline ECM
equations used in Figure 4. It writes stationarity, model, residual,
complete-effect, and calendar-break sensitivity tables.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews


ROOT = Path(__file__).resolve().parents[1]
BASE_CODE = Path(__file__).with_name("07_make_figure4.py")
BASELINE_FILE = ROOT / "Data" / "Figure_4_ECM_model_results.csv"
TABLE_DIR = ROOT / "Data" / "Supplementary_Table_Sources"
QA_DIR = ROOT / "QA"
BREAK_DATES = (
    "2020-06-30",
    "2020-09-30",
    "2020-12-31",
    "2021-03-31",
    "2021-06-30",
    "2021-09-30",
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("figure4_model", BASE_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_kpss(values: pd.Series, regression: str) -> tuple[float, float, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, pvalue, lags, _ = kpss(values, regression=regression, nlags="auto")
    return float(statistic), float(pvalue), int(lags)


def stationarity_table(module, frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    series_meta = (
        baseline[["measurement", "metric", "label", "dependent_variable", "predictor_variable"]]
        .melt(
            id_vars=["measurement", "metric", "label"],
            value_vars=["dependent_variable", "predictor_variable"],
            value_name="series",
        )
        .drop(columns="variable")
        .drop_duplicates("series")
        .sort_values(["measurement", "metric", "series"])
    )
    rows: list[dict[str, object]] = []
    for row in series_meta.itertuples(index=False):
        values = pd.to_numeric(frame[row.series], errors="coerce").dropna().astype(float)
        differences = values.diff().dropna()
        adf_c = adfuller(values, regression="c", autolag="BIC")
        adf_ct = adfuller(values, regression="ct", autolag="BIC")
        adf_diff = adfuller(differences, regression="c", autolag="BIC")
        kpss_c = safe_kpss(values, "c")
        kpss_ct = safe_kpss(values, "ct")
        kpss_diff = safe_kpss(differences, "c")
        za = zivot_andrews(values, regression="ct", maxlag=14, autolag="BIC")
        break_index = int(za[4])
        rows.append(
            {
                "measurement": row.measurement,
                "metric": row.metric,
                "label": row.label,
                "series": row.series,
                "topic": "climate" if row.series.startswith("climate_") else "COVID-19",
                "n": len(values),
                "adf_level_c_stat": float(adf_c[0]),
                "adf_level_c_p": float(adf_c[1]),
                "adf_level_ct_stat": float(adf_ct[0]),
                "adf_level_ct_p": float(adf_ct[1]),
                "adf_diff_c_stat": float(adf_diff[0]),
                "adf_diff_c_p": float(adf_diff[1]),
                "kpss_level_c_stat": kpss_c[0],
                "kpss_level_c_p": kpss_c[1],
                "kpss_level_ct_stat": kpss_ct[0],
                "kpss_level_ct_p": kpss_ct[1],
                "kpss_diff_c_stat": kpss_diff[0],
                "kpss_diff_c_p": kpss_diff[1],
                "za_ct_stat": float(za[0]),
                "za_ct_p": float(za[1]),
                "za_break_date": frame.loc[break_index, "date"].strftime("%Y-%m-%d"),
                "adf_diff_reject_0_05": bool(adf_diff[1] < 0.05),
                "kpss_diff_nonreject_0_05": bool(kpss_diff[1] >= 0.05),
                "no_i2_evidence_by_adf_diff": bool(adf_diff[1] < 0.05),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 34:
        raise RuntimeError(f"Expected 34 unique series, found {len(output)}")
    return output


def complete_effects(baseline: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "measurement",
        "metric",
        "label",
        "direction",
        "dependent_variable",
        "predictor_variable",
        "n",
        "selected_p",
        "selected_q",
    ]
    rows: list[dict[str, object]] = []
    for row in baseline.itertuples(index=False):
        base = {column: getattr(row, column) for column in id_columns}
        rows.append(
            {
                **base,
                "horizon": "short run",
                "coefficient": row.short_run_coefficient,
                "hac_standard_error": row.short_run_hac_se,
                "ci95_low": row.short_run_ci95_low,
                "ci95_high": row.short_run_ci95_high,
                "p_value": row.short_run_pvalue,
            }
        )
        rows.append(
            {
                **base,
                "horizon": "long run",
                "coefficient": row.long_run_coefficient,
                "hac_standard_error": row.long_run_hac_se,
                "ci95_low": row.long_run_ci95_low,
                "ci95_high": row.long_run_ci95_high,
                "p_value": row.long_run_pvalue,
            }
        )
    return pd.DataFrame(rows)


def fit_calendar_break(module, frame: pd.DataFrame, row: pd.Series, break_date: str) -> dict[str, object]:
    y_column = str(row["dependent_variable"])
    x_column = str(row["predictor_variable"])
    y = module.zscore(frame[y_column].rename(y_column))
    x = pd.DataFrame({x_column: module.zscore(frame[x_column].rename(x_column))})
    fixed = module.fixed_controls(frame).copy()
    breakpoint = pd.Timestamp(break_date)
    fixed["calendar_break_level"] = (frame["date"] > breakpoint).astype(float)
    fixed["calendar_break_slope_years"] = np.maximum(
        (frame["date"] - breakpoint).dt.days, 0
    ) / 365.2425
    model = ARDL(
        y,
        lags=int(row["selected_p"]),
        exog=x,
        order=int(row["selected_q"]),
        trend="ct",
        fixed=fixed,
        hold_back=module.HOLD_BACK,
        missing="drop",
    )
    fit = module.ErrorCorrectionModel.from_ardl(model).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": module.HAC_LAGS, "use_correction": True},
    )
    y_level = f"{y_column}.L1"
    x_level = f"{x_column}.L1"
    short_name = f"D.{x_column}.L0"
    a = float(fit.params[y_level])
    b = float(fit.params[x_level])
    covariance = fit.cov_params()
    long_run = -b / a
    gradient = np.array([b / a**2, -1 / a])
    level_covariance = covariance.loc[[y_level, x_level], [y_level, x_level]].to_numpy()
    long_se = float(np.sqrt(max(gradient @ level_covariance @ gradient, 0)))
    long_p = float(2 * st.norm.sf(abs(long_run / long_se))) if long_se > 0 else np.nan
    short_run = float(fit.params[short_name])
    return {
        "break_date": break_date,
        "measurement": row["measurement"],
        "metric": row["metric"],
        "label": row["label"],
        "direction": row["direction"],
        "n": int(fit.nobs),
        "selected_p_fixed": int(row["selected_p"]),
        "selected_q_fixed": int(row["selected_q"]),
        "baseline_short_run": float(row["short_run_coefficient"]),
        "baseline_short_p": float(row["short_run_pvalue"]),
        "short_run": short_run,
        "short_run_hac_se": float(fit.bse[short_name]),
        "short_run_p_value": float(fit.pvalues[short_name]),
        "baseline_long_run": float(row["long_run_coefficient"]),
        "baseline_long_p": float(row["long_run_pvalue"]),
        "long_run": long_run,
        "long_run_hac_se": long_se,
        "long_run_p_value": long_p,
        "short_run_same_sign": bool(np.sign(short_run) == np.sign(row["short_run_coefficient"])),
        "long_run_same_sign": bool(np.sign(long_run) == np.sign(row["long_run_coefficient"])),
        "break_level": float(fit.params["calendar_break_level"]),
        "break_level_p_value": float(fit.pvalues["calendar_break_level"]),
        "break_slope": float(fit.params["calendar_break_slope_years"]),
        "break_slope_p_value": float(fit.pvalues["calendar_break_slope_years"]),
    }


def calendar_break_tables(module, frame: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    total = len(BREAK_DATES) * len(baseline)
    counter = 0
    for break_date in BREAK_DATES:
        for _, equation in baseline.iterrows():
            counter += 1
            if counter % 20 == 0 or counter == total:
                print(f"calendar-break models: {counter}/{total}")
            rows.append(fit_calendar_break(module, frame, equation, break_date))
    full = pd.DataFrame(rows)
    summary = (
        full.groupby("break_date", as_index=False)
        .agg(
            equations=("metric", "size"),
            short_run_same_sign=("short_run_same_sign", "sum"),
            long_run_same_sign=("long_run_same_sign", "sum"),
            short_run_p_lt_0_05=("short_run_p_value", lambda x: int((x < 0.05).sum())),
            long_run_p_lt_0_05=("long_run_p_value", lambda x: int((x < 0.05).sum())),
            break_level_p_lt_0_05=("break_level_p_value", lambda x: int((x < 0.05).sum())),
            break_slope_p_lt_0_05=("break_slope_p_value", lambda x: int((x < 0.05).sum())),
        )
    )
    return full, summary


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    module = load_base_module()
    frame = module.load_daily_data()
    baseline = pd.read_csv(BASELINE_FILE)
    if len(frame) != 711 or len(baseline) != 34:
        raise RuntimeError("Expected 711 daily observations and 34 baseline equations")

    stationarity = stationarity_table(module, frame, baseline)
    effects = complete_effects(baseline)
    bounds = baseline[
        [
            "measurement",
            "metric",
            "label",
            "direction",
            "n",
            "selected_p",
            "selected_q",
            "bounds_f",
            "pss_case",
            "pss_k",
            "pss_n",
            "pss_covariance",
            "pss_core_fixed_controls",
            "pss_critical_value_method",
            "pss_90_lower",
            "pss_90_upper",
            "pss_95_lower",
            "pss_95_upper",
            "pss_99_lower",
            "pss_99_upper",
            "pss_95_decision",
            "ecm_coefficient",
            "ecm_hac_se",
            "ecm_pvalue",
            "bic",
            "aic",
        ]
    ].copy()
    diagnostics = baseline[
        [
            "measurement", "metric", "label", "direction", "n", "ljung_box_p_lag7",
            "ljung_box_p_lag14", "ljung_box_p_lag30", "arch_lm_pvalue_lag7",
            "jarque_bera_pvalue", "cusum_pvalue", "minimum_root_modulus",
        ]
    ].copy()
    break_full, break_summary = calendar_break_tables(module, frame, baseline)

    paths = {
        "stationarity": TABLE_DIR / "Supplementary_Table_7a_Stationarity_Tests_source.csv",
        "bounds": TABLE_DIR / "Supplementary_Table_7b_PSS_Bounds_and_ECM_source.csv",
        "effects": TABLE_DIR / "Supplementary_Table_6_Complete_ARDL_ECM_Effects_source.csv",
        "diagnostics": TABLE_DIR / "Supplementary_Table_8a_Residual_and_Stability_Diagnostics_source.csv",
        "break_full": TABLE_DIR / "Supplementary_Table_9_Calendar_Break_Sensitivity_full_source.csv",
        "break_summary": TABLE_DIR / "Supplementary_Table_9_Calendar_Break_Sensitivity_summary_source.csv",
    }
    for key, table in {
        "stationarity": stationarity,
        "bounds": bounds,
        "effects": effects,
        "diagnostics": diagnostics,
        "break_full": break_full,
        "break_summary": break_summary,
    }.items():
        table.to_csv(paths[key], index=False, encoding="utf-8-sig")

    workbook = TABLE_DIR / "Supplementary_Tables_6_7a_7b_8a_9_source.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        stationarity.to_excel(writer, sheet_name="S7a_stationarity", index=False)
        bounds.to_excel(writer, sheet_name="S7b_bounds_ECM", index=False)
        effects.to_excel(writer, sheet_name="S6_effects", index=False)
        diagnostics.to_excel(writer, sheet_name="S8a_diagnostics", index=False)
        break_full.to_excel(writer, sheet_name="S9_break_full", index=False)
        break_summary.to_excel(writer, sheet_name="S9_break_summary", index=False)

    audit = {
        "status": "complete",
        "daily_rows": len(frame),
        "stationarity_series": len(stationarity),
        "adf_first_difference_rejections_0_05": int(stationarity["adf_diff_reject_0_05"].sum()),
        "kpss_first_difference_nonrejections_0_05": int(stationarity["kpss_diff_nonreject_0_05"].sum()),
        "baseline_equations": len(baseline),
        "complete_effects": len(effects),
        "pss_bounds_test": {
            "case": 5,
            "k": 1,
            "covariance": "classical nonrobust",
            "core_fixed_controls": 0,
            "n": sorted(int(value) for value in bounds["pss_n"].unique()),
            "f_range": [
                float(bounds["bounds_f"].min()),
                float(bounds["bounds_f"].max()),
            ],
            "critical_95": {
                "lower": float(bounds["pss_95_lower"].iloc[0]),
                "upper": float(bounds["pss_95_upper"].iloc[0]),
            },
            "decisions": bounds["pss_95_decision"].value_counts().to_dict(),
            "note": (
                "PSS bounds statistics use a core y-x ECM with unrestricted "
                "constant and trend. HAC(7) is retained separately for the "
                "substantive ECM coefficient inference."
            ),
        },
        "negative_significant_ecm_terms": int(((bounds["ecm_coefficient"] < 0) & (bounds["ecm_pvalue"] < 0.05)).sum()),
        "ljung_box_pass_counts": {
            "lag7": int((diagnostics["ljung_box_p_lag7"] >= 0.05).sum()),
            "lag14": int((diagnostics["ljung_box_p_lag14"] >= 0.05).sum()),
            "lag30": int((diagnostics["ljung_box_p_lag30"] >= 0.05).sum()),
        },
        "arch_lm_pass_lag7": int((diagnostics["arch_lm_pvalue_lag7"] >= 0.05).sum()),
        "cusum_pass": int((diagnostics["cusum_pvalue"] >= 0.05).sum()),
        "calendar_break_models": len(break_full),
        "calendar_break_summary": break_summary.to_dict(orient="records"),
        "inference": "exact two-sided Newey-West HAC(7) p values reported",
        "workbook": str(workbook),
    }
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "Figure4_17_Indicator_Diagnostics_Audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
