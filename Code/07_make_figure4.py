"""Estimate Figure 4 ECMs for fastText-English NRC/LIWC/VADER series.

Run this file once to reproduce the complete Figure 4 workflow. It reads the
daily analysis table, estimates 34 ECMs (17 indicators x 2 equation
orientations), exports the model results, and then draws all 68 short- and
long-run coefficients. Exact two-sided HAC(7) p values are exported for every reported estimate.

Coefficient inference uses Newey-West HAC(7). The separate Pesaran-Shin-Smith
(PSS) bounds test uses the classical nonrobust covariance required by the PSS
critical-value framework, Case V, one x variable, and the common estimation
sample. The PSS statistic is a joint test of the lagged level terms.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib.colors import TwoSlopeNorm
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.ardl import ARDL, UECM as ErrorCorrectionModel


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_DIR / "Data" / "daily_emotion_scores.csv"
)
OUTPUT_DIR = PROJECT_DIR / "Figures"
OUTPUT = OUTPUT_DIR / "Figure_4"
MODEL_OUTPUT = PROJECT_DIR / "Data" / "Figure_4_ECM_model_results.csv"
SOURCE_OUTPUT = PROJECT_DIR / "Data" / "Figure_4_source_data.csv"

HAC_LAGS = 7
HOLD_BACK = 6
MAX_LAG = 6
PSS_CASE = 5
PSS_X_COUNT = 1
PSS_NSIM = 100_000
PSS_SEED = 20260816
PSS_SIGNIFICANCE_LEVEL = 95.0
PSS_CRITICAL_VALUE_CACHE: dict[int, dict[str, float]] = {}
PSS_VERIFIED_FINITE_SAMPLE_CRITICAL_VALUES = {
    705: {
        "lower_90": 5.5914384136604856,
        "upper_90": 6.278261785461958,
        "lower_95": 6.587714741611035,
        "upper_95": 7.339057663328569,
        "lower_99": 8.733965950084494,
        "upper_99": 9.633686109641985,
    }
}

NRC_ROWS = [
    ("anger", "Anger"),
    ("sadness", "Sadness"),
    ("fear", "Fear"),
    ("disgust", "Disgust"),
    ("joy", "Joy"),
    ("anticipation", "Anticipation"),
    ("trust", "Trust"),
    ("surprise", "Surprise"),
    ("negative", "Negative emotion"),
    ("positive", "Positive emotion"),
]
LIWC_ROWS = [
    ("emo_anger", "Anger"),
    ("emo_sad", "Sadness"),
    ("emo_anx", "Anxiety"),
    ("emo_neg", "Negative emotion"),
    ("emo_pos", "Positive emotion"),
]
VADER_ROWS = [
    ("negative", "Negative sentiment"),
    ("positive", "Positive sentiment"),
]
ROW_KEYS = [
    *(("NRC Emotion Lexicon", metric, label) for metric, label in NRC_ROWS),
    *(("LIWC-22", metric, label) for metric, label in LIWC_ROWS),
    *(("VADER", metric, label) for metric, label in VADER_ROWS),
]
COLUMN_KEYS = [
    ("COVID-to-climate", "short_run", "COVID-19 $\\rightarrow$ Climate\n(Short-run)"),
    ("COVID-to-climate", "long_run", "COVID-19 $\\rightarrow$ Climate\n(Long-run)"),
    ("climate-to-COVID", "short_run", "Climate $\\rightarrow$ COVID-19\n(Short-run)"),
    ("climate-to-COVID", "long_run", "Climate $\\rightarrow$ COVID-19\n(Long-run)"),
]

CONTINUOUS_CONTROLS = [
    "US_daily_covid_death",
    "climatenews",
    "GovernmentResponseIndex_Average",
]
BINARY_CONTROLS = [
    "debates",
    "WinterStorm",
    "Wildfire",
    "TropicalCyclone",
    "SevereStorm",
    "Flood",
    "Drought",
]

INK = "#222222"
GROUP = "#59636E"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    standard_deviation = numeric.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError(f"Cannot standardize {values.name}")
    return (numeric - numeric.mean()) / standard_deviation


def series_column(measurement: str, group: str, metric: str) -> str:
    if measurement == "NRC Emotion Lexicon":
        return f"{group}_{metric}_score_freq"
    if measurement == "LIWC-22":
        return f"{group}_liwc_{metric}_pct"
    if measurement == "VADER":
        return f"{group}_vader_{metric}_pct"
    raise ValueError(f"Unknown measurement family: {measurement}")


def load_daily_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    emotion_columns = {
        series_column(measurement, group, metric)
        for measurement, metric, _ in ROW_KEYS
        for group in ("climate", "covid")
    }
    required = {"date", *emotion_columns, *CONTINUOUS_CONTROLS, *BINARY_CONTROLS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(frame) != 711 or frame["date"].duplicated().any():
        raise ValueError("Expected 711 unique daily observations")
    if frame[list(required - {"date"})].isna().any().any():
        raise ValueError("The Figure 4 input contains missing model values")

    frame["day_index"] = (frame["date"] - frame["date"].min()).dt.days.astype(float)
    for harmonic in (1, 2):
        angle = 2 * math.pi * harmonic * frame["day_index"] / 365.2425
        frame[f"annual_sin_{harmonic}"] = np.sin(angle)
        frame[f"annual_cos_{harmonic}"] = np.cos(angle)
    weekdays = pd.get_dummies(
        frame["date"].dt.dayofweek,
        prefix="weekday",
        drop_first=True,
        dtype=float,
    )
    return pd.concat([frame, weekdays], axis=1)


def fixed_controls(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = pd.DataFrame(index=frame.index)
    for column in CONTINUOUS_CONTROLS:
        fixed[f"z_{column}"] = zscore(frame[column].rename(column))
    fixed = pd.concat(
        [
            fixed,
            frame[BINARY_CONTROLS].astype(float),
            frame[[column for column in frame if column.startswith("weekday_")]].astype(float),
            frame[["annual_sin_1", "annual_cos_1", "annual_sin_2", "annual_cos_2"]].astype(float),
        ],
        axis=1,
    )
    return fixed.astype(float)


def wald_f_for_level_terms(fit, y_level: str, x_level: str) -> float:
    """Return the classical joint F statistic for the two lagged levels."""
    names = [y_level, x_level]
    coefficients = fit.params.loc[names].to_numpy(dtype=float)
    covariance = fit.cov_params().loc[names, names].to_numpy(dtype=float)
    quadratic_form = coefficients @ np.linalg.solve(covariance, coefficients)
    return float(quadratic_form / len(names))


def classify_pss_bounds(statistic: float, lower: float, upper: float) -> str:
    if statistic > upper:
        return "above_upper_bound"
    if statistic < lower:
        return "below_lower_bound"
    return "inconclusive_between_bounds"


def finite_sample_pss_critical_values(fit) -> dict[str, float]:
    """Simulate and cache finite-sample Case-V critical values for one x."""
    nobs = int(fit.nobs)
    if nobs not in PSS_CRITICAL_VALUE_CACHE:
        if nobs in PSS_VERIFIED_FINITE_SAMPLE_CRITICAL_VALUES:
            PSS_CRITICAL_VALUE_CACHE[nobs] = dict(
                PSS_VERIFIED_FINITE_SAMPLE_CRITICAL_VALUES[nobs]
            )
        else:
            simulated = fit.bounds_test(
                case=PSS_CASE,
                cov_type="nonrobust",
                asymptotic=False,
                nsim=PSS_NSIM,
                seed=PSS_SEED,
            )
            PSS_CRITICAL_VALUE_CACHE[nobs] = {
                "lower_90": float(simulated.crit_vals.loc[90.0, "lower"]),
                "upper_90": float(simulated.crit_vals.loc[90.0, "upper"]),
                "lower_95": float(simulated.crit_vals.loc[95.0, "lower"]),
                "upper_95": float(simulated.crit_vals.loc[95.0, "upper"]),
                "lower_99": float(simulated.crit_vals.loc[99.0, "lower"]),
                "upper_99": float(simulated.crit_vals.loc[99.0, "upper"]),
            }
    return PSS_CRITICAL_VALUE_CACHE[nobs]


def formal_pss_bounds_test(
    y: pd.Series,
    x: pd.DataFrame,
    p: int,
    q: int,
) -> dict[str, object]:
    """Run the formal core PSS Case-V test on the common estimation sample.

    The PSS refit contains y, the single x series, an unrestricted intercept,
    and an unrestricted linear trend. The core bounds-test specification omits
    fixed covariates. The coefficient model uses the full controls and HAC(7)
    inference.
    """
    core_model = ErrorCorrectionModel(
        y,
        p,
        x,
        order=q,
        trend="ct",
        hold_back=HOLD_BACK,
    )
    core_fit = core_model.fit(cov_type="nonrobust")
    y_level = f"{y.name}.L1"
    x_level = f"{x.columns[0]}.L1"
    statistic = wald_f_for_level_terms(core_fit, y_level, x_level)
    critical = finite_sample_pss_critical_values(core_fit)
    lower = critical["lower_95"]
    upper = critical["upper_95"]
    return {
        "bounds_f": statistic,
        "pss_case": PSS_CASE,
        "pss_k": PSS_X_COUNT,
        "pss_n": int(core_fit.nobs),
        "pss_covariance": "classical nonrobust",
        "pss_core_fixed_controls": 0,
        "pss_critical_value_method": (
            f"finite-sample Monte Carlo; nsim={PSS_NSIM}; seed={PSS_SEED}"
        ),
        "pss_90_lower": critical["lower_90"],
        "pss_90_upper": critical["upper_90"],
        "pss_95_lower": lower,
        "pss_95_upper": upper,
        "pss_99_lower": critical["lower_99"],
        "pss_99_upper": critical["upper_99"],
        "pss_95_decision": classify_pss_bounds(statistic, lower, upper),
    }


def select_model(y: pd.Series, x: pd.DataFrame, fixed: pd.DataFrame) -> tuple[ARDL, object, int]:
    best: tuple[float, ARDL, object] | None = None
    fitted_candidates = 0
    for p in range(1, MAX_LAG + 1):
        for q in range(1, MAX_LAG + 1):
            try:
                model = ARDL(
                    y,
                    lags=p,
                    exog=x,
                    order=q,
                    trend="ct",
                    fixed=fixed,
                    hold_back=HOLD_BACK,
                    missing="drop",
                )
                fitted = model.fit()
            except (ValueError, np.linalg.LinAlgError):
                continue
            fitted_candidates += 1
            candidate = (float(fitted.bic), model, fitted)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise RuntimeError(f"No estimable ECM candidate for {y.name}")
    return best[1], best[2], fitted_candidates


def estimate_equation(
    frame: pd.DataFrame,
    fixed: pd.DataFrame,
    measurement: str,
    metric: str,
    label: str,
    direction: str,
) -> dict[str, object]:
    if direction == "COVID-to-climate":
        y_column = series_column(measurement, "climate", metric)
        x_column = series_column(measurement, "covid", metric)
    else:
        y_column = series_column(measurement, "covid", metric)
        x_column = series_column(measurement, "climate", metric)

    y = zscore(frame[y_column].rename(y_column))
    x = pd.DataFrame({x_column: zscore(frame[x_column].rename(x_column))})
    model, ardl_fit, candidate_count = select_model(y, x, fixed)
    fit = ErrorCorrectionModel.from_ardl(model).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    selected_p, selected_q = model.ardl_order[:2]
    pss = formal_pss_bounds_test(y, x, int(selected_p), int(selected_q))

    y_level = f"{y_column}.L1"
    x_level = f"{x_column}.L1"
    short_name = f"D.{x_column}.L0"
    adjustment = float(fit.params[y_level])
    other_level = float(fit.params[x_level])
    covariance = fit.cov_params()

    long_run = -other_level / adjustment
    gradient = np.array([other_level / adjustment**2, -1 / adjustment])
    level_covariance = covariance.loc[[y_level, x_level], [y_level, x_level]].to_numpy()
    long_se = float(np.sqrt(max(gradient @ level_covariance @ gradient, 0)))
    long_z = long_run / long_se if long_se > 0 else np.nan
    long_p = float(2 * st.norm.sf(abs(long_z))) if np.isfinite(long_z) else np.nan

    residuals = pd.Series(ardl_fit.resid).dropna()
    ljung = acorr_ljungbox(residuals, lags=[7, 14, 30], return_df=True)
    arch = het_arch(residuals, nlags=7)
    jarque = jarque_bera(residuals)
    cusum = breaks_cusumolsresid(residuals, ddof=len(ardl_fit.params))
    try:
        minimum_root_modulus = float(np.min(np.abs(ardl_fit.roots)))
    except Exception:
        minimum_root_modulus = np.nan

    short = float(fit.params[short_name])
    short_se = float(fit.bse[short_name])
    return {
        "measurement": measurement,
        "metric": metric,
        "label": label,
        "direction": direction,
        "specification": "ECM",
        "dependent_variable": y_column,
        "predictor_variable": x_column,
        "n": int(fit.nobs),
        "selected_p": int(selected_p),
        "selected_q": int(selected_q),
        "candidate_models_fitted": candidate_count,
        "bic": float(ardl_fit.bic),
        "aic": float(ardl_fit.aic),
        **pss,
        "ecm_coefficient": adjustment,
        "ecm_hac_se": float(fit.bse[y_level]),
        "ecm_pvalue": float(fit.pvalues[y_level]),
        "short_run_coefficient": short,
        "short_run_hac_se": short_se,
        "short_run_ci95_low": short - 1.96 * short_se,
        "short_run_ci95_high": short + 1.96 * short_se,
        "short_run_pvalue": float(fit.pvalues[short_name]),
        "long_run_coefficient": long_run,
        "long_run_hac_se": long_se,
        "long_run_ci95_low": long_run - 1.96 * long_se,
        "long_run_ci95_high": long_run + 1.96 * long_se,
        "long_run_pvalue": long_p,
        "ljung_box_p_lag7": float(ljung.loc[7, "lb_pvalue"]),
        "ljung_box_p_lag14": float(ljung.loc[14, "lb_pvalue"]),
        "ljung_box_p_lag30": float(ljung.loc[30, "lb_pvalue"]),
        "arch_lm_pvalue_lag7": float(arch[1]),
        "jarque_bera_pvalue": float(jarque[1]),
        "cusum_pvalue": float(cusum[1]),
        "minimum_root_modulus": minimum_root_modulus,
        "inference": "exact two-sided Newey-West HAC(7) p values reported",
    }


def estimate_all_models(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = fixed_controls(frame)
    records = []
    total = len(ROW_KEYS) * 2
    for index, (measurement, metric, label) in enumerate(ROW_KEYS, start=1):
        for direction_index, direction in enumerate(("COVID-to-climate", "climate-to-COVID")):
            number = (index - 1) * 2 + direction_index + 1
            print(f"[{number:02d}/{total}] Estimating {measurement}: {label}, {direction}")
            records.append(
                estimate_equation(frame, fixed, measurement, metric, label, direction)
            )
    models = pd.DataFrame(records)
    expected_models = len(ROW_KEYS) * 2
    if len(models) != expected_models or sorted(models["n"].unique()) != [705]:
        raise RuntimeError(
            f"The ECM workflow did not produce {expected_models} "
            "common-sample equations"
        )
    return models


def build_source_data(models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, model in models.iterrows():
        for horizon in ("short_run", "long_run"):
            rows.append(
                {
                    "measurement": model["measurement"],
                    "metric": model["metric"],
                    "label": model["label"],
                    "measurement_order": {
                        "NRC Emotion Lexicon": 0,
                        "LIWC-22": 1,
                        "VADER": 2,
                    }[model["measurement"]],
                    "direction": model["direction"],
                    "specification": model["specification"],
                    "n": model["n"],
                    "selected_p": model["selected_p"],
                    "selected_q": model["selected_q"],
                    "horizon_key": horizon,
                    "estimate": model[f"{horizon}_coefficient"],
                    "se": model[f"{horizon}_hac_se"],
                    "ci_low": model[f"{horizon}_ci95_low"],
                    "ci_high": model[f"{horizon}_ci95_high"],
                    "pvalue": model[f"{horizon}_pvalue"],
                }
            )
    source = pd.DataFrame(rows)
    expected_coefficients = len(ROW_KEYS) * len(COLUMN_KEYS)
    if len(source) != expected_coefficients or source.duplicated(
        ["measurement", "metric", "direction", "horizon_key"]
    ).any():
        raise RuntimeError(
            f"Expected {expected_coefficients} unique Figure 4 coefficients"
        )
    return source


def stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def build_matrices(source: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    estimates = np.full((len(ROW_KEYS), len(COLUMN_KEYS)), np.nan)
    pvalues = np.full_like(estimates, np.nan)
    for row_index, (measurement, metric, _) in enumerate(ROW_KEYS):
        for column_index, (direction, horizon, _) in enumerate(COLUMN_KEYS):
            match = source[
                source["measurement"].eq(measurement)
                & source["metric"].eq(metric)
                & source["direction"].eq(direction)
                & source["horizon_key"].eq(horizon)
            ]
            if len(match) != 1:
                raise RuntimeError("A Figure 4 coefficient is missing or duplicated")
            estimates[row_index, column_index] = float(match.iloc[0]["estimate"])
            pvalues[row_index, column_index] = float(match.iloc[0]["pvalue"])
    return estimates, pvalues


def annotation_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.48 else INK


def save_figure(fig: plt.Figure) -> None:
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
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


def draw(source: pd.DataFrame) -> None:
    estimates, pvalues = build_matrices(source)
    limit = max(0.1, np.ceil(np.abs(estimates).max() * 10) / 10)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = mpl.colormaps["RdBu_r"]

    fig, ax = plt.subplots(figsize=(12, 11.4))
    image = ax.imshow(estimates, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(COLUMN_KEYS)), [item[2] for item in COLUMN_KEYS])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.set_yticks(np.arange(len(ROW_KEYS)), [item[2] for item in ROW_KEYS])
    ax.tick_params(axis="y", length=0, pad=7)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xticks(np.arange(-0.5, len(COLUMN_KEYS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROW_KEYS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(1.5, color="white", linewidth=5.0)
    ax.axhline(len(NRC_ROWS) - 0.5, color=GROUP, linewidth=2.0)
    ax.axhline(len(NRC_ROWS) + len(LIWC_ROWS) - 0.5, color=GROUP, linewidth=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(estimates.shape[0]):
        for column in range(estimates.shape[1]):
            value = estimates[row, column]
            marker = stars(pvalues[row, column])
            label = f"{value:.3f}\n{marker}" if marker else f"{value:.3f}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9.5,
                color=annotation_color(cmap(norm(value))),
                fontweight="bold" if marker else "normal",
                linespacing=0.9,
            )

    transform = ax.get_yaxis_transform()
    group_label_x = -0.03
    group_label_size = float(mpl.rcParams["ytick.labelsize"]) + 2
    group_label_style = {
        "transform": transform,
        "ha": "right",
        "va": "center",
        "fontsize": group_label_size,
        "fontweight": "bold",
        "color": INK,
        "clip_on": False,
        "bbox": {"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    }
    ax.text(
        group_label_x,
        -0.5,
        "NRC Emotion Lexicon",
        **group_label_style,
    )
    ax.text(
        group_label_x,
        len(NRC_ROWS) + len(LIWC_ROWS) - 0.5,
        "VADER",
        **group_label_style,
    )
    ax.text(
        group_label_x,
        len(NRC_ROWS) - 0.5,
        "LIWC",
        **group_label_style,
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.055, shrink=0.78)
    colorbar.set_label("Standardised ECM coefficient", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=9.5)
    colorbar.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.20, right=0.90, top=0.88, bottom=0.04)
    save_figure(fig)
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_daily_data()
    models = estimate_all_models(frame)
    source = build_source_data(models)
    models.to_csv(MODEL_OUTPUT, index=False)
    source.to_csv(SOURCE_OUTPUT, index=False)
    draw(source)
    print(f"Model results: {MODEL_OUTPUT}")
    print(f"Figure source data: {SOURCE_OUTPUT}")
    print(f"Figure exports: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
