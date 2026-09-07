"""Independently verify the formal PSS Case-V bounds results for Figure 4.

The substantive ECM retains all controls and Newey-West HAC(7) coefficient
inference. This audit reconstructs the separate core y-x ECM used for the PSS
bounds test, preserves the common six-day hold-back, uses classical nonrobust
covariance, and verifies the finite-sample k=1 critical values and decisions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import UECM


ROOT = Path(__file__).resolve().parents[1]
BASE_CODE = Path(__file__).with_name("07_make_figure4.py")
BASELINE_FILE = (
    ROOT / "Data" / "Figure_4_ECM_model_results.csv"
)
OUTPUT_CSV = ROOT / "QA" / "Bounds_Test_Covariance_Verification_34.csv"
OUTPUT_JSON = ROOT / "QA" / "Bounds_Test_Critical_Values_Verification.json"
CASE = 5
K = 1
NSIM = 100_000
SEED = 20260816


def load_base_module():
    spec = importlib.util.spec_from_file_location("figure4_model", BASE_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wald_f(fit, y_level: str, x_level: str) -> float:
    names = [y_level, x_level]
    coefficients = fit.params.loc[names].to_numpy(dtype=float)
    covariance = fit.cov_params().loc[names, names].to_numpy(dtype=float)
    return float(
        coefficients @ np.linalg.solve(covariance, coefficients) / len(names)
    )


def classify(statistic: float, lower: float, upper: float) -> str:
    if statistic > upper:
        return "above_upper_bound"
    if statistic < lower:
        return "below_lower_bound"
    return "inconclusive_between_bounds"


def main() -> None:
    module = load_base_module()
    frame = module.load_daily_data()
    baseline = pd.read_csv(BASELINE_FILE)
    if len(baseline) != 34:
        raise RuntimeError(f"Expected 34 baseline equations, found {len(baseline)}")

    representative = baseline.loc[baseline["selected_p"].eq(module.HOLD_BACK)].iloc[0]
    representative_y_column = str(representative["dependent_variable"])
    representative_x_column = str(representative["predictor_variable"])
    representative_y = module.zscore(
        frame[representative_y_column].rename(representative_y_column)
    )
    representative_x = pd.DataFrame(
        {
            representative_x_column: module.zscore(
                frame[representative_x_column].rename(representative_x_column)
            )
        }
    )
    representative_core = UECM(
        representative_y,
        int(representative["selected_p"]),
        representative_x,
        order=int(representative["selected_q"]),
        trend="ct",
        hold_back=module.HOLD_BACK,
    ).fit(cov_type="nonrobust")
    simulated_critical_values = representative_core.bounds_test(
        case=CASE,
        cov_type="nonrobust",
        asymptotic=False,
        nsim=NSIM,
        seed=SEED,
    ).crit_vals

    rows: list[dict[str, object]] = []
    for number, row in enumerate(baseline.itertuples(index=False), start=1):
        print(f"[{number:02d}/34] {row.measurement}: {row.label}, {row.direction}")
        y_column = str(row.dependent_variable)
        x_column = str(row.predictor_variable)
        y = module.zscore(frame[y_column].rename(y_column))
        x = pd.DataFrame({x_column: module.zscore(frame[x_column].rename(x_column))})

        core = UECM(
            y,
            int(row.selected_p),
            x,
            order=int(row.selected_q),
            trend="ct",
            hold_back=module.HOLD_BACK,
        ).fit(cov_type="nonrobust")
        statistic = wald_f(core, f"{y_column}.L1", f"{x_column}.L1")
        lower = float(simulated_critical_values.loc[95.0, "lower"])
        upper = float(simulated_critical_values.loc[95.0, "upper"])
        decision = classify(statistic, lower, upper)
        rows.append(
            {
                "measurement": row.measurement,
                "metric": row.metric,
                "label": row.label,
                "direction": row.direction,
                "selected_p": int(row.selected_p),
                "selected_q": int(row.selected_q),
                "pss_case": CASE,
                "pss_k": K,
                "pss_n": int(core.nobs),
                "pss_covariance": "classical nonrobust",
                "reported_formal_pss_f": float(row.bounds_f),
                "independently_recomputed_pss_f": statistic,
                "absolute_reproduction_error": abs(float(row.bounds_f) - statistic),
                "reported_95_lower": float(row.pss_95_lower),
                "reported_95_upper": float(row.pss_95_upper),
                "simulated_95_lower": lower,
                "simulated_95_upper": upper,
                "reported_95_decision": row.pss_95_decision,
                "independently_recomputed_95_decision": decision,
                "decision_matches": bool(row.pss_95_decision == decision),
                "fixed_controls_in_pss_core_refit": 0,
                "substantive_ecm_inference": (
                    "full controls; two-sided Newey-West HAC(7)"
                ),
            }
        )

    output = pd.DataFrame(rows)
    max_error = float(output["absolute_reproduction_error"].max())
    max_critical_value_error = float(
        max(
            (output["reported_95_lower"] - output["simulated_95_lower"]).abs().max(),
            (output["reported_95_upper"] - output["simulated_95_upper"]).abs().max(),
        )
    )
    if max_error > 1e-10:
        raise RuntimeError(f"Formal PSS statistics did not reproduce: {max_error}")
    if max_critical_value_error > 1e-10:
        raise RuntimeError(
            "Finite-sample PSS critical values did not reproduce: "
            f"{max_critical_value_error}"
        )
    if not output["decision_matches"].all():
        raise RuntimeError("At least one formal PSS decision failed to reproduce")
    decisions = output["reported_95_decision"].value_counts().to_dict()
    if decisions != {"above_upper_bound": 31, "below_lower_bound": 3}:
        raise RuntimeError(f"Unexpected formal PSS decision counts: {decisions}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    critical = {
        "statsmodels_version": __import__("statsmodels").__version__,
        "test": "Pesaran-Shin-Smith bounds test",
        "case": CASE,
        "case_definition": (
            "unrestricted constant and unrestricted trend; neither is included "
            "in the joint restriction"
        ),
        "k": K,
        "k_definition": "number of x variables in the level relationship",
        "joint_restriction": "lagged y level = lagged x level = 0",
        "covariance": "classical nonrobust",
        "core_fixed_controls": 0,
        "common_nobs": sorted(int(value) for value in output["pss_n"].unique()),
        "simulation_nsim": NSIM,
        "simulation_seed": SEED,
        "finite_sample_critical_values": simulated_critical_values.to_dict(),
        "f_range": [
            float(output["reported_formal_pss_f"].min()),
            float(output["reported_formal_pss_f"].max()),
        ],
        "decisions_at_5_percent": decisions,
        "below_lower_bound_equations": output.loc[
            output["reported_95_decision"].eq("below_lower_bound"),
            ["measurement", "label", "direction"],
        ].to_dict(orient="records"),
        "max_statistic_reproduction_error": max_error,
        "max_critical_value_reproduction_error": max_critical_value_error,
        "coefficient_inference_note": (
            "Newey-West HAC(7) is used for ECM coefficient inference, whereas "
            "classical covariance is used to construct the PSS F statistic."
        ),
        "comparison_csv": str(OUTPUT_CSV),
    }
    OUTPUT_JSON.write_text(
        json.dumps(critical, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(critical, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
