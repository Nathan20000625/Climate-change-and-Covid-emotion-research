"""Compare the Figure 4 ECM lag-search ceilings of 6 and 14 days.

This sensitivity workflow writes to a separate output directory and estimates
three specifications:

1. the existing max-lag-6 results (read from disk; hold-back 6),
2. max lag 6 with hold-back 14, and
3. max lag 14 with hold-back 14.

Comparing specifications 2 and 3 isolates the effect of expanding the BIC lag
search because both use the same 697-day estimation sample. Specification 1 is
also compared with specification 3 to show the combined impact of the wider lag
search and the common-sample change.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PRIMARY_CODE = Path(__file__).with_name("07_make_figure4.py")
PRIMARY_RESULTS = PROJECT_DIR / "Data" / "Figure_4_ECM_model_results.csv"
OUTPUT_DIR = PROJECT_DIR / "QA" / "Figure4_lag14_sensitivity"
TABLE_DIR = PROJECT_DIR / "Data" / "Supplementary_Table_Sources"


def load_primary_module():
    spec = importlib.util.spec_from_file_location("figure4_primary", PRIMARY_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import primary Figure 4 code: {PRIMARY_CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate_specification(module, frame: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    module.MAX_LAG = max_lag
    module.HOLD_BACK = 14
    controls = module.fixed_controls(frame)
    records: list[dict[str, object]] = []
    total = len(module.ROW_KEYS) * 2

    for index, (measurement, metric, label) in enumerate(module.ROW_KEYS, start=1):
        for direction_index, direction in enumerate(
            ("COVID-to-climate", "climate-to-COVID")
        ):
            number = (index - 1) * 2 + direction_index + 1
            print(
                f"[max={max_lag}: {number:02d}/{total}] "
                f"{measurement}: {label}, {direction}",
                flush=True,
            )
            record = module.estimate_equation(
                frame, controls, measurement, metric, label, direction
            )
            record["lag_search_max"] = max_lag
            record["hold_back"] = 14
            records.append(record)

    models = pd.DataFrame(records)
    if len(models) != total or sorted(models["n"].unique()) != [697]:
        raise RuntimeError(
            f"Expected {total} equations with n=697 for max lag {max_lag}"
        )
    # Formal PSS bounds verification is handled separately from lag selection.
    pss_columns = [column for column in models if column.startswith("pss_")]
    models = models.drop(columns=["bounds_f", *pss_columns], errors="ignore")
    return models


def add_comparison_flags(comparison: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    for horizon in ("short_run", "long_run"):
        left_coef = comparison[f"{horizon}_coefficient_{left}"]
        right_coef = comparison[f"{horizon}_coefficient_{right}"]
        left_p = comparison[f"{horizon}_pvalue_{left}"]
        right_p = comparison[f"{horizon}_pvalue_{right}"]
        comparison[f"{horizon}_sign_same"] = (
            np.sign(left_coef) == np.sign(right_coef)
        )
        comparison[f"{horizon}_significant_{left}"] = left_p < 0.05
        comparison[f"{horizon}_significant_{right}"] = right_p < 0.05
        comparison[f"{horizon}_significance_same"] = (
            comparison[f"{horizon}_significant_{left}"]
            == comparison[f"{horizon}_significant_{right}"]
        )
        comparison[f"{horizon}_coefficient_change"] = right_coef - left_coef
        comparison[f"{horizon}_absolute_change"] = (
            comparison[f"{horizon}_coefficient_change"].abs()
        )
    comparison["selected_p_same"] = (
        comparison[f"selected_p_{left}"] == comparison[f"selected_p_{right}"]
    )
    comparison["selected_q_same"] = (
        comparison[f"selected_q_{left}"] == comparison[f"selected_q_{right}"]
    )
    return comparison


def compare(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    keys = ["measurement", "metric", "label", "direction"]
    fields = [
        *keys,
        "n",
        "selected_p",
        "selected_q",
        "bic",
        "ecm_coefficient",
        "ecm_pvalue",
        "short_run_coefficient",
        "short_run_pvalue",
        "long_run_coefficient",
        "long_run_pvalue",
    ]
    merged = left_frame[fields].merge(
        right_frame[fields],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=(f"_{left}", f"_{right}"),
    )
    if len(merged) != 34:
        raise RuntimeError(f"Expected 34 matched equations, found {len(merged)}")
    return add_comparison_flags(merged, left, right)


def lag_distribution(frame: pd.DataFrame) -> str:
    p_counts = frame["selected_p"].value_counts().sort_index().to_dict()
    q_counts = frame["selected_q"].value_counts().sort_index().to_dict()
    return f"p={p_counts}; q={q_counts}"


def comparison_summary(
    comparison: pd.DataFrame, left: str, right: str, title: str
) -> list[str]:
    lines = [f"## {title}"]
    lines.append(
        f"- Identical selected p: {int(comparison['selected_p_same'].sum())}/34; "
        f"identical selected q: {int(comparison['selected_q_same'].sum())}/34."
    )
    for horizon, label in (("short_run", "Short-run"), ("long_run", "Long-run")):
        lines.append(
            f"- {label} coefficient signs unchanged: "
            f"{int(comparison[f'{horizon}_sign_same'].sum())}/34; "
            f"p<0.05 classification unchanged: "
            f"{int(comparison[f'{horizon}_significance_same'].sum())}/34; "
            f"maximum absolute coefficient change: "
            f"{comparison[f'{horizon}_absolute_change'].max():.4f}."
        )

    changes = comparison[
        ~comparison["short_run_significance_same"]
        | ~comparison["long_run_significance_same"]
        | ~comparison["short_run_sign_same"]
        | ~comparison["long_run_sign_same"]
    ]
    if changes.empty:
        lines.append("- No coefficient changed sign or p < 0.05 classification.")
    else:
        lines.append("- Equations with a sign or p < 0.05 classification change:")
        for _, row in changes.iterrows():
            details = []
            for horizon, label in (("short_run", "short"), ("long_run", "long")):
                if not row[f"{horizon}_sign_same"] or not row[f"{horizon}_significance_same"]:
                    details.append(
                        f"{label}: {row[f'{horizon}_coefficient_{left}']:.3f} "
                        f"(p={row[f'{horizon}_pvalue_{left}']:.3g}) -> "
                        f"{row[f'{horizon}_coefficient_{right}']:.3f} "
                        f"(p={row[f'{horizon}_pvalue_{right}']:.3g})"
                    )
            lines.append(
                f"  - {row['measurement']} / {row['label']} / "
                f"{row['direction']}: {'; '.join(details)}."
            )
    lines.append("")
    return lines


def write_summary(
    current6: pd.DataFrame,
    common6: pd.DataFrame,
    lag14: pd.DataFrame,
    fair: pd.DataFrame,
    practical: pd.DataFrame,
) -> None:
    at_boundary = lag14[
        lag14["selected_p"].eq(14) | lag14["selected_q"].eq(14)
    ]
    lines = [
        "# Figure 4 ECM lag-ceiling sensitivity: 6 versus 14 days",
        "",
        "The same-sample comparison (max lag 6 versus max lag 14, both with "
        "hold-back 14) is the primary diagnostic because it isolates the lag-search "
        "ceiling from the eight-day change in the estimation sample.",
        "",
        "## Lag-order distributions",
        f"- Existing max-6 model (hold-back 6): {lag_distribution(current6)}",
        f"- Max-6 common-sample model (hold-back 14): {lag_distribution(common6)}",
        f"- Max-14 common-sample model (hold-back 14): {lag_distribution(lag14)}",
        f"- Max-14 boundary selections: {len(at_boundary)}/34.",
        "",
    ]
    if not at_boundary.empty:
        lines.append("Models selecting p=14 or q=14:")
        for _, row in at_boundary.iterrows():
            lines.append(
                f"- {row['measurement']} / {row['label']} / {row['direction']}: "
                f"p={int(row['selected_p'])}, q={int(row['selected_q'])}."
            )
        lines.append("")

    lines.extend(
        comparison_summary(
            fair,
            "max6_hb14",
            "max14_hb14",
            "Same-sample comparison: max lag 6 versus 14",
        )
    )
    lines.extend(
        comparison_summary(
            practical,
            "current6",
            "max14_hb14",
            "Practical comparison: current primary model versus max lag 14",
        )
    )
    (OUTPUT_DIR / "Lag6_vs_Lag14_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    module = load_primary_module()
    frame = module.load_daily_data()

    if not PRIMARY_RESULTS.exists():
        raise FileNotFoundError(f"Primary max-lag-6 results not found: {PRIMARY_RESULTS}")
    current6 = pd.read_csv(PRIMARY_RESULTS)

    common6 = estimate_specification(module, frame, max_lag=6)
    common6_path = OUTPUT_DIR / "ECM_results_maxlag6_holdback14.csv"
    common6.to_csv(common6_path, index=False)

    lag14 = estimate_specification(module, frame, max_lag=14)
    lag14_path = OUTPUT_DIR / "ECM_results_maxlag14_holdback14.csv"
    lag14.to_csv(lag14_path, index=False)

    fair = compare(common6, lag14, "max6_hb14", "max14_hb14")
    fair_path = TABLE_DIR / "Supplementary_Table_8b_Lag_Ceiling_Sensitivity_source.csv"
    fair.to_csv(fair_path, index=False)

    practical = compare(current6, lag14, "current6", "max14_hb14")
    practical_path = OUTPUT_DIR / "Lag6_current_vs_Lag14_comparison.csv"
    practical.to_csv(practical_path, index=False)

    module.OUTPUT_DIR = OUTPUT_DIR
    module.OUTPUT = OUTPUT_DIR / "Figure_4_maxlag14_sensitivity_heatmap"
    module.MODEL_OUTPUT = lag14_path
    module.SOURCE_OUTPUT = OUTPUT_DIR / "Figure_4_maxlag14_source_data.csv"
    source14 = module.build_source_data(lag14)
    source14.to_csv(module.SOURCE_OUTPUT, index=False)
    module.configure_style()
    module.draw(source14)

    write_summary(current6, common6, lag14, fair, practical)
    print(f"Max-lag-6 common-sample results: {common6_path}")
    print(f"Max-lag-14 results: {lag14_path}")
    print(f"Same-sample comparison: {fair_path}")
    print(f"Summary: {OUTPUT_DIR / 'Lag6_vs_Lag14_summary.md'}")


if __name__ == "__main__":
    main()
