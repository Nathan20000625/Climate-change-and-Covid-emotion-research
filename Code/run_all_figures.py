"""Run the main-figure workflow in a fixed order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent

MAIN_SCRIPTS = [
    "04_make_figure1.py",
    "05_make_figure2.py",
    "06_make_figure3.py",
    "07_make_figure4.py",
    "08_make_figure5.py",
]

SUPPLEMENTARY_SCRIPTS = [
    "04b_export_figure1_full_models.py",
    "07b_figure4_diagnostics.py",
    "07c_figure4_lag14_sensitivity.py",
    "07d_verify_bounds_test_covariance.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-supplementary",
        action="store_true",
        help="Also regenerate model tables, diagnostics, and sensitivity exports.",
    )
    return parser.parse_args()


def run_script(filename: str) -> None:
    path = CODE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"\n=== Running {filename} ===", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=CODE_DIR, check=True)


def main() -> None:
    args = parse_args()
    for filename in MAIN_SCRIPTS:
        run_script(filename)
    if args.include_supplementary:
        for filename in SUPPLEMENTARY_SCRIPTS:
            run_script(filename)
    print("\nAll requested scripts completed successfully.")


if __name__ == "__main__":
    main()
