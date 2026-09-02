# Reproducibility code

This directory contains the Python workflow for the study
**Attention Displacement and Cross-Topic Emotional Coupling in Climate Change and COVID-19 Discourse**.
All paths are resolved relative to the package root; no author-specific absolute
paths are used.

## Directory layout

- `01_build_analysis_corpus.py`: screens the historical comment files, classifies
  topic membership, applies the early character-language, length, bot and date
  rules, removes normalized duplicate bodies, applies fastText top-1 English
  classification, and then restricts the output to the final analysis window.
- `02_score_nrc_vader.py`: calculates comment-level NRC category counts,
  NRC percentages using total whitespace-delimited word count, and VADER scores.
- `03_build_daily_time_series.py`: aggregates comment-level NRC, optional native
  LIWC-22, and VADER values by date and discourse group and joins the daily controls.
- `04_make_figure1.py`: estimates the Figure 1 OLS models and draws Figure 1.
- `04b_export_figure1_full_models.py`: exports complete Figure 1 coefficient and
  model-metadata tables.
- `05_make_figure2.py`: draws the descriptive co-mention-share figure.
- `06_make_figure3.py`: runs the paired-date Friedman tests, exports descriptive
  means, and draws the 17-indicator discourse-group comparison. Its
  supplementary outputs correspond to Supplementary Tables 4 and 5.
- `07_make_figure4.py`: selects and estimates the 34 ARDL-ECM equations and draws
  all 68 short- and long-run estimates.
- `07b_figure4_diagnostics.py`: exports stationarity, PSS bounds, coefficient,
  residual, stability, and calendar-break results for Supplementary Tables 6,
  7a, 7b, 8a, and 9.
- `07c_figure4_lag14_sensitivity.py`: compares maximum lag ceilings of six and
  fourteen days on a common estimation sample for Supplementary Table 8b.
- `07d_verify_bounds_test_covariance.py`: verifies the formal PSS Case V covariance
  specification and finite-sample critical values.
- `08_make_figure5.py`: draws the two-panel qualitative results figure from the
  coded source data.
- `run_all_figures.py`: runs the five main-figure scripts in order; the optional
  `--include-supplementary` flag also runs the table and diagnostic exports.

## Data availability within the package

The companion `Data` directory contains the daily aggregate files and figure
source data required to reproduce Figures 1-5. Comment-level records and licensed
LIWC-22 files are outside this package. Rebuilding the daily data from comment-level
records requires appending native LIWC-22 percentages before running step 03.

## Environment

Python 3.12.9 was used for the final verification. Create a fresh environment and
install the compatible ranges in `requirements.txt`. The exact tested environment
is recorded in `requirements-lock.txt`. The `fasttext-wheel` package is included
for Windows-compatible fastText language classification.

The packaged NRC rescoring script was verified with NRCLex 4.1.0 and TextBlob
0.20.1. The supplied daily aggregate and figure source files are the inputs for
reproducing the reported models and figures.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce the main figures

From this directory, run:

```bash
python run_all_figures.py
```

The scripts read from `../Data` and write journal-ready PNG, PDF, SVG,
JPEG, and TIFF outputs to `../Figures` as supported by each figure module.

To regenerate the supplementary statistical exports as well:

```bash
python run_all_figures.py --include-supplementary
```

## Rebuild from comment-level data

Steps 01-03 accept command-line arguments for the historical monthly files,
fastText model path, controls file, and licensed LIWC output. Use each script's
`--help` option to inspect the required arguments and processing summaries:

```bash
python 01_build_analysis_corpus.py --help
python 02_score_nrc_vader.py --help
python 03_build_daily_time_series.py --help
```

The retained screening chronology is encoded explicitly in step 01. The preliminary
screen begins on 20 January 2020, normalized-body deduplication precedes the final
fastText validation, and the analytic output begins on 21 January 2020. The broader
processing order is corpus screening, NRC/VADER scoring, optional LIWC-22 merge,
daily aggregation, statistical modelling, diagnostics, and figure export.
Diagnostic and processing-summary files are written to `../QA`.
