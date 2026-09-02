#!/usr/bin/env python3
"""Aggregate the screened, scored comments into daily analysis series.

This is pipeline step 3. It reads the comment-level CSV produced by
``02_score_nrc_vader.py`` (or the same file after the licensed LIWC-22 scores
have been merged) and creates the two compact daily tables used downstream:

* ``daily_emotion_scores.csv``: daily mean NRC, optional LIWC-22, and VADER
  scores for climate-only, COVID-only, and co-mention comments, followed by
  the date-level controls; and
* ``daily_comment_volume_and_covid_deaths.csv``: daily comment counts, the
  co-mention share, and daily US COVID-19 deaths.

NRC values are comment-level category matches divided by the whitespace word
count and multiplied by 100. VADER positive and negative proportions are also
multiplied by 100. If all five native LIWC-22 emotion columns are present,
their exported percentages are averaged without changing the LIWC denominator.

The input is read in chunks. The daily outputs contain aggregate values and omit
comment text, usernames, and comment identifiers. The script rejects dates outside the declared
analysis window by default, validates all date-group combinations, and writes
the final CSV files atomically only after every check passes.

Example
-------
python 03_build_daily_time_series.py ^
  --input "path/to/scored_comments.csv" ^
  --controls "path/to/daily_controls.csv" ^
  --output-emotions "path/to/daily_emotion_scores.csv" ^
  --output-volume "path/to/daily_comment_volume_and_covid_deaths.csv" ^
  --require-liwc --overwrite
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMOTION_OUTPUT = PROJECT_DIR / "Data" / "daily_emotion_scores.csv"
DEFAULT_VOLUME_OUTPUT = (
    PROJECT_DIR / "Data" / "daily_comment_volume_and_covid_deaths.csv"
)
DEFAULT_AUDIT_OUTPUT = PROJECT_DIR / "QA" / "daily_time_series.audit.json"

DEFAULT_START_DATE = "2020-01-21"
DEFAULT_END_DATE = "2021-12-31"
DEFAULT_EXPECTED_ROWS = 25_911_526
DEFAULT_CHUNK_SIZE = 100_000

GROUP_ORDER = ("climate", "covid", "both")
EXPECTED_GROUP_COUNTS = {
    "climate": 1_351_032,
    "covid": 24_438_852,
    "both": 121_642,
}

NRC_METRICS = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
    "positive",
    "negative",
)
LIWC_METRICS = ("emo_pos", "emo_neg", "emo_anx", "emo_anger", "emo_sad")
VADER_METRICS = ("positive", "negative")

LIWC_SOURCE_COLUMNS = {
    metric: f"LIWC_{metric}_native_pct" for metric in LIWC_METRICS
}
VADER_SOURCE_COLUMNS = {
    "positive": "VADER_positive",
    "negative": "VADER_negative",
}

CONTROL_COLUMNS = (
    "US_daily_covid_confirm",
    "US_daily_covid_death",
    "debates",
    "climatenews",
    "WinterStorm",
    "Wildfire",
    "TropicalCyclone",
    "SevereStorm",
    "Flood",
    "Drought",
    "GovernmentResponseIndex_Average",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Comment-level scored CSV.")
    parser.add_argument(
        "--controls",
        required=True,
        help="Daily CSV containing date and all established control columns.",
    )
    parser.add_argument(
        "--output-emotions",
        default=str(DEFAULT_EMOTION_OUTPUT),
        help="Daily emotion-score output CSV.",
    )
    parser.add_argument(
        "--output-volume",
        default=str(DEFAULT_VOLUME_OUTPUT),
        help="Daily comment-volume output CSV.",
    )
    parser.add_argument(
        "--audit-output",
        default=str(DEFAULT_AUDIT_OUTPUT),
        help="JSON audit record.",
    )
    parser.add_argument("--created-column", default="created")
    parser.add_argument("--climate-column", default="climate")
    parser.add_argument("--covid-column", default="covid")
    parser.add_argument("--word-count-column", default="word_count")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
        help="Expected input rows; set to 0 to disable this check.",
    )
    parser.add_argument(
        "--require-liwc",
        action="store_true",
        help="Stop unless all five native LIWC-22 emotion columns are present.",
    )
    parser.add_argument(
        "--allow-group-count-change",
        action="store_true",
        help="Do not enforce the three expected discourse-group totals.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs after all validation checks pass.",
    )
    parser.add_argument("--encoding", default="utf-8")
    return parser.parse_args()


def resolve_nrc_sources(columns: set[str]) -> tuple[dict[str, str], str]:
    explicit = {metric: f"NRC_{metric}_pct" for metric in NRC_METRICS}
    counts = {metric: f"{metric}_score" for metric in NRC_METRICS}
    if set(explicit.values()).issubset(columns):
        return explicit, "explicit_comment_level_percentages"
    if set(counts.values()).issubset(columns):
        return counts, "raw_counts_divided_by_whitespace_word_count"
    missing_explicit = sorted(set(explicit.values()).difference(columns))
    missing_counts = sorted(set(counts.values()).difference(columns))
    raise ValueError(
        "The input has neither a complete NRC percentage set nor a complete "
        f"NRC count set. Missing percentages: {missing_explicit}; "
        f"missing counts: {missing_counts}"
    )


def resolve_liwc_sources(
    columns: set[str], *, require_liwc: bool
) -> dict[str, str]:
    present = {name for name in LIWC_SOURCE_COLUMNS.values() if name in columns}
    expected = set(LIWC_SOURCE_COLUMNS.values())
    if present == expected:
        return dict(LIWC_SOURCE_COLUMNS)
    if present:
        raise ValueError(
            "Only part of the five native LIWC-22 emotion columns is present: "
            f"{sorted(present)}"
        )
    if require_liwc:
        raise ValueError(
            "--require-liwc was specified, but the five native LIWC-22 "
            "emotion columns are absent"
        )
    return {}


def load_controls(
    path: Path, *, start_date: str, end_date: str
) -> pd.DataFrame:
    required = {"date", *CONTROL_COLUMNS}
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"Control file is missing columns: {missing}")

    controls = pd.read_csv(path, usecols=["date", *CONTROL_COLUMNS])
    dates = pd.to_datetime(controls["date"], errors="raise").dt.strftime("%Y-%m-%d")
    controls["date"] = dates
    controls = controls.loc[controls["date"].between(start_date, end_date)].copy()
    if controls["date"].duplicated().any():
        duplicates = controls.loc[controls["date"].duplicated(), "date"].tolist()
        raise ValueError(f"Control file contains duplicate dates: {duplicates[:10]}")

    expected_dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    controls = controls.set_index("date").reindex(expected_dates)
    if controls.isna().any().any():
        bad = controls.index[controls.isna().any(axis=1)].tolist()
        raise ValueError(f"Controls are incomplete for dates: {bad[:10]}")
    controls.index.name = "date"
    return controls.reset_index()


def classify_groups(climate: pd.Series, covid: pd.Series) -> pd.Series:
    conditions = (
        climate.eq(1) & covid.eq(0),
        climate.eq(0) & covid.eq(1),
        climate.eq(1) & covid.eq(1),
    )
    return pd.Series(
        np.select(conditions, GROUP_ORDER, default="neither"),
        index=climate.index,
        dtype="string",
    )


def add_frame(
    accumulated: pd.DataFrame | None, current: pd.DataFrame
) -> pd.DataFrame:
    return current if accumulated is None else accumulated.add(current, fill_value=0.0)


def add_series(accumulated: pd.Series | None, current: pd.Series) -> pd.Series:
    return current if accumulated is None else accumulated.add(current, fill_value=0)


def aggregate_comments(
    source: Path,
    *,
    args: argparse.Namespace,
    nrc_sources: dict[str, str],
    nrc_mode: str,
    liwc_sources: dict[str, str],
) -> tuple[pd.DataFrame, pd.Series, dict[str, int], int]:
    base_columns = [
        args.created_column,
        args.climate_column,
        args.covid_column,
        *nrc_sources.values(),
        *liwc_sources.values(),
        *VADER_SOURCE_COLUMNS.values(),
    ]
    if nrc_mode == "raw_counts_divided_by_whitespace_word_count":
        base_columns.append(args.word_count_column)
    read_columns = list(dict.fromkeys(base_columns))

    accumulated_sum: pd.DataFrame | None = None
    accumulated_size: pd.Series | None = None
    group_counts: Counter[str] = Counter()
    total_rows = 0
    expected_metric_columns = [
        *(f"nrc_{metric}" for metric in NRC_METRICS),
        *(f"liwc_{metric}" for metric in liwc_sources),
        *(f"vader_{metric}" for metric in VADER_METRICS),
    ]

    reader = pd.read_csv(
        source,
        usecols=read_columns,
        chunksize=args.chunk_size,
        encoding=args.encoding,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        date_text = chunk[args.created_column].astype("string").str.slice(0, 10)
        parsed_dates = pd.to_datetime(date_text, errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError(f"Invalid created date in input chunk {chunk_number}")
        chunk["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
        outside = ~chunk["date"].between(args.start_date, args.end_date)
        if outside.any():
            examples = chunk.loc[outside, "date"].value_counts().head().to_dict()
            raise ValueError(
                "The scored input contains rows outside the declared analysis "
                f"window in chunk {chunk_number}: {examples}"
            )

        climate = pd.to_numeric(chunk[args.climate_column], errors="coerce")
        covid = pd.to_numeric(chunk[args.covid_column], errors="coerce")
        if climate.isna().any() or covid.isna().any():
            raise ValueError(f"Missing topic flag in input chunk {chunk_number}")
        if (~climate.isin((0, 1))).any() or (~covid.isin((0, 1))).any():
            raise ValueError(f"Non-binary topic flag in input chunk {chunk_number}")
        chunk["group"] = classify_groups(climate, covid)
        if (chunk["group"] == "neither").any():
            raise ValueError(f"Comment without either topic in chunk {chunk_number}")

        if nrc_mode == "raw_counts_divided_by_whitespace_word_count":
            denominator = pd.to_numeric(
                chunk[args.word_count_column], errors="coerce"
            )
            if denominator.isna().any() or denominator.le(0).any():
                raise ValueError(f"Invalid word_count in input chunk {chunk_number}")
        else:
            denominator = None

        work = chunk[["date", "group"]].copy()
        for metric, source_column in nrc_sources.items():
            values = pd.to_numeric(chunk[source_column], errors="coerce")
            if values.isna().any():
                raise ValueError(
                    f"Missing NRC value in {source_column}, chunk {chunk_number}"
                )
            if denominator is not None:
                values = values / denominator * 100.0
            work[f"nrc_{metric}"] = values

        for metric, source_column in liwc_sources.items():
            values = pd.to_numeric(chunk[source_column], errors="coerce")
            if values.isna().any() or values.lt(0).any() or values.gt(100).any():
                raise ValueError(
                    f"Invalid native LIWC value in {source_column}, "
                    f"chunk {chunk_number}"
                )
            work[f"liwc_{metric}"] = values

        for metric, source_column in VADER_SOURCE_COLUMNS.items():
            values = pd.to_numeric(chunk[source_column], errors="coerce")
            if values.isna().any() or values.lt(0).any() or values.gt(1).any():
                raise ValueError(
                    f"Invalid VADER value in {source_column}, chunk {chunk_number}"
                )
            work[f"vader_{metric}"] = values * 100.0

        if list(work.columns[2:]) != expected_metric_columns:
            raise RuntimeError("Internal daily metric order changed unexpectedly")
        grouped = work.groupby(["date", "group"], observed=True, sort=False)
        accumulated_sum = add_frame(
            accumulated_sum, grouped[expected_metric_columns].sum()
        )
        accumulated_size = add_series(accumulated_size, grouped.size())

        counts = work["group"].value_counts()
        group_counts.update({str(key): int(value) for key, value in counts.items()})
        if chunk_number % 20 == 0:
            print(
                f"Aggregated chunks={chunk_number:,}; rows={total_rows:,}",
                flush=True,
            )

    if accumulated_sum is None or accumulated_size is None:
        raise RuntimeError("No comment rows were aggregated")
    if args.expected_rows and total_rows != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows:,} input rows, found {total_rows:,}"
        )
    observed_counts = {group: int(group_counts.get(group, 0)) for group in GROUP_ORDER}
    if sum(observed_counts.values()) != total_rows:
        raise RuntimeError("Discourse-group counts do not sum to total input rows")
    if not args.allow_group_count_change and observed_counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(
            f"Expected group counts {EXPECTED_GROUP_COUNTS}, found {observed_counts}"
        )

    mean = accumulated_sum.div(accumulated_size, axis=0).sort_index()
    return mean, accumulated_size.astype(int).sort_index(), observed_counts, total_rows


def complete_index(start_date: str, end_date: str) -> pd.MultiIndex:
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    return pd.MultiIndex.from_product(
        [dates, GROUP_ORDER], names=["date", "group"]
    )


def build_emotion_table(
    mean: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    liwc_sources: dict[str, str],
) -> pd.DataFrame:
    index = complete_index(start_date, end_date)
    missing = index.difference(mean.index)
    extra = mean.index.difference(index)
    if len(missing) or len(extra):
        raise RuntimeError(
            f"Incomplete daily panel; missing={missing.tolist()[:10]}, "
            f"extra={extra.tolist()[:10]}"
        )
    mean = mean.reindex(index)
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    daily = pd.DataFrame({"date": dates})

    for metric in NRC_METRICS:
        pivot = mean[f"nrc_{metric}"].unstack("group")
        for group in GROUP_ORDER:
            daily[f"{group}_{metric}_score_freq"] = pivot[group].to_numpy()
    for metric in liwc_sources:
        pivot = mean[f"liwc_{metric}"].unstack("group")
        for group in GROUP_ORDER:
            daily[f"{group}_liwc_{metric}_pct"] = pivot[group].to_numpy()
    for metric in VADER_METRICS:
        pivot = mean[f"vader_{metric}"].unstack("group")
        for group in GROUP_ORDER:
            daily[f"{group}_vader_{metric}_pct"] = pivot[group].to_numpy()

    daily = daily.merge(controls, on="date", how="left", validate="one_to_one")
    if daily.isna().any().any():
        raise RuntimeError("Daily emotion table contains missing values")
    return daily


def build_volume_table(
    counts: pd.Series,
    controls: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    index = complete_index(start_date, end_date)
    counts = counts.reindex(index, fill_value=0).unstack("group")
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    volume = pd.DataFrame(
        {
            "date": dates,
            "climate_only_comments": counts["climate"].to_numpy(dtype=int),
            "covid_only_comments": counts["covid"].to_numpy(dtype=int),
            "co_mention_comments": counts["both"].to_numpy(dtype=int),
        }
    )
    count_columns = (
        "climate_only_comments",
        "covid_only_comments",
        "co_mention_comments",
    )
    volume["total_comments"] = volume[list(count_columns)].sum(axis=1)
    climate_related_total = (
        volume["climate_only_comments"] + volume["co_mention_comments"]
    )
    volume["co_mention_share_pct"] = (
        volume["co_mention_comments"] / climate_related_total * 100.0
    )
    volume = volume.merge(
        controls[["date", "US_daily_covid_death"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    if volume.isna().any().any():
        raise RuntimeError("Daily volume table contains missing values")
    return volume


def write_csv_atomic(frame: pd.DataFrame, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    frame.to_csv(temporary, index=False, lineterminator="\n")
    check = pd.read_csv(temporary)
    if list(check.columns) != list(frame.columns) or len(check) != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Written CSV failed independent validation: {output}")
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.expected_rows < 0:
        raise ValueError("chunk-size must be positive and expected-rows non-negative")
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start > end:
        raise ValueError("start-date must not be later than end-date")

    source = Path(args.input).expanduser().resolve()
    controls_path = Path(args.controls).expanduser().resolve()
    emotion_output = Path(args.output_emotions).expanduser().resolve()
    volume_output = Path(args.output_volume).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Scored comment CSV not found: {source}")
    if not controls_path.is_file():
        raise FileNotFoundError(f"Control CSV not found: {controls_path}")
    if emotion_output == volume_output:
        raise ValueError("Emotion and volume outputs must be different files")
    if not args.overwrite:
        existing = [path for path in (emotion_output, volume_output) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output exists; use --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    input_signature = file_signature(source)
    control_signature = file_signature(controls_path)

    header = set(pd.read_csv(source, nrows=0).columns)
    required_base = {
        args.created_column,
        args.climate_column,
        args.covid_column,
        *VADER_SOURCE_COLUMNS.values(),
    }
    missing_base = sorted(required_base.difference(header))
    if missing_base:
        raise ValueError(f"Scored input is missing columns: {missing_base}")
    nrc_sources, nrc_mode = resolve_nrc_sources(header)
    if nrc_mode == "raw_counts_divided_by_whitespace_word_count":
        if args.word_count_column not in header:
            raise ValueError(
                f"NRC counts require denominator column: {args.word_count_column}"
            )
    liwc_sources = resolve_liwc_sources(header, require_liwc=args.require_liwc)

    # Load all controls before any output replacement. This also permits an
    # existing daily_emotion_scores.csv to serve as the control source.
    controls = load_controls(
        controls_path, start_date=args.start_date, end_date=args.end_date
    )
    mean, counts, group_counts, total_rows = aggregate_comments(
        source,
        args=args,
        nrc_sources=nrc_sources,
        nrc_mode=nrc_mode,
        liwc_sources=liwc_sources,
    )
    emotions = build_emotion_table(
        mean,
        controls,
        start_date=args.start_date,
        end_date=args.end_date,
        liwc_sources=liwc_sources,
    )
    volume = build_volume_table(
        counts,
        controls,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    expected_dates = len(pd.date_range(args.start_date, args.end_date, freq="D"))
    if len(emotions) != expected_dates or len(volume) != expected_dates:
        raise RuntimeError("Daily output row count does not match the date window")
    if int(volume["total_comments"].sum()) != total_rows:
        raise RuntimeError("Daily comment totals do not reproduce input row count")

    write_csv_atomic(emotions, emotion_output, overwrite=args.overwrite)
    write_csv_atomic(volume, volume_output, overwrite=args.overwrite)

    audit = {
        "status": "complete",
        "completed_utc": utc_now(),
        "input": input_signature,
        "control_source": control_signature,
        "analysis_window": [args.start_date, args.end_date],
        "input_rows": total_rows,
        "group_counts": group_counts,
        "daily_rows": expected_dates,
        "daily_emotion_columns": len(emotions.columns),
        "daily_volume_columns": len(volume.columns),
        "nrc_source_mode": nrc_mode,
        "nrc_formula": (
            "comment-level NRC category matches / whitespace word_count * 100; "
            "arithmetic mean by date and discourse group"
        ),
        "liwc_included": bool(liwc_sources),
        "liwc_formula": (
            "native LIWC-22 percentage using LIWC tokenization and WC; "
            "arithmetic mean by date and discourse group"
            if liwc_sources
            else None
        ),
        "vader_formula": (
            "VADER positive or negative proportion * 100; arithmetic mean by "
            "date and discourse group"
        ),
        "emotion_output": file_signature(emotion_output),
        "volume_output": file_signature(volume_output),
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
