#!/usr/bin/env python3
"""Calculate comment-level NRC and VADER scores for the screened corpus.

This is pipeline step 2. It reads the output of
``01_build_analysis_corpus.py`` without applying any additional sample filter.
For each retained comment it calculates:

* raw NRC match counts for eight emotions plus positive and negative;
* NRC intensity as category matches / whitespace word_count * 100; and
* VADER neg, neu, pos, and compound scores.

The ``<category>_score`` NRC count columns are retained for downstream daily
aggregation. The explicit ``NRC_<category>_pct`` columns record percentages
based on each comment's whitespace-delimited word count. LIWC scores are
calculated separately with the licensed LIWC software.

Scoring is chunked and parallel. Each completed chunk is stored as a compressed
Parquet part with metadata, allowing an interrupted run to resume without
rescoring completed chunks. The parts are concatenated into one final CSV only
after every source chunk has been validated.

Example
-------
python 02_score_nrc_vader.py \
    --input "path/to/analysis_corpus.csv" \
    --output "path/to/scored_nrc_vader.csv"
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from nrclex import NRCLex
except ImportError:
    try:
        from NRCLex import NRCLex
    except ImportError:
        NRCLex = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


NRC_CATEGORIES = (
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
VADER_COLUMNS = (
    "VADER_negative",
    "VADER_neutral",
    "VADER_positive",
    "VADER_compound",
)

DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_BATCH_SIZE = 250
DEFAULT_WORKERS = min(12, max(1, (os.cpu_count() or 4) - 2))
SCORING_SCHEMA_VERSION = 2

_VADER_ANALYZER = None
_NRC_ANALYZER = None
_NRC_LEXICON = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def scoring_configuration(*, allow_scoring_errors: bool) -> dict[str, object]:
    """Return every setting that can change a completed part's scores."""
    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "nrclex_version": package_version("nrclex"),
        "vader_sentiment_version": package_version("vaderSentiment"),
        "nrc_categories": list(NRC_CATEGORIES),
        "nrc_engine": (
            "NRCLex bundled lexicon with TextBlob word tokenization and "
            "default lemmatization; raw-count equivalent to NRCLex 4.1"
        ),
        "nrc_denominator": "input whitespace word_count",
        "nrc_scale": 100.0,
        "vader_columns": list(VADER_COLUMNS),
        "allow_scoring_errors": allow_scoring_errors,
    }


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def initialize_worker() -> None:
    global _NRC_ANALYZER, _NRC_LEXICON, _VADER_ANALYZER
    if NRCLex is None:
        raise ImportError("Install NRCLex: pip install nrclex")
    if TextBlob is None:
        raise ImportError("Install TextBlob: pip install textblob")
    if SentimentIntensityAnalyzer is None:
        raise ImportError("Install vaderSentiment: pip install vaderSentiment")
    # NRCLex 4.x loads the lexicon in the constructor and analyzes text through
    # load_raw_text(). Reuse one lexicon object per worker. NRCLex 3.x instead
    # accepts text in its constructor, so no persistent analyzer is created.
    if hasattr(NRCLex, "load_raw_text"):
        _NRC_ANALYZER = NRCLex()
        _NRC_LEXICON = _NRC_ANALYZER.__lexicon__
    _VADER_ANALYZER = SentimentIntensityAnalyzer()


def nrc_raw_scores(text: str) -> dict[str, int]:
    """Return raw NRC counts, avoiding NRCLex work unrelated to raw scores.

    NRCLex 4.1 ``load_raw_text`` performs word tokenization, default
    lemmatization, sentence segmentation, lexicon lookup, and top-emotion
    calculation. Raw counts depend only on tokenization, lemmatization, and
    lookup. Repeating those three operations gives identical raw counts while
    avoiding the expensive sentence and top-emotion work.
    """
    global _NRC_ANALYZER, _NRC_LEXICON
    if NRCLex is None:
        raise ImportError("Install NRCLex: pip install nrclex")
    if hasattr(NRCLex, "load_raw_text"):
        if _NRC_ANALYZER is None:
            _NRC_ANALYZER = NRCLex()
            _NRC_LEXICON = _NRC_ANALYZER.__lexicon__
        words = [word.lemmatize() for word in TextBlob(text).words]
        labels = chain.from_iterable(
            _NRC_LEXICON[word] for word in words if word in _NRC_LEXICON
        )
        return dict(Counter(labels))
    return dict(NRCLex(text).raw_emotion_scores)


def score_batch(
    texts: Sequence[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, int, int]:
    """Score one text batch inside a worker process."""
    if NRCLex is None:
        raise ImportError("Install NRCLex: pip install nrclex")
    if _VADER_ANALYZER is None:
        initialize_worker()

    n = len(texts)
    nrc_counts = np.zeros((n, len(NRC_CATEGORIES)), dtype=np.int32)
    dominant_emotions: list[str] = []
    confidence = np.zeros(n, dtype=np.float32)
    vader = np.zeros((n, len(VADER_COLUMNS)), dtype=np.float32)
    nrc_errors = 0
    vader_errors = 0

    for index, text in enumerate(texts):
        try:
            raw_scores = nrc_raw_scores(text)
            row = np.asarray(
                [int(raw_scores.get(category, 0)) for category in NRC_CATEGORIES],
                dtype=np.int32,
            )
            nrc_counts[index] = row
            total = int(row.sum())
            if total > 0:
                best_index = int(np.argmax(row))
                dominant_emotions.append(NRC_CATEGORIES[best_index])
                confidence[index] = float(row[best_index] / total)
            else:
                dominant_emotions.append("neutral")
        except Exception:
            nrc_errors += 1
            dominant_emotions.append("neutral")

        try:
            scores = _VADER_ANALYZER.polarity_scores(text)
            vader[index] = (
                scores["neg"],
                scores["neu"],
                scores["pos"],
                scores["compound"],
            )
        except Exception:
            vader_errors += 1

    return nrc_counts, dominant_emotions, confidence, vader, nrc_errors, vader_errors


def score_parallel(
    texts: Sequence[str], executor: ProcessPoolExecutor, batch_size: int
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, int, int]:
    results = executor.map(score_batch, batches(texts, batch_size), chunksize=1)
    count_parts: list[np.ndarray] = []
    emotion_parts: list[str] = []
    confidence_parts: list[np.ndarray] = []
    vader_parts: list[np.ndarray] = []
    nrc_errors = 0
    vader_errors = 0
    for counts, emotions, confidence, vader, nrc_bad, vader_bad in results:
        count_parts.append(counts)
        emotion_parts.extend(emotions)
        confidence_parts.append(confidence)
        vader_parts.append(vader)
        nrc_errors += nrc_bad
        vader_errors += vader_bad

    return (
        np.concatenate(count_parts, axis=0),
        emotion_parts,
        np.concatenate(confidence_parts, axis=0),
        np.concatenate(vader_parts, axis=0),
        nrc_errors,
        vader_errors,
    )


def append_scores(
    chunk: pd.DataFrame,
    *,
    source_start_row: int,
    body_column: str,
    word_count_column: str,
    executor: ProcessPoolExecutor,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add NRC/VADER columns without changing corpus membership or row order."""
    texts = chunk[body_column].fillna("").astype(str).tolist()
    word_count = pd.to_numeric(chunk[word_count_column], errors="coerce")
    if word_count.isna().any() or word_count.le(0).any():
        raise ValueError("All screened comments must have a positive word_count")

    started = time.perf_counter()
    counts, emotions, confidence, vader, nrc_errors, vader_errors = score_parallel(
        texts, executor, batch_size
    )
    scoring_seconds = time.perf_counter() - started

    chunk = chunk.copy()
    if "analysis_id" in chunk.columns:
        observed = pd.to_numeric(chunk["analysis_id"], errors="coerce")
        expected = np.arange(source_start_row + 1, source_start_row + len(chunk) + 1)
        if observed.isna().any() or not np.array_equal(observed.to_numpy(), expected):
            raise ValueError("Existing analysis_id is not sequential in source row order")
    else:
        chunk.insert(
            0,
            "analysis_id",
            np.arange(source_start_row + 1, source_start_row + len(chunk) + 1),
        )

    denominator = word_count.to_numpy(dtype=np.float64)
    for column_index, category in enumerate(NRC_CATEGORIES):
        legacy_count_column = f"{category}_score"
        intensity_column = f"NRC_{category}_pct"
        chunk[legacy_count_column] = counts[:, column_index]
        chunk[intensity_column] = counts[:, column_index] / denominator * 100.0

    chunk["emotion"] = emotions
    chunk["emotion_confidence"] = confidence
    for column_index, column in enumerate(VADER_COLUMNS):
        chunk[column] = vader[:, column_index]

    component_error = np.abs(vader[:, :3].sum(axis=1) - 1.0)
    metadata = {
        "input_rows": len(chunk),
        "output_rows": len(chunk),
        "source_start_row": source_start_row,
        "nrc_errors": nrc_errors,
        "vader_errors": vader_errors,
        "scoring_seconds": scoring_seconds,
        "maximum_abs_error_vader_neg_neu_pos_sum_minus_one": float(
            component_error.max(initial=0.0)
        ),
    }
    return chunk, metadata


def part_paths(parts_dir: Path, chunk_number: int) -> tuple[Path, Path]:
    stem = f"part_{chunk_number:04d}"
    return parts_dir / f"{stem}.parquet", parts_dir / f"{stem}.json"


def completed_part_is_valid(
    parquet_path: Path,
    metadata_path: Path,
    *,
    source: dict[str, object],
    configuration: dict[str, object],
    source_start_row: int,
    input_rows: int,
) -> bool:
    if not parquet_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "status": "complete",
        "source": source,
        "scoring_configuration": configuration,
        "source_start_row": source_start_row,
        "input_rows": input_rows,
        "output_rows": input_rows,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def write_part_atomic(
    frame: pd.DataFrame,
    metadata: dict[str, object],
    parquet_path: Path,
    metadata_path: Path,
) -> None:
    temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    frame.to_parquet(temporary_parquet, index=False, compression="zstd")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_parquet.replace(parquet_path)
    temporary_metadata.replace(metadata_path)


def finalize_csv(parts_dir: Path, chunk_count: int, output: Path) -> int:
    """Join validated parts while checking schema and analysis_id continuity."""
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    header_written = False
    rows_written = 0
    output_columns: list[str] | None = None
    next_analysis_id = 1
    for chunk_number in range(chunk_count):
        parquet_path, metadata_path = part_paths(parts_dir, chunk_number)
        if not parquet_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Missing completed part {chunk_number}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise RuntimeError(f"Incomplete metadata for part {chunk_number}")
        frame = pd.read_parquet(parquet_path)
        if len(frame) != int(metadata["output_rows"]):
            raise RuntimeError(f"Row-count mismatch in part {chunk_number}")
        if output_columns is None:
            output_columns = list(frame.columns)
        elif list(frame.columns) != output_columns:
            raise RuntimeError(f"Column mismatch in part {chunk_number}")
        if "analysis_id" not in frame.columns:
            raise RuntimeError(f"analysis_id missing from part {chunk_number}")
        observed_ids = pd.to_numeric(frame["analysis_id"], errors="coerce").to_numpy()
        expected_ids = np.arange(next_analysis_id, next_analysis_id + len(frame))
        if np.isnan(observed_ids).any() or not np.array_equal(observed_ids, expected_ids):
            raise RuntimeError(f"analysis_id sequence mismatch in part {chunk_number}")
        frame.to_csv(
            partial,
            mode="a",
            header=not header_written,
            index=False,
            lineterminator="\n",
        )
        header_written = True
        rows_written += len(frame)
        next_analysis_id += len(frame)
        print(
            f"Finalizing {chunk_number + 1}/{chunk_count}: {len(frame):,} rows",
            flush=True,
        )
    partial.replace(output)
    return rows_written


def validate_dependencies() -> None:
    if NRCLex is None:
        raise ImportError("Install NRCLex before running: pip install nrclex")
    if SentimentIntensityAnalyzer is None:
        raise ImportError(
            "Install vaderSentiment before running: pip install vaderSentiment"
        )
    if TextBlob is None:
        raise ImportError("Install TextBlob before running: pip install textblob")
    try:
        if hasattr(NRCLex, "load_raw_text"):
            probe = NRCLex()
            probe_texts = (
                "Climate policy can inspire hope and trust.",
                "Wildfires caused fear, sadness, and anger.",
                "The studies were surprising but the result was not bad.",
            )
            for probe_text in probe_texts:
                probe.load_raw_text(probe_text)
                native_scores = dict(probe.raw_emotion_scores)
                optimized_scores = nrc_raw_scores(probe_text)
                if native_scores != optimized_scores:
                    raise RuntimeError(
                        "Optimized NRC lookup differs from the installed "
                        "NRCLex implementation"
                    )
        else:
            probe = NRCLex("climate policy can inspire hope and trust")
            _ = probe.raw_emotion_scores
    except Exception as exc:
        raise RuntimeError(
            "NRCLex could not tokenize a test sentence. Install its language "
            "resources, for example: python -m textblob.download_corpora"
        ) from exc
    analyzer = SentimentIntensityAnalyzer()
    probe_vader = analyzer.polarity_scores("This validation test is good.")
    if set(probe_vader) != {"neg", "neu", "pos", "compound"}:
        raise RuntimeError("Unexpected VADER output schema")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Screened corpus CSV.")
    parser.add_argument("--output", required=True, help="Final scored CSV.")
    parser.add_argument("--body-column", default="body")
    parser.add_argument("--word-count-column", default="word_count")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--parts-dir",
        help="Optional resumable part directory; defaults beside --output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output and completed parts before starting.",
    )
    parser.add_argument(
        "--allow-scoring-errors",
        action="store_true",
        help=(
            "Continue if individual NRC/VADER calls fail and retain zero defaults. "
            "The scientific default is to stop on any scoring error."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("Chunk size, batch size, and workers must be positive")
    validate_dependencies()

    source_path = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Screened corpus not found: {source_path}")
    if source_path == output:
        raise ValueError("Input and output paths must differ")

    parts_dir = (
        Path(args.parts_dir).expanduser().resolve()
        if args.parts_dir
        else output.parent / f"{output.stem}_parts"
    )
    audit_output = output.with_suffix(".audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    if args.overwrite:
        for path in (output, output.with_suffix(output.suffix + ".partial"), audit_output):
            if path.exists():
                path.unlink()
        for pattern in ("part_*.parquet", "part_*.json", "*.tmp"):
            for path in parts_dir.glob(pattern):
                path.unlink()

    source = file_signature(source_path)
    configuration = scoring_configuration(
        allow_scoring_errors=args.allow_scoring_errors
    )
    required = {args.body_column, args.word_count_column}
    source_start_row = 0
    chunk_count = 0
    total_nrc_errors = 0
    total_vader_errors = 0
    maximum_vader_component_error = 0.0
    scoring_seconds = 0.0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
    ) as executor:
        reader = pd.read_csv(
            source_path,
            chunksize=args.chunk_size,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
            encoding=args.encoding,
            encoding_errors="replace",
        )
        for chunk_number, chunk in enumerate(reader):
            chunk_count += 1
            missing = sorted(required.difference(chunk.columns))
            if missing:
                raise ValueError(f"Missing required input columns: {missing}")

            parquet_path, metadata_path = part_paths(parts_dir, chunk_number)
            if completed_part_is_valid(
                parquet_path,
                metadata_path,
                source=source,
                configuration=configuration,
                source_start_row=source_start_row,
                input_rows=len(chunk),
            ):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not args.allow_scoring_errors and (
                    int(metadata["nrc_errors"]) > 0
                    or int(metadata["vader_errors"]) > 0
                ):
                    raise RuntimeError(
                        f"Completed part {chunk_number} contains scoring errors; "
                        "rerun with --overwrite after investigating"
                    )
                total_nrc_errors += int(metadata["nrc_errors"])
                total_vader_errors += int(metadata["vader_errors"])
                scoring_seconds += float(metadata["scoring_seconds"])
                maximum_vader_component_error = max(
                    maximum_vader_component_error,
                    float(metadata["maximum_abs_error_vader_neg_neu_pos_sum_minus_one"]),
                )
                print(
                    f"[{chunk_number + 1}] Reusing completed part: {len(chunk):,} rows",
                    flush=True,
                )
            else:
                scored, metrics = append_scores(
                    chunk,
                    source_start_row=source_start_row,
                    body_column=args.body_column,
                    word_count_column=args.word_count_column,
                    executor=executor,
                    batch_size=args.batch_size,
                )
                metadata = {
                    "status": "complete",
                    "completed_utc": utc_now(),
                    "source": source,
                    "scoring_configuration": configuration,
                    "chunk_number": chunk_number,
                    **metrics,
                }
                if not args.allow_scoring_errors and (
                    int(metrics["nrc_errors"]) > 0
                    or int(metrics["vader_errors"]) > 0
                ):
                    raise RuntimeError(
                        f"Scoring failed for {metrics['nrc_errors']} NRC and "
                        f"{metrics['vader_errors']} VADER comments in chunk "
                        f"{chunk_number}; no part was written"
                    )
                write_part_atomic(scored, metadata, parquet_path, metadata_path)
                total_nrc_errors += int(metrics["nrc_errors"])
                total_vader_errors += int(metrics["vader_errors"])
                scoring_seconds += float(metrics["scoring_seconds"])
                maximum_vader_component_error = max(
                    maximum_vader_component_error,
                    float(metrics["maximum_abs_error_vader_neg_neu_pos_sum_minus_one"]),
                )
                print(
                    f"[{chunk_number + 1}] Scored and saved: {len(chunk):,} rows",
                    flush=True,
                )
            source_start_row += len(chunk)

    if chunk_count == 0:
        raise RuntimeError("The screened input corpus contains no rows")
    rows_written = finalize_csv(parts_dir, chunk_count, output)
    if rows_written != source_start_row:
        raise RuntimeError("Final output row count differs from screened input")

    payload = {
        "status": "complete",
        "completed_utc": utc_now(),
        "pipeline_step": 2,
        "input": source,
        "output": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "rows": rows_written,
        },
        "parts_directory": str(parts_dir),
        "chunk_count": chunk_count,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "scoring_configuration": configuration,
        "nrc": {
            "categories": list(NRC_CATEGORIES),
            "count_columns": [f"{category}_score" for category in NRC_CATEGORIES],
            "intensity_columns": [f"NRC_{category}_pct" for category in NRC_CATEGORIES],
            "formula": "category match count / whitespace word_count * 100",
            "errors": total_nrc_errors,
            "package_version": package_version("nrclex"),
        },
        "vader": {
            "columns": list(VADER_COLUMNS),
            "errors": total_vader_errors,
            "maximum_abs_error_neg_neu_pos_sum_minus_one": maximum_vader_component_error,
            "package_version": package_version("vaderSentiment"),
        },
        "scoring_seconds_summed_across_parts": scoring_seconds,
        "sample_membership_changed": False,
        "row_order_changed": False,
        "liwc_included": False,
    }
    audit_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed: {output}", flush=True)
    print(f"Rows scored: {rows_written:,}", flush=True)
    print(f"Audit: {audit_output}", flush=True)


if __name__ == "__main__":
    main()
