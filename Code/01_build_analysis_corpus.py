#!/usr/bin/env python3
"""Build the screened Reddit analysis corpus in the specified processing order.

The script reads one or more raw Reddit CSV files in deterministic file and
row order, processes them in chunks, and writes a new comment-level CSV. NRC,
LIWC, and VADER scoring occurs in subsequent steps. The operations comprise topic
assignment; invalid-body and observable-bot screens;
the character language prefilter; the word-count and preliminary date rules;
normalized-body deduplication; fastText top-1 English classification; and the
final analysis-window restriction. Deduplication precedes fastText because the
26,527,237-row language-classification input had already passed the preceding
screening and deduplication steps.

Author names are used transiently for the bot screen and are omitted from the
output by default. The audit JSON records mutually exclusive removal counts at
each step. The output order determines which copy is retained during dedup.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    import fasttext
except ImportError:  # The help screen remains available without fastText.
    fasttext = None


PRESCREEN_START_DATE = "2020-01-20"
START_DATE = "2020-01-21"
END_DATE = "2021-12-31"
DEFAULT_CHUNK_SIZE = 100_000

CLIMATE_PATTERN = re.compile(
    r"(?:global\s+warm|global\s+warming|climate\s+change|"
    r"climate\s+changing|climate\s+crisis|climate\s+shift|"
    r"climate\s+variation|climate\s+fluctuations|climate\s+emergency|"
    r"climate\s+instability|atmospheric\s+change|"
    r"environmental\s+climate\s+transformation|climate\s+related|"
    r"climate\s+induced|climate\s+affected|climate\s+based|"
    r"climate\s+driven|disturbing\s+the\s+climate|"
    r"upsetting\s+the\s+climate\s+balance|modifying\s+the\s+climate|"
    r"varying\s+the\s+climate|global\s+climate\s+disruption)",
    flags=re.IGNORECASE,
)
COVID_PATTERN = re.compile(
    r"(?:COVID-19|COVID|SARS-CoV-2|coronavirus|pandemic)",
    flags=re.IGNORECASE,
)

BOT_BODY_PHRASES = (
    "i am a bot",
    "i'm a bot",
    "this action was performed automatically",
    "this message was posted by a bot.",
    "this comment was left automatically (by a bot).",
    "you can summon this bot any time in",
)
BOT_AUTHOR_PATTERNS = ("-bot", "bot-", "_bot", "bot_")
INVALID_BODIES = {"", "[deleted]", "[removed]"}

STEP_NAMES = (
    "topic",
    "valid_body",
    "bot_screen",
    "character_language_prefilter",
    "word_count",
    "prescreen_time_window",
    "deduplication",
    "fasttext_language_screen",
    "analysis_time_window",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def batches(values: Sequence[str], size: int = 5_000) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def resolve_input_files(values: Sequence[str], recursive: bool) -> list[Path]:
    """Resolve files, directories, and shell-independent glob expressions."""
    resolved: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_file():
            resolved.append(path.resolve())
            continue
        if path.is_dir():
            iterator = path.rglob("*.csv*") if recursive else path.glob("*.csv*")
            resolved.extend(candidate.resolve() for candidate in iterator if candidate.is_file())
            continue

        matches = [Path(match).resolve() for match in glob.glob(value, recursive=recursive)]
        resolved.extend(match for match in matches if match.is_file())

    unique = {str(path).casefold(): path for path in resolved}
    files = sorted(unique.values(), key=lambda path: str(path).casefold())
    if not files:
        raise FileNotFoundError("No input CSV files matched --input")
    return files


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed Reddit timestamp strings as UTC."""
    return pd.to_datetime(values, errors="coerce", utc=True)


def character_heuristic(text: str) -> bool:
    """Apply the first-500-character ASCII/Latin-letter rule."""
    sample = text[:500]
    if not sample:
        return False
    ascii_ratio = sum(ord(character) < 128 for character in sample) / len(sample)
    letter_ratio = sum(
        ("a" <= character.lower() <= "z") for character in sample
    ) / len(sample)
    return ascii_ratio > 0.70 and letter_ratio > 0.30


def normalize_fasttext_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def predict_fasttext_top1(
    texts: Sequence[str], model, batch_size: int
) -> tuple[list[str], list[float]]:
    """Return fastText top-1 language labels and probabilities."""
    all_languages: list[str] = []
    all_probabilities: list[float] = []
    for group in batches(texts, batch_size):
        normalized = [normalize_fasttext_text(text) for text in group]
        labels, probabilities = model.predict(normalized, k=1)
        for row_labels, row_probabilities in zip(labels, probabilities):
            label = row_labels if isinstance(row_labels, str) else row_labels[0]
            probability = (
                row_probabilities
                if isinstance(row_probabilities, (float, int))
                else row_probabilities[0]
            )
            all_languages.append(str(label).removeprefix("__label__"))
            all_probabilities.append(float(probability))
    return all_languages, all_probabilities


def body_hash(text: str) -> bytes:
    """Match the old dedup rule: trim, lowercase, then hash with MD5."""
    normalized = text.strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).digest()


def after_step(audit: Counter[str], step: str, rows: int) -> None:
    audit[f"rows_after_{step}"] += int(rows)


def screen_chunk(
    chunk: pd.DataFrame,
    *,
    model,
    seen_hashes: set[bytes],
    prescreen_start: pd.Timestamp,
    analysis_start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    body_column: str,
    author_column: str,
    created_column: str,
    keep_author: bool,
    fasttext_batch_size: int,
    audit: Counter[str],
    language_counts: Counter[str],
) -> pd.DataFrame:
    """Apply the documented screening sequence to one source chunk."""
    audit["source_rows"] += len(chunk)

    # 1. Keyword classification and mutually exclusive discourse membership.
    bodies = chunk[body_column].fillna("").astype(str)
    chunk["climate"] = bodies.str.contains(CLIMATE_PATTERN, na=False).astype("int8")
    chunk["covid"] = bodies.str.contains(COVID_PATTERN, na=False).astype("int8")
    either_topic = chunk["climate"].eq(1) | chunk["covid"].eq(1)
    audit["excluded_no_topic"] += int((~either_topic).sum())
    chunk = chunk.loc[either_topic].copy()
    after_step(audit, "topic", len(chunk))
    if chunk.empty:
        return chunk

    # 2. Empty, deleted, and removed bodies.
    bodies = chunk[body_column].fillna("").astype(str)
    stripped_lower = bodies.str.strip().str.lower()
    valid_body = ~stripped_lower.isin(INVALID_BODIES)
    audit["excluded_invalid_body"] += int((~valid_body).sum())
    chunk = chunk.loc[valid_body].copy()
    after_step(audit, "valid_body", len(chunk))
    if chunk.empty:
        return chunk

    # 3. Observable bot markers in comment text and author name.
    bodies_lower = chunk[body_column].astype(str).str.lower()
    bot_body = bodies_lower.apply(
        lambda text: any(phrase in text for phrase in BOT_BODY_PHRASES)
    )
    audit["excluded_bot_body"] += int(bot_body.sum())
    chunk = chunk.loc[~bot_body].copy()
    if chunk.empty:
        after_step(audit, "bot_screen", 0)
        return chunk

    authors_lower = chunk[author_column].fillna("").astype(str).str.lower()
    bot_author = authors_lower.apply(
        lambda author: any(pattern in author for pattern in BOT_AUTHOR_PATTERNS)
    )
    audit["excluded_bot_author"] += int(bot_author.sum())
    chunk = chunk.loc[~bot_author].copy()
    after_step(audit, "bot_screen", len(chunk))
    if chunk.empty:
        return chunk

    # 4. Computationally inexpensive character language prefilter.
    bodies = chunk[body_column].astype(str)
    heuristic_english = bodies.apply(character_heuristic)
    audit["excluded_character_language_rule"] += int((~heuristic_english).sum())
    chunk = chunk.loc[heuristic_english].copy()
    after_step(audit, "character_language_prefilter", len(chunk))
    if chunk.empty:
        return chunk

    # 5. Whitespace-delimited word count greater than 15.
    bodies = chunk[body_column].astype(str)
    chunk["word_count"] = bodies.str.split().str.len().astype("int32")
    long_enough = chunk["word_count"].gt(15)
    audit["excluded_word_count_le_15"] += int((~long_enough).sum())
    chunk = chunk.loc[long_enough].copy()
    after_step(audit, "word_count", len(chunk))
    if chunk.empty:
        return chunk

    # 6. Preliminary retained-file window used before deduplication.
    parsed = parse_dates(chunk[created_column])
    valid_date = parsed.notna()
    in_prescreen_window = (
        valid_date & parsed.ge(prescreen_start) & parsed.lt(end_exclusive)
    )
    audit["excluded_invalid_date"] += int((~valid_date).sum())
    audit["excluded_outside_prescreen_window"] += int(
        (valid_date & ~in_prescreen_window).sum()
    )
    chunk = chunk.loc[in_prescreen_window].copy()
    parsed = parsed.loc[in_prescreen_window]
    chunk["date"] = parsed.dt.strftime("%Y-%m-%d").to_numpy()
    after_step(audit, "prescreen_time_window", len(chunk))
    if chunk.empty:
        return chunk

    # 7. Keep the first normalized body encountered across all chunks/files.
    hashes = chunk[body_column].astype(str).map(body_hash)
    keep: list[bool] = []
    for digest in hashes:
        if digest in seen_hashes:
            keep.append(False)
        else:
            seen_hashes.add(digest)
            keep.append(True)
    duplicate_count = len(keep) - sum(keep)
    audit["excluded_duplicate_body"] += duplicate_count
    chunk = chunk.loc[keep].copy()
    after_step(audit, "deduplication", len(chunk))
    if chunk.empty:
        return chunk

    # 8. fastText top-1 English, without a probability cutoff.
    texts = chunk[body_column].astype(str).tolist()
    languages, probabilities = predict_fasttext_top1(
        texts, model, fasttext_batch_size
    )
    language_counts.update(languages)
    chunk["fasttext_top_language"] = languages
    chunk["fasttext_top_probability"] = probabilities
    fasttext_english = chunk["fasttext_top_language"].eq("en")
    audit["excluded_fasttext_non_english"] += int((~fasttext_english).sum())
    chunk = chunk.loc[fasttext_english].copy()
    after_step(audit, "fasttext_language_screen", len(chunk))
    if chunk.empty:
        return chunk

    # 9. Final analysis window; the retained prescreen source included 20 January.
    analysis_window = chunk["date"].ge(analysis_start.strftime("%Y-%m-%d"))
    audit["excluded_before_analysis_start"] += int((~analysis_window).sum())
    chunk = chunk.loc[analysis_window].copy()
    after_step(audit, "analysis_time_window", len(chunk))

    if not keep_author:
        chunk = chunk.drop(columns=[author_column])

    audit["retained_rows"] += len(chunk)
    audit["retained_climate_only"] += int(
        (chunk["climate"].eq(1) & chunk["covid"].eq(0)).sum()
    )
    audit["retained_covid_only"] += int(
        (chunk["climate"].eq(0) & chunk["covid"].eq(1)).sum()
    )
    audit["retained_both"] += int(
        (chunk["climate"].eq(1) & chunk["covid"].eq(1)).sum()
    )
    return chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Raw CSV files, directories, or glob expressions.",
    )
    parser.add_argument("--output", required=True, help="Final screened CSV path.")
    parser.add_argument(
        "--fasttext-model",
        required=True,
        help="Path to the official fastText lid.176.bin model.",
    )
    parser.add_argument(
        "--prescreen-start-date",
        default=PRESCREEN_START_DATE,
        help="Earliest date retained before deduplication and fastText.",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help="First date retained in the final analytic corpus.",
    )
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--body-column", default="body")
    parser.add_argument("--author-column", default="author")
    parser.add_argument("--created-column", default="created")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--fasttext-batch-size", type=int, default=5_000)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--keep-author",
        action="store_true",
        help="Retain the author column in the private output (default: drop it).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and partial output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if fasttext is None:
        raise ImportError("Install fastText before running: pip install fasttext")
    if args.chunk_size <= 0 or args.fasttext_batch_size <= 0:
        raise ValueError("Chunk and fastText batch sizes must be positive")

    inputs = resolve_input_files(args.input, args.recursive)
    output = Path(args.output).expanduser().resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    audit_output = output.with_suffix(".audit.json")
    model_path = Path(args.fasttext_model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"fastText model not found: {model_path}")

    forbidden_inputs = {str(output).casefold(), str(partial).casefold()}
    inputs = [path for path in inputs if str(path).casefold() not in forbidden_inputs]
    if not inputs:
        raise ValueError("The output path cannot be the only matched input")

    for path in (output, partial, audit_output):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {path}")
            path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    prescreen_start = pd.Timestamp(args.prescreen_start_date, tz="UTC")
    analysis_start = pd.Timestamp(args.start_date, tz="UTC")
    end_exclusive = pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1)
    if prescreen_start > analysis_start:
        raise ValueError("--prescreen-start-date must not be later than --start-date")
    if analysis_start >= end_exclusive:
        raise ValueError("--start-date must not be later than --end-date")

    print(f"Loading fastText model: {model_path}", flush=True)
    language_model = fasttext.load_model(str(model_path))

    required = {args.body_column, args.author_column, args.created_column}
    expected_source_columns: list[str] | None = None
    output_columns: list[str] | None = None
    header_written = False
    chunk_number = 0
    audit: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    seen_hashes: set[bytes] = set()

    for file_index, input_path in enumerate(inputs, start=1):
        print(f"[{file_index}/{len(inputs)}] Reading {input_path}", flush=True)
        reader = pd.read_csv(
            input_path,
            chunksize=args.chunk_size,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
            encoding=args.encoding,
            encoding_errors="replace",
        )
        for chunk in reader:
            chunk_number += 1
            if expected_source_columns is None:
                expected_source_columns = list(chunk.columns)
                missing = sorted(required.difference(expected_source_columns))
                if missing:
                    raise ValueError(f"Missing required source columns: {missing}")
            elif list(chunk.columns) != expected_source_columns:
                raise ValueError(
                    f"Column order differs in {input_path}; all monthly files "
                    "must have the same schema"
                )

            retained = screen_chunk(
                chunk,
                model=language_model,
                seen_hashes=seen_hashes,
                prescreen_start=prescreen_start,
                analysis_start=analysis_start,
                end_exclusive=end_exclusive,
                body_column=args.body_column,
                author_column=args.author_column,
                created_column=args.created_column,
                keep_author=args.keep_author,
                fasttext_batch_size=args.fasttext_batch_size,
                audit=audit,
                language_counts=language_counts,
            )
            if not retained.empty:
                if output_columns is None:
                    output_columns = list(retained.columns)
                elif list(retained.columns) != output_columns:
                    retained = retained.reindex(columns=output_columns)
                retained.to_csv(
                    partial,
                    mode="a",
                    header=not header_written,
                    index=False,
                    lineterminator="\n",
                )
                header_written = True

            print(
                f"  chunk {chunk_number}: read={len(chunk):,}, "
                f"retained={len(retained):,}, total_retained={audit['retained_rows']:,}",
                flush=True,
            )

    if not header_written:
        raise RuntimeError("No comments passed all seven screening steps")
    partial.replace(output)

    group_total = (
        audit["retained_climate_only"]
        + audit["retained_covid_only"]
        + audit["retained_both"]
    )
    if group_total != audit["retained_rows"]:
        raise RuntimeError("Retained discourse-group counts do not sum to total")

    excluded_keys = [key for key in audit if key.startswith("excluded_")]
    excluded_total = sum(audit[key] for key in excluded_keys)
    if audit["source_rows"] - audit["retained_rows"] != excluded_total:
        raise RuntimeError("Sequential exclusion counts do not reconcile")

    payload = {
        "status": "complete",
        "completed_utc": utc_now(),
        "processing_steps": list(STEP_NAMES),
        "inputs": [file_record(path) for path in inputs],
        "output": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "author_column_retained": bool(args.keep_author),
        },
        "rules": {
            "prescreen_window_utc": [args.prescreen_start_date, args.end_date],
            "analysis_window_utc": [args.start_date, args.end_date],
            "topic_rule": "retain climate keyword and/or COVID-19 keyword match",
            "invalid_bodies": sorted(INVALID_BODIES),
            "word_count": "whitespace-delimited word_count > 15",
            "bot_body_phrases": list(BOT_BODY_PHRASES),
            "bot_author_patterns": list(BOT_AUTHOR_PATTERNS),
            "character_language_rule": (
                "first 500 characters: ASCII ratio > 0.70 and A-Z/a-z ratio > 0.30"
            ),
            "deduplication": (
                "MD5(body.strip().lower()); keep first in input order before fastText"
            ),
            "fasttext_language_rule": (
                "top-1 label == en after deduplication; no probability cutoff"
            ),
        },
        "fasttext_model": file_record(model_path),
        "counts": dict(sorted(audit.items())),
        "excluded_total": excluded_total,
        "fasttext_top_language_counts_before_final_language_filter": dict(
            sorted(language_counts.items())
        ),
        "chunk_count": chunk_number,
    }
    audit_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Completed: {output}", flush=True)
    print(f"Retained rows: {audit['retained_rows']:,}", flush=True)
    print(f"Audit: {audit_output}", flush=True)


if __name__ == "__main__":
    main()
