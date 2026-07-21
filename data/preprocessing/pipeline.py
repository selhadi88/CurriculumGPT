"""
Raw → processed data pipeline.

RECONSTRUCTED — this module does not exist anywhere in the original repository
history and there is no design doc, docstring, or partial implementation of it
to recover from (checked exhaustively: no reference to "dedup", "language
filter", "text cleaning", or "metadata flatten" anywhere in the codebase). The
only concrete evidence available was indirect:

  - the raw JSONL shape written by scripts/scrape.py's save_jsonl():
    {"title": ..., "description": ..., "source": ..., **metadata}
    under data/raw/{courses,jobs}/*.jsonl
  - the processed CSV schema expected by backend/app/db/init_db.py and
    scripts/build_dataset.py under data/processed/{courses,jobs}/*.csv
  - README.md's own admission that data/scrapers/ and data/preprocessing/
    were excluded from the repo and "would need to be built or restored"

Everything below — the cleaning rules, the dedup strategy, the language
filter, and the metadata-flattening scheme — is a fresh, from-scratch
implementation built to those two known endpoints (raw shape in, processed
schema out), not a recovery of lost code. See README.md for the up-to-date
list of what was reconstructed vs. what is unchanged.

Pipeline steps (per raw record, in order):
    1. text cleaning   — strip HTML tags, normalize unicode (NFKC), collapse
                          whitespace, drop control characters
    2. language filter — keep only records whose title+description is
                          detected as English (langdetect, with a heuristic
                          ASCII/stopword fallback when langdetect can't decide
                          on very short text)
    3. metadata flatten — nested dict values become dot-notation columns;
                          list/tuple values are JSON-stringified so every
                          record fits a flat CSV row
    4. deduplication    — drop exact duplicates on normalized (lowercased,
                          whitespace-collapsed) title+description

Usage:
    from data.preprocessing.pipeline import load_processed, run_full_pipeline

    courses = load_processed("courses")
    counts = run_full_pipeline()   # {"courses": N, "jobs": M}
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _REPO_ROOT / "data" / "raw"
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Fallback English heuristic, used only when langdetect can't make a call
# (empty/very short text raises LangDetectException). Not a real language
# model — just enough signal to reject obviously non-English/garbage text
# when the proper detector has nothing to work with.
_ENGLISH_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was",
    "will", "have", "has", "you", "your", "our", "their", "of", "to", "in",
    "is", "on", "as", "an", "a",
}


# --------------------------------------------------------------------------- cleaning

def _fix_mojibake(text: str) -> str:
    """
    Repair double-UTF-8-encoded text (e.g. RemoteOK's API, observed live:
    "Herói" -> correct UTF-8 bytes -> misread as Latin-1 -> re-encoded to
    UTF-8 -> "HerÃ³i"). ftfy detects this pattern (and several others) and
    only touches text that actually looks mis-encoded, so it's safe to run
    unconditionally on already-correct text too.
    """
    try:
        import ftfy

        return ftfy.fix_text(text)
    except ImportError:
        logger.debug("ftfy not installed — skipping mojibake repair")
        return text


def clean_text(text: Any) -> str:
    """
    Repair mojibake, strip HTML tags, normalize unicode to NFKC, drop control
    characters, and collapse all whitespace (including newlines/tabs) to
    single spaces.
    """
    if text is None:
        return ""
    text = str(text)
    text = _fix_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ------------------------------------------------------------------- language filter

def _heuristic_is_english(text: str) -> bool:
    """Cheap fallback when langdetect can't decide (text too short/empty)."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_ratio = sum(1 for c in letters if ord(c) < 128) / len(letters)
    if ascii_ratio < 0.85:
        return False
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    # A genuine English sentence of any length almost always contains at
    # least one common stopword; pure ASCII gibberish or another
    # Latin-alphabet language usually won't overlap this specific set.
    return bool(words & _ENGLISH_STOPWORDS) or len(words) <= 3


def is_english(text: str) -> bool:
    """Best-effort English-language filter: langdetect first, heuristic fallback."""
    text = text.strip()
    if not text:
        return False
    try:
        from langdetect import DetectorFactory, LangDetectException, detect

        DetectorFactory.seed = 0  # deterministic results across runs
        return detect(text) == "en"
    except ImportError:
        logger.debug("langdetect not installed — using heuristic English filter")
        return _heuristic_is_english(text)
    except LangDetectException:
        # Too short / no textual features for langdetect to work with.
        return _heuristic_is_english(text)


# ------------------------------------------------------------------ metadata flatten

def flatten_record(record: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten a raw scraped record into single-level scalar columns so it fits
    a flat CSV row:
      - nested dicts    -> dot-notation keys, recursively
      - lists/tuples    -> JSON-stringified (can't otherwise fit a CSV cell)
      - scalars         -> passed through unchanged
    """
    flat: dict[str, Any] = {}
    for key, value in record.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_record(value, full_key, sep))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = json.dumps(list(value), default=str)
        else:
            flat[full_key] = value
    return flat


# -------------------------------------------------------------------------- dedup

def _dedup_key(record: dict[str, Any]) -> str:
    title = clean_text(record.get("title", "")).lower()
    description = clean_text(record.get("description", "")).lower()
    return _WHITESPACE_RE.sub(" ", f"{title}|{description}").strip()


def dedup_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact duplicates on normalized (lowercased, whitespace-collapsed) title+description."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in records:
        key = _dedup_key(rec)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped


# ------------------------------------------------------------------------- raw -> processed

def _process_one_file(path: Path) -> list[dict[str, Any]]:
    """Load one raw JSONL file and run it through clean -> language-filter -> flatten."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s:%d (%s)", path.name, line_no, exc)
                continue

            title = clean_text(rec.get("title", ""))
            description = clean_text(rec.get("description", rec.get("content", "")))
            if not title:
                continue

            if not is_english(f"{title} {description}"):
                continue

            rec = dict(rec)
            rec["title"] = title
            rec["description"] = description
            records.append(flatten_record(rec))

    return records


def run_full_pipeline(
    force: bool = False,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> dict[str, int]:
    """
    Process every raw JSONL file under data/raw/{record_type}/*.jsonl into a
    cleaned, deduplicated, English-only CSV under
    data/processed/{record_type}/{source}.csv (one output file per raw
    source file, mirroring its stem).

    force=False (default): skip a raw file if its processed CSV already exists.
    force=True: reprocess every raw file regardless.

    Returns {"courses": n_records, "jobs": n_records} — total cleaned records
    written across all sources for each type. A raw/{type}/ directory that
    doesn't exist, or contains no *.jsonl files, contributes 0 (not an error —
    this repo ships with no raw scraped data; see README.md).
    """
    raw_root = raw_dir or _RAW_DIR
    processed_root = processed_dir or _PROCESSED_DIR

    counts: dict[str, int] = {}

    for record_type in ("courses", "jobs"):
        type_raw_dir = raw_root / record_type
        type_processed_dir = processed_root / record_type
        type_processed_dir.mkdir(parents=True, exist_ok=True)

        if not type_raw_dir.exists():
            logger.warning(
                "No raw data directory at %s — 0 %s processed. "
                "Run `python scripts/scrape.py` first to populate it.",
                type_raw_dir, record_type,
            )
            counts[record_type] = 0
            continue

        raw_files = sorted(type_raw_dir.glob("*.jsonl"))
        if not raw_files:
            logger.warning("No *.jsonl files found under %s — 0 %s processed.",
                           type_raw_dir, record_type)
            counts[record_type] = 0
            continue

        total = 0
        for raw_path in raw_files:
            out_path = type_processed_dir / f"{raw_path.stem}.csv"
            if out_path.exists() and not force:
                logger.info("Skipping %s — %s already exists (force=False)", raw_path.name, out_path.name)
                existing = pd.read_csv(out_path)
                total += len(existing)
                continue

            raw_records = _process_one_file(raw_path)
            n_before = sum(1 for _ in raw_path.open(encoding="utf-8")) if raw_path.exists() else 0
            deduped = dedup_records(raw_records)

            if deduped:
                pd.DataFrame(deduped).to_csv(out_path, index=False)
            logger.info(
                "%s: %d raw -> %d cleaned+English -> %d after dedup -> %s",
                raw_path.name, n_before, len(raw_records), len(deduped), out_path.name,
            )
            total += len(deduped)

        counts[record_type] = total

    return counts


# ------------------------------------------------------------------------- processed loader

_COURSE_COLUMN_ALIASES = {"content": "description", "Category": "category", "tags": "skills"}
_JOB_COLUMN_ALIASES = {"requirement": "description", "category": "domain", "tags": "skills"}


def _load_csvs(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(
            f"No processed data directory at {directory}. "
            f"Run `python scripts/scrape.py` and `python scripts/preprocess.py` first."
        )
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {directory}. "
            f"Run `python scripts/scrape.py` and `python scripts/preprocess.py` first."
        )

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["_source_file"] = csv_path.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_processed(kind: str) -> pd.DataFrame:
    """
    Load and normalize every processed CSV under data/processed/{kind}/.

    kind: "courses" or "jobs"

    Normalizes column-name variants seen across sources (e.g. "content" ->
    "description") so downstream code (scripts/build_dataset.py,
    scripts/train.py) can rely on consistent "title"/"description"/"category"
    columns. For "jobs" specifically, both "domain" (the DB-facing name used
    by backend/app/db/init_db.py) and "category" (the alias
    scripts/build_dataset.py filters on) are guaranteed present and kept in
    sync, since different call sites in this repo expect one or the other.
    """
    if kind not in ("courses", "jobs"):
        raise ValueError(f"kind must be 'courses' or 'jobs', got {kind!r}")

    df = _load_csvs(_PROCESSED_DIR / kind)

    aliases = _COURSE_COLUMN_ALIASES if kind == "courses" else _JOB_COLUMN_ALIASES
    for source_col, target_col in aliases.items():
        if source_col in df.columns and target_col not in df.columns:
            df[target_col] = df[source_col]
        elif source_col in df.columns and target_col in df.columns:
            df[target_col] = df[target_col].fillna(df[source_col])

    if "title" not in df.columns:
        raise ValueError(f"Processed {kind} data is missing a required 'title' column")
    if "description" not in df.columns:
        df["description"] = ""

    if kind == "jobs":
        if "domain" not in df.columns and "category" in df.columns:
            df["domain"] = df["category"]
        if "category" not in df.columns and "domain" in df.columns:
            df["category"] = df["domain"]
    elif "category" not in df.columns:
        df["category"] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

    return df
