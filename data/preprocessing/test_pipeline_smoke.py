"""
Self-contained smoke test for data/preprocessing/pipeline.py.

Exercises all four processing steps (text cleaning, language filter,
metadata flatten, dedup) against a synthetic raw fixture written to a temp
directory — never touches the real data/raw or data/processed trees, so it's
safe to run against a live checkout without polluting seeded demo data.

Usage:
    python data/preprocessing/test_pipeline_smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.preprocessing.pipeline import (  # noqa: E402
    clean_text,
    dedup_records,
    flatten_record,
    is_english,
    load_processed,
    run_full_pipeline,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def test_clean_text() -> None:
    print("\n1. text cleaning")
    dirty = "  <p>Learn <b>Python</b>\tand\nSQL.</p>  \x07  café  "
    cleaned = clean_text(dirty)
    check("strips HTML tags", "<p>" not in cleaned and "<b>" not in cleaned, cleaned)
    check("collapses whitespace/newlines/tabs", "  " not in cleaned and "\n" not in cleaned, cleaned)
    check("drops control characters", "\x07" not in cleaned, repr(cleaned))
    check("keeps real unicode content (NFKC-normalized)", "café" in cleaned, cleaned)
    check("trims outer whitespace", cleaned == cleaned.strip(), repr(cleaned))


def test_language_filter() -> None:
    print("\n2. language filter")
    check("accepts clear English", is_english("This course teaches machine learning and data analysis."))
    check("rejects clear French", not is_english("Ce cours enseigne l'apprentissage automatique et l'analyse de donnees."))
    check("rejects empty text", not is_english(""))


def test_flatten() -> None:
    print("\n3. metadata flatten")
    nested = {
        "title": "Intro to ML",
        "description": "desc",
        "instructor": {"name": "Dr. Smith", "rating": 4.8},
        "tags": ["ml", "python", "beginner"],
        "duration_weeks": 6,
    }
    flat = flatten_record(nested)
    check("flattens nested dict to dot-notation key", "instructor.name" in flat and flat["instructor.name"] == "Dr. Smith")
    check("flattens second nested key too", "instructor.rating" in flat and flat["instructor.rating"] == 4.8)
    check("no leftover nested dict value", not any(isinstance(v, dict) for v in flat.values()))
    check("JSON-stringifies list values", isinstance(flat["tags"], str) and json.loads(flat["tags"]) == ["ml", "python", "beginner"])
    check("scalar values pass through unchanged", flat["duration_weeks"] == 6)


def test_dedup() -> None:
    print("\n4. deduplication")
    records = [
        {"title": "Intro to Python", "description": "Learn Python basics."},
        {"title": "  Intro to Python  ", "description": "Learn Python basics."},  # dup (whitespace)
        {"title": "INTRO TO PYTHON", "description": "LEARN PYTHON BASICS."},       # dup (case)
        {"title": "Intro to SQL", "description": "Learn SQL basics."},             # distinct
    ]
    deduped = dedup_records(records)
    check("drops exact/whitespace/case duplicates", len(deduped) == 2, f"got {len(deduped)}")
    check("keeps distinct records", any(r["title"].strip() == "Intro to SQL" for r in deduped))


def test_run_full_pipeline_end_to_end() -> None:
    print("\n5. run_full_pipeline (end-to-end, temp dirs — real data/ untouched)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        courses_raw = raw_dir / "courses"
        courses_raw.mkdir(parents=True)

        records = [
            # clean, unique, English -> kept
            {"title": "Intro to Machine Learning", "description": "<p>Covers <b>supervised</b> learning.</p>",
             "source": "test", "instructor": {"name": "A. Lee"}, "topics": ["ml", "supervised"]},
            # exact duplicate (case/whitespace) -> dropped
            {"title": "  INTRO TO MACHINE LEARNING  ", "description": "Covers supervised learning.",
             "source": "test"},
            # non-English -> dropped
            {"title": "Introduction a l'apprentissage automatique",
             "description": "Ce cours couvre l'apprentissage supervise et non supervise en detail.",
             "source": "test"},
            # distinct, kept
            {"title": "Intro to Databases", "description": "Covers relational database design.",
             "source": "test"},
        ]
        with (courses_raw / "fixture.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # jobs/ deliberately left absent to also verify the "no raw dir" path.
        counts = run_full_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        check("courses: 4 raw -> 2 after language-filter+dedup", counts.get("courses") == 2,
              f"counts={counts}")
        check("jobs: missing raw dir handled gracefully (0, no crash)", counts.get("jobs") == 0,
              f"counts={counts}")

        out_csv = processed_dir / "courses" / "fixture.csv"
        check("processed CSV was written", out_csv.exists())

        import pandas as pd
        df = pd.read_csv(out_csv)
        check("HTML stripped from persisted description",
              not df["description"].str.contains("<p>|<b>", regex=True).any())
        check("nested instructor dict flattened to a column",
              "instructor.name" in df.columns)
        check("list field JSON-stringified in persisted CSV",
              "topics" in df.columns and isinstance(df.loc[df["title"].str.contains("Machine"), "topics"].iloc[0], str))

        # load_processed() against these same temp dirs — patch the module
        # constant rather than duplicating its loader logic here.
        import data.preprocessing.pipeline as pipeline_mod
        original_processed_dir = pipeline_mod._PROCESSED_DIR
        pipeline_mod._PROCESSED_DIR = processed_dir
        try:
            loaded = load_processed("courses")
            check("load_processed() reads it back with title+description+category",
                  {"title", "description", "category"}.issubset(loaded.columns))
            check("load_processed() row count matches pipeline output",
                  len(loaded) == counts["courses"], f"{len(loaded)} vs {counts['courses']}")
        finally:
            pipeline_mod._PROCESSED_DIR = original_processed_dir


def test_load_processed_against_real_seed_data() -> None:
    print("\n6. load_processed() against the real shipped seed data")
    try:
        courses = load_processed("courses")
        jobs = load_processed("jobs")
        check("real seed courses load with title/description/category",
              {"title", "description", "category"}.issubset(courses.columns) and len(courses) > 0,
              f"{len(courses)} rows")
        check("real seed jobs load with both domain and category columns",
              {"title", "description", "domain", "category"}.issubset(jobs.columns) and len(jobs) > 0,
              f"{len(jobs)} rows")
        check("jobs domain/category kept in sync",
              (jobs["domain"] == jobs["category"]).all())
    except FileNotFoundError as exc:
        check("real seed data present under data/processed/", False, str(exc))


if __name__ == "__main__":
    test_clean_text()
    test_language_filter()
    test_flatten()
    test_dedup()
    test_run_full_pipeline_end_to_end()
    test_load_processed_against_real_seed_data()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")
