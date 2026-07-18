"""
Build the labeled curriculum-job pair dataset for CurriculumGPT.

Pipeline:
  1. Load Courses.csv + Jobs.csv (columns normalized by load_processed()).
  2. Filter jobs to discipline-relevant categories.
  3. Sample candidate pairs (mix of same-discipline and cross-discipline).
  4. Silver-label each pair from skill-vector cosine + category-alignment boost.
  5. (Optional) Validate the most uncertain pairs with Google Gemini Flash (free).
  6. Attach Bloom histograms + skill multi-hot.
  7. Write data/labeled/pairs.parquet and a 300-row gold subset.

Usage:
    python scripts/build_dataset.py --n-pairs 4000
    python scripts/build_dataset.py --n-pairs 4000 --gemini-validate 500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.pipeline import load_processed  # noqa: E402
from ml.components import bloom_depth, skill_utils  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("build_dataset")

_LABELED_DIR = Path(__file__).parent.parent / "data" / "labeled"

# Course category -> job categories that count as aligned.
CATEGORY_MAP = {
    "Computer Science": ["Information Technology", "Engineering"],
    "Data Analytics": ["Information Technology", "Finance"],
    "Business": ["Finance", "Marketing", "Operations", "Human Resources"],
    "Finance": ["Finance"],
    "Engineering": ["Engineering"],
    "Marketing": ["Marketing"],
    "Sales": ["Marketing"],
    "HR": ["Human Resources"],
    "Operations": ["Operations"],
}
RELEVANT_JOB_CATS = ["Information Technology", "Engineering", "Finance",
                     "Marketing", "Human Resources", "Operations"]

# Labeling combines two signals into a continuous alignment score, then a
# calibrated percentile threshold targets the paper's ~58% aligned split:
#   - category alignment (reliable): is the job in this course's aligned set?
#   - skill-vector cosine (sparse on short text): refines within a category.
CAT_WEIGHT = 0.6
SKILL_WEIGHT = 0.4
TARGET_POSITIVE_RATE = 0.58


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _bge_similarity_matrix(course_texts: list[str], job_texts: list[str]):
    """
    (n_courses, n_jobs) cosine matrix from BGE backbone embeddings, or None if
    BGE can't load. Runs on GPU when available — fast for a few-thousand pool.
    """
    try:
        import torch
        from ml.models.bi_encoder import BGEBiEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Embedding %d courses + %d jobs with BGE on %s for retrieval…",
                    len(course_texts), len(job_texts), device)
        model = BGEBiEncoder()
        model.eval()
        ce = model.encode_backbone_texts(course_texts, batch_size=64, device=device)
        je = model.encode_backbone_texts(job_texts, batch_size=64, device=device)
        return (ce @ je.T).cpu().numpy()
    except Exception as exc:  # noqa: BLE001
        logger.warning("BGE retrieval unavailable (%s) — falling back to random pairing.", exc)
        return None


def alignment_signal(course_skills, job_skills, course_cat, job_cat) -> tuple[float, float]:
    """Continuous alignment score in [0,1] and the raw skill cosine."""
    sim = _cosine(course_skills, job_skills)
    cat_match = 1.0 if job_cat in CATEGORY_MAP.get(course_cat, []) else 0.0
    score = CAT_WEIGHT * cat_match + SKILL_WEIGHT * sim
    return float(score), float(sim)


def _difficulty(sim: float) -> str:
    if sim > 0.6 or sim < 0.15:
        return "easy"
    if sim < 0.35:
        return "medium"
    return "hard"


def build_pairs(n_pairs: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    courses = load_processed("courses")
    jobs = load_processed("jobs")
    jobs = jobs[jobs["category"].isin(RELEVANT_JOB_CATS)].reset_index(drop=True)
    logger.info("Loaded %d courses, %d relevant jobs", len(courses), len(jobs))

    # Working pool — cap for speed.
    pool_courses = courses.sample(min(len(courses), 2500), random_state=seed).reset_index(drop=True)
    pool_jobs = jobs.sample(min(len(jobs), 2500), random_state=seed).reset_index(drop=True)

    def text_of(row, body):
        return f"{str(row['title'])}. {str(row.get(body, ''))}"[:1000]

    course_texts = [text_of(r, "description") for _, r in pool_courses.iterrows()]
    job_texts = [text_of(r, "description") for _, r in pool_jobs.iterrows()]
    course_skv = [skill_utils.detect_skill_vector(t) for t in course_texts]
    job_skv = [skill_utils.detect_skill_vector(t) for t in job_texts]

    # Semantic retrieval (BGE) so candidate pairs are a genuine mix of aligned
    # (top-similar jobs) and misaligned (random jobs), rather than random pairs
    # that are almost never truly aligned. This is what makes LLM labels balanced.
    sims = _bge_similarity_matrix(course_texts, job_texts)  # (n_courses, n_jobs) or None

    rows: list[dict] = []
    n_courses_used = min(len(pool_courses), max(1, n_pairs // 4))
    course_order = rng.permutation(len(pool_courses))[:n_courses_used]
    pairs_per_course = max(1, n_pairs // n_courses_used)

    for ci in course_order:
        ci = int(ci)
        c_cat = str(pool_courses.iloc[ci]["category"])
        if sims is not None:
            ranked = np.argsort(-sims[ci])           # most→least similar jobs
            n_pos = max(1, pairs_per_course // 2)
            pos_js = ranked[:n_pos].tolist()         # likely-aligned candidates
            neg_js = rng.choice(ranked[len(ranked) // 2:], size=n_pos, replace=False).tolist()
            chosen_js = pos_js + neg_js
        else:
            chosen_js = rng.integers(0, len(pool_jobs), size=pairs_per_course).tolist()

        for ji in chosen_js:
            ji = int(ji)
            signal, sim = alignment_signal(
                course_skv[ci], job_skv[ji], c_cat, str(pool_jobs.iloc[ji]["category"])
            )
            bge_sim = float(sims[ci, ji]) if sims is not None else None
            # Deterministic content-based id so the LLM-verdict cache matches
            # across rebuilds (UUIDs would never re-match).
            import hashlib
            pid = hashlib.md5(
                (course_texts[ci][:200] + "||" + job_texts[ji][:200]).encode("utf-8")
            ).hexdigest()[:16]
            rows.append({
                "pair_id": pid,
                "curriculum_text": course_texts[ci],
                "job_text": job_texts[ji],
                "label": -1,  # filled after calibration / Gemini
                "sim_target": round(sim, 4),
                "signal": round(signal, 4),
                "bge_sim": round(bge_sim, 4) if bge_sim is not None else None,
                "skill_multihot": np.nonzero(course_skv[ci] > 0)[0].tolist(),
                "job_skill_mh": np.nonzero(job_skv[ji] > 0)[0].tolist(),
                "bloom_curriculum": [round(float(x), 4) for x in bloom_depth.compute_bloom_histogram(course_texts[ci])],
                "bloom_job": [round(float(x), 4) for x in bloom_depth.compute_bloom_histogram(job_texts[ji])],
                "pub_year": None,
                "course_category": c_cat,
                "job_category": str(pool_jobs.iloc[ji]["category"]),
                "difficulty": _difficulty(sim),
            })
            if len(rows) >= n_pairs:
                break
        if len(rows) >= n_pairs:
            break

    df = pd.DataFrame(rows)
    df["label"] = _calibrate_labels(df["signal"].to_numpy(), TARGET_POSITIVE_RATE, seed)
    # `boosted` kept as an alias of signal for the Gemini-uncertainty selection.
    df["boosted"] = df["signal"]
    pos = df["label"].mean()
    logger.info("Built %d pairs — %.1f%% aligned (target %.0f%%)",
                len(df), 100 * pos, 100 * TARGET_POSITIVE_RATE)
    return df


def _calibrate_labels(signal: np.ndarray, target_rate: float, seed: int) -> np.ndarray:
    """
    Assign binary labels so ~target_rate are positive, robust to tied signals.
    Pairs strictly above the cutoff are positive; pairs strictly below are
    negative; the tied mass straddling the cutoff is split randomly to hit the
    exact target (these are the genuinely ambiguous category-match-only pairs).
    """
    rng = np.random.default_rng(seed)
    n = len(signal)
    n_pos_target = int(round(target_rate * n))
    cutoff = float(np.quantile(signal, 1.0 - target_rate))

    labels = np.where(signal > cutoff, 1, 0)
    tied = np.where(np.isclose(signal, cutoff))[0]
    n_pos_now = int(labels.sum())
    need = n_pos_target - n_pos_now
    if need > 0 and len(tied) > 0:
        chosen = rng.choice(tied, size=min(need, len(tied)), replace=False)
        labels[chosen] = 1
    return labels.astype(int)


import os as _os

# LLM labeling supports two free providers. Groq is preferred (higher free-tier
# limits → ~1000 labels/day) with Gemini as a fallback.
_GEMINI_MODEL = _os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_GROQ_MODEL = _os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_LLM_PROMPT = (
    "You are a curriculum-quality expert. Decide whether this university course "
    "meaningfully prepares a student for this job — i.e. it covers at least 50% of "
    "the job's core skill requirements.\n\n"
    "Course: {course}\n"
    "Job: {job}\n\n"
    'Reply ONLY with valid JSON: {{"aligned": true/false, "confidence": 0.0-1.0, '
    '"sim": 0.0-1.0}}  where "sim" is a continuous alignment strength.'
)
_GEMINI_PROMPT = _LLM_PROMPT  # backward-compat alias


def _read_env_key(name: str) -> str:
    """Read an API key from the environment, falling back to .env on disk."""
    import os
    key = os.getenv(name, "")
    if not key:
        env = Path(__file__).parent.parent / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith(name + "="):
                    key = line.split("=", 1)[1].strip()
    return key


def _load_cache(cache_path: Path) -> dict:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                cache[rec["pair_id"]] = rec
            except Exception:  # noqa: BLE001
                continue
    return cache


# Default 2s pacing matches Groq's ~30 req/min free tier (avoids 429 churn).
# Set LLM_MIN_INTERVAL_S=0 to go as fast as the provider allows.
_MIN_INTERVAL_S = float(_os.getenv("LLM_MIN_INTERVAL_S", "2.0"))
_last_call_t = [0.0]


def _build_caller(provider: str):
    """
    Returns (call_fn, model_name, cache_filename) for the chosen provider.
    call_fn(prompt) -> dict|None handles its own retry/backoff.
    """
    import time

    def _pace():
        if _MIN_INTERVAL_S > 0:
            elapsed = time.time() - _last_call_t[0]
            if elapsed < _MIN_INTERVAL_S:
                time.sleep(_MIN_INTERVAL_S - elapsed)
        _last_call_t[0] = time.time()

    if provider == "groq":
        key = _read_env_key("GROQ_API_KEY")
        if not key:
            logger.warning("GROQ_API_KEY not set — skipping Groq labeling.")
            return None
        try:
            from groq import Groq
        except ImportError:
            logger.warning("groq not installed (`pip install groq`) — skipping.")
            return None
        client = Groq(api_key=key)

        def call(prompt, max_retries=5):
            for attempt in range(max_retries):
                _pace()
                try:
                    resp = client.chat.completions.create(
                        model=_GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                    )
                    return json.loads(resp.choices[0].message.content)
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    if "429" in msg or "rate" in msg.lower():
                        backoff = min(60, 4 * (2 ** attempt))
                        logger.warning("Groq rate limited — backing off %ds (%d/%d)",
                                       backoff, attempt + 1, max_retries)
                        time.sleep(backoff); continue
                    logger.debug("Groq call failed: %s", msg[:120])
                    return None
            return None

        return call, _GROQ_MODEL, "groq_cache.jsonl"

    # default: gemini
    key = _read_env_key("GEMINI_API_KEY")
    if not key:
        logger.warning("GEMINI_API_KEY not set — skipping Gemini labeling.")
        return None
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not installed — skipping.")
        return None
    client = genai.Client(api_key=key)

    def call(prompt, max_retries=5):
        for attempt in range(max_retries):
            _pace()
            try:
                resp = client.models.generate_content(
                    model=_GEMINI_MODEL, contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                return json.loads(resp.text)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    backoff = min(60, 8 * (2 ** attempt))
                    logger.warning("Gemini rate limited — backing off %ds (%d/%d)",
                                   backoff, attempt + 1, max_retries)
                    time.sleep(backoff); continue
                logger.debug("Gemini call failed: %s", msg[:120])
                return None
        return None

    return call, _GEMINI_MODEL, "gemini_cache.jsonl"


def llm_label(df: pd.DataFrame, n: int | None = None, provider: str = "groq") -> pd.DataFrame:
    """
    Label pairs with an LLM as the PRIMARY ground truth (free tier).
    Overwrites the weak silver `label` with the LLM judgment and stores the
    continuous `sim` in `sim_target`. Resumable via an on-disk cache.

    provider: "groq" (preferred, higher free limits) or "gemini".
    n=None labels every pair; otherwise the first n (rest keep silver labels).
    """
    built = _build_caller(provider)
    if built is None:
        return df
    call_fn, model_name, cache_name = built

    _LABELED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _LABELED_DIR / cache_name
    cache = _load_cache(cache_path)

    df = df.copy()
    targets = df if n is None else df.head(n)
    logger.info("LLM-labeling %d pairs with %s/%s (cached: %d)…",
                len(targets), provider, model_name, len(cache))

    df["llm_labeled"] = False
    labeled = failed = 0
    partial_path = _LABELED_DIR / "pairs.parquet"
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for idx, row in targets.iterrows():
            pid = row["pair_id"]
            verdict = cache.get(pid)
            if verdict is None:
                prompt = _LLM_PROMPT.format(
                    course=str(row["curriculum_text"])[:600],
                    job=str(row["job_text"])[:600],
                )
                verdict = call_fn(prompt)
                if verdict is None:
                    failed += 1
                    continue
                verdict["pair_id"] = pid
                cache_file.write(json.dumps(verdict) + "\n")
                cache_file.flush()
            df.at[idx, "label"] = 1 if verdict.get("aligned") else 0
            df.at[idx, "llm_labeled"] = True
            if verdict.get("sim") is not None:
                df.at[idx, "sim_target"] = round(float(verdict["sim"]), 4)
            labeled += 1
            if labeled % 50 == 0:
                logger.info("  …%d / %d labeled (%d failed)", labeled, len(targets), failed)
                # Incremental save: a stall always leaves a usable dataset.
                try:
                    df[df["llm_labeled"]].to_parquet(partial_path, index=False)
                except Exception:  # noqa: BLE001
                    pass

    df["label"] = df["label"].astype(int)
    n_llm = int(df["llm_labeled"].sum())
    logger.info("LLM labeling done — %d labeled (%d via LLM), %.1f%% aligned",
                labeled, n_llm, 100 * df[df["llm_labeled"]]["label"].mean() if n_llm else 0)
    return df


# Backward-compat alias.
def gemini_label(df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    return llm_label(df, n, provider="gemini")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--llm", action="store_true",
                    help="Use an LLM as the PRIMARY labeler (real ground truth, free)")
    ap.add_argument("--provider", choices=["groq", "gemini"], default="groq",
                    help="LLM provider for labeling (default: groq — higher free limits)")
    ap.add_argument("--gemini", action="store_true",
                    help="Deprecated alias for --llm --provider gemini")
    ap.add_argument("--llm-n", type=int, default=None,
                    help="Limit LLM labeling to the first N pairs (default: all)")
    ap.add_argument("--gold-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _LABELED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_pairs(args.n_pairs, args.seed)

    used_llm = False
    if args.gemini:
        df = llm_label(df, args.llm_n, provider="gemini"); used_llm = True
    elif args.llm:
        df = llm_label(df, args.llm_n, provider=args.provider); used_llm = True

    # When LLM-labeling, keep ONLY the real LLM-labeled rows (drop heuristic ones).
    if used_llm and "llm_labeled" in df.columns:
        df = df[df["llm_labeled"]].reset_index(drop=True)

    pairs_path = _LABELED_DIR / "pairs.parquet"
    df.to_parquet(pairs_path, index=False)
    logger.info("Wrote %s (%d rows, %.0f%% aligned)",
                pairs_path, len(df), 100 * df["label"].mean())

    gold = df.sample(min(args.gold_size, len(df)), random_state=args.seed)
    gold_path = _LABELED_DIR / "gold.parquet"
    gold.to_parquet(gold_path, index=False)
    logger.info("Wrote %s (%d rows)", gold_path, len(gold))


if __name__ == "__main__":
    main()
