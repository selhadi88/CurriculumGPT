"""
Build the labeled curriculum-job pair dataset for CurriculumGPT.

Pipeline (rebuilt to run on the real collected corpus — MIT OCW courses +
O*NET/USAJobs/RemoteOK jobs — instead of the hand-authored seed data):
  1. Load Courses.csv + Jobs.csv (columns normalized by load_processed()).
  2. Filter both to the 6-domain evaluation scope via `unified_domain`
     (computer_science, data_science, cloud_devops, cybersecurity, business,
     design — web_development excluded, see README.md).
  3. BGE-similarity retrieval generates candidate pairs: same-domain
     (positive-leaning) and cross-domain (negative-leaning).
  4. Label from a FIXED signal threshold — no calibration to a target rate.
  5. Explicit hard-negative mining: same domain, low skill overlap, forced
     label=0 (see HARD NEGATIVES section for why this overrides the signal).
  6. Attach Bloom histograms + skill multi-hot.
  7. Write data/labeled/pairs.parquet and a genuinely disjoint, stratified
     data/labeled/gold.parquet (not a resample of the same rows).

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --pairs-per-course 10 --hard-neg-per-domain 40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.pipeline import load_processed  # noqa: E402
from ml.components import bloom_depth, skill_utils  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("build_dataset")

_LABELED_DIR = Path(__file__).parent.parent / "data" / "labeled"

# Evaluation scope: 6 of the taxonomy's 7 domains. web_development excluded —
# the real MIT OCW course corpus classifies essentially zero courses into it
# (1 out of 2565, confirmed after fixing the unified classifier's keyword
# false-positives) — see README.md "Evaluation scope" section.
TARGET_DOMAINS = ["computer_science", "data_science", "cloud_devops",
                  "cybersecurity", "business", "design"]

CAT_WEIGHT = 0.6
SKILL_WEIGHT = 0.4

# Fixed threshold — NOT calibrated to any target positive rate. Justified by
# the formula's own arithmetic: same_domain in {0,1} contributes CAT_WEIGHT
# when course/job share a unified_domain, 0 otherwise; skill_cosine in [0,1]
# contributes at most SKILL_WEIGHT. Since SKILL_WEIGHT (0.4) < CAT_WEIGHT
# (0.6), a same-domain pair's signal is ALWAYS in [0.6, 1.0] and a
# different-domain pair's signal is ALWAYS in [0.0, 0.4] — regardless of the
# skill cosine. Any threshold in the open interval (0.4, 0.6) therefore
# produces the identical partition; 0.5 is the midpoint, giving a full 0.1
# margin on both sides against floating-point edge cases. Skill cosine still
# matters for RANKING within the same-domain group (used by hard-negative
# mining below), just not for crossing this threshold.
SIGNAL_THRESHOLD = 0.5


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


def alignment_signal(course_skills, job_skills, course_domain, job_domain) -> tuple[float, float]:
    """Continuous alignment score in [0,1] and the raw skill cosine."""
    sim = _cosine(course_skills, job_skills)
    same_domain = 1.0 if (course_domain == job_domain and course_domain in TARGET_DOMAINS) else 0.0
    score = CAT_WEIGHT * same_domain + SKILL_WEIGHT * sim
    return float(score), float(sim)


def _difficulty(sim: float) -> str:
    if sim > 0.6 or sim < 0.15:
        return "easy"
    if sim < 0.35:
        return "medium"
    return "hard"


def _pair_id(course_text: str, job_text: str) -> str:
    import hashlib
    return hashlib.md5((course_text[:200] + "||" + job_text[:200]).encode("utf-8")).hexdigest()[:16]


def _load_domain_corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Courses/jobs restricted to the 6-domain evaluation scope via
    `unified_domain` (added by ml.components.skill_utils.classify_domain,
    the single classifier used for both sides — see README.md). Rows without
    a unified_domain (the hand-authored seed data, which predates that
    column, and anything the classifier couldn't match) are excluded, so this
    always reflects the real collected corpus only.
    """
    courses = load_processed("courses")
    jobs = load_processed("jobs")
    if "unified_domain" not in courses.columns or "unified_domain" not in jobs.columns:
        raise RuntimeError(
            "unified_domain column missing — run the domain reclassification "
            "pass (ml.components.skill_utils.classify_domain over title+description) "
            "before building pairs."
        )
    courses = courses[courses["unified_domain"].isin(TARGET_DOMAINS)].reset_index(drop=True)
    jobs = jobs[jobs["unified_domain"].isin(TARGET_DOMAINS)].reset_index(drop=True)
    return courses, jobs


def _text_of(row: pd.Series) -> str:
    return f"{str(row['title'])}. {str(row.get('description', ''))}"[:1000]


def build_pairs(
    seed: int = 42,
    pairs_per_course: int = 10,
    hard_neg_per_domain: int = 40,
) -> pd.DataFrame:
    """
    Generate pairs at a scale the real corpus supports (not a fixed n_pairs
    target) — each classified course is paired with up to `pairs_per_course`
    jobs (half same-domain top-BGE-similarity, half cross-domain random),
    plus a fixed hard-negative allotment per domain. Small domains
    (e.g. cloud_devops has only 8 classified courses) simply contribute
    fewer pairs — sizes aren't forced to be equal across domains.
    """
    rng = np.random.default_rng(seed)
    courses, jobs = _load_domain_corpus()
    logger.info("Domain-scoped corpus: %d courses, %d jobs (6 target domains)", len(courses), len(jobs))
    for d in TARGET_DOMAINS:
        logger.info("  %-18s courses=%-5d jobs=%-5d", d,
                    (courses["unified_domain"] == d).sum(), (jobs["unified_domain"] == d).sum())

    course_texts = [_text_of(r) for _, r in courses.iterrows()]
    job_texts = [_text_of(r) for _, r in jobs.iterrows()]
    course_titles = courses["title"].astype(str).tolist()
    course_descs = courses.get("description", pd.Series([""] * len(courses))).astype(str).tolist()
    job_titles = jobs["title"].astype(str).tolist()
    job_descs = jobs.get("description", pd.Series([""] * len(jobs))).astype(str).tolist()
    course_domains = courses["unified_domain"].tolist()
    job_domains = jobs["unified_domain"].tolist()
    course_skv = [skill_utils.detect_skill_vector(t) for t in course_texts]
    job_skv = [skill_utils.detect_skill_vector(t) for t in job_texts]

    sims = _bge_similarity_matrix(course_texts, job_texts)  # (n_courses, n_jobs) or None

    jobs_by_domain: dict[str, list[int]] = {d: [] for d in TARGET_DOMAINS}
    for j, d in enumerate(job_domains):
        jobs_by_domain[d].append(j)

    rows: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()

    # ---- general candidates: same-domain (BGE top-sim) + cross-domain (random) ----
    for ci in range(len(courses)):
        c_domain = course_domains[ci]
        same_js = jobs_by_domain.get(c_domain, [])
        other_js = [j for j in range(len(jobs)) if job_domains[j] != c_domain]
        if not same_js:
            continue

        n_pos = max(1, min(len(same_js), pairs_per_course // 2))
        if sims is not None:
            ranked = sorted(same_js, key=lambda j: -sims[ci, j])
            pos_js = ranked[:n_pos]
        else:
            pos_js = rng.choice(same_js, size=n_pos, replace=False).tolist()

        n_neg = max(1, min(len(other_js), pairs_per_course - n_pos))
        neg_js = rng.choice(other_js, size=n_neg, replace=False).tolist() if other_js else []

        for ji in pos_js + neg_js:
            if (ci, ji) in seen_pairs:
                continue
            seen_pairs.add((ci, ji))
            signal, sim = alignment_signal(course_skv[ci], job_skv[ji], c_domain, job_domains[ji])
            bge_sim = float(sims[ci, ji]) if sims is not None else None
            rows.append({
                "pair_id": _pair_id(course_texts[ci], job_texts[ji]),
                "curriculum_text": course_texts[ci],
                "job_text": job_texts[ji],
                "label": int(signal >= SIGNAL_THRESHOLD),
                "sim_target": round(sim, 4),
                "signal": round(signal, 4),
                "bge_sim": round(bge_sim, 4) if bge_sim is not None else None,
                "skill_multihot": np.nonzero(course_skv[ci] > 0)[0].tolist(),
                "job_skill_mh": np.nonzero(job_skv[ji] > 0)[0].tolist(),
                "bloom_curriculum": [round(float(x), 4) for x in bloom_depth.compute_bloom_histogram(course_texts[ci])],
                "bloom_job": [round(float(x), 4) for x in bloom_depth.compute_bloom_histogram(job_texts[ji])],
                "pub_year": None,
                "course_domain": c_domain,
                "job_domain": job_domains[ji],
                "course_title": course_titles[ci],
                "course_description": course_descs[ci][:500],
                "job_title": job_titles[ji],
                "job_description": job_descs[ji][:500],
                "difficulty": _difficulty(sim),
                "pair_type": "general",
            })

    logger.info("General candidates: %d pairs (label=1: %.1f%%)",
               len(rows), 100 * np.mean([r["label"] for r in rows]) if rows else 0.0)

    # ---- hard negatives: same domain, LOW skill overlap, label forced to 0 ----
    # This deliberately contradicts what alignment_signal()/SIGNAL_THRESHOLD
    # would say — a same-domain pair normally scores >= 0.6 (positive) purely
    # from the domain match, regardless of skill overlap. A "hard negative" is
    # exactly a pair that LOOKS plausible on domain alone but genuinely isn't a
    # good match on actual skill content — the whole point is to stop a model
    # from learning "same domain = aligned" as a shortcut. So these are mined
    # and labeled independently of the general signal formula, not filtered
    # through it.
    n_hard_total = 0
    for domain in TARGET_DOMAINS:
        c_idxs = [i for i, d in enumerate(course_domains) if d == domain]
        j_idxs = jobs_by_domain.get(domain, [])
        if len(c_idxs) < 1 or len(j_idxs) < 1:
            continue
        combos = []
        for ci in c_idxs:
            for ji in j_idxs:
                if (ci, ji) in seen_pairs:
                    continue
                sim = _cosine(course_skv[ci], job_skv[ji])
                combos.append((ci, ji, sim))
        combos.sort(key=lambda x: x[2])  # ascending: lowest skill overlap first
        chosen = combos[:min(hard_neg_per_domain, len(combos))]
        for ci, ji, sim in chosen:
            seen_pairs.add((ci, ji))
            signal, _ = alignment_signal(course_skv[ci], job_skv[ji], domain, domain)  # would be >= 0.6
            bge_sim = float(sims[ci, ji]) if sims is not None else None
            rows.append({
                "pair_id": _pair_id(course_texts[ci], job_texts[ji]),
                "curriculum_text": course_texts[ci],
                "job_text": job_texts[ji],
                "label": 0,  # forced — see comment above
                "sim_target": round(sim, 4),
                "signal": round(signal, 4),
                "bge_sim": round(bge_sim, 4) if bge_sim is not None else None,
                "skill_multihot": np.nonzero(course_skv[ci] > 0)[0].tolist(),
                "job_skill_mh": np.nonzero(job_skv[ji] > 0)[0].tolist(),
                "bloom_curriculum": [round(float(x), 4) for x in bloom_depth.compute_bloom_histogram(course_texts[ci])],
                "bloom_job": [round(float(x), 4) for x in bloom_depth.compute_bloom_histogram(job_texts[ji])],
                "pub_year": None,
                "course_domain": domain,
                "job_domain": domain,
                "course_title": course_titles[ci],
                "course_description": course_descs[ci][:500],
                "job_title": job_titles[ji],
                "job_description": job_descs[ji][:500],
                "difficulty": "hard",
                "pair_type": "hard_negative",
            })
            n_hard_total += 1

    logger.info("Hard negatives: %d pairs added (same-domain, lowest skill overlap, label forced 0)", n_hard_total)

    df = pd.DataFrame(rows)
    df["boosted"] = df["signal"]  # kept for compatibility with llm_label()'s uncertainty selection
    pos_rate = df["label"].mean()
    logger.info("Built %d total pairs — %.1f%% positive (natural, no calibration)", len(df), 100 * pos_rate)
    return df


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

_GRADED_LLM_PROMPT = (
    "You are a curriculum-quality expert. Grade whether this university course "
    "prepares a student for this job, on a 3-point scale:\n"
    '  "aligned"     — the course covers most (roughly 50%+) of the job\'s core skill requirements\n'
    '  "partial"     — the course covers some relevant skills but falls meaningfully short of full preparation\n'
    '  "not_aligned" — the course has little to no bearing on this job\'s requirements\n\n'
    "Course: {course}\n"
    "Job: {job}\n\n"
    'Reply ONLY with valid JSON: {{"verdict": "aligned"|"partial"|"not_aligned", '
    '"confidence": 0.0-1.0, "sim": 0.0-1.0}}  where "sim" is a continuous alignment strength.'
)
_VALID_VERDICTS = {"aligned", "partial", "not_aligned"}


def _signal_fallback_label(signal: float) -> str:
    """
    3-way fallback grading from the heuristic alignment_signal — used ONLY
    when an LLM call fails after all retries; a tiebreak, never a primary
    labeling strategy. Boundaries mirror SIGNAL_THRESHOLD's own reasoning:
    a same-domain pair's signal is always >= CAT_WEIGHT (0.6) and a
    cross-domain pair's is always <= SKILL_WEIGHT (0.4), so "partial" covers
    the (0.4, 0.6) band that's otherwise almost empty.
    """
    if signal >= 0.6:
        return "aligned"
    if signal >= 0.4:
        return "partial"
    return "not_aligned"


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
        # Supports a fallback key (GROQ_API_KEY_2): when the active key's
        # daily token quota (TPD) is exhausted, switch to the next key and
        # keep going rather than falling back to the heuristic signal.
        keys = [k for k in (_read_env_key("GROQ_API_KEY"), _read_env_key("GROQ_API_KEY_2")) if k]
        if not keys:
            logger.warning("GROQ_API_KEY not set — skipping Groq labeling.")
            return None
        try:
            from groq import Groq
        except ImportError:
            logger.warning("groq not installed (`pip install groq`) — skipping.")
            return None
        clients = [Groq(api_key=k) for k in keys]
        # A daily token-quota 429 (e.g. "tokens per day (TPD): Limit 100000,
        # Used 99998") won't clear within any retry budget we'd sanely wait
        # out — confirmed live: retrying it just burns 5x exponential
        # backoff (~2min) per row for zero benefit. Once seen for a given
        # key, mark it exhausted and switch to the next configured key;
        # only fall back to the heuristic signal once every key is spent.
        exhausted = [False] * len(clients)
        active = [0]

        def call(prompt, max_retries=5):
            for attempt in range(max_retries):
                while exhausted[active[0]]:
                    nxt = next((i for i in range(len(clients)) if not exhausted[i]), None)
                    if nxt is None:
                        return None
                    if nxt != active[0]:
                        logger.warning("Switching to Groq key #%d (key #%d's daily quota is exhausted)",
                                       nxt + 1, active[0] + 1)
                    active[0] = nxt
                client = clients[active[0]]
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
                    if "per day" in msg.lower() or "tpd" in msg.lower():
                        logger.warning("Groq key #%d daily token quota exhausted: %s",
                                       active[0] + 1, msg[:200])
                        exhausted[active[0]] = True
                        continue  # re-enters the while-loop above and switches keys
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


def llm_label_graded(
    df: pd.DataFrame,
    n: int | None = None,
    provider: str = "groq",
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """
    Graded silver labeling: aligned / partial / not_aligned, via LLM as the
    PRIMARY source. Leaves the existing heuristic `label` column (0/1, from
    alignment_signal at SIGNAL_THRESHOLD) untouched and adds `llm_label` (+
    `llm_label_source`, `llm_confidence`) — both columns are kept for
    provenance. The heuristic `signal` is used ONLY as a tiebreak fallback
    (via _signal_fallback_label) for rows where the LLM call fails after
    retries; it never overrides or calibrates a successful LLM verdict.

    Resumable at two levels: the on-disk verdict cache (`graded_{provider}_cache.jsonl`,
    append-only, keyed by pair_id) means a restart re-derives already-labeled
    rows without re-calling the API; `checkpoint_path` (written every
    `checkpoint_every` rows) means a crash mid-run still leaves a usable
    partial dataset on disk. checkpoint_path is NEVER data/labeled/pairs.parquet
    by default — callers pick their own output file so this never clobbers
    the heuristic pairs dataset.
    """
    built = _build_caller(provider)
    if built is None:
        raise RuntimeError(
            f"No usable LLM caller for provider={provider!r} — set GROQ_API_KEY "
            "(or GEMINI_API_KEY for provider=gemini) before running graded labeling."
        )
    call_fn, model_name, _default_cache_name = built

    _LABELED_DIR.mkdir(parents=True, exist_ok=True)
    # Separate cache from the binary llm_label() cache — the verdict schema
    # (graded "verdict" string vs. boolean "aligned") is incompatible.
    cache_path = _LABELED_DIR / f"graded_{provider}_cache.jsonl"
    cache = _load_cache(cache_path)

    df = df.copy()
    targets = df if n is None else df.head(n)
    logger.info("Graded LLM-labeling %d pairs with %s/%s (cached: %d)…",
                len(targets), provider, model_name, len(cache))

    df["llm_label"] = pd.Series([None] * len(df), dtype=object)
    df["llm_label_source"] = pd.Series([None] * len(df), dtype=object)
    df["llm_confidence"] = np.nan

    labeled = fallback = 0
    partial_path = checkpoint_path or (_LABELED_DIR / "pairs_llm.parquet")
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for idx, row in targets.iterrows():
            pid = row["pair_id"]
            verdict = cache.get(pid)
            if verdict is None:
                prompt = _GRADED_LLM_PROMPT.format(
                    course=str(row["curriculum_text"])[:600],
                    job=str(row["job_text"])[:600],
                )
                verdict = call_fn(prompt)
                if verdict is not None and verdict.get("verdict") not in _VALID_VERDICTS:
                    verdict = None  # malformed reply — treat like a failed call
                if verdict is None:
                    df.at[idx, "llm_label"] = _signal_fallback_label(float(row["signal"]))
                    df.at[idx, "llm_label_source"] = "heuristic_fallback"
                    fallback += 1
                else:
                    verdict["pair_id"] = pid
                    cache_file.write(json.dumps(verdict) + "\n")
                    cache_file.flush()
            if verdict is not None:
                df.at[idx, "llm_label"] = verdict["verdict"]
                df.at[idx, "llm_label_source"] = "llm"
                if verdict.get("confidence") is not None:
                    df.at[idx, "llm_confidence"] = round(float(verdict["confidence"]), 4)
                if verdict.get("sim") is not None:
                    df.at[idx, "sim_target"] = round(float(verdict["sim"]), 4)
                labeled += 1

            if (labeled + fallback) % checkpoint_every == 0:
                logger.info("  …%d / %d processed (%d LLM, %d fallback)",
                            labeled + fallback, len(targets), labeled, fallback)
                # Incremental save: a crash mid-run still leaves a usable,
                # resumable checkpoint (resume also skips cached pair_ids).
                try:
                    df.to_parquet(partial_path, index=False)
                except Exception:  # noqa: BLE001
                    pass

    logger.info("Graded labeling done — %d via LLM, %d fell back to heuristic signal (LLM call failed/malformed)",
                labeled, fallback)
    return df


def _report_llm_agreement(df: pd.DataFrame) -> None:
    """LLM label distribution, agreement vs. the domain-heuristic `label`, and per-domain breakdown."""
    total = len(df)
    logger.info("=== Graded LLM label report (%d pairs) ===", total)

    dist = df["llm_label"].value_counts()
    logger.info("LLM label distribution:")
    for v in ("aligned", "partial", "not_aligned"):
        c = int(dist.get(v, 0))
        logger.info("  %-12s %5d (%.1f%%)", v, c, 100 * c / total)

    src = df["llm_label_source"].value_counts()
    n_llm = int(src.get("llm", 0))
    n_fb = int(src.get("heuristic_fallback", 0))
    logger.info("Label source: llm=%d (%.1f%%), heuristic_fallback=%d (%.1f%%)",
                n_llm, 100 * n_llm / total, n_fb, 100 * n_fb / total)

    logger.info("Agreement vs. domain-heuristic label (rows=heuristic 0/1, cols=llm_label):")
    crosstab = pd.crosstab(df["label"], df["llm_label"])
    for line in crosstab.to_string().splitlines():
        logger.info("  %s", line)

    agree = ((df["label"] == 1) & (df["llm_label"] == "aligned")) | \
            ((df["label"] == 0) & (df["llm_label"] == "not_aligned"))
    partial_rate = (df["llm_label"] == "partial").mean()
    logger.info("Strict agreement (heuristic 1<->aligned, 0<->not_aligned): %.1f%% "
                "— partial verdicts (%.1f%% of all pairs) counted as neither agree nor disagree",
                100 * agree.mean(), 100 * partial_rate)

    logger.info("Per-domain breakdown:")
    for d in TARGET_DOMAINS:
        sub = df[df["course_domain"] == d]
        if not len(sub):
            continue
        sd = sub["llm_label"].value_counts()
        sub_agree = ((sub["label"] == 1) & (sub["llm_label"] == "aligned")) | \
                    ((sub["label"] == 0) & (sub["llm_label"] == "not_aligned"))
        logger.info("  %-18s n=%-5d aligned=%-4d partial=%-4d not_aligned=%-4d agreement=%.1f%%",
                    d, len(sub), int(sd.get("aligned", 0)), int(sd.get("partial", 0)),
                    int(sd.get("not_aligned", 0)), 100 * sub_agree.mean())


def _report_distribution(df: pd.DataFrame) -> None:
    """Natural (uncalibrated) class distribution + a histogram summary of the signal."""
    pos_rate = df["label"].mean()
    logger.info("Natural class distribution: %d positive / %d negative (%.1f%% positive)",
                int(df["label"].sum()), int((df["label"] == 0).sum()), 100 * pos_rate)
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(df["signal"].to_numpy(), bins=edges)
    logger.info("Signal histogram (bin width 0.1):")
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        bar = "#" * max(1, int(60 * c / max(counts.max(), 1))) if c else ""
        logger.info("  [%.1f, %.1f): %5d %s", lo, hi, int(c), bar)
    logger.info("Per-domain pair counts (course_domain):")
    for d in TARGET_DOMAINS:
        sub = df[df["course_domain"] == d]
        if len(sub):
            logger.info("  %-18s total=%-5d positive=%-5d (%.1f%%) hard_neg=%d",
                        d, len(sub), int(sub["label"].sum()), 100 * sub["label"].mean(),
                        int((sub["pair_type"] == "hard_negative").sum()))


def _stratified_gold_split(df: pd.DataFrame, gold_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genuinely disjoint holdout: stratify by (course_domain, signal tercile) so
    gold mirrors the domain/difficulty mix of the full pairs pool, then REMOVE
    the chosen rows from the training set entirely — unlike df.sample(), which
    leaves the same rows in both files.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    q1, q2 = df["signal"].quantile([1 / 3, 2 / 3]).to_numpy()

    def tercile(s: float) -> str:
        if s <= q1:
            return "low"
        if s <= q2:
            return "mid"
        return "high"

    strata_key = df["course_domain"].astype(str) + "|" + df["signal"].apply(tercile)
    strata: dict[str, list[int]] = {}
    for i, key in enumerate(strata_key):
        strata.setdefault(key, []).append(i)

    target_per_stratum = max(1, gold_size // max(1, len(strata)))
    chosen: list[int] = []
    for idxs in strata.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        chosen.extend(idxs[:min(target_per_stratum, len(idxs))])

    if len(chosen) > gold_size:
        rng.shuffle(chosen)
        chosen = chosen[:gold_size]
    elif len(chosen) < gold_size:
        remaining = [i for i in range(len(df)) if i not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(remaining[:gold_size - len(chosen)])

    chosen_set = set(chosen)
    gold = df.iloc[sorted(chosen_set)].reset_index(drop=True)
    train = df.iloc[[i for i in range(len(df)) if i not in chosen_set]].reset_index(drop=True)
    return train, gold


def _export_review_sample(df: pd.DataFrame, path: Path, seed: int, n_pos: int = 10, n_hard_neg: int = 5) -> None:
    """15 example pairs for human review — spread across domains, one CSV
    with course/job title+description, domain, signal score, and label."""
    rng = np.random.default_rng(seed)

    def spread_sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if pool.empty:
            return pool
        domains = list(pool["course_domain"].unique())
        rng.shuffle(domains)
        buckets = {d: pool[pool["course_domain"] == d].sample(frac=1, random_state=seed) for d in domains}
        chosen_rows = []
        while len(chosen_rows) < n and any(len(b) for b in buckets.values()):
            for d in domains:
                if len(buckets[d]) == 0:
                    continue
                chosen_rows.append(buckets[d].iloc[0])
                buckets[d] = buckets[d].iloc[1:]
                if len(chosen_rows) >= n:
                    break
        return pd.DataFrame(chosen_rows)

    positives = df[(df["label"] == 1) & (df["pair_type"] == "general")]
    hard_negs = df[df["pair_type"] == "hard_negative"]
    sample = pd.concat([spread_sample(positives, n_pos), spread_sample(hard_negs, n_hard_neg)], ignore_index=True)

    cols = ["course_title", "course_description", "job_title", "job_description",
            "course_domain", "signal", "label", "pair_type"]
    sample[cols].rename(columns={"course_domain": "domain"}).to_csv(path, index=False)
    logger.info("Wrote %s (%d rows: %d positive, %d hard_negative)",
                path, len(sample), int((sample["label"] == 1).sum()), int((sample["pair_type"] == "hard_negative").sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-course", type=int, default=10,
                    help="Max candidate pairs generated per classified course (half same-domain, half cross-domain)")
    ap.add_argument("--hard-neg-per-domain", type=int, default=40,
                    help="Hard-negative pairs mined per domain (same domain, lowest skill overlap)")
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
    ap.add_argument("--relabel-graded", action="store_true",
                     help="Skip rebuilding pairs; load the existing data/labeled/pairs.parquet, "
                          "run graded LLM labeling (aligned/partial/not_aligned) over every row "
                          "via llm_label_graded(), write the result to --out (default "
                          "data/labeled/pairs_llm.parquet, NOT pairs.parquet), and print the "
                          "distribution/agreement/per-domain report.")
    ap.add_argument("--input", type=str, default=None,
                     help="Input parquet for --relabel-graded (default: data/labeled/pairs.parquet). "
                          "E.g. data/labeled/gold.parquet to grade the gold holdout instead.")
    ap.add_argument("--out", type=str, default=None,
                     help="Output path for --relabel-graded (default: data/labeled/pairs_llm.parquet). "
                          "Also used as the resumable checkpoint path, saved every --checkpoint-every rows.")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                     help="Rows between checkpoint saves during --relabel-graded (default: 50)")
    args = ap.parse_args()

    _LABELED_DIR.mkdir(parents=True, exist_ok=True)

    if args.relabel_graded:
        pairs_path = Path(args.input) if args.input else (_LABELED_DIR / "pairs.parquet")
        out_path = Path(args.out) if args.out else (_LABELED_DIR / "pairs_llm.parquet")
        df = pd.read_parquet(pairs_path)
        logger.info("Loaded %d existing pairs from %s for graded relabeling (input untouched)", len(df), pairs_path)
        df = llm_label_graded(df, n=args.llm_n, provider=args.provider,
                               checkpoint_path=out_path, checkpoint_every=args.checkpoint_every)
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows, columns include both `label` [heuristic] and `llm_label` [graded])",
                    out_path, len(df))
        _report_llm_agreement(df)
        return

    df = build_pairs(seed=args.seed, pairs_per_course=args.pairs_per_course,
                      hard_neg_per_domain=args.hard_neg_per_domain)

    used_llm = False
    if args.gemini:
        df = llm_label(df, args.llm_n, provider="gemini"); used_llm = True
    elif args.llm:
        df = llm_label(df, args.llm_n, provider=args.provider); used_llm = True

    # When LLM-labeling, keep ONLY the real LLM-labeled rows (drop heuristic ones).
    if used_llm and "llm_labeled" in df.columns:
        df = df[df["llm_labeled"]].reset_index(drop=True)

    # pair_id is a content hash of the first 200 chars of each side (see
    # _pair_id) so the LLM-verdict cache matches across rebuilds. A handful of
    # MIT OCW rows share identical title+description text (reruns/duplicate
    # listings), which produces exact pair_id collisions when paired with the
    # same job — dedupe before splitting so train/gold stay disjoint by
    # pair_id, not just by row position.
    before = len(df)
    df = df.drop_duplicates(subset="pair_id", keep="first").reset_index(drop=True)
    if before != len(df):
        logger.info("Dropped %d duplicate-content pairs (pair_id collisions)", before - len(df))

    _report_distribution(df)

    train, gold = _stratified_gold_split(df, args.gold_size, args.seed)
    overlap = set(train["pair_id"]) & set(gold["pair_id"])
    if overlap:
        raise RuntimeError(f"gold/train overlap detected ({len(overlap)} pair_ids) — split is not disjoint")
    logger.info("Gold split: %d held out (disjoint from %d training pairs, verified 0 overlap)", len(gold), len(train))

    pairs_path = _LABELED_DIR / "pairs.parquet"
    train.to_parquet(pairs_path, index=False)
    logger.info("Wrote %s (%d rows, %.1f%% aligned)", pairs_path, len(train), 100 * train["label"].mean())

    gold_path = _LABELED_DIR / "gold.parquet"
    gold.to_parquet(gold_path, index=False)
    logger.info("Wrote %s (%d rows, %.1f%% aligned)", gold_path, len(gold), 100 * gold["label"].mean())

    review_path = _LABELED_DIR / "review_sample.csv"
    _export_review_sample(df, review_path, args.seed)


if __name__ == "__main__":
    main()
