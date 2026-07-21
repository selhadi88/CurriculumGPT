"""
Recall check on the unified domain classifier (ml.components.skill_utils.classify_domain).

Two parts:
  1. Manual review sample: 30 random unclassified courses + 30 random
     unclassified jobs (title + first 200 chars of description), written to
     CSV for human inspection.
  2. Automated centroid probe: embed every classified AND unclassified
     course/job with the BGE backbone (cached on disk, keyed by content hash,
     so reruns don't re-embed). For each taxonomy domain, build a centroid
     from that domain's classified records (courses + jobs pooled) and take
     the median cosine similarity of those same records to their own
     centroid as a per-domain threshold. Any unclassified record whose
     cosine similarity to a domain centroid EXCEEDS that domain's own median
     is a candidate false negative — a record the keyword classifier missed
     but that reads as semantically closer to the domain than a typical
     genuine member of it.

Usage:
    python scripts/recall_check.py
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.pipeline import load_processed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("recall_check")

_LABELED_DIR = Path(__file__).parent.parent / "data" / "labeled"
_CACHE_PATH = _LABELED_DIR / "embedding_cache.joblib"
_SEED = 42


def _text_of(row: pd.Series) -> str:
    return f"{str(row['title'])}. {str(row.get('description', ''))}"[:1000]


def _rec_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    import joblib
    if _CACHE_PATH.exists():
        return joblib.load(_CACHE_PATH)
    return {}


def _save_cache(cache: dict) -> None:
    import joblib
    _LABELED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, _CACHE_PATH)


def _embed_missing(missing_texts: list[str], cache: dict, chunk_size: int = 300) -> None:
    """Embed any texts not already in the on-disk cache, saving after each chunk."""
    if not missing_texts:
        logger.info("All texts already cached — no embedding needed.")
        return
    import torch
    from ml.models.bi_encoder import BGEBiEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading BGE backbone on %s to embed %d new texts…", device, len(missing_texts))
    model = BGEBiEncoder()
    model.eval()
    for i in range(0, len(missing_texts), chunk_size):
        chunk = missing_texts[i:i + chunk_size]
        emb = model.encode_backbone_texts(chunk, batch_size=64, device=device).cpu().numpy()
        for t, e in zip(chunk, emb):
            cache[_rec_hash(t)] = e
        _save_cache(cache)
        logger.info("  embedded %d / %d (cache saved)", min(i + chunk_size, len(missing_texts)), len(missing_texts))


def _emb_matrix(texts: list[str], cache: dict) -> np.ndarray:
    return np.stack([cache[_rec_hash(t)] for t in texts]) if texts else np.zeros((0, 1024))


def _normalize(m: np.ndarray) -> np.ndarray:
    if len(m) == 0:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def main() -> None:
    courses = load_processed("courses")
    jobs = load_processed("jobs")

    c_classified = courses[courses["unified_domain"].notna()].reset_index(drop=True)
    c_unclassified = courses[courses["unified_domain"].isna()].reset_index(drop=True)
    j_classified = jobs[jobs["unified_domain"].notna()].reset_index(drop=True)
    j_unclassified = jobs[jobs["unified_domain"].isna()].reset_index(drop=True)

    logger.info("Courses: %d classified, %d unclassified", len(c_classified), len(c_unclassified))
    logger.info("Jobs: %d classified, %d unclassified", len(j_classified), len(j_unclassified))

    # ---- Part 1: manual review sample (30 unclassified courses + 30 unclassified jobs) ----
    c_sample = c_unclassified.sample(min(30, len(c_unclassified)), random_state=_SEED)
    j_sample = j_unclassified.sample(min(30, len(j_unclassified)), random_state=_SEED)
    review_rows = (
        [{"type": "course", "title": r["title"], "excerpt": str(r.get("description", ""))[:200]}
         for _, r in c_sample.iterrows()]
        + [{"type": "job", "title": r["title"], "excerpt": str(r.get("description", ""))[:200]}
           for _, r in j_sample.iterrows()]
    )
    review_df = pd.DataFrame(review_rows)
    review_path = _LABELED_DIR / "unclassified_review_sample.csv"
    _LABELED_DIR.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(review_path, index=False)
    logger.info("Wrote %s (%d rows: 30 courses + 30 jobs, for manual review)", review_path, len(review_df))

    # ---- Part 2: automated centroid probe over the FULL classified/unclassified pools ----
    cache = _load_cache()

    c_class_texts = [_text_of(r) for _, r in c_classified.iterrows()]
    j_class_texts = [_text_of(r) for _, r in j_classified.iterrows()]
    c_unclass_texts = [_text_of(r) for _, r in c_unclassified.iterrows()]
    j_unclass_texts = [_text_of(r) for _, r in j_unclassified.iterrows()]

    all_texts = c_class_texts + j_class_texts + c_unclass_texts + j_unclass_texts
    unique_texts = list(dict.fromkeys(all_texts))
    missing = [t for t in unique_texts if _rec_hash(t) not in cache]
    logger.info("%d unique texts total, %d already cached, %d need embedding",
                len(unique_texts), len(unique_texts) - len(missing), len(missing))
    _embed_missing(missing, cache)

    c_class_emb = _normalize(_emb_matrix(c_class_texts, cache))
    j_class_emb = _normalize(_emb_matrix(j_class_texts, cache))
    c_unclass_emb = _normalize(_emb_matrix(c_unclass_texts, cache))
    j_unclass_emb = _normalize(_emb_matrix(j_unclass_texts, cache))

    pooled_class_emb = np.vstack([c_class_emb, j_class_emb])
    pooled_class_domain = c_classified["unified_domain"].tolist() + j_classified["unified_domain"].tolist()

    pooled_unclass_emb = np.vstack([c_unclass_emb, j_unclass_emb]) if (len(c_unclass_emb) + len(j_unclass_emb)) else pooled_class_emb[:0]
    pooled_unclass_meta = (
        [{"type": "course", "title": str(t)} for t in c_unclassified["title"].tolist()]
        + [{"type": "job", "title": str(t)} for t in j_unclassified["title"].tolist()]
    )

    all_domains = sorted(set(pooled_class_domain))
    domain_to_idx: dict[str, list[int]] = {d: [] for d in all_domains}
    for i, d in enumerate(pooled_class_domain):
        domain_to_idx[d].append(i)

    centroids: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for d in all_domains:
        vecs = pooled_class_emb[domain_to_idx[d]]
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm else centroid
        centroids[d] = centroid
        thresholds[d] = float(np.median(vecs @ centroid))

    logger.info("=== Domain centroids (from classified records) ===")
    for d in all_domains:
        logger.info("  %-18s n_classified=%-5d median_cos_to_own_centroid=%.4f",
                    d, len(domain_to_idx[d]), thresholds[d])

    candidates = []
    for d, centroid in centroids.items():
        if len(pooled_unclass_emb) == 0:
            break
        sims = pooled_unclass_emb @ centroid
        thr = thresholds[d]
        for i in np.where(sims > thr)[0]:
            candidates.append({
                "domain": d,
                "cosine": float(sims[i]),
                "median_threshold": thr,
                "margin": float(sims[i] - thr),
                "type": pooled_unclass_meta[i]["type"],
                "title": pooled_unclass_meta[i]["title"],
            })
    cand_df = pd.DataFrame(candidates).sort_values("margin", ascending=False) if candidates else pd.DataFrame(
        columns=["domain", "cosine", "median_threshold", "margin", "type", "title"])

    total_unclassified = len(pooled_unclass_meta)
    logger.info("=== Candidate false-negative counts per domain (unclassified pool = %d) ===", total_unclassified)
    for d in all_domains:
        n = int((cand_df["domain"] == d).sum())
        logger.info("  %-18s %5d candidates (%.1f%% of unclassified pool)",
                    d, n, 100 * n / total_unclassified if total_unclassified else 0.0)

    if len(cand_df):
        n_unique = cand_df[["type", "title"]].drop_duplicates().shape[0]
        logger.info("Unique unclassified records flagged for >=1 domain: %d / %d (%.1f%%)",
                    n_unique, total_unclassified, 100 * n_unique / total_unclassified)

    cand_path = _LABELED_DIR / "recall_candidates.csv"
    cand_df.to_csv(cand_path, index=False)
    logger.info("Wrote %s (%d candidate domain-matches)", cand_path, len(cand_df))

    logger.info("=== Top 10 example candidates (highest margin above domain median) ===")
    for _, r in cand_df.head(10).iterrows():
        logger.info("  [%-6s] domain=%-18s cos=%.4f  median=%.4f  margin=+%.4f  %s",
                    r["type"], r["domain"], r["cosine"], r["median_threshold"], r["margin"], str(r["title"])[:90])


if __name__ == "__main__":
    main()
