"""
Generate the report figures under results/figures/ (PNG, 300 DPI).

Figure 1: alignment heatmap — curriculum domain x job domain, mean zero-shot
          BGE cosine similarity (bge_sim), from the gold set.
Figure 2: industry coverage vs. curriculum coverage per domain — fraction of
          distinct jobs/courses (in data/labeled/pairs.parquet) with at least
          one candidate pair at or above the coverage threshold (0.35, same
          default as ml/evaluation/metrics.py's AlignmentMetrics).
Figure 3: skill gap severity per domain — average job-side skill presence
          minus average curriculum-side skill presence, bucketed into
          high (>=0.4) / medium (0.2-0.4) / low (0-0.2), same thresholds as
          AlignmentMetrics._compute_skill_gaps.
Figure 4: baseline (zero-shot BGE) vs. fine-tuned checkpoint performance on
          the gold set — accuracy, AUC, NDCG@10.
Figure 5: LLM label distribution (aligned/partial/not_aligned) for the
          training sample (pairs_sample_500_llm.parquet) and the gold set
          (gold_llm.parquet).

Usage:
    python scripts/generate_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.components import skill_utils  # noqa: E402

_ROOT = Path(__file__).parent.parent
_LABELED = _ROOT / "data" / "labeled"
_RESULTS = _ROOT / "results"
_FIG_DIR = _RESULTS / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["computer_science", "data_science", "cloud_devops", "cybersecurity", "business", "design"]
DPI = 300
COVERAGE_THRESHOLD = 0.35  # matches AlignmentMetrics default
GAP_HIGH, GAP_MED = 0.4, 0.2  # matches AlignmentMetrics default


def _vec_from_indices(idx, n: int) -> np.ndarray:
    v = np.zeros(n)
    for i in idx:
        if 0 <= i < n:
            v[i] = 1.0
    return v


def figure1_alignment_heatmap() -> None:
    df = pd.read_parquet(_LABELED / "gold_llm.parquet")
    mat = np.full((len(DOMAINS), len(DOMAINS)), np.nan)
    for i, cd in enumerate(DOMAINS):
        for j, jd in enumerate(DOMAINS):
            sub = df[(df["course_domain"] == cd) & (df["job_domain"] == jd)]
            if len(sub):
                mat[i, j] = sub["bge_sim"].mean()

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e8e8e8")
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    mid = (vmin + vmax) / 2

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(DOMAINS))); ax.set_xticklabels(DOMAINS, rotation=40, ha="right")
    ax.set_yticks(range(len(DOMAINS))); ax.set_yticklabels(DOMAINS)
    ax.set_xlabel("Job Domain")
    ax.set_ylabel("Curriculum Domain")
    ax.set_title("Figure 1: Curriculum–Job Alignment Heatmap\n"
                  "Mean zero-shot BGE cosine similarity — gold set (n=300)")
    for i in range(len(DOMAINS)):
        for j in range(len(DOMAINS)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < mid else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean cosine similarity")
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "figure1_alignment_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("Wrote figure1_alignment_heatmap.png")


def figure2_coverage_by_domain() -> None:
    df = pd.read_parquet(_LABELED / "pairs.parquet")
    rows = []
    for d in DOMAINS:
        jobs_d = df[df["job_domain"] == d]
        courses_d = df[df["course_domain"] == d]
        industry_cov = (
            jobs_d.groupby("job_text")["bge_sim"].max().ge(COVERAGE_THRESHOLD).mean()
            if len(jobs_d) else 0.0
        )
        curriculum_cov = (
            courses_d.groupby("curriculum_text")["bge_sim"].max().ge(COVERAGE_THRESHOLD).mean()
            if len(courses_d) else 0.0
        )
        rows.append({"domain": d, "industry_coverage": industry_cov, "curriculum_coverage": curriculum_cov})
    cov_df = pd.DataFrame(rows)

    x = np.arange(len(DOMAINS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - width / 2, cov_df["industry_coverage"] * 100, width,
                label="Industry coverage (% jobs matched)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, cov_df["curriculum_coverage"] * 100, width,
                label="Curriculum coverage (% courses matched)", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(DOMAINS, rotation=30, ha="right")
    ax.set_ylabel(f"% covered (BGE cosine ≥ {COVERAGE_THRESHOLD})")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 2: Industry vs. Curriculum Coverage by Domain\n"
                  "data/labeled/pairs.parquet (n=4,361 candidate pairs)")
    ax.legend()
    ax.bar_label(b1, fmt="%.0f%%", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.0f%%", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "figure2_coverage_by_domain.png", dpi=DPI)
    plt.close(fig)
    print("Wrote figure2_coverage_by_domain.png")


def figure3_skill_gap_severity() -> None:
    df = pd.read_parquet(_LABELED / "pairs.parquet")
    n_skills = skill_utils.get_num_skills()
    rows = []
    for d in DOMAINS:
        curr_unique = df[df["course_domain"] == d].drop_duplicates("curriculum_text")
        job_unique = df[df["job_domain"] == d].drop_duplicates("job_text")
        if not len(curr_unique) or not len(job_unique):
            rows.append({"domain": d, "high": 0, "medium": 0, "low": 0})
            continue
        curr_vecs = np.stack([_vec_from_indices(x, n_skills) for x in curr_unique["skill_multihot"]])
        job_vecs = np.stack([_vec_from_indices(x, n_skills) for x in job_unique["job_skill_mh"]])
        gaps = np.maximum(0.0, job_vecs.mean(axis=0) - curr_vecs.mean(axis=0))
        rows.append({
            "domain": d,
            "high": int((gaps >= GAP_HIGH).sum()),
            "medium": int(((gaps >= GAP_MED) & (gaps < GAP_HIGH)).sum()),
            "low": int(((gaps > 0.0) & (gaps < GAP_MED)).sum()),
        })
    gap_df = pd.DataFrame(rows)

    x = np.arange(len(DOMAINS))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, gap_df["high"], label=f"High gap (≥{GAP_HIGH})", color="#d62728")
    ax.bar(x, gap_df["medium"], bottom=gap_df["high"],
           label=f"Medium gap ({GAP_MED}–{GAP_HIGH})", color="#ff7f0e")
    ax.bar(x, gap_df["low"], bottom=gap_df["high"] + gap_df["medium"],
           label=f"Low gap (0–{GAP_MED})", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(DOMAINS, rotation=30, ha="right")
    ax.set_ylabel("Number of taxonomy skills")
    ax.set_title("Figure 3: Skill Gap Severity by Domain\n"
                  "Job-side minus curriculum-side skill presence, averaged per domain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "figure3_skill_gap_severity.png", dpi=DPI)
    plt.close(fig)
    print("Wrote figure3_skill_gap_severity.png")


def figure4_baseline_vs_finetuned() -> None:
    baseline = json.loads((_RESULTS / "gold_evaluation.json").read_text())["overall"]
    finetuned = json.loads((_RESULTS / "finetuned_evaluation.json").read_text())["overall"]

    metrics = ["accuracy", "auc", "ndcg_at_10"]
    labels = ["Accuracy", "AUC", "NDCG@10"]
    base_vals = [baseline[m] for m in metrics]
    fine_vals = [finetuned[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - width / 2, base_vals, width, label="Zero-shot BGE (baseline, production)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, fine_vals, width, label="Fine-tuned (500-pair sample, reverted)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Figure 4: Baseline vs. Fine-tuned Performance\nGold set (n=300)")
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=3)
    ax.bar_label(b2, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "figure4_baseline_vs_finetuned.png", dpi=DPI)
    plt.close(fig)
    print("Wrote figure4_baseline_vs_finetuned.png")


def figure5_llm_label_distribution() -> None:
    train_df = pd.read_parquet(_LABELED / "pairs_sample_500_llm.parquet")
    gold_df = pd.read_parquet(_LABELED / "gold_llm.parquet")

    order = ["aligned", "partial", "not_aligned"]
    colors = {"aligned": "#2ca02c", "partial": "#ff7f0e", "not_aligned": "#d62728"}

    train_counts = train_df["llm_label"].value_counts().reindex(order).fillna(0)
    gold_counts = gold_df["llm_label"].value_counts().reindex(order).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    panels = [
        (axes[0], train_counts, f"Training sample\npairs_sample_500_llm.parquet (n={int(train_counts.sum())})"),
        (axes[1], gold_counts, f"Gold set\ngold_llm.parquet (n={int(gold_counts.sum())})"),
    ]
    for ax, counts, title in panels:
        ax.pie(
            counts.values,
            labels=[f"{label}\n({int(c)})" for label, c in zip(order, counts.values)],
            colors=[colors[label] for label in order],
            autopct="%1.1f%%", startangle=90,
        )
        ax.set_title(title)
    fig.suptitle("Figure 5: LLM Label Distribution — Training Sample vs. Gold Set", y=1.03)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "figure5_llm_label_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figure5_llm_label_distribution.png")


def main() -> None:
    figure1_alignment_heatmap()
    figure2_coverage_by_domain()
    figure3_skill_gap_severity()
    figure4_baseline_vs_finetuned()
    figure5_llm_label_distribution()
    print(f"\nAll figures written to {_FIG_DIR}/")


if __name__ == "__main__":
    main()
