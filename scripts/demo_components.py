"""
Standalone demo of the seven-component CurriculumGPT score — NO database needed.

Scores one (or a few) curriculum vs. job pairs and prints the full 7-component
breakdown, fusion weights, and the fused alignment score. Optionally writes an
HTML page you can open in a browser to *see* the bars.

Usage:
    python scripts/demo_components.py
    python scripts/demo_components.py --html demo.html
    python scripts/demo_components.py \
        --curriculum "Machine learning course: neural networks in Python, data analysis" \
        --job "ML Engineer: build deep learning models in Python, analyze data"
    python scripts/demo_components.py --checkpoint checkpoints/cgpt_epochXX.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from ml.models.curriculum_gpt import COMPONENT_NAMES, CurriculumGPT  # noqa: E402
from ml.models.cgpt_features import build_batch  # noqa: E402

logging.basicConfig(level=logging.WARNING)

DEFAULT_PAIRS = [
    ("Machine Learning course: students implement neural networks in Python, "
     "analyze datasets, evaluate models, and design experiments.",
     "Machine Learning Engineer: build and deploy deep learning models in Python, "
     "perform data analysis, and design scalable pipelines."),
    ("Introduction to Marketing: brand strategy, advertising campaigns, "
     "consumer behavior, and market research.",
     "Backend Software Engineer: design REST APIs in Java, manage SQL databases, "
     "and build distributed systems."),
]


def score_pairs(pairs, checkpoint=None):
    model = CurriculumGPT()
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"[loaded checkpoint: {checkpoint}]\n")
    model.eval()
    tok = model.bge.tokenizer

    curr = [p[0] for p in pairs]
    jobs = [p[1] for p in pairs]
    batch = build_batch(curr, jobs, tok, device="cpu")
    with torch.no_grad():
        out = model(batch)

    weights = dict(zip(COMPONENT_NAMES, out["weights"].tolist()))
    rows = []
    for i in range(len(pairs)):
        comps = dict(zip(COMPONENT_NAMES, out["component_scores"][i].tolist()))
        rows.append({"curriculum": curr[i], "job": jobs[i],
                     "score": float(out["alignment_score"][i]), "components": comps})
    return rows, weights


def print_report(rows, weights):
    print("=" * 70)
    print("FUSION WEIGHTS  softmax(theta)")
    for name, w in weights.items():
        print(f"  {name:10s} {'#' * int(w * 50):50s} {w * 100:5.1f}%")
    print("=" * 70)
    for i, row in enumerate(rows, 1):
        print(f"\nPAIR {i}   fused alignment score = {row['score']:.3f}")
        print(f"  curriculum: {row['curriculum'][:60]}...")
        print(f"  job:        {row['job'][:60]}...")
        print("  components:")
        for name, val in row["components"].items():
            print(f"    {name:10s} {'#' * int(val * 40):40s} {val:.3f}")


def write_html(rows, weights, path):
    def bar(val, color):
        return (f'<div style="background:#1e293b;border-radius:6px;height:14px;flex:1;overflow:hidden">'
                f'<div style="background:{color};height:100%;width:{val*100:.0f}%"></div></div>')

    blocks = []
    wrows = "".join(
        f'<div style="display:flex;gap:8px;align-items:center;margin:4px 0">'
        f'<span style="width:90px;color:#cbd5e1;font-size:13px">{n}</span>'
        f'{bar(w, "#818cf8")}<span style="width:48px;text-align:right;color:#e2e8f0">{w*100:.0f}%</span></div>'
        for n, w in weights.items())
    blocks.append(f'<h2 style="color:#f1f5f9">Fusion weights (softmax θ)</h2>{wrows}')

    for i, row in enumerate(rows, 1):
        crows = "".join(
            f'<div style="display:flex;gap:8px;align-items:center;margin:4px 0">'
            f'<span style="width:90px;color:#cbd5e1;font-size:13px;text-transform:capitalize">{n}</span>'
            f'{bar(v, "#34d399")}<span style="width:48px;text-align:right;color:#e2e8f0">{v:.2f}</span></div>'
            for n, v in row["components"].items())
        blocks.append(
            f'<div style="margin-top:24px;padding:16px;background:#0f172a;border:1px solid #1e293b;border-radius:10px">'
            f'<h3 style="color:#f1f5f9;margin:0 0 4px">Pair {i} — fused score '
            f'<span style="color:#a5b4fc">{row["score"]:.3f}</span></h3>'
            f'<p style="color:#94a3b8;font-size:12px;margin:2px 0">📘 {row["curriculum"][:90]}…</p>'
            f'<p style="color:#94a3b8;font-size:12px;margin:2px 0 12px">💼 {row["job"][:90]}…</p>'
            f'{crows}</div>')

    html = (f'<html><body style="font-family:system-ui;background:#020617;padding:40px;max-width:760px;margin:auto">'
            f'<h1 style="color:#f8fafc">CurriculumGPT — Seven-Component Breakdown</h1>'
            f'{"".join(blocks)}</body></html>')
    Path(path).write_text(html, encoding="utf-8")
    print(f"\nHTML written to {path} — open it in a browser.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum")
    ap.add_argument("--job")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--html", default=None)
    args = ap.parse_args()

    pairs = [(args.curriculum, args.job)] if args.curriculum and args.job else DEFAULT_PAIRS
    print("Loading CurriculumGPT (BGE semantic core)…\n")
    rows, weights = score_pairs(pairs, args.checkpoint)
    print_report(rows, weights)
    if args.html:
        write_html(rows, weights, args.html)


if __name__ == "__main__":
    main()
