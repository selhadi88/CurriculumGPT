"""
Visual research report generator.

Reads a run folder (outputs/runs/<run_id>/) and produces report.html with:
  - run metadata (device, git commit, dataset size)
  - model comparison table + bar chart
  - component ablation chart (Table 2)
  - latency vs paper target
  - statistical summary

Charts are rendered to PNG with matplotlib and embedded as base64 so the HTML is
a single self-contained file (easy to share / attach to a paper).
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _comparison_chart(results: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    accs = [results[n].get("accuracy", 0) for n in names]
    f1s = [results[n].get("f1", 0) for n in names]
    order = sorted(range(len(names)), key=lambda i: accs[i])
    names = [names[i] for i in order]; accs = [accs[i] for i in order]; f1s = [f1s[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1))
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], accs, height=0.4, label="Accuracy", color="#6366f1")
    ax.barh([i - 0.2 for i in y], f1s, height=0.4, label="F1", color="#34d399")
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("Score"); ax.legend(loc="lower right")
    ax.set_title("Model comparison (held-out test set)")
    ax.grid(axis="x", alpha=0.3)
    return _fig_to_b64(fig)


def _ablation_chart(components: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comps = [c["component"] for c in components]
    deltas = [c["delta_accuracy"] for c in components]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#ef4444" if d > 0 else "#94a3b8" for d in deltas]
    ax.bar(comps, deltas, color=colors)
    ax.set_ylabel("Δ accuracy when removed")
    ax.set_title("Component ablation — importance (higher = more important)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    return _fig_to_b64(fig)


def _table_html(results: dict) -> str:
    rows = sorted(results.items(), key=lambda kv: -(kv[1].get("accuracy") or 0))
    head = ("<tr><th>Method</th><th>Accuracy</th><th>Precision</th>"
            "<th>Recall</th><th>F1</th><th>AUC</th></tr>")
    body = ""
    for name, m in rows:
        hl = "background:#1e293b" if "CurriculumGPT (7" in name else ""
        body += (f'<tr style="{hl}"><td>{name}</td>'
                 f'<td>{m.get("accuracy",0):.3f}</td><td>{m.get("precision",0):.3f}</td>'
                 f'<td>{m.get("recall",0):.3f}</td><td>{m.get("f1",0):.3f}</td>'
                 f'<td>{(m.get("auc") or 0):.3f}</td></tr>')
    return f'<table>{head}{body}</table>'


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    manifest = _load(run_dir / "manifest.json") or {}
    comparison = _load(run_dir / "model_comparison.json") or {}
    ablation = _load(run_dir / "ablation.json") or {}
    latency = _load(run_dir / "latency.json") or {}
    stats = _load(run_dir / "statistics.json") or {}

    env = manifest.get("environment", {})
    cfg = manifest.get("config", {})
    results = comparison.get("results", {})

    sections = []

    # Header / metadata
    sections.append(f"""
    <h1>CurriculumGPT — Research Run Report</h1>
    <div class="meta">
      <span><b>Run:</b> {manifest.get('run_id','?')}</span>
      <span><b>Device:</b> {env.get('device','?')}</span>
      <span><b>Torch:</b> {env.get('torch','?')}</span>
      <span><b>Commit:</b> {env.get('git_commit','?')}</span>
      <span><b>Pairs:</b> {cfg.get('n_pairs','?')}</span>
      <span><b>Epochs:</b> {cfg.get('epochs','?')}</span>
      <span><b>UTC:</b> {env.get('timestamp_utc','?')}</span>
    </div>""")

    # Model comparison
    if results:
        chart = _comparison_chart(results)
        sections.append(f"""
        <h2>1. Model Comparison</h2>
        {_table_html(results)}
        <img src="data:image/png;base64,{chart}"/>""")
        if stats.get("improvement_over_best_baseline_pp") is not None:
            sections.append(
                f'<p class="callout">CurriculumGPT improves on the best baseline '
                f'(<b>{stats.get("best_baseline")}</b>) by '
                f'<b>{stats["improvement_over_best_baseline_pp"]:+.2f} pp</b> accuracy.</p>')

    # Ablation
    comps = ablation.get("components", [])
    if comps:
        chart = _ablation_chart(comps)
        rows = "".join(f'<tr><td>{c["component"]}</td><td>{c["weight"]:.3f}</td>'
                       f'<td>{c["delta_accuracy"]:+.3f}</td></tr>' for c in comps)
        sections.append(f"""
        <h2>2. Component Ablation (7 components)</h2>
        <p>Full-system accuracy: <b>{ablation.get('full_accuracy','?')}</b></p>
        <table><tr><th>Component</th><th>Fusion weight</th><th>Δ accuracy</th></tr>{rows}</table>
        <img src="data:image/png;base64,{chart}"/>""")

    # Latency
    if latency:
        sections.append(f"""
        <h2>3. Inference Latency</h2>
        <p><b>{latency.get('mean_s_per_pair','?')} ± {latency.get('std_s_per_pair','?')} s/pair</b>
        on {latency.get('device','?')} &nbsp;|&nbsp; paper target: {latency.get('paper_target','?')}</p>""")

    # Limitations (always shown — research integrity)
    sections.append("""
    <h2>4. Limitations & Honest Notes</h2>
    <ul class="limits">
      <li>Labels are Gemini-Flash silver labels, not expert annotations (κ unknown vs paper's 0.87).</li>
      <li>Semantic core is BGE-large, not the paper's DeBERTa+RoBERTa ensemble (documented substitution).</li>
      <li>Recency (Srec) and Trend (Strend) default to ~0.5 — current data has no posting dates.</li>
      <li>Structure (Sstruct) ≈ 0.5 — no syllabus sequencing metadata available.</li>
      <li>Single held-out split shown here; run nested 6-fold CV for publication-grade CIs.</li>
    </ul>""")

    css = """
    body{font-family:system-ui,Segoe UI,sans-serif;background:#020617;color:#e2e8f0;
         max-width:920px;margin:auto;padding:40px}
    h1{color:#f8fafc} h2{color:#c7d2fe;border-bottom:1px solid #1e293b;padding-bottom:6px;margin-top:36px}
    .meta{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:#94a3b8;margin:10px 0 20px}
    table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}
    th,td{border:1px solid #1e293b;padding:7px 10px;text-align:left}
    th{background:#0f172a;color:#cbd5e1}
    img{max-width:100%;border-radius:8px;margin:12px 0;background:#fff;padding:6px}
    .callout{background:#0f172a;border-left:3px solid #6366f1;padding:10px 14px;border-radius:6px}
    .limits li{margin:4px 0;color:#cbd5e1}"""

    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(sections)}</body></html>"
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Report written: %s", out)
    return out


if __name__ == "__main__":
    import sys
    from ml.evaluation.run_manifest import RunManifest
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else RunManifest.latest()
    if target:
        print(build_report(target))
