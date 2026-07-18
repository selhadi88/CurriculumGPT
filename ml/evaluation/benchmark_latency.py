"""
Inference-latency benchmark (paper target: 3.2 ± 0.8 s/pair).

Measures per-pair wall-clock time for the curriculum_gpt backend after warm-up.

Run:
    python -m ml.evaluation.benchmark_latency
    python -m ml.evaluation.benchmark_latency --n-measure 50
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ml.models.alignment import AlignmentModel

logger = logging.getLogger(__name__)

_RESULTS = Path(__file__).resolve().parent / "results"

_SAMPLE_C = ("Machine Learning course: students implement neural networks in Python, "
             "analyze datasets, evaluate models, and design experiments.")
_SAMPLE_J = ("Machine Learning Engineer: build and deploy deep learning models in Python, "
             "perform data analysis, and design scalable model pipelines.")


def run_benchmark(n_warmup: int, n_measure: int) -> dict:
    model = AlignmentModel(model_type="curriculum_gpt")
    model._load()
    device = model._device

    for _ in range(n_warmup):
        model.score([_SAMPLE_C], [_SAMPLE_J])

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model.score([_SAMPLE_C], [_SAMPLE_J])
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    result = {
        "mean_s_per_pair": round(float(arr.mean()), 4),
        "std_s_per_pair": round(float(arr.std()), 4),
        "min_s": round(float(arr.min()), 4),
        "max_s": round(float(arr.max()), 4),
        "n_measure": n_measure,
        "device": str(device),
        "paper_target": "3.2 ± 0.8 s/pair (4xA100)",
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "latency.json").write_text(json.dumps(result, indent=2))
    logger.info("Latency: %.3f ± %.3f s/pair on %s",
                result["mean_s_per_pair"], result["std_s_per_pair"], device)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-measure", type=int, default=50)
    args = ap.parse_args()
    run_benchmark(args.n_warmup, args.n_measure)


if __name__ == "__main__":
    main()
