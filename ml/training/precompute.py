"""
Precompute step of the frozen-backbone training pipeline (paired with the
projection-head-only InfoNCE trainer in ml/training/trainer.py).

Runs BGE-large-en-v1.5 exactly once, in inference mode (no gradients, fp16
where the runtime supports it, small batch size), over every curriculum and
job text and caches the raw 1024-dim [CLS] embeddings to disk. The training
step never loads the backbone at all — it only reads this cache — so the
335M-parameter backbone's weights/gradients/optimizer state never enter the
training process, which is what keeps training memory to a few hundred MB
instead of several GB.

Usage:
    from ml.training.precompute import precompute_and_cache
    path = precompute_and_cache(curriculum_texts, job_texts, "data/labeled/embedding_cache.pt")
"""
from __future__ import annotations

import logging
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def _peak_rss_mib() -> float:
    """Peak resident set size of this process so far, in MiB (Linux/macOS only)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@dataclass
class PrecomputeReport:
    n_curriculum: int
    n_job: int
    dtype_used: str
    fp16_requested: bool
    fp16_fallback_reason: Optional[str]
    batch_size: int
    elapsed_seconds: float
    peak_rss_mib: float
    cache_path: str


@torch.no_grad()
def _encode_texts(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer,
    batch_size: int,
    max_length: int,
    device: str,
) -> torch.Tensor:
    """Raw (unprojected) [CLS] embeddings, upcast to fp32 before returning —
    the tiny disk/memory cost of fp32 storage buys numerical stability for
    the training step, which is the part that actually needs precision."""
    model.eval()
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = model(**encoded)
        cls = out.last_hidden_state[:, 0, :]  # (B, 1024)
        embeddings.append(cls.float().cpu())
    return torch.cat(embeddings, dim=0)


def _load_backbone(
    model_name: str, cache_dir: Optional[str], device: str, use_fp16: bool,
) -> tuple[torch.nn.Module, object, str, Optional[str]]:
    """
    Loads the backbone directly in the target dtype (never materializes an
    fp32 copy first when fp16 is requested and works, so peak memory during
    loading matches the target dtype, not fp32 + fp16 combined).

    Returns (model, tokenizer, dtype_used, fallback_reason).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    fallback_reason: Optional[str] = None
    if use_fp16:
        try:
            model = AutoModel.from_pretrained(
                model_name, cache_dir=cache_dir, torch_dtype=torch.float16,
            )
            model.to(device)
            # Cheap real forward pass (not just a dtype check) to catch any
            # op that silently misbehaves in fp16 on this specific CPU build.
            with torch.no_grad():
                probe = tokenizer(["sanity check"], return_tensors="pt").to(device)
                out = model(**probe)
                if torch.isnan(out.last_hidden_state).any() or torch.isinf(out.last_hidden_state).any():
                    raise RuntimeError("fp16 forward produced NaN/Inf")
            return model, tokenizer, "float16", None
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("fp16 backbone load/probe failed (%s) — falling back to fp32", fallback_reason)

    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float32)
    model.to(device)
    return model, tokenizer, "float32", fallback_reason


def precompute_and_cache(
    curriculum_texts: list[str],
    job_texts: list[str],
    output_path: str | Path,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cpu",
    use_fp16: bool = True,
    cache_dir: Optional[str] = None,
) -> PrecomputeReport:
    """
    Encode every curriculum/job text with the frozen backbone and write a
    single cache file to output_path containing:
        {"curriculum_embeddings": (N_c, 1024) fp32 tensor,
         "job_embeddings":        (N_j, 1024) fp32 tensor,
         "curriculum_texts": [...], "job_texts": [...],
         "model_name": ..., "dtype_used": ...}

    The backbone is dropped (goes out of scope) as soon as encoding finishes
    — nothing about it is retained past this function returning.
    """
    cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", ".model_cache")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info(
        "Precompute: loading %s (fp16=%s, batch_size=%d, device=%s)…",
        model_name, use_fp16, batch_size, device,
    )
    model, tokenizer, dtype_used, fallback_reason = _load_backbone(
        model_name, cache_dir, device, use_fp16,
    )
    logger.info("Backbone loaded in dtype=%s (peak RSS so far: %.0f MiB)",
                dtype_used, _peak_rss_mib())

    logger.info("Encoding %d curriculum texts…", len(curriculum_texts))
    curr_emb = _encode_texts(curriculum_texts, model, tokenizer, batch_size, max_length, device)

    logger.info("Encoding %d job texts…", len(job_texts))
    job_emb = _encode_texts(job_texts, model, tokenizer, batch_size, max_length, device)

    # Explicitly drop the backbone before saving/reporting — it must not
    # survive past this function, since the whole point is that it never
    # reaches the training step.
    del model, tokenizer

    peak_rss = _peak_rss_mib()
    elapsed = time.time() - t0

    torch.save(
        {
            "curriculum_embeddings": curr_emb,
            "job_embeddings": job_emb,
            "curriculum_texts": curriculum_texts,
            "job_texts": job_texts,
            "model_name": model_name,
            "dtype_used": dtype_used,
        },
        output_path,
    )
    logger.info("Cached embeddings written to %s (%.1f KiB)",
                output_path, output_path.stat().st_size / 1024)

    return PrecomputeReport(
        n_curriculum=len(curriculum_texts),
        n_job=len(job_texts),
        dtype_used=dtype_used,
        fp16_requested=use_fp16,
        fp16_fallback_reason=fallback_reason,
        batch_size=batch_size,
        elapsed_seconds=round(elapsed, 2),
        peak_rss_mib=round(peak_rss, 1),
        cache_path=str(output_path),
    )


def load_embedding_cache(path: str | Path) -> dict:
    """Load a cache file written by precompute_and_cache()."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)
