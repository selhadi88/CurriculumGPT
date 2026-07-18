"""
AlignmentModel — wraps either BGEBiEncoder or CurriculumTransformer
and exposes a unified inference API used by the backend service.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .bi_encoder import BGEBiEncoder
from .transformer import CurriculumTransformer

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    overall_score: float                # 0–1
    curriculum_skills: list[float]      # per-skill logits / probabilities
    job_skills: list[float]
    curriculum_embedding: list[float]
    job_embedding: list[float]
    top_skill_gaps: list[tuple[int, float]] = field(default_factory=list)
    # Seven-component breakdown (populated by the curriculum_gpt backend)
    component_scores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0


class AlignmentModel:
    """
    Unified inference interface for both model backends.
    Lazy-loads on first call; holds a singleton per process.
    """

    def __init__(
        self,
        model_type: str = "bge",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        num_skills: int = 150,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._model_type = model_type
        self._checkpoint = checkpoint_path
        self._num_skills = num_skills
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", ".model_cache")
        self._model: Optional[BGEBiEncoder | CurriculumTransformer] = None
        self._tokenizer: Optional[object] = None

    def _load(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading %s model on %s…", self._model_type, self._device)

        if self._model_type == "bge":
            model = BGEBiEncoder(num_skills=self._num_skills)
        elif self._model_type == "custom":
            model = CurriculumTransformer(num_skills=self._num_skills)
            from transformers import AutoTokenizer
            logger.info("Loading bert-base-uncased tokenizer (cache: %s)…", self._cache_dir)
            self._tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                cache_dir=self._cache_dir,
            )
            logger.info("Tokenizer ready")
        elif self._model_type == "curriculum_gpt":
            from .curriculum_gpt import CurriculumGPT
            # num_skills=None → CurriculumGPT derives it from the taxonomy so the
            # trend BiLSTM input dim matches the actual skill-vector length.
            model = CurriculumGPT(num_skills=None, cache_dir=self._cache_dir)
            # CurriculumGPT uses the BGE tokenizer for its semantic core.
            self._tokenizer = model.bge.tokenizer
            logger.info("CurriculumGPT (7-component) ready")
        else:
            raise ValueError(
                f"Unknown model_type: {self._model_type!r}. Use 'bge', 'custom', or 'curriculum_gpt'."
            )

        if self._checkpoint and Path(self._checkpoint).exists():
            state = torch.load(self._checkpoint, map_location=self._device)
            model.load_state_dict(state["model_state_dict"], strict=False)
            logger.info("Loaded checkpoint from %s", self._checkpoint)

        model.eval()
        model.to(self._device)
        self._model = model
        logger.info("%s model ready", self._model_type)

    @property
    def model(self) -> BGEBiEncoder | CurriculumTransformer:
        self._load()
        assert self._model is not None
        return self._model

    @torch.no_grad()
    def score(
        self,
        curriculum_texts: list[str],
        job_texts: list[str],
        batch_size: int = 32,
        pub_years: Optional[list] = None,
        curriculum_order: Optional[list] = None,
    ) -> list[AlignmentResult]:
        """
        Score all (curriculum, job) pairs.
        Returns one AlignmentResult per pair.

        pub_years: optional per-job posting year (powers Recency).
        curriculum_order: optional per-curriculum sequence index (powers Structure).
        """
        self._load()
        assert self._model is not None

        if self._model_type == "bge":
            return self._score_bge(curriculum_texts, job_texts, batch_size)
        if self._model_type == "curriculum_gpt":
            return self._score_curriculum_gpt(
                curriculum_texts, job_texts, batch_size, pub_years, curriculum_order
            )
        return self._score_custom(curriculum_texts, job_texts, batch_size)

    def _score_curriculum_gpt(
        self,
        curriculum_texts: list[str],
        job_texts: list[str],
        batch_size: int,
        pub_years: Optional[list] = None,
        curriculum_order: Optional[list] = None,
    ) -> list[AlignmentResult]:
        from .cgpt_features import build_batch
        from .curriculum_gpt import COMPONENT_NAMES

        model = self._model
        results: list[AlignmentResult] = []

        for start in range(0, len(curriculum_texts), batch_size):
            curr_chunk = curriculum_texts[start : start + batch_size]
            job_chunk = job_texts[start : start + batch_size]
            year_chunk = pub_years[start : start + batch_size] if pub_years else None
            batch = build_batch(
                curr_chunk, job_chunk, self._tokenizer,
                pub_years=year_chunk, device=self._device,
            )
            out = model(batch)  # type: ignore[operator]

            scores = out["alignment_score"].tolist()
            comp = out["component_scores"].tolist()                # (B, 7)
            weights = {n: float(w) for n, w in zip(COMPONENT_NAMES, out["weights"].tolist())}
            curr_emb = out["curriculum_embedding"].tolist()
            job_emb = out["job_embedding"].tolist()

            for i, score in enumerate(scores):
                results.append(AlignmentResult(
                    overall_score=float(score),
                    curriculum_skills=[0.0] * self._num_skills,
                    job_skills=[0.0] * self._num_skills,
                    curriculum_embedding=curr_emb[i],
                    job_embedding=job_emb[i],
                    component_scores={n: float(v) for n, v in zip(COMPONENT_NAMES, comp[i])},
                    weights=weights,
                ))

        return results

    def _score_bge(
        self,
        curriculum_texts: list[str],
        job_texts: list[str],
        batch_size: int,
    ) -> list[AlignmentResult]:
        assert isinstance(self._model, BGEBiEncoder)

        # Use backbone CLS embeddings directly — projection heads are randomly
        # initialized until fine-tuned, so bypassing them gives correct zero-shot scores.
        # Must be a falsy check, not `is None`: MODEL_CHECKPOINT commonly arrives as ""
        # (e.g. docker-compose's `${MODEL_CHECKPOINT:-}`), which is not None but still
        # means "no checkpoint" — `is None` would wrongly route through the untrained heads.
        use_backbone = not self._checkpoint

        if use_backbone:
            curr_embs = self._model.encode_backbone_texts(curriculum_texts, batch_size=batch_size)
            job_embs = self._model.encode_backbone_texts(job_texts, batch_size=batch_size)
        else:
            curr_embs = self._model.encode_texts(curriculum_texts, encoder="curriculum", batch_size=batch_size)
            job_embs = self._model.encode_texts(job_texts, encoder="job", batch_size=batch_size)

        scores = (curr_embs * job_embs).sum(dim=-1).tolist()

        results: list[AlignmentResult] = []
        for i, score in enumerate(scores):
            results.append(AlignmentResult(
                overall_score=float(score),
                curriculum_skills=[0.0] * self._num_skills,
                job_skills=[0.0] * self._num_skills,
                curriculum_embedding=curr_embs[i].tolist(),
                job_embedding=job_embs[i].tolist(),
                top_skill_gaps=[],
            ))

        return results

    def _score_custom(
        self,
        curriculum_texts: list[str],
        job_texts: list[str],
        batch_size: int,
    ) -> list[AlignmentResult]:
        assert isinstance(self._model, CurriculumTransformer)
        tokenizer = self._tokenizer

        results: list[AlignmentResult] = []
        dev = torch.device(self._device)

        for curr_text, job_text in zip(curriculum_texts, job_texts):
            curr_enc = tokenizer(curr_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            job_enc = tokenizer(job_text, return_tensors="pt", truncation=True, max_length=512, padding=True)

            out = self._model(
                curr_enc["input_ids"].to(dev),
                curr_enc["attention_mask"].to(dev),
                job_enc["input_ids"].to(dev),
                job_enc["attention_mask"].to(dev),
            )

            score = out["alignment_score"].item()
            curr_skills = out["curriculum_skills"].sigmoid().squeeze(0).tolist()
            job_skills = out["job_skills"].sigmoid().squeeze(0).tolist()

            results.append(AlignmentResult(
                overall_score=float(score),
                curriculum_skills=curr_skills,
                job_skills=job_skills,
                curriculum_embedding=out["curriculum_embedding"].squeeze(0).tolist(),
                job_embedding=out["job_embedding"].squeeze(0).tolist(),
                top_skill_gaps=self._compute_gaps(curr_skills, job_skills),
            ))

        return results

    @staticmethod
    def _compute_gaps(
        curr_skills: list[float],
        job_skills: list[float],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return top-k (skill_index, gap_magnitude) pairs sorted by gap descending."""
        gaps = [(i, max(0.0, j - c)) for i, (c, j) in enumerate(zip(curr_skills, job_skills))]
        gaps.sort(key=lambda x: x[1], reverse=True)
        return gaps[:top_k]


# Module-level singleton — lazy-loaded by the backend service
_default_model: Optional[AlignmentModel] = None


def get_model(
    model_type: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> AlignmentModel:
    global _default_model
    if _default_model is None:
        _default_model = AlignmentModel(
            model_type=model_type or os.getenv("MODEL_TYPE", "bge"),
            checkpoint_path=checkpoint or os.getenv("MODEL_CHECKPOINT"),
        )
    return _default_model
