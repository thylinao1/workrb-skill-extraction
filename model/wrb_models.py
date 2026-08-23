"""Submission-ready WorkRB models for RecSys-HR 2026.

FusionRetriever      : z-weighted CurriculumMatch + ConTeXT, 0.6501 on validation. Submittable as it stands.
CrossEncoderReranker : FusionRetriever candidates -> a cross-encoder checkpoint reranks top-K -> lift.
                       Drop in the the notebook environment-fine-tuned checkpoint via `ce_path` and it is submittable.

Both are real `workrb.models.ModelInterface` models, so `submission/generate_submission_file.py` and the
official scorer treat them like any baseline. nDCG is order-only, so scores are not calibrated, only ordered.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from workrb.models import ModelInterface, CurriculumMatchModel, ConTeXTMatchModel
from workrb.types import ModelInputType
from workrb_challenge.models import WorkrbSaveable

# BUGFIX 2026-07-28: generate_submission_file.py::_load_model requires exactly one ModelInterface
# subclass visible in the def file. This module imports CurriculumMatchModel and ConTeXTMatchModel
# (both ModelInterface subclasses) and defines two of its own, so the loader saw 4 candidates and
# raised "Expected exactly one ModelInterface subclass". Pin the intended entry point. Swap to
# "CrossEncoderReranker" when submitting that variant instead.
__all__ = ["FusionRetriever"]

ESCO_PREFIX = "http://data.europa.eu/esco/skill/"
# Built-in models require non-None input types; the task supplies them, these are the extraction defaults.
_DEF_QIT = ModelInputType.SKILL_SENTENCE
_DEF_TIT = ModelInputType.SKILL_NAME


def _to_t(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def _zscore_rows(m: torch.Tensor) -> torch.Tensor:
    mu = m.mean(dim=1, keepdim=True)
    sd = m.std(dim=1, keepdim=True) + 1e-9
    return (m - mu) / sd


class FusionRetriever(nn.Module, ModelInterface, WorkrbSaveable):
    """z-weighted per-query fusion of CurriculumMatch + ConTeXT (w*curric + (1-w)*context, z-scored)."""

    def __init__(self, w: float = 0.6,
                 leaderboard_name: str = "Fusion-Curric-ConTeXT",
                 leaderboard_description: str = "Per-query z-weighted fusion of CurriculumMatch + ConTeXT-Match retrievers."):
        super().__init__()
        self.w = float(w)
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description
        self.curric = CurriculumMatchModel()
        self.context = ConTeXTMatchModel()

    def _rank(self, model, queries, targets, qit, tit) -> torch.Tensor:
        with torch.no_grad():
            r = model.compute_rankings(queries, targets, query_input_type=qit, target_input_type=tit)
        return _to_t(r)

    def _compute_rankings(self, queries, targets, query_input_type=None, target_input_type=None) -> torch.Tensor:
        qit = query_input_type or _DEF_QIT
        tit = target_input_type or _DEF_TIT
        c = _zscore_rows(self._rank(self.curric, queries, targets, qit, tit))
        x = _zscore_rows(self._rank(self.context, queries, targets, qit, tit))
        return self.w * c + (1.0 - self.w) * x

    def _compute_classification(self, texts, targets, input_type=None, target_input_type=None):
        return self._compute_rankings(texts, targets)

    @property
    def name(self): return self._leaderboard_name
    @property
    def description(self): return self._leaderboard_description
    @property
    def classification_label_space(self): return None

    def _save_extra(self, path):
        return {"w": self.w, "leaderboard_name": self._leaderboard_name,
                "leaderboard_description": self._leaderboard_description}


class CrossEncoderReranker(nn.Module, ModelInterface, WorkrbSaveable):
    """Retriever candidates -> cross-encoder reranks top-K over (query, 'title. description') -> lift.

    ce_path: a fine-tuned cross-encoder checkpoint dir (sentence-transformers CrossEncoder) OR a HF id.
    esco_repo: a *-graded repo whose 'corpus' config supplies title->description (all 5 share the ESCO corpus).
    """

    def __init__(self, ce_path: str = "BAAI/bge-reranker-base", w: float = 0.6, top_k: int = 200,
                 max_length: int = 512, esco_repo: str = "TechWolf/Skill-extraction-Tech-graded",
                 leaderboard_name: str = "Fusion+CE-reranker",
                 leaderboard_description: str = "z-weighted fusion retrieval + cross-encoder rerank of the top-K."):
        super().__init__()
        self.ce_path = ce_path
        self.w = float(w)
        self.top_k = int(top_k)
        self.max_length = int(max_length)
        self.esco_repo = esco_repo
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description
        self.retriever = FusionRetriever(w=w)
        self._ce = None
        self._t2d = None

    def _title2desc(self):
        if self._t2d is None:
            from datasets import load_dataset
            df = load_dataset(self.esco_repo, "corpus", split="corpus").to_pandas()
            m = {}
            for _, r in df.iterrows():
                t = str(r["title"]).strip(); d = str(r.get("text", "") or "").strip()
                m[t] = f"{t}. {d}".strip() if d and d.lower() != "nan" else t
            self._t2d = m
        return self._t2d

    def _cross_encoder(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._ce = CrossEncoder(self.ce_path, max_length=self.max_length, device=device)
        return self._ce

    def _compute_rankings(self, queries, targets, query_input_type=None, target_input_type=None) -> torch.Tensor:
        base = self.retriever._compute_rankings(queries, targets, query_input_type, target_input_type)   # (Nq, Nt) fused
        base_np = base.numpy()
        t2d = self._title2desc()
        ce = self._cross_encoder()
        k = min(self.top_k, base_np.shape[1])
        final = base_np.copy()
        for qi in range(base_np.shape[0]):
            row = base_np[qi]
            cand = np.argpartition(row, -k)[-k:]
            docs = [t2d.get(targets[j], targets[j]) for j in cand]
            scores = np.asarray(ce.predict([(queries[qi], d) for d in docs],
                                           batch_size=16, show_progress_bar=False), dtype=np.float32)
            final[qi, cand] = row.max() + 1.0 + (scores - scores.min())
        return torch.from_numpy(final)

    def _compute_classification(self, texts, targets, input_type=None, target_input_type=None):
        return self._compute_rankings(texts, targets)

    @property
    def name(self): return self._leaderboard_name
    @property
    def description(self): return self._leaderboard_description
    @property
    def classification_label_space(self): return None

    def _save_extra(self, path):
        return {"ce_path": self.ce_path, "w": self.w, "top_k": self.top_k, "max_length": self.max_length,
                "esco_repo": self.esco_repo, "leaderboard_name": self._leaderboard_name,
                "leaderboard_description": self._leaderboard_description}
