"""Submittable ModelInterface for the LEARNED (LambdaRank) reranker.

The honest CV number is produced by the local training harness, but a CV score cannot be uploaded. This
wraps the same feature construction plus a LightGBM model trained on ALL 286 graded queries into a
real workrb ModelInterface, so submission/generate_submission_file.py and the official scorer treat
it like any baseline.

Two invariants that must hold or the submission silently mis-scores:
  1. FEATURE ORDER AND SEMANTICS MUST MATCH eval/ltr_build.py EXACTLY. The order here is asserted
     against the meta json written at build time, so a drift fails loudly instead of quietly
     feeding the model shuffled columns.
  2. Every lane must be computable from scratch at inference (test queries have no cache). Each
     lane below is derived only from (queries, targets) plus the ESCO description table.

Memory: lanes are computed ONE encoder at a time and freed, matching the local memory limit.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from workrb.models import ModelInterface, CurriculumMatchModel, ConTeXTMatchModel
from workrb.types import ModelInputType
from workrb_challenge.models import WorkrbSaveable

_DEF_QIT = ModelInputType.SKILL_SENTENCE
_DEF_TIT = ModelInputType.SKILL_NAME

# REQUIRED by submission/generate_submission_file.py::_load_model. It scans this module with
# inspect.getmembers and raises "Expected exactly one ModelInterface subclass" if more than one is
# visible. The imports above pull in CurriculumMatchModel and ConTeXTMatchModel, which ARE
# ModelInterface subclasses, so without this pin the loader sees 3 candidates and refuses to build.
__all__ = ["LTRReranker"]

# MUST mirror the --lanes order passed to eval/ltr_build.py for the shipped model, exactly.
# The booster indexes features positionally; a reordering here silently feeds it shuffled columns
# and there is no error, only a worse score. _booster_() asserts the COUNT as a tripwire.
LANES = ["CurriculumMatch", "ConTeXTMatch", "minilm", "minilmdesc",
         "tfidfword", "tfidfchar", "bm25", "bm25desc", "editdist",
         "jobbert", "curricdesc", "fuzzset"]
TASK_ORDER = ["House", "Tech", "SkillSkape", "SkillNorm"]


def _z(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


def _ranks(m: np.ndarray) -> np.ndarray:
    order = np.argsort(-m, axis=1)
    r = np.empty_like(order)
    r[np.arange(m.shape[0])[:, None], order] = np.arange(m.shape[1])[None, :]
    return r.astype(np.float32)


class LTRReranker(nn.Module, ModelInterface, WorkrbSaveable):
    """Multi-lane retrieval + LambdaRank reranking of the fused top-K."""

    def __init__(self, model_path: str = "",
                 meta_path: str = "",
                 w: float = 0.6, top_k: int = 200,
                 task_hint: str = "House",
                 ce_path: str = "models/workrb-ce-all", ce_maxlen: int = 160,
                 ce_batch: int = 64,
                 esco_repo: str = "TechWolf/Skill-extraction-Tech-graded",
                 leaderboard_name: str = "Fusion+LambdaRank-reranker",
                 leaderboard_description: str = (
                     "Nine-lane retrieval fusion reranked by a LambdaRank model trained on graded "
                     "relevance with query-grouped folds.")):
        super().__init__()
        # Resolve defaults from THIS FILE, not the cwd. generate_submission_file.py may be invoked
        # from the repo root or from challenge/, and the LightGBM model lives one level ABOVE
        # challenge/, so cwd-relative defaults silently point at a path that does not exist.
        _chal = Path(__file__).resolve().parents[1]     # <challenge repo>
        _root = Path(__file__).resolve().parents[2]     # <project root>
        self.model_path = str(Path(model_path)) if model_path else str(_root / "models" / "ltr_full.txt")
        self.meta_path = str(Path(meta_path)) if meta_path else str(
            _chal / "eval" / "cache" / "ltr_feats_v2_meta.json")
        if ce_path and not Path(ce_path).is_absolute() and not Path(ce_path).exists():
            ce_path = str(_root / ce_path)
        self.w = float(w)
        self.top_k = int(top_k)
        self.task_hint = task_hint
        self.ce_path = ce_path
        self.ce_maxlen = int(ce_maxlen)
        self.ce_batch = int(ce_batch)
        self._ce = None
        self.esco_repo = esco_repo
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description
        self._booster = None
        self._t2d = None
        self._feat_names = None
        # The 5 task blocks share the IDENTICAL 13,891-item ESCO target space, and
        # generate_submission_file.py calls this model once per block. Without this cache every
        # encoder re-encodes all 13,891 targets five times over. Keyed by (lane, target fingerprint)
        # so a genuinely different target space still recomputes.
        self._target_cache: dict = {}
        self.use_ce = False      # set from the model meta in _booster_()
        self.use_onehot = False  # set by comparing booster arity to the meta

    # ---------- resources ----------
    def _booster_(self):
        """Read the shipped model's arity WITHOUT constructing a Booster in this process.

        lightgbm.Booster() segfaults here once torch is loaded (see _ltr_predict_worker.py), and
        torch is always loaded by the time the submission script imports this file. The LightGBM
        text format records the feature count on a `max_feature_idx=` line, so the tripwire that
        catches feature drift can be enforced by reading the file.
        """
        if self._booster is None:
            meta = json.loads(Path(self.meta_path).read_text())
            feats = list(meta["features"])
            self.use_ce = any(f.startswith("ce_") for f in feats)
            txt = Path(self.model_path).read_text()
            n_model = None
            for line in txt.splitlines():
                if line.startswith("max_feature_idx="):
                    n_model = int(line.split("=")[1]) + 1
                    break
            if n_model is None:
                raise RuntimeError(f"could not read max_feature_idx from {self.model_path}")
            self.use_onehot = (n_model == len(feats) + len(TASK_ORDER))
            if self.use_onehot:
                feats = feats + [f"task_{t}" for t in TASK_ORDER]
            self._feat_names = feats
            assert n_model == len(feats), (
                f"feature-count drift: booster expects {n_model}, builder meta describes "
                f"{len(feats)}. Rebuild the model and the meta together "
                f"(eval/ltr_build.py then eval/ltr_fit_full.py with the SAME --lanes and --ce).")
            self._booster = True
            print(f"[LTRReranker] {n_model} features | CE={self.use_ce} | onehot={self.use_onehot}",
                  flush=True)
        return self._booster

    def _predict_rows(self, X: np.ndarray) -> np.ndarray:
        """Score (N, F) features in a FRESH process so lightgbm never shares this one with torch."""
        import subprocess, sys as _sys, tempfile, os
        d = tempfile.mkdtemp(prefix="ltrpred_")
        fp, op = os.path.join(d, "X.npy"), os.path.join(d, "y.npy")
        np.save(fp, np.ascontiguousarray(X, dtype=np.float32))
        worker = str(Path(__file__).resolve().parent / "_ltr_predict_worker.py")
        r = subprocess.run([_sys.executable, worker, fp, self.model_path, op],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(op):
            raise RuntimeError(f"LTR predict worker failed (rc={r.returncode}): "
                               f"{r.stderr[-800:]}")
        y = np.load(op)
        for f in (fp, op):
            try: os.remove(f)
            except OSError: pass
        try: os.rmdir(d)
        except OSError: pass
        return y

    def _title2desc(self):
        if self._t2d is None:
            from datasets import load_dataset
            df = load_dataset(self.esco_repo, "corpus", split="corpus").to_pandas()
            m = {}
            for _, r in df.iterrows():
                t = str(r["title"]).strip()
                d = str(r.get("text", "") or "").strip()
                m[t] = f"{t}. {d}" if d and d.lower() != "nan" else t
            self._t2d = m
        return self._t2d

    # ---------- lanes ----------
    def _lane(self, lane: str, queries, targets) -> np.ndarray:
        docs = [self._title2desc().get(t, t) for t in targets]
        if lane == "CurriculumMatch":
            m = CurriculumMatchModel()
            out = np.asarray(m.compute_rankings(queries, targets, _DEF_QIT, _DEF_TIT), dtype=np.float32)
            del m; gc.collect(); return out
        if lane == "ConTeXTMatch":
            m = ConTeXTMatchModel(scoring_batch_size=8)
            out = np.asarray(m.compute_rankings(queries, targets, _DEF_QIT, _DEF_TIT), dtype=np.float32)
            del m; gc.collect(); return out
        if lane in ("minilm", "minilmdesc"):
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            sm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            sm.max_seq_length = 256
            tgt = docs if lane == "minilmdesc" else list(targets)
            key = (lane, len(targets), hash(targets[0]) ^ hash(targets[-1]))
            qe = sm.encode(list(queries), batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
            te = self._target_cache.get(key)
            if te is None:
                te = sm.encode(tgt, batch_size=64, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
                self._target_cache[key] = te
            del sm; gc.collect()
            return (qe @ te.T).astype(np.float32)
        if lane == "tfidfword":
            from workrb.models import TfIdfModel
            return np.asarray(TfIdfModel().compute_rankings(queries, targets, None, None), dtype=np.float32)
        if lane == "tfidfchar":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), lowercase=True)
            T = v.fit_transform(list(targets))
            return np.asarray(cosine_similarity(v.transform(list(queries)), T), dtype=np.float32)
        if lane == "bm25":
            from workrb.models import BM25Model
            return np.asarray(BM25Model().compute_rankings(queries, targets, None, None), dtype=np.float32)
        if lane == "bm25desc":
            import unicodedata
            from rank_bm25 import BM25Okapi
            bm = BM25Okapi([unicodedata.normalize("NFKD", d).lower().split() for d in docs])
            return np.asarray([bm.get_scores(unicodedata.normalize("NFKD", q).lower().split())
                               for q in queries], dtype=np.float32)
        if lane in ("editdist", "fuzzset"):
            from rapidfuzz import fuzz
            from rapidfuzz.process import cdist
            scorer = fuzz.ratio if lane == "editdist" else fuzz.token_set_ratio
            return np.asarray(cdist(list(queries), list(targets), scorer=scorer, workers=4),
                              dtype=np.float32)
        if lane == "jobbert":
            from workrb.models import JobBERTModel
            torch.set_num_threads(4)
            m = JobBERTModel()
            out = np.asarray(m.compute_rankings(queries, targets, _DEF_QIT, _DEF_TIT), dtype=np.float32)
            del m; gc.collect(); return out
        if lane == "curricdesc":
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            sm = SentenceTransformer("Aleksandruz/skillmatch-mpnet-curriculum-retriever", device="cpu")
            sm.max_seq_length = 256
            key = ("curricdesc", len(targets), hash(targets[0]) ^ hash(targets[-1]))
            qe = sm.encode(list(queries), batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
            te = self._target_cache.get(key)
            if te is None:
                te = sm.encode(docs, batch_size=64, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
                self._target_cache[key] = te
            del sm; gc.collect()
            return (qe @ te.T).astype(np.float32)
        raise ValueError(lane)

    def _cross_encoder(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self.ce_path, max_length=self.ce_maxlen, device="cpu")
        return self._ce

    # ---------- ranking ----------
    def _compute_rankings(self, queries, targets, query_input_type=None, target_input_type=None) -> torch.Tensor:
        queries, targets = list(queries), list(targets)
        booster = self._booster_()   # sets use_ce / use_onehot from the model meta

        mats = {}
        for lane in LANES:
            mats[lane] = self._lane(lane, queries, targets)
            gc.collect()

        base = (self.w * _z(mats["CurriculumMatch"]) + (1 - self.w) * _z(mats["ConTeXTMatch"])).astype(np.float32)
        zl = {l: _z(m) for l, m in mats.items()}
        rk = {l: _ranks(m) for l, m in mats.items()}
        rbase = _ranks(base)

        nq, nt = base.shape
        K = min(self.top_k, nt)
        ti = TASK_ORDER.index(self.task_hint) if self.task_hint in TASK_ORDER else 0
        final = base.copy()
        all_X, all_cand = [], []

        for i in range(nq):
            part = np.argpartition(base[i], -K)[-K:]
            cand = part[np.argsort(base[i][part])[::-1]]
            fz = np.stack([zl[l][i][cand] for l in LANES], axis=1)
            fr = np.stack([np.log1p(rk[l][i][cand]) for l in LANES], axis=1)
            a50 = np.stack([(rk[l][i][cand] < 50) for l in LANES], axis=1).sum(1, keepdims=True).astype(np.float32)
            a200 = np.stack([(rk[l][i][cand] < 200) for l in LANES], axis=1).sum(1, keepdims=True).astype(np.float32)
            X = np.concatenate([
                fz, fr, base[i][cand][:, None], np.log1p(rbase[i][cand])[:, None], a50, a200,
                fz.mean(1, keepdims=True), fz.std(1, keepdims=True), fz.max(1, keepdims=True),
                np.array([len(targets[j].split()) for j in cand], dtype=np.float32)[:, None],
                np.full((K, 1), len(queries[i].split()), dtype=np.float32),
            ], axis=1).astype(np.float32)

            # Cross-encoder feature, appended LAST to match eval/ltr_build.py, which concatenates
            # (ce_z, ce_logrank) after every lane/aggregate feature. Same per-query normalisation.
            if self.use_ce:
                ce = self._cross_encoder()
                cdocs = [self._title2desc().get(targets[j], targets[j]) for j in cand]
                s_ce = np.asarray(ce.predict([(queries[i], d) for d in cdocs],
                                             batch_size=self.ce_batch, show_progress_bar=False),
                                  dtype=np.float32)
                ce_z = (s_ce - s_ce.mean()) / (s_ce.std() + 1e-9)
                ce_lr = np.log1p(np.argsort(np.argsort(-s_ce)).astype(np.float32))
                X = np.concatenate([X, ce_z[:, None], ce_lr[:, None]], axis=1)

            if self.use_onehot:
                oh = np.zeros((K, 4), dtype=np.float32); oh[:, ti] = 1.0
                X = np.concatenate([X, oh], axis=1)
            all_X.append(X)
            all_cand.append(cand)

        # ONE worker call per task block, not per query: the process spawn is ~0.5s and there can be
        # hundreds of queries. Rows are stacked and split back by the fixed candidate width K.
        scores = self._predict_rows(np.concatenate(all_X, axis=0))
        for i, cand in enumerate(all_cand):
            s = scores[i * K:(i + 1) * K]
            # lift the reranked candidates above every non-candidate; nDCG is order-only so the
            # absolute offset is irrelevant, only that no non-candidate outranks a candidate.
            final[i, cand] = base[i].max() + 1.0 + (s - s.min())

        return torch.from_numpy(final.astype(np.float32))

    def _compute_classification(self, texts, targets, input_type=None, target_input_type=None):
        return self._compute_rankings(texts, targets)

    @property
    def name(self): return self._leaderboard_name

    @property
    def description(self): return self._leaderboard_description

    @property
    def classification_label_space(self): return None

    def _save_extra(self, path):
        return {"model_path": self.model_path, "meta_path": self.meta_path, "w": self.w,
                "top_k": self.top_k, "task_hint": self.task_hint, "esco_repo": self.esco_repo,
                "ce_path": self.ce_path, "ce_maxlen": self.ce_maxlen, "ce_batch": self.ce_batch,
                "leaderboard_name": self._leaderboard_name,
                "leaderboard_description": self._leaderboard_description}
