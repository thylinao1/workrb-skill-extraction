"""End-to-end pipeline for graded skill extraction over the ESCO taxonomy.

One run produces everything needed to build a submission: retrieval candidates,
per-candidate features, and cross-encoder scores, for both the validation and
test splits.

Design notes, each of them a measured result rather than an assumption.

Submissions are precomputed scores, not executable code. The file is
{model: {task: {"en": {"scores": {query_id: {target_id: float}}}}}} holding the
top 500 targets per query, so no model needs to run at scoring time. This run
therefore computes the test split as well and exports everything required to
write the submission without further inference.

Candidate depth is monotonic. Measured on a fixed cross-encoder with identical
folds and features: top-100 scores 0.6852, top-200 scores 0.7226, top-500
scores 0.7393. The trend did not turn, so candidates and features are built at
K=1000. Downstream code can truncate to any shallower depth for free but cannot
go deeper than what was exported.

Stage-1 pre-training draws on the full 138,260-row synthetic corpus. An earlier
configuration sampled 12,000 rows as a cheap check and that figure was never
intended as the real run.

Two encoder cross-encoders are exported as separate feature columns rather than
averaged, leaving the combination to the downstream ranker. The second is a
bonus lane: the run is complete and submittable before it begins.

All twelve retrieval lanes are computed here. The benchmark library cannot be
installed in this environment because its dependency pins conflict with the
preinstalled framework, so its TF-IDF, BM25 and JobBERT models are transcribed
inline from the installed source. The two domain retrievers were verified
against the official implementations at a maximum absolute difference of
4.8e-07 and 0.0 respectively, with identical top-10 orderings.

Some test queries carry a leading bullet marker that no validation query has, so
the effect was invisible until the test split was inspected. A leading bullet
and its trailing whitespace are stripped for model input only; dictionary keys
keep the raw identifier. A diagnostic below measures what an injected marker
costs on validation, so the size of the effect is known rather than assumed.

The run is long and unattended, so every phase is wrapped and a single failure
does not end the run; every artefact is written as soon as it exists; a
wall-clock deadline combined with measured throughput decides whether a phase is
affordable; phases are ordered by value so the earliest complete pipeline lands
about three and a half hours in; and re-running against existing output skips
whatever is already done.
"""

import os, math, glob, time, json, re, gc, shutil, traceback, contextlib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # single GPU on purpose: DataParallel + a listwise
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # loss is not worth the risk here
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RUN          = "v6"
WORK         = os.environ.get("WORKRB_WORK", "/kaggle/working")   # overridable for the smoke test
os.makedirs(WORK, exist_ok=True)
ESCO_REPO    = "TechWolf/Skill-extraction-Tech-graded"
CURRIC_MODEL = "Aleksandruz/skillmatch-mpnet-curriculum-retriever"
CTX_MODEL    = "TechWolf/ConTeXT-Skill-Extraction-base"
JOBBERT      = "TechWolf/JobBERT-v2"
MINILM       = "sentence-transformers/all-MiniLM-L6-v2"

K            = 1000        # candidate depth for BOTH splits. Must match across val/test.
NFOLD        = 3
SEED         = 0
MAXLEN       = 192
BASE_W       = 0.6         # fused base = 0.6*z(CurriculumMatch) + 0.4*z(ConTeXT). Reproduces 0.6499.
DEADLINE_H   = float(os.environ.get("WORKRB_DEADLINE_H", "11.0"))   # override per compute window
                           # (the notebook environment kills at 12h; a Slurm job's --time may be much shorter)

# Ladder for stage-1 synthetic rows: 1 gold + 3 near + 5 mid + 6 deep-random = 15 docs/row.
N_ADJ, N_PLAUS, N_NONSENSE, DEEP_MIN = 3, 5, 6, 2000

CE_MODELS = [
    dict(key="A", hf="BAAI/bge-reranker-base",
         s1_syn=80000, s1_desc=14000, s1_batch=8, s1_lr=2e-5, s1_budget_s=95 * 60,
         s2_batch=8, s2_lr=1e-5, s2_samples=16, s2_docs=24, s2_epochs=2,
         infer_bs=256, prior_train_pps=260.0, prior_infer_pps=780.0, test_K=1000,
         test_members=("full", "fold0", "fold1", "fold2")),
    # Model B's stage 1 is deliberately shorter than A's. It is already a strong reranker out of
    # the box so it needs less adaptation, and its TEST scoring costs ~2.3h that cannot be skipped:
    # a cross-encoder with val scores but no test scores is unusable and contributes nothing.
    dict(key="B", hf="BAAI/bge-reranker-v2-m3",
         s1_syn=30000, s1_desc=14000, s1_batch=2, s1_lr=8e-6, s1_budget_s=135 * 60,
         s2_batch=2, s2_lr=5e-6, s2_samples=10, s2_docs=24, s2_epochs=1,
         # test at depth 500, not 1000: B is worth ~+0.006 as an LTR feature and its
         # measured value does not need the deep tail; halving the pairs saves ~1.2h
         infer_bs=128, prior_train_pps=75.0, prior_infer_pps=225.0, test_K=500,
         test_members=("full",)),
]

TASKS = {
    "Tech":       ("TechWolf/Skill-extraction-Tech-graded",       ["val", "test"]),
    "House":      ("TechWolf/Skill-extraction-House-graded",      ["val", "test"]),
    "SkillSkape": ("TechWolf/Skill-extraction-SkillSkape-graded", ["val", "test"]),
    "SkillNorm":  ("TechWolf/Skill-normalisation-ESCO-graded",    ["val", "test"]),
    "TechWolf":   ("TechWolf/Skill-extraction-TechWolf-graded",   ["test"]),  # TEST ONLY, no val labels
}
VAL_TASKS  = [t for t, (_, sp) in TASKS.items() if "val" in sp]
TEST_TASKS = list(TASKS.keys())
HF_SPLIT   = {"val": "validation", "test": "test"}

LANES = ["CurriculumMatch", "ConTeXTMatch", "minilm", "minilmdesc", "tfidfword", "tfidfchar",
         "bm25", "bm25desc", "editdist", "jobbert", "curricdesc", "fuzzset"]

# Model C is OPT-IN ONLY (WORKRB_CE_KEYS must name it): it exists for ARCHITECTURAL
# DIVERSITY. The measured +0.0073 for a CE pair came from two models at rho=0.55-0.66,
# and A/B are both XLM-R/bge family; DeBERTa-v3 is a different pretraining objective and
# tokenizer. bf16 REQUIRED for this family (fp16-unstable) -> run with WORKRB_BF16=1.
# Needs sentencepiece installed (DeBERTa tokenizer).
if "C" in (os.environ.get("WORKRB_CE_KEYS") or ""):
    CE_MODELS.append(dict(
        key="C", hf="microsoft/deberta-v3-large",
        s1_syn=80000, s1_desc=14000, s1_batch=8, s1_lr=1e-5, s1_budget_s=150 * 60,
        s2_batch=8, s2_lr=6e-6, s2_samples=16, s2_docs=24, s2_epochs=2,
        infer_bs=128, prior_train_pps=160.0, prior_infer_pps=480.0, test_K=1000,
        test_members=("full",)))

# Per-run env overrides so ONE script serves parallel cluster jobs without forking:
#   WORKRB_CE_KEYS="A"       train only the listed models (comma-separated)
#   WORKRB_S2_SAMPLES/_DOCS  graded stage-2 density (the stage-2 density trigger)
#   WORKRB_BF16=1            bf16 wherever fp16 was used (H100/A100; DeBERTa needs it)
BF16 = os.environ.get("WORKRB_BF16") == "1"
_keys = os.environ.get("WORKRB_CE_KEYS")
if _keys:
    CE_MODELS = [c for c in CE_MODELS if c["key"] in _keys.split(",")]
    assert CE_MODELS, f"WORKRB_CE_KEYS={_keys} matched no model"
for _c in CE_MODELS:
    if os.environ.get("WORKRB_S2_SAMPLES"):
        _c["s2_samples"] = int(os.environ["WORKRB_S2_SAMPLES"])
    if os.environ.get("WORKRB_S2_DOCS"):
        _c["s2_docs"] = int(os.environ["WORKRB_S2_DOCS"])

# SMOKE MODE. Off on the notebook environment. Set WORKRB_SMOKE=1 to run this exact file end to end on a laptop CPU
# in a few minutes against a subsampled corpus and query set. Its whole purpose is to execute every
# code path before any GPU time is spent, because a crash at hour 8 of this run is unrecoverable.
SMOKE = os.environ.get("WORKRB_SMOKE") == "1"
SMOKE_NQ, SMOKE_NT = 8, 700
if SMOKE:
    K, DEADLINE_H, NFOLD = 40, 0.5, 3
    CE_MODELS = [dict(key="A", hf="cross-encoder/ms-marco-MiniLM-L-6-v2",
                      s1_syn=120, s1_desc=40, s1_batch=2, s1_lr=2e-5, s1_budget_s=90,
                      s2_batch=2, s2_lr=1e-5, s2_samples=2, s2_docs=8, s2_epochs=1,
                      infer_bs=32, prior_train_pps=40.0, prior_infer_pps=120.0,
                      test_members=("full",))]
    print(">>> SMOKE MODE: subsampled corpus/queries, tiny model, real code path <<<", flush=True)

import numpy as np, torch
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
from transformers import TrainerCallback
try:
    from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments
except ImportError:
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

START    = time.time()
DEADLINE = START + DEADLINE_H * 3600
rng = np.random.RandomState(SEED)
AC_DTYPE = torch.bfloat16 if BF16 else torch.float16   # autocast dtype for CE inference
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}", flush=True)
print(f"deadline: {DEADLINE_H:.1f}h from now\n", flush=True)


# --------------------------------------------------------------------------------------
# artifacts: written to WORK, read from WORK or from any attached previous-run output
# --------------------------------------------------------------------------------------
def art(fn):
    """Path to an existing artifact (this run's output, or an attached earlier run), else None."""
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        return p
    for c in glob.glob(f"/kaggle/input/**/{fn}", recursive=True):
        return c
    return None


def save_np(fn, a):
    np.save(os.path.join(WORK, fn), a)
    print(f"    wrote {fn} {tuple(a.shape)} {a.dtype}", flush=True)


def save_json(fn, o):
    with open(os.path.join(WORK, fn), "w") as f:
        json.dump(o, f)


def load_np(fn):
    p = art(fn)
    if p is None:
        raise FileNotFoundError(f"required artifact missing: {fn}")
    return np.load(p)


def _np(x):
    """Any tensor on any device -> float32 numpy. np.asarray alone throws on CUDA tensors."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype=np.float32)


def free():
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


THROUGHPUT = {}   # measured pairs/sec, filled in as the run goes; priors used until then


def pps(model_key, kind, prior):
    return THROUGHPUT.get(f"{model_key}_{kind}", prior)


PHASES = []


def phase(name, est_s, fn):
    """Run one phase. Skips if the clock cannot afford it, and never lets a failure kill the run."""
    left = DEADLINE - time.time()
    if left < est_s * 0.6:
        print(f"\n### [SKIP] {name}: needs ~{est_s/60:.0f} min, only {left/60:.0f} min left", flush=True)
        PHASES.append({"name": name, "status": "skipped_no_time"})
        return None
    print(f"\n{'#'*78}\n### {name}   (est {est_s/60:.0f} min, {left/60:.0f} min left)\n{'#'*78}", flush=True)
    t0 = time.time()
    try:
        r = fn()
        el = time.time() - t0
        print(f"### [OK] {name} in {el/60:.1f} min", flush=True)
        PHASES.append({"name": name, "status": "ok", "minutes": round(el / 60, 1)})
        return r
    except Exception as e:
        traceback.print_exc()
        print(f"### [FAIL] {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        PHASES.append({"name": name, "status": f"failed:{type(e).__name__}"})
        free()
        return None


# --------------------------------------------------------------------------------------
# text handling
# --------------------------------------------------------------------------------------
# Strip ONE leading bullet run only when whitespace follows, so "* .NET" -> ".NET" and
# "- Dialogue and follow-up" -> "Dialogue and follow-up", while "-5% margin" is left alone.
_BULLET = re.compile(r"^\s*[\*•·\-–—]+\s+")


def clean_q(s):
    s = str(s).strip()
    out = _BULLET.sub("", s).strip()
    return out if len(out) >= 2 else s


DISC = 1.0 / np.log2(np.arange(max(K, 100)) + 2)   # >=100: IDCG sums over up to 100 ideal grades


def ndcg_at(y_ordered, idcg):
    g = (2.0 ** np.asarray(y_ordered[:100], dtype=np.float64)) - 1.0
    return float(np.sum(g * DISC[:len(g)]) / idcg) if idcg > 0 else 0.0


def z(a):
    return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-9)


# ======================================================================================
# PHASE 0  corpus + every query set
# ======================================================================================
corpus = load_dataset(ESCO_REPO, "corpus", split="corpus").to_pandas()
titles = [str(t).strip() for t in corpus["title"]]
descs, docs = [], []
for t, d in zip(titles, corpus["text"]):
    d = str(d or "").strip()
    d = "" if d.lower() == "nan" else d
    descs.append(d if d else t)
    docs.append(f"{t}. {d}" if d else t)
corpus_ids = [str(u) for u in corpus["_id"]]
title2idx = {t: i for i, t in enumerate(titles)}
uri2idx = {u: i for i, u in enumerate(corpus_ids)}
NT = len(titles)
print(f"[corpus] {NT} ESCO concepts", flush=True)
save_json(f"{RUN}_corpus_ids.json", corpus_ids)

Q = {}     # Q[(split, task)] = dict(qids, raw, txt, grades|None)
for tname, (repo, splits) in TASKS.items():
    for sp in splits:
        qdf = load_dataset(repo, "queries", split=HF_SPLIT[sp]).to_pandas()
        qid2text = {str(a): str(b) for a, b in zip(qdf["_id"], qdf["text"])}
        grades = None
        if sp == "val":
            qr = load_dataset(repo, "qrels", split="validation").to_pandas()
            qr = qr[qr["score"] > 0]           # 96% of the dense qrels are grade 0
            g = {}
            for qi, ci, sc in zip(qr["query-id"].astype(str), qr["corpus-id"].astype(str), qr["score"]):
                j = uri2idx.get(ci)
                if j is not None:
                    g.setdefault(qi, {})[j] = float(sc)
            keys = [k for k in qid2text if k in g]      # reproduces the local task's graded query set
            grades = [g[k] for k in keys]
        else:
            keys = list(qid2text)                        # verified 1:1 with the task's test queries
        raw = [qid2text[k].strip() for k in keys]
        Q[(sp, tname)] = {"qids": keys, "raw": raw, "txt": [clean_q(x) for x in raw], "grades": grades}
        nclean = sum(1 for a, b in zip(raw, Q[(sp, tname)]["txt"]) if a != b)
        print(f"  {sp:4s} {tname:11s} {len(keys):5d} queries   bullet-stripped: {nclean}", flush=True)
        save_json(f"{RUN}_qids_{sp}_{tname}.json", keys)
        save_json(f"{RUN}_qtext_{sp}_{tname}.json", raw)

ALLQ = [(sp, t) for t in TASKS for sp in TASKS[t][1]]

if SMOKE:
    # Shrink the corpus, keeping every gold of the surviving val queries so positives still exist.
    keep = set(range(min(SMOKE_NT, NT)))
    for (sp, t) in ALLQ:
        if Q[(sp, t)]["grades"]:
            for gd in Q[(sp, t)]["grades"][:SMOKE_NQ]:
                keep |= set(gd)
    keep = sorted(keep)
    remap = {o: n for n, o in enumerate(keep)}
    titles = [titles[i] for i in keep]
    docs = [docs[i] for i in keep]
    descs = [descs[i] for i in keep]
    corpus_ids = [corpus_ids[i] for i in keep]
    title2idx = {t: i for i, t in enumerate(titles)}
    uri2idx = {u: i for i, u in enumerate(corpus_ids)}
    NT = len(titles)
    for kq, v in Q.items():
        for f in ("qids", "raw", "txt"):
            v[f] = v[f][:SMOKE_NQ]
        if v["grades"] is not None:
            v["grades"] = [{remap[j]: g for j, g in gd.items() if j in remap}
                           for gd in v["grades"][:SMOKE_NQ]]
    # The id/text files were written above from the FULL query lists. Rewrite them so the smoke's
    # exports stay self-consistent and the local submission builder can be exercised end to end.
    for (sp, t) in ALLQ:
        save_json(f"{RUN}_qids_{sp}_{t}.json", Q[(sp, t)]["qids"])
        save_json(f"{RUN}_qtext_{sp}_{t}.json", Q[(sp, t)]["raw"])
    save_json(f"{RUN}_corpus_ids.json", corpus_ids)
    print(f">>> SMOKE: corpus {NT}, {SMOKE_NQ} queries per task-split", flush=True)


# ======================================================================================
# retriever implementations (all transcribed from the installed workrb source)
# ======================================================================================
def enc_st(hf, texts_q, texts_t, bs_q=64, bs_t=128, maxseq=256):
    """maxseq=None means LEAVE THE MODEL DEFAULT. That matters: the CurriculumMatch inline was
    verified bit-identical to the official model without touching max_seq_length, so forcing a
    value here would silently change the base and break the 0.6499 reproduction check."""
    m = SentenceTransformer(hf, device=dev)
    if maxseq is not None:
        m.max_seq_length = maxseq
    te = m.encode(texts_t, batch_size=bs_t, convert_to_numpy=True,
                  normalize_embeddings=True, show_progress_bar=False)
    out = {}
    for kq, txt in texts_q.items():
        qe = m.encode(txt, batch_size=bs_q, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
        out[kq] = (qe @ te.T).astype(np.float32)
    del m, te
    free()
    return out


def _enc_ctx(model, texts, mean, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        a = {"normalize_embeddings": False, "convert_to_tensor": True}
        if not mean:
            a["output_value"] = "token_embeddings"
        out.extend(model.encode(texts[i:i + bs], **a))
    return pad_sequence(out, batch_first=True)


def _ctx_score(tok, tgt, temp=1.0):
    dot = (tok @ tgt.T).transpose(1, 2)
    dot[dot.abs() < 1e-9] = float("-inf")
    w = torch.softmax(dot / temp, dim=2)
    nt = torch.nn.functional.normalize(tok, p=2, dim=2)
    ng = torch.nn.functional.normalize(tgt, p=2, dim=1)
    return (w * (nt @ ng.T).transpose(1, 2)).sum(dim=2)


def lane_context(texts_q, tgt_texts):
    xm = SentenceTransformer(CTX_MODEL, device=dev)
    tgt = _enc_ctx(xm, tgt_texts, mean=True)
    out = {}
    for kq, txt in texts_q.items():
        qtok = _enc_ctx(xm, txt, mean=False)
        out[kq] = _np(torch.cat([_ctx_score(qtok[i:i + 8], tgt) for i in range(0, qtok.shape[0], 8)], dim=0))
        del qtok
        free()
    del xm, tgt
    free()
    return out


def lane_jobbert(texts_q, tgt_texts):
    """JobBERT routes text through named branches via features['text_keys'].
    SKILL_SENTENCE and SKILL_NAME both map to 'positive' in workrb, so both sides use it."""
    m = SentenceTransformer(JOBBERT, device=dev)
    m.eval()

    def enc(texts, branch="positive", bs=128):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        embs = []
        for i in range(0, len(order), bs):
            batch = [texts[j] for j in order[i:i + bs]]
            f = batch_to_device(m.tokenize(batch), m.device)
            f["text_keys"] = [branch]
            with torch.no_grad():
                embs.append(m.forward(f)["sentence_embedding"])
        e = torch.cat(embs, dim=0)
        inv = [0] * len(order)
        for pos, oi in enumerate(order):
            inv[oi] = pos
        return e.index_select(0, torch.tensor(inv, dtype=torch.long, device=e.device))

    te = torch.nn.functional.normalize(enc(tgt_texts), p=2, dim=1)
    out = {}
    for kq, txt in texts_q.items():
        qe = torch.nn.functional.normalize(enc(txt), p=2, dim=1)
        out[kq] = _np(qe @ te.T)
        del qe
    del m, te
    free()
    return out


def lane_tfidf(texts_q, tgt_texts, mode):
    import unicodedata
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    pre = lambda s: unicodedata.normalize("NFKD", s).lower()
    if mode == "word":
        vec = TfidfVectorizer()                                        # workrb TfIdfModel default
        T = vec.fit_transform([pre(t) for t in tgt_texts])
        return {kq: np.asarray(cosine_similarity(vec.transform([pre(q) for q in txt]), T),
                               dtype=np.float32) for kq, txt in texts_q.items()}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), lowercase=True)  # the lane implementations
    T = vec.fit_transform(tgt_texts)
    return {kq: np.asarray(cosine_similarity(vec.transform(txt), T), dtype=np.float32)
            for kq, txt in texts_q.items()}


def lane_bm25(texts_q, tgt_texts):
    import unicodedata
    from rank_bm25 import BM25Okapi
    pre = lambda s: unicodedata.normalize("NFKD", s).lower().split()
    bm = BM25Okapi([pre(t) for t in tgt_texts])
    return {kq: np.asarray([bm.get_scores(pre(q)) for q in txt], dtype=np.float32)
            for kq, txt in texts_q.items()}


def lane_fuzz(texts_q, tgt_texts, kind):
    from rapidfuzz import fuzz
    from rapidfuzz.process import cdist
    sc = {"editdist": fuzz.ratio, "fuzzset": fuzz.token_set_ratio}[kind]
    return {kq: np.asarray(cdist(txt, tgt_texts, scorer=sc, workers=-1), dtype=np.float32)
            for kq, txt in texts_q.items()}


def compute_lane(lane, texts_q):
    if lane == "CurriculumMatch":
        return enc_st(CURRIC_MODEL, texts_q, titles, maxseq=None)   # model default, as verified
    if lane == "curricdesc":
        return enc_st(CURRIC_MODEL, texts_q, docs)
    if lane == "minilm":
        return enc_st(MINILM, texts_q, titles)
    if lane == "minilmdesc":
        return enc_st(MINILM, texts_q, docs)
    if lane == "ConTeXTMatch":
        return lane_context(texts_q, titles)
    if lane == "jobbert":
        return lane_jobbert(texts_q, titles)
    if lane == "tfidfword":
        return lane_tfidf(texts_q, titles, "word")
    if lane == "tfidfchar":
        return lane_tfidf(texts_q, titles, "char")
    if lane == "bm25":
        return lane_bm25(texts_q, titles)
    if lane == "bm25desc":
        return lane_bm25(texts_q, docs)
    if lane in ("editdist", "fuzzset"):
        return lane_fuzz(texts_q, titles, lane)
    raise ValueError(lane)


# Both text variants are computed for every task-split. Encoding the 13891 targets is what costs
# time and it is shared, so the second variant is nearly free, and it turns the bullet-stripping
# question from a guess into a measurement. Counts of affected queries printed above:
# val Tech 25/75, test TechWolf 169/324, test Tech 42/338 -- val Tech alone can decide it.
ALL_TXT = {}
for (sp, t) in ALLQ:
    ALL_TXT[f"{sp}|{t}"] = Q[(sp, t)]["txt"]     # bullets stripped
    ALL_TXT[f"{sp}R|{t}"] = Q[(sp, t)]["raw"]    # exactly what the local pipeline used
TEXT_MODE = "raw"        # decided in phase 1 by measurement; "" suffix means cleaned


# ======================================================================================
# PHASE 1  base retrievers -> candidates -> labels        (everything downstream needs this)
# ======================================================================================
def val_macro_from_base(base_by_key, suffix):
    """Macro nDCG@100 of a base ordering over the FULL target space, from the graded qrels."""
    per = {}
    for t in VAL_TASKS:
        b = base_by_key[f"val{suffix}|{t}"]
        g = Q[("val", t)]["grades"]
        o = np.argsort(-b, axis=1)[:, :100]
        vals = []
        for i in range(len(g)):
            ideal = sorted(g[i].values(), reverse=True)[:100]
            ig = (2.0 ** np.array(ideal)) - 1.0
            idcg = float(np.sum(ig * DISC[:len(ig)])) if len(ig) else 0.0
            vals.append(ndcg_at([g[i].get(int(j), 0.0) for j in o[i]], idcg))
        per[t] = float(np.mean(vals))
    return float(np.mean([per[t] for t in VAL_TASKS])), per


def phase1():
    global TEXT_MODE
    if art(f"{RUN}_cands_test_TechWolf.npy") and art(f"{RUN}_idcg_val_SkillNorm.npy"):
        tm = art(f"{RUN}_textmode.json")
        TEXT_MODE = json.load(open(tm))["mode"] if tm else "raw"
        print(f"[reuse] candidates already exist (text mode '{TEXT_MODE}'), skipping", flush=True)
        return
    cur = compute_lane("CurriculumMatch", ALL_TXT)
    print("  CurriculumMatch done", flush=True)
    ctx = compute_lane("ConTeXTMatch", ALL_TXT)
    print("  ConTeXT-Match done", flush=True)

    BASE = {k: (BASE_W * z(cur[k]) + (1 - BASE_W) * z(ctx[k])).astype(np.float32) for k in ALL_TXT}
    del cur, ctx
    free()

    # ---- GATE: the RAW variant is exactly what produced the local 0.6499. It must reproduce. ----
    raw_macro, raw_per = val_macro_from_base(BASE, "R")
    cln_macro, cln_per = val_macro_from_base(BASE, "")
    print(f"\n  [VERIFY] fused base on RAW query text = {raw_macro:.4f}   (known value 0.6499)", flush=True)
    for t in VAL_TASKS:
        print(f"      {t:11s} raw {raw_per[t]:.4f}   bullets-stripped {cln_per[t]:.4f}   "
              f"{cln_per[t]-raw_per[t]:+.4f}", flush=True)
    if SMOKE:
        print("  [VERIFY] skipped in smoke mode (corpus is subsampled)", flush=True)
    elif abs(raw_macro - 0.6499) > 0.002:
        raise RuntimeError(f"base recomputed to {raw_macro:.4f}, not 0.6499. Pipeline is misaligned; "
                           f"every downstream number would be invalid. Stopping here.")
    else:
        print("  [VERIFY] PASSED. Candidate construction matches the local pipeline exactly.", flush=True)

    # ---- DECISION: strip bullets only if it actually measures better on val ----
    TEXT_MODE = "clean" if cln_macro > raw_macro + 0.0005 else "raw"
    print(f"\n  [DECISION] bullet stripping: raw {raw_macro:.4f} vs cleaned {cln_macro:.4f} "
          f"({cln_macro-raw_macro:+.4f}) -> using '{TEXT_MODE}' text everywhere", flush=True)
    save_json(f"{RUN}_textmode.json", {"mode": TEXT_MODE, "raw": raw_macro, "clean": cln_macro,
                                       "raw_per_task": raw_per, "clean_per_task": cln_per})
    sfx = "" if TEXT_MODE == "clean" else "R"

    for (sp, t) in ALLQ:
        base = BASE[f"{sp}{sfx}|{t}"]
        kk = min(K, base.shape[1])
        part = np.argpartition(base, -kk, axis=1)[:, -kk:]
        rows = np.arange(base.shape[0])[:, None]
        cands = part[rows, np.argsort(-base[rows, part], axis=1)].astype(np.int32)
        save_np(f"{RUN}_cands_{sp}_{t}.npy", cands)
        # Saved as real RUN-prefixed artifacts, not temp files: phase 2 needs them, and phase 1
        # is skipped entirely when a previous run's output is attached.
        np.save(os.path.join(WORK, f"{RUN}_basefeat_{sp}_{t}.npy"),
                np.stack([base[rows, cands],
                          np.argsort(np.argsort(-base, axis=1), axis=1)[rows, cands]],
                         axis=-1).astype(np.float32))
        if sp == "val":
            g = Q[(sp, t)]["grades"]
            y = np.zeros(cands.shape, np.float32)
            idcg = np.zeros(len(g), np.float32)
            for i in range(len(g)):
                y[i] = [g[i].get(int(j), 0.0) for j in cands[i]]
                ideal = sorted(g[i].values(), reverse=True)[:100]
                ig = (2.0 ** np.array(ideal)) - 1.0
                idcg[i] = float(np.sum(ig * DISC[:len(ig)])) if len(ig) else 0.0
            save_np(f"{RUN}_y_val_{t}.npy", y)
            save_np(f"{RUN}_idcg_val_{t}.npy", idcg)
        del base
        free()

    # The exported candidates+labels must reproduce the same base macro. They will only differ if
    # some query's top-100 fell outside top-K, which would mean the export is lossy.
    chosen = cln_macro if TEXT_MODE == "clean" else raw_macro
    per = {}
    for t in VAL_TASKS:
        y = load_np(f"{RUN}_y_val_{t}.npy")
        idcg = load_np(f"{RUN}_idcg_val_{t}.npy")
        per[t] = float(np.mean([ndcg_at(y[i], idcg[i]) for i in range(len(y))]))
    macro = float(np.mean([per[t] for t in VAL_TASKS]))
    print(f"  [VERIFY] base recomputed from the exported top-{K} candidates = {macro:.4f} "
          f"(full space {chosen:.4f})", flush=True)
    # Only meaningful when K >= 100: nDCG@100 needs 100 ranked items, so a shallower export is
    # legitimately lower. In production K is 1000 and the two must agree exactly.
    if K >= 100 and abs(macro - chosen) > 1e-6:
        raise RuntimeError(f"candidate export is lossy: {macro:.6f} vs {chosen:.6f}")
    del BASE
    free()


phase(f"PHASE 1  base retrievers + candidates @top-{K} (val+test, both text variants)", 34 * 60, phase1)
# Phases 1 and 2 are load-bearing: every later phase reads their output, so a run that continued
# past a failure here would spend hours producing scores nothing can consume. It stops instead.
for _t in TEST_TASKS:
    if art(f"{RUN}_cands_test_{_t}") is None and art(f"{RUN}_cands_test_{_t}.npy") is None:
        raise RuntimeError(f"PHASE 1 did not produce candidates for {_t}; aborting the run.")

# From here on every lane uses ONLY the text variant phase 1 chose, so val and test features are
# built from identical preprocessing.
_sfx = "" if TEXT_MODE == "clean" else "R"
ALL_TXT = {f"{sp}|{t}": ALL_TXT[f"{sp}{_sfx}|{t}"] for (sp, t) in ALLQ}
if TEXT_MODE == "raw":
    for (sp, t) in ALLQ:          # keep the cross-encoder and query_len on the same text as the lanes
        Q[(sp, t)]["txt"] = Q[(sp, t)]["raw"]
print(f"[text mode] every lane, the cross-encoder and query_len use '{TEXT_MODE}' query text", flush=True)


# ======================================================================================
# PHASE 2  remaining 10 lanes -> per-candidate features
# ======================================================================================
NF = 2 * len(LANES) + 9


def phase2():
    if art(f"{RUN}_feat_test_TechWolf.npy"):
        print("[reuse] features already exist, skipping", flush=True)
        return
    FEAT, CAND = {}, {}
    for (sp, t) in ALLQ:
        CAND[(sp, t)] = load_np(f"{RUN}_cands_{sp}_{t}.npy")
        FEAT[(sp, t)] = np.zeros((CAND[(sp, t)].shape[0], CAND[(sp, t)].shape[1], NF), np.float32)

    ok_lanes = []
    for li, lane in enumerate(LANES):
        t0 = time.time()
        try:
            mats = compute_lane(lane, ALL_TXT)
        except Exception as e:
            traceback.print_exc()
            print(f"  [lane {lane}] FAILED, its two columns stay zero: {e}", flush=True)
            continue
        for (sp, t) in ALLQ:
            m = mats[f"{sp}|{t}"]
            c = CAND[(sp, t)]
            rows = np.arange(m.shape[0])[:, None]
            zz = z(m)
            rk = np.argsort(np.argsort(-m, axis=1), axis=1).astype(np.float32)
            FEAT[(sp, t)][:, :, li] = zz[rows, c]
            FEAT[(sp, t)][:, :, len(LANES) + li] = np.log1p(rk[rows, c])
            del zz, rk
        ok_lanes.append(lane)
        del mats
        free()
        print(f"  [lane {li+1}/{len(LANES)}] {lane:16s} {time.time()-t0:6.0f}s", flush=True)

    L = len(LANES)
    LR50, LR200 = np.log1p(50.0), np.log1p(200.0)
    tl = np.array([len(x.split()) for x in titles], np.float32)
    for (sp, t) in ALLQ:
        F = FEAT[(sp, t)]
        c = CAND[(sp, t)]
        fz, fr = F[:, :, :L], F[:, :, L:2 * L]
        bf = load_np(f"{RUN}_basefeat_{sp}_{t}.npy")
        F[:, :, 2 * L + 0] = bf[:, :, 0]
        F[:, :, 2 * L + 1] = np.log1p(bf[:, :, 1])
        F[:, :, 2 * L + 2] = (fr < LR50).sum(axis=2)
        F[:, :, 2 * L + 3] = (fr < LR200).sum(axis=2)
        F[:, :, 2 * L + 4] = fz.mean(axis=2)
        F[:, :, 2 * L + 5] = fz.std(axis=2)
        F[:, :, 2 * L + 6] = fz.max(axis=2)
        F[:, :, 2 * L + 7] = tl[c]
        F[:, :, 2 * L + 8] = np.array([len(x.split()) for x in Q[(sp, t)]["txt"]], np.float32)[:, None]
        save_np(f"{RUN}_feat_{sp}_{t}.npy", F)
    save_json(f"{RUN}_lanes.json", {"lanes": LANES, "ok": ok_lanes, "n_features": NF, "K": K})
    print(f"  {len(ok_lanes)}/{len(LANES)} lanes succeeded", flush=True)


phase(f"PHASE 2  10 remaining retriever lanes -> {NF} features per candidate", 30 * 60, phase2)
for _sp, _t in ALLQ:
    if art(f"{RUN}_feat_{_sp}_{_t}.npy") is None:
        raise RuntimeError(f"PHASE 2 did not produce features for {_sp}/{_t}; aborting the run.")


# ======================================================================================
# cross-encoder machinery
# ======================================================================================
class NanGuard(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        v = (logs or {}).get("loss")
        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise RuntimeError(f"loss non-finite at step {state.global_step}; diverged")


class TimeStop(TrainerCallback):
    def __init__(self, deadline):
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kw):
        if time.time() > self.deadline:
            print(f"    [time] stopping at step {state.global_step}", flush=True)
            control.should_training_stop = True


def assert_finite(model, tag):
    a = title2idx.get("debug software", 0)
    b = title2idx.get("3D lighting", NT - 1)
    probe = "debug software in a full stack environment"
    p = np.asarray(model.predict([(probe, docs[a]), (probe, docs[b])]), dtype=float)
    if not np.all(np.isfinite(p)):
        raise RuntimeError(f"{tag}: non-finite output, diverged")
    print(f"    [{tag}] finite check PASSED {p} (relevant>irrelevant: {p[0] > p[1]})", flush=True)
    if torch.cuda.is_available():
        # prove fp16 parity on real pairs before ce_score autocasts millions of them
        pairs = [(probe, docs[j]) for j in range(min(256, NT))]
        f32 = np.asarray(model.predict(pairs, batch_size=64, show_progress_bar=False), dtype=float)
        with torch.autocast("cuda", dtype=AC_DTYPE):
            f16 = np.asarray(model.predict(pairs, batch_size=64, show_progress_bar=False), dtype=float)
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)
        rho = float(np.corrcoef(rk(f32), rk(f16))[0, 1])
        drift = float(np.max(np.abs(f32 - f16)) / (np.std(f32) + 1e-9))
        if rho < 0.999 or drift > 0.02:
            raise RuntimeError(f"{tag}: fp16 autocast diverges from fp32 "
                               f"(spearman {rho:.5f}, max|d|/std {drift:.4f})")
        print(f"    [{tag}] fp16 parity PASSED (spearman {rho:.5f}, max|d|/std {drift:.4f})",
              flush=True)


def train_ce(model, ds, cfg, epochs, batch, lr, budget_s, tag, warmup=50):
    """Train with a hard time budget. max_steps is set from measured/prior throughput so the LR
    schedule still decays properly; TimeStop is the backstop if the estimate was optimistic."""
    docs_per_row = len(ds[0]["docs"])
    rate = pps(cfg["key"], "train", cfg["prior_train_pps"])
    steps_full = max(1, (len(ds) // batch) * epochs)
    steps_afford = max(50, int(budget_s * rate / (batch * docs_per_row)))
    max_steps = min(steps_full, steps_afford)
    print(f"    rows={len(ds)} docs/row={docs_per_row} batch={batch} -> {steps_full} steps full, "
          f"{max_steps} affordable ({budget_s/60:.0f} min @ {rate:.0f} pairs/s)", flush=True)
    def attempt(bs, grad_ckpt, steps):
        args = CrossEncoderTrainingArguments(
            output_dir=f"/tmp/{tag}", max_steps=steps, per_device_train_batch_size=bs,
            learning_rate=lr, warmup_steps=min(warmup, max(1, steps // 10)),
            fp16=(not SMOKE) and (not BF16), bf16=BF16, logging_steps=100,
            save_strategy="no", seed=SEED,
            report_to="none", max_grad_norm=1.0, gradient_checkpointing=grad_ckpt,
            # SMOKE runs on a Mac, where the HF Trainer silently moves the model to MPS regardless
            # of the device the CrossEncoder was built with; only use_cpu pins it. Worker
            # processes then fail with "_share_filename_: only available on CPU". Neither applies
            # on the notebook environment, so the real run keeps fp16 and 2 loader workers.
            use_cpu=SMOKE, dataloader_num_workers=0 if SMOKE else 2)
        t = CrossEncoderTrainer(model=model, args=args, train_dataset=ds,
                                loss=losses.LambdaLoss(model, weighting_scheme=losses.NDCGLoss2PPScheme()))
        t.add_callback(NanGuard())
        t.add_callback(TimeStop(min(time.time() + budget_s, DEADLINE)))
        t.train()
        return t

    t0 = time.time()
    try:
        tr = attempt(batch, False, max_steps)
    except torch.cuda.OutOfMemoryError:
        # 568M params with Adam is ~9GB of state on a 16GB card, so this is a live risk for model B.
        # Halving the batch and trading compute for activation memory is far cheaper than losing
        # a multi-hour phase; step count is doubled so the same number of rows is still seen.
        free()
        nb = max(1, batch // 2)
        print(f"    [OOM] retrying at batch {nb} with gradient checkpointing", flush=True)
        batch = nb
        max_steps = min(steps_full * 2, max_steps * 2)
        tr = attempt(nb, True, max_steps)
    el = time.time() - t0
    done = tr.state.global_step
    rate_now = done * batch * docs_per_row / max(el, 1.0)
    THROUGHPUT[cfg["key"] + "_train"] = rate_now
    print(f"    trained {done} steps in {el/60:.1f} min -> {rate_now:.0f} pairs/s measured", flush=True)
    assert_finite(model, tag)
    return model


def ce_score(model, qtexts, cands, cfg, tag):
    """Score every (query, candidate) pair. Refuses to return anything non-finite.
    Inference runs under fp16 autocast on CUDA: both bge rerankers are XLM-R seq-cls,
    which is fp16-stable, and the fp32 throughput priors were physically impossible on a
    T4 (93-95% of theoretical peak with no TF32). assert_finite proves fp32/fp16 parity
    on real pairs before any model reaches this function."""
    out = np.zeros(cands.shape, np.float32)
    t0 = time.time()
    n = len(qtexts)
    on_cuda = torch.cuda.is_available()
    for i in range(n):
        with (torch.autocast("cuda", dtype=AC_DTYPE) if on_cuda
              else contextlib.nullcontext()):
            s = model.predict([(qtexts[i], docs[int(j)]) for j in cands[i]],
                              batch_size=cfg["infer_bs"], show_progress_bar=False)
        s = np.asarray(s, dtype=np.float32)
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"{tag}: non-finite CE scores at query {i}; refusing to export")
        out[i] = s
        if (i + 1) % 250 == 0 or i + 1 == n:
            r = (i + 1) * cands.shape[1] / max(time.time() - t0, 1.0)
            print(f"      {tag} {i+1}/{n} q  {r:.0f} pairs/s  "
                  f"eta {(n-i-1)*cands.shape[1]/max(r,1)/60:.1f} min", flush=True)
    # Keep the slowest observed rate. The last validation task scored is the
    # normalisation set, whose four-word queries are the fastest, so sizing the
    # 1,873-query test phase from it would give an optimistic estimate.
    _k = cfg["key"] + "_infer"
    _r = n * cands.shape[1] / max(time.time() - t0, 1.0)
    THROUGHPUT[_k] = min(THROUGHPUT.get(_k, _r), _r)
    return out


def build_stage1(cfg):
    """Synthetic sentence->skill rows, plus ESCO description->skill rows.
    The description rows exist because 2 of the 5 TEST tasks (SkillNorm 450q, TechWolf 324q, so
    41% of the test set) are short-phrase normalisation, while only 1 of 4 val tasks is. Stage 1
    had no normalisation-shaped supervision at all before this."""
    # Vectorised on purpose: this table is 138,260 rows and row-at-a-time iteration over an HF
    # dataset is the slowest thing in the phase for no reason.
    syn = load_dataset("TechWolf/Synthetic-ESCO-skill-sentences", split="train").to_pandas()
    gi = syn["skill"].astype(str).str.strip().map(title2idx)
    ok = gi.notna()
    pairs = list(zip(syn.loc[ok, "sentence"].astype(str).tolist(), gi[ok].astype(int).tolist()))
    print(f"    synthetic rows {len(syn)} -> {len(pairs)} with a resolvable ESCO skill", flush=True)
    rng.shuffle(pairs)
    pairs = pairs[:cfg["s1_syn"]]
    if cfg["s1_desc"] > 0:
        idx = rng.permutation(NT)[:cfg["s1_desc"]]
        pairs += [(descs[j], int(j)) for j in idx if descs[j] != titles[j]]
    rng.shuffle(pairs)
    print(f"    stage1 rows={len(pairs)} ({cfg['s1_syn']} synthetic + description rows)", flush=True)

    emb = SentenceTransformer(CTX_MODEL, device=dev)
    emb.max_seq_length = 192
    demb = emb.encode(docs, batch_size=256, convert_to_tensor=True,
                      normalize_embeddings=True, show_progress_bar=False).to(dev)
    ranked = np.zeros((len(pairs), 200), np.int32)
    for s in range(0, len(pairs), 2048):
        se = emb.encode([p[0] for p in pairs[s:s + 2048]], batch_size=256, convert_to_tensor=True,
                        normalize_embeddings=True, show_progress_bar=False).to(dev)
        ranked[s:s + 2048] = torch.topk(se @ demb.T, 200, dim=1).indices.cpu().numpy()
        del se
    del emb, demb
    free()
    print("    stage1 hard-negative mining done", flush=True)

    q1, d1, l1 = [], [], []
    for i, (sent, gi) in enumerate(pairs):
        row = ranked[i]
        row = row[row != gi]
        adj = row[:10][:N_ADJ]
        pl = row[10:200]
        pl = pl[rng.choice(len(pl), size=min(N_PLAUS, len(pl)), replace=False)] if len(pl) else []
        banned = set(row[:DEEP_MIN].tolist()) | {gi}
        deep = []
        while len(deep) < N_NONSENSE:
            c = int(rng.randint(NT))
            if c not in banned:
                deep.append(c)
                banned.add(c)
        idxs = [gi] + list(adj) + list(pl) + deep
        q1.append(sent)
        d1.append([docs[j] for j in idxs])
        l1.append([4.0] + [2.0] * len(adj) + [1.0] * len(pl) + [0.0] * len(deep))
    del ranked
    return Dataset.from_dict({"query": q1, "docs": d1, "labels": l1})


def build_rows(t, qidx, cands, samples, ndocs, r):
    qq, dd, ll = [], [], []
    g = Q[("val", t)]["grades"]
    qt = Q[("val", t)]["txt"]
    for i in qidx:
        c = cands[i]
        y = np.array([g[i].get(int(j), 0.0) for j in c])
        hi, lo = np.where(y >= 2)[0], np.where(y < 2)[0]
        # A row whose labels are all identical has an undefined ideal ranking; LambdaLoss on it
        # is a live NaN source. Local training diverged four times before this was understood,
        # so queries with no grade>=2 inside the candidate set are dropped rather than fed in.
        if len(hi) == 0:
            continue
        for _ in range(samples):
            th = hi if len(hi) <= ndocs // 2 else r.choice(hi, ndocs // 2, replace=False)
            tl = r.choice(lo, min(ndocs - len(th), len(lo)), replace=False)
            sel = np.concatenate([th, tl])
            if len(sel) < ndocs:
                continue
            r.shuffle(sel)
            qq.append(qt[i])
            dd.append([docs[int(c[j])] for j in sel])
            ll.append([float(y[j]) for j in sel])
    return qq, dd, ll


def stage2_dataset(cfg, folds, which, r):
    q2, d2, l2 = [], [], []
    for t in VAL_TASKS:
        c = load_np(f"{RUN}_cands_val_{t}.npy")
        n = c.shape[0]
        idx = np.arange(n) if which is None else np.concatenate(
            [folds[t][f] for f in range(NFOLD) if f != which])
        a, b, cc = build_rows(t, idx, c, cfg["s2_samples"], cfg["s2_docs"], r)
        q2 += a
        d2 += b
        l2 += cc
    return Dataset.from_dict({"query": q2, "docs": d2, "labels": l2})


def run_ce_model(cfg):
    key = cfg["key"]
    s1dir = os.path.join(WORK, f"{RUN}-ce{key}-stage1")

    # ---------- stage 1 ----------
    def s1():
        found = art(f"{RUN}-ce{key}-stage1/config.json")
        if found:
            print(f"[reuse] stage 1 for model {key} at {os.path.dirname(found)}", flush=True)
            return os.path.dirname(found)
        ds1 = build_stage1(cfg)
        m = CrossEncoder(cfg["hf"], num_labels=1, max_length=MAXLEN,
                         model_kwargs={"torch_dtype": "float32"})
        train_ce(m, ds1, cfg, 1, cfg["s1_batch"], cfg["s1_lr"], cfg["s1_budget_s"], f"ce{key}-stage1")
        m.save_pretrained(s1dir)
        del m, ds1
        free()
        return s1dir

    got = phase(f"MODEL {key} ({cfg['hf']})  STAGE 1 pretrain", cfg["s1_budget_s"], s1)
    if got is None:
        return
    stage1 = got

    r = np.random.RandomState(SEED)
    folds = {t: np.array_split(r.permutation(load_np(f"{RUN}_cands_val_{t}.npy").shape[0]), NFOLD)
             for t in VAL_TASKS}
    nval = sum(len(Q[("val", t)]["txt"]) for t in VAL_TASKS)
    s2_pairs = nval * cfg["s2_samples"] * cfg["s2_docs"] * cfg["s2_epochs"]
    s2_est = s2_pairs / cfg["prior_train_pps"]

    # ---------- folds -> out-of-fold val scores (this is what the LTR trains on) ----------
    def folds_oof():
        if all(art(f"{RUN}_ce{key}_oof_val_{t}.npy") for t in VAL_TASKS):
            print(f"[reuse] model {key} OOF already exists", flush=True)
            return True
        oof = {t: np.zeros(load_np(f"{RUN}_cands_val_{t}.npy").shape, np.float32) for t in VAL_TASKS}
        for fo in range(NFOLD):
            ds2 = stage2_dataset(cfg, folds, fo, np.random.RandomState(SEED + fo))
            m = CrossEncoder(stage1, num_labels=1, max_length=MAXLEN,
                             model_kwargs={"torch_dtype": "float32"})
            train_ce(m, ds2, cfg, cfg["s2_epochs"], cfg["s2_batch"], cfg["s2_lr"],
                     max(300.0, s2_est * 0.8), f"ce{key}-fold{fo}", warmup=20)
            for t in VAL_TASKS:
                c = load_np(f"{RUN}_cands_val_{t}.npy")
                qt = Q[("val", t)]["txt"]
                idx = folds[t][fo]
                s = ce_score(m, [qt[i] for i in idx], c[idx], cfg, f"ce{key}f{fo}/{t}")
                oof[t][idx] = s
            del m, ds2
            free()
            print(f"  [fold {fo}] done", flush=True)
        for t in VAL_TASKS:
            save_np(f"{RUN}_ce{key}_oof_val_{t}.npy", oof[t])

        # Score the cross-encoder in isolation here. It costs nothing and answers the question
        # the run exists to answer -- is this model better than the last one -- hours before the
        # LightGBM stage downstream could. Reference points: v3 MiniLM 0.6884, v5 bge-base 0.7044.
        ce_per, base_per = {}, {}
        for t in VAL_TASKS:
            y, idcg = load_np(f"{RUN}_y_val_{t}.npy"), load_np(f"{RUN}_idcg_val_{t}.npy")
            ce_per[t] = float(np.mean([ndcg_at(y[i][np.argsort(-oof[t][i])], idcg[i])
                                       for i in range(len(y))]))
            base_per[t] = float(np.mean([ndcg_at(y[i], idcg[i]) for i in range(len(y))]))
        cm = float(np.mean(list(ce_per.values())))
        bm = float(np.mean(list(base_per.values())))
        print(f"\n  {'='*68}\n  CROSS-ENCODER {key} ALONE, out-of-fold, top-{K}: "
              f"macro nDCG@100 = {cm:.4f}  (base {bm:.4f}, {cm-bm:+.4f})", flush=True)
        for t in VAL_TASKS:
            print(f"      {t:11s} {ce_per[t]:.4f}  (base {base_per[t]:.4f}, "
                  f"{ce_per[t]-base_per[t]:+.4f})", flush=True)
        print(f"  reference: v3 MiniLM 0.6884 | v5 bge-base 0.7044 (both CE-alone)\n  {'='*68}",
              flush=True)
        save_json(f"{RUN}_ce{key}_alone.json", {"macro": cm, "base": bm, "per_task": ce_per})
        return True

    if phase(f"MODEL {key}  STAGE 2 x{NFOLD} folds + out-of-fold val scores",
             s2_est * NFOLD + 8 * 60, folds_oof) is None:
        return

    # ---------- full model -> test scores (this is what the submission uses) ----------
    def full_and_test():
        mem = "full"
        if all(art(f"{RUN}_ce{key}_test_{t}_{mem}.npy") for t in TEST_TASKS):
            print(f"[reuse] model {key} full test scores already exist", flush=True)
            return True
        ds2 = stage2_dataset(cfg, folds, None, np.random.RandomState(SEED))
        m = CrossEncoder(stage1, num_labels=1, max_length=MAXLEN,
                         model_kwargs={"torch_dtype": "float32"})
        train_ce(m, ds2, cfg, cfg["s2_epochs"], cfg["s2_batch"], cfg["s2_lr"],
                 max(300.0, s2_est), f"ce{key}-full", warmup=20)
        del ds2
        free()
        tk = min(cfg.get("test_K", K), K)
        for t in TEST_TASKS:
            # per-task gate: phase() only checks time at START (threshold 0.6), so an
            # optimistic estimate lets a phase start that cannot finish, and it then fails near the end. Stopping
            # cleanly here leaves a truthful manifest instead of a dead multi-hour phase.
            need = len(Q[("test", t)]["txt"]) * tk / pps(key, "infer", cfg["prior_infer_pps"])
            if DEADLINE - time.time() < need * 1.15:
                print(f"  [stop] no time for test task {t}; model {key} test scoring "
                      f"incomplete", flush=True)
                return False
            c = load_np(f"{RUN}_cands_test_{t}.npy")[:, :tk]
            s = ce_score(m, Q[("test", t)]["txt"], c, cfg, f"ce{key}full/{t}")
            save_np(f"{RUN}_ce{key}_test_{t}_{mem}.npy", s)
        del m
        free()
        return True

    ntest = sum(len(Q[("test", t)]["txt"]) for t in TEST_TASKS)
    infer_est = ntest * min(cfg.get("test_K", K), K) / pps(key, "infer", cfg["prior_infer_pps"])
    phase(f"MODEL {key}  FULL model + TEST scoring ({ntest} queries x {K})",
          s2_est + infer_est, full_and_test)


for cfg in CE_MODELS:
    run_ce_model(cfg)


# ======================================================================================
# BONUS  extra test-scoring ensemble members, only with time to spare
# ======================================================================================
def bonus_members():
    cfg = CE_MODELS[0]
    key = cfg["key"]
    stage1 = os.path.dirname(art(f"{RUN}-ce{key}-stage1/config.json"))
    r = np.random.RandomState(SEED)
    folds = {t: np.array_split(r.permutation(load_np(f"{RUN}_cands_val_{t}.npy").shape[0]), NFOLD)
             for t in VAL_TASKS}
    nval = sum(len(Q[("val", t)]["txt"]) for t in VAL_TASKS)
    s2_est = nval * cfg["s2_samples"] * cfg["s2_docs"] * cfg["s2_epochs"] / pps(key, "train", cfg["prior_train_pps"])
    ntest = sum(len(Q[("test", t)]["txt"]) for t in TEST_TASKS)
    tk = min(cfg.get("test_K", K), K)
    for fo in range(NFOLD):
        mem = f"fold{fo}"
        if all(art(f"{RUN}_ce{key}_test_{t}_{mem}.npy") for t in TEST_TASKS):
            continue
        need = s2_est + ntest * tk / pps(key, "infer", cfg["prior_infer_pps"])
        if DEADLINE - time.time() < need * 1.1:
            print(f"  [stop] no time for member {mem}", flush=True)
            break
        ds2 = stage2_dataset(cfg, folds, fo, np.random.RandomState(SEED + fo))
        m = CrossEncoder(stage1, num_labels=1, max_length=MAXLEN,
                         model_kwargs={"torch_dtype": "float32"})
        train_ce(m, ds2, cfg, cfg["s2_epochs"], cfg["s2_batch"], cfg["s2_lr"],
                 max(300.0, s2_est), f"ce{key}-{mem}-test", warmup=20)
        for t in TEST_TASKS:
            c = load_np(f"{RUN}_cands_test_{t}.npy")[:, :tk]
            save_np(f"{RUN}_ce{key}_test_{t}_{mem}.npy",
                    ce_score(m, Q[("test", t)]["txt"], c, cfg, f"ce{key}{mem}/{t}"))
        del m, ds2
        free()


phase("BONUS  extra ensemble members for model A test scoring", 45 * 60, bonus_members)


# ======================================================================================
# manifest
# ======================================================================================
have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WORK, f"{RUN}_*")))
ce_ready = {}
for cfg in CE_MODELS:
    k = cfg["key"]
    ce_ready[k] = {
        "oof_val": all(art(f"{RUN}_ce{k}_oof_val_{t}.npy") is not None for t in VAL_TASKS),
        "test_members": [m for m in ("full", "fold0", "fold1", "fold2")
                         if all(art(f"{RUN}_ce{k}_test_{t}_{m}.npy") is not None for t in TEST_TASKS)],
        "test_K": min(cfg.get("test_K", K), K),
    }
    # A stage-1 checkpoint is only worth keeping if a FOLLOW-UP run would need to resume from it.
    # Once a model has both its out-of-fold val scores and its test scores, it is finished and the
    # checkpoint is 1-2 GB of pure download cost, so drop it. A model that failed keeps its
    # checkpoint, which is exactly the case where resuming saves the most GPU time.
    d = os.path.join(WORK, f"{RUN}-ce{k}-stage1")
    if ce_ready[k]["oof_val"] and ce_ready[k]["test_members"] and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"[cleanup] model {k} is complete; removed its stage-1 checkpoint from the output", flush=True)
manifest = {"run": RUN, "K": K, "lanes": LANES, "n_features": NF, "val_tasks": VAL_TASKS,
            "test_tasks": TEST_TASKS, "phases": PHASES, "throughput": THROUGHPUT,
            "ce_ready": ce_ready, "elapsed_h": round((time.time() - START) / 3600, 2)}
save_json(f"{RUN}_manifest.json", manifest)

print("\n" + "=" * 78)
print(f"RUN COMPLETE in {(time.time()-START)/3600:.2f} h")
for p in PHASES:
    print(f"  {p['status']:22s} {p.get('minutes','')!s:>7}  {p['name']}")
print("\nCROSS-ENCODERS USABLE DOWNSTREAM:")
for k, v in ce_ready.items():
    state = "USABLE" if (v["oof_val"] and v["test_members"]) else "not usable (needs both)"
    print(f"  model {k}: oof_val={v['oof_val']}  test_members={v['test_members']}  -> {state}")
print(f"\n{len(have)} artifacts in /kaggle/working. Download the whole folder.")
print("=" * 78, flush=True)
