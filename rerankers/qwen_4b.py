#!/usr/bin/env python
"""Cross-encoder trainer and scorer built on Qwen3-Reranker-4B.

The script rebuilds the stage-1 and stage-2 datasets with the same code and
parameters as the main pipeline, reads the retrieval artefacts produced by that
pipeline, and never recomputes retrieval itself. It emits out-of-fold validation
scores over three query-grouped folds, full-fit test scores for every task, and a
manifest recording the run.

Loading path. Qwen3-Reranker is a causal language model with no classifier head.
Loading it through AutoModelForSequenceClassification attaches a randomly
initialised head and discards the reranking behaviour the checkpoint already
has. This script loads AutoModelForCausalLM instead and scores with the
converted-head identity:

    score = h_last @ (lm_head.weight[yes] - lm_head.weight[no])

which equals the difference between the two logits at the final position,
verified to 2.4e-8 against evaluating the full logit vector. Applying a sigmoid
to that score reproduces the model card's binary probability for "yes" exactly,
and the full-vocabulary logits are never materialised. The published input
template is used verbatim, with left padding so that the final position holds
real content. Loading information from from_pretrained is checked for
non-tied missing keys to catch a silent re-initialisation, and every checkpoint
load refuses unexpected keys and refuses to mix LoRA and full fine-tune state.

Loss. A port of the LambdaLoss implementation in sentence-transformers under the
NDCGLoss2++ scheme (mu=10, k=None, sigma=1.0, eps=1e-10, binary reduction log,
averaged over valid pairs), applied to per-row document groups of yes-minus-no
scores. Verified bit-identical in both values and gradients against the
installed sentence-transformers implementation, and to 6e-8 against the
reference allRank implementation.

Resumption. Re-running the same command resumes at every granularity: artefact
level skips, step level training checkpoints, per-fold reuse of trained state so
a crash between training and scoring does not force a retrain, and partial score
files within a test task.

Environment variables:

    WORKRB_WORK            artefact directory holding the pipeline's retrieval
                           and feature outputs. Required in practice; the script
                           refuses to run without them.
    WORKRB_ART_DIRS        colon-separated additional read-only artefact dirs
    WORKRB_DEADLINE_H      wall-clock budget in hours (default 24.0)
    WORKRB_SMOKE=1         substitute a tiny random causal model and run on CPU
                           in minutes. Requires a prior smoke run of the main
                           pipeline in the same work directory.
    WORKRB_S2_SAMPLES      stage-2 rows sampled per query
    WORKRB_S2_DOCS         documents per stage-2 row
    WORKRB_Q_FULL_FT=1     full fine-tune instead of LoRA
    WORKRB_Q_LORA_R        LoRA rank (default 16, alpha = 2r)
    WORKRB_Q_S1_LR         stage-1 learning rate (default 1e-4 LoRA, 6e-6 full)
    WORKRB_Q_S2_LR         stage-2 learning rate (default 5e-5 LoRA, 3e-6 full)
    WORKRB_Q_S1_BUDGET_MIN stage-1 budget in minutes (default 240)
    WORKRB_Q_MICRO_ROWS    rows per forward pass (default 2)
    WORKRB_Q_ACCUM         gradient accumulation steps (default 4)
    WORKRB_Q_INFER_BS      inference pair batch, halved automatically on OOM
                           (default 64)
    WORKRB_Q_TRAIN_PPS     training throughput prior in pairs per second
    WORKRB_Q_INFER_PPS     inference throughput prior, deliberately conservative
                           until measured
    WORKRB_Q_CKPT_EVERY    checkpoint every N optimizer steps (default 250)
    WORKRB_Q_PARITY_RHO    bf16 parity gate on rank correlation (default 0.999)
    WORKRB_Q_PARITY_DRIFT  bf16 parity gate on score drift (default 0.05). Drift
                           between 0.02 and 0.05 passes with a warning: bf16 has
                           eight mantissa bits against fp16's eleven, so rank
                           agreement is the binding check.
    WORKRB_Q_INSTRUCT      override the fixed English instruction string
"""

print("WorkRB v6 model Q  |  Qwen3-Reranker-4B  |  OOF val + full test  |  key 'Q'")
print("If you do NOT see this banner you are running an old file.")
print("=" * 78, flush=True)

import os, math, glob, time, json, re, gc, traceback, contextlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # single GPU on purpose (v6 rule);
                                                     # setdefault: Slurm may have set it
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RUN          = "v6"
CE_KEY       = "Q"
WORK         = os.environ.get("WORKRB_WORK", "/kaggle/working")
os.makedirs(WORK, exist_ok=True)
ART_DIRS     = [d for d in os.environ.get("WORKRB_ART_DIRS", "").split(":") if d]
ESCO_REPO    = "TechWolf/Skill-extraction-Tech-graded"
CTX_MODEL    = "TechWolf/ConTeXT-Skill-Extraction-base"   # stage-1 hard-negative miner (v6)

NFOLD        = 3
SEED         = 0
MAXLEN       = 192          # v6 pair-token budget (query+doc)
TOTAL_LEN    = MAXLEN + 64  # + ~45 template tokens (prefix/suffix/<Instruct> scaffold)
DEADLINE_H   = float(os.environ.get("WORKRB_DEADLINE_H", "24.0"))

# Ladder for stage-1 synthetic rows (v6): 1 gold + 3 near + 5 mid + 6 deep = 15 docs/row.
N_ADJ, N_PLAUS, N_NONSENSE, DEEP_MIN = 3, 5, 6, 2000

TASKS = {
    "Tech":       ("TechWolf/Skill-extraction-Tech-graded",       ["val", "test"]),
    "House":      ("TechWolf/Skill-extraction-House-graded",      ["val", "test"]),
    "SkillSkape": ("TechWolf/Skill-extraction-SkillSkape-graded", ["val", "test"]),
    "SkillNorm":  ("TechWolf/Skill-normalisation-ESCO-graded",    ["val", "test"]),
    "TechWolf":   ("TechWolf/Skill-extraction-TechWolf-graded",   ["test"]),  # TEST ONLY
}
VAL_TASKS  = [t for t, (_, sp) in TASKS.items() if "val" in sp]
TEST_TASKS = list(TASKS.keys())
HF_SPLIT   = {"val": "validation", "test": "test"}

SMOKE = os.environ.get("WORKRB_SMOKE") == "1"
SMOKE_NQ, SMOKE_NT = 8, 700          # MUST match the main pipeline for artifact parity

FULL_FT   = os.environ.get("WORKRB_Q_FULL_FT") == "1"
LORA_R    = int(os.environ.get("WORKRB_Q_LORA_R", "16"))
S1_LR     = float(os.environ.get("WORKRB_Q_S1_LR", "6e-6" if FULL_FT else "1e-4"))
S2_LR     = float(os.environ.get("WORKRB_Q_S2_LR", "3e-6" if FULL_FT else "5e-5"))
S1_BUDGET = float(os.environ.get("WORKRB_Q_S1_BUDGET_MIN", "240")) * 60
MICRO     = int(os.environ.get("WORKRB_Q_MICRO_ROWS", "2"))
ACCUM     = int(os.environ.get("WORKRB_Q_ACCUM", "4"))
INFER_BS  = int(os.environ.get("WORKRB_Q_INFER_BS", "64"))
CKPT_EVERY = int(os.environ.get("WORKRB_Q_CKPT_EVERY", "250"))
PARITY_RHO   = float(os.environ.get("WORKRB_Q_PARITY_RHO", "0.999"))
PARITY_DRIFT = float(os.environ.get("WORKRB_Q_PARITY_DRIFT", "0.05"))
V6_FP16_DRIFT = 0.02                 # v6's fp16 bound; drift above it warns, not fails

CFG = dict(key=CE_KEY, hf="Qwen/Qwen3-Reranker-4B",
           s1_syn=80000, s1_desc=14000,                     # model-A/C sized stage 1
           s2_samples=16, s2_docs=24, s2_epochs=2,          # model-A graded density
           prior_train_pps=float(os.environ.get("WORKRB_Q_TRAIN_PPS", "40.0")),
           # conservative infer prior until a measured rate replaces it ():
           prior_infer_pps=float(os.environ.get("WORKRB_Q_INFER_PPS", "150.0")),
           test_K=1000, test_members=("full",))
if os.environ.get("WORKRB_S2_SAMPLES"):
    CFG["s2_samples"] = int(os.environ["WORKRB_S2_SAMPLES"])
if os.environ.get("WORKRB_S2_DOCS"):
    CFG["s2_docs"] = int(os.environ["WORKRB_S2_DOCS"])

if SMOKE:
    DEADLINE_H = 0.5
    S1_BUDGET, MICRO, ACCUM, INFER_BS = 90.0, 2, 1, 8
    S1_LR, S2_LR = 1e-4, 5e-5
    # trl-internal-testing id: the hf-internal-testing one 404s, and this one is the
    # same Qwen3 architecture as the real reranker (q/k norm included)
    CFG.update(hf="trl-internal-testing/tiny-Qwen3ForCausalLM",
               s1_syn=120, s1_desc=40, s2_samples=2, s2_docs=8, s2_epochs=1,
               prior_train_pps=40.0, prior_infer_pps=120.0)
    print(">>> SMOKE MODE: tiny stand-in causal LM, CPU, subsampled artifacts <<<", flush=True)

import numpy as np, torch
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer          # stage-1 mining only
from transformers import AutoTokenizer, AutoModelForCausalLM

START    = time.time()
DEADLINE = START + DEADLINE_H * 3600
rng = np.random.RandomState(SEED)          # consumed ONLY by build_stage1, like v6
torch.manual_seed(SEED)
if SMOKE:
    torch.set_num_threads(4)               # local memory limit: cap CPU threads
AC_DTYPE = torch.bfloat16                  # bf16 always: ms-swift trains this model in
                                           # bf16; the fp16+FA2 model-card note is
                                           # inference-only. WORKRB_BF16 is irrelevant.
dev = "cpu" if SMOKE else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}", flush=True)
print(f"deadline: {DEADLINE_H:.1f}h | mode: {'FULL FT' if FULL_FT else f'LoRA r={LORA_R}'}"
      f" | s1_lr {S1_LR:g} s2_lr {S2_LR:g} | micro {MICRO} x accum {ACCUM}\n", flush=True)


# --------------------------------------------------------------------------------------
# artifact helpers (verbatim semantics from the main pipeline, + WORKRB_ART_DIRS)
# --------------------------------------------------------------------------------------
def art(fn):
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        return p
    for d in ART_DIRS:
        p = os.path.join(d, fn)
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


def free():
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


THROUGHPUT = {}


def pps(model_key, kind, prior):
    return THROUGHPUT.get(f"{model_key}_{kind}", prior)


PHASES = []


def phase(name, est_s, fn):
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


_BULLET = re.compile(r"^\s*[\*•·\-–—]+\s+")


def clean_q(s):
    s = str(s).strip()
    out = _BULLET.sub("", s).strip()
    return out if len(out) >= 2 else s


DISC = None   # set after K is known (read from the candidate artifacts)


def ndcg_at(y_ordered, idcg):
    g = (2.0 ** np.asarray(y_ordered[:100], dtype=np.float64)) - 1.0
    return float(np.sum(g * DISC[:len(g)]) / idcg) if idcg > 0 else 0.0


# ======================================================================================
# PREFLIGHT: this script consumes phase-1/2 artifacts; it never recomputes retrieval
# ======================================================================================
ALLQ_STATIC = [(sp, t) for t in TASKS for sp in TASKS[t][1]]
_need = [f"{RUN}_corpus_ids.json", f"{RUN}_textmode.json"]
_need += [f"{RUN}_qids_{sp}_{t}.json" for (sp, t) in ALLQ_STATIC]
_need += [f"{RUN}_cands_{sp}_{t}.npy" for (sp, t) in ALLQ_STATIC]
_need += [f"{RUN}_y_val_{t}.npy" for t in VAL_TASKS]
_need += [f"{RUN}_idcg_val_{t}.npy" for t in VAL_TASKS]
_missing = [f for f in _need if art(f) is None]
if _missing:
    raise RuntimeError(
        f"{len(_missing)} required v6 artifacts missing from WORK={WORK} "
        f"(first few: {_missing[:5]}). Run the main pipeline phases 1-2 first "
        f"(or point WORKRB_WORK / WORKRB_ART_DIRS at the directory holding its outputs).")
print(f"[preflight] all {len(_need)} phase-1/2 artifacts present", flush=True)


# ======================================================================================
# PHASE 0  corpus + query sets  (verbatim construction from the main pipeline, then
#          VERIFIED against the exported v6_corpus_ids/v6_qids artifacts -- the row-order
#          gate that caught the real v5 bug. No json artifacts are rewritten here.)
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

Q = {}
for tname, (repo, splits) in TASKS.items():
    for sp in splits:
        qdf = load_dataset(repo, "queries", split=HF_SPLIT[sp]).to_pandas()
        qid2text = {str(a): str(b) for a, b in zip(qdf["_id"], qdf["text"])}
        grades = None
        if sp == "val":
            qr = load_dataset(repo, "qrels", split="validation").to_pandas()
            qr = qr[qr["score"] > 0]
            g = {}
            for qi, ci, sc in zip(qr["query-id"].astype(str), qr["corpus-id"].astype(str), qr["score"]):
                j = uri2idx.get(ci)
                if j is not None:
                    g.setdefault(qi, {})[j] = float(sc)
            keys = [k for k in qid2text if k in g]
            grades = [g[k] for k in keys]
        else:
            keys = list(qid2text)
        raw = [qid2text[k].strip() for k in keys]
        Q[(sp, tname)] = {"qids": keys, "raw": raw, "txt": [clean_q(x) for x in raw], "grades": grades}
        print(f"  {sp:4s} {tname:11s} {len(keys):5d} queries", flush=True)

ALLQ = [(sp, t) for t in TASKS for sp in TASKS[t][1]]

if SMOKE:
    # Verbatim smoke subsample from the main pipeline so the K=40 smoke artifacts align.
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
    print(f">>> SMOKE: corpus {NT}, {SMOKE_NQ} queries per task-split", flush=True)

# ---- ROW-ORDER GATE: never re-derive query/corpus order without checking the json ----
_cj = json.load(open(art(f"{RUN}_corpus_ids.json")))
if _cj != corpus_ids:
    raise RuntimeError("corpus id order does not match v6_corpus_ids.json -- every int32 "
                       "candidate index would point at the wrong concept. Refusing.")
for (sp, t) in ALLQ:
    want = json.load(open(art(f"{RUN}_qids_{sp}_{t}.json")))
    if want != Q[(sp, t)]["qids"]:
        raise RuntimeError(f"query id order mismatch for {sp}/{t} vs {RUN}_qids_{sp}_{t}.json "
                           f"-- row alignment with the candidate and out-of-fold artefacts would be wrong.")
print("[verify] corpus order + every query row order match the v6 artifacts", flush=True)

# ---- TEXT MODE: read the measured phase-1 decision; use exactly that text ----
TEXT_MODE = json.load(open(art(f"{RUN}_textmode.json")))["mode"]
if TEXT_MODE == "raw":
    for (sp, t) in ALLQ:
        Q[(sp, t)]["txt"] = Q[(sp, t)]["raw"]
print(f"[text mode] '{TEXT_MODE}' query text everywhere (from {RUN}_textmode.json)", flush=True)

# ---- K from the exported candidates; nDCG discounts; known-value gate ----
K = int(load_np(f"{RUN}_cands_val_{VAL_TASKS[0]}.npy").shape[1])
for (sp, t) in ALLQ:
    w = int(load_np(f"{RUN}_cands_{sp}_{t}.npy").shape[1])
    if w != K:
        raise RuntimeError(f"candidate width mismatch: {sp}/{t} has {w}, expected {K}")
DISC = 1.0 / np.log2(np.arange(max(K, 100)) + 2)
print(f"[cands] K={K} on every split", flush=True)

if not SMOKE:
    # Base macro recomputed from exported y/idcg must reproduce the stored phase-1 value
    # (which itself passed the 0.6499 gate). Catches a stale or foreign artifact set.
    _tm = json.load(open(art(f"{RUN}_textmode.json")))
    _want = _tm["clean"] if TEXT_MODE == "clean" else _tm["raw"]
    per = {}
    for t in VAL_TASKS:
        y, idcg = load_np(f"{RUN}_y_val_{t}.npy"), load_np(f"{RUN}_idcg_val_{t}.npy")
        per[t] = float(np.mean([ndcg_at(y[i], idcg[i]) for i in range(len(y))]))
    _macro = float(np.mean([per[t] for t in VAL_TASKS]))
    print(f"[verify] base macro from exported y/idcg = {_macro:.4f} (stored {_want:.4f}, "
          f"known raw value 0.6499)", flush=True)
    if abs(_macro - _want) > 1e-4:
        raise RuntimeError(f"exported labels reproduce {_macro:.4f}, textmode.json says {_want:.4f}"
                           f" -- artifact set is misaligned. Refusing to continue.")


# ======================================================================================
# Qwen3-Reranker-4B machinery: template, scorer module, listwise loss, train loop
# ======================================================================================
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = os.environ.get(
    "WORKRB_Q_INSTRUCT",
    "Given a sentence from a job posting or a short skill phrase, judge whether the ESCO "
    "skill (title and description) matches the skill expressed in the query")
YES_NO_REFERENCE = (9693, 2152)      # published Qwen3-Reranker ids (, model card)

tok = AutoTokenizer.from_pretrained(CFG["hf"], padding_side="left")   # LEFT: score is read
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id  # at pos -1
PRE_IDS = tok(PREFIX, add_special_tokens=False)["input_ids"]
SUF_IDS = tok(SUFFIX, add_special_tokens=False)["input_ids"]
BODY_LEN = TOTAL_LEN - len(PRE_IDS) - len(SUF_IDS)
print(f"[template] prefix {len(PRE_IDS)} + body {BODY_LEN} + suffix {len(SUF_IDS)} tokens", flush=True)


def encode_pairs(pairs):
    bodies = [f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
    enc = tok(bodies, add_special_tokens=False, truncation=True, max_length=BODY_LEN)["input_ids"]
    return [PRE_IDS + e + SUF_IDS for e in enc]


def pad_left(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, L - len(s):] = 1
    return ids, mask


class QwenScorer(torch.nn.Module):
    """Causal-LM trunk + the converted yes-minus-no head as a single trainable vector.
    score = h_last @ w  with  w init = lm_head.weight[yes] - lm_head.weight[no], so
    sigmoid(score) is bit-equivalent to the model card's binary softmax P("yes").
    Only the trunk's last hidden state is projected: full-vocab logits are never built
    (the dominant activation cost in the memory arithmetic)."""

    def __init__(self, causal, w_vec):
        super().__init__()
        self.causal = causal                       # raw HF model or peft-wrapped
        self.w = torch.nn.Parameter(w_vec.float())

    def _trunk(self):
        m = self.causal
        if m.__class__.__name__.startswith("Peft"):
            m = m.base_model.model                 # PeftModel -> LoraModel -> causal LM
        return m.get_decoder()

    def forward(self, input_ids, attention_mask):
        h = self._trunk()(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state[:, -1, :]
        if h.is_cuda:                              # keep the head readout in true fp32
            with torch.autocast("cuda", enabled=False):
                return h.float() @ self.w
        return h.float() @ self.w


def _resolve_token(s, V):
    """Single-token id for s, with an encode fallback for tokenizers where the plain
    surface form is not a vocab token. None if unresolvable."""
    i = tok.convert_tokens_to_ids(s)
    if i is not None and i != tok.unk_token_id and 0 <= i < V:
        return int(i)
    e = tok(s, add_special_tokens=False)["input_ids"]
    if len(e) == 1 and 0 <= e[0] < V:
        return int(e[0])
    return None


def build_model():
    causal, load_info = AutoModelForCausalLM.from_pretrained(
        CFG["hf"], torch_dtype=torch.float32,      # fp32 master weights; bf16 via autocast
        attn_implementation="sdpa",                # (classic AMP -- makes the parity gate real)
        output_loading_info=True)
    # Silent-reinit guard (): any missing key that is not weight-tied means part
    # of the network was randomly initialized -- the exact bug class this design forbids.
    tied = set(getattr(causal, "_tied_weights_keys", None) or [])
    reinit = [k for k in load_info.get("missing_keys", []) if k not in tied]
    if reinit:
        msg = (f"from_pretrained reports non-tied missing keys {reinit[:5]} -- part of the "
               f"model would be RANDOM. Refusing (random-head bug class).")
        if SMOKE:
            print(f"    [smoke] {msg}", flush=True)
        else:
            raise RuntimeError(msg)
    causal.config.use_cache = False
    V = causal.get_output_embeddings().weight.shape[0]
    yes_id, no_id = _resolve_token("yes", V), _resolve_token("no", V)
    if yes_id is None or no_id is None or yes_id == no_id:
        if not SMOKE:
            raise RuntimeError("yes/no token ids unresolvable -- wrong tokenizer/checkpoint. "
                               "This is the random-head bug class; refusing.")
        yes_id, no_id = 1, 2
    if CFG["hf"].startswith("Qwen/Qwen3-Reranker") and (yes_id, no_id) != YES_NO_REFERENCE:
        print(f"    [head] WARNING: yes/no ids ({yes_id},{no_id}) differ from the published "
              f"{YES_NO_REFERENCE}; verify the tokenizer revision before trusting scores", flush=True)
    w = (causal.get_output_embeddings().weight[yes_id]
         - causal.get_output_embeddings().weight[no_id]).detach().clone()
    print(f"    [head] yes={yes_id} no={no_id} |w|={float(w.float().norm()):.3f}", flush=True)
    if not SMOKE:
        causal.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not FULL_FT:
        try:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"])
            causal = get_peft_model(causal, lcfg)
            if hasattr(causal, "enable_input_require_grads"):
                causal.enable_input_require_grads()   # redundant with checkpointing
            ntr = sum(p.numel() for p in causal.parameters() if p.requires_grad)
            print(f"    [lora] r={LORA_R} alpha={2*LORA_R} trainable {ntr/1e6:.1f}M", flush=True)
        except ImportError:
            if not SMOKE:
                raise
            print("    [smoke] peft unavailable, full-FT the tiny model instead", flush=True)
    model = QwenScorer(causal, w).to(dev)
    return model


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    st = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
    if FULL_FT:                                    # 16GB fp32 -> 8GB bf16 on disk
        st = {k: v.to(torch.bfloat16) for k, v in st.items()}
    return st


def save_state(path, st):
    tmp = path + ".tmp"
    torch.save(st, tmp)
    os.replace(tmp, path)
    print(f"    wrote {os.path.basename(path)}", flush=True)


def _guard_state_mode(st, where):
    """Cross-mode guard: a LoRA run must never silently consume a state
    blob without LoRA tensors (e.g. after a WORKRB_Q_FULL_FT flip between requeues) --
    load_state_dict(strict=False) would no-op it and score an untrained model."""
    if SMOKE or FULL_FT:
        return
    if not any("lora_" in k for k in st):
        raise RuntimeError(f"{where}: state has no LoRA tensors but mode is LoRA -- "
                           f"cross-mode checkpoint (full-FT flip between requeues?). Refusing.")


def load_state_into(model, path):
    st = torch.load(path, map_location="cpu")
    _guard_state_mode(st, path)
    missing, unexpected = model.load_state_dict(st, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys loading {path}: {unexpected[:5]}")
    print(f"    loaded {len(st)} trainable tensors from {os.path.basename(path)}", flush=True)


def lambda_ndcg2pp_loss(scores, labels, k=None, sigma=1.0, eps=1e-10, mu=10.0):
    """Faithful port of sentence_transformers.cross_encoder.losses.LambdaLoss with
    NDCGLoss2PPScheme(mu=10), k=None, sigma=1.0, eps=1e-10, reduction_log='binary',
    mean reduction over valid pairs -- the exact v6 training loss, applied to a dense
    (batch, n_docs) score matrix. Padded docs (unused here: rows are fixed 15/24 docs)
    are marked with float('-inf') labels, as in the original. Verified bit-identical
    (values AND grads) to the installed sentence-transformers () and to 6e-8
    against allRank master ()."""
    device = scores.device
    B, N = scores.shape
    finite = torch.isfinite(labels)
    logits_matrix = torch.where(finite, scores, torch.full_like(scores, -1e16))
    logits_sorted, indices_pred = logits_matrix.sort(descending=True, dim=-1)
    labels_sorted, _ = labels.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(labels, 1, indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs) & (true_diffs > 0)
    k_ = k or N
    ndcg_at_k_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k_, :k_] = 1
    true_sorted_by_preds = true_sorted_by_preds.clamp(min=0.0)
    labels_sorted = labels_sorted.clamp(min=0.0)
    pos_idxs = torch.arange(1, N + 1, device=device)
    discount = torch.log2(1.0 + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, labels_sorted) - 1) / discount)[:, :k_], dim=-1).clamp(min=eps)
    gain = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]
    # NDCGLoss2 scheme
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(discount[0, delta_idxs - 1]), -1.0)
                       - torch.pow(torch.abs(discount[0, delta_idxs]), -1.0))
    deltas.diagonal().zero_()
    gain_diffs = torch.abs(gain[:, :, None] - gain[:, None, :])
    ndcg2_w = deltas[None, :, :] * gain_diffs
    # LambdaRank scheme
    lambda_w = torch.abs(torch.pow(discount[:, :, None], -1.0)
                         - torch.pow(discount[:, None, :], -1.0)) * gain_diffs
    weights = mu * ndcg2_w + lambda_w
    scores_diffs = (logits_sorted[:, :, None] - logits_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs = torch.where(torch.isnan(scores_diffs), torch.zeros_like(scores_diffs), scores_diffs)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    losses = torch.log2(weighted_probas)
    masked = losses[padded_pairs_mask & ndcg_at_k_mask]
    if masked.numel() == 0:            # all-identical labels: undefined ideal ranking
        return scores.sum() * 0.0      # (guarded upstream by build_rows; redundant by design)
    return -torch.mean(masked)


INFER = {"bs": INFER_BS}     # inference batch; halved on OOM, reduction persists (draft 2)


@torch.no_grad()
def qpredict(model, pairs, batch_size, use_autocast=True):
    """Score (query, doc) pairs. Length-sorted batching, order restored on output,
    OOM batch halving (persists via INFER['bs'] for later calls)."""
    model.eval()
    seqs = encode_pairs(pairs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros(len(seqs), np.float32)
    bs = max(1, batch_size)
    s = 0
    while s < len(order):
        idxs = order[s:s + bs]
        try:
            ids, mask = pad_left([seqs[i] for i in idxs])
            ctx = (torch.autocast("cuda", dtype=AC_DTYPE)
                   if (use_autocast and dev == "cuda") else contextlib.nullcontext())
            with ctx:
                sc = model(ids.to(dev), mask.to(dev))
            out[np.array(idxs)] = sc.float().cpu().numpy()
            s += len(idxs)
        except torch.cuda.OutOfMemoryError:
            free()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            INFER["bs"] = min(INFER["bs"], bs)
            print(f"    [OOM] inference batch -> {bs}", flush=True)
    return out


def assert_finite(model, tag):
    """v6 gate, adapted: probe-pair finite check + (CUDA) bf16-vs-fp32 parity on 256
    real pairs. Weights are fp32 masters, so the no-autocast pass is honest fp32.
    Rank agreement (spearman >= PARITY_RHO) is the binding gate; amplitude drift in
    (0.02, PARITY_DRIFT] passes with a warning (bf16: 8 mantissa bits vs fp16's 11)."""
    a = title2idx.get("debug software", 0)
    b = title2idx.get("3D lighting", NT - 1)
    probe = "debug software in a full stack environment"
    p = qpredict(model, [(probe, docs[a]), (probe, docs[b])], batch_size=2, use_autocast=False)
    if not np.all(np.isfinite(p)):
        raise RuntimeError(f"{tag}: non-finite output, diverged")
    print(f"    [{tag}] finite check PASSED {p} (relevant>irrelevant: {p[0] > p[1]})", flush=True)
    if not SMOKE and not (p[0] > p[1]):
        # the only zero-gradient stationary point found is a fully-saturated
        # sign flip; a mis-ordered probe on the real model is its cheapest early symptom.
        print(f"    [{tag}] WARNING: probe ordering is WRONG (relevant scored below "
              f"irrelevant). Inspect the OOF ladder before trusting this model.", flush=True)
    if dev == "cuda":
        pairs = [(probe, docs[j]) for j in range(min(256, NT))]
        f32 = qpredict(model, pairs, batch_size=32, use_autocast=False)
        fbf = qpredict(model, pairs, batch_size=32, use_autocast=True)
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)
        rho = float(np.corrcoef(rk(f32), rk(fbf))[0, 1])
        drift = float(np.max(np.abs(f32 - fbf)) / (np.std(f32) + 1e-9))
        if rho < PARITY_RHO or drift > PARITY_DRIFT:
            raise RuntimeError(f"{tag}: bf16 autocast diverges from fp32 "
                               f"(spearman {rho:.5f}, max|d|/std {drift:.4f})")
        lvl = ("PASSED" if drift <= V6_FP16_DRIFT
               else "PASSED (drift above v6's fp16 0.02 bound, rank-safe)")
        print(f"    [{tag}] bf16 parity {lvl} (spearman {rho:.5f}, max|d|/std {drift:.4f})",
              flush=True)


def _forward_loss(model, qs, dls, lls, rows_idx):
    pairs, labs = [], []
    ndoc = len(dls[rows_idx[0]])
    for i in rows_idx:
        assert len(dls[i]) == ndoc, "mixed docs/row inside a micro-batch"
        pairs += [(qs[i], d) for d in dls[i]]
        labs.append(lls[i])
    ids, mask = pad_left(encode_pairs(pairs))
    ctx = (torch.autocast("cuda", dtype=AC_DTYPE) if dev == "cuda" else contextlib.nullcontext())
    with ctx:
        sc = model(ids.to(dev), mask.to(dev))
    y = torch.tensor(labs, dtype=torch.float32, device=sc.device)
    loss = lambda_ndcg2pp_loss(sc.float().view(len(rows_idx), ndoc), y)
    if not torch.isfinite(loss):                      # NanGuard
        raise RuntimeError("loss non-finite; diverged")
    return loss


def train_qwen(model, ds, tag, lr, budget_s, epochs, warmup=50):
    """Budget-driven trainer mirroring v6 train_ce semantics: max_steps from measured/
    prior throughput so the LR schedule decays properly, TimeStop as backstop, NanGuard,
    grad-clip 1.0, OOM halving of the micro-batch, measured-rate accounting, and
    periodic resumable checkpoints in WORK (preemption survival)."""
    global MICRO
    qs, dls, lls = ds["query"], ds["docs"], ds["labels"]
    if len(qs) == 0:
        raise RuntimeError(f"{tag}: empty training dataset (every query dropped?)")
    nrows, docs_per_row = len(qs), len(dls[0])
    eff = MICRO * ACCUM
    rate = pps(CE_KEY, "train", CFG["prior_train_pps"])
    steps_full = max(1, (nrows // eff) * epochs)
    steps_afford = max(50, int(budget_s * rate / (eff * docs_per_row)))
    max_steps = min(steps_full, steps_afford)
    warmup_steps = min(warmup, max(1, max_steps // 10))
    print(f"    rows={nrows} docs/row={docs_per_row} eff-batch={eff} -> {steps_full} steps full, "
          f"{max_steps} affordable ({budget_s/60:.0f} min @ {rate:.0f} pairs/s)", flush=True)

    r = np.random.RandomState(SEED)
    reps = math.ceil(max_steps * eff / nrows) + 1
    order = np.concatenate([r.permutation(nrows) for _ in range(reps)])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        return max(0.0, (max_steps - s) / max(1, max_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step = 0
    ck = os.path.join(WORK, f"{RUN}-ce{CE_KEY}-ckpt-{tag}.pt")
    if os.path.exists(ck):
        blob = torch.load(ck, map_location="cpu")
        _guard_state_mode(blob["model"], ck)  # no silent cross-mode resume
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys resuming {ck}: {unexpected[:5]}")
        if blob.get("opt") is not None:
            opt.load_state_dict(blob["opt"])
        step = int(blob["step"])
        for _ in range(step):
            sched.step()
        print(f"    [resume] {tag} from step {step}", flush=True)

    stop_at = min(time.time() + budget_s, DEADLINE)
    model.train()
    t0, pairs_done, last_loss = time.time(), 0, float("nan")
    while step < max_steps:
        if time.time() > stop_at:
            print(f"    [time] stopping at step {step}", flush=True)
            break
        rows = order[step * eff:(step + 1) * eff]
        opt.zero_grad(set_to_none=True)
        mi = 0
        while mi < len(rows):
            sub = rows[mi:mi + MICRO]
            try:
                loss = _forward_loss(model, qs, dls, lls, list(sub))
                (loss / max(1, math.ceil(len(rows) / MICRO))).backward()
                last_loss = float(loss.detach())
                mi += len(sub)
            except torch.cuda.OutOfMemoryError:
                free()
                if MICRO == 1:
                    raise
                MICRO = max(1, MICRO // 2)
                print(f"    [OOM] micro-rows -> {MICRO}, redoing step", flush=True)
                opt.zero_grad(set_to_none=True)
                mi = 0
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        pairs_done += len(rows) * docs_per_row
        if step % 20 == 0 or step == max_steps:
            el = time.time() - t0
            print(f"      {tag} step {step}/{max_steps}  loss {last_loss:.4f}  "
                  f"{pairs_done/max(el,1):.0f} pairs/s  lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if step > warmup_steps and last_loss > 2.5:
                # normal converged loss is ~0.02-0.6; a loss stuck 5-100x above
                # that after warmup is the saturation/misordering signature.
                print(f"      [warn] {tag}: loss {last_loss:.2f} still high after warmup -- "
                      f"possible sigmoid saturation or label misordering", flush=True)
        if step % CKPT_EVERY == 0 and step < max_steps:
            save_state(ck, {"model": trainable_state(model),
                            "opt": opt.state_dict() if not FULL_FT else None, "step": step})
    el = time.time() - t0
    if pairs_done:
        THROUGHPUT[CE_KEY + "_train"] = pairs_done / max(el, 1.0)
    print(f"    trained to step {step} in {el/60:.1f} min -> "
          f"{THROUGHPUT.get(CE_KEY+'_train', rate):.0f} pairs/s measured", flush=True)
    del opt, sched
    free()
    assert_finite(model, tag)
    return model


def clear_ckpt(tag):
    p = os.path.join(WORK, f"{RUN}-ce{CE_KEY}-ckpt-{tag}.pt")
    if os.path.exists(p):
        os.remove(p)


def ce_score(model, qtexts, cands, tag, part_name=None):
    """v6 ce_score semantics: per-query scoring, non-finite refusal, progress with eta,
    and the SLOWEST observed rate kept (SkillNorm-last bias rule). part_name enables
    mid-task resume via .part files (draft 2) -- used for the long test tasks."""
    out = np.zeros(cands.shape, np.float32)
    start = 0
    if part_name:
        pn, pj = art(part_name + ".part.npy"), art(part_name + ".part.json")
        if pn and pj:
            prev = np.load(pn)
            if prev.shape == out.shape:
                with open(pj) as f:
                    start = int(json.load(f)["done"])
                out[:start] = prev[:start]
                print(f"      {tag} resuming at query {start}", flush=True)
    t0 = time.time()
    n = len(qtexts)
    for i in range(start, n):
        s = qpredict(model, [(qtexts[i], docs[int(j)]) for j in cands[i]],
                     batch_size=INFER["bs"], use_autocast=True)
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"{tag}: non-finite CE scores at query {i}; refusing to export")
        out[i] = s
        if (i + 1) % 50 == 0 or i + 1 == n:
            r = (i + 1 - start) * cands.shape[1] / max(time.time() - t0, 1.0)
            print(f"      {tag} {i+1}/{n} q  {r:.0f} pairs/s  "
                  f"eta {(n-i-1)*cands.shape[1]/max(r,1)/60:.1f} min", flush=True)
        if part_name and ((i + 1) % 200 == 0):
            np.save(os.path.join(WORK, part_name + ".part.npy"), out)
            save_json(part_name + ".part.json", {"done": i + 1})
    _k = CE_KEY + "_infer"
    if n > start:
        _r = (n - start) * cands.shape[1] / max(time.time() - t0, 1.0)
        THROUGHPUT[_k] = min(THROUGHPUT.get(_k, _r), _r)
    if part_name:
        for ext in (".part.npy", ".part.json"):
            with contextlib.suppress(OSError):
                os.remove(os.path.join(WORK, part_name + ext))
    return out


# ======================================================================================
# dataset builders -- VERBATIM from the main pipeline (do not edit; recipe fidelity)
# ======================================================================================
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


# ======================================================================================
# RUN: stage 1 -> folds/OOF -> full model + test          (v6 run_ce_model, key "Q")
# ======================================================================================
S1_FINAL = f"{RUN}-ce{CE_KEY}-stage1-final.pt"


def s1():
    found = art(S1_FINAL)
    if found:
        print(f"[reuse] stage 1 for model {CE_KEY} at {found}", flush=True)
        return found
    ds1 = build_stage1(CFG)
    m = build_model()
    train_qwen(m, ds1, f"ce{CE_KEY}-stage1", S1_LR, S1_BUDGET, epochs=1, warmup=50)
    save_state(os.path.join(WORK, S1_FINAL), trainable_state(m))
    clear_ckpt(f"ce{CE_KEY}-stage1")
    del m, ds1
    free()
    return os.path.join(WORK, S1_FINAL)


got = phase(f"MODEL {CE_KEY} ({CFG['hf']})  STAGE 1 pretrain", S1_BUDGET + 20 * 60, s1)
if got is None:
    raise RuntimeError("stage 1 unavailable; nothing downstream can run. Aborting.")
STAGE1_PATH = got

# ---- fold construction: VERBATIM v6 (single RandomState(SEED) consumed across
# VAL_TASKS in dict order Tech, House, SkillSkape, SkillNorm) so folds are bit-identical
# to the ones behind v6_ceA/B_oof_val_* ----
r = np.random.RandomState(SEED)
folds = {t: np.array_split(r.permutation(load_np(f"{RUN}_cands_val_{t}.npy").shape[0]), NFOLD)
         for t in VAL_TASKS}
nval = sum(len(Q[("val", t)]["txt"]) for t in VAL_TASKS)
s2_pairs = nval * CFG["s2_samples"] * CFG["s2_docs"] * CFG["s2_epochs"]
s2_est = s2_pairs / CFG["prior_train_pps"]


def fresh_from_state(path):
    m = build_model()
    load_state_into(m, path)
    return m


def folds_oof():
    if all(art(f"{RUN}_ce{CE_KEY}_oof_val_{t}.npy") for t in VAL_TASKS):
        print(f"[reuse] model {CE_KEY} OOF already exists", flush=True)
        return True
    oof = {t: np.zeros(load_np(f"{RUN}_cands_val_{t}.npy").shape, np.float32) for t in VAL_TASKS}
    for fo in range(NFOLD):
        parts = {t: f"{RUN}_ce{CE_KEY}_oofpart_val_{t}_f{fo}.npy" for t in VAL_TASKS}
        if all(art(p) for p in parts.values()):
            for t in VAL_TASKS:
                oof[t][folds[t][fo]] = load_np(parts[t])
            print(f"  [reuse] fold {fo} parts", flush=True)
            continue
        # Fold-model persistence (draft 2): a crash between training and scoring must
        # not cost a retrain -- reload the trained fold state if it exists.
        fold_final = f"{RUN}-ce{CE_KEY}-fold{fo}-final.pt"
        ds2 = None
        if art(fold_final):
            print(f"  [reuse] fold {fo} model already trained", flush=True)
            m = fresh_from_state(art(fold_final))
            assert_finite(m, f"ce{CE_KEY}-fold{fo}-reloaded")   # gate BEFORE any scoring
        else:
            ds2 = stage2_dataset(CFG, folds, fo, np.random.RandomState(SEED + fo))
            m = fresh_from_state(STAGE1_PATH)
            train_qwen(m, ds2, f"ce{CE_KEY}-fold{fo}", S2_LR, max(300.0, s2_est * 0.8),
                       CFG["s2_epochs"], warmup=20)
            save_state(os.path.join(WORK, fold_final), trainable_state(m))
            clear_ckpt(f"ce{CE_KEY}-fold{fo}")
        for t in VAL_TASKS:
            if art(parts[t]):
                oof[t][folds[t][fo]] = load_np(parts[t])
                print(f"  [reuse] fold {fo} part for {t}", flush=True)
                continue
            c = load_np(f"{RUN}_cands_val_{t}.npy")
            qt = Q[("val", t)]["txt"]
            idx = folds[t][fo]
            s = ce_score(m, [qt[i] for i in idx], c[idx], f"ce{CE_KEY}f{fo}/{t}")
            oof[t][idx] = s
            save_np(parts[t], s)
        if FULL_FT:            # ~8GB each; LoRA finals (~130MB) are kept as bonus members
            with contextlib.suppress(OSError):
                os.remove(os.path.join(WORK, fold_final))
        del m
        if ds2 is not None:
            del ds2
        free()
        print(f"  [fold {fo}] done", flush=True)
    for t in VAL_TASKS:
        save_np(f"{RUN}_ce{CE_KEY}_oof_val_{t}.npy", oof[t])

    # KNOWN-VALUE GATE: CE-Q alone, out-of-fold, vs every reference point on record.
    ce_per, base_per = {}, {}
    for t in VAL_TASKS:
        y, idcg = load_np(f"{RUN}_y_val_{t}.npy"), load_np(f"{RUN}_idcg_val_{t}.npy")
        ce_per[t] = float(np.mean([ndcg_at(y[i][np.argsort(-oof[t][i])], idcg[i])
                                   for i in range(len(y))]))
        base_per[t] = float(np.mean([ndcg_at(y[i], idcg[i]) for i in range(len(y))]))
    cm = float(np.mean(list(ce_per.values())))
    bm = float(np.mean(list(base_per.values())))
    print(f"\n  {'='*68}\n  CROSS-ENCODER {CE_KEY} ALONE, out-of-fold, top-{K}: "
          f"macro nDCG@100 = {cm:.4f}  (base {bm:.4f}, {cm-bm:+.4f})", flush=True)
    for t in VAL_TASKS:
        print(f"      {t:11s} {ce_per[t]:.4f}  (base {base_per[t]:.4f}, "
              f"{ce_per[t]-base_per[t]:+.4f})", flush=True)
    print(f"  reference: base 0.6499 | v3 MiniLM 0.6884 | v5 bge-base 0.7044 | "
          f"v6 A bge-base 0.7327 (all CE-alone)", flush=True)
    if not SMOKE and cm < 0.7044:
        print("  WARNING: below v5's 278M bge-base; a 4B model earning less than a "
              "14x-smaller one is a red flag -- inspect before spending test GPU time",
              flush=True)
    print(f"  {'='*68}", flush=True)
    save_json(f"{RUN}_ce{CE_KEY}_alone.json", {"macro": cm, "base": bm, "per_task": ce_per})
    return True


if phase(f"MODEL {CE_KEY}  STAGE 2 x{NFOLD} folds + out-of-fold val scores",
         s2_est * NFOLD + nval * K / CFG["prior_infer_pps"] + 10 * 60, folds_oof) is None:
    raise RuntimeError("OOF phase failed; a CE without OOF val scores is unusable downstream.")


FULL_FINAL = f"{RUN}-ce{CE_KEY}-full-final.pt"


def full_and_test():
    mem = "full"
    if all(art(f"{RUN}_ce{CE_KEY}_test_{t}_{mem}.npy") for t in TEST_TASKS):
        print(f"[reuse] model {CE_KEY} full test scores already exist", flush=True)
        return True
    if art(FULL_FINAL):
        print(f"  [reuse] full model already trained", flush=True)
        m = fresh_from_state(art(FULL_FINAL))
        assert_finite(m, f"ce{CE_KEY}-full-reloaded")           # gate BEFORE any scoring
    else:
        ds2 = stage2_dataset(CFG, folds, None, np.random.RandomState(SEED))
        m = fresh_from_state(STAGE1_PATH)
        train_qwen(m, ds2, f"ce{CE_KEY}-full", S2_LR, max(300.0, s2_est),
                   CFG["s2_epochs"], warmup=20)
        save_state(os.path.join(WORK, FULL_FINAL), trainable_state(m))
        clear_ckpt(f"ce{CE_KEY}-full")
        del ds2
        free()
    tk = min(CFG.get("test_K", K), K)
    complete = True
    for t in TEST_TASKS:
        if art(f"{RUN}_ce{CE_KEY}_test_{t}_{mem}.npy"):
            print(f"  [reuse] test scores for {t}", flush=True)
            continue
        # per-task gate: phase() only checks time at START (threshold 0.6), so an
        # optimistic estimate lets a phase start that cannot finish, and it then fails near the end. Stopping
        # cleanly here leaves a truthful manifest instead of a dead multi-hour phase.
        need = len(Q[("test", t)]["txt"]) * tk / pps(CE_KEY, "infer", CFG["prior_infer_pps"])
        if DEADLINE - time.time() < need * 1.15:
            print(f"  [stop] no time for test task {t}; model {CE_KEY} test scoring "
                  f"incomplete (full model kept for resume)", flush=True)
            complete = False
            break
        c = load_np(f"{RUN}_cands_test_{t}.npy")[:, :tk]
        s = ce_score(m, Q[("test", t)]["txt"], c, f"ce{CE_KEY}full/{t}",
                     part_name=f"{RUN}_ce{CE_KEY}_test_{t}_{mem}")
        save_np(f"{RUN}_ce{CE_KEY}_test_{t}_{mem}.npy", s)
    del m
    free()
    if complete and FULL_FT:   # ~8GB; the LoRA full final (~130MB) is kept
        with contextlib.suppress(OSError):
            os.remove(os.path.join(WORK, FULL_FINAL))
    return complete


ntest = sum(len(Q[("test", t)]["txt"]) for t in TEST_TASKS)
infer_est = ntest * min(CFG.get("test_K", K), K) / pps(CE_KEY, "infer", CFG["prior_infer_pps"])
phase(f"MODEL {CE_KEY}  FULL model + TEST scoring ({ntest} queries x {K})",
      s2_est + infer_est, full_and_test)


# ======================================================================================
# manifest: merge ce_ready["Q"] into the run manifest (v6 contract; art() prefers WORK)
# and write a standalone v6_manifest_ceQ.json record of this run (redundant by design).
# ======================================================================================
tk = min(CFG.get("test_K", K), K)
ready = {
    "oof_val": all(art(f"{RUN}_ce{CE_KEY}_oof_val_{t}.npy") is not None for t in VAL_TASKS),
    "test_members": [m for m in ("full",)
                     if all(art(f"{RUN}_ce{CE_KEY}_test_{t}_{m}.npy") is not None
                            for t in TEST_TASKS)],
    "test_K": tk,
}
mp = art(f"{RUN}_manifest.json")
manifest = json.load(open(mp)) if mp else {"run": RUN, "K": K, "val_tasks": VAL_TASKS,
                                           "test_tasks": TEST_TASKS, "ce_ready": {}}
manifest.setdefault("ce_ready", {})[CE_KEY] = ready
manifest.setdefault("throughput", {}).update(THROUGHPUT)
manifest.setdefault("phases", []).extend(PHASES)
save_json(f"{RUN}_manifest.json", manifest)
save_json(f"{RUN}_manifest_ce{CE_KEY}.json", {
    "run": RUN, "key": CE_KEY, "hf": CFG["hf"],
    "mode": "full_ft" if FULL_FT else f"lora_r{LORA_R}",
    "loss": "LambdaLoss(NDCGLoss2PPScheme) over yes-minus-no logits",
    "phases": PHASES, "throughput": THROUGHPUT, "ce_ready": {CE_KEY: ready},
    "elapsed_h": round((time.time() - START) / 3600, 2)})

# Full-FT stage-1 checkpoints are ~8GB of pure storage cost once the model is finished;
# the LoRA finals (~130MB each) are kept either way (cheap; enable bonus fold members).
if FULL_FT and ready["oof_val"] and ready["test_members"]:
    p = os.path.join(WORK, S1_FINAL)
    if os.path.exists(p):
        os.remove(p)
        print("[cleanup] model Q complete; removed its full-FT stage-1 state", flush=True)

print("\n" + "=" * 78)
print(f"RUN COMPLETE in {(time.time()-START)/3600:.2f} h")
for p in PHASES:
    print(f"  {p['status']:22s} {p.get('minutes','')!s:>7}  {p['name']}")
state = "USABLE" if (ready["oof_val"] and ready["test_members"]) else "not usable (needs both)"
print(f"\nmodel {CE_KEY}: oof_val={ready['oof_val']}  test_members={ready['test_members']}  "
      f"test_K={ready['test_K']}  -> {state}")
print("=" * 78, flush=True)
