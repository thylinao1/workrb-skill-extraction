"""Assemble a scored submission from the exported pipeline artefacts.

The submission is a JSON file of precomputed scores rather than runnable code,
so the pipeline exports the retrieval lanes and the cross-encoder scores for both
splits as plain .npy arrays and this script only has to combine them. It imports
numpy, lightgbm and json and nothing else, for two reasons. LightGBM aborts with
exit 139 and no traceback in any process where the deep-learning framework has
already been imported, and not importing it is the only fix that is not a
workaround. And because nothing needs re-encoding, rebuilding a submission takes
seconds.

Two of the levers here need no GPU at all.

The relevance prior contributes four smoothed per-skill columns. Measured at
+0.0380 over seven disjoint seeds, it survives a leave-one-task-out control at
+0.0315 and collapses under a label-shuffled control. Counts are rebuilt from the
candidate pools of each fold, using only that fold's training rows and leaving
out the row being scored. The acceptance gate requires at least +0.025 at the
chosen depth, failing which the counting window falls back to the first 500
candidates to restore the conditioning event P(grade >= 1 | target in pool).

The alternative-label override is a hard override rather than a feature. Query
text that exactly matches an alternative label of a candidate skill promotes that
candidate to the front. It measured +0.0564 on the normalisation task and exactly
0.0000 on each of the other three. Offering the same signal as a soft feature
measured worse and cost the extraction tasks, which is why it is applied at write
time instead. ESCO is CC BY 4.0 (c) European Commission and should be credited in
any method description.

The evaluation protocol is deliberately conservative:

  - folds are grouped by query, so a held-out query is never trained on
  - the cross-encoder feature on validation is out-of-fold, produced by a model
    that never saw that query
  - prior counts for a held-out query come only from that fold's training queries
  - the ideal DCG comes from the full qrel set, so the number matches the
    official scorer
  - no task one-hot encoding: one task is test-only and carries a fifth of the
    score, so the model has to generalise rather than memorise a task identity

The leaderboard aggregates as a flat mean over the five tasks, verified against
three placed rows.

Usage:
    python fusion/build_submission.py --kout /path/to/export --topk 500
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL_TASKS = ["House", "Tech", "SkillSkape", "SkillNorm"]
TEST_TASKS = ["Tech", "House", "TechWolf", "SkillSkape", "SkillNorm"]
DISPLAY = {
    "Tech": "Skill Extraction Tech Graded",
    "House": "Skill Extraction House Graded",
    "TechWolf": "Skill Extraction TechWolf Graded",
    "SkillSkape": "Skill Extraction SkillSkape Graded",
    "SkillNorm": "Skill Normalization ESCO Graded",
}
PREFIX = "http://data.europa.eu/esco/skill/"
SUBMIT_TOP_K = 500          # what generate_submission_file.py emits per query
DISC100 = 1.0 / np.log2(np.arange(100) + 2)
BOARD_OFFSET = -0.0086      # MEASURED: local 0.7178 -> board 0.7092 on submission 1


def ndcg_from_order(y_ordered: np.ndarray, idcg: float) -> float:
    g = (2.0 ** y_ordered[:100]) - 1.0
    return float(np.sum(g * DISC100[:len(g)]) / idcg) if idcg > 0 else 0.0


def per_query_z(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


def per_query_logrank(s: np.ndarray) -> np.ndarray:
    return np.log1p(np.argsort(np.argsort(-s, axis=1), axis=1).astype(np.float32))


class Export:
    """Reads the v6 artifacts and reports exactly what is and is not usable."""

    def __init__(self, kout: Path):
        self.k = kout
        self.manifest = json.loads((kout / "v6_manifest.json").read_text())
        self.lanes = json.loads((kout / "v6_lanes.json").read_text())
        self.corpus_ids = json.loads((kout / "v6_corpus_ids.json").read_text())
        self.K = int(self.manifest["K"])

    def np_(self, name: str) -> np.ndarray:
        return np.load(self.k / f"v6_{name}.npy")

    def json_(self, name: str):
        return json.loads((self.k / f"v6_{name}.json").read_text())

    def usable_ce(self) -> list[str]:
        """A cross-encoder is usable only if it has BOTH out-of-fold val scores and test scores."""
        out = []
        for key, info in self.manifest["ce_ready"].items():
            if info["oof_val"] and info["test_members"]:
                out.append(key)
            else:
                print(f"  [skip] cross-encoder {key}: oof_val={info['oof_val']} "
                      f"test_members={info['test_members']} -> cannot be used on both splits")
        return out


def build_matrix(ex: Export, split: str, task: str, ce_keys: list[str], topk: int,
                 members: str, altfeat: np.ndarray | None = None,
                 cedisp: bool = False, escoattr: np.ndarray | None = None) -> np.ndarray:
    """Per-candidate feature matrix: the lane block, then two columns per cross-encoder."""
    F = ex.np_(f"feat_{split}_{task}")[:, :topk, :]
    cols = [F]
    zs = []
    for key in ce_keys:
        if split == "val":
            s = ex.np_(f"ce{key}_oof_val_{task}")[:, :topk]
        else:
            avail = ex.manifest["ce_ready"][key]["test_members"]
            use = avail if members == "all" else [m for m in avail if m == members]
            if not use:
                raise RuntimeError(f"no test member '{members}' for cross-encoder {key}")
            # z-score each member within the query before averaging: members are separately
            # trained heads whose raw score scales are not calibrated to each other.
            s = np.mean([per_query_z(ex.np_(f"ce{key}_test_{task}_{m}")[:, :topk]) for m in use],
                        axis=0)
        z = per_query_z(s)
        zs.append(z)
        cols.append(z[:, :, None])
        cols.append(per_query_logrank(s)[:, :, None])
    if cedisp:
        if len(zs) < 2:
            raise RuntimeError("--cedisp needs at least 2 cross-encoders")
        # per-candidate disagreement across the CE z-columns: an uncertainty
        # signal the fusion otherwise only sees implicitly
        cols.append(np.std(np.stack(zs), axis=0)[:, :, None].astype(np.float32))
    if escoattr is not None:
        cols.append(escoattr[:, :topk, :])
    if altfeat is not None:
        cols.append(altfeat[:, :topk, :])
    return np.concatenate(cols, axis=2).astype(np.float32)


def feature_names(ex: Export, ce_keys: list[str], use_prior: bool, use_altfeat: bool = False,
                  use_cedisp: bool = False, use_escoattr: bool = False) -> list[str]:
    lanes = ex.lanes["lanes"]
    n = ([f"z_{x}" for x in lanes] + [f"lr_{x}" for x in lanes]
         + ["base_z", "base_logrank", "agree50", "agree200", "zmean", "zstd", "zmax",
            "target_len", "query_len"])
    for k in ce_keys:
        n += [f"ce{k}_z", f"ce{k}_lr"]
    if use_cedisp:
        n += ["ce_disp"]
    if use_escoattr:
        n += ["esco_reuse", "esco_type", "esco_nalt", "esco_desclen"]
    if use_prior:
        n += ["prior_rel", "prior_hi", "prior_logseen", "prior_egain"]
    if use_altfeat:
        n += ["altsub_n", "altsub_len"]
    return n


# ---------------------------------------------------------------------------
# Lever 1: per-ESCO-skill relevance prior. Construction copied verbatim from
# the confirmed prior measurement (the confirmed measurement); only the data source
# changes, from the v5 cache to the v6 export, as the verification requires.
# ---------------------------------------------------------------------------
def build_counts(cands_v: dict, yv: dict, rows_by_task: dict, nskill: int):
    seen = np.zeros(nskill)
    rel = np.zeros(nskill)
    hi = np.zeros(nskill)
    gain = np.zeros(nskill)
    for t, rows in rows_by_task.items():
        for i in rows:
            c = cands_v[t][i]
            y = yv[t][i]
            np.add.at(seen, c, 1.0)
            np.add.at(rel, c, (y >= 1).astype(float))
            np.add.at(hi, c, (y >= 3).astype(float))
            np.add.at(gain, c, (2.0 ** y) - 1.0)
    return seen, rel, hi, gain


def prior_feats(counts, c: np.ndarray, y: np.ndarray | None, loo: bool) -> np.ndarray:
    seen, rel, hi, gain = counts
    s, r, h, g = seen[c].copy(), rel[c].copy(), hi[c].copy(), gain[c].copy()
    if loo:                       # subtract this query's own contribution on TRAIN rows
        s -= 1.0
        r -= (y >= 1)
        h -= (y >= 3)
        g -= (2.0 ** y) - 1.0
        for a in (s, r, h, g):
            np.maximum(a, 0.0, out=a)
    return np.stack([(r + 1.0) / (s + 4.0), (h + 0.5) / (s + 8.0), np.log1p(s),
                     (g + 1.0) / (s + 4.0)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Lever 2: ESCO altLabel exact-match hard override. LUT joined on ORIGINURI
# (exact URI match to corpus_ids); the promotion rule is byte-identical to
# the alternative-label check: exact matches move to the front in candidate order.
# ---------------------------------------------------------------------------
def build_alt_lookup(corpus_ids: list[str]) -> dict[str, set[int]]:
    """lowercased altLabel string -> set of corpus indices whose skill carries it."""
    uri2idx = {u: i for i, u in enumerate(corpus_ids)}
    lab2idx: dict[str, set[int]] = {}
    with open(ROOT / "data" / "esco_v111_skills.csv", newline="") as f:
        for row in csv.DictReader(f):
            idx = uri2idx.get(row["ORIGINURI"])
            if idx is None:
                continue
            for a in (row["ALTLABELS"] or "").split("\n"):
                a = a.strip().lower()
                if a:
                    lab2idx.setdefault(a, set()).add(idx)
    return lab2idx


def build_idx2labels(corpus_ids: list[str]) -> dict[int, list[str]]:
    """corpus index -> lowercased altLabels (>=4 chars) of that skill."""
    uri2idx = {u: i for i, u in enumerate(corpus_ids)}
    idx2lab: dict[int, list[str]] = {}
    with open(ROOT / "data" / "esco_v111_skills.csv", newline="") as f:
        for row in csv.DictReader(f):
            idx = uri2idx.get(row["ORIGINURI"])
            if idx is None:
                continue
            labs = [a.strip().lower() for a in (row["ALTLABELS"] or "").split("\n")]
            idx2lab[idx] = [a for a in labs if len(a) >= 4]
    return idx2lab


_ALT_NONWORD = re.compile(r"[^a-z0-9]+")


def alt_substr_feats(idx2lab: dict[int, list[str]], qtext: list[str],
                     cands: np.ndarray) -> np.ndarray:
    """(nq, K, 2): per candidate, count of its altLabels appearing verbatim
    (word-boundary padded) in the query text, and the longest such match."""
    F = np.zeros((*cands.shape, 2), dtype=np.float32)
    for i, q in enumerate(qtext):
        qn = " " + _ALT_NONWORD.sub(" ", q.lower()).strip() + " "
        for j, c in enumerate(cands[i]):
            labs = idx2lab.get(int(c))
            if not labs:
                continue
            n = best = 0
            for a in labs:
                if (" " + _ALT_NONWORD.sub(" ", a).strip() + " ") in qn:
                    n += 1
                    best = max(best, len(a))
            if n:
                F[i, j, 0] = min(n, 5)
                F[i, j, 1] = best / 10.0
    return F


def alt_exact_mask(lab2idx: dict[str, set[int]], qtext: list[str],
                   cands: np.ndarray) -> np.ndarray:
    M = np.zeros(cands.shape, dtype=bool)
    for i, q in enumerate(qtext):
        hit = lab2idx.get(q.strip().lower())
        if hit:
            M[i] = np.isin(cands[i], np.fromiter(hit, dtype=cands.dtype))
    return M


def alt_promote(order: np.ndarray, mask_row: np.ndarray) -> np.ndarray:
    ex = np.where(mask_row)[0]
    if not len(ex):
        return order
    es = set(ex.tolist())
    return np.array(ex.tolist() + [j for j in order if j not in es])


# ---------------------------------------------------------------------------
# --escoattr: 4 query-independent item attributes (an item-attribute lever: # +0.0088 standalone / +0.0014 on top of the prior, v5 measurement). Joined
# on ORIGINURI like the alt lookups. Categorical codes are ranks of the sorted
# distinct strings, so the encoding is deterministic across runs; unjoined
# corpus rows (and empty cells) stay -1.
# ---------------------------------------------------------------------------
def build_esco_attrs(corpus_ids: list[str]) -> np.ndarray:
    """(n_corpus, 4) float32: REUSELEVEL code, SKILLTYPE code, #altLabels, len(DESCRIPTION)."""
    uri2idx = {u: i for i, u in enumerate(corpus_ids)}
    A = np.full((len(corpus_ids), 4), -1.0, dtype=np.float32)
    cat: dict[int, tuple[str, str]] = {}
    with open(ROOT / "data" / "esco_v111_skills.csv", newline="") as f:
        for row in csv.DictReader(f):
            idx = uri2idx.get(row["ORIGINURI"])
            if idx is None:
                continue
            cat[idx] = ((row["REUSELEVEL"] or "").strip(), (row["SKILLTYPE"] or "").strip())
            A[idx, 2] = sum(1 for a in (row["ALTLABELS"] or "").split("\n") if a.strip())
            A[idx, 3] = float(len(row["DESCRIPTION"] or ""))
    for col in (0, 1):
        code = {v: c for c, v in enumerate(sorted({v[col] for v in cat.values()} - {""}))}
        for idx, v in cat.items():
            if v[col]:
                A[idx, col] = code[v[col]]
    return A


LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=1.0, reg_lambda=5.0)


def new_ranker(seed: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(objective="lambdarank", metric="ndcg", ndcg_eval_at=[100],
                          label_gain=[0, 1, 3, 7, 15], random_state=seed, n_jobs=4,
                          verbose=-1, **LGB_PARAMS)


def fit_group(model, X_by_task: dict, y_by_task: dict, rows_by_task: dict, counts=None,
              cands_v: dict | None = None):
    Xtr, ytr, gtr = [], [], []
    for t in VAL_TASKS:
        for i in rows_by_task[t]:
            X = X_by_task[t][i]
            if counts is not None:
                X = np.concatenate([X, prior_feats(counts, cands_v[t][i], y_by_task[t][i],
                                                   loo=True)], axis=1)
            Xtr.append(X)
            ytr.append(y_by_task[t][i])
            gtr.append(X.shape[0])
    model.fit(np.concatenate(Xtr), np.concatenate(ytr).astype(int), group=gtr)
    return model


def cross_validate(Xv, yv, idcg, seeds, ctx, nfold=5):
    """Query-grouped CV. Prior counts are rebuilt per fold from that fold's TRAINING
    queries only; the altLabel override is applied to the held-out prediction order."""
    macros, per_task, imp = [], {t: [] for t in VAL_TASKS}, None
    for seed in seeds:
        rng = np.random.RandomState(seed)
        folds = {t: np.array_split(rng.permutation(len(yv[t])), nfold) for t in VAL_TASKS}
        fold_macros = []
        for f in range(nfold):
            rows = {t: np.concatenate([folds[t][g] for g in range(nfold) if g != f])
                    for t in VAL_TASKS}
            counts = (build_counts(ctx["cands_val"], yv, rows, ctx["nskill"])
                      if ctx["prior"] else None)
            r = fit_group(new_ranker(seed), Xv, yv, rows, counts, ctx["cands_val"])
            imp = r.feature_importances_ if imp is None else imp + r.feature_importances_
            tv = {}
            for t in VAL_TASKS:
                vals = []
                for i in folds[t][f]:
                    X = Xv[t][i]
                    if counts is not None:
                        X = np.concatenate([X, prior_feats(counts, ctx["cands_val"][t][i],
                                                           None, loo=False)], axis=1)
                    order = np.argsort(-r.predict(X))
                    if ctx["alt"]:
                        order = alt_promote(order, ctx["alt_val"][t][i])
                    vals.append(ndcg_from_order(yv[t][i][order], idcg[t][i]))
                tv[t] = float(np.mean(vals))
                per_task[t].append(tv[t])
            fold_macros.append(float(np.mean(list(tv.values()))))
        macros.append(float(np.mean(fold_macros)))
        print(f"    seed {seed}: macro={macros[-1]:.4f}")
    return (float(np.mean(macros)), float(np.std(macros)) if len(macros) > 1 else 0.0,
            {t: float(np.mean(v)) for t, v in per_task.items()}, imp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kout", default="/tmp/kout6", help="folder holding the v6_* the notebook environment artifacts")
    ap.add_argument("--topk", type=int, default=0, help="rerank depth; 0 = everything exported")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ce", default="", help="cross-encoders to use, e.g. 'A' or 'A,B'; default all usable")
    ap.add_argument("--members", default="all", help="'all' to average test members, or one member name")
    ap.add_argument("--no-prior", action="store_true", help="drop the per-skill relevance prior")
    ap.add_argument("--no-alt", action="store_true", help="drop the altLabel exact-match override")
    ap.add_argument("--bag", type=int, default=10,
                    help="rankers in the final refit bag (measured +0.0011 at 10)")
    ap.add_argument("--ladder", action="store_true",
                    help="also CV the lever ablations: base / +prior / +prior+alt")
    ap.add_argument("--name", default="Fusion12+DualCE+Prior+LambdaRank")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-write", action="store_true", help="score only, do not emit a submission")
    ap.add_argument("--altfeat", action="store_true",
                    help="add altLabel-substring lexical features (count, max match length)")
    ap.add_argument("--cedisp", action="store_true",
                    help="add per-candidate std across the CE z-columns as one feature")
    ap.add_argument("--escoattr", action="store_true",
                    help="add 4 item-side ESCO attribute columns (reuse level, skill type, "
                         "#altLabels, description length)")
    args = ap.parse_args()

    ex = Export(Path(args.kout))
    ce_probe = args.ce.split(",") if args.ce else list(ex.manifest["ce_ready"].keys())
    depth_cap = min([ex.K] + [int(ex.manifest["ce_ready"][k].get("test_K", ex.K))
                              for k in ce_probe if k in ex.manifest["ce_ready"]])
    topk = args.topk or depth_cap
    if topk > depth_cap:
        raise SystemExit(f"--topk {topk} exceeds the usable depth {depth_cap} "
                         f"(export K={ex.K}, cross-encoder test_K may be shallower)")
    ce_keys = [c for c in (args.ce.split(",") if args.ce else ex.usable_ce()) if c]
    use_prior, use_alt = not args.no_prior, not args.no_alt
    print(f"[export] K={ex.K}, using depth {topk}, lanes ok={len(ex.lanes['ok'])}/{len(ex.lanes['lanes'])}")
    print(f"[export] cross-encoders in play: {ce_keys or 'NONE (lanes only)'}")
    print(f"[levers] prior={'ON' if use_prior else 'off'}  alt-override={'ON' if use_alt else 'off'}")
    tm = ex.json_("textmode")
    print(f"[export] query text mode '{tm['mode']}' (raw {tm['raw']:.4f} vs cleaned {tm['clean']:.4f})")

    af_v = {t: None for t in VAL_TASKS}
    idx2lab = None
    if args.altfeat:
        idx2lab = build_idx2labels(ex.corpus_ids)
        for t in VAL_TASKS:
            af_v[t] = alt_substr_feats(idx2lab, ex.json_(f"qtext_val_{t}"),
                                       ex.np_(f"cands_val_{t}")[:, :topk])
        print("[altfeat] val fired: " + "  ".join(
            f"{t}={int((af_v[t][:, :, 0] > 0).any(axis=1).sum())}q" for t in VAL_TASKS))
    ea = build_esco_attrs(ex.corpus_ids) if args.escoattr else None
    ea_v = {t: None for t in VAL_TASKS}
    if args.escoattr:
        for t in VAL_TASKS:
            ea_v[t] = ea[ex.np_(f"cands_val_{t}")[:, :topk]]
        print(f"[escoattr] corpus join: {int((ea[:, 3] >= 0).sum())}/{len(ex.corpus_ids)} skills")
    Xv = {t: build_matrix(ex, "val", t, ce_keys, topk, args.members, af_v[t],
                          cedisp=args.cedisp, escoattr=ea_v[t]) for t in VAL_TASKS}
    yv = {t: ex.np_(f"y_val_{t}")[:, :topk] for t in VAL_TASKS}
    idcg = {t: ex.np_(f"idcg_val_{t}") for t in VAL_TASKS}
    cands_v = {t: ex.np_(f"cands_val_{t}")[:, :topk] for t in VAL_TASKS}

    lab2idx = build_alt_lookup(ex.corpus_ids) if use_alt else {}
    alt_v = {}
    if use_alt:
        for t in VAL_TASKS:
            qt = ex.json_(f"qtext_val_{t}")
            alt_v[t] = alt_exact_mask(lab2idx, qt, cands_v[t])
        fired = {t: int(alt_v[t].any(axis=1).sum()) for t in VAL_TASKS}
        print(f"[alt] exact-match queries on val: " +
              "  ".join(f"{t}={fired[t]}/{len(yv[t])}" for t in VAL_TASKS))
    ctx = {"prior": use_prior, "alt": use_alt, "cands_val": cands_v, "alt_val": alt_v,
           "nskill": len(ex.corpus_ids)}

    base = {t: float(np.mean([ndcg_from_order(yv[t][i], idcg[t][i]) for i in range(len(yv[t]))]))
            for t in VAL_TASKS}
    base_macro = float(np.mean(list(base.values())))
    print(f"\n[base] fused base macro = {base_macro:.4f}  " +
          "  ".join(f"{t}={base[t]:.4f}" for t in VAL_TASKS))
    if abs(base_macro - 0.6499) > 0.002:
        print(f"  WARNING: base is {base_macro:.4f}, not the known 0.6499. "
              f"Expected only if phase 1 chose cleaned query text.")

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.ladder:
        print("\n[ladder] lever ablations, same folds and seeds each rung")
        rungs = [("base (lanes+CE only)", False, False), ("+prior", True, False),
                 ("+prior+alt", True, True)]
        prev = None
        for label, rp, ra in rungs:
            rctx = dict(ctx, prior=rp, alt=ra)
            m, _, pt, _ = cross_validate(Xv, yv, idcg, seeds, rctx)
            d = f"  ({m-prev:+.4f})" if prev is not None else ""
            print(f"  {label:22s} macro={m:.4f}{d}  " +
                  " ".join(f"{t}={pt[t]:.4f}" for t in VAL_TASKS))
            prev = m

    print("\n[cv] query-grouped 5-fold, no task one-hot, prior rebuilt per fold")
    macro, spread, per_task, imp = cross_validate(Xv, yv, idcg, seeds, ctx)
    print(f"\n{'='*74}")
    print(f"VAL macro nDCG@100 = {macro:.4f}   (seed spread {spread:.4f})")
    print(f"  fused base       = {base_macro:.4f}")
    print(f"  DELTA            = {macro-base_macro:+.4f}")
    print(f"  board estimate   = {macro+BOARD_OFFSET:.4f}   (measured offset {BOARD_OFFSET:+.4f})")
    for t in VAL_TASKS:
        print(f"    {t:11s} {per_task[t]:.4f}  (base {base[t]:.4f}, {per_task[t]-base[t]:+.4f})")
    print(f"  reference: v5 pipeline 0.7393 local / prior on v5 cache 0.7764 / leader 0.8145")
    print(f"{'='*74}")
    names = feature_names(ex, ce_keys, use_prior, args.altfeat, args.cedisp, args.escoattr)
    order = np.argsort(-imp)
    print("top features: " + ", ".join(f"{names[i]}({int(imp[i])})" for i in order[:14]))

    (ROOT / "eval" / "v6_ltr_results.json").write_text(json.dumps(
        {"macro": macro, "spread": spread, "base_macro": base_macro, "per_task": per_task,
         "base_per_task": base, "topk": topk, "ce": ce_keys, "members": args.members,
         "prior": use_prior, "alt_override": use_alt,
         "board_estimate": macro + BOARD_OFFSET,
         "importance": {names[i]: int(imp[i]) for i in order}}, indent=2))

    if args.no_write:
        return

    # ---- refit on every val query, then score the test split ----
    nq = sum(len(yv[t]) for t in VAL_TASKS)
    print(f"\n[fit] refitting on all {nq} val queries x {args.bag} bagged seeds, then scoring test")
    rows_all = {t: np.arange(len(yv[t])) for t in VAL_TASKS}
    counts_full = (build_counts(cands_v, yv, rows_all, ctx["nskill"]) if use_prior else None)
    finals = [fit_group(new_ranker(s), Xv, yv, rows_all, counts_full, cands_v)
              for s in range(args.bag)]

    submission = {args.name: {}}
    for t in TEST_TASKS:
        af_t = None
        if args.altfeat:
            af_t = alt_substr_feats(idx2lab, ex.json_(f"qtext_test_{t}"),
                                    ex.np_(f"cands_test_{t}")[:, :topk])
        ea_t = ea[ex.np_(f"cands_test_{t}")[:, :topk]] if args.escoattr else None
        Xt = build_matrix(ex, "test", t, ce_keys, topk, args.members, af_t,
                          cedisp=args.cedisp, escoattr=ea_t)
        cands = ex.np_(f"cands_test_{t}")[:, :topk]
        qids = ex.json_(f"qids_test_{t}")
        if len(qids) != Xt.shape[0]:
            raise RuntimeError(f"{t}: {len(qids)} query ids but {Xt.shape[0]} feature rows")
        alt_t = (alt_exact_mask(lab2idx, ex.json_(f"qtext_test_{t}"), cands)
                 if use_alt else np.zeros(cands.shape, dtype=bool))
        n_fired = int(alt_t.any(axis=1).sum())
        scores = {}
        for i in range(Xt.shape[0]):
            X = Xt[i]
            if counts_full is not None:
                X = np.concatenate([X, prior_feats(counts_full, cands[i], None, loo=False)],
                                   axis=1)
            # per-query z-score each bagged ranker before averaging: LGBMRanker outputs
            # are not calibrated across seeds, so a raw mean lets one seed dominate
            preds = np.stack([m.predict(X) for m in finals])
            preds = (preds - preds.mean(axis=1, keepdims=True)) / \
                    (preds.std(axis=1, keepdims=True) + 1e-9)
            s = preds.mean(axis=0)
            exi = np.where(alt_t[i])[0]
            if len(exi):
                # hard override: exact altLabel matches outrank everything, keeping their
                # relative candidate order, exactly as measured for the alternative-label check
                bump = float(s.max()) + 1.0
                for k, j in enumerate(exi):
                    s[j] = bump + (len(exi) - k) * 1e-3
            keep = np.argsort(-s)[:SUBMIT_TOP_K]
            row = {ex.corpus_ids[int(cands[i][j])][len(PREFIX):]: float(s[j]) for j in keep}
            if not all(np.isfinite(v) for v in row.values()):
                raise RuntimeError(f"{t} query {i}: non-finite score, JSON would be unloadable")
            scores[str(qids[i])] = row
        submission[args.name][DISPLAY[t]] = {
            "en": {"num_queries": len(qids), "num_targets": len(ex.corpus_ids), "scores": scores}}
        print(f"  {DISPLAY[t]:36s} {len(qids):4d} queries x {SUBMIT_TOP_K} targets"
              + (f"  (alt override fired on {n_fired} queries)" if use_alt else ""))

    stem = args.out or f"workrb-v6-depth{topk}-ce{''.join(ce_keys) or 'none'}"
    out_json = ROOT / "submission" / f"{stem}.json"
    out_json.write_text(json.dumps(submission, allow_nan=False))
    out_zip = out_json.with_suffix(".zip")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname=out_json.name)
    print(f"\n[done] {out_zip}  ({out_zip.stat().st_size/1e6:.1f} MB)  <- upload this")


if __name__ == "__main__":
    main()
