"""Isolated LightGBM prediction worker. Imports numpy and lightgbm only, never torch.

On this machine `lightgbm.Booster(model_file=...)` segfaults with exit 139 whenever torch has
already been imported in the same process, because two OpenMP runtimes end up loaded. Measured:

    import lightgbm; Booster(...)                      -> OK
    import torch; import lightgbm; Booster(...)        -> SIGSEGV
    import lightgbm; import torch; Booster(...)        -> OK
    KMP_DUPLICATE_LIB_OK=TRUE                          -> still SIGSEGV

submission/generate_submission_file.py imports workrb (and therefore torch) at module load, before
it imports the participant model file, so the import order cannot be fixed from this end. Running
the booster in a fresh process side-steps the conflict entirely and costs one process spawn per
task block, not per query.

Contract: argv = <features.npy> <model.txt> <out.npy>
  features.npy : float32 (N, F)
  out.npy      : float32 (N,) raw booster scores
"""
from __future__ import annotations
import sys
import numpy as np
import lightgbm as lgb


def main() -> None:
    feat_path, model_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    X = np.load(feat_path)
    booster = lgb.Booster(model_file=model_path)
    if X.shape[1] != booster.num_feature():
        raise SystemExit(
            f"feature drift: booster expects {booster.num_feature()} columns, got {X.shape[1]}")
    np.save(out_path, np.asarray(booster.predict(X), dtype=np.float32))


if __name__ == "__main__":
    main()
