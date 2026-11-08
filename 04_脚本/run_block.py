import numpy as np
import pandas as pd
import random
import common as C

g, X, y = C.load("native")
xy = np.c_[g.geometry.x.to_numpy(), g.geometry.y.to_numpy()]
n = len(y)


def block_ids(size):
    bx = np.floor(xy[:, 0] / size).astype(int)
    by = np.floor(xy[:, 1] / size).astype(int)
    return pd.Series([f"{a}_{b}" for a, b in zip(bx, by)], index=g.index)


def fold_manual(bid):
    # exactly the manual's field-calculator logic: random.seed(block_id); randint(1,5)
    out = {}
    for b in bid.unique():
        random.seed(b)
        out[b] = random.randint(1, 5)
    return bid.map(out)


def fold_balanced(bid):
    # blocks sorted by point count, dealt round-robin -> balanced fold sizes
    cnt = bid.value_counts()
    order = cnt.sort_values(ascending=False).index
    f = {b: (i % 5) + 1 for i, b in enumerate(order)}
    return bid.map(f)


def run_cv(folds, label, size):
    oof = np.full(n, np.nan)
    per_fold = []
    for k in range(1, 6):
        va = np.where(folds.to_numpy() == k)[0]
        tr = np.where(folds.to_numpy() != k)[0]
        p = C.fit_predict(X.iloc[tr], y[tr], X.iloc[va])
        oof[va] = p
        m = C.metrics(y[va], p)
        per_fold.append((k, len(va), m["R2"], m["MAE"], m["RMSE"]))
    ok = ~np.isnan(oof)
    m = C.metrics(y[ok], oof[ok])
    nblk = folds.nunique()
    print(f"\n{label}  block={size//1000}km  blocks={nblk}")
    for k, nv, r2, mae, rmse in per_fold:
        print(f"   fold{k}: n={nv:5d}  R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    print(f"   POOLED  R2={m['R2']:.4f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}"
          f"   per-fold R2 range {min(p[2] for p in per_fold):.3f}-{max(p[2] for p in per_fold):.3f}")
    return m, nblk, per_fold


if __name__ == "__main__":
    print(f"n={n}")
    for size in (10000, 20000, 40000):
        bid = block_ids(size)
        run_cv(fold_manual(bid), "manual random folds", size)
        run_cv(fold_balanced(bid), "balanced folds", size)
