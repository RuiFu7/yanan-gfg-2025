import numpy as np
import common as C

g, X, y = C.load("native")
n = len(y)
print("n =", n)

rows = []
for i in range(10):
    rng = np.random.RandomState(2024 + i)
    idx = rng.permutation(n)
    nva = int(round(n * 0.2))
    va, tr = idx[:nva], idx[nva:]
    p = C.fit_predict(X.iloc[tr], y[tr], X.iloc[va])
    m = C.metrics(y[va], p)
    ptr = C.fit_predict(X.iloc[tr], y[tr], X.iloc[tr])
    mtr = C.metrics(y[tr], ptr)
    rows.append((i + 1, m["R2"], m["MAE"], m["RMSE"], mtr["R2"]))
    print(f"run {i+1:2d}  R2={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} trainR2={mtr['R2']:.4f}")

a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print("\nrandom 20% holdout x10")
print(f"  R2   mean={a[:,0].mean():.4f}  range {a[:,0].min():.4f}-{a[:,0].max():.4f}")
print(f"  MAE  mean={a[:,1].mean():.4f}")
print(f"  RMSE mean={a[:,2].mean():.4f}")
print(f"  train R2 mean={a[:,3].mean():.4f}")
