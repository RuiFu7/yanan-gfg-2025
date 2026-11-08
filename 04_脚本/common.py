import numpy as np
import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PTS = r"D:\GFG\xgb\pts.shp"
FEATS = ["GFG", "CROPKEPT", "FOR99", "BUILDNEW", "PRECD", "TMPD", "PREC", "TMP",
         "DEM", "SLOPE", "ASPECT", "SOIL", "POP00", "DROAD"]
TARGET = "DNDVI"
PARAMS = dict(n_estimators=800, max_depth=4, learning_rate=0.05,
              min_child_weight=5, reg_lambda=1.0, gamma=0.0,
              subsample=1.0, colsample_bytree=1.0,
              objective="reg:squarederror", tree_method="hist",
              random_state=2024, n_jobs=-1)


def load(cat="native"):
    g = gpd.read_file(PTS)
    X = g[FEATS].copy()
    if cat == "native":
        X["SOIL"] = X["SOIL"].astype(int).astype("category")
    elif cat == "onehot":
        X = pd.get_dummies(X, columns=["SOIL"], drop_first=True)
    elif cat == "numeric":
        pass
    y = g[TARGET].to_numpy()
    return g, X, y


def fit_predict(Xtr, ytr, Xte):
    m = XGBRegressor(**PARAMS, enable_categorical=True)
    m.fit(Xtr, ytr, verbose=False)
    return m.predict(Xte)


def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))))
