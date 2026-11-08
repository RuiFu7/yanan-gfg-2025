import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import common as C

VAR = r"D:\GFG\var250"
FILES = {
    "GFG": "gfg_frac_250.tif", "CROPKEPT": "crop_kept_frac_250.tif",
    "FOR99": "forest99_frac_250.tif", "BUILDNEW": "build_new_frac_250.tif",
    "PRECD": "precd_250.tif", "TMPD": "tmpd_250.tif", "PREC": "prec0025_250.tif",
    "TMP": "tmp0025_250.tif", "DEM": "dem_250.tif", "SLOPE": "slope_250.tif",
    "ASPECT": "aspect_250.tif", "SOIL": "soil_250.tif", "POP00": "pop2000_250.tif",
    "DROAD": "dist_road_250.tif",
}

with rasterio.open(f"{VAR}\\dndvi_250.tif") as d:
    prof = d.profile
    dndvi = d.read(1)
    nod = d.nodata
    transform, shape = d.transform, d.shape

arrs = {}
for k, f in FILES.items():
    with rasterio.open(f"{VAR}\\{f}") as d:
        arrs[k] = d.read(1)

valid = (dndvi != nod)
for k in arrs:
    valid &= (arrs[k] != nod) & np.isfinite(arrs[k])
print("valid pixels:", valid.sum())

g, X, y = C.load("native")
m = C.fit_predict  # not used; train full model
from xgboost import XGBRegressor
model = XGBRegressor(**C.PARAMS, enable_categorical=True)
model.fit(X, y, verbose=False)

idx = np.where(valid.ravel())[0]
df = pd.DataFrame({k: arrs[k].ravel()[idx] for k in FILES})
df["SOIL"] = df["SOIL"].astype(int).astype("category")
# align categories to training
df["SOIL"] = df["SOIL"].cat.set_categories(X["SOIL"].cat.categories)
pred_real = model.predict(df)

cf = df.copy()
cf["CROPKEPT"] = np.clip(cf["CROPKEPT"] + cf["GFG"], 0, 1)
cf["GFG"] = 0.0
pred_cf = model.predict(cf)

contrib = pred_real - pred_cf
obs = dndvi.ravel()[idx]
gfg = df["GFG"].to_numpy()

print(f"city mean obs={obs.mean():.4f} pred_real={pred_real.mean():.4f} "
      f"contrib={contrib.mean():.4f} share={contrib.mean()/obs.mean()*100:.2f}%")
print(f"GFG area share: >0 {np.mean(gfg>0)*100:.2f}%   >0.3 {np.mean(gfg>0.3)*100:.2f}%")

rows = []
for name, msk in [("全市", np.ones_like(gfg, bool)),
                  ("退耕比例>0", gfg > 0),
                  ("退耕比例>0.3", gfg > 0.3)]:
    rows.append(dict(分区=name, 像元数=int(msk.sum()), 观测=obs[msk].mean(),
                     退耕贡献=contrib[msk].mean(),
                     占观测=contrib[msk].mean() / obs[msk].mean() * 100,
                     反事实=pred_cf[msk].mean(), 实测减反事实=obs[msk].mean() - pred_cf[msk].mean()))

# slope classes
with rasterio.open(f"{VAR}\\slope_cls_250.tif") as d:
    sc = d.read(1).ravel()[idx]
for k in range(1, 6):
    msk = sc == k
    if msk.sum() == 0:
        continue
    rows.append(dict(分区=f"坡度等级{k}", 像元数=int(msk.sum()), 观测=obs[msk].mean(),
                     退耕贡献=contrib[msk].mean(),
                     占观测=contrib[msk].mean() / obs[msk].mean() * 100,
                     反事实=pred_cf[msk].mean(), 实测减反事实=obs[msk].mean() - pred_cf[msk].mean()))

# counties
cty = gpd.read_file(r"D:\GFG\raw\bnd\610600_full.json").to_crs(prof["crs"])
name_field = "name" if "name" in cty.columns else cty.columns[1]
cty["cid"] = np.arange(1, len(cty) + 1)
rst = rasterize(((geom, c) for geom, c in zip(cty.geometry, cty["cid"])),
                out_shape=shape, transform=transform, fill=0, dtype="int32").ravel()[idx]
for _, r in cty.iterrows():
    msk = rst == r["cid"]
    if msk.sum() == 0:
        continue
    rows.append(dict(分区=str(r[name_field]), 像元数=int(msk.sum()), 观测=obs[msk].mean(),
                     退耕贡献=contrib[msk].mean(),
                     占观测=contrib[msk].mean() / obs[msk].mean() * 100,
                     反事实=pred_cf[msk].mean(), 实测减反事实=obs[msk].mean() - pred_cf[msk].mean()))

tab = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(tab.round(4).to_string(index=False))
tab.to_csv(r"E:\zcodedata\.zcode\workspace\draft-7b3e295b-0f02-46aa-b58d-6b65ef80f64f\blkcv\contrib_zonal.csv", index=False)

np.save(r"E:\zcodedata\.zcode\workspace\draft-7b3e295b-0f02-46aa-b58d-6b65ef80f64f\blkcv\contrib_flat.npy",
        np.c_[obs, pred_real, pred_cf, contrib, gfg].astype(np.float32))
print("saved")
