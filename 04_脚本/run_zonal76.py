import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

V = r"D:\GFG\var250"
X = r"D:\GFG\xgb"

with rasterio.open(f"{V}\\dndvi_250.tif") as d:
    prof = d.profile
    obs = d.read(1)
    nod = d.nodata
    transform, shape = d.transform, d.shape

with rasterio.open(f"{X}\\gfg_contrib_250.tif") as d:
    contrib = d.read(1)
with rasterio.open(f"{V}\\gfg_frac_250.tif") as d:
    gfg = d.read(1)
with rasterio.open(f"{V}\\slope_250.tif") as d:
    slope = d.read(1)

valid = (obs != nod) & (contrib != nod) & (gfg != nod) & (slope != nod)
PX = 0.0625


def stat(mask, name):
    n = int(mask.sum())
    o = float(obs[mask].mean())
    c = float(contrib[mask].mean())
    return dict(区域=name, 像元数=n, 面积_km2=round(n * PX, 1),
                观测DNDVI=round(o, 4), 退耕贡献=round(c, 4),
                占观测_pct=round(c / o * 100, 2))


rows = [stat(valid, "全市"),
        stat(valid & (gfg > 0), "退耕比例 > 0"),
        stat(valid & (gfg > 0.3), "退耕比例 > 0.3")]

# fixed degree breaks computed from slope_250 (250 m slope = 3x3 mean of 30 m slope)
breaks = [(-1, 6, "<6°"), (6, 15, "6—15°"), (15, 25, "15—25°"),
          (25, 35, "25—35°"), (35, 91, "≥35°")]
for lo, hi, lab in breaks:
    m = valid & (slope >= lo) & (slope < hi)
    rows.append(stat(m, f"坡度 {lab}"))

# 4-class version used in section 4.5
for lo, hi, lab in [(-1, 6, "<6°"), (6, 15, "6—15°"), (15, 25, "15—25°"), (25, 91, "≥25°")]:
    m = valid & (slope >= lo) & (slope < hi)
    rows.append(stat(m, f"坡度(4级) {lab}"))

cty = gpd.read_file(r"D:\GFG\raw\bnd\610600_full.json").to_crs(prof["crs"])
cty["cid"] = np.arange(1, len(cty) + 1)
rst = rasterize(((g, c) for g, c in zip(cty.geometry, cty.cid)),
                out_shape=shape, transform=transform, fill=0, dtype="int32")
for _, r in cty.iterrows():
    m = valid & (rst == r["cid"])
    if m.sum():
        rows.append(stat(m, str(r["name"])))

tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

note = pd.DataFrame({"项": ["数据来源", "贡献定义", "坡度口径", "未含项", "说明"],
                     "说明": [
    "pred_real_250.tif - pred_nogfg_250.tif（ArcGIS 随机森林，1000 棵/深度 12/叶子 5，验证 R2 0.85）",
    "模型内反事实边际贡献：把退耕比例设为 0、退耕地加回耕地后的预测差，不是因果效应",
    "slope_cls_250.tif 与 slope_250 相关系数仅 0.03，不是坡度分级，本表改用 slope_250 固定分级重算",
    "降水变化贡献（需要第三次预测 pred_prec0，PRECD 置 0）",
    "观测 DNDVI 为 2000—2002 与 2023—2025 之差；面积 = 像元数 x 0.0625 km2",
]})

out = r"D:\GFG\tab\t7_contrib.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as w:
    tab.to_excel(w, sheet_name="贡献分区统计", index=False)
    note.to_excel(w, sheet_name="说明", index=False)
print("written", out)
