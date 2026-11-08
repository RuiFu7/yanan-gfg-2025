import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

V = r"D:\GFG\chg"
PX = 0.0009


def rd(n):
    with rasterio.open(rf"{V}\{n}.tif") as d:
        return d.read(1), d.nodata, d.transform, d.shape, d.crs


g, ng, T, shp, crs = rd("gfg30")
g0, ng0, _, _, _ = rd("gfg30_00")
c99, n99, _, _, _ = rd("crop99_30")
c25, n25, _, _, _ = rd("crop25_30")
sc, nsc, _, _, _ = rd("slope_cls30")

# ---- slope table: merge class 1 (slope=0) into <6 ----
rows = []
for lab, ks in [("<6°", (1, 2)), ("6—15°", (3,)), ("15—25°", (4,)), ("≥25°", (5,))]:
    m = np.isin(sc, ks) & (g != ng) & (c99 != n99) & (c25 != n25)
    a = c99[m].sum() * PX
    b = (g[m] == 1).sum() * PX
    b0 = (g0[m] == 1).sum() * PX
    e = (c25[m] == 1).sum() * PX
    rows.append({"坡度": lab, "1999耕地": round(a, 1), "遥感退耕": round(b, 1),
                 "退耕率1999起点": round(b / a * 100, 1),
                 "退耕率2000起点": round(b0 / (a - b + b0) * 100, 1),
                 "2025耕地": round(e, 1)})
slope_tab = pd.DataFrame(rows)
print(slope_tab.to_string(index=False))

# ---- county table ----
cty = gpd.read_file(r"D:\GFG\raw\bnd\610600_full.json").to_crs(crs)
cty["cid"] = np.arange(1, len(cty) + 1)
rst = rasterize(((geom, c) for geom, c in zip(cty.geometry, cty.cid)),
                out_shape=shp, transform=T, fill=0, dtype="int32")
rows = []
for _, r in cty.iterrows():
    m = (rst == r["cid"]) & (g != ng) & (c99 != n99) & (c25 != n25)
    a = c99[m].sum() * PX
    b = (g[m] == 1).sum() * PX
    e = (c25[m] == 1).sum() * PX
    rows.append({"县区": r["name"], "1999耕地": round(a, 1), "遥感退耕": round(b, 1),
                 "退耕率": round(b / a * 100, 1), "2025耕地": round(e, 1)})
cty_tab = pd.DataFrame(rows).sort_values("遥感退耕", ascending=False)
print(cty_tab.to_string(index=False))

with pd.ExcelWriter(r"D:\GFG\tab\t4_gfg_slope.xlsx", engine="openpyxl") as w:
    slope_tab.to_excel(w, sheet_name="分坡度", index=False)
out_county = r"D:\GFG\tab\t4_gfg_county_2025.xlsx"
try:
    with pd.ExcelWriter(r"D:\GFG\tab\t4_gfg_county.xlsx", engine="openpyxl") as w:
        cty_tab.to_excel(w, sheet_name="分县", index=False)
    out_county = r"D:\GFG\tab\t4_gfg_county.xlsx"
except PermissionError:
    with pd.ExcelWriter(out_county, engine="openpyxl") as w:
        cty_tab.to_excel(w, sheet_name="分县", index=False)
print("written t4_gfg_slope.xlsx /", out_county)

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200,
                               gridspec_kw={"width_ratios": [1, 1.15]})
x = np.arange(len(slope_tab))
w = 0.38
b1 = ax1.bar(x - w / 2, slope_tab["退耕率1999起点"], w, label="1999 起点", color="#1b6b3a")
b2 = ax1.bar(x + w / 2, slope_tab["退耕率2000起点"], w, label="2000 起点", color="#8ccf7e")
for bars in (b1, b2):
    for r in bars:
        ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 1,
                 f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=8)
ax1.set_xticks(x, slope_tab["坡度"])
ax1.set_ylabel("退耕率 (%)")
ax1.set_ylim(0, 100)
ax1.set_title("按坡度：退耕率（30 m，→2025）", fontsize=11)
ax1.legend(frameon=False, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)

ct = cty_tab.iloc[::-1]
y = np.arange(len(ct))
ax2.barh(y, ct["遥感退耕"], color="#1b6b3a", height=0.65)
for yy, (_, r) in zip(y, ct.iterrows()):
    ax2.text(r["遥感退耕"] + 6, yy, f'{r["遥感退耕"]:.0f} km² ({r["退耕率"]:.0f}%)',
             va="center", fontsize=8)
ax2.set_yticks(y, ct["县区"])
ax2.set_xlabel("遥感退耕面积 (km²)")
ax2.set_xlim(0, ct["遥感退耕"].max() * 1.32)
ax2.set_title("分县遥感退耕（括号内为占 1999 耕地比例）", fontsize=11)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = r"D:\GFG\fig\fig4_3_gfg_slope_county.png"
fig.savefig(out, bbox_inches="tight")
print("written", out)
