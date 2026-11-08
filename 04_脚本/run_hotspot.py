import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.stats import norm
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

BAND = 3000
SRC = r"D:\GFG\var250\gfg_frac_250.tif"
OUT_TIF = r"D:\GFG\chg\gfg_hotspot_1km.tif"
OUT_Z = r"D:\GFG\chg\gfg_giz_1km.tif"
OUT_SHP = r"D:\GFG\chg\gfg_hotspot_1km.shp"
OUT_PNG = r"D:\GFG\fig\fig_gfg_hotspot.png"

with rasterio.open(SRC) as d:
    a = d.read(1).astype("float64")
    nod = d.nodata
    T = d.transform
    crs = d.crs
    left, top = T.c, T.f
a[a == nod] = np.nan

H, W = a.shape
H4 = (H + 3) // 4 * 4
W4 = (W + 3) // 4 * 4
pad = np.full((H4, W4), np.nan)
pad[:H, :W] = a
blk = pad.reshape(H4 // 4, 4, W4 // 4, 4)
cnt = np.sum(~np.isnan(blk), axis=(1, 3))
agg = np.nanmean(blk, axis=(1, 3))
agg[cnt < 8] = np.nan
rows, cols = np.where(~np.isnan(agg))
vals = agg[rows, cols]
x = left + 500 + 1000 * cols
y = top - 500 - 1000 * rows
pts = np.c_[x, y]
n = len(vals)
print("1 km cells:", n)

tree = cKDTree(pts)
pairs = tree.query_pairs(r=BAND, output_type="ndarray")
i = np.r_[pairs[:, 0], pairs[:, 1], np.arange(n)]
j = np.r_[pairs[:, 1], pairs[:, 0], np.arange(n)]
Wm = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
sum_w = np.asarray(Wm.sum(axis=1)).ravel()
sum_w2 = np.asarray(Wm.multiply(Wm).sum(axis=1)).ravel()
sum_wx = Wm @ vals
xbar = vals.mean()
S = np.sqrt(np.sum(vals ** 2) / n - xbar ** 2)
z = (sum_wx - xbar * sum_w) / (S * np.sqrt((n * sum_w2 - sum_w ** 2) / (n - 1)))
p = 2 * (1 - norm.cdf(np.abs(z)))

order = np.argsort(p)
rank = np.empty(n, int)
rank[order] = np.arange(1, n + 1)
p_adj = np.minimum(1, p * n / rank)
ps = p_adj[order]
ps = np.minimum.accumulate(ps[::-1])[::-1]
p_adj[order] = ps


def bin_from_p(zv, pv):
    b = np.zeros(len(zv), int)
    for th, k in [(0.10, 1), (0.05, 2), (0.01, 3)]:
        b[(pv < th) & (zv > 0)] = k
        b[(pv < th) & (zv < 0)] = -k
    return b


gi = bin_from_p(z, p_adj)
print("Gi_Bin:", dict(zip(*np.unique(gi, return_counts=True))))

grid_bin = np.full(agg.shape, -128, dtype="int16")
grid_z = np.full(agg.shape, np.nan, dtype="float32")
grid_bin[rows, cols] = gi
grid_z[rows, cols] = z
T1 = from_origin(left, top, 1000, 1000)
prof = dict(driver="GTiff", width=agg.shape[1], height=agg.shape[0], count=1,
            crs=crs, transform=T1, compress="deflate")
with rasterio.open(OUT_TIF, "w", dtype="int16", nodata=-128, **prof) as w:
    w.write(grid_bin, 1)
with rasterio.open(OUT_Z, "w", dtype="float32", nodata=np.nan, **prof) as w:
    w.write(grid_z, 1)

geoms = [box(left + 1000 * c, top - 1000 * (r + 1), left + 1000 * (c + 1), top - 1000 * r)
         for r, c in zip(rows, cols)]
gdf = gpd.GeoDataFrame({"GFG": vals.round(4), "GiZ": z.round(3), "P": p.round(4),
                        "P_FDR": p_adj.round(4), "Gi_Bin": gi},
                       geometry=geoms, crs=crs)
gdf.to_file(OUT_SHP, encoding="UTF-8")

cty = gpd.read_file(r"D:\GFG\raw\bnd\610600_full.json").to_crs(crs)
bnd = gpd.read_file(r"D:\GFG\raw\bnd\yanan_bnd.shp").to_crs(crs)

colors = {3: "#7f0000", 2: "#d7301f", 1: "#fc8d59", 0: "#ededed",
          -1: "#9ecae1", -2: "#4292c6", -3: "#08519c"}
cmap = ListedColormap([colors[k] for k in range(-3, 4)])
normc = BoundaryNorm(np.arange(-3.5, 4.5, 1), cmap.N)
show = np.ma.masked_where(grid_bin == -128, grid_bin)
ext = [left, left + 1000 * agg.shape[1], top - 1000 * agg.shape[0], top]

fig, ax = plt.subplots(figsize=(10, 8), dpi=220)
ax.imshow(show, cmap=cmap, norm=normc, extent=ext, interpolation="nearest")
cty.boundary.plot(ax=ax, color="#7a7a7a", linewidth=0.45)
bnd.boundary.plot(ax=ax, color="black", linewidth=1.0)
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

# scale bar 20 km
sx, sy = ext[0] + 8000, ext[2] + 9000
ax.plot([sx, sx + 20000], [sy, sy], color="black", lw=3, solid_capstyle="butt")
ax.plot([sx, sx], [sy - 1500, sy + 1500], color="black", lw=1.5)
ax.plot([sx + 20000, sx + 20000], [sy - 1500, sy + 1500], color="black", lw=1.5)
ax.text(sx + 10000, sy + 3000, "20 km", ha="center", va="bottom", fontsize=8)

# north arrow
ax.annotate("", xy=(ext[1] - 12000, ext[3] - 8000), xytext=(ext[1] - 12000, ext[3] - 26000),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
ax.text(ext[1] - 12000, ext[3] - 4500, "N", ha="center", va="bottom", fontsize=10, fontweight="bold")

labels = ["99% 冷点", "95% 冷点", "90% 冷点", "不显著", "90% 热点", "95% 热点", "99% 热点"]
handles = [Patch(facecolor=colors[k], edgecolor="#666", label=labels[i]) for i, k in enumerate(range(-3, 4))]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
          frameon=False, title="Getis-Ord Gi*", title_fontsize=10)
ax.set_title("延安市退耕比例热点分析（1 km 网格，3 km 距离带，FDR 校正）", fontsize=12, pad=12)
fig.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight")
print("written", OUT_PNG)

pts_gdf = gpd.GeoDataFrame({"Gi_Bin": gi, "GFG": vals},
                           geometry=gpd.points_from_xy(x, y), crs=crs)
join = gpd.sjoin(pts_gdf, cty[["name", "geometry"]], predicate="within", how="left")
tab = join.groupby("name").apply(
    lambda d: pd.Series({"格数": len(d),
                         "热点占比%": round((d.Gi_Bin > 0).mean() * 100, 1),
                         "冷点占比%": round((d.Gi_Bin < 0).mean() * 100, 1),
                         "平均退耕比例": round(d.GFG.mean(), 3)}),
    include_groups=False).sort_values("热点占比%", ascending=False)
print(tab.to_string())
tab.to_excel(r"D:\GFG\tab\t_gfg_hotspot_county.xlsx")
print("written D:\\GFG\\tab\\t_gfg_hotspot_county.xlsx")
