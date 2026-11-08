import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import pyogrio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import uniform_filter

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

V = r"D:\GFG\var250"
X = r"D:\GFG\xgb"
FIG = r"D:\GFG\fig"
CRS = None
EXT = None


def load(path):
    with rasterio.open(path) as d:
        a = d.read(1).astype(float)
        nod = d.nodata
        global CRS, EXT
        CRS = d.crs
        EXT = [d.bounds.left, d.bounds.right, d.bounds.bottom, d.bounds.top]
    a[a == nod] = np.nan
    return a


gfg = load(f"{V}\\gfg_frac_250.tif")
contrib = load(f"{X}\\gfg_contrib_250.tif")
pred = load(f"{X}\\pred_real_250.tif")
obs = load(f"{V}\\dndvi_250.tif")

_m = ~np.isnan(contrib)
_f = np.nan_to_num(contrib)
_num = uniform_filter(_f, size=5, mode="constant")
_den = uniform_filter(_m.astype(float), size=5, mode="constant")
contrib_s = _num / np.maximum(_den, 1e-6)
contrib_s[_den < 0.5] = np.nan

cty = gpd.read_file(r"D:\GFG\raw\bnd\610600_full.json").to_crs(CRS)
bnd = gpd.read_file(r"D:\GFG\raw\bnd\yanan_bnd.shp").to_crs(CRS)


def decorate(ax, title, letter=None):
    cty.boundary.plot(ax=ax, color="#8a8a8a", linewidth=0.35)
    bnd.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, pad=6)
    if letter:
        ax.text(0.02, 0.97, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")


def scale_north(ax):
    sx = EXT[0] + 8000; sy = EXT[2] + 9000
    ax.plot([sx, sx + 20000], [sy, sy], color="black", lw=2.5, solid_capstyle="butt")
    ax.plot([sx, sx], [sy - 1200, sy + 1200], color="black", lw=1.2)
    ax.plot([sx + 20000, sx + 20000], [sy - 1200, sy + 1200], color="black", lw=1.2)
    ax.text(sx + 10000, sy + 2500, "20 km", ha="center", va="bottom", fontsize=7)
    ax.annotate("", xy=(EXT[1] - 12000, EXT[3] - 7000),
                xytext=(EXT[1] - 12000, EXT[3] - 24000),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    ax.text(EXT[1] - 12000, EXT[3] - 4000, "N", ha="center", va="bottom",
            fontsize=9, fontweight="bold")


# ---------- 图 7-2 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=220)
im1 = axes[0].imshow(gfg, cmap="YlGn", vmin=0, vmax=0.35, extent=EXT, interpolation="nearest")
decorate(axes[0], "（a）退耕比例")
cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.035, pad=0.01)
cb1.set_label("退耕面积占比", fontsize=8); cb1.ax.tick_params(labelsize=7)

norm = TwoSlopeNorm(vmin=-0.012, vcenter=0.0, vmax=0.022)
im2 = axes[1].imshow(contrib_s, cmap="RdBu_r", norm=norm, extent=EXT, interpolation="nearest")
decorate(axes[1], "（b）退耕贡献")
cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.035, pad=0.01)
cb2.set_label("ΔNDVI 增量", fontsize=8); cb2.ax.tick_params(labelsize=7)
scale_north(axes[0])
fig.suptitle("延安市退耕比例与退耕贡献（250 m，随机森林反事实）", fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig(f"{FIG}\\fig7_2_gfg_frac_contrib.png", bbox_inches="tight")
print("ok fig7_2")

# ---------- 图 7-3 ----------
imp = pyogrio.read_dataframe(r"D:\GFG\pro\GFG.gdb", layer="vimp1000_3")
imp = imp.sort_values("Importance", ascending=True)
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=220)
bars = ax.barh(imp["Variables"], imp["Importance"] * 100, color="#2f6b4f", height=0.65)
for b, v in zip(bars, imp["Importance"] * 100):
    ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8)
ax.set_xlabel("重要性（Gain，%）")
ax.set_xlim(0, 52)
ax.set_title("变量重要性", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}\\fig7_3_vimp.png", bbox_inches="tight")
print("ok fig7_3")

# ---------- 预测 vs 观测 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=220)
im1 = axes[0].imshow(pred, cmap="YlGn", vmin=0, vmax=0.35, extent=EXT, interpolation="nearest")
decorate(axes[0], "（a）模型预测 ΔNDVI")
cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.035, pad=0.01)
cb1.set_label("ΔNDVI", fontsize=8); cb1.ax.tick_params(labelsize=7)
im2 = axes[1].imshow(obs, cmap="YlGn", vmin=0, vmax=0.35, extent=EXT, interpolation="nearest")
decorate(axes[1], "（b）观测 ΔNDVI")
cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.035, pad=0.01)
cb2.set_label("ΔNDVI", fontsize=8); cb2.ax.tick_params(labelsize=7)
scale_north(axes[0])
fig.suptitle("预测与观测 ΔNDVI 对比（250 m）", fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig(f"{FIG}\\fig_pred_vs_obs.png", bbox_inches="tight")
print("ok pred_vs_obs")

# ---------- Excel ----------
acc = pd.DataFrame({
    "项目": ["样本量", "树数", "最大树深", "最小叶子", "随机变量数", "每棵树可用数据",
             "训练 R²", "验证 R²", "训练 RMSE", "验证 RMSE", "训练 MAE", "验证 MAE"],
    "值": [9939, 1000, 12, 5, 5, "100%", 0.90, 0.85, 0.03, 0.04, 0.02, 0.03],
})
imp_out = imp.sort_values("Importance", ascending=False).copy()
imp_out["重要性_%"] = (imp_out["Importance"] * 100).round(1)
imp_out = imp_out[["Variables", "重要性_%"]].rename(columns={"Variables": "变量"})
contrib_tab = pd.read_excel(r"D:\GFG\tab\t7_contrib.xlsx", sheet_name="贡献分区统计")

with pd.ExcelWriter(r"D:\GFG\tab\t7_figures_data.xlsx", engine="openpyxl") as w:
    acc.to_excel(w, sheet_name="模型精度", index=False)
    imp_out.to_excel(w, sheet_name="变量重要性", index=False)
    contrib_tab.to_excel(w, sheet_name="贡献分区", index=False)
print("ok excel")
