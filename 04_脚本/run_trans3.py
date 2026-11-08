import numpy as np
import pandas as pd
import rasterio

NAMES = {1: "耕地", 2: "林地", 3: "草地", 4: "水域", 5: "建设用地", 6: "未利用地"}
PX = 0.0009  # 30 m x 30 m, km2

arr = {}
for y in (1999, 2010, 2025):
    with rasterio.open(rf"D:\GFG\lu30\lu{y}.tif") as d:
        arr[y] = d.read(1)
        nod = d.nodata

valid = (arr[1999] != nod) & (arr[2010] != nod) & (arr[2025] != nod) & \
        (arr[1999] > 0) & (arr[2010] > 0) & (arr[2025] > 0)
print("valid pixels", valid.sum(), "area %.2f km2" % (valid.sum() * PX))


def matrix(y0, y1):
    ct = pd.crosstab(arr[y0][valid], arr[y1][valid]) * PX
    ct.index = [NAMES[i] for i in ct.index]
    ct.columns = [NAMES[i] for i in ct.columns]
    ct = ct.reindex(index=list(NAMES.values()), columns=list(NAMES.values()), fill_value=0)
    ct["合计"] = ct.sum(axis=1)
    ct.loc["合计"] = ct.sum(axis=0)
    return ct.round(2)


m1 = matrix(1999, 2010)
m2 = matrix(2010, 2025)
m3 = matrix(1999, 2025)

for name, m in [("1999→2010", m1), ("2010→2025", m2), ("1999→2025", m3)]:
    print("\n=====", name)
    print(m.to_string())

area = pd.DataFrame({
    "1999": [m3.loc[c, "合计"] for c in NAMES.values()],
    "2010": [m1.loc[c, "合计"] for c in NAMES.values()],
    "2025": [m2.loc[c, "合计"] for c in NAMES.values()],
}, index=list(NAMES.values()))
area["1999-2010"] = area["2010"] - area["1999"]
area["2010-2025"] = area["2025"] - area["2010"]
area["1999-2025"] = area["2025"] - area["1999"]
print("\n===== 面积")
print(area.round(2).to_string())

m1.to_csv(r"D:\GFG\tab\trans_1999_2010.csv", encoding="utf-8-sig")
m2.to_csv(r"D:\GFG\tab\trans_2010_2025.csv", encoding="utf-8-sig")

with pd.ExcelWriter(r"D:\GFG\tab\t_transition_3periods.xlsx", engine="openpyxl") as w:
    area.round(2).to_excel(w, sheet_name="面积变化")
    m1.to_excel(w, sheet_name="1999-2010")
    m2.to_excel(w, sheet_name="2010-2025")
    m3.to_excel(w, sheet_name="1999-2025")
print("\nwritten: trans_1999_2010.csv, trans_2010_2025.csv, t_transition_3periods.xlsx")

# 关键流量
print("\n耕地转出:")
for y0, y1, m, tag in [(1999, 2010, m1, "1999-2010"), (2010, 2025, m2, "2010-2025")]:
    row = m.loc["耕地"]
    out = row["合计"] - row["耕地"]
    print(f"  {tag}: 转出 {out:.2f} km2, 其中→林地 {row['林地']:.2f}, →草地 {row['草地']:.2f}, "
          f"→建设 {row['建设用地']:.2f}")
