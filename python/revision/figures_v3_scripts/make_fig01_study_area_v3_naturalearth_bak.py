"""Figure 1 v3: study-area cascade; panel (d) = v3 QC'd holdout by class + training + 600 visual pts."""
from pathlib import Path
import sys
import geopandas as gpd, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
sys.path.insert(0, str(Path(__file__).parent))
from v3_polys import build

ROOT = Path("/workspace/apple_orchard_project")
OUT = ROOT / "outputs/revision/figures_v3"; OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "outputs/revision/basemaps"
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False,
                     "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7})
C_XJ, C_ILI, C_GONG = "#f4a261", "#2a9d8f", "#e76f51"
C_APPLE, C_OTHER, C_NON = "#d62728", "#ff9f1c", "#1f77b4"
C_TRAIN = "#555555"; C_SENS = "#2ca02c"

def north(ax, x=0.93, y=0.93, s=0.07):
    ax.annotate("N", xy=(x, y), xytext=(x, y - s), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))

def scalebar(ax, km, loc):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    deg = km / (111.32 * np.cos(np.deg2rad(np.mean(yl))))
    x0 = xl[0] + (xl[1] - xl[0]) * loc[0]; y0 = yl[0] + (yl[1] - yl[0]) * loc[1]
    dy = (yl[1] - yl[0]) * 0.01
    ax.plot([x0, x0 + deg], [y0, y0], "k", lw=2.2, solid_capstyle="butt", zorder=10)
    for xx in (x0, x0 + deg): ax.plot([xx, xx], [y0 - dy, y0 + dy], "k", lw=1.1, zorder=10)
    ax.text(x0 + deg / 2, y0 + 2 * dy, f"{km} km", ha="center", va="bottom", fontsize=7, zorder=10)

def style(ax):
    ax.set_aspect("equal", adjustable="box"); ax.tick_params(length=2, pad=1)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)

countries = gpd.read_file(BASE / "ne_110m_admin_0/ne_110m_admin_0_countries.shp")
china = countries[countries.ADMIN == "China"]
nbr = countries[countries.ADMIN.isin(["Mongolia", "Russia", "Kazakhstan", "Kyrgyzstan", "India", "Nepal",
      "Bhutan", "Myanmar", "Laos", "Vietnam", "North Korea", "South Korea", "Japan", "Taiwan", "Tajikistan", "Pakistan", "Afghanistan"])]
prov = gpd.read_file(BASE / "ne_50m_admin_1/ne_50m_admin_1_states_provinces.shp")
prov = prov[prov.admin == "China"]; xj = prov[prov.name == "Xinjiang"]
counties = gpd.read_file(ROOT / "data/roi/Yili_county_full.geojson").to_crs(4326)
gl = counties[counties.name == "巩留县"]
sens = pd.read_csv(ROOT / "data/interpretation_kit/Gongliu_independent_val_points.csv", usecols=["lon", "lat"])
P = build()
cen = P.geometry.to_crs(32644).centroid.to_crs(4326)
P["cenx"], P["ceny"] = cen.x.values, cen.y.values
tr = P[P.split == "train"]; ho = P[P.split == "val_holdout"]
ib = counties.total_bounds; minx, miny, maxx, maxy = ib

fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), gridspec_kw=dict(height_ratios=[1, 0.95]))
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.97, wspace=0.14, hspace=0.12)

# (a)
ax = axes[0, 0]
nbr.boundary.plot(ax=ax, lw=0.3, color="#cccccc")
china.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#444", lw=0.6)
prov.boundary.plot(ax=ax, lw=0.25, color="#bbbbbb")
xj.plot(ax=ax, facecolor=C_XJ, edgecolor="#9a3412", lw=0.9)
ax.add_patch(mpatches.Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, ec=C_ILI, lw=1.5, zorder=5))
ax.text(87, 41.5, "新疆", fontsize=9, fontweight="bold", color="#7c2d12", ha="center")
ax.annotate("伊犁河谷", xy=((minx + maxx) / 2, maxy), xytext=((minx + maxx) / 2 - 2, maxy + 5),
            fontsize=7.5, color=C_ILI, ha="center", arrowprops=dict(arrowstyle="-", color=C_ILI, lw=0.7))
ax.set_xlim(72, 136); ax.set_ylim(17, 55)
ax.set_title("(a) 中国 · 新疆", fontsize=10, fontweight="bold", loc="left"); style(ax); north(ax)

# (b)
ax = axes[0, 1]
prov.boundary.plot(ax=ax, lw=0.3, color="#dddddd")
xj.plot(ax=ax, facecolor="#fff7ed", edgecolor="#9a3412", lw=0.9)
counties.plot(ax=ax, facecolor=C_ILI, edgecolor="#0f766e", lw=0.4, alpha=0.85)
gl.plot(ax=ax, facecolor=C_GONG, edgecolor="#7c2d12", lw=0.8)
xb = xj.total_bounds; ax.set_xlim(xb[0] - 1, xb[2] + 1); ax.set_ylim(xb[1] - 1, xb[3] + 1)
gp = gl.geometry.iloc[0].representative_point()
ax.annotate("巩留县", xy=(gp.x, gp.y), xytext=(gp.x + 3, gp.y - 2.2), fontsize=8, color=C_GONG,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_GONG, lw=0.9))
ax.set_title("(b) 新疆 · 伊犁河谷", fontsize=10, fontweight="bold", loc="left"); style(ax); north(ax)
ax.legend(handles=[mpatches.Patch(fc=C_ILI, ec="#0f766e", label="伊犁河谷11县市"),
                   mpatches.Patch(fc=C_GONG, ec="#7c2d12", label="巩留县（空间留出县）")],
          loc="lower left", fontsize=7, frameon=True, fancybox=False, edgecolor="#ccc")

# (c)
ax = axes[1, 0]
counties.plot(ax=ax, facecolor="#e8f5f3", edgecolor="#555", lw=0.55)
gl.plot(ax=ax, facecolor="#fbd3c4", edgecolor="#7c2d12", lw=1.2)
for _, r in counties.iterrows():
    c = r.geometry.representative_point()
    if r["name"] == "巩留县":
        from shapely.geometry import Point; c = Point(83.05, 43.13)
    nm = r["name"].replace("察布查尔锡伯自治县", "察布查尔").replace("霍尔果斯市", "霍尔果斯").replace("可克达拉市", "可克达拉")
    ax.text(c.x, c.y, nm, fontsize=6.3 if nm != "巩留县" else 7.5, ha="center", va="center",
            color="#7c2d12" if nm == "巩留县" else "#333", fontweight="bold" if nm == "巩留县" else "normal", zorder=3)
ax.scatter(tr.cenx, tr.ceny, s=16, facecolors="none", edgecolors=C_TRAIN, lw=0.8, marker="o", zorder=5)
for k, col, mk in [(0, C_NON, "s"), (2, C_OTHER, "D"), (1, C_APPLE, "^")]:
    s = ho[ho.class_code == k]
    ax.scatter(s.cenx, s.ceny, s=16, c=col, marker=mk, edgecolors="white", lw=0.3, zorder=6)
ax.set_xlim(ib[0] - 0.12, ib[2] + 0.12); ax.set_ylim(ib[1] - 0.12, ib[3] + 0.12)
ax.set_title("(c) 伊犁河谷县市 · 训练面与留出面质心", fontsize=10, fontweight="bold", loc="left")
style(ax); north(ax); scalebar(ax, 50, (0.06, 0.06))

# (d)
ax = axes[1, 1]
counties.plot(ax=ax, facecolor="#fafafa", edgecolor="#bbbbbb", lw=0.5, zorder=0)
gl.plot(ax=ax, facecolor="#fff5f0", edgecolor="#7c2d12", lw=1.3, zorder=1)
ax.scatter(sens.lon, sens.lat, s=3, c=C_SENS, alpha=0.5, lw=0, zorder=2)
ax.scatter(tr.cenx, tr.ceny, s=22, facecolors="none", edgecolors=C_TRAIN, lw=0.9, marker="o", zorder=5)
for k, col, mk in [(0, C_NON, "s"), (2, C_OTHER, "D"), (1, C_APPLE, "^")]:
    s = ho[ho.class_code == k]
    ax.scatter(s.cenx, s.ceny, s=20, c=col, marker=mk, edgecolors="white", lw=0.35, zorder=6)
xl = (81.55, 83.47); yl = (42.90, 44.02)
ax.set_xlim(*xl); ax.set_ylim(*yl)
for nm, (x, y) in {"巩留县": (82.9, 43.05), "特克斯县": (81.75, 43.0), "伊宁县": (81.65, 43.93), "新源县": (82.35, 43.80)}.items():
    if nm != "巩留县": ax.text(x, y, nm, fontsize=6.5, color="#888", ha="center")
    else: ax.text(x, y, nm, fontsize=8, color="#7c2d12", fontweight="bold", ha="center")
ax.set_title("(d) 巩留县：v3 留出园块（按类别）与目视敏感性点", fontsize=10, fontweight="bold", loc="left")
style(ax); north(ax, 0.95, 0.30, 0.07); scalebar(ax, 20, (0.30, 0.04))
# inset: dense NW cluster, show real polygon shapes
ins = inset_axes(ax, width="34%", height="34%", loc="upper right", borderpad=0.5)
for k, col in [(0, C_NON), (2, C_OTHER), (1, C_APPLE)]:
    s = ho[ho.class_code == k]
    if len(s): s.plot(ax=ins, facecolor=col, edgecolor="k", lw=0.5, alpha=0.85)
ins.set_xlim(82.402, 82.440); ins.set_ylim(43.483, 43.512); ins.set_aspect("equal")
ins.set_xticks([]); ins.set_yticks([]); ins.set_title("局部放大：园块边界", fontsize=6.5, pad=1.5)
mark_inset(ax, ins, loc1=2, loc2=3, fc="none", ec="#555", lw=0.7)

n = ho.class_code.value_counts()
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=C_TRAIN, markersize=7,
           label=f"训练面（非巩留，n={len(tr)}：苹果{(tr.class_code==1).sum()}/非果园{(tr.class_code==0).sum()}）"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_APPLE, markersize=8, label=f"留出·苹果园面（n={n[1]}）"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=C_OTHER, markersize=7, label=f"留出·其他果园面（n={n[2]}）"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_NON, markersize=7, label=f"留出·非果园面（n={n[0]}）"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SENS, markersize=5, label="目视敏感性点（n=600，辅助）"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=True, fancybox=False,
           edgecolor="#ccc", bbox_to_anchor=(0.5, 0.005), columnspacing=1.5)
out = OUT / "fig01_study_area_v3.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig01_study_area_v3.pdf", bbox_inches="tight", facecolor="white")
print(out, n.to_dict(), len(tr))
