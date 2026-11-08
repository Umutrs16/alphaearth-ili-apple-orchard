# -*- coding: utf-8 -*-
"""v4 retrain: 图11 candidate for an EXPANDED-training model - wall-to-wall Gongliu map (30 m or 10 m) + four zooms cropped from the SAME
model's county map (10 m if available, else 30 m), official GS(2024)0650 boundary. Args: SCALE TAG CMDIR."""
import sys, json
from pathlib import Path
import numpy as np, rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager
from shapely.geometry import box
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision/figures_v3_scripts"))
import official_boundary as ob
from v3_polys import build
SCALE = int(sys.argv[1]); TAG = sys.argv[2]; CMN = Path(sys.argv[3])
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "font.size": 11})
O = ROOT / "outputs/revision/v4_experiments"; CM = O / "county_map"; MV = ROOT / "outputs/revision/maps_v2"
meta = json.load(open(CMN / f"county_map_meta_{SCALE}m.json")); zmeta = json.load(open(MV / "MAP_META.json"))
DEC = 3 if SCALE == 10 else 1
def rd(p, dtype=None):
    with rasterio.open(p) as s:
        a = s.read(1, out_shape=(s.height // DEC, s.width // DEC), resampling=Resampling.nearest); b = s.bounds
    return a, (b.left, b.right, b.bottom, b.top)
pu, ext = rd(CMN / f"gongliu_{TAG}_prob_{SCALE}m.tif"); prob = np.where(pu == 255, np.nan, pu / 250.0)
cu, _ = rd(CMN / f"gongliu_{TAG}_class_{SCALE}m.tif"); cls = np.where(cu == 255, np.nan, cu.astype(float))
elev = None
ep = CM / "ref_elev_30m.tif"
if ep.exists():
    with rasterio.open(ep) as s: elev = s.read(1, out_shape=(s.height // (1 if SCALE == 30 else 1), s.width), resampling=Resampling.nearest).astype(float); eext = (s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top)
gl = ob.gongliu().geometry.iloc[0]; H = build(); H = H[H.split == "val_holdout"]
fig = plt.figure(figsize=(15.5, 8.6))
gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[0.9, 1.0], hspace=0.12, wspace=0.10)
axa = fig.add_subplot(gs[0, 0:2]); axb = fig.add_subplot(gs[0, 2:4])
ima = axa.imshow(prob, extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
axb.imshow(cls, extent=ext, cmap=ListedColormap(["#2c3e50", "#f1c40f"]), vmin=0, vmax=1, interpolation="nearest")
if elev is not None:
    hi = np.where(np.isfinite(np.where(np.isnan(np.repeat(np.repeat(cls, 1, 0), 1, 1)), np.nan, 1)) if False else None, 1, 1) if False else None
    E = np.where(elev > 1217, 1.0, np.nan)
    with rasterio.open(CM / "gongliu_B3poly2021v2_class_30m.tif") as s: inside = s.read(1) != 255
    E[~inside] = np.nan
    axb.contourf(np.linspace(eext[0], eext[1], E.shape[1]), np.linspace(eext[3], eext[2], E.shape[0]), np.nan_to_num(E, nan=0), levels=[0.5, 1.5],
                 colors="none", hatches=["////"], zorder=4)
    axb.contour(np.linspace(eext[0], eext[1], E.shape[1]), np.linspace(eext[3], eext[2], E.shape[0]), np.nan_to_num(E, nan=0), levels=[0.5], colors="white", linewidths=0.4, zorder=4)
for ax in (axa, axb):
    for g in (gl.geoms if gl.geom_type == "MultiPolygon" else [gl]):
        x, y = g.exterior.xy; ax.plot(x, y, color="#e74c3c", lw=1.0, zorder=5)
    ax.scatter(H.geometry.representative_point().x, H.geometry.representative_point().y, s=6, c="white", edgecolors="k", lw=0.3, zorder=6)
    for zz in zmeta["zooms"]:
        w, s_, e, n = zz["bbox"]; ax.add_patch(plt.Rectangle((w, s_), e - w, n - s_, fill=False, ec="#ff00ff", lw=1.4, zorder=7))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.set_xticks([81.75, 82.25, 82.75, 83.25]); ax.tick_params(labelsize=9); ax.set_xlabel("经度 (°E)", fontsize=10)
axa.set_ylabel("纬度 (°N)", fontsize=10); axb.set_yticklabels([])
axa.set_title(f"(a) 巩留县全域苹果园概率 P(apple)（{SCALE} m，扩充训练 {TAG[:-8]}）", loc="left", fontsize=12)
axb.set_title(f"(b) 阈值0.5二值结果（黄=苹果园；斜线=高于原38面训练集最高高程1217 m）", loc="left", fontsize=12)
cax = axa.inset_axes([0.68, 0.90, 0.30, 0.035]); cb = fig.colorbar(ima, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8); cb.ax.set_title("P(apple)", fontsize=8, pad=2)
axb.legend(handles=[Patch(fc="#f1c40f", label="预测苹果园"), Patch(fc="#2c3e50", label="预测非苹果"), Patch(fc="white", ec="k", hatch="////", label="原模型高程外推区（>1217 m）"),
                    Line2D([0], [0], color="#e74c3c", lw=1.2, label="巩留县界"), Line2D([0], [0], marker="o", color="w", mfc="white", mec="k", ms=4, label="留出面"),
                    Line2D([0], [0], color="#ff00ff", lw=1.4, label="10 m 放大窗")], loc="lower left", fontsize=8, ncol=2, framealpha=0.95)
axb.text(0.995, 0.01, "审图号：GS（2024）0650号", transform=axb.transAxes, ha="right", va="bottom", fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1), zorder=9)
labels = {"c_nw_holdout": "西北留出簇", "d_se_holdout": "东南留出簇", "e_highprob": "原模型高概率区", "f_fp_prone": "原模型未验证区"}
cols = {1: ("#ff2d55", "-"), 2: ("#ff9f1c", "-"), 0: ("#00d4ff", "--")}
last = None
for i, (zz, let) in enumerate(zip(zmeta["zooms"], "cdef")):
    ax = fig.add_subplot(gs[1, i])
    with rasterio.open(zz["tif"]) as r: b = r.bounds
    e2 = (b.left, b.right, b.bottom, b.top)
    zsrc = CMN / f"gongliu_{TAG}_prob_10m.tif"; zres = 10
    if not zsrc.exists(): zsrc = CMN / f"gongliu_{TAG}_prob_30m.tif"; zres = 30
    with rasterio.open(zsrc) as r:
        win = rasterio.windows.from_bounds(b.left, b.bottom, b.right, b.top, transform=r.transform)
        pz = r.read(1, window=win, boundless=True, fill_value=255).astype(float)
    pz = np.where(pz == 255, np.nan, pz / 250.0)
    rgb = np.load(MV / "zooms" / f"{zz['id']}_s2rgb.npy")
    ax.imshow(rgb, extent=e2, zorder=0); last = ax.imshow(pz, extent=e2, cmap="viridis", vmin=0, vmax=1, alpha=0.55, interpolation="nearest", zorder=1)
    w, s_, e, n = zz["bbox"]
    for k, (c, ls) in cols.items():
        for g in H[(H.class_code == k) & H.intersects(box(w, s_, e, n))].geometry:
            for pp in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                x, y = pp.exterior.xy; ax.plot(x, y, color=c, lw=1.3, ls=ls, zorder=5)
    ax.set_xlim(e2[0], e2[1]); ax.set_ylim(e2[2], e2[3]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"({let}) {labels[zz['id']]}  {zz['lon']:.3f}°E, {zz['lat']:.3f}°N · {zres} m", fontsize=9.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#ff00ff"); sp.set_linewidth(1.5)
cbar_ax = fig.add_axes([0.30, 0.075, 0.40, 0.013]); cb2 = fig.colorbar(last, cax=cbar_ax, orientation="horizontal")
cb2.set_label(f"P(apple)（放大窗 {zres} m，半透明叠加于 Sentinel-2 2021 真彩色；线框为留出面：红=苹果园，橙=其他果园，青虚线=非果园；无叠加处为县界外）", fontsize=8.5)
out = CMN.parent / f"fig11_county_map_v4_{TAG}_{SCALE}m.png"; fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print(out)
