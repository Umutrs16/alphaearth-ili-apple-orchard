"""Re-render Gongliu map (same B3_poly_v2 predictions, cached) with fixed layout, no embedded figure number,
v3 holdout polygons overlaid by class. S2 RGB for zoom backgrounds fetched from GEE (fallback: none)."""
import sys, importlib.util, json
from pathlib import Path
import numpy as np, rasterio, geopandas as gpd
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib import font_manager
from shapely.geometry import box
ROOT = Path("/workspace/apple_orchard_project")
spec = importlib.util.spec_from_file_location("m20", ROOT / "python/revision/20_gongliu_hires_map.py")
m20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m20)
sys.path.insert(0, str(Path(__file__).parent)); from v3_polys import build
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})
MV = ROOT / "outputs/revision/maps_v2"; OUT = ROOT / "outputs/revision/figures_v3"
meta = json.load(open(MV / "MAP_META.json"))
z = np.load(MV / "gongliu_focus_map_2021_100m.npz")
prob, cls = z["prob"], z["cls"]; W, E, S, N = z["extent"]
import official_boundary as ob
county = ob.gongliu().geometry.iloc[0]   # official GS(2024)0650 boundary (display only; cached predictions unchanged)
P = build(); H = P[P.split == "val_holdout"]
cols = {1: ("#ff2d55", "-"), 2: ("#ff9f1c", "-"), 0: ("#00d4ff", "--")}
def draw(ax, gdf, lw):
    for k, (c, ls) in cols.items():
        for g in gdf[gdf.class_code == k].geometry:
            for pp in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                x, y = pp.exterior.xy; ax.plot(x, y, color=c, lw=lw, ls=ls, zorder=5)
RGB_CACHE = MV / "zooms"
try:
    m20.init_ee(); ee_ok = True
except Exception as e:
    print("EE init failed", e); ee_ok = False
fig = plt.figure(figsize=(14.5, 9.0))
gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1.0], hspace=0.25, wspace=0.12)
axa = fig.add_subplot(gs[0, 0:2]); axb = fig.add_subplot(gs[0, 2:4])
cg = county.intersection(box(*m20.FOCUS_BBOX)); cgs = list(cg.geoms) if cg.geom_type == "MultiPolygon" else [cg]
ima = axa.imshow(prob, origin="upper", extent=[W, E, S, N], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
cs = cls.astype(float); cs[cls < 0] = np.nan
axb.imshow(cs, origin="upper", extent=[W, E, S, N], cmap=ListedColormap(["#2c3e50", "#f1c40f"]), vmin=0, vmax=1, interpolation="nearest")
for ax in (axa, axb):
    for g in cgs:
        x, y = g.exterior.xy; ax.plot(x, y, color="white", lw=1.0)
    draw(ax, H, 0.8)
    for zz in meta["zooms"]:
        w, s, e, n = zz["bbox"]; ax.add_patch(plt.Rectangle((w, s), e - w, n - s, fill=False, ec="#ffcc00", lw=1.3, zorder=6))
    ax.set_xlim(W, E); ax.set_ylim(S, N); ax.set_xlabel("经度 (°E)", fontsize=9); ax.tick_params(labelsize=8)
axa.set_ylabel("纬度 (°N)", fontsize=9); axb.set_yticklabels([])
axa.set_title("(a) 焦点区苹果园概率 P(apple)（约100 m）", fontsize=10)
axb.set_title("(b) 阈值0.5二值结果（黄=苹果园）", fontsize=10)
axb.text(0.99, 0.01, "审图号：GS（2024）0650号", transform=axb.transAxes, ha="right", va="bottom", fontsize=6.5, color="#333", zorder=9, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
cax = axa.inset_axes([0.86, 0.52, 0.025, 0.42]); cb = fig.colorbar(ima, cax=cax); cb.ax.tick_params(labelsize=7); cb.set_label("P(apple)", fontsize=7)
axb.legend(handles=[Line2D([0], [0], color="#ff2d55", lw=2, label="留出·苹果园"),
                    Line2D([0], [0], color="#ff9f1c", lw=2, label="留出·其他果园"),
                    Line2D([0], [0], color="#00d4ff", lw=2, ls="--", label="留出·非果园"),
                    Line2D([0], [0], color="#ffcc00", lw=2, label="10 m 放大窗")], loc="upper right", fontsize=7.5, framealpha=0.95)
labels = {"c_nw_holdout": "西北留出簇", "d_se_holdout": "东南留出簇", "e_highprob": "高概率斑块", "f_fp_prone": "未验证高概率区"}
last = None
for i, (zz, let) in enumerate(zip(meta["zooms"], "cdef")):
    ax = fig.add_subplot(gs[1, i])
    with rasterio.open(zz["tif"]) as r:
        pz = r.read(1).astype(float); b = r.bounds
    pz[pz < 0] = np.nan
    ext = (b.left, b.right, b.bottom, b.top)
    rgb = None
    rp = RGB_CACHE / f"{zz['id']}_s2rgb.npy"
    if rp.exists(): rgb = np.load(rp)
    elif ee_ok:
        rgb = m20.try_s2_rgb(*zz["bbox"], scale=20)
        if rgb is not None: np.save(rp, rgb)
    if rgb is not None:
        ax.imshow(rgb, origin="upper", extent=ext, zorder=0)
        last = ax.imshow(pz, origin="upper", extent=ext, cmap="viridis", vmin=0, vmax=1, alpha=0.55, interpolation="nearest", zorder=1)
    else:
        last = ax.imshow(pz, origin="upper", extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    w, s, e, n = zz["bbox"]; draw(ax, H[H.intersects(box(w, s, e, n))], 1.3)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"({let}) {labels[zz['id']]}\n{zz['lon']:.3f}°E, {zz['lat']:.3f}°N · 10 m", fontsize=9)
    print(zz["id"], "rgb" if rgb is not None else "no-rgb")
cbar_ax = fig.add_axes([0.30, 0.07, 0.40, 0.014])
cb2 = fig.colorbar(last, cax=cbar_ax, orientation="horizontal"); cb2.set_label("P(apple)（放大窗，叠加于 Sentinel-2 2021 真彩色）", fontsize=8)
out = OUT / "fig11_gongliu_map_2021_v3.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print(out)
