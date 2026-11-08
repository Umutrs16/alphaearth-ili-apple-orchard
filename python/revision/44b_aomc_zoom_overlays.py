# -*- coding: utf-8 -*-
"""A (visual): zoom overlays of AOMC-2021 apple pixels vs expanded-B3 10 m apple map vs challenge-set polygons,
on Sentinel-2 2021 (Jun-Sep median, true colour) and Esri World Imagery (acquisition date unknown/current).
Windows (1.5 km) chosen automatically on the 10 m grid: max AOMC density, max model density, max overlap, and the densest holdout apple cluster.
Overview panel uses the official GS(2024)0650 Gongliu boundary."""
import sys, io, json, math, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, ee
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
from shapely.geometry import shape, box
from PIL import Image
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision")); import official_boundary as ob
sys.path.insert(0, str(ROOT / "outputs/revision/v4_writing/figures")); from v4style import plt, CM
E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; OUT = ROOT / "outputs/revision/v41_review/aomc_alignment"; FO = ROOT / "outputs/revision/v41_review/figures"
TC = OUT / "_tiles"; TC.mkdir(exist_ok=True)
with rasterio.open(E / "county_map_B3/gongliu_B3expanded_class_10m.tif") as s: m10 = s.read(1); T10 = s.transform; C10 = s.crs
a10 = np.zeros(m10.shape, np.uint8)
with rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2021.tif") as s:
    reproject(rasterio.band(s, 1), a10, dst_transform=T10, dst_crs=C10, resampling=Resampling.nearest)
a10[m10 == 255] = 0
P = gpd.read_file(ROOT / "outputs/revision/polygon_samples/v3/polygons_v3_all.geojson"); H = P[P.split == "val_holdout"]
B = 167  # ~1.5 km
def blocksum(x):
    h, w = (x.shape[0] // B) * B, (x.shape[1] // B) * B
    return x[:h, :w].reshape(h // B, B, w // B, B).sum((1, 3))
sa, sm, so = blocksum(a10 == 1), blocksum(m10 == 1), blocksum((a10 == 1) & (m10 == 1))
from rasterio.features import rasterize
hap = rasterize([(g, 1) for g in H[H.class_code == 1].geometry], out_shape=m10.shape, transform=T10, fill=0, dtype="uint8"); sh = blocksum(hap == 1)
chosen = []
def pick(S, name):
    S = S.astype(float).copy()
    for (i, j) in [c[1] for c in chosen]: S[max(0, i - 2):i + 3, max(0, j - 2):j + 3] = -1
    i, j = np.unravel_index(np.argmax(S), S.shape); chosen.append((name, (i, j)))
pick(sa, "AOMC密集区"); pick(sm, "模型苹果密集区"); pick(so, "二者重叠最多区"); pick(sh, "挑战集苹果面集中区")
wins = []
for name, (i, j) in chosen:
    x0, y0 = T10 * (j * B, i * B); x1, y1 = T10 * ((j + 1) * B, (i + 1) * B)
    r0, c0 = i * B, j * B; a = a10[r0:r0 + B, c0:c0 + B]; m = m10[r0:r0 + B, c0:c0 + B]
    wins.append({"name": name, "bbox": [x0, y1, x1, y0], "aomc_px": int((a == 1).sum()), "model_px": int((m == 1).sum()), "overlap_px": int(((a == 1) & (m == 1)).sum())})
json.dump(wins, open(OUT / "zoom_windows.json", "w"), ensure_ascii=False, indent=1)
ee.Initialize(project="hallowed-chain-458005-g9")
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate("2021-06-01", "2021-10-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).median() \
     .select(["B4", "B3", "B2"]).visualize(min=0, max=2200, gamma=1.2)
def s2img(bb, k):
    p = TC / f"s2_{k}.tif"
    if not p.exists():
        url = s2.getDownloadURL({"region": ee.Geometry.Rectangle(bb, proj="EPSG:4326", geodesic=False), "scale": 10, "crs": "EPSG:4326", "format": "GEO_TIFF"})
        urllib.request.urlretrieve(url, p)
    with rasterio.open(p) as s: return np.moveaxis(s.read()[:3], 0, -1), rasterio.plot.plotting_extent(s) if False else (s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top)
def deg2num(lon, lat, z):
    n = 2 ** z; return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
def num2deg(x, y, z):
    n = 2 ** z; return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
def esri(bb, z=16):
    x0, y0 = deg2num(bb[0], bb[3], z); x1, y1 = deg2num(bb[2], bb[1], z)
    tx = range(int(x0), int(x1) + 1); ty = range(int(y0), int(y1) + 1)
    im = Image.new("RGB", (256 * len(tx), 256 * len(ty)))
    for a, x in enumerate(tx):
        for b_, y in enumerate(ty):
            p = TC / f"esri_{z}_{x}_{y}.jpg"
            if not p.exists():
                req = urllib.request.Request(f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", headers={"User-Agent": "Mozilla/5.0"})
                p.write_bytes(urllib.request.urlopen(req, timeout=30).read())
            im.paste(Image.open(p).convert("RGB"), (256 * a, 256 * b_))
    # tile grid is Mercator in y; for a 1.5 km window the lat distortion is negligible -> map rows linearly in lat of tile edges
    L, T_ = num2deg(tx[0], ty[0], z); R_, Bt = num2deg(tx[-1] + 1, ty[-1] + 1, z)
    return np.asarray(im), (L, R_, Bt, T_)
def outlines(mask, r0, c0):
    t = T10 * rasterio.Affine.translation(c0, r0)
    return [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=t) if v == 1]
fig = plt.figure(figsize=(17.5 * CM, 21 * CM))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])
axo = fig.add_subplot(gs[0, 0])
g = ob.gongliu(); g.boundary.plot(ax=axo, color="k", lw=0.6)
yy, xx = np.nonzero(a10[::5, ::5] == 1); x, y = T10 * (xx * 5, yy * 5); axo.scatter(x, y, s=0.2, c="#00bcd4", lw=0, label="AOMC 2021苹果")
yy, xx = np.nonzero(m10[::5, ::5] == 1); x, y = T10 * (xx * 5, yy * 5); axo.scatter(x, y, s=0.2, c="#e91e63", lw=0, label="新B3苹果（10 m）")
for k, w in enumerate(wins):
    bb = w["bbox"]; axo.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="orange", lw=1.2))
    axo.text(bb[2], bb[3], str(k + 1), color="orange", fontsize=8, weight="bold")
axo.set_title("(a) 巩留县总览与放大窗口", loc="left"); axo.set_xticks([]); axo.set_yticks([]); axo.set_aspect(1 / math.cos(math.radians(43.4)))
axo.legend(frameon=False, markerscale=15, fontsize=6, loc="lower left")
pos = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
lab = "bcdefghi"; kk = 0
for k, w in enumerate(wins):
    bb = w["bbox"]; i, j = chosen[k][1]; r0, c0 = i * B, j * B
    ao = outlines(a10[r0:r0 + B, c0:c0 + B] == 1, r0, c0); mo = outlines(m10[r0:r0 + B, c0:c0 + B] == 1, r0, c0)
    hp = H[H.intersects(box(*bb))]
    for src in ("s2", "esri"):
        if kk >= len(pos): break
        ax = fig.add_subplot(gs[pos[kk]])
        img, ext = s2img(bb, k) if src == "s2" else esri(bb)
        ax.imshow(img, extent=ext, interpolation="bilinear")
        for p in ao: gpd.GeoSeries([p]).boundary.plot(ax=ax, color="#00e5ff", lw=0.6)
        for p in mo: gpd.GeoSeries([p]).boundary.plot(ax=ax, color="#ff2d7a", lw=0.6)
        for r in hp.itertuples(): gpd.GeoSeries([r.geometry]).boundary.plot(ax=ax, color={1: "yellow", 2: "#7CFC00", 0: "white"}[r.class_code], lw=1.0, ls="--")
        ax.set_xlim(bb[0], bb[2]); ax.set_ylim(bb[1], bb[3]); ax.set_aspect(1 / math.cos(math.radians((bb[1] + bb[3]) / 2)))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({lab[kk]}) 窗口{k + 1} {w['name']}｜{'S2 2021' if src == 's2' else 'Esri高分影像'}", loc="left", fontsize=6.8)
        ax.text(0.02, 0.02, f"AOMC {w['aomc_px']}  模型 {w['model_px']}  重叠 {w['overlap_px']} 像元", transform=ax.transAxes, fontsize=5.5, color="white",
                bbox=dict(fc="black", alpha=0.5, lw=0, pad=1))
        kk += 1
from matplotlib.lines import Line2D
hs = [Line2D([], [], color="#00e5ff", lw=1, label="AOMC 2021苹果像元边界"), Line2D([], [], color="#ff2d7a", lw=1, label="新B3苹果像元边界（10 m，p≥0.5）"),
      Line2D([], [], color="yellow", lw=1, ls="--", label="挑战集苹果面"), Line2D([], [], color="#7CFC00", lw=1, ls="--", label="挑战集其他果园面"), Line2D([], [], color="grey", lw=1, ls="--", label="挑战集非果园面")]
fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FO / "figS7_aomc_vs_B3exp_zoom_v41.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.03)
print(json.dumps(wins, ensure_ascii=False, indent=1))
