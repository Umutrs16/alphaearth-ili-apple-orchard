# -*- coding: utf-8 -*-
"""v4.1 review item A: diagnose low AOMC-2021 vs expanded-B3 overlap in Gongliu.
(1) CRS/transform of both rasters; (2) overlap under different grids / aggregation rules;
(3) systematic-offset test: shift AOMC by -3..+3 px (native 30 m) and recompute overlap with (a) our map and (b) our digitized
apple / non-apple polygons (independent of any model); (4) same for our 10 m map vs challenge-set polygons (±3 px at 10 m)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision/figures_v3_scripts")); import official_boundary as ob
V4 = ROOT / "outputs/revision/v4_experiments"; E = V4 / "retrain_expanded"; OUT = ROOT / "outputs/revision/v41_review/aomc_alignment"; OUT.mkdir(parents=True, exist_ok=True)
AOMC = ROOT / "data/aomc/AOMC_30_Xinjiang_2021.tif"
M10 = E / "county_map_B3/gongliu_B3expanded_class_10m.tif"; M30 = E / "county_map_B3/gongliu_B3expanded_class_30m.tif"
P10 = E / "county_map_B3/gongliu_B3expanded_prob_10m.tif"
res = {}
for k, p in (("AOMC_2021", AOMC), ("B3exp_10m", M10), ("B3exp_30m", M30)):
    with rasterio.open(p) as s:
        res.setdefault("rasters", {})[k] = {"crs": s.crs.to_string(), "crs_wkt_name": s.crs.to_wkt().split('"')[1], "transform": list(s.transform)[:6],
                                            "res": list(s.res), "shape": [s.height, s.width], "nodata": s.nodata, "dtype": s.dtypes[0]}
# ---------- AOMC native window over Gongliu (+margin) and our 10 m map resampled onto it
gl = ob.gongliu(); glb = gl.total_bounds
with rasterio.open(AOMC) as s:
    b = transform_bounds("EPSG:4326", s.crs, glb[0] - 0.05, glb[1] - 0.05, glb[2] + 0.05, glb[3] + 0.05)
    w = from_bounds(*b, s.transform).round_offsets().round_lengths(); A = s.read(1, window=w); TA = s.window_transform(w); CA = s.crs
res["aomc_window_values"] = {str(int(v)): int(c) for v, c in zip(*np.unique(A, return_counts=True))}
H, W = A.shape
cmask = rasterize([(g, 1) for g in gl.to_crs(CA).geometry], out_shape=(H, W), transform=TA, fill=0, dtype="uint8").astype(bool)
with rasterio.open(M10) as s:
    m10 = s.read(1); T10 = s.transform; C10 = s.crs
frac = np.full((H, W), np.nan, np.float32)
src = np.where(m10 == 255, np.nan, (m10 == 1).astype(np.float32))
reproject(src, frac, src_transform=T10, src_crs=C10, dst_transform=TA, dst_crs=CA, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
valid = cmask & np.isfinite(frac)
aomc1 = (A == 1)
def ov(mp, ref, v):
    mp = mp & v; ref = ref & v; o = (mp & ref).sum()
    return {"map_px": int(mp.sum()), "ref_px": int(ref.sum()), "overlap_px": int(o), "frac_map_in_ref": float(o / max(mp.sum(), 1)), "frac_ref_in_map": float(o / max(ref.sum(), 1)),
            "IoU": float(o / max((mp | ref).sum(), 1))}
res["native_AOMC_grid"] = {"our10m_to_30m_majority(frac>=0.5)": ov(frac >= 0.5, aomc1, valid), "our10m_to_30m_any(frac>0)": ov(frac > 0, aomc1, valid),
                           "our10m_to_30m_frac>=0.25": ov(frac >= 0.25, aomc1, valid)}
# 30 m map (AE resampled to 30 m in GEE then classified) onto AOMC grid, nearest
with rasterio.open(M30) as s:
    m30 = s.read(1); T30 = s.transform
m30a = np.full((H, W), 255, np.uint8)
reproject(m30, m30a, src_transform=T30, src_crs=C10, dst_transform=TA, dst_crs=CA, resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
res["native_AOMC_grid"]["our30m_nearest"] = ov(m30a == 1, aomc1, cmask & (m30a != 255))
# AOMC -> our 10 m grid (nearest)
a10 = np.zeros(m10.shape, np.uint8)
with rasterio.open(AOMC) as s:
    reproject(rasterio.band(s, 1), a10, dst_transform=T10, dst_crs=C10, resampling=Resampling.nearest)
res["our_10m_grid"] = {"AOMC_nearest_vs_our10m": ov(m10 == 1, a10 == 1, m10 != 255)}
# ---------- shift test AOMC vs our map (native grid)
def shift(a, dy, dx, fill=False):
    o = np.full_like(a, fill); H_, W_ = a.shape
    ys, yd = (slice(0, H_ - dy), slice(dy, H_)) if dy >= 0 else (slice(-dy, H_), slice(0, H_ + dy))
    xs, xd = (slice(0, W_ - dx), slice(dx, W_)) if dx >= 0 else (slice(-dx, W_), slice(0, W_ + dx))
    o[yd, xd] = a[ys, xs]; return o
S = []
maj = frac >= 0.5
for dy in range(-3, 4):
    for dx in range(-3, 4):
        a = shift(aomc1, dy, dx); r = ov(maj, a, valid); r.update(dy=dy, dx=dx); S.append(r)
S = pd.DataFrame(S); S.to_csv(OUT / "shift_AOMC_vs_B3exp_majority30m.csv", index=False)
best = S.loc[S.IoU.idxmax()]; res["shift_AOMC_vs_map"] = {"best_dy_dx": [int(best.dy), int(best.dx)], "best_IoU": float(best.IoU),
      "IoU_at_0": float(S[(S.dy == 0) & (S.dx == 0)].IoU.iloc[0]), "IoU_range": [float(S.IoU.min()), float(S.IoU.max())]}
# ---------- AOMC vs our digitized polygons (all counties), shift test; model independent
P = pd.concat([gpd.read_file(ROOT / "outputs/revision/polygon_samples/v3/polygons_v3_all.geojson")[["name", "class_code", "split", "geometry"]],
               gpd.read_file(E / "training_new_kept.geojson").rename(columns={"poly_name": "name"}).assign(split="train_new")[["name", "class_code", "split", "geometry"]]], ignore_index=True)
with rasterio.open(AOMC) as s:
    b = transform_bounds("EPSG:4326", s.crs, *(P.total_bounds + np.array([-0.02, -0.02, 0.02, 0.02])))
    w = from_bounds(*b, s.transform).round_offsets().round_lengths(); A2 = s.read(1, window=w); T2 = s.window_transform(w)
Pp = P.to_crs(CA)
ids = rasterize([(g, i + 1) for i, g in enumerate(Pp.geometry)], out_shape=A2.shape, transform=T2, fill=0, dtype="int32")
cls = np.zeros_like(ids) - 1; cc = P.class_code.values
cls[ids > 0] = cc[ids[ids > 0] - 1]
SP = []
for dy in range(-3, 4):
    for dx in range(-3, 4):
        a = shift(A2 == 1, dy, dx)
        SP.append({"dy": dy, "dx": dx, "apple_poly_px_AOMC_frac": float(a[cls == 1].mean()), "other_orchard_px_AOMC_frac": float(a[cls == 2].mean()),
                   "nonorchard_px_AOMC_frac": float(a[cls == 0].mean())})
SP = pd.DataFrame(SP); SP.to_csv(OUT / "shift_AOMC_vs_digitized_polygons.csv", index=False)
b2 = SP.loc[SP.apple_poly_px_AOMC_frac.idxmax()]
res["shift_AOMC_vs_polygons"] = {"n_polys": {int(k): int(v) for k, v in P.class_code.value_counts().items()}, "px_by_class": {int(k): int((cls == k).sum()) for k in (0, 1, 2)},
      "at_0": SP[(SP.dy == 0) & (SP.dx == 0)].iloc[0].to_dict(), "best_apple": b2.to_dict()}
# per-polygon AOMC coverage (no shift)
pc = []
for i, r in P.iterrows():
    m = ids == i + 1
    if m.sum() == 0: continue
    pc.append({"name": r["name"], "class_code": int(r.class_code), "split": r.split, "n_px": int(m.sum()), "aomc_frac": float((A2[m] == 1).mean())})
pc = pd.DataFrame(pc); pc.to_csv(OUT / "polygon_AOMC_coverage.csv", index=False)
res["polygon_AOMC_coverage"] = {str(c): {"n": int(len(g)), "any_AOMC": int((g.aomc_frac > 0).sum()), "ge50pct_AOMC": int((g.aomc_frac >= 0.5).sum()),
      "mean_frac": float(g.aomc_frac.mean())} for c, g in pc.groupby("class_code")}
res["polygon_AOMC_coverage_holdout"] = {str(c): {"n": int(len(g)), "any_AOMC": int((g.aomc_frac > 0).sum()), "ge50pct_AOMC": int((g.aomc_frac >= 0.5).sum())}
      for c, g in pc[pc.split == "val_holdout"].groupby("class_code")}
# ---------- our 10 m map vs holdout polygons, shift ±3 px (10 m)
H10 = P[P.split == "val_holdout"]
hid = rasterize([(g, int(c) + 1) for g, c in zip(H10.geometry, H10.class_code)], out_shape=m10.shape, transform=T10, fill=0, dtype="uint8")
SM = []
for dy in range(-3, 4):
    for dx in range(-3, 4):
        m = shift(m10 == 1, dy, dx)
        SM.append({"dy": dy, "dx": dx, "apple_px_mapped": float(m[hid == 2].mean()), "nonapple_px_mapped": float(m[(hid == 1) | (hid == 3)].mean())})
SM = pd.DataFrame(SM); SM.to_csv(OUT / "shift_B3exp10m_vs_holdout_polygons.csv", index=False)
res["shift_map_vs_holdout"] = {"at_0": SM[(SM.dy == 0) & (SM.dx == 0)].iloc[0].to_dict(), "best": SM.loc[SM.apple_px_mapped.idxmax()].to_dict()}
# ---------- where are AOMC apple px classified by our prob? prob distribution inside AOMC
with rasterio.open(P10) as s: p10r = s.read(1)
p10 = np.where(p10r == 255, np.nan, p10r / 250.0).astype(np.float32)  # stored as uint8 p*250, 255 nodata
inA = (a10 == 1) & (m10 != 255)
res["our_prob_inside_AOMC_apple_10m"] = {"n_px": int(inA.sum()), "median_prob": float(np.nanmedian(p10[inA])), "frac_ge0.5": float((p10[inA] >= 0.5).mean()),
      "frac_ge0.3": float((p10[inA] >= 0.3).mean()), "frac_lt0.1": float((p10[inA] < 0.1).mean())}
np.save(OUT / "_frac_on_aomc_grid.npy", frac); json.dump({"TA": list(TA)[:6], "crs": CA.to_wkt()}, open(OUT / "_aomc_grid.json", "w"))
json.dump(res, open(OUT / "aomc_alignment_diagnostics.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print(S.pivot(index="dy", columns="dx", values="IoU").round(4)); print(SP.pivot(index="dy", columns="dx", values="apple_poly_px_AOMC_frac").round(3))
print(SM.pivot(index="dy", columns="dx", values="apple_px_mapped").round(3))
