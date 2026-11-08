# -*- coding: utf-8 -*-
"""新源 second-county holdout, step 3: digitizing kit. Candidates are drawn WITHOUT any model probability:
apple  : AOMC 2021 apple patches (>=3 px, 30 m) inside 新源 (official GS(2024)0650 boundary), stratified by SRTM elevation band
         (<900, 900-1200, 1200-1400, >=1400 m), random within band, >=500 m spacing, >=200 m from removed training polygons.
other orchard hints : ESA WorldCover 2021 tree cover <1400 m within 1 km of an AOMC (2019-2021) apple patch but not on one.
non-orchard : WorldCover 2021 x elevation-band stratified random points (cropland, tree/shelterbelt, grassland, built-up, bare, shrub),
         incl. >=1400 m mountain grass/forest; points on/adjacent to AOMC apple (any year) dropped.
Seed 20260927. Output folder: outputs/revision/xinyuan_holdout/kit"""
import sys, json, time, zlib
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, ee
from rasterio.mask import mask
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape, mapping, Point
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"
sys.path.insert(0, str(REV)); sys.path.insert(0, str(REV / "figures_v3_scripts"))
import official_boundary as ob
from kml_utils import write_kml, style
OUT = ROOT / "outputs/revision/xinyuan_holdout"; KIT = OUT / "kit"; KIT.mkdir(parents=True, exist_ok=True)
SEED = 20260927; rng = np.random.default_rng(SEED)
BANDS = [(0, 900, "<900"), (900, 1200, "900-1200"), (1200, 1400, "1200-1400"), (1400, 9000, ">=1400")]
WCN = {10: "tree", 20: "shrub", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
ALB = "ESRI:102025"
cty = ob.county(); XY = cty[cty.name == "新源县"]; xy = XY.geometry.iloc[0]; xy_u = XY.to_crs(32644).geometry.iloc[0]
removed = gpd.read_file(OUT / "removed_training_polygons.geojson")
rem_u = removed.to_crs(32644); rem_union = rem_u.unary_union
ee.Initialize(project="hallowed-chain-458005-g9")
dem = ee.Image("USGS/SRTMGL1_003").rename("elev")

def ee_elev(df, chunk=400):
    out = []
    for s in range(0, len(df), chunk):
        sub = df.iloc[s:s + chunk]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"i": int(i)}) for i, r in sub.iterrows()])
        for att in range(5):
            try:
                out += [(f["properties"]["i"], f["properties"].get("elev")) for f in dem.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()["features"]]; break
            except Exception as e: print("retry", att, str(e)[:120]); time.sleep(10)
    e = dict(out); return df.index.map(lambda i: e.get(int(i), np.nan))

def band_of(e):
    for lo, hi, l in BANDS:
        if lo <= e < hi: return l
    return "nodata"

# ---------------- AOMC patches (2021) + multi-year flags
from rasterio.warp import reproject, Resampling
A = {}
with rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2021.tif") as s:
    a, T = mask(s, XY.to_crs(ALB).geometry, crop=True, nodata=255); A[2021] = a[0]; CRS21 = s.crs
for yr in (2019, 2020):   # other years have different grids -> nearest onto the 2021 window
    with rasterio.open(ROOT / f"data/aomc/AOMC_30_Xinjiang_{yr}.tif") as s:
        dst = np.full(A[2021].shape, 255, np.uint8)
        reproject(rasterio.band(s, 1), dst, dst_transform=T, dst_crs=CRS21, resampling=Resampling.nearest, dst_nodata=255); A[yr] = dst
assert A[2019].shape == A[2021].shape
lab, n = ndimage.label(A[2021] == 1, structure=np.ones((3, 3))); sz = np.bincount(lab.ravel())
idx_all = np.arange(n + 1)
F20 = ndimage.mean((A[2020] == 1).astype(float), lab, idx_all); F19 = ndimage.mean((A[2019] == 1).astype(float), lab, idx_all)
rows = []
for geom, v in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=T, connectivity=8):
    v = int(v)
    if sz[v] < 3: continue
    rows.append({"patch": v, "n_px": int(sz[v]), "frac2020": float(F20[v]), "frac2019": float(F19[v]), "geometry": shape(geom)})
P = gpd.GeoDataFrame(rows, crs=ALB).dissolve("patch", aggfunc="first").reset_index()
P = P.to_crs(4326); P = P[P.within(xy)].copy()
Pu = P.to_crs(32644); P["area_ha"] = Pu.area.values / 1e4
c = Pu.centroid.to_crs(4326); P["lon"], P["lat"] = c.x.values, c.y.values
P["d_removed_m"] = Pu.distance(rem_union).values
P["elev"] = ee_elev(P[["lon", "lat"]]); P["band"] = P.elev.map(band_of)
P["aomc_years"] = 1 + (P.frac2019 >= 0.5).astype(int) + (P.frac2020 >= 0.5).astype(int)
print("AOMC 2021 patches >=3px inside 新源:", len(P), P.band.value_counts().to_dict())
P.drop(columns="geometry").to_csv(KIT / "_aomc_patches_xinyuan.csv", index=False)

def spaced_pick(df, n, taken_u, min_m):
    idx = rng.permutation(np.array(df.index.values)); pick = []
    for i in idx:
        if len(pick) >= n: break
        p = Point(*gpd.GeoSeries([Point(df.at[i, "lon"], df.at[i, "lat"])], crs=4326).to_crs(32644).iloc[0].coords[0])
        if all(p.distance(q) >= min_m for q in taken_u): pick.append(i); taken_u.append(p)
    return pick

taken = []
# apple: 20 per band target (80), shortfall redistributed to other bands
elig = P[(P.d_removed_m >= 200)]
alloc = {l: 20 for _, _, l in BANDS}; avail = elig.band.value_counts().to_dict()
short = sum(max(0, alloc[l] - avail.get(l, 0)) for l in alloc)
for l in alloc: alloc[l] = min(alloc[l], avail.get(l, 0))
while short > 0:
    grow = [l for l in alloc if avail.get(l, 0) > alloc[l]]
    if not grow: break
    for l in grow:
        if short == 0: break
        alloc[l] += 1; short -= 1
apple = []
for l, k in alloc.items():
    apple += spaced_pick(elig[elig.band == l], k, taken, 500)
AP = P.loc[apple].copy(); AP["cand_type"] = "c1_apple_AOMC_hint"
print("apple candidates by band:", AP.band.value_counts().to_dict(), "alloc", alloc)

# ---------------- WorldCover x elevation-band stratified random points
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
bandimg = dem.gte(900).add(dem.gte(1200)).add(dem.gte(1400))
st = wc.multiply(10).add(bandimg).rename("st")
region = ee.Geometry(mapping(xy.simplify(0.002)))
cls = [10, 20, 30, 40, 50, 60]; vals = [c_ * 10 + b for c_ in cls for b in range(4)]
for att in range(5):
    try:
        fc = st.stratifiedSample(numPoints=0, classBand="st", region=region, scale=30, seed=SEED % 100000, geometries=True,
                                 classValues=vals, classPoints=[40] * len(vals), tileScale=8)
        info = fc.getInfo()["features"]; break
    except Exception as e: print("retry wc", att, str(e)[:150]); time.sleep(15)
W = pd.DataFrame([{"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1], "st": int(f["properties"]["st"])} for f in info])
W["wc"] = W.st // 10; W["band"] = W.st % 10; W["band"] = W.band.map({i: b[2] for i, b in enumerate(BANDS)})
W = gpd.GeoDataFrame(W, geometry=gpd.points_from_xy(W.lon, W.lat), crs=4326); W = W[W.within(xy)].copy()
W["elev"] = ee_elev(W[["lon", "lat"]])
# AOMC any-year flag (3x3 neighbourhood) and distance to AOMC patches
anyA = ndimage.maximum_filter(((A[2019] == 1) | (A[2020] == 1) | (A[2021] == 1)).astype(np.uint8), size=3)
wa = W.to_crs(ALB); rc = [rasterio.transform.rowcol(T, x, y) for x, y in zip(wa.geometry.x, wa.geometry.y)]
W["aomc_adj"] = [int(anyA[r, c_]) if 0 <= r < anyA.shape[0] and 0 <= c_ < anyA.shape[1] else 0 for r, c_ in rc]
allA = gpd.GeoDataFrame(geometry=[shape(g) for g, v in features.shapes(((A[2019] == 1) | (A[2020] == 1) | (A[2021] == 1)).astype(np.uint8), mask=anyA > 0, transform=T) if v == 1], crs=ALB).to_crs(32644)
Wu = W.to_crs(32644); W["d_aomc_m"] = Wu.distance(allA.unary_union).values; W["d_removed_m"] = Wu.distance(rem_union).values
W = W[(W.d_removed_m >= 200)]
print("WorldCover strata points:", W.groupby(["wc", "band"]).size().to_dict())
# other-orchard hints: tree <1400 m, within 1 km of AOMC, not adjacent
o_pool = W[(W.wc == 10) & (W.band != ">=1400") & (W.d_aomc_m <= 1000) & (W.aomc_adj == 0)]
OO = W.loc[spaced_pick(o_pool, 20, taken, 500)].copy(); OO["cand_type"] = "c2_other_orchard_hint"
# non-orchard plan: (wc, band) -> n
plan = {(40, "<900"): 6, (40, "900-1200"): 6, (40, "1200-1400"): 4, (10, "<900"): 4, (10, "900-1200"): 4, (10, "1200-1400"): 2,
        (30, "<900"): 3, (30, "900-1200"): 4, (30, "1200-1400"): 3, (50, "<900"): 3, (50, "900-1200"): 3, (60, "<900"): 2, (60, "900-1200"): 2,
        (20, "900-1200"): 2, (20, "1200-1400"): 2, (30, ">=1400"): 6, (10, ">=1400"): 4, (20, ">=1400"): 1, (60, ">=1400"): 1}
NO = []
npool = W[W.aomc_adj == 0]
for (w, b), k in plan.items():
    pk = spaced_pick(npool[(npool.wc == w) & (npool.band == b)], k, taken, 500); NO += pk
    if len(pk) < k: print("shortfall", w, b, len(pk), "/", k)
NOd = W.loc[NO].copy(); NOd["cand_type"] = "c0_non_orchard_" + NOd.wc.map(WCN)
print("non-orchard candidates:", len(NOd), NOd.groupby(["wc", "band"]).size().to_dict())

# ---------------- assemble + IDs (apple 0001.., other orchard, non-orchard)
C = pd.concat([AP.assign(wc=np.nan).drop(columns="geometry"), OO.drop(columns="geometry"), NOd.drop(columns="geometry")], ignore_index=True)
C["pref_class"] = C.cand_type.str[:2]
C = C.sort_values(["pref_class", "band", "lat"], key=lambda s: s.map({"c1": 0, "c2": 1, "c0": 2}) if s.name == "pref_class" else s).reset_index(drop=True)
order = {"c1": 0, "c2": 1, "c0": 2}; C["o"] = C.pref_class.map(order); C = C.sort_values(["o", "band", "lat"]).reset_index(drop=True)
C["cand_id"] = [f"XY-HO-{i:04d}" for i in range(1, len(C) + 1)]
C["wc_name"] = C.wc.map(lambda v: WCN.get(int(v), "") if pd.notna(v) else "")
keep = ["cand_id", "pref_class", "cand_type", "lon", "lat", "elev", "band", "wc_name", "area_ha", "n_px", "aomc_years", "frac2019", "frac2020", "d_aomc_m", "d_removed_m"]
C[[k for k in keep if k in C]].to_csv(KIT / "xinyuan_candidates.csv", index=False, encoding="utf-8-sig")
pgeo = dict(zip(AP.index, AP.geometry))
AP_geoms = {r.cand_id: P.loc[P.patch == r.patch].geometry.iloc[0] for r in C.itertuples() if r.pref_class == "c1"}

# ---------------- KML
def desc(r):
    h = [f"候选类型={r.cand_type}", f"高程_SRTM={r.elev:.0f} m", f"高程带={r.band}"]
    if r.pref_class == "c1": h += [f"AOMC斑块={r.n_px} px ({r.area_ha:.2f} ha)", f"AOMC年份数(2019-2021)={int(r.aomc_years)}"]
    else: h += [f"WorldCover2021={r.wc_name}", f"距AOMC苹果斑块={r.d_aomc_m:.0f} m"]
    return ("【提示，仅供定位，不代表真实类别】 " + "; ".join(h) +
            "\n勾绘后把面命名为: " + f"{r.cand_id} | c?  (c1苹果 / c2其他果园 / c0非果园，按影像实际判定)" +
            "\n描述填写: img_date=YYYY-MM-DD; temporal_stable=Y/N; confidence=high/medium/low; subtype=; notes=")
S = [style("bd", "ff00ffff", width=3), style("rm", "ff9e9e9e", "4d9e9e9e", 2), style("c1", "ff0000ff", "330000ff", 2.5), style("c2", "ff00a5ff", "", 2),
     style("c0", "ffffff00", "", 2), style("tpl", "ff00ff00", "3300ff00", 2)]
fold = [("新源县边界 (GS(2024)0650)", [{"name": "新源县", "geom": xy.boundary.buffer(0.0003) if False else xy, "style": "bd", "desc": "官方县界 GS(2024)0650"}])]
fold[0][1][0]["geom"] = xy
fold.append(("已移除的训练面（仅作参考，勿在其上勾绘留出面）", [{"name": f"{r.name} | c{int(r.class_code)} (已移除:{r.drop_reason})", "geom": r.geometry, "style": "rm",
             "desc": f"训练面已从训练集移除 ({r.drop_reason}); 县={r.county_official}"} for r in removed.itertuples()]))
for pc, fname, sty in (("c1", "苹果候选 c1 (AOMC弱提示; 目标勾绘30–50个)", "c1"), ("c2", "其他果园候选 c2 (林地且邻近AOMC; 需目视确认)", "c2"),
                       ("c0", "非果园候选 c0 (耕地/防护林/草地/建设用地/裸地/灌丛/高海拔; 与c2合计目标30–50)", "c0")):
    items = []
    for r in C[C.pref_class == pc].itertuples():
        items.append({"name": f"{r.cand_id} (候选{pc})", "desc": desc(r), "geom": AP_geoms.get(r.cand_id), "point": (r.lon, r.lat), "style": sty})
    fold.append((fname, items))
fold.append(("在此文件夹勾绘留出面 (XY-HO-#### | cK)", []))
write_kml(KIT / "xinyuan_holdout_candidates.kml", "新源县 第二县域最终留出集 勾绘候选", S, fold,
          description="候选不使用任何模型概率生成。请在最后一个文件夹中勾绘，完成后导出为 xinyuan_holdout.kml")
# CSV template for record keeping
T2 = C[["cand_id", "pref_class", "lon", "lat", "band"]].copy()
for k in ("final_class", "img_date", "temporal_stable", "confidence", "subtype", "notes"): T2[k] = ""
T2.to_excel(KIT / "xinyuan_holdout_record_template.xlsx", index=False)
summ = {"seed": SEED, "aomc2021_patches_ge3px_in_xinyuan": int(len(P)), "aomc_patch_band_counts": P.band.value_counts().to_dict(),
        "apple_alloc": alloc, "n_candidates": {k: int(v) for k, v in C.pref_class.value_counts().items()},
        "by_class_band": C.groupby(["pref_class", "band"]).size().rename("n").reset_index().to_dict("records"),
        "nonorchard_by_type": NOd.cand_type.value_counts().to_dict(), "removed_training_reference": int(len(removed)),
        "model_probability_used": False}
json.dump(summ, open(KIT / "kit_summary.json", "w"), ensure_ascii=False, indent=1, default=int); print(json.dumps(summ, ensure_ascii=False, indent=1, default=int))
