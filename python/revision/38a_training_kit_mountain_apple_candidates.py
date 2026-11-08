# -*- coding: utf-8 -*-
"""v4 kit finalisation (stage a): extra candidate pools OUTSIDE Gongliu.
(1) Mountain stratum: ESA WorldCover v200 2021 (10 m) x SRTM elevation zones sampled in GEE in the
    counties around Gongliu (NOT Gongliu): zone T = 1217-1500 m (transition), zone M = >=1500 m.
    classes: 30 grassland/meadow, 20 shrub, 10 tree (mostly spruce at >=1500 m).
(2) Apple candidates: AOMC 2019/2020/2021 apple pixels (local 30 m rasters) outside Gongliu, preferably apple in all
    three years, spread over elevation bins; plus field-survey apple points not yet used as training polygons.
All candidates: AE 2021 sampled at 10 m and scored with frozen B3_poly_2021_v2; SRTM elevation sampled.
Also adds elevation to the existing 174 YL-TRN points (for the KML description)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, ee, joblib, rasterio
from shapely.geometry import mapping
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision"))
import official_boundary as ob
KIT = ROOT / "outputs/revision/v4_experiments/training_expansion_kit"
COUNTIES = ["特克斯县", "伊宁县", "新源县", "察布查尔锡伯自治县", "尼勒克县", "霍城县"]
c = ob.county(); reg = c[c.name.isin(COUNTIES)]
gl = ob.gongliu().geometry.iloc[0]
ee.Initialize(project="hallowed-chain-458005-g9")
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
ae = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2021-01-01", "2022-01-01").mosaic()

def getinfo(fn, tries=5):
    for a in range(tries):
        try: return fn()
        except Exception as e: print("retry", a, str(e)[:150], flush=True); time.sleep(10 * (a + 1))
    raise RuntimeError("GEE failed")

# ---------- (1) mountain pool
zone = ee.Image(0).where(dem.gte(1217).And(dem.lt(1500)), 1).where(dem.gte(1500), 2)
code = wc.multiply(10).add(zone).updateMask(zone.gt(0)).rename("code").toInt()
# codes: 301/302 grass T/M, 201/202 shrub T/M, 101/102 tree T/M
vals = [301, 302, 202, 102, 201]; npts = [40, 60, 40, 45, 10]
cand = []
for _, r in reg.iterrows():
    geom = ee.Geometry(mapping(r.geometry.simplify(0.002)))
    info = getinfo(lambda: code.stratifiedSample(numPoints=0, classBand="code", region=geom, scale=10, seed=7, geometries=True,
                                                 classValues=vals, classPoints=npts, tileScale=4).getInfo()["features"])
    for f in info:
        x, y = f["geometry"]["coordinates"]; cand.append({"lon": x, "lat": y, "code": f["properties"]["code"], "county": r["name"], "pool": "mountain"})
    print("mountain", r["name"], len(info), flush=True)
M = pd.DataFrame(cand); M["wc"] = M.code // 10; M["zone"] = np.where(M.code % 10 == 1, "T1217-1500", "M>=1500")

# ---------- (2) apple pool (AOMC 3-year consistent) + field points
rows = []
with rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2021.tif") as s21, rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2020.tif") as s20, \
        rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2019.tif") as s19:
    for _, r in reg.iterrows():
        g = gpd.GeoSeries([r.geometry], crs=4326).to_crs(s21.crs).iloc[0]
        win = rasterio.windows.from_bounds(*g.bounds, transform=s21.transform).round_offsets().round_lengths()
        a21 = s21.read(1, window=win); T = s21.window_transform(win)
        def rd(src):
            w2 = rasterio.windows.from_bounds(*g.bounds, transform=src.transform).round_offsets().round_lengths()
            a = src.read(1, window=w2, boundless=True, fill_value=0)
            return a[:a21.shape[0], :a21.shape[1]] if a.shape[0] >= a21.shape[0] and a.shape[1] >= a21.shape[1] else None
        a20, a19 = rd(s20), rd(s19)
        m = (a21 == 1)
        rr, cc = np.nonzero(m)
        xs, ys = rasterio.transform.xy(T, rr, cc)
        d = pd.DataFrame({"x": xs, "y": ys})
        d["aomc_years"] = 1 + (0 if a20 is None else (a20[rr, cc] == 1).astype(int)) + (0 if a19 is None else (a19[rr, cc] == 1).astype(int))
        pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.x, d.y), crs=s21.crs).to_crs(4326)
        pts = pts[pts.within(r.geometry)]
        pts = pts.sample(min(len(pts), 400), random_state=42, weights=(pts.aomc_years.values ** 3).astype(float))
        rows.append(pd.DataFrame({"lon": pts.geometry.x, "lat": pts.geometry.y, "aomc_years": pts.aomc_years, "county": r["name"], "pool": "aomc_apple"}))
        print("aomc", r["name"], int(m.sum()), "->", len(pts), flush=True)
A = pd.concat(rows, ignore_index=True)
F = pd.read_csv(ROOT / "data/field_survey/usable_candidates/ili_field_apple_usable.csv")
F = pd.DataFrame({"lon": F._lon, "lat": F._lat, "county": F.county, "pool": "field_apple", "field_note": F.text})
A = pd.concat([A, F], ignore_index=True); A["wc"] = np.nan

# ---------- existing 174
E = pd.read_csv(KIT / "YL_TRN_candidates.csv")
E0 = E[["id", "lon", "lat"]].assign(pool="existing")

allp = pd.concat([M.assign(src="M"), A.assign(src="A"), E0.assign(src="E")], ignore_index=True)
img = ae.addBands(dem.rename("elev")).addBands(wc.rename("wc_s"))
out = []
for s in range(0, len(allp), 200):
    sub = allp.iloc[s:s + 200]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"i": int(i)}) for i, r in sub.iterrows()])
    out += [f["properties"] for f in getinfo(lambda: img.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"])]
    print("sampled", s + len(sub), "/", len(allp), flush=True)
S = pd.DataFrame(out).set_index("i")
b = joblib.load(ROOT / "outputs/revision/polygon_samples/models/B3_poly_2021_v2.joblib")
S = S.rename(columns={f"A{i:02d}": f"AE_{i:02d}" for i in range(64)})
allp = allp.join(S[b["features"] + ["elev", "wc_s"]], how="left")
ok = allp[b["features"]].notna().all(1)
allp.loc[ok, "b3_prob"] = b["model"].predict_proba(allp.loc[ok, b["features"]].values.astype(float))[:, 1]
allp = allp.drop(columns=b["features"])
allp["in_gongliu"] = gpd.GeoSeries(gpd.points_from_xy(allp.lon, allp.lat), crs=4326).within(gl).values
allp.to_csv(KIT / "_candidates_mountain_apple_scored.csv", index=False, encoding="utf-8-sig")
print(allp.groupby("pool").agg(n=("lon", "size"), elev_med=("elev", "median"), hi=("b3_prob", lambda v: int((v >= 0.5).sum())), in_gl=("in_gongliu", "sum")))
