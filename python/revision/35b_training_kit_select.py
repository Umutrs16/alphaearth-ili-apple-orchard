# -*- coding: utf-8 -*-
"""v4 task 5 (stage b): select stratified candidates and write the digitizing KML (IDs YL-TRN-0001...).
Rules: outside Gongliu (official GS2024 boundary), >=2 km from every Gongliu holdout polygon (v3, 113),
>=300 m from existing training polygons, >=800 m between selected points, not AOMC-2021 apple (3x3 window) for negative strata.
Hard negatives prioritised: ~70% of each stratum drawn from B3_poly_2021_v2 prob >= 0.5 where available."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, ee
from shapely.geometry import Point
from shapely.ops import unary_union
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision"))
import official_boundary as ob
KIT = ROOT / "outputs/revision/v4_experiments/training_expansion_kit"
d = pd.read_csv(KIT / "_candidates_scored.csv").dropna(subset=["b3_prob"]).reset_index(drop=True)
g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(32644)
P = gpd.read_file(ROOT / "outputs/revision/polygon_samples/v3/polygons_v3_all.geojson").to_crs(32644)
ho = unary_union(P[P.split == "val_holdout"].geometry.values); trp = unary_union(P[P.split == "train"].geometry.values)
gl = ob.gongliu().to_crs(32644).geometry.iloc[0]
g["d_holdout_km"] = g.distance(ho) / 1000; g["d_train_m"] = g.distance(trp); g["in_gongliu"] = g.within(gl)
# AOMC 2021 apple in 3x3 window
with rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2021.tif") as src:
    xy = gpd.GeoSeries(gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(src.crs)
    arr_flags = []
    for p in xy:
        r, c = src.index(p.x, p.y)
        w = src.read(1, window=((r - 1, r + 2), (c - 1, c + 2)))
        arr_flags.append(int((w == 1).any()))
g["aomc_apple_3x3"] = arr_flags
# tree-cover fraction within 30 m (WorldCover 10 m) for tree candidates -> shelterbelt-like (linear) vs block
ee.Initialize(project="hallowed-chain-458005-g9")
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
frac = wc.eq(10).reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(30, "meters")).rename("tree_frac")
t = g[g.wc == 10]
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"i": int(i)}) for i, r in t.iterrows()])
tf = {}
for s in range(0, len(t), 300):
    sub = ee.FeatureCollection(fc.toList(300, s))
    for f in frac.sampleRegions(collection=sub, scale=10, geometries=False).getInfo()["features"]:
        tf[f["properties"]["i"]] = f["properties"]["tree_frac"]
g["tree_frac30m"] = pd.Series(tf)
elig = g[(~g.in_gongliu) & (g.d_holdout_km >= 2) & (g.d_train_m >= 300)].copy()
STRATA = [("OTH", "其他果园候选（地块状林木，非AOMC苹果）", lambda x: (x.wc == 10) & (x.aomc_apple_3x3 == 0) & (x.tree_frac30m >= 0.7), 45),
          ("CRP", "农田候选", lambda x: (x.wc == 40) & (x.aomc_apple_3x3 == 0), 45),
          ("FOR", "防护林/人工林候选（线状或稀疏林木）", lambda x: (x.wc == 10) & (x.aomc_apple_3x3 == 0) & (x.tree_frac30m < 0.7), 28),
          ("GRS", "草地/灌丛候选", lambda x: x.wc.isin([20, 30]) & (x.aomc_apple_3x3 == 0), 28),
          ("HNG", "建设用地/裸地等其他硬负样本候选", lambda x: x.wc.isin([50, 60, 70, 100]) & (x.aomc_apple_3x3 == 0), 28)]
rng = np.random.default_rng(42); chosen = []; used = gpd.GeoSeries([], crs=32644)
def far_enough(pt):
    return len(used) == 0 or used.distance(pt).min() >= 800
for code, zh, f, n in STRATA:
    pool = elig[f(elig)].copy(); pool["hi"] = pool.b3_prob >= 0.5
    n_hi = int(round(0.7 * n)); picked = []
    for part, k in [(pool[pool.hi].sort_values("b3_prob", ascending=False), n_hi), (pool[~pool.hi].sample(frac=1, random_state=42), n)]:
        # round-robin across counties for spatial spread
        groups = {cn: q for cn, q in part.groupby("county")}
        order = []
        while any(len(q) for q in groups.values()):
            for cn in list(groups):
                if len(groups[cn]): order.append(groups[cn].index[0]); groups[cn] = groups[cn].iloc[1:]
        for i in order:
            if len(picked) >= (k if part is not None and k == n_hi and len(picked) < n_hi else n): pass
            if len([p for p in picked]) >= n: break
            if part is pool[pool.hi] and False: pass
            pt = elig.geometry[i]
            if far_enough(pt):
                picked.append(i); used = pd.concat([used, gpd.GeoSeries([pt], crs=32644)], ignore_index=True)
                if k == n_hi and sum(elig.loc[picked, "b3_prob"] >= 0.5) >= n_hi: break
    sel = elig.loc[picked].copy(); sel["stratum"] = code; sel["stratum_zh"] = zh; chosen.append(sel)
    print(code, len(pool), "hi-prob pool", int(pool.hi.sum()), "->", len(sel), "hi", int((sel.b3_prob >= 0.5).sum()))
S = pd.concat(chosen).reset_index(drop=True)
S["id"] = [f"YL-TRN-{i + 1:04d}" for i in range(len(S))]
WCN = {10: "树木", 20: "灌丛", 30: "草地", 40: "农田", 50: "建设用地", 60: "裸地/稀疏植被", 70: "冰雪", 100: "苔藓地衣"}
S["wc_name"] = S.wc.map(WCN)
cols = ["id", "stratum", "stratum_zh", "county", "lon", "lat", "wc", "wc_name", "b3_prob", "tree_frac30m", "aomc_apple_3x3", "d_holdout_km", "d_train_m"]
S[cols].round({"b3_prob": 3, "tree_frac30m": 2, "d_holdout_km": 2, "d_train_m": 0}).to_csv(KIT / "YL_TRN_candidates.csv", index=False, encoding="utf-8-sig")
# KML
colors = {"OTH": "ff1c9fff", "CRP": "ff00d7ff", "FOR": "ff2ca02c", "GRS": "ff8fd6b0", "HNG": "ff7f7f7f"}
kml = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>YL-TRN 训练样本扩充候选点（巩留县外）</name>']
for code in colors:
    kml.append(f'<Style id="{code}"><IconStyle><color>{colors[code]}</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>')
for code, zh, *_ in STRATA:
    kml.append(f"<Folder><name>{code} {zh}</name>")
    for r in S[S.stratum == code].itertuples():
        desc = (f"stratum={r.stratum}; 县={r.county}; WorldCover2021={r.wc_name}; B3概率={r.b3_prob:.2f}; "
                f"距巩留留出面={r.d_holdout_km:.1f} km; class_code=待判（1苹果园/2其他果园/0非果园）; crop_type=; confidence=")
        kml.append(f"<Placemark><name>{r.id}</name><description>{desc}</description><styleUrl>#{code}</styleUrl>"
                   f"<Point><coordinates>{r.lon:.6f},{r.lat:.6f},0</coordinates></Point></Placemark>")
    kml.append("</Folder>")
kml.append("</Document></kml>")
(KIT / "YL_TRN_training_expansion_candidates.kml").write_text("\n".join(kml), encoding="utf-8")
summ = {"n": int(len(S)), "by_stratum": S.stratum.value_counts().to_dict(), "by_county": S.county.value_counts().to_dict(),
        "hi_prob_by_stratum": S[S.b3_prob >= 0.5].stratum.value_counts().to_dict(),
        "min_d_holdout_km": float(S.d_holdout_km.min()), "any_in_gongliu": bool(S.in_gongliu.any()), "min_d_train_m": float(S.d_train_m.min())}
json.dump(summ, open(KIT / "kit_summary.json", "w"), ensure_ascii=False, indent=1); print(summ)
