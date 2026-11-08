# -*- coding: utf-8 -*-
"""v4 task 5 (stage a): candidate points for a training-expansion digitizing kit OUTSIDE Gongliu.
ESA WorldCover v200 (2021, 10 m) strata sampled in GEE (valley floor, elevation < 1500 m) in counties around Gongliu;
AOMC 2021 apple flag checked locally; AlphaEarth 2021 sampled at 10 m and scored with the frozen B3_poly_2021_v2."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, ee, joblib
from shapely.geometry import mapping
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision"))
import official_boundary as ob
KIT = ROOT / "outputs/revision/v4_experiments/training_expansion_kit"
COUNTIES = ["特克斯县", "伊宁县", "新源县", "察布查尔锡伯自治县", "尼勒克县", "霍城县"]
c = ob.county(); reg = c[c.name.isin(COUNTIES)]
ee.Initialize(project="hallowed-chain-458005-g9")
cand = []
for _, r in reg.iterrows():
    geom = ee.Geometry(mapping(r.geometry.simplify(0.002)))
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    dem = ee.Image("USGS/SRTMGL1_003")
    strat = wc.updateMask(dem.lt(1500)).rename("wc")
    for att in range(4):
        try:
            fc = strat.stratifiedSample(numPoints=90, classBand="wc", region=geom, scale=10, seed=42, geometries=True,
                                        classValues=[10, 20, 30, 40, 50, 60, 80, 90], classPoints=[160, 40, 60, 90, 50, 30, 0, 0], tileScale=4)
            info = fc.getInfo()["features"]; break
        except Exception as e:
            print("retry", r["name"], att, str(e)[:150]); time.sleep(10)
    for f in info:
        x, y = f["geometry"]["coordinates"]; cand.append({"lon": x, "lat": y, "wc": f["properties"]["wc"], "county": r["name"]})
    print(r["name"], len(info), flush=True)
df = pd.DataFrame(cand); df.to_csv(KIT / "_candidates_raw.csv", index=False)
# AE 2021 at 10 m (same as polygon samples) -> B3 prob
ae = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2021-01-01", "2022-01-01").mosaic()
rows = []
for s in range(0, len(df), 200):
    sub = df.iloc[s:s + 200]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"i": int(i)}) for i, r in sub.iterrows()])
    for att in range(4):
        try:
            rows += [f["properties"] for f in ae.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"]]; break
        except Exception as e:
            print("ae retry", att, str(e)[:120]); time.sleep(10)
a = pd.DataFrame(rows).set_index("i")
a = a.rename(columns={f"A{i:02d}": f"AE_{i:02d}" for i in range(64)})
b = joblib.load(ROOT / "outputs/revision/polygon_samples/models/B3_poly_2021_v2.joblib")
df = df.join(a[b["features"]], how="left")
ok = df[b["features"]].notna().all(1)
df.loc[ok, "b3_prob"] = b["model"].predict_proba(df.loc[ok, b["features"]].values.astype(float))[:, 1]
df.drop(columns=b["features"]).to_csv(KIT / "_candidates_scored.csv", index=False)
print("scored", ok.sum(), len(df))
