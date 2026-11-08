# -*- coding: utf-8 -*-
"""v4 task 1: sample 2021 S2/S1/DEM features (identical definitions to the AOMC B0-B6 ablation,
python/gee_sample_features.py build_s2/build_s1/build_dem, scale=30 m) at the SAME 910 polygon sample points
(389 train + 521 Gongliu holdout) used by B3_poly_2021_v2. Cached, chunked getInfo."""
import sys, time, json
from pathlib import Path
import pandas as pd, ee
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python"))
import gee_sample_features as g
OUT = ROOT / "outputs/revision/v4_experiments"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "poly_s1s2dem_features_2021_v4.csv"
pts = pd.read_csv(ROOT / "outputs/revision/polygon_samples/sample_points_v3.csv")
ee.Initialize(project=g.PROJECT)
aoi = ee.Geometry.Rectangle([pts.lon.min() - 0.05, pts.lat.min() - 0.05, pts.lon.max() + 0.05, pts.lat.max() + 0.05])
img = g.build_s2(2021, aoi).addBands(g.build_s1(2021, aoi)).addBands(g.build_dem(aoi))
done = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["sample_id"])
todo = pts[~pts.sample_id.isin(done.sample_id)]
rows = done.to_dict("records"); log = {"errors": []}; t0 = time.time()
for s in range(0, len(todo), 80):
    sub = todo.iloc[s:s + 80]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"sample_id": r.sample_id}) for r in sub.itertuples()])
    for att in range(4):
        try:
            info = img.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()["features"]
            rows += [f["properties"] for f in info]; break
        except Exception as e:
            log["errors"].append(str(e)[:200]); print("retry", att, e); time.sleep(10 * (att + 1))
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"{s + len(sub)}/{len(todo)} rows={len(rows)} {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows); print("final", len(df), "missing", len(set(pts.sample_id) - set(df.sample_id)))
log.update({"n": len(df), "n_points": len(pts), "scale_m": 30, "year": 2021, "seconds": time.time() - t0})
json.dump(log, open(OUT / "poly_s1s2dem_fetch_log.json", "w"), indent=1)
