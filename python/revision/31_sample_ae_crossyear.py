# -*- coding: utf-8 -*-
"""v4 task 3: sample AlphaEarth annual embeddings for years 2017-2025 at the FIXED AOMC weak-label point sets of
2019/2020/2021 (data/samples/real/points_gongliu_2019_2021.csv, 3112 points), using the same build_ae() and
sampleRegions(scale=30) as the original feature extraction (python/gee_sample_features.py)."""
import sys, time, json
from pathlib import Path
import pandas as pd, ee
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python"))
import gee_sample_features as g
OUT = ROOT / "outputs/revision/v4_experiments/crossyear"; OUT.mkdir(parents=True, exist_ok=True)
pts = pd.read_csv(ROOT / "data/samples/real/points_gongliu_2019_2021.csv").reset_index().rename(columns={"index": "pid"})
pts.to_csv(OUT / "label_points_2019_2021_with_pid.csv", index=False)
ee.Initialize(project=g.PROJECT)
aoi = ee.Geometry.Rectangle([pts.lon.min() - 0.02, pts.lat.min() - 0.02, pts.lon.max() + 0.02, pts.lat.max() + 0.02])
years = [int(y) for y in (sys.argv[1].split(",") if len(sys.argv) > 1 else range(2017, 2026))]
for fy in years:
    cache = OUT / f"ae_{fy}_at_labelpts.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["pid"])
    todo = pts[~pts.pid.isin(done.pid)]
    rows = done.to_dict("records"); img = g.build_ae(fy, aoi); t0 = time.time()
    for s in range(0, len(todo), 300):
        sub = todo.iloc[s:s + 300]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"pid": int(r.pid)}) for r in sub.itertuples()])
        for att in range(5):
            try:
                info = img.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()["features"]
                rows += [f["properties"] for f in info]; break
            except Exception as e:
                print("retry", fy, att, str(e)[:150], flush=True); time.sleep(10 * (att + 1))
        pd.DataFrame(rows).to_csv(cache, index=False)
    print(fy, "rows", len(rows), f"{time.time() - t0:.0f}s", flush=True)
