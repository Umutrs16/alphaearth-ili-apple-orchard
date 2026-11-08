# -*- coding: utf-8 -*-
"""A (supplement): AOMC-2021 vs digitized apple polygons, wider shift search ±10 px (±300 m) on the native AOMC grid."""
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, json
from rasterio.warp import transform_bounds
from rasterio.features import rasterize
from rasterio.windows import from_bounds
ROOT = Path("/workspace/apple_orchard_project"); E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; OUT = ROOT / "outputs/revision/v41_review/aomc_alignment"
P = pd.concat([gpd.read_file(ROOT / "outputs/revision/polygon_samples/v3/polygons_v3_all.geojson")[["name", "class_code", "geometry"]],
               gpd.read_file(E / "training_new_kept.geojson").rename(columns={"poly_name": "name"})[["name", "class_code", "geometry"]]], ignore_index=True)
with rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2021.tif") as s:
    b = transform_bounds("EPSG:4326", s.crs, *(P.total_bounds + np.array([-0.05, -0.05, 0.05, 0.05])))
    w = from_bounds(*b, s.transform).round_offsets().round_lengths(); A = s.read(1, window=w) == 1; T = s.window_transform(w); C = s.crs
cls = rasterize([(g, int(c) + 1) for g, c in zip(P.to_crs(C).geometry, P.class_code)], out_shape=A.shape, transform=T, fill=0, dtype="uint8")
yy, xx = {}, {}
for k in (1, 2, 3): yy[k], xx[k] = np.nonzero(cls == k)
R = []
for dy in range(-10, 11):
    for dx in range(-10, 11):
        r = {"dy": dy, "dx": dx}
        for k, nm in ((2, "apple"), (3, "other_orchard"), (1, "nonorchard")):
            y, x = yy[k] - dy, xx[k] - dx; ok = (y >= 0) & (y < A.shape[0]) & (x >= 0) & (x < A.shape[1]); r[nm] = float(A[y[ok], x[ok]].mean())
        R.append(r)
R = pd.DataFrame(R); R.to_csv(OUT / "shift_AOMC_vs_digitized_polygons_pm10.csv", index=False)
b = R.loc[R.apple.idxmax()]; z = R[(R.dy == 0) & (R.dx == 0)].iloc[0]
out = {"at_0": z.to_dict(), "best_apple": b.to_dict(), "apple_frac_range": [float(R.apple.min()), float(R.apple.max())]}
json.dump(out, open(OUT / "shift_polygons_pm10_summary.json", "w"), indent=1); print(out)
