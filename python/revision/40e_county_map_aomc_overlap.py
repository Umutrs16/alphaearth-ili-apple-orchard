# -*- coding: utf-8 -*-
"""v4 retrain: overlap of the 30 m Gongliu county maps (old B3, expanded B3, expanded HIER) with AOMC 2021 apple
(weak reference only; AOMC reprojected nearest-neighbour onto the map grid). Raw pixel-count areas, unadjusted."""
import json
from pathlib import Path
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
V4 = Path("/workspace/apple_orchard_project/outputs/revision/v4_experiments"); R = V4 / "retrain_expanded"
maps = {"B3_old": V4 / "county_map/gongliu_B3poly2021v2_class_30m.tif", "B3_expanded": R / "county_map_B3/gongliu_B3expanded_class_30m.tif",
        "HIER_expanded": R / "county_map_HIER/gongliu_HIERexpanded_class_30m.tif"}
with rasterio.open(maps["B3_old"]) as s: T, crs, H, W = s.transform, s.crs, s.height, s.width
aomc = np.zeros((H, W), np.uint8)
with rasterio.open("/workspace/apple_orchard_project/data/aomc/AOMC_30_Xinjiang_2021.tif") as src:
    reproject(rasterio.band(src, 1), aomc, dst_transform=T, dst_crs=crs, resampling=Resampling.nearest)
lat = T.f + (np.arange(H) + 0.5) * T.e; Rr = 6371008.8
pa = ((np.deg2rad(T.a) * Rr * np.cos(np.deg2rad(lat))) * (np.deg2rad(-T.e) * Rr))[:, None]
res = {}
for k, p in maps.items():
    with rasterio.open(p) as s: c = s.read(1)
    v = c != 255; a = (c == 1); ao = (aomc == 1) & v
    res[k] = {"map_apple_ha": float((pa * a).sum() / 1e4), "aomc2021_apple_ha_in_county": float((pa * ao).sum() / 1e4),
              "overlap_ha": float((pa * (a & ao)).sum() / 1e4),
              "frac_map_apple_that_is_aomc": float((pa * (a & ao)).sum() / max((pa * a).sum(), 1)),
              "frac_aomc_apple_mapped": float((pa * (a & ao)).sum() / max((pa * ao).sum(), 1))}
res["note"] = "AOMC 2021 is a weak reference (not ground truth); areas are raw pixel counts, not bias-corrected."
json.dump(res, open(R / "county_map_aomc_overlap.json", "w"), indent=1); print(json.dumps(res, indent=1))
