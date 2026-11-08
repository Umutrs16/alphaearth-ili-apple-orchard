# -*- coding: utf-8 -*-
"""v4 task 4: fetch SRTM elevation and ESA WorldCover 2021 on the exact 30 m county-map grid (for interpreting the map only)."""
import json, urllib.request
from pathlib import Path
import numpy as np, rasterio, ee
O = Path("/workspace/apple_orchard_project/outputs/revision/v4_experiments/county_map")
ee.Initialize(project="hallowed-chain-458005-g9")
with rasterio.open(O / "gongliu_B3poly2021v2_class_30m.tif") as s: T, H, W, prof = s.transform, s.height, s.width, s.profile
imgs = {"elev": ee.Image("USGS/SRTMGL1_003").toInt16(), "wc": ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").toUint8()}
for name, img in imgs.items():
    parts = []
    for c0 in range(0, W, 1900):
        w = min(1900, W - c0); t = [T.a, 0, T.c + c0 * T.a, 0, T.e, T.f]
        url = img.getDownloadURL({"crs": "EPSG:4326", "crs_transform": t, "dimensions": f"{w}x{H}", "format": "GEO_TIFF"})
        fp = O / f"_{name}_{c0}.tif"; urllib.request.urlretrieve(url, fp)
        with rasterio.open(fp) as s: parts.append(s.read(1)); 
        fp.unlink(); print(name, c0, parts[-1].shape, flush=True)
    a = np.concatenate(parts, axis=1)
    p = prof.copy(); p.update(dtype=a.dtype.name, nodata=None, compress="deflate")
    with rasterio.open(O / f"ref_{name}_30m.tif", "w", **p) as d: d.write(a, 1)
