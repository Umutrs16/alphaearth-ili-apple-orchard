# -*- coding: utf-8 -*-
"""v4 task 4: wall-to-wall Gongliu county B3_poly_2021_v2 map. AlphaEarth 2021 64-band tiles are downloaded via
getDownloadURL (EPSG:4326, given scale), the frozen LightGBM model is applied locally (same as maps_v2 focus map),
per-tile probability GeoTIFFs are mosaicked and masked with the official GS(2024)0650 Gongliu boundary."""
import sys, os, time, json, urllib.request
from pathlib import Path
import numpy as np, joblib, rasterio, ee
from shapely.geometry import box
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision/figures_v3_scripts"))
import official_boundary as ob
SCALE = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TILE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
MAXT = int(sys.argv[3]) if len(sys.argv) > 3 else 10**6
WORKER, NW = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
OUT = Path(os.environ.get("CM_OUT", str(ROOT / "outputs/revision/v4_experiments/county_map"))); TD = OUT / f"tiles_{SCALE}m"; TD.mkdir(parents=True, exist_ok=True)  # CM_OUT/CM_MODEL env overrides used by 39_retrain_expanded.py
b = joblib.load(os.environ.get("CM_MODEL", str(ROOT / "outputs/revision/polygon_samples/models/B3_poly_2021_v2.joblib"))); clf, feats = b["model"], b["features"]
gl = ob.gongliu().geometry.iloc[0]
ee.Initialize(project="hallowed-chain-458005-g9")
ae = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2021-01-01", "2021-12-31").mosaic().toFloat()  # as maps_v2; float32 to fit 48 MB request limit
minx, miny, maxx, maxy = gl.bounds
xs = np.arange(np.floor(minx / TILE) * TILE, maxx, TILE); ys = np.arange(np.floor(miny / TILE) * TILE, maxy, TILE)
tiles = [(round(x, 4), round(y, 4), round(x + TILE, 4), round(y + TILE, 4)) for y in ys for x in xs if box(x, y, x + TILE, y + TILE).intersects(gl)]
print("tiles", len(tiles), "worker", WORKER, "of", NW, flush=True)
log = {"scale_m": SCALE, "tile_deg": TILE, "n_tiles": len(tiles), "tiles": []}
t_all = time.time()
for i, (w, s, e, n) in enumerate(tiles[:MAXT]):
    if i % NW != WORKER: continue
    pp = TD / f"prob_{w:.3f}_{s:.3f}.tif"
    if pp.exists(): continue
    ap = TD / f"_ae_{w:.3f}_{s:.3f}.tif"; t0 = time.time(); err = None
    for att in range(4):
        try:
            url = ae.getDownloadURL({"scale": SCALE, "crs": "EPSG:4326", "format": "GEO_TIFF",
                                     "region": ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)})
            urllib.request.urlretrieve(url, ap); break
        except Exception as ex:
            err = str(ex)[:200]; print("retry", i, att, err, flush=True); time.sleep(5 * (att + 1))
    else:
        log["tiles"].append({"tile": [w, s, e, n], "error": err}); continue
    t_dl = time.time() - t0
    with rasterio.open(ap) as src:
        d = src.read(); prof = src.profile
    flat = d[:64].reshape(64, -1).T.astype(np.float32); del d
    valid = np.isfinite(flat).all(1) & (np.abs(flat).sum(1) > 0)
    prob = np.full(flat.shape[0], np.nan, np.float32)
    if valid.any(): prob[valid] = clf.predict_proba(flat[valid])[:, 1]
    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(pp, "w", **prof) as dst: dst.write(prob.reshape(prof["height"], prof["width"]), 1)
    sz = ap.stat().st_size; ap.unlink()
    log["tiles"].append({"tile": [w, s, e, n], "dl_s": round(t_dl, 1), "total_s": round(time.time() - t0, 1), "ae_bytes": sz, "shape": [prof["height"], prof["width"]]})
    print(f"{i + 1}/{len(tiles)} dl {t_dl:.1f}s total {time.time() - t0:.1f}s {sz / 1e6:.1f} MB", flush=True)
    json.dump(log, open(OUT / f"tile_log_{SCALE}m_w{WORKER}.json", "w"), indent=1)
log["elapsed_s"] = time.time() - t_all
json.dump(log, open(OUT / f"tile_log_{SCALE}m_w{WORKER}.json", "w"), indent=1)
