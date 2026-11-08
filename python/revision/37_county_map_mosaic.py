# -*- coding: utf-8 -*-
"""v4 task 4: mosaic per-tile B3 probability tiles -> county GeoTIFFs (prob uint8 0-250 = p*250, 255 nodata; class uint8 0/1, 255 nodata),
masked by the official GS(2024)0650 Gongliu boundary; raw (unadjusted) mapped apple area."""
import sys, os, json, math
from pathlib import Path
import numpy as np, rasterio
from rasterio.merge import merge
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import mapping
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision/figures_v3_scripts"))
import official_boundary as ob
SCALE = int(sys.argv[1]); O = Path(os.environ.get("CM_OUT", str(ROOT / "outputs/revision/v4_experiments/county_map"))); TAG = os.environ.get("CM_TAG", "B3poly2021v2"); TD = O / f"tiles_{SCALE}m"; TMP = O / f"_u8_{SCALE}m"; TMP.mkdir(exist_ok=True)
gl = ob.gongliu().geometry.iloc[0]
tiles = sorted(TD.glob("prob_*.tif")); print("tiles", len(tiles))
for t in tiles:
    for kind in ("p", "c"):
        o = TMP / f"{kind}_{t.stem}.tif"
        if o.exists(): continue
        with rasterio.open(t) as s:
            a = s.read(1); prof = s.profile
        v = np.isfinite(a)
        u = np.full(a.shape, 255, np.uint8)
        if kind == "p": u[v] = np.round(a[v] * 250).astype(np.uint8)
        else: u[v] = (a[v] >= 0.5).astype(np.uint8)
        prof.update(dtype="uint8", nodata=255, compress="deflate")
        with rasterio.open(o, "w", **prof) as d: d.write(u, 1)
out = {}
for kind, name in (("p", f"gongliu_{TAG}_prob_{SCALE}m.tif"), ("c", f"gongliu_{TAG}_class_{SCALE}m.tif")):
    dst = O / name
    merge(sorted(TMP.glob(f"{kind}_*.tif")), nodata=255, dst_path=str(dst), method="first")
    out[kind] = dst
# mask outside county + area, blockwise
with rasterio.open(out["c"], "r+") as dc, rasterio.open(out["p"], "r+") as dp:
    H, W, T = dc.height, dc.width, dc.transform; res_x, res_y = T.a, -T.e
    R = 6371008.8; n_valid = n_apple = 0; area_apple = area_valid = 0.0; n_nodata_in = 0
    step = 512
    for r0 in range(0, H, step):
        h = min(step, H - r0); win = Window(0, r0, W, h); wt = dc.window_transform(win)
        inside = ~geometry_mask([mapping(gl)], out_shape=(h, W), transform=wt, all_touched=False)
        c = dc.read(1, window=win); p = dp.read(1, window=win)
        c[~inside] = 255; p[~inside] = 255; dc.write(c, 1, window=win); dp.write(p, 1, window=win)
        lat = T.f + (np.arange(r0, r0 + h) + 0.5) * T.e
        pa = (np.deg2rad(res_x) * R * np.cos(np.deg2rad(lat))) * (np.deg2rad(res_y) * R)  # m2 per pixel per row
        valid = inside & (c != 255); ap = c == 1
        n_nodata_in += int((inside & (c == 255)).sum())
        area_valid += float((valid.sum(1) * pa).sum()); area_apple += float((ap.sum(1) * pa).sum())
        n_valid += int(valid.sum()); n_apple += int(ap.sum())
meta = {"scale_m": SCALE, "n_tiles": len(tiles), "shape": [H, W], "pixel_deg": [res_x, res_y], "n_valid_px": n_valid, "n_apple_px": n_apple,
        "n_nodata_px_inside_county": n_nodata_in,
        "county_area_valid_km2": area_valid / 1e6, "official_county_area_km2_ref": 4113.46,
        "apple_area_unadjusted_ha": area_apple / 1e4, "apple_fraction_of_mapped_county": area_apple / area_valid,
        "threshold": 0.5, "model": os.environ.get("CM_MODEL", "B3_poly_2021_v2 (frozen)"), "embedding": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2021, getDownloadURL float32, EPSG:4326",
        "note": "Raw pixel-count area; NOT a bias-corrected (stratified, error-adjusted) area estimate."}
json.dump(meta, open(O / f"county_map_meta_{SCALE}m.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(meta, ensure_ascii=False, indent=1))
