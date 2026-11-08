# -*- coding: utf-8 -*-
"""Real AOMC purification over Gongliu: morphological positives, hard negatives ≥60 m,
spatial block IDs, years 2019–2021.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, load_config, setup_logging, spatial_block_id

log = setup_logging("purify_aomc_real")


def load_gongliu_mask_wgs84():
    import geopandas as gpd
    from shapely.ops import unary_union

    gpath = ROOT / "data" / "roi" / "Gongliu.geojson"
    if not gpath.exists():
        raise FileNotFoundError(f"Missing {gpath}; run export_roi.py first")
    gdf = gpd.read_file(gpath)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    return unary_union(gdf.geometry.values)


def make_positive_core(orchard: np.ndarray, min_core: int = 200) -> np.ndarray:
    """Prefer eroded/opened cores; gracefully relax if orchards are fragmented."""
    st = generate_binary_structure(2, 1)
    opened = binary_opening(orchard, structure=np.ones((3, 3)))
    eroded = binary_erosion(opened, structure=st, iterations=1)
    if int(eroded.sum()) >= min_core:
        log.info("  positives: eroded core n=%d", int(eroded.sum()))
        return eroded
    if int(opened.sum()) >= min_core:
        log.info("  positives: opened (no erode) n=%d", int(opened.sum()))
        return opened
    # majority filter style: require >=5 orchard neighbors in 3x3 including self
    from scipy.ndimage import uniform_filter

    frac = uniform_filter(orchard.astype(np.float32), size=3, mode="constant")
    majority = orchard & (frac >= (5.0 / 9.0))
    if int(majority.sum()) >= min_core:
        log.info("  positives: majority5 n=%d", int(majority.sum()))
        return majority
    log.info("  positives: raw orchard n=%d (fragmented)", int(orchard.sum()))
    return orchard


def purify_year(
    tif: Path,
    geom_wgs84,
    n_pos: int,
    n_neg: int,
    seed: int,
    block_size_deg: float,
    buffer_px: int = 2,
) -> pd.DataFrame:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_bounds, transform as rio_transform
    from shapely.geometry import mapping
    import geopandas as gpd

    rng = np.random.default_rng(seed)
    year = int(tif.stem.split("_")[-1])

    with rasterio.open(tif) as ds:
        left, bottom, right, top = transform_bounds(
            CRS.from_epsg(4326), ds.crs, *geom_wgs84.bounds
        )
        pad = 500
        from rasterio.windows import from_bounds

        window = from_bounds(
            left - pad, bottom - pad, right + pad, top + pad, transform=ds.transform
        )
        data = ds.read(1, window=window, boundless=True, fill_value=0)
        transform = ds.window_transform(window)
        crs = ds.crs
        gdf = gpd.GeoDataFrame(geometry=[geom_wgs84], crs=4326).to_crs(crs)
        county_mask = ~geometry_mask(
            [mapping(gdf.geometry.iloc[0])],
            out_shape=data.shape,
            transform=transform,
            invert=False,
        )

    orchard = (data > 0) & county_mask
    core = make_positive_core(orchard, min_core=min(200, max(50, n_pos // 2)))
    near = binary_dilation(orchard, structure=np.ones((3, 3)), iterations=buffer_px)
    hard_neg = county_mask & (~near)

    ys_p, xs_p = np.where(core)
    ys_n, xs_n = np.where(hard_neg)
    log.info(
        "%d: orchard=%d core=%d hard_neg=%d",
        year,
        int(orchard.sum()),
        int(core.sum()),
        int(hard_neg.sum()),
    )
    if len(ys_p) < 30:
        raise RuntimeError(f"Too few positives in {tif.name}: {len(ys_p)}")

    n_pos_u = min(n_pos, len(ys_p))
    n_neg_u = min(n_neg, len(ys_n))
    sel_p = rng.choice(len(ys_p), n_pos_u, replace=False)
    sel_n = rng.choice(len(ys_n), n_neg_u, replace=False)

    def pix_to_ll(rr, cc):
        xs_c, ys_c = rasterio.transform.xy(transform, rr, cc)
        lons, lats = rio_transform(crs, CRS.from_epsg(4326), xs_c, ys_c)
        return np.asarray(lons), np.asarray(lats)

    frames = []
    for label, sel, ys, xs in (
        (1, sel_p, ys_p, xs_p),
        (0, sel_n, ys_n, xs_n),
    ):
        lon, lat = pix_to_ll(ys[sel], xs[sel])
        df = pd.DataFrame(
            {
                "lon": lon,
                "lat": lat,
                "year": year,
                "label": label,
                "label_raw": label,
                "block_id": spatial_block_id(lon, lat, block_size_deg),
                "source": f"aomc_core_{tif.name}",
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=600)
    ap.add_argument("--n-neg", type=int, default=600)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    geom = load_gongliu_mask_wgs84()
    aomc_dir = ROOT / cfg["data"]["aomc_dir"]
    tifs = sorted(aomc_dir.glob("AOMC_30_Xinjiang_*.tif"))
    if not tifs:
        raise FileNotFoundError("No AOMC tifs")

    block = cfg["training"]["spatial_block_size_deg"]
    frames = []
    for i, tif in enumerate(tifs):
        df = purify_year(
            tif,
            geom,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            seed=cfg["training"]["random_seed"] + i,
            block_size_deg=block,
            buffer_px=2,
        )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out_dir = ROOT / "data" / "samples" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    pts_path = out_dir / "points_gongliu_2019_2021.csv"
    out.to_csv(pts_path, index=False, encoding="utf-8-sig")
    meta = {
        "aoi": "巩留县",
        "n_rows": len(out),
        "by_year_label": {
            f"{y}_{l}": int(c)
            for (y, l), c in out.groupby(["year", "label"]).size().items()
        },
        "pos": int((out["label"] == 1).sum()),
        "neg": int((out["label"] == 0).sum()),
        "hard_neg_buffer_m": 60,
        "block_size_deg": block,
        "points_csv": str(pts_path),
    }
    with open(out_dir / "points_gongliu_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s (%d rows)", pts_path, len(out))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return pts_path


if __name__ == "__main__":
    main()
