# -*- coding: utf-8 -*-
"""Publication-quality Gongliu (巩留) apple-orchard map with polygon-trained B3 v2.

Focus AOI (~82.35–82.95°E, 43.20–43.55°N ∩ 巩留县) at ~100 m via tiled
getDownloadURL; four ~2.5 km zoom windows at 10 m (fallback 30 m).
GEE project hallowed-chain-458005-g9 is in restricted mode — no full-county
10 m Export.toDrive.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ee
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds, Affine
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping, shape, Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "revision"))
from common import setup_matplotlib_zh, save_json  # noqa: E402

PROJECT = "hallowed-chain-458005-g9"
YEAR = 2021
MODEL_PATH = (
    ROOT
    / "outputs"
    / "revision"
    / "polygon_samples"
    / "models"
    / "B3_poly_2021_v2.joblib"
)
ROI_PATH = ROOT / "data" / "roi" / "Yili_county_full.geojson"
KML_PATH = (
    ROOT
    / "outputs"
    / "revision"
    / "polygon_samples"
    / "Confirmed_apple_field_points_area.kml"
)
SPLIT_PATH = (
    ROOT
    / "outputs"
    / "revision"
    / "polygon_samples"
    / "split_manifest_v2.csv"
)
MAPS_V2 = ROOT / "outputs" / "revision" / "maps_v2"
FIGS_V2 = ROOT / "outputs" / "revision" / "figures_v2"
TILES_DIR = MAPS_V2 / "tiles"
ZOOMS_DIR = MAPS_V2 / "zooms"

# Focus AOI covering both holdout clusters; clipped to county
FOCUS_BBOX = (82.35, 43.20, 82.95, 43.55)  # lon_min, lat_min, lon_max, lat_max
GRID_SCALE_M = 100  # ~90–100 m
TILE_DEG = 0.12
TILE_OVERLAP = 0.005
ZOOM_HALF_DEG = 0.012  # ~2.7 km box
THRESH = 0.5
SLEEP = 0.6


def init_ee() -> None:
    ee.Initialize(project=PROJECT)


def load_gongliu_geom():
    import geopandas as gpd

    gdf = gpd.read_file(ROI_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gl = gdf[gdf["name"] == "巩留县"]
    if gl.empty:
        raise RuntimeError("巩留县 not found in ROI geojson")
    return unary_union(gl.geometry.values)


def focus_aoi(county_geom):
    fb = box(*FOCUS_BBOX)
    aoi = county_geom.intersection(fb)
    if aoi.is_empty:
        raise RuntimeError("Focus AOI ∩ Gongliu is empty")
    return aoi


def load_holdout_polygons() -> Tuple[Any, Any, pd.DataFrame]:
    """Return (apple_gdf, nonapple_gdf, holdout_manifest)."""
    import geopandas as gpd
    from pyogrio import list_layers

    manifest = pd.read_csv(SPLIT_PATH)
    hold = manifest[manifest["split"] == "val_holdout"].copy()
    hold_names = set(hold["name"].astype(str).tolist())

    frames = []
    for lyr, _ in list_layers(str(KML_PATH)):
        g = gpd.read_file(KML_PATH, layer=lyr)
        g["layer"] = lyr
        frames.append(g)
    allg = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    polys = allg[allg.geom_type.astype(str).str.contains("Polygon", na=False)].copy()
    polys["Name"] = polys["Name"].astype(str)
    # Prefer holdout names; also keep all Gongliu layer polys for context
    hold_polys = polys[polys["Name"].isin(hold_names)].copy()
    # Join class from manifest
    hold_polys = hold_polys.merge(
        hold[["name", "class_code", "cen_lon", "cen_lat"]],
        left_on="Name",
        right_on="name",
        how="left",
    )
    apple = hold_polys[hold_polys["class_code"] == 1]
    nonapple = hold_polys[hold_polys["class_code"] == 0]
    return apple, nonapple, hold


def ae_image(year: int = YEAR):
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )


def download_ae_tile(
    ae: ee.Image,
    west: float,
    south: float,
    east: float,
    north: float,
    scale: int,
    out_path: Path,
    retries: int = 3,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    region = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
    last_err = None
    for attempt in range(retries):
        try:
            url = ae.getDownloadURL(
                {
                    "scale": scale,
                    "crs": "EPSG:4326",
                    "region": region,
                    "format": "GEO_TIFF",
                }
            )
            urllib.request.urlretrieve(url, out_path)
            if out_path.stat().st_size > 1000:
                return out_path
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"getDownloadURL failed for {out_path.name}: {last_err}")


def tile_bounds(aoi, tile_deg=TILE_DEG, overlap=TILE_OVERLAP) -> List[Tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = aoi.bounds
    xs = []
    x = minx
    while x < maxx - 1e-9:
        xs.append(x)
        x += tile_deg - overlap
    ys = []
    y = miny
    while y < maxy - 1e-9:
        ys.append(y)
        y += tile_deg - overlap
    tiles = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_deg, maxx)
            y1 = min(y0 + tile_deg, maxy)
            tb = box(x0, y0, x1, y1)
            if not tb.intersects(aoi):
                continue
            # slightly pad to avoid gaps
            tiles.append((x0, y0, x1, y1))
    return tiles


def predict_raster(
    tif_path: Path,
    clf,
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return prob (H,W), cls (H,W), profile."""
    with rasterio.open(tif_path) as src:
        data = src.read()  # (64, H, W)
        profile = src.profile.copy()
        transform = src.transform
        height, width = src.height, src.width
    if data.shape[0] < 64:
        raise RuntimeError(f"Expected 64 AE bands, got {data.shape[0]}")
    # bands A00..A63 in order
    H, W = height, width
    flat = data[:64].reshape(64, -1).T.astype(np.float32)  # (N, 64)
    valid = np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 0)
    prob_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
    if valid.any():
        X = flat[valid]
        # LightGBM predict_proba
        proba = clf.predict_proba(X)[:, 1].astype(np.float32)
        prob_flat[valid] = proba
    prob = prob_flat.reshape(H, W)
    cls = np.full((H, W), -1, dtype=np.int16)
    cls[np.isfinite(prob)] = (prob[np.isfinite(prob)] >= THRESH).astype(np.int16)
    return prob, cls, {**profile, "transform": transform, "height": H, "width": W}


def write_prob_class_geotiff(
    out_path: Path,
    prob: np.ndarray,
    cls: np.ndarray,
    ref_profile: dict,
    mask_geom=None,
) -> None:
    transform = ref_profile["transform"]
    height, width = prob.shape
    if mask_geom is not None:
        m = geometry_mask(
            [mapping(mask_geom)],
            transform=transform,
            invert=True,
            out_shape=(height, width),
        )
        prob = prob.copy()
        cls = cls.copy()
        prob[~m] = np.nan
        cls[~m] = -1
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "nodata": np.nan,
    }
    # store class as float in band2 for simplicity (nodata nan); -1 -> nan
    cls_f = cls.astype(np.float32)
    cls_f[cls < 0] = np.nan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob.astype(np.float32), 1)
        dst.write(cls_f, 2)
        dst.set_band_description(1, "apple_prob")
        dst.set_band_description(2, "apple_class")


def mosaic_tiles(
    tile_paths: List[Path],
    aoi,
    out_prob_tif: Path,
    out_npz: Path,
) -> Tuple[np.ndarray, np.ndarray, Affine, Tuple[float, float, float, float], int]:
    """Mosaic AE tiles → predict per-tile then mosaic probs (avoids huge 64-band mosaic)."""
    # Actually we already have per-tile predictions saved as geotiffs with 2 bands
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = rio_merge(srcs, nodata=np.nan)
    # mosaic shape (2, H, W) if we merge pred tifs; but we merge AE then predict
    for s in srcs:
        s.close()
    # This function expects AE tiles — predict after merge is memory heavy.
    # Instead: merge already-predicted 2-band tifs.
    raise NotImplementedError


def build_focus_map(clf, features, aoi, county_geom) -> Dict[str, Any]:
    init_ee()
    ae = ae_image(YEAR)
    tiles = tile_bounds(aoi)
    print(f"Focus AOI tiles: {len(tiles)} at scale={GRID_SCALE_M}m")
    pred_paths: List[Path] = []
    failures: List[str] = []
    for i, (w, s, e, n) in enumerate(tiles):
        tag = f"tile_{i:03d}_{w:.3f}_{s:.3f}"
        ae_path = TILES_DIR / f"{tag}_ae.tif"
        pred_path = TILES_DIR / f"{tag}_pred.tif"
        try:
            print(f"  [{i+1}/{len(tiles)}] download {tag} …")
            download_ae_tile(ae, w, s, e, n, GRID_SCALE_M, ae_path)
            time.sleep(SLEEP)
            if not pred_path.exists():
                prob, cls, prof = predict_raster(ae_path, clf, features)
                # clip each tile with aoi
                write_prob_class_geotiff(pred_path, prob, cls, prof, mask_geom=aoi)
            pred_paths.append(pred_path)
            print(f"    ok → {pred_path.name}")
        except Exception as e:
            msg = f"{tag}: {e}"
            print("    FAIL", msg)
            failures.append(msg)
            time.sleep(2.0)

    if not pred_paths:
        raise RuntimeError("No focus tiles succeeded")

    srcs = [rasterio.open(p) for p in pred_paths]
    mosaic, transform = rio_merge(srcs, nodata=np.nan)
    meta0 = srcs[0].meta.copy()
    for s in srcs:
        s.close()
    prob = mosaic[0].astype(np.float32)
    cls_f = mosaic[1].astype(np.float32)
    height, width = prob.shape
    # Re-mask to AOI
    m = geometry_mask(
        [mapping(aoi)],
        transform=transform,
        invert=True,
        out_shape=(height, width),
    )
    prob[~m] = np.nan
    cls_f[~m] = np.nan
    cls = np.full_like(prob, -1, dtype=np.int16)
    valid = np.isfinite(prob)
    cls[valid] = (prob[valid] >= THRESH).astype(np.int16)

    out_tif = MAPS_V2 / "gongliu_focus_prob_class_2021_100m.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "nodata": np.nan,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(prob, 1)
        dst.write(cls_f, 2)
        dst.set_band_description(1, "apple_prob")
        dst.set_band_description(2, "apple_class")

    # CSV + NPZ of valid pixels
    rows, cols = np.where(valid)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    df = pd.DataFrame(
        {
            "lon": xs,
            "lat": ys,
            "prob": prob[valid],
            "pred": cls[valid],
        }
    )
    csv_path = MAPS_V2 / "gongliu_focus_grid_pred_2021_100m.csv"
    df.to_csv(csv_path, index=False)
    npz_path = MAPS_V2 / "gongliu_focus_map_2021_100m.npz"
    # store extent
    west = transform.c
    north = transform.f
    east = west + width * transform.a
    south = north + height * transform.e
    np.savez_compressed(
        npz_path,
        prob=prob,
        cls=cls,
        transform=np.array(transform),
        extent=np.array([west, east, south, north]),
        scale_m=GRID_SCALE_M,
    )
    n_pix = int(valid.sum())
    print(f"Focus map: {width}×{height}, valid pixels={n_pix}, wrote {out_tif.name}")
    return {
        "prob": prob,
        "cls": cls,
        "transform": transform,
        "extent": (west, east, south, north),
        "n_pixels": n_pix,
        "width": width,
        "height": height,
        "tif": str(out_tif),
        "csv": str(csv_path),
        "npz": str(npz_path),
        "failures": failures,
        "n_tiles_ok": len(pred_paths),
        "n_tiles": len(tiles),
    }


def pick_zoom_centers(hold: pd.DataFrame, focus: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Four windows: NW/SE holdout, ag-belt high-prob, unverified high-P (FP-prone)."""
    from scipy.ndimage import uniform_filter

    nw = hold[hold["cen_lon"] < 82.6]
    se = hold[hold["cen_lon"] >= 82.6]
    c_nw = (float(nw["cen_lon"].mean()), float(nw["cen_lat"].mean()))
    c_se = (float(se["cen_lon"].mean()), float(se["cen_lat"].mean()))

    prob = focus["prob"]
    transform = focus["transform"]
    p = np.nan_to_num(prob, nan=0.0)
    sm = uniform_filter(p, size=9)
    H, W = prob.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xs = transform.c + (xx + 0.5) * transform.a
    ys = transform.f + (yy + 0.5) * transform.e

    def far_mask(min_d_nw=0.05, min_d_se=0.05, lat0=43.28, lat1=43.48, extra=None):
        m = (
            (ys >= lat0)
            & (ys <= lat1)
            & (np.hypot(xs - c_nw[0], ys - c_nw[1]) > min_d_nw)
            & (np.hypot(xs - c_se[0], ys - c_se[1]) > min_d_se)
            & np.isfinite(prob)
        )
        if extra is not None:
            m = m & (np.hypot(xs - extra[0], ys - extra[1]) > 0.05)
        return m

    sm_hi = sm.copy()
    sm_hi[~far_mask()] = 0
    if np.nanmax(sm_hi) > 0.1:
        j, i = np.unravel_index(np.nanargmax(sm_hi), sm_hi.shape)
        c_hi = (float(xs[j, i]), float(ys[j, i]))
    else:
        c_hi = (82.70, 43.28)

    sm_fp = sm.copy()
    sm_fp[~far_mask(min_d_nw=0.06, min_d_se=0.06, lat0=43.30, lat1=43.47, extra=c_hi)] = 0
    if np.nanmax(sm_fp) > 0.1:
        j, i = np.unravel_index(np.nanargmax(sm_fp), sm_fp.shape)
        c_fp = (float(xs[j, i]), float(ys[j, i]))
    else:
        c_fp = (82.44, 43.34)

    return [
        {"id": "c_nw_holdout", "label_zh": "西北留出簇", "label_en": "NW holdout", "lon": c_nw[0], "lat": c_nw[1]},
        {"id": "d_se_holdout", "label_zh": "东南留出簇", "label_en": "SE holdout", "lon": c_se[0], "lat": c_se[1]},
        {"id": "e_highprob", "label_zh": "高概率斑块", "label_en": "High-prob patch", "lon": c_hi[0], "lat": c_hi[1]},
        {"id": "f_fp_prone", "label_zh": "易误判/未验证高概率", "label_en": "Unverified high-P", "lon": c_fp[0], "lat": c_fp[1]},
    ]


def try_s2_rgb(west, south, east, north, scale=30) -> Optional[np.ndarray]:
    """Optional S2 true-color; return HxWx3 uint8 or None."""
    try:
        init_ee()
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(f"{YEAR}-06-01", f"{YEAR}-09-30")
            .filterBounds(ee.Geometry.Rectangle([west, south, east, north]))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .select(["B4", "B3", "B2"])
        )
        region = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
        url = s2.getDownloadURL(
            {"scale": scale, "crs": "EPSG:4326", "region": region, "format": "GEO_TIFF"}
        )
        tmp = ZOOMS_DIR / f"_s2_tmp_{west:.3f}_{south:.3f}.tif"
        urllib.request.urlretrieve(url, tmp)
        with rasterio.open(tmp) as src:
            rgb = src.read()  # 3,H,W
        arr = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
        # percentile stretch
        out = np.zeros_like(arr, dtype=np.uint8)
        for b in range(3):
            band = arr[:, :, b]
            lo, hi = np.nanpercentile(band, [2, 98])
            if hi <= lo:
                hi = lo + 1
            out[:, :, b] = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        tmp.unlink(missing_ok=True)
        return out
    except Exception as e:
        print(f"  S2 RGB skipped: {e}")
        return None


def build_zoom(
    ae,
    clf,
    features,
    z: Dict[str, Any],
    apple,
    nonapple,
) -> Dict[str, Any]:
    half = ZOOM_HALF_DEG
    w, s = z["lon"] - half, z["lat"] - half
    e, n = z["lon"] + half, z["lat"] + half
    out: Dict[str, Any] = {**z, "bbox": (w, s, e, n), "failures": []}
    scale_used = None
    for scale in (10, 30):
        ae_path = ZOOMS_DIR / f"{z['id']}_ae_{scale}m.tif"
        pred_path = ZOOMS_DIR / f"{z['id']}_pred_{scale}m.tif"
        try:
            print(f"  Zoom {z['id']} scale={scale} …")
            download_ae_tile(ae, w, s, e, n, scale, ae_path)
            time.sleep(SLEEP)
            prob, cls, prof = predict_raster(ae_path, clf, features)
            write_prob_class_geotiff(pred_path, prob, cls, prof)
            west = prof["transform"].c
            north = prof["transform"].f
            height, width = prob.shape
            east = west + width * prof["transform"].a
            south = north + height * prof["transform"].e
            out.update(
                {
                    "prob": prob,
                    "cls": cls,
                    "extent": (west, east, south, north),
                    "scale_m": scale,
                    "tif": str(pred_path),
                    "n_pixels": int(np.isfinite(prob).sum()),
                }
            )
            scale_used = scale
            break
        except Exception as ex:
            msg = f"scale={scale}: {ex}"
            print("    FAIL", msg)
            out["failures"].append(msg)
            time.sleep(1.5)
    if scale_used is None:
        return out
    # optional RGB
    rgb = try_s2_rgb(w, s, e, n, scale=max(scale_used, 20))
    out["rgb"] = rgb
    return out


def draw_polygons(ax, apple, nonapple, lw=1.2):
    from matplotlib.patches import Polygon as MPLPoly

    for gdf, color, ls in (
        (apple, "#ff2d55", "-"),
        (nonapple, "#00d4ff", "--"),
    ):
        if gdf is None or len(gdf) == 0:
            continue
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in geoms:
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=5)


def build_figure(
    focus: Dict[str, Any],
    zooms: List[Dict[str, Any]],
    county_geom,
    apple,
    nonapple,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.gridspec as gridspec

    setup_matplotlib_zh()
    fig = plt.figure(figsize=(14.5, 10.5), dpi=300)
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    zoom_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    west, east, south, north = focus["extent"]
    # (a) probability
    im_a = ax_a.imshow(
        focus["prob"],
        origin="upper",
        extent=[west, east, south, north],
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    # county outline clipped to focus
    try:
        cg = county_geom.intersection(box(*FOCUS_BBOX))
        geoms = list(cg.geoms) if cg.geom_type == "MultiPolygon" else [cg]
        for g in geoms:
            xs, ys = g.exterior.xy
            ax_a.plot(xs, ys, color="white", lw=1.0, alpha=0.9)
    except Exception:
        pass
    draw_polygons(ax_a, apple, nonapple, lw=0.9)
    # zoom boxes
    for z, col in zip(zooms, ["#ffcc00", "#ffcc00", "#ffcc00", "#ffcc00"]):
        if "bbox" not in z:
            continue
        w, s, e, n = z["bbox"]
        ax_a.add_patch(
            plt.Rectangle(
                (w, s),
                e - w,
                n - s,
                fill=False,
                edgecolor=col,
                linewidth=1.2,
                zorder=6,
            )
        )
    ax_a.set_xlim(west, east)
    ax_a.set_ylim(south, north)
    ax_a.set_xlabel("经度 Lon (°E)")
    ax_a.set_ylabel("纬度 Lat (°N)")
    ax_a.set_title("(a) 巩留焦点区苹果园概率\nGongliu focus AOI · P(apple) · B3_poly_v2 · ~100 m", fontsize=10)
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.035, pad=0.02)
    cb_a.set_label("概率 Probability")

    # (b) binary
    cls_show = focus["cls"].astype(float)
    cls_show[focus["cls"] < 0] = np.nan
    cmap_bin = ListedColormap(["#2c3e50", "#f1c40f"])
    im_b = ax_b.imshow(
        cls_show,
        origin="upper",
        extent=[west, east, south, north],
        cmap=cmap_bin,
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    try:
        for g in geoms:
            xs, ys = g.exterior.xy
            ax_b.plot(xs, ys, color="white", lw=1.0, alpha=0.9)
    except Exception:
        pass
    draw_polygons(ax_b, apple, nonapple, lw=0.9)
    ax_b.set_xlim(west, east)
    ax_b.set_ylim(south, north)
    ax_b.set_xlabel("经度 Lon (°E)")
    ax_b.set_ylabel("纬度 Lat (°N)")
    ax_b.set_title(f"(b) 二值分类 Binary (阈值 threshold={THRESH})\n苹果园 / 非苹果园", fontsize=10)
    legend_elems = [
        Line2D([0], [0], color="#ff2d55", lw=2, label="留出苹果 Holdout apple"),
        Line2D([0], [0], color="#00d4ff", lw=2, ls="--", label="留出非苹果 Holdout non-apple"),
        Line2D([0], [0], color="#ffcc00", lw=2, label="放大窗 Zoom box"),
    ]
    ax_b.legend(handles=legend_elems, loc="lower right", fontsize=7, framealpha=0.85)

    panel_letters = ["c", "d", "e", "f"]
    for ax, z, let in zip(zoom_axes, zooms, panel_letters):
        if "prob" not in z:
            ax.set_title(f"({let}) {z.get('label_zh','')} FAILED")
            ax.text(0.5, 0.5, "\n".join(z.get("failures", ["no data"])[:2]), ha="center", va="center", transform=ax.transAxes, fontsize=7)
            continue
        ext = z["extent"]
        if z.get("rgb") is not None:
            ax.imshow(z["rgb"], origin="upper", extent=ext, aspect="equal", zorder=0)
            im = ax.imshow(
                z["prob"],
                origin="upper",
                extent=ext,
                cmap="viridis",
                vmin=0,
                vmax=1,
                aspect="equal",
                alpha=0.55,
                interpolation="nearest",
                zorder=1,
            )
        else:
            im = ax.imshow(
                z["prob"],
                origin="upper",
                extent=ext,
                cmap="viridis",
                vmin=0,
                vmax=1,
                aspect="equal",
                interpolation="nearest",
            )
        # clip polys to zoom
        w, s, e, n = z["bbox"]
        zb = box(w, s, e, n)
        apple_z = apple[apple.intersects(zb)] if len(apple) else apple
        non_z = nonapple[nonapple.intersects(zb)] if len(nonapple) else nonapple
        draw_polygons(ax, apple_z, non_z, lw=1.4)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_title(
            f"({let}) {z['label_zh']} / {z['label_en']}\n"
            f"{z['lon']:.3f}°E, {z['lat']:.3f}°N · {z['scale_m']} m",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "图9  巩留县苹果园制图（面样本训练 B3 / AlphaEarth 2021）\n"
        "Fig.9  Gongliu apple-orchard mapping — polygon-trained B3_poly_v2, AlphaEarth 2021",
        fontsize=12,
        y=0.98,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    # also copy into figures_v2 with alt name
    plt.close(fig)
    print(f"Wrote figure {out_png}")


def main():
    MAPS_V2.mkdir(parents=True, exist_ok=True)
    FIGS_V2.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    ZOOMS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model…")
    obj = joblib.load(MODEL_PATH)
    clf = obj["model"]
    features = list(obj["features"])
    print(f"  model={type(clf).__name__} features={len(features)} year={obj.get('year')} ver={obj.get('version')}")

    print("Loading geometries…")
    county = load_gongliu_geom()
    aoi = focus_aoi(county)
    apple, nonapple, hold = load_holdout_polygons()
    print(f"  holdout polys: apple={len(apple)} nonapple={len(nonapple)} manifest_n={len(hold)}")
    print(f"  focus AOI bounds={aoi.bounds}")

    print("Building focus ~100 m map…")
    focus = build_focus_map(clf, features, aoi, county)

    print("Selecting zoom centers…")
    try:
        centers = pick_zoom_centers(hold, focus)
    except Exception as e:
        print("scipy filter failed, using fallback centers:", e)
        centers = [
            {"id": "c_nw_holdout", "label_zh": "西北留出簇", "label_en": "NW holdout", "lon": 82.422, "lat": 43.498},
            {"id": "d_se_holdout", "label_zh": "东南留出簇", "label_en": "SE holdout", "lon": 82.846, "lat": 43.255},
            {"id": "e_highprob", "label_zh": "高概率斑块", "label_en": "High-prob patch", "lon": 82.55, "lat": 43.35},
            {"id": "f_fp_prone", "label_zh": "易误判区", "label_en": "FP-prone", "lon": 82.835, "lat": 43.252},
        ]
    for c in centers:
        print(f"  {c['id']}: {c['lon']:.4f}, {c['lat']:.4f}")

    print("Building zoom windows…")
    init_ee()
    ae = ae_image(YEAR)
    zooms = []
    for z in centers:
        zooms.append(build_zoom(ae, clf, features, z, apple, nonapple))

    fig_path = FIGS_V2 / "fig09_gongliu_map_2021.png"
    print("Composing figure…")
    build_figure(focus, zooms, county, apple, nonapple, fig_path)
    # also place a copy under maps_v2
    import shutil

    shutil.copy2(fig_path, MAPS_V2 / "fig09_gongliu_map_2021.png")

    zoom_meta = []
    all_failures = list(focus.get("failures", []))
    for z in zooms:
        zoom_meta.append(
            {
                "id": z["id"],
                "label_zh": z["label_zh"],
                "label_en": z["label_en"],
                "lon": z["lon"],
                "lat": z["lat"],
                "bbox": list(z.get("bbox", [])),
                "scale_m": z.get("scale_m"),
                "n_pixels": z.get("n_pixels"),
                "tif": z.get("tif"),
                "has_rgb": z.get("rgb") is not None,
                "failures": z.get("failures", []),
            }
        )
        all_failures.extend(z.get("failures", []))

    meta = {
        "title": "Gongliu focus AOI apple-orchard map 2021 (B3_poly_v2)",
        "model_path": str(MODEL_PATH),
        "model_keys": ["model", "features", "year", "feature_set", "version"],
        "model_year": int(obj.get("year", YEAR)),
        "model_version": obj.get("version"),
        "feature_set": obj.get("feature_set", "B3"),
        "features": features,
        "gee_project": PROJECT,
        "gee_mode": "restricted — getInfo/chunk sampling and getDownloadURL work; large Export.toDrive may fail",
        "limitation": (
            "Full Gongliu county (~4000+ km²) at 10 m is not feasible under GEE restricted mode; "
            "mapping uses a focus AOI covering both spatial holdout clusters at ~100 m, "
            "plus four ~2–3 km zoom windows at 10 m (30 m fallback)."
        ),
        "focus_aoi_bbox_requested": list(FOCUS_BBOX),
        "focus_aoi_bounds_actual": list(aoi.bounds),
        "grid_scale_m": GRID_SCALE_M,
        "grid_approx_deg": 0.001,
        "n_pixels_valid": focus["n_pixels"],
        "raster_width": focus["width"],
        "raster_height": focus["height"],
        "extent_wgs84": list(focus["extent"]),
        "threshold": THRESH,
        "threshold_note": "Default 0.5; no calibrated threshold shipped with B3_poly_v2 joblib",
        "n_tiles": focus["n_tiles"],
        "n_tiles_ok": focus["n_tiles_ok"],
        "outputs": {
            "focus_geotiff": focus["tif"],
            "focus_csv": focus["csv"],
            "focus_npz": focus["npz"],
            "figure": str(fig_path),
            "zooms": [z.get("tif") for z in zooms if z.get("tif")],
        },
        "holdout": {
            "n_apple": int(len(apple)),
            "n_nonapple": int(len(nonapple)),
            "clusters": "NW ~82.41–82.44/43.48–43.51; SE ~82.83–82.87/43.25–43.26",
        },
        "zooms": zoom_meta,
        "failures": all_failures,
        "colormap": "viridis (probability); binary dark/gold",
        "paper_caveat_zh_en": (
            "制图在 GEE 非商业计算配额受限模式下完成：巩留全县约四千余平方公里无法按 10 m 全覆盖导出，"
            "故采用覆盖两处空间留出簇的焦点 AOI，以约 100 m 规则格网通过 getDownloadURL 分块下载 "
            "AlphaEarth 2021 年度嵌入并本地预测；另对四处约 2–3 km 窗口按 10 m（失败则 30 m）下载预测。"
            "图面展示的是模型概率与阈值 0.5 的二值结果及留出面样本叠加，"
            "并非全县精细面积统计产品，亦未宣称县域制图精度。"
            " Mapping was produced under GEE restricted mode: full-county 10 m export is not feasible; "
            "a focus AOI (~100 m) covering both holdout clusters plus four ~2–3 km zoom windows "
            "(10 m / 30 m fallback) were predicted locally from AlphaEarth 2021 embeddings via getDownloadURL. "
            "The figure shows model probabilities and a 0.5-threshold binary map with holdout polygons overlaid; "
            "it is not a county-wide area statistic and does not claim county-level mapping accuracy."
        ),
    }
    meta_path = MAPS_V2 / "MAP_META.json"
    save_json(meta, meta_path)
    print("META →", meta_path)
    print("DONE")
    print(json.dumps({k: meta[k] for k in ("n_pixels_valid", "grid_scale_m", "failures", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
