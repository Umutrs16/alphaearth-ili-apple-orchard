# -*- coding: utf-8 -*-
"""Classify Gongliu preview maps with B3 (AE-only) for available years.

Under GEE restricted mode: sample a regular grid via getInfo chunks,
predict locally, write GeoTIFF-like NPZ + RGB preview PNGs.
Years: 2019, 2021, 2022, 2024 (fallback 2023).
"""
from __future__ import annotations

import time
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd

from common import (
    FEATURE_SETS,
    FIGURES,
    MAPS,
    METRICS,
    MODELS,
    ensure_out_dirs,
    get_estimator,
    load_config,
    load_feature_years,
    matrix_X,
    save_json,
    setup_matplotlib_zh,
)

PROJECT = "hallowed-chain-458005-g9"
ROI_ASSET = "projects/hallowed-chain-458005-g9/assets/Yili_county"


def init_ee():
    ee.Initialize(project=PROJECT)


def gongliu_bounds():
    import json
    from pathlib import Path as P

    g = json.load(open(P("/workspace/apple_orchard_project/data/roi/Gongliu.geojson")))
    coords = g["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def make_grid(step_deg=0.01):
    """~1 km grid for preview."""
    import json
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    import geopandas as gpd

    gdf = gpd.read_file("/workspace/apple_orchard_project/data/roi/Gongliu.geojson")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    geom = unary_union(gdf.geometry.values)
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx, maxx, step_deg)
    ys = np.arange(miny, maxy, step_deg)
    pts = []
    for y in ys:
        for x in xs:
            if geom.contains(Point(x, y)):
                pts.append((x, y))
    return np.array(pts), (minx, miny, maxx, maxy, step_deg)


def sample_ae_grid(pts, year, chunk=80):
    init_ee()
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    rows = []
    for i in range(0, len(pts), chunk):
        batch = pts[i : i + chunk]
        feats = [
            ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"i": int(i + k)})
            for k, (x, y) in enumerate(batch)
        ]
        fc = ee.FeatureCollection(feats)
        try:
            info = ae.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"]
        except Exception as e:
            print(f"year {year} chunk {i} fail:", e)
            time.sleep(2)
            continue
        for f in info:
            props = f["properties"]
            idx = props.get("i")
            emb = []
            keys = sorted([k for k in props.keys() if k != "i"])
            # keep numeric embedding-like
            num_keys = []
            for k in keys:
                v = props[k]
                if isinstance(v, (int, float)):
                    num_keys.append(k)
            # prefer A## order
            akeys = sorted([k for k in num_keys if str(k).startswith("A")])
            use = akeys if len(akeys) >= 32 else num_keys[:64]
            for bi in range(64):
                if bi < len(use):
                    emb.append(float(props[use[bi]]))
                else:
                    emb.append(np.nan)
            if idx is None:
                continue
            x, y = pts[int(idx)]
            rows.append({"lon": x, "lat": y, **{f"AE_{b:02d}": emb[b] for b in range(64)}})
        print(f"{year}: grid AE {len(rows)}/{len(pts)}")
        time.sleep(0.35)
    return pd.DataFrame(rows)


def train_b3():
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    path = MODELS / "B3_full_2019_2021.joblib"
    if path.exists():
        return joblib.load(path)
    train = load_feature_years([2019, 2020, 2021])
    Xtr, cols = matrix_X(train, FEATURE_SETS["B3"], cfg)
    clf = get_estimator("lgbm", seed)
    clf.fit(Xtr, train["label"].values.astype(int))
    obj = {"model": clf, "cols": cols}
    joblib.dump(obj, path)
    return obj


def rasterize_preds(df, bounds, step):
    minx, miny, maxx, maxy = bounds[:4]
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    prob = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    cls = np.full((len(ys), len(xs)), -1, dtype=np.int16)
    x_to_i = {round(x, 6): i for i, x in enumerate(xs)}
    y_to_j = {round(y, 6): j for j, y in enumerate(ys)}
    for _, r in df.iterrows():
        i = x_to_i.get(round(float(r.lon), 6))
        j = y_to_j.get(round(float(r.lat), 6))
        if i is None or j is None:
            # nearest
            i = int(np.argmin(np.abs(xs - r.lon)))
            j = int(np.argmin(np.abs(ys - r.lat)))
        prob[j, i] = r["prob"]
        cls[j, i] = int(r["pred"])
    return xs, ys, prob, cls


def main():
    ensure_out_dirs()
    plt = setup_matplotlib_zh()
    obj = train_b3()
    clf, cols = obj["model"], obj["cols"]
    pts, bounds = make_grid(step_deg=0.012)  # ~1.3 km — keep quota small
    print("grid points:", len(pts), "bounds", bounds)

    years = [2019, 2021, 2022, 2024]
    done = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    for ai, year in enumerate(years):
        try:
            gdf = sample_ae_grid(pts, year, chunk=70)
            if len(gdf) < 20:
                raise RuntimeError(f"too few AE samples for {year}")
            X = gdf[cols].fillna(gdf[cols].median()).values.astype(float)
            prob = clf.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)
            gdf = gdf.assign(prob=prob, pred=pred)
            gdf.to_csv(MAPS / f"grid_pred_{year}.csv", index=False)
            xs, ys, P, C = rasterize_preds(gdf, bounds, bounds[4])
            np.savez_compressed(MAPS / f"map_{year}.npz", xs=xs, ys=ys, prob=P, cls=C)
            ax = axes[ai]
            im = ax.imshow(
                P,
                origin="lower",
                extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            ax.set_title(f"{year} 苹果园概率 (B3/AE)")
            ax.set_xlabel("经度")
            ax.set_ylabel("纬度")
            plt.colorbar(im, ax=ax, fraction=0.046)
            done.append(year)
        except Exception as e:
            print("year failed", year, e)
            if year == 2024:
                # fallback 2023
                try:
                    gdf = sample_ae_grid(pts, 2023, chunk=70)
                    X = gdf[cols].fillna(gdf[cols].median()).values.astype(float)
                    prob = clf.predict_proba(X)[:, 1]
                    gdf = gdf.assign(prob=prob, pred=(prob >= 0.5).astype(int))
                    gdf.to_csv(MAPS / "grid_pred_2023.csv", index=False)
                    xs, ys, P, C = rasterize_preds(gdf, bounds, bounds[4])
                    np.savez_compressed(MAPS / "map_2023.npz", xs=xs, ys=ys, prob=P, cls=C)
                    ax = axes[ai]
                    im = ax.imshow(P, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                                   cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
                    ax.set_title("2023 苹果园概率 (B3/AE, 2024回退)")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    done.append(2023)
                except Exception as e2:
                    axes[ai].set_title(f"{year} 失败")
                    axes[ai].text(0.5, 0.5, str(e)[:80], ha="center", va="center", transform=axes[ai].transAxes, fontsize=8)
                    print("fallback also failed", e2)
            else:
                axes[ai].set_title(f"{year} 失败")
                axes[ai].text(0.5, 0.5, str(e)[:80], ha="center", va="center", transform=axes[ai].transAxes, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig09_classification_maps.png", bbox_inches="tight")
    # also copy as main fig name
    fig.savefig(MAPS / "preview_maps_panel.png", bbox_inches="tight")
    plt.close(fig)
    save_json({"years_done": done, "grid_n": int(len(pts)), "step_deg": bounds[4]}, METRICS / "maps_meta.json")
    print("maps done", done)


if __name__ == "__main__":
    main()
