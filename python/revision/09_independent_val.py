# -*- coding: utf-8 -*-
"""Build independent validation set (≥300) that NEVER enters training.

Protocol (semi-auto + QC flags):
1. Stratified candidate points across Gongliu from AOMC (apple) and
   ESA WorldCover (cropland/forest/shrub/built/bare) via GEE.
2. Exclude any point within 200 m of training feature locations (2019–2023).
3. Collapse multi-class → apple vs non-apple for metrics; keep multi-class column.
4. Sample AlphaEarth (+ light S2 stats if quota allows) at candidates.
5. Evaluate best AE-only (B3) model trained on 2019–2021 (excluding near-val).

Residual uncertainty is labeled in `qc_flag` / README note.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import (
    FEATURE_SETS,
    IND_VAL,
    METRICS,
    MODELS,
    TABLES,
    ensure_out_dirs,
    get_estimator,
    load_config,
    load_feature_years,
    matrix_X,
    save_json,
)
from utils import ROOT, compute_metrics, spatial_block_id

PROJECT = "hallowed-chain-458005-g9"
ROI_ASSET = "projects/hallowed-chain-458005-g9/assets/Yili_county"


def init_ee():
    ee.Initialize(project=PROJECT)


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def sample_candidates_local_aomc(n_apple=200, n_other=400, seed=42, exclude_xy=None, min_dist_m=200):
    """Local AOMC + random county negatives with distance exclusion."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.features import geometry_mask
    from rasterio.warp import transform as rio_transform, transform_bounds
    from shapely.geometry import mapping, shape
    import geopandas as gpd
    from shapely.ops import unary_union

    rng = np.random.default_rng(seed)
    gpath = ROOT / "data" / "roi" / "Gongliu.geojson"
    gdf = gpd.read_file(gpath)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    geom = unary_union(gdf.geometry.values)
    tif = ROOT / "data" / "aomc" / "AOMC_30_Xinjiang_2021.tif"

    with rasterio.open(tif) as ds:
        left, bottom, right, top = transform_bounds(CRS.from_epsg(4326), ds.crs, *geom.bounds)
        from rasterio.windows import from_bounds

        window = from_bounds(left - 300, bottom - 300, right + 300, top + 300, transform=ds.transform)
        data = ds.read(1, window=window, boundless=True, fill_value=0)
        transform = ds.window_transform(window)
        gdf_p = gpd.GeoDataFrame(geometry=[geom], crs=4326).to_crs(ds.crs)
        county = ~geometry_mask([mapping(gdf_p.geometry.iloc[0])], out_shape=data.shape, transform=transform, invert=False)

    orchard = (data > 0) & county
    from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

    st = generate_binary_structure(2, 1)
    core = binary_erosion(orchard, structure=st, iterations=1)
    if core.sum() < 100:
        core = orchard
    near = binary_dilation(orchard, structure=np.ones((3, 3)), iterations=3)
    hard_neg = county & (~near)

    def xy_to_lonlat(xs, ys):
        # pixel centers
        xs = xs.astype(float) + 0.5
        ys = ys.astype(float) + 0.5
        x = transform.c + xs * transform.a + ys * transform.b
        y = transform.f + xs * transform.d + ys * transform.e
        lon, lat = rio_transform(ds.crs, CRS.from_epsg(4326), x, y)
        return np.asarray(lon), np.asarray(lat)

    with rasterio.open(tif) as ds:  # reopen for crs
        crs = ds.crs

    def pick(mask, n, class_name, apple_label):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return []
        # subsample candidates
        take = min(len(ys), n * 20)
        sel = rng.choice(len(ys), take, replace=False)
        xs, ys = xs[sel], ys[sel]
        # transform
        xs_f = xs.astype(float) + 0.5
        ys_f = ys.astype(float) + 0.5
        x = transform.c + xs_f * transform.a + ys_f * transform.b
        y = transform.f + xs_f * transform.d + ys_f * transform.e
        lon, lat = rio_transform(crs, CRS.from_epsg(4326), x, y)
        lon, lat = np.asarray(lon), np.asarray(lat)
        out = []
        for lo, la in zip(lon, lat):
            if exclude_xy is not None:
                d = haversine_m(lo, la, exclude_xy[:, 0], exclude_xy[:, 1])
                if d.min() < min_dist_m:
                    continue
            out.append({
                "lon": float(lo),
                "lat": float(la),
                "class_multi": class_name,
                "label_apple": int(apple_label),
                "source": "AOMC2021_core" if apple_label else "AOMC_hardneg_buffer",
                "qc_flag": "semi_auto_aomc",
            })
            if len(out) >= n:
                break
        return out

    rows = []
    rows += pick(core, n_apple, "apple_orchard", 1)
    # diversify non-apple with random hard-neg (later refined by WorldCover if available)
    rows += pick(hard_neg, n_other, "non_apple_untyped", 0)
    return pd.DataFrame(rows)


def refine_with_worldcover(df: pd.DataFrame, chunk=80):
    """Attach ESA WorldCover class via GEE; map to multi-class names."""
    init_ee()
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    # 10 tree, 20 shrub, 30 grassland, 40 cropland, 50 built, 60 bare, 80 water, ...
    mapping = {
        10: "forest",
        20: "shrub",
        30: "grassland",
        40: "cropland",
        50: "built_up",
        60: "bare",
        80: "water",
        90: "wetland",
        95: "mangrove",
        100: "moss",
    }
    classes = []
    for i in range(0, len(df), chunk):
        sub = df.iloc[i : i + chunk]
        feats = [
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"i": int(j)})
            for j, r in sub.iterrows()
        ]
        fc = ee.FeatureCollection(feats)
        sampled = wc.sampleRegions(collection=fc, scale=10, geometries=False)
        try:
            info = sampled.getInfo()["features"]
            by_i = {f["properties"]["i"]: f["properties"].get("Map", None) for f in info}
        except Exception as e:
            print("WorldCover chunk failed:", e)
            by_i = {}
            time.sleep(1)
        for j, r in sub.iterrows():
            code = by_i.get(int(j))
            classes.append(code)
        time.sleep(0.3)
    df = df.copy()
    df["worldcover"] = classes
    # refine multi-class for non-apple; apple stays apple_orchard if AOMC
    multi = []
    for _, r in df.iterrows():
        if r["label_apple"] == 1:
            multi.append("apple_orchard")
            continue
        name = mapping.get(int(r["worldcover"]) if pd.notna(r["worldcover"]) else -1, "other")
        # note: other_orchard not separable from WorldCover alone
        if name == "forest":
            multi.append("forest")
        elif name == "shrub":
            multi.append("shrub")
        elif name == "cropland":
            multi.append("cropland")
        elif name == "built_up":
            multi.append("built_up")
        else:
            multi.append(name if name != "other" else "other")
    df["class_multi"] = multi
    df["qc_flag"] = df.apply(
        lambda r: "aomc_apple" if r.label_apple == 1 else f"wc_{r.class_multi}",
        axis=1,
    )
    # residual: cannot reliably separate other orchard species → flag
    df.loc[df["label_apple"] == 0, "uncertainty_note"] = "other_orchard_not_separated_from_WC"
    df.loc[df["label_apple"] == 1, "uncertainty_note"] = "aomc_domain_shift_possible"
    return df


def sample_ae_features(df: pd.DataFrame, year=2021, chunk=60):
    init_ee()
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    bands = [f"A{i:02d}" for i in range(64)]  # GEE uses A00..; we'll rename
    # Actual AlphaEarth band names are A00..A63 in some versions; check
    # Catalog uses embedding bands often named A00 or band_0 — probe first image
    try:
        probe = (
            ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .first()
            .bandNames()
            .getInfo()
        )
    except Exception:
        probe = [f"A{i:02d}" for i in range(64)]
    print("AE bands sample:", probe[:5], "... n=", len(probe))

    rows_out = []
    for i in range(0, len(df), chunk):
        sub = df.iloc[i : i + chunk]
        feats = []
        for j, r in sub.iterrows():
            feats.append(ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"sid": int(j)}))
        fc = ee.FeatureCollection(feats)
        sampled = ae.sampleRegions(collection=fc, scale=10, geometries=False)
        try:
            info = sampled.getInfo()["features"]
        except Exception as e:
            print("AE sample fail chunk", i, e)
            time.sleep(2)
            continue
        by_sid = {f["properties"]["sid"]: f["properties"] for f in info}
        for j, r in sub.iterrows():
            props = by_sid.get(int(j))
            if not props:
                continue
            rec = r.to_dict()
            # map embedding bands to AE_00..
            emb_keys = [k for k in props.keys() if k not in ("sid", "label_apple", "lon", "lat")]
            # prefer sorted A## or band_##
            emb_keys = sorted([k for k in emb_keys if k.startswith("A") or k.startswith("b") or k.startswith("embedding")])
            if not emb_keys:
                emb_keys = [k for k in props.keys() if isinstance(props[k], (int, float))]
                emb_keys = emb_keys[:64]
            for bi, k in enumerate(emb_keys[:64]):
                rec[f"AE_{bi:02d}"] = props[k]
            # fill missing
            for bi in range(64):
                rec.setdefault(f"AE_{bi:02d}", np.nan)
            rec["year"] = year
            rows_out.append(rec)
        print(f"AE sampled {len(rows_out)}/{len(df)}")
        time.sleep(0.4)
    return pd.DataFrame(rows_out)


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]

    train_pts = load_feature_years([2019, 2020, 2021, 2022, 2023])[["lon", "lat"]].drop_duplicates()
    exclude_xy = train_pts.values.astype(float)

    # Build more candidates then filter
    cand = sample_candidates_local_aomc(
        n_apple=250, n_other=500, seed=seed, exclude_xy=exclude_xy, min_dist_m=200
    )
    print("candidates after exclusion:", len(cand), "apple", int((cand.label_apple == 1).sum()))

    # WorldCover refine (best-effort under restricted mode)
    try:
        cand = refine_with_worldcover(cand)
        print("WorldCover refined")
    except Exception as e:
        print("WorldCover refine failed, keeping AOMC labels:", e)
        cand["worldcover"] = np.nan
        cand["uncertainty_note"] = "worldcover_unavailable"

    # Stratified downsample to ~400–600 if possible, else ≥300
    rng = np.random.default_rng(seed)
    parts = []
    # apple
    apple = cand[cand.label_apple == 1]
    n_apple = min(200, len(apple))
    parts.append(apple.sample(n=n_apple, random_state=seed) if len(apple) > n_apple else apple)
    # non-apple by class
    non = cand[cand.label_apple == 0]
    if "class_multi" in non.columns:
        for cls, grp in non.groupby("class_multi"):
            take = min(80, len(grp))
            if take > 0:
                parts.append(grp.sample(n=take, random_state=seed))
    else:
        take = min(350, len(non))
        parts.append(non.sample(n=take, random_state=seed))
    val = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["lon", "lat"])
    # ensure ≥300
    if len(val) < 300 and len(cand) >= 300:
        extra = cand.sample(n=min(400, len(cand)), random_state=seed + 1)
        val = pd.concat([val, extra]).drop_duplicates(subset=["lon", "lat"]).head(500)
    print("val size before AE:", len(val), "pos", int((val.label_apple == 1).sum()))

    # Sample AE features
    try:
        val_feat = sample_ae_features(val, year=2021, chunk=50)
    except Exception as e:
        print("AE sampling failed:", e)
        val_feat = val.copy()
        for i in range(64):
            val_feat[f"AE_{i:02d}"] = np.nan

    # Drop rows without AE
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    if all(c in val_feat.columns for c in ae_cols):
        val_feat = val_feat.dropna(subset=ae_cols, how="all")
        # if many NaN, drop incomplete
        val_feat = val_feat.dropna(subset=ae_cols[:8])

    val_feat["label"] = val_feat["label_apple"].astype(int)
    val_feat["block_id"] = spatial_block_id(val_feat["lon"].values, val_feat["lat"].values, 0.05)
    val_feat["independent"] = True
    val_path = IND_VAL / "independent_val_points.csv"
    val_feat.to_csv(val_path, index=False)
    print("saved", val_path, "n=", len(val_feat))

    # Train B3 on 2019-2021 excluding any train point within 200m of val (already excluded at candidate stage)
    train = load_feature_years([2019, 2020, 2021])
    Xtr, cols = matrix_X(train, FEATURE_SETS["B3"], cfg)
    clf = get_estimator("lgbm", seed)
    clf.fit(Xtr, train["label"].values.astype(int))
    joblib.dump({"model": clf, "cols": cols}, MODELS / "B3_full_2019_2021.joblib")

    # Evaluate if AE present
    if len(val_feat) >= 50 and ae_cols[0] in val_feat.columns and val_feat[ae_cols[0]].notna().any():
        Xva = val_feat[ae_cols].fillna(val_feat[ae_cols].median()).values.astype(float)
        yva = val_feat["label"].values.astype(int)
        prob = clf.predict_proba(Xva)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = compute_metrics(yva, pred, prob)
        # per-class confusion helpers
        cm_by = (
            pd.crosstab(
                val_feat["class_multi"].fillna("unk"),
                pred,
                rownames=["class_multi"],
                colnames=["pred_apple"],
            ).to_dict()
            if "class_multi" in val_feat.columns
            else {}
        )
        out = {
            "n": int(len(val_feat)),
            "n_apple": int((yva == 1).sum()),
            "n_nonapple": int((yva == 0).sum()),
            "metrics": m,
            "protocol": "semi_auto_AOMC2021_core_plus_ESA_WorldCover_excl200m_from_train",
            "residual_uncertainty": [
                "other_orchard_species_not_separated",
                "AOMC_commission_omission_may_propagate_to_apple_labels",
                "no_full_visual_interpretation_of_all_points",
            ],
            "pred_by_class_multi": cm_by,
        }
        save_json(out, METRICS / "independent_val_B3.json")
        pd.DataFrame([{"model": "B3_AE", **m, "n": len(val_feat)}]).to_csv(
            TABLES / "independent_val_metrics.csv", index=False
        )
        print("Independent val B3:", m)
    else:
        save_json(
            {"n": int(len(val_feat)), "error": "insufficient_AE_features", "path": str(val_path)},
            METRICS / "independent_val_B3.json",
        )
        print("Independent points saved but AE eval incomplete")

    # meta
    save_json(
        {
            "n_candidates": int(len(cand)),
            "n_val": int(len(val_feat)),
            "exclusion_m": 200,
            "year_ae": 2021,
        },
        IND_VAL / "meta.json",
    )


if __name__ == "__main__":
    main()
