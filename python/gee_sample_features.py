# -*- coding: utf-8 -*-
"""Sample REAL GEE features (AlphaEarth + S2 + S1 + DEM) at purified points.
Quota-aware: chunked getInfo ≤500, Gongliu-first, small n.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, setup_logging

log = setup_logging("gee_sample")
PROJECT = "hallowed-chain-458005-g9"
CHUNK = 400


def init_ee():
    ee.Initialize(project=PROJECT)


def build_s2(year: int, aoi):
    def maskS2(img):
        scl = img.select("SCL")
        mask = (
            scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        )
        qa = img.select("QA60")
        maskQa = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return (
            img.updateMask(mask.And(maskQa))
            .divide(10000)
            .copyProperties(img, ["system:time_start"])
        )

    def addIndices(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        evi = img.expression(
            "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
            {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
        ).rename("EVI")
        ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ndre = img.normalizedDifference(["B8", "B5"]).rename("NDRE")
        ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
        lswi = img.normalizedDifference(["B8", "B11"]).rename("LSWI")
        return img.addBands([ndvi, evi, ndwi, ndre, ndmi, lswi])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(f"{year}-04-01", f"{year}-11-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(maskS2)
        .map(addIndices)
    )
    ndvi = s2.select("NDVI")
    pheno = ee.Image.cat(
        [
            ndvi.max().rename("NDVI_max"),
            ndvi.mean().rename("NDVI_mean"),
            ndvi.reduce(ee.Reducer.stdDev()).rename("NDVI_std"),
            ndvi.reduce(ee.Reducer.percentile([10, 90])),
            s2.select("EVI").max().rename("EVI_max"),
            s2.select("EVI").mean().rename("EVI_mean"),
            s2.select("NDWI").mean().rename("NDWI_mean"),
            s2.select("NDRE").mean().rename("NDRE_mean"),
            s2.select("NDMI").mean().rename("NDMI_mean"),
            s2.select("LSWI").mean().rename("LSWI_mean"),
            s2.select("B4").mean().rename("B4_mean"),
            s2.select("B8").mean().rename("B8_mean"),
            s2.select("B11").mean().rename("B11_mean"),
            s2.select("B12").mean().rename("B12_mean"),
        ]
    )
    pheno = pheno.addBands(
        pheno.select("NDVI_max").subtract(pheno.select("NDVI_p10")).rename("NDVI_amp")
    )

    def add_doy(img):
        doy = ee.Image(img.date().getRelative("day", "year")).toFloat().rename("doy")
        return img.addBands(doy)

    # peak NDVI DOY -> greenup_doy (schema compat); senescence via fall inverted mosaic
    peak = s2.map(add_doy).qualityMosaic("NDVI").select("doy").rename("greenup_doy")
    s2_fall = s2.filterDate(f"{year}-08-01", f"{year}-11-01")

    def add_inv(img):
        return add_doy(img).addBands(img.select("NDVI").multiply(-1).rename("NDVIinv"))

    sen = (
        s2_fall.map(add_inv)
        .qualityMosaic("NDVIinv")
        .select("doy")
        .rename("senescence_doy")
    )
    # if fall empty, mosaic may be empty-masked; unmask with peak+50
    sen = sen.unmask(peak.add(50))
    return pheno.addBands(peak).addBands(sen)


def build_s1(year: int, aoi):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(f"{year}-04-01", f"{year}-11-01")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    s1f = ee.Image.cat(
        [
            s1.select("VV").mean().rename("VV_mean"),
            s1.select("VV").reduce(ee.Reducer.stdDev()).rename("VV_std"),
            s1.select("VH").mean().rename("VH_mean"),
            s1.select("VH").reduce(ee.Reducer.stdDev()).rename("VH_std"),
        ]
    )
    s1f = s1f.addBands(
        s1f.select("VV_mean").subtract(s1f.select("VH_mean")).rename("VV_VH_ratio")
    )
    vvlin = ee.Image(10).pow(s1f.select("VV_mean").divide(10))
    vhlin = ee.Image(10).pow(s1f.select("VH_mean").divide(10))
    rvi = vhlin.multiply(4).divide(vvlin.add(vhlin)).rename("RVI")
    return s1f.addBands(rvi)


def build_dem(aoi):
    dem = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003"))
    elev = dem.select("elevation")
    slope = dem.select("slope")
    aspect = dem.select("aspect")
    return ee.Image.cat(
        [
            elev,
            slope,
            aspect.multiply(np.pi / 180).sin().rename("aspect_sin"),
            aspect.multiply(np.pi / 180).cos().rename("aspect_cos"),
        ]
    )


def build_ae(year: int, aoi):
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(aoi)
        .mosaic()
    )
    # A00..A63 -> AE_00..AE_63
    names = ae.bandNames()
    new_names = names.map(lambda b: ee.String("AE_").cat(ee.String(b).slice(1)))
    return ae.rename(new_names)


def build_feature_image(year: int, aoi):
    return (
        build_s2(year, aoi)
        .addBands(build_s1(year, aoi))
        .addBands(build_dem(aoi))
        .addBands(build_ae(year, aoi))
    )


def points_to_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for i, r in df.iterrows():
        feats.append(
            ee.Feature(
                ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                {
                    "sid": int(i),
                    "year": int(r["year"]),
                    "label": int(r["label"]),
                    "block_id": int(r["block_id"]),
                    "lon": float(r["lon"]),
                    "lat": float(r["lat"]),
                },
            )
        )
    return ee.FeatureCollection(feats)


def sample_chunk(feat_img, df_chunk: pd.DataFrame, scale=30) -> list:
    fc = points_to_fc(df_chunk)
    sampled = feat_img.sampleRegions(
        collection=fc, properties=["sid", "year", "label", "block_id", "lon", "lat"],
        scale=scale, geometries=False,
    )
    return sampled.getInfo()["features"]


def features_to_rows(features: list) -> list:
    rows = []
    for f in features:
        p = f["properties"]
        rows.append(p)
    return rows


def sample_year(df_year: pd.DataFrame, year: int, aoi, chunk: int = CHUNK) -> pd.DataFrame:
    log.info("Building feature image for %d (%d pts)...", year, len(df_year))
    feat_img = build_feature_image(year, aoi)
    all_rows = []
    n = len(df_year)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = df_year.iloc[start:end].copy()
        # reindex sid locally unique
        sub = sub.reset_index(drop=True)
        sub.index = range(start, start + len(sub))
        attempt = 0
        while attempt < 4:
            try:
                t0 = time.time()
                feats = sample_chunk(feat_img, sub, scale=30)
                rows = features_to_rows(feats)
                log.info(
                    "  chunk %d-%d: got %d / %d in %.1fs",
                    start, end, len(rows), len(sub), time.time() - t0,
                )
                all_rows.extend(rows)
                break
            except Exception as e:
                attempt += 1
                log.warning("  chunk %d-%d attempt %d failed: %s", start, end, attempt, e)
                time.sleep(5 * attempt)
                if attempt >= 4:
                    # try half chunk
                    if len(sub) > 50:
                        mid = len(sub) // 2
                        log.info("  splitting chunk...")
                        for sub2 in (sub.iloc[:mid], sub.iloc[mid:]):
                            try:
                                feats = sample_chunk(feat_img, sub2.reset_index(drop=True))
                                all_rows.extend(features_to_rows(feats))
                            except Exception as e2:
                                log.error("  subchunk failed: %s", e2)
                    else:
                        log.error("  giving up on chunk %d-%d", start, end)
        time.sleep(1.0)
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def sample_transfer_years(base_pts: pd.DataFrame, years: list, aoi, n_per_class: int, seed: int):
    """Reuse 2021 locations (or mixed orchard/non) for post-2021 feature years."""
    rng = np.random.default_rng(seed)
    src = base_pts[base_pts["year"] == 2021].copy()
    if len(src) == 0:
        src = base_pts.copy()
    frames = []
    for year in years:
        parts = []
        for lab in (0, 1):
            sub = src[src["label"] == lab]
            take = min(n_per_class, len(sub))
            if take == 0:
                continue
            sel = sub.sample(n=take, random_state=int(rng.integers(0, 1e9)))
            sel = sel.copy()
            sel["year"] = year
            parts.append(sel)
        if not parts:
            continue
        pts = pd.concat(parts, ignore_index=True)
        log.info("Transfer sample year=%d n=%d", year, len(pts))
        try:
            df = sample_year(pts, year, aoi, chunk=min(CHUNK, 250))
            if len(df):
                frames.append(df)
                out_dir = ROOT / "data" / "samples" / "real"
                df.to_csv(out_dir / f"features_{year}.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            log.error("Transfer year %d failed: %s", year, e)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--transfer-years", default="2022")
    ap.add_argument("--transfer-n", type=int, default=200, help="per class per transfer year")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--years", default="2019,2020,2021")
    args = ap.parse_args()

    init_ee()
    out_dir = ROOT / "data" / "samples" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)

    pts_path = Path(args.points) if args.points else out_dir / "points_gongliu_2019_2021.csv"
    pts = pd.read_csv(pts_path)
    years = [int(y) for y in args.years.split(",") if y.strip()]

    # AOI from Gongliu asset
    aoi = (
        ee.FeatureCollection("projects/hallowed-chain-458005-g9/assets/Yili_county")
        .filter(ee.Filter.eq("name", "巩留县"))
        .geometry()
    )

    run_log = {"years_ok": [], "years_fail": [], "errors": [], "counts": {}}
    frames = []
    for year in years:
        dfy = pts[pts["year"] == year].copy().reset_index(drop=True)
        if len(dfy) == 0:
            continue
        try:
            sampled = sample_year(dfy, year, aoi, chunk=args.chunk)
            if len(sampled) == 0:
                run_log["years_fail"].append(year)
                run_log["errors"].append(f"{year}: empty")
                continue
            sampled.to_csv(out_dir / f"features_{year}.csv", index=False, encoding="utf-8-sig")
            frames.append(sampled)
            run_log["years_ok"].append(year)
            run_log["counts"][str(year)] = {
                "n": len(sampled),
                "pos": int((sampled["label"] == 1).sum()),
                "neg": int((sampled["label"] == 0).sum()),
            }
            log.info("Year %d saved %d rows", year, len(sampled))
        except Exception as e:
            log.exception("Year %d failed", year)
            run_log["years_fail"].append(year)
            run_log["errors"].append(f"{year}: {e}")

    if not args.skip_transfer and args.transfer_years.strip():
        ty = [int(y) for y in args.transfer_years.split(",") if y.strip()]
        tframes = sample_transfer_years(pts, ty, aoi, args.transfer_n, seed=42)
        for df in tframes:
            y = int(df["year"].iloc[0])
            frames.append(df)
            run_log["years_ok"].append(y)
            run_log["counts"][str(y)] = {
                "n": len(df),
                "pos": int((df["label"] == 1).sum()),
                "neg": int((df["label"] == 0).sum()),
            }

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # ensure AE columns exist / rename if needed
        # fill missing model schema columns with NaN
        all_path = out_dir / "purified_samples_real.csv"
        all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        run_log["combined_csv"] = str(all_path)
        run_log["combined_n"] = len(all_df)
        log.info("Combined %d rows -> %s", len(all_df), all_path)
    with open(out_dir / "gee_sample_log.json", "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)
    print(json.dumps(run_log, ensure_ascii=False, indent=2))
    return run_log


if __name__ == "__main__":
    main()
