# -*- coding: utf-8 -*-
"""Export Gongliu + full Yili county geometries from GEE asset to GeoJSON."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, setup_logging

log = setup_logging("export_roi")
ASSET = "projects/hallowed-chain-458005-g9/assets/Yili_county"
PROJECT = "hallowed-chain-458005-g9"


def main():
    ee.Initialize(project=PROJECT)
    out = ROOT / "data" / "roi"
    out.mkdir(parents=True, exist_ok=True)
    fc = ee.FeatureCollection(ASSET)
    # full valley
    full_info = fc.getInfo()
    full_path = out / "Yili_county_full.geojson"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_info, f, ensure_ascii=False)
    log.info("Wrote %s (%d features)", full_path, len(full_info.get("features", [])))

    gongliu = fc.filter(ee.Filter.eq("name", "巩留县"))
    g_info = gongliu.getInfo()
    g_path = out / "Gongliu.geojson"
    with open(g_path, "w", encoding="utf-8") as f:
        json.dump(g_info, f, ensure_ascii=False)
    # also simplified bounds meta
    geom = gongliu.geometry()
    bounds = geom.bounds().getInfo()
    area_km2 = geom.area(maxError=100).divide(1e6).getInfo()
    meta = {
        "asset": ASSET,
        "project": PROJECT,
        "name": "巩留县",
        "area_km2": area_km2,
        "bounds": bounds,
        "n_features_full": len(full_info.get("features", [])),
    }
    with open(out / "roi_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Gongliu area=%.1f km2", area_km2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return g_path, full_path


if __name__ == "__main__":
    main()
