# -*- coding: utf-8 -*-
"""Try sampling 2024 AE (+ optional S2/S1) at a subset of 2021 points if quota allows."""
from __future__ import annotations

import time

import ee
import numpy as np
import pandas as pd

from common import IND_VAL, METRICS, TABLES, ensure_out_dirs, load_feature_years, save_json
from utils import ROOT

PROJECT = "hallowed-chain-458005-g9"


def main():
    ensure_out_dirs()
    ee.Initialize(project=PROJECT)
    src = load_feature_years([2021])
    # balanced subset
    pos = src[src.label == 1].sample(n=min(80, (src.label == 1).sum()), random_state=42)
    neg = src[src.label == 0].sample(n=min(80, (src.label == 0).sum()), random_state=42)
    pts = pd.concat([pos, neg])[["sid", "lon", "lat", "label", "block_id"]].reset_index(drop=True)

    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate("2024-01-01", "2025-01-01")
        .mosaic()
    )
    rows = []
    chunk = 40
    for i in range(0, len(pts), chunk):
        sub = pts.iloc[i : i + chunk]
        feats = [
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"sid": int(r.sid) if pd.notna(r.sid) else int(i + k), "label": int(r.label)})
            for k, r in sub.iterrows()
        ]
        try:
            info = ae.sampleRegions(ee.FeatureCollection(feats), scale=10, geometries=False).getInfo()["features"]
        except Exception as e:
            print("2024 AE fail", e)
            save_json({"ok": False, "error": str(e)}, METRICS / "sample_2024.json")
            return
        for f in info:
            props = f["properties"]
            rec = {"label": props.get("label"), "sid": props.get("sid")}
            keys = sorted([k for k in props if k.startswith("A")])
            if not keys:
                keys = [k for k, v in props.items() if isinstance(v, (int, float)) and k not in ("sid", "label")][:64]
            for bi, k in enumerate(keys[:64]):
                rec[f"AE_{bi:02d}"] = props[k]
            rows.append(rec)
        print("2024 AE", len(rows))
        time.sleep(0.4)

    out = pd.DataFrame(rows)
    # attach lon/lat
    out = out.merge(pts.drop(columns=["label"]), on="sid", how="left")
    out["year"] = 2024
    path = ROOT / "data" / "samples" / "real" / "features_2024_subset.csv"
    out.to_csv(path, index=False)
    save_json({"ok": True, "n": len(out), "path": str(path)}, METRICS / "sample_2024.json")
    print("wrote", path, len(out))


if __name__ == "__main__":
    main()
