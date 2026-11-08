# -*- coding: utf-8 -*-
"""Shortcut diagnostics on the v3 expanded Gongliu holdout (reuses 21_shortcut_diagnostics logic)."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sd21", HERE / "21_shortcut_diagnostics.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)

OUT = sd.OUT
V3 = OUT / "v3"
FIG3 = ROOT / "outputs" / "revision" / "figures_v3"
CACHE_V3 = OUT / "poly_terrain_features_2021_v3.csv"
sd.TERRAIN_CACHE = CACHE_V3
sd.SUPTITLE = "巩留面样本空间 holdout v3（113 面/521 点；面级多数投票计 101 面[≥3 点]）：地理捷径诊断（苹果 vs 非苹果[0+2]）"


def main():
    FIG3.mkdir(parents=True, exist_ok=True)
    if not CACHE_V3.exists() and (OUT / "poly_terrain_features_2021.csv").exists():
        shutil.copy(OUT / "poly_terrain_features_2021.csv", CACHE_V3)  # seed with v2 points
    pts = pd.read_csv(OUT / "sample_points_v3.csv")
    ae = pd.read_csv(OUT / "poly_ae_features_2021_v3.csv")
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    terrain, gee_errors = sd.sample_terrain(pts)
    df = (pts.merge(ae[["sample_id"] + ae_cols], on="sample_id", how="left")
             .merge(terrain[["sample_id", "elevation", "slope"]], on="sample_id", how="left"))
    ok = df[ae_cols + ["elevation", "slope"]].notna().all(axis=1)
    train = df[(df.split == "train") & ok].copy()
    hold = df[(df.split == "val_holdout") & ok].copy()
    hold["class_code_3"] = hold["class_code"]
    hold["class_code"] = (hold["class_code_3"] == 1).astype(int)  # binary apple vs (0+2)
    print(f"train={len(train)} hold={len(hold)} polys={hold.poly_name.nunique()} dropped={(~ok).sum()}")

    sets = [("AE-only", ae_cols, True), ("DEM-only", ["elevation", "slope"], False),
            ("lat/lon-only", ["lon", "lat"], False), ("AE+DEM", ae_cols + ["elevation", "slope"], False),
            ("AE+lat/lon", ae_cols + ["lon", "lat"], False)]
    results = []
    for name, cols, reload_ in sets:
        r = sd.eval_feature_set(name, cols, train, hold, reload_ae=reload_)
        results.append(r)
        print(f"{name}: pt F1={r['point_level']['F1']:.3f} OA={r['point_level']['OA']:.3f} AP={r['point_level'].get('AP', float('nan')):.3f} | "
              f"poly n={r['poly_majority']['n_polys']} F1={r['poly_majority']['F1']:.3f} OA={r['poly_majority']['OA']:.3f}")
    ae_r = results[0]
    for r in results:
        r["delta_F1_point_vs_AE"] = r["point_level"]["F1"] - ae_r["point_level"]["F1"]
        r["delta_F1_poly_vs_AE"] = r["poly_majority"]["F1"] - ae_r["poly_majority"]["F1"]
    payload = {"version": "v3", "seed": sd.SEED, "binary": "apple(1) vs non-apple(0+2)",
               "n_train": int(len(train)), "n_holdout": int(len(hold)), "gee_errors": gee_errors,
               "terrain_cache": str(CACHE_V3), "results": results}
    sd.save_json(payload, V3 / "metrics_shortcut_diagnostics_v3.json")
    sd.save_json(payload, FIG3 / "metrics_shortcut_diagnostics_v3.json")
    sd.plot_shortcut_bars(results, ae_r["point_level"]["F1"], ae_r["poly_majority"]["F1"],
                          FIG3 / "fig_shortcut_diagnostics_v3.png")
    # retitle to v3
    import matplotlib.image as mpimg  # noqa: F401
    print("DONE")


if __name__ == "__main__":
    main()
