# -*- coding: utf-8 -*-
"""Scale consistency: max 1 sample per 30 m parent cell vs naive 10 m-like dense sampling.

Training features were sampled at point locations; AOMC is 30 m. We approximate
parent cells by rounding lon/lat to ~30 m grids and keep ≤1 sample per cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    FEATURE_SETS,
    METRICS,
    TABLES,
    ensure_out_dirs,
    get_estimator,
    load_config,
    load_feature_years,
    matrix_X,
    save_json,
    spatial_split,
)
from utils import compute_metrics


def parent_cell_id(lon, lat, cell_m: float = 30.0):
    # WGS84 approx metres at 43N
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(43.3))
    ix = np.floor(lon * m_per_deg_lon / cell_m).astype(np.int64)
    iy = np.floor(lat * m_per_deg_lat / cell_m).astype(np.int64)
    return ix * 10_000_000 + iy


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    df = load_feature_years([2019, 2020, 2021]).copy()
    df["cell30"] = parent_cell_id(df["lon"].values, df["lat"].values, 30.0)

    # naive: all points; thinned: 1 per 30m cell (prefer positive)
    df_sorted = df.sort_values(["label", "year"], ascending=[False, True])
    thinned = df_sorted.drop_duplicates("cell30", keep="first")

    rows = []
    for tag, data in (("naive_all_points", df), ("max1_per_30m_cell", thinned)):
        tr, te = spatial_split(data, 0.3, seed)
        train_df, test_df = data.iloc[tr], data.iloc[te]
        for bid in ("B3", "B6"):
            Xtr, _ = matrix_X(train_df, FEATURE_SETS[bid], cfg)
            Xte, _ = matrix_X(test_df, FEATURE_SETS[bid], cfg)
            clf = get_estimator("lgbm", seed)
            clf.fit(Xtr, train_df["label"].values.astype(int))
            pred = clf.predict(Xte)
            m = compute_metrics(test_df["label"].values.astype(int), pred)
            rec = {"scale_mode": tag, "model": bid, "n_samples": len(data), "n_train": len(train_df), "n_test": len(test_df), **m}
            rows.append(rec)
            print(tag, bid, f"n={len(data)} F1={m['F1']:.4f}")

    pd.DataFrame(rows).to_csv(TABLES / "scale_consistency.csv", index=False)
    save_json({
        "note": "Features sampled at point locations; AOMC reference is 30 m. "
                "max1_per_30m_cell keeps ≤1 training/eval sample per ~30 m parent cell "
                "to reduce pseudo-replication vs naive multi-points-per-cell.",
        "rows": rows,
    }, METRICS / "scale_consistency.json")


if __name__ == "__main__":
    main()
