# -*- coding: utf-8 -*-
"""5-fold spatial block CV at ~2/5/10 km for B3 and B6."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import (
    BLOCK_SIZES_DEG,
    FEATURE_SETS,
    METRICS,
    TABLES,
    ensure_out_dirs,
    get_estimator,
    load_config,
    load_feature_years,
    matrix_X,
    save_json,
)
from utils import compute_metrics, spatial_block_id


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    df0 = load_feature_years([2019, 2020, 2021])
    rows, results = [], {}

    for bname, bsize in BLOCK_SIZES_DEG.items():
        df = df0.copy()
        df["cv_block"] = spatial_block_id(df["lon"].values, df["lat"].values, bsize)
        n_blocks = df["cv_block"].nunique()
        for bid in ("B3", "B6"):
            groups = FEATURE_SETS[bid]
            X, _ = matrix_X(df, groups, cfg)
            y = df["label"].values.astype(int)
            gkf = GroupKFold(n_splits=5)
            fold_f1 = []
            for fold, (tr, te) in enumerate(gkf.split(X, y, df["cv_block"].values)):
                clf = get_estimator("lgbm", seed)
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                prob = clf.predict_proba(X[te])[:, 1]
                m = compute_metrics(y[te], pred, prob)
                fold_f1.append(m["F1"])
                rows.append({
                    "block_size": bname,
                    "block_deg": bsize,
                    "n_blocks": n_blocks,
                    "model": bid,
                    "fold": fold,
                    **m,
                })
            results[f"{bid}_{bname}"] = {
                "F1_mean": float(np.mean(fold_f1)),
                "F1_sd": float(np.std(fold_f1)),
                "folds": fold_f1,
                "n_blocks": int(n_blocks),
            }
            print(f"{bid} {bname}: F1={np.mean(fold_f1):.4f}±{np.std(fold_f1):.4f} (blocks={n_blocks})")

    pd.DataFrame(rows).to_csv(TABLES / "spatial_cv.csv", index=False)
    save_json(results, METRICS / "spatial_cv.json")


if __name__ == "__main__":
    main()
