# -*- coding: utf-8 -*-
"""RF vs XGBoost vs LightGBM on B3 (AE) and B6 (full fusion)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    FEATURE_SETS,
    METRICS,
    TABLES,
    block_bootstrap_ci,
    ensure_out_dirs,
    get_estimator,
    load_config,
    load_feature_years,
    matrix_X,
    save_json,
)
from utils import compute_metrics


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    df = load_feature_years([2019, 2020, 2021])
    split = np.load(METRICS / "holdout_split.npz")
    train_df, test_df = df.iloc[split["tr_idx"]], df.iloc[split["te_idx"]]

    rows, results = [], {}
    for bid in ("B3", "B6"):
        groups = FEATURE_SETS[bid]
        Xtr, _ = matrix_X(train_df, groups, cfg)
        Xte, _ = matrix_X(test_df, groups, cfg)
        ytr = train_df["label"].values.astype(int)
        yte = test_df["label"].values.astype(int)
        for cname in ("rf", "xgboost", "lightgbm"):
            clf = get_estimator(cname, seed)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = compute_metrics(yte, pred, prob)
            ci = block_bootstrap_ci(yte, pred, test_df["block_id"].values, prob, n_boot=400, seed=seed)
            key = f"{bid}_{cname}"
            rec = {"id": bid, "classifier": cname, **m, "F1_ci_lo": ci["F1"]["lo"], "F1_ci_hi": ci["F1"]["hi"]}
            results[key] = rec
            rows.append(rec)
            print(f"{key}: F1={m['F1']:.4f}")

    pd.DataFrame(rows).to_csv(TABLES / "classifier_compare.csv", index=False)
    save_json(results, METRICS / "classifier_compare.json")


if __name__ == "__main__":
    main()
