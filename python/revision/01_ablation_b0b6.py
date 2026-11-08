# -*- coding: utf-8 -*-
"""Unified LightGBM ablation B0–B6 with block-bootstrap CIs."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from common import (
    FEATURE_SETS,
    METRICS,
    MODELS,
    TABLES,
    block_bootstrap_ci,
    ensure_out_dirs,
    get_estimator,
    load_config,
    load_feature_years,
    matrix_X,
    save_json,
    spatial_split,
)
from utils import compute_metrics


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    df = load_feature_years([2019, 2020, 2021])
    tr_idx, te_idx = spatial_split(df, test_size=0.3, seed=seed)
    train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]

    rows = []
    results = {}
    for bid, groups in FEATURE_SETS.items():
        Xtr, cols = matrix_X(train_df, groups, cfg)
        Xte, _ = matrix_X(test_df, groups, cfg)
        ytr = train_df["label"].values.astype(int)
        yte = test_df["label"].values.astype(int)
        clf = get_estimator("lgbm", seed)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = compute_metrics(yte, pred, prob)
        ci = block_bootstrap_ci(yte, pred, test_df["block_id"].values, prob, n_boot=500, seed=seed)
        rec = {
            "id": bid,
            "features": groups,
            "n_features": len(cols),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            **m,
            "F1_ci_lo": ci["F1"]["lo"],
            "F1_ci_hi": ci["F1"]["hi"],
            "OA_ci_lo": ci["OA"]["lo"],
            "OA_ci_hi": ci["OA"]["hi"],
            "block_bootstrap": ci,
        }
        results[bid] = rec
        rows.append({k: rec[k] for k in [
            "id", "features", "n_features", "n_train", "n_test",
            "OA", "Precision", "Recall", "F1", "IoU", "Kappa", "AUC",
            "F1_ci_lo", "F1_ci_hi", "OA_ci_lo", "OA_ci_hi",
        ]})
        joblib.dump({"model": clf, "cols": cols, "groups": groups}, MODELS / f"ablation_{bid}.joblib")
        print(f"{bid} F1={m['F1']:.4f} [{ci['F1']['lo']:.4f},{ci['F1']['hi']:.4f}] OA={m['OA']:.4f}")

    pd.DataFrame(rows).to_csv(TABLES / "ablation_B0B6.csv", index=False)
    save_json(results, METRICS / "ablation_B0B6.json")
    # also store split indices for reuse
    np.savez(
        METRICS / "holdout_split.npz",
        tr_idx=tr_idx,
        te_idx=te_idx,
        years=np.array([2019, 2020, 2021]),
    )
    print("Wrote ablation_B0B6")


if __name__ == "__main__":
    main()
