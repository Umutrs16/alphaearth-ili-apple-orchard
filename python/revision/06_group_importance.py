# -*- coding: utf-8 -*-
"""Group permutation importance (AE/S2/S1/DEM) + optional SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    FEATURE_SETS,
    GROUP_COLS,
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
from utils import compute_metrics
import joblib


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    df = load_feature_years([2019, 2020, 2021])
    split = np.load(METRICS / "holdout_split.npz")
    train_df, test_df = df.iloc[split["tr_idx"]], df.iloc[split["te_idx"]]

    # use B6 full model
    groups = FEATURE_SETS["B6"]
    Xtr, cols = matrix_X(train_df, groups, cfg)
    Xte, _ = matrix_X(test_df, groups, cfg)
    ytr = train_df["label"].values.astype(int)
    yte = test_df["label"].values.astype(int)
    clf = get_estimator("lgbm", seed)
    clf.fit(Xtr, ytr)
    base_pred = clf.predict(Xte)
    base_f1 = compute_metrics(yte, base_pred)["F1"]

    rng = np.random.default_rng(seed)
    rows = []
    for gname, gcols in GROUP_COLS.items():
        idxs = [cols.index(c) for c in gcols if c in cols]
        if not idxs:
            continue
        deltas = []
        for _ in range(20):
            Xp = Xte.copy()
            for j in idxs:
                Xp[:, j] = rng.permutation(Xp[:, j])
            f1 = compute_metrics(yte, clf.predict(Xp))["F1"]
            deltas.append(base_f1 - f1)
        rows.append({
            "group": gname,
            "n_features": len(idxs),
            "base_F1": base_f1,
            "delta_F1_mean": float(np.mean(deltas)),
            "delta_F1_sd": float(np.std(deltas)),
        })
        print(f"perm {gname}: ΔF1={np.mean(deltas):.4f}±{np.std(deltas):.4f}")

    pd.DataFrame(rows).to_csv(TABLES / "group_permutation_importance.csv", index=False)
    save_json({"base_F1": base_f1, "groups": rows}, METRICS / "group_importance.json")

    # optional SHAP (TreeExplainer, subsample)
    shap_ok = False
    try:
        import shap

        explainer = shap.TreeExplainer(clf)
        sample = Xte[: min(200, len(Xte))]
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        shap_rows = []
        for gname, gcols in GROUP_COLS.items():
            idxs = [cols.index(c) for c in gcols if c in cols]
            if idxs:
                shap_rows.append({"group": gname, "mean_abs_shap": float(mean_abs[idxs].sum())})
        pd.DataFrame(shap_rows).to_csv(TABLES / "shap_group.csv", index=False)
        save_json({"shap_group": shap_rows}, METRICS / "shap_group.json")
        shap_ok = True
        print("SHAP group summary written")
    except Exception as e:
        save_json({"shap_error": str(e)}, METRICS / "shap_group.json")
        print("SHAP skipped:", e)

    joblib.dump({"model": clf, "cols": cols}, MODELS / "B6_for_importance.joblib")
    print("group importance done; shap_ok=", shap_ok)


if __name__ == "__main__":
    main()
