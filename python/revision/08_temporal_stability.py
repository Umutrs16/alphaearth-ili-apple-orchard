# -*- coding: utf-8 -*-
"""Feature temporal stability (same-location / same-label-protocol transfer).

Honest framing: NOT independent temporal orchard mapping unless independent
labels exist for target years. 2022–2023 labels still come from AOMC-style
purification pipeline → report as feature temporal stability / cross-year
generalization under consistent weak-label protocol.
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
)
from utils import compute_metrics


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    all_df = load_feature_years([2019, 2020, 2021, 2022, 2023])
    years = sorted(all_df["year"].unique())
    rows = []
    heat = {bid: {} for bid in ("B3", "B6", "B0")}

    for bid in ("B0", "B3", "B6"):
        groups = FEATURE_SETS[bid]
        for ty in years:
            train = all_df[all_df["year"] == ty]
            if train["label"].nunique() < 2 or len(train) < 50:
                continue
            Xtr, _ = matrix_X(train, groups, cfg)
            ytr = train["label"].values.astype(int)
            clf = get_estimator("lgbm", seed)
            clf.fit(Xtr, ytr)
            heat[bid][str(ty)] = {}
            for ey in years:
                test = all_df[all_df["year"] == ey]
                Xte, _ = matrix_X(test, groups, cfg)
                yte = test["label"].values.astype(int)
                pred = clf.predict(Xte)
                prob = clf.predict_proba(Xte)[:, 1]
                m = compute_metrics(yte, pred, prob)
                rec = {
                    "model": bid,
                    "train_year": int(ty),
                    "eval_year": int(ey),
                    "n_train": len(train),
                    "n_eval": len(test),
                    "design": "feature_temporal_stability_weaklabel_protocol",
                    **m,
                }
                rows.append(rec)
                heat[bid][str(ty)][str(ey)] = m["F1"]
                if ty != ey:
                    print(f"{bid} {ty}->{ey}: F1={m['F1']:.4f}")

    # also train on 2019-2021 -> eval 2022/2023 (multi-year train)
    for bid in ("B0", "B3", "B6"):
        train = all_df[all_df["year"].isin([2019, 2020, 2021])]
        Xtr, _ = matrix_X(train, FEATURE_SETS[bid], cfg)
        clf = get_estimator("lgbm", seed)
        clf.fit(Xtr, train["label"].values.astype(int))
        for ey in (2022, 2023):
            test = all_df[all_df["year"] == ey]
            if len(test) == 0:
                continue
            Xte, _ = matrix_X(test, FEATURE_SETS[bid], cfg)
            m = compute_metrics(test["label"].values.astype(int), clf.predict(Xte), clf.predict_proba(Xte)[:, 1])
            rows.append({
                "model": bid,
                "train_year": "2019-2021",
                "eval_year": ey,
                "n_train": len(train),
                "n_eval": len(test),
                "design": "feature_temporal_stability_weaklabel_protocol",
                **m,
            })
            print(f"{bid} 2019-2021->{ey}: F1={m['F1']:.4f}")

    pd.DataFrame(rows).to_csv(TABLES / "temporal_stability.csv", index=False)
    save_json({"heatmap_F1": heat, "note": "Not independent temporal orchard mapping"}, METRICS / "temporal_stability.json")


if __name__ == "__main__":
    main()
