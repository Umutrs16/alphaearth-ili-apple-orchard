# -*- coding: utf-8 -*-
"""Label-efficiency curves with spatial-block-aware subsampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SETS = {"S2": "B0", "S1+S2": "B1", "AE": "B3", "AE+S1+S2": "B5"}


def subsample_by_blocks(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    blocks = df["block_id"].unique()
    n_keep = max(1, int(round(len(blocks) * frac)))
    # stratified-ish: keep blocks with positives preferentially when rare
    keep = rng.choice(blocks, size=n_keep, replace=False)
    sub = df[df["block_id"].isin(keep)].copy()
    # if still too few positives, add random positive rows from leftover
    if (sub["label"] == 1).sum() < 5:
        pos = df[df["label"] == 1]
        extra = pos.sample(n=min(20, len(pos)), random_state=seed)
        sub = pd.concat([sub, extra]).drop_duplicates()
    return sub


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    df = load_feature_years([2019, 2020, 2021])
    # fixed spatial holdout test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(df, df["label"], df["block_id"]))
    train_pool, test_df = df.iloc[tr].copy(), df.iloc[te].copy()

    rows = []
    curve = {}
    for label, bid in SETS.items():
        groups = FEATURE_SETS[bid]
        Xte, _ = matrix_X(test_df, groups, cfg)
        yte = test_df["label"].values.astype(int)
        curve[label] = []
        for frac in FRACTIONS:
            # average over 5 seeds for stability at low frac
            f1s = []
            for s in range(5):
                sub = subsample_by_blocks(train_pool, frac, seed + s * 17)
                if len(sub) < 10 or sub["label"].nunique() < 2:
                    continue
                Xtr, _ = matrix_X(sub, groups, cfg)
                ytr = sub["label"].values.astype(int)
                clf = get_estimator("lgbm", seed + s)
                clf.fit(Xtr, ytr)
                pred = clf.predict(Xte)
                m = compute_metrics(yte, pred)
                f1s.append(m["F1"])
            mean_f1 = float(np.mean(f1s)) if f1s else float("nan")
            sd_f1 = float(np.std(f1s)) if f1s else float("nan")
            rec = {
                "feature_set": label,
                "bid": bid,
                "frac": frac,
                "F1_mean": mean_f1,
                "F1_sd": sd_f1,
                "n_runs": len(f1s),
            }
            rows.append(rec)
            curve[label].append(rec)
            print(f"{label} frac={frac:.0%} F1={mean_f1:.4f}±{sd_f1:.4f}")

    pd.DataFrame(rows).to_csv(TABLES / "label_efficiency.csv", index=False)
    save_json(curve, METRICS / "label_efficiency.json")


if __name__ == "__main__":
    main()
