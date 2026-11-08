# -*- coding: utf-8 -*-
"""Isotonic vs Platt calibration on held-out calib split (not test)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit

from common import (
    FEATURE_SETS,
    FIGURES,
    METRICS,
    TABLES,
    ensure_out_dirs,
    expected_calibration_error,
    get_estimator,
    load_config,
    load_feature_years,
    matrix_X,
    save_json,
    setup_matplotlib_zh,
)
from utils import compute_metrics


def calibrate_probs(method: str, p_cal, y_cal, p_test):
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_cal, y_cal)
        return ir.predict(p_test)
    if method == "sigmoid":
        # Platt scaling on 1D probs
        lr = LogisticRegression(max_iter=1000)
        lr.fit(p_cal.reshape(-1, 1), y_cal)
        return lr.predict_proba(p_test.reshape(-1, 1))[:, 1]
    raise ValueError(method)


def main():
    ensure_out_dirs()
    cfg = load_config()
    seed = cfg["training"]["random_seed"]
    df = load_feature_years([2019, 2020, 2021])
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    tr, rest = next(gss1.split(df, df["label"], df["block_id"]))
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    ca, te = next(gss2.split(rest_df, rest_df["label"], rest_df["block_id"]))
    train_df = df.iloc[tr]
    calib_df = rest_df.iloc[ca]
    test_df = rest_df.iloc[te]

    rows, results = [], {}
    plt = setup_matplotlib_zh()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax_i, bid in enumerate(("B3", "B6")):
        groups = FEATURE_SETS[bid]
        Xtr, _ = matrix_X(train_df, groups, cfg)
        Xca, _ = matrix_X(calib_df, groups, cfg)
        Xte, _ = matrix_X(test_df, groups, cfg)
        ytr = train_df["label"].values.astype(int)
        yca = calib_df["label"].values.astype(int)
        yte = test_df["label"].values.astype(int)

        base = get_estimator("lgbm", seed)
        base.fit(Xtr, ytr)
        raw_te = base.predict_proba(Xte)[:, 1]
        raw_ca = base.predict_proba(Xca)[:, 1]

        variants = {"raw": raw_te}
        for method in ("isotonic", "sigmoid"):
            variants[method] = calibrate_probs(method, raw_ca, yca, raw_te)

        for name, prob in variants.items():
            pred = (prob >= 0.5).astype(int)
            m = compute_metrics(yte, pred, prob)
            rec = {
                "model": bid,
                "calib": name,
                "Brier": float(brier_score_loss(yte, prob)),
                "LogLoss": float(log_loss(yte, np.clip(prob, 1e-6, 1 - 1e-6))),
                "ECE": expected_calibration_error(yte, prob),
                **{k: m[k] for k in ("OA", "F1", "AUC")},
                "n_train": len(train_df),
                "n_calib": len(calib_df),
                "n_test": len(test_df),
            }
            rows.append(rec)
            results[f"{bid}_{name}"] = rec
            print(f"{bid} {name}: Brier={rec['Brier']:.4f} ECE={rec['ECE']:.4f} F1={rec['F1']:.4f}")

        ax = axes[ax_i]
        for name, prob in variants.items():
            frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=8, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{bid} 可靠性图")
        ax.set_xlabel("预测概率")
        ax.set_ylabel("观测频率")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_reliability_calibration.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TABLES / "calibration.csv", index=False)
    save_json(results, METRICS / "calibration.json")


if __name__ == "__main__":
    main()
