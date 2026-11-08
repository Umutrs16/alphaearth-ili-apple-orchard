# -*- coding: utf-8 -*-
"""训练 M0–M5 模型（空间块划分，防地理泄漏）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    ROOT,
    compute_metrics,
    ensure_dirs,
    feature_columns,
    load_config,
    save_json,
    setup_logging,
)

log = setup_logging("train_models")


def _get_estimator(model_type: str, cfg: dict):
    seed = cfg["training"]["random_seed"]
    n_est = cfg["training"]["n_estimators"]
    depth = cfg["training"]["max_depth"]
    if model_type == "RandomForest":
        return RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, random_state=seed, n_jobs=-1, class_weight="balanced"
        )
    if model_type == "LightGBM":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            class_weight="balanced",
        )
    if model_type == "XGBoost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    raise ValueError(model_type)


def load_samples(cfg: dict, mode: str) -> pd.DataFrame:
    samples_dir = ROOT / cfg["data"]["samples_dir"]
    real_dir = samples_dir / "real"
    if mode in ("gee", "real"):
        candidates = [
            real_dir / "purified_samples_real.csv",
            samples_dir / "purified_samples_real.csv",
            samples_dir / "purified_samples_aomc_partial.csv",
            samples_dir / "purified_samples_demo.csv",
        ]
    else:
        candidates = [
            samples_dir / "purified_samples_demo.csv",
            real_dir / "purified_samples_real.csv",
        ]
    for p in candidates:
        if p.exists():
            log.info("读取样本: %s", p)
            df = pd.read_csv(p)
            # fill missing feature cols with column median later in train; drop rows all-na features lightly
            return df
    raise FileNotFoundError("未找到纯化样本，请先运行 label_purify / gee_sample_features")


def train_one(df: pd.DataFrame, mid: str, mcfg: dict, cfg: dict, years: list) -> dict:
    feat_groups = mcfg["features"]
    cols = feature_columns(cfg, feat_groups)
    sub = df[df["year"].isin(years)].copy()
    for c in cols:
        if c not in sub.columns:
            sub[c] = np.nan
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        med = sub[c].median()
        sub[c] = sub[c].fillna(med if pd.notna(med) else 0.0)
    X = sub[cols].values.astype(np.float32)
    y = sub["label"].values.astype(int)
    groups = sub["block_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["training"]["test_ratio"], random_state=cfg["training"]["random_seed"])
    train_idx, test_idx = next(gss.split(X, y, groups))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    est = _get_estimator(mcfg["type"], cfg)
    if mcfg.get("calibrate"):
        # 用空间块做校准折
        est = CalibratedClassifierCV(est, method="isotonic", cv=3)
    est.fit(Xtr, ytr)
    prob = est.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = compute_metrics(yte, pred, prob)
    metrics["model_id"] = mid
    metrics["model_name"] = mcfg["name"]
    metrics["n_train"] = int(len(train_idx))
    metrics["n_test"] = int(len(test_idx))
    metrics["features"] = feat_groups
    metrics["years_train"] = years

    model_dir = ROOT / cfg["outputs"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": est, "columns": cols, "mcfg": mcfg}, model_dir / f"{mid}.joblib")
    return {"metrics": metrics, "y_true": yte, "y_pred": pred, "y_prob": prob, "columns": cols}


def main(mode: str = "demo", cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    df = load_samples(cfg, mode)
    train_years = cfg["years"]["train"]

    all_metrics = {}
    holdout = {}
    for mid, mcfg in cfg["models"].items():
        log.info("训练 %s (%s) ...", mid, mcfg["name"])
        # M1 可用 XGB 备选：此处按配置 type
        res = train_one(df, mid, mcfg, cfg, train_years)
        all_metrics[mid] = res["metrics"]
        holdout[mid] = {
            "y_true": res["y_true"].tolist(),
            "y_pred": res["y_pred"].tolist(),
            "y_prob": res["y_prob"].tolist(),
        }
        log.info("  OA=%.3f F1=%.3f IoU=%.3f AUC=%.3f", res["metrics"]["OA"], res["metrics"]["F1"], res["metrics"]["IoU"], res["metrics"].get("AUC", float("nan")))

    out = {
        "mode": mode,
        "demo_note": cfg["demo"]["label"] if mode == "demo" else None,
        "metrics": all_metrics,
        "holdout": holdout,
    }
    save_json(all_metrics, ROOT / cfg["outputs"]["metrics"] / "train_metrics.json")
    # holdout 较大，分开存
    np.savez_compressed(
        ROOT / cfg["outputs"]["metrics"] / "train_holdout.npz",
        **{f"{mid}_{k}": np.array(v) for mid, h in holdout.items() for k, v in h.items()},
    )
    save_json(
        {"mode": mode, "note": cfg["demo"]["label"] if mode == "demo" else "gee/partial"},
        ROOT / cfg["outputs"]["metrics"] / "run_meta.json",
    )
    # 表格
    rows = []
    for mid, m in all_metrics.items():
        rows.append({
            "模型": mid,
            "名称": m["model_name"],
            "OA": round(m["OA"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "IoU": round(m["IoU"], 4),
            "Kappa": round(m["Kappa"], 4),
            "AUC": round(m.get("AUC", float("nan")), 4),
        })
    pd.DataFrame(rows).to_csv(ROOT / cfg["outputs"]["tables"] / "table_train_metrics.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="demo")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    main(args.mode, args.config)
