# -*- coding: utf-8 -*-
"""跨年度迁移评估：2019–2021 训练 → 2022–2025 测试。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    ROOT,
    bootstrap_ci,
    compute_metrics,
    ensure_dirs,
    feature_columns,
    load_config,
    save_json,
    setup_logging,
)
from train_models import load_samples

log = setup_logging("transfer_eval")


def main(mode: str = "demo", cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    df = load_samples(cfg, mode)
    transfer_years = cfg["years"]["transfer"]
    model_dir = ROOT / cfg["outputs"]["models"]

    results = {}
    ci_all = {}
    rows = []
    for mid in cfg["models"]:
        bundle = joblib.load(model_dir / f"{mid}.joblib")
        model = bundle["model"]
        cols = bundle["columns"]
        year_metrics = {}
        for year in transfer_years:
            sub = df[df["year"] == year]
            if len(sub) < 20:
                continue
            for c in cols:
                if c not in sub.columns:
                    sub[c] = np.nan
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
                med = sub[c].median()
                sub[c] = sub[c].fillna(med if pd.notna(med) else 0.0)
            X = sub[cols].values.astype(np.float32)
            y = sub["label"].values.astype(int)
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = compute_metrics(y, pred, prob)
            m["year"] = year
            year_metrics[str(year)] = m
            ci = bootstrap_ci(y, pred, prob, n_boot=cfg["training"]["n_bootstrap"], seed=cfg["training"]["random_seed"])
            ci_all[f"{mid}_{year}"] = ci
            rows.append({
                "模型": mid,
                "年份": year,
                "OA": round(m["OA"], 4),
                "F1": round(m["F1"], 4),
                "IoU": round(m["IoU"], 4),
                "Kappa": round(m["Kappa"], 4),
                "AUC": round(m.get("AUC", float("nan")), 4),
                "F1_CI_lo": round(ci["F1"]["lo"], 4),
                "F1_CI_hi": round(ci["F1"]["hi"], 4),
            })
            log.info("%s @ %d: OA=%.3f F1=%.3f IoU=%.3f", mid, year, m["OA"], m["F1"], m["IoU"])
        results[mid] = year_metrics

    save_json(results, ROOT / cfg["outputs"]["metrics"] / "transfer_metrics.json")
    save_json(ci_all, ROOT / cfg["outputs"]["metrics"] / "transfer_bootstrap_ci.json")
    pd.DataFrame(rows).to_csv(ROOT / cfg["outputs"]["tables"] / "table_transfer_metrics.csv", index=False, encoding="utf-8-sig")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="demo")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    main(args.mode, args.config)
