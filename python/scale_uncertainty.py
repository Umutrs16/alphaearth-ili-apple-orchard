# -*- coding: utf-8 -*-
"""面积尺度估算与不确定性（像素计数 + Bootstrap）。

Demo：基于样本概率与 AOI 面积做示意性面积推估；正式生产应以 GEE 分类图为准。
"""
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
    area_relative_error,
    ensure_dirs,
    load_config,
    save_json,
    setup_logging,
)
from train_models import load_samples

log = setup_logging("scale_uncertainty")


def aoi_area_ha(cfg: dict) -> float:
    """粗略球面矩形面积（公顷）。"""
    aoi = cfg["aoi"]
    lat_c = 0.5 * (aoi["lat_min"] + aoi["lat_max"])
    # 1 deg lat ≈ 111.32 km; lon 随 lat 缩放
    dlat = (aoi["lat_max"] - aoi["lat_min"]) * 111.32
    dlon = (aoi["lon_max"] - aoi["lon_min"]) * 111.32 * np.cos(np.radians(lat_c))
    return float(dlat * dlon * 100)  # km^2 → ha


def main(mode: str = "demo", cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    df = load_samples(cfg, mode)
    mid = "M4"  # 主融合模型
    bundle = joblib.load(ROOT / cfg["outputs"]["models"] / f"{mid}.joblib")
    model, cols = bundle["model"], bundle["columns"]

    aoi_ha = aoi_area_ha(cfg)
    # 参考面积：用训练年正样本比例 × AOI（demo 示意）
    train_df = df[df["year"].isin(cfg["years"]["train"])]
    ref_frac = float((train_df["label"] == 1).mean())
    ref_ha = aoi_ha * ref_frac * 0.15  # 果园仅占 AOI 一小部分，demo 缩放

    rng = np.random.default_rng(cfg["training"]["random_seed"])
    rows = []
    area_stats = {}
    for year in cfg["years"]["all"]:
        sub = df[df["year"] == year].copy()
        if len(sub) < 10:
            log.warning("skip year %s: insufficient samples (%d)", year, len(sub))
            continue
        for c in cols:
            if c not in sub.columns:
                sub[c] = np.nan
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
            med = float(sub[c].median()) if sub[c].notna().any() else 0.0
            sub[c] = sub[c].fillna(med)
        X = sub[cols].values.astype(np.float32)
        y = sub["label"].values.astype(int)
        prob = model.predict_proba(X)[:, 1]
        # 用样本先验估计面积：平均概率 × 参考尺度
        mean_p = float(prob.mean())
        # 校准：正类样本占比加权
        pred_frac = float((prob >= 0.5).mean())
        pred_ha = ref_ha * (pred_frac / (ref_frac + 1e-12))
        # bootstrap 面积
        boots = []
        n = len(prob)
        for _ in range(cfg["training"]["n_bootstrap"]):
            idx = rng.integers(0, n, n)
            pf = float((prob[idx] >= 0.5).mean())
            boots.append(ref_ha * (pf / (ref_frac + 1e-12)))
        boots = np.array(boots)
        re = area_relative_error(pred_ha, ref_ha)
        rec = {
            "year": year,
            "pred_ha": round(pred_ha, 2),
            "ref_ha_demo": round(ref_ha, 2),
            "RE": round(re, 4),
            "ha_ci_lo": round(float(np.quantile(boots, 0.025)), 2),
            "ha_ci_hi": round(float(np.quantile(boots, 0.975)), 2),
            "mean_prob": round(mean_p, 4),
            "pred_frac": round(pred_frac, 4),
        }
        area_stats[str(year)] = rec
        rows.append(rec)
        log.info("%d: pred=%.1f ha RE=%.3f CI=[%.1f, %.1f]", year, pred_ha, re, rec["ha_ci_lo"], rec["ha_ci_hi"])

    out = {
        "mode": mode,
        "model": mid,
        "aoi_ha_approx": round(aoi_ha, 2),
        "ref_ha_demo": round(ref_ha, 2),
        "note": cfg["demo"]["label"] if mode == "demo" else (
            "REAL mode: scale analysis on sample-level / spatial-block aggregation "
            "(no full Gongliu probability raster export under restricted quota)"
        ),
        "method": "sample_block_aggregation",
        "by_year": area_stats,
    }
    save_json(out, ROOT / cfg["outputs"]["metrics"] / "area_uncertainty.json")
    pd.DataFrame(rows).to_csv(ROOT / cfg["outputs"]["tables"] / "table_area_uncertainty.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="demo")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    main(args.mode, args.config)
