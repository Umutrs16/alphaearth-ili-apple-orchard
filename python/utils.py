# -*- coding: utf-8 -*-
"""公用工具：配置、路径、日志、空间块划分、指标计算。"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)

ROOT = Path(__file__).resolve().parents[1]


def setup_logging(name: str = "orchard") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    for key in ("figures", "tables", "metrics", "paper", "models"):
        p = ROOT / cfg["outputs"][key]
        p.mkdir(parents=True, exist_ok=True)
    for d in ("samples", "features", "aomc"):
        (ROOT / "data" / d).mkdir(parents=True, exist_ok=True)


def spatial_block_id(lon: np.ndarray, lat: np.ndarray, block_size: float = 0.05) -> np.ndarray:
    """按经纬度网格划分空间块，用于空间块交叉验证 / 防泄漏。"""
    ix = np.floor(lon / block_size).astype(np.int64)
    iy = np.floor(lat / block_size).astype(np.int64)
    return ix * 100000 + iy


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    iou = tp / (tp + fp + fn + 1e-12)
    out = {
        "OA": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "IoU": float(iou),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }
    if y_prob is not None:
        try:
            out["AUC"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["AUC"] = float("nan")
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["OA", "Precision", "Recall", "F1", "IoU", "Kappa", "AUC"]
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        yp_prob = y_prob[idx] if y_prob is not None else None
        m = compute_metrics(yt, yp, yp_prob)
        for k in keys:
            if k in m and not (isinstance(m[k], float) and np.isnan(m[k])):
                store[k].append(m[k])
    ci = {}
    for k, vals in store.items():
        if not vals:
            ci[k] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
            continue
        arr = np.array(vals)
        ci[k] = {
            "mean": float(np.mean(arr)),
            "lo": float(np.quantile(arr, alpha / 2)),
            "hi": float(np.quantile(arr, 1 - alpha / 2)),
        }
    return ci


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def feature_columns(cfg: dict, feature_groups: List[str]) -> List[str]:
    """按特征组返回列名（与 demo 合成样本一致）。"""
    cols: List[str] = []
    if "s2" in feature_groups:
        cols += [
            "NDVI_max", "NDVI_mean", "NDVI_std", "NDVI_p10", "NDVI_p90",
            "EVI_max", "EVI_mean", "NDWI_mean", "NDRE_mean", "NDMI_mean", "LSWI_mean",
            "B4_mean", "B8_mean", "B11_mean", "B12_mean",
            "NDVI_amp", "greenup_doy", "senescence_doy",
        ]
    if "s1" in feature_groups:
        cols += ["VV_mean", "VV_std", "VH_mean", "VH_std", "VV_VH_ratio", "RVI"]
    if "dem" in feature_groups:
        cols += ["elevation", "slope", "aspect_sin", "aspect_cos"]
    if "ae" in feature_groups:
        cols += [f"AE_{i:02d}" for i in range(64)]
    return cols


def watermark_axes(ax, text: str = "[DEMO]") -> None:
    ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes, fontsize=28, color="red",
        alpha=0.25, ha="center", va="center", rotation=25, zorder=100,
    )


def area_relative_error(pred_ha: float, ref_ha: float) -> float:
    return float((pred_ha - ref_ha) / (ref_ha + 1e-12))
