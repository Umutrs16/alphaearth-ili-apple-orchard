# -*- coding: utf-8 -*-
"""Shared helpers for revision experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from utils import compute_metrics, feature_columns, load_config, spatial_block_id  # noqa: E402

OUT = ROOT / "outputs" / "revision"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
TABLES = OUT / "tables"
MAPS = OUT / "maps"
PAPER = OUT / "paper"
IND_VAL = ROOT / "data" / "independent_val"

FEATURE_SETS = {
    "B0": ["s2"],
    "B1": ["s1", "s2"],
    "B2": ["s1", "s2", "dem"],
    "B3": ["ae"],
    "B4": ["ae", "s2"],
    "B5": ["ae", "s1", "s2"],
    "B6": ["ae", "s1", "s2", "dem"],
}

GROUP_COLS = {
    "AE": [f"AE_{i:02d}" for i in range(64)],
    "S2": [
        "NDVI_max", "NDVI_mean", "NDVI_std", "NDVI_p10", "NDVI_p90",
        "EVI_max", "EVI_mean", "NDWI_mean", "NDRE_mean", "NDMI_mean", "LSWI_mean",
        "B4_mean", "B8_mean", "B11_mean", "B12_mean",
        "NDVI_amp", "greenup_doy", "senescence_doy",
    ],
    "S1": ["VV_mean", "VV_std", "VH_mean", "VH_std", "VV_VH_ratio", "RVI"],
    "DEM": ["elevation", "slope", "aspect_sin", "aspect_cos"],
}

# ~2/5/10 km at ~43°N
BLOCK_SIZES_DEG = {
    "2km": 0.02,
    "5km": 0.05,
    "10km": 0.10,
}


def ensure_out_dirs() -> None:
    for p in (OUT, METRICS, FIGURES, MODELS, TABLES, MAPS, PAPER, IND_VAL):
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_feature_years(years: Sequence[int] = (2019, 2020, 2021, 2022, 2023)) -> pd.DataFrame:
    frames = []
    for y in years:
        p = ROOT / "data" / "samples" / "real" / f"features_{y}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["year"] = int(y)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No feature CSVs found")
    df = pd.concat(frames, ignore_index=True)
    if "block_id" not in df.columns:
        df["block_id"] = spatial_block_id(df["lon"].values, df["lat"].values, 0.05)
    return df


def fill_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        med = out[c].median()
        out[c] = out[c].fillna(med if pd.notna(med) else 0.0)
    return out


def make_lgbm(seed: int = 42, n_estimators: int = 200, max_depth: int = 12):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        class_weight="balanced",
    )


def make_rf(seed: int = 42, n_estimators: int = 200, max_depth: int = 12):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )


def make_xgb(seed: int = 42, n_estimators: int = 200, max_depth: int = 12):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
    )


def get_estimator(name: str, seed: int = 42):
    name = name.lower()
    if name in ("lgbm", "lightgbm"):
        return make_lgbm(seed)
    if name in ("rf", "randomforest"):
        return make_rf(seed)
    if name in ("xgb", "xgboost"):
        return make_xgb(seed)
    raise ValueError(name)


def spatial_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
    group_col: str = "block_id",
) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col].values
    tr, te = next(gss.split(df, df["label"], groups))
    return tr, te


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    blocks: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Resample whole spatial blocks (not pixels)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    keys = ["OA", "Precision", "Recall", "F1", "IoU", "Kappa", "AUC"]
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in chosen])
        yt, yp = y_true[idx], y_pred[idx]
        yp_prob = y_prob[idx] if y_prob is not None else None
        # need both classes for AUC
        m = compute_metrics(yt, yp, yp_prob if len(np.unique(yt)) > 1 else None)
        for k in keys:
            if k in m and not (isinstance(m[k], float) and np.isnan(m[k])):
                store[k].append(m[k])
    ci = {}
    for k, vals in store.items():
        if not vals:
            ci[k] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
            continue
        arr = np.asarray(vals, dtype=float)
        ci[k] = {
            "mean": float(np.mean(arr)),
            "lo": float(np.quantile(arr, alpha / 2)),
            "hi": float(np.quantile(arr, 1 - alpha / 2)),
        }
    return ci


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def setup_matplotlib_zh():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 11
    return plt


def matrix_X(df: pd.DataFrame, feat_groups: List[str], cfg=None) -> Tuple[np.ndarray, List[str]]:
    cfg = cfg or load_config()
    cols = feature_columns(cfg, feat_groups)
    df2 = fill_features(df, cols)
    return df2[cols].values.astype(float), cols
