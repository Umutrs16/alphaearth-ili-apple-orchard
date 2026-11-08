# -*- coding: utf-8 -*-
"""生成论文图件（DEMO 模式加水印）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, ensure_dirs, load_config, setup_logging, watermark_axes

log = setup_logging("make_figures")

# 中文字体回退
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _demo(ax, cfg, mode):
    if mode == "demo" and cfg["demo"].get("watermark", True):
        watermark_axes(ax, "[DEMO]")


def fig_study_area(cfg, out_dir, mode):
    fig, ax = plt.subplots(figsize=(7, 6))
    drawn = False
    try:
        import json
        from matplotlib.patches import Polygon as MplPolygon
        full_path = ROOT / "data" / "roi" / "Yili_county_full.geojson"
        g_path = ROOT / "data" / "roi" / "Gongliu.geojson"
        if full_path.exists():
            gj = json.loads(full_path.read_text(encoding="utf-8"))
            for feat in gj.get("features", []):
                name = feat.get("properties", {}).get("name", "")
                geom = feat.get("geometry", {})
                coords_list = []
                if geom.get("type") == "Polygon":
                    coords_list = [geom["coordinates"][0]]
                elif geom.get("type") == "MultiPolygon":
                    coords_list = [p[0] for p in geom["coordinates"]]
                for ring in coords_list:
                    xs, ys = zip(*[(c[0], c[1]) for c in ring])
                    ax.plot(xs, ys, color="steelblue", lw=0.8, alpha=0.8)
                    if name == "巩留县":
                        ax.fill(xs, ys, color="crimson", alpha=0.25)
                drawn = True
            ax.set_title("Study Area: Yili Counties (REAL) — Gongliu highlighted")
            # legend proxy
            ax.plot([], [], color="steelblue", lw=1.5, label="Yili_county asset (11)")
            ax.fill([], [], color="crimson", alpha=0.25, label="Gongliu (巩留县) AOI")
        if g_path.exists() and not drawn:
            pass
    except Exception as e:
        log.warning("real ROI map failed: %s", e)
        drawn = False
    if not drawn:
        aoi = cfg["aoi"]
        ax.add_patch(plt.Rectangle(
            (aoi["lon_min"], aoi["lat_min"]),
            aoi["lon_max"] - aoi["lon_min"],
            aoi["lat_max"] - aoi["lat_min"],
            fill=False, edgecolor="crimson", linewidth=2, label="Gongliu core AOI",
        ))
        ax.add_patch(plt.Rectangle((81.0, 42.8), 3.5, 1.5, fill=False, edgecolor="steelblue",
                                   linestyle="--", linewidth=1.5, label="Ili Valley (expandable)"))
        ax.scatter([82.4], [43.35], c="red", s=60, zorder=5, label="Gongliu")
        ax.set_title("Study Area: Gongliu / Ili Valley Apple Orchards")
    ax.set_xlim(80.5, 85.0)
    ax.set_ylim(42.5, 44.5)
    ax.set_xlabel("Longitude (E)")
    ax.set_ylabel("Latitude (N)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    _demo(ax, cfg, mode)
    # real-mode caption tag
    if mode != "demo":
        ax.text(0.02, 0.02, "REAL / Gongliu", transform=ax.transAxes, fontsize=9,
                color="darkgreen", ha="left", va="bottom")
    fig.tight_layout()
    p = out_dir / "fig01_study_area.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_model_comparison(cfg, out_dir, mode):
    metrics_path = ROOT / cfg["outputs"]["metrics"] / "train_metrics.json"
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    mids = list(metrics.keys())
    f1 = [metrics[m]["F1"] for m in mids]
    iou = [metrics[m]["IoU"] for m in mids]
    oa = [metrics[m]["OA"] for m in mids]
    x = np.arange(len(mids))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, oa, w, label="OA")
    ax.bar(x, f1, w, label="F1")
    ax.bar(x + w, iou, w, label="IoU")
    ax.set_xticks(x)
    ax.set_xticklabels(mids)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("M0–M5 Spatial-Block Holdout (REAL Gongliu, 2019–2021)" if mode!="demo" else "M0–M5 Performance on Spatial-Block Holdout (2019–2021) [DEMO]")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _demo(ax, cfg, mode)
    fig.tight_layout()
    p = out_dir / "fig02_model_comparison.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_transfer(cfg, out_dir, mode):
    df = pd.read_csv(ROOT / cfg["outputs"]["tables"] / "table_transfer_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for mid in df["模型"].unique():
        sub = df[df["模型"] == mid].sort_values("年份")
        ax.plot(sub["年份"], sub["F1"], marker="o", label=mid)
    ax.set_xlabel("Year")
    ax.set_ylabel("F1")
    ax.set_title("Cross-year Transfer F1 (train 2019–2021 → test 2022–2025)")
    ax.legend(ncol=3, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    _demo(ax, cfg, mode)
    fig.tight_layout()
    p = out_dir / "fig03_transfer_f1.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_confusion_roc(cfg, out_dir, mode):
    npz = np.load(ROOT / cfg["outputs"]["metrics"] / "train_holdout.npz")
    mid = "M4"
    yt = npz[f"{mid}_y_true"]
    yp = npz[f"{mid}_y_pred"]
    yprob = npz[f"{mid}_y_prob"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Non-orchard", "Orchard"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"{mid} Confusion Matrix")
    _demo(axes[0], cfg, mode)
    fpr, tpr, _ = roc_curve(yt, yprob)
    axes[1].plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title(f"{mid} ROC")
    axes[1].legend()
    _demo(axes[1], cfg, mode)
    fig.tight_layout()
    p = out_dir / "fig04_confusion_roc_M4.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_area_timeseries(cfg, out_dir, mode):
    df = pd.read_csv(ROOT / cfg["outputs"]["tables"] / "table_area_uncertainty.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["year"], df["pred_ha"], "o-", color="forestgreen", label="Predicted area (ha)")
    ax.fill_between(df["year"], df["ha_ci_lo"], df["ha_ci_hi"], alpha=0.25, color="forestgreen", label="95% CI")
    ax.axhline(df["ref_ha_demo"].iloc[0], color="gray", linestyle="--", label=("Reference (sample-prior)" if mode!="demo" else "Demo reference"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (ha)")
    ax.set_title("Area Estimate (sample/block agg, M4)" + ("" if mode=="demo" else " — REAL Gongliu"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _demo(ax, cfg, mode)
    fig.tight_layout()
    p = out_dir / "fig05_area_uncertainty.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_feature_importance(cfg, out_dir, mode):
    import joblib
    import lightgbm as lgb

    mid = "M4"
    bundle = joblib.load(ROOT / cfg["outputs"]["models"] / f"{mid}.joblib")
    model = bundle["model"]
    cols = bundle["columns"]
    # CalibratedClassifierCV → 取第一折 base
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    if hasattr(base, "feature_importances_"):
        imp = base.feature_importances_
    else:
        imp = np.ones(len(cols))
    # 聚合组
    groups = {"S2": 0.0, "S1": 0.0, "DEM": 0.0, "AE": 0.0}
    for c, v in zip(cols, imp):
        if c.startswith("AE_"):
            groups["AE"] += v
        elif c in ("elevation", "slope", "aspect_sin", "aspect_cos"):
            groups["DEM"] += v
        elif c.startswith("VV") or c.startswith("VH") or c in ("RVI", "VV_VH_ratio"):
            groups["S1"] += v
        else:
            groups["S2"] += v
    # top 单特征
    order = np.argsort(imp)[::-1][:15]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(range(15), imp[order][::-1], color="teal")
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels([cols[i] for i in order][::-1], fontsize=8)
    axes[0].set_title("Top-15 Feature Importance (M4)")
    _demo(axes[0], cfg, mode)
    gnames = list(groups.keys())
    axes[1].bar(gnames, [groups[g] for g in gnames], color=["#2ca02c", "#1f77b4", "#8c564b", "#ff7f0e"])
    axes[1].set_title("Grouped Importance")
    _demo(axes[1], cfg, mode)
    fig.tight_layout()
    p = out_dir / "fig06_feature_importance.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_ablation_radar(cfg, out_dir, mode):
    with open(ROOT / cfg["outputs"]["metrics"] / "train_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    keys = ["OA", "Precision", "Recall", "F1", "IoU", "AUC"]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for mid in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        vals = [metrics[mid].get(k, 0) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, label=mid)
        ax.fill(angles, vals, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1)
    ax.set_title("Ablation Radar (Holdout)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    # polar 水印
    if mode == "demo":
        ax.text(0.5, 0.5, "[DEMO]", transform=ax.transAxes, fontsize=28, color="red",
                alpha=0.25, ha="center", va="center", rotation=25)
    fig.tight_layout()
    p = out_dir / "fig07_ablation_radar.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_pipeline(cfg, out_dir, mode):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    steps = ["ROI", "AOMC\npurify", "S1/S2/AE\n/DEM", "M0–M5\ntrain", "Cross-year\ntransfer", "Scale &\nuncertainty", "Figures\n& Paper"]
    for i, s in enumerate(steps):
        ax.add_patch(plt.Rectangle((i * 1.35, 0.3), 1.15, 0.5, fill=True, color="#dceaf7", ec="#1f4e79", lw=1.5))
        ax.text(i * 1.35 + 0.575, 0.55, s, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=((i + 1) * 1.35, 0.55), xytext=(i * 1.35 + 1.15, 0.55),
                        arrowprops=dict(arrowstyle="->", color="#1f4e79"))
    ax.set_xlim(-0.2, len(steps) * 1.35)
    ax.set_ylim(0, 1.2)
    ax.set_title("Technical Pipeline", pad=10)
    _demo(ax, cfg, mode)
    fig.tight_layout()
    p = out_dir / "fig08_pipeline.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_phenology_schematic(cfg, out_dir, mode):
    doy = np.arange(1, 366)
    # 示意苹果园 vs 作物 NDVI
    orchard = 0.25 + 0.45 * np.exp(-0.5 * ((doy - 180) / 55) ** 2)
    crop = 0.2 + 0.5 * np.exp(-0.5 * ((doy - 160) / 35) ** 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(doy, orchard, label="Apple orchard (schematic)", color="green", lw=2)
    ax.plot(doy, crop, label="Annual crop (schematic)", color="orange", lw=2, linestyle="--")
    ax.axvline(105, color="gray", ls=":", label="green-up")
    ax.axvline(280, color="gray", ls="--", label="senescence")
    ax.set_xlabel("DOY")
    ax.set_ylabel("NDVI")
    ax.set_title("Optical Phenology Schematic (S2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _demo(ax, cfg, mode)
    fig.tight_layout()
    p = out_dir / "fig09_phenology_schematic.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def main(mode: str = "demo", cfg_path: str | None = None) -> list:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    out_dir = ROOT / cfg["outputs"]["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in (
        fig_study_area,
        fig_model_comparison,
        fig_transfer,
        fig_confusion_roc,
        fig_area_timeseries,
        fig_feature_importance,
        fig_ablation_radar,
        fig_pipeline,
        fig_phenology_schematic,
    ):
        p = fn(cfg, out_dir, mode)
        log.info("写出 %s", p.name)
        paths.append(p)
    # 清单
    with open(out_dir / "figure_list.txt", "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p.name + "\n")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="demo")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    main(args.mode, args.config)
