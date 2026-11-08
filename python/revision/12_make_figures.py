# -*- coding: utf-8 -*-
"""Publication-quality figures (Chinese labels, no DEMO watermark)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import FIGURES, METRICS, TABLES, ensure_out_dirs, load_feature_years, setup_matplotlib_zh
from utils import ROOT


def fig1_locator(plt):
    import geopandas as gpd
    from shapely.geometry import box

    fig = plt.figure(figsize=(10, 8))
    # nested axes manually
    ax0 = fig.add_axes([0.08, 0.55, 0.35, 0.4])  # China rough
    ax1 = fig.add_axes([0.48, 0.55, 0.45, 0.4])  # Xinjiang/Ili
    ax2 = fig.add_axes([0.08, 0.08, 0.55, 0.4])  # Gongliu
    ax3 = fig.add_axes([0.68, 0.08, 0.28, 0.4])  # samples

    # China box schematic
    ax0.set_xlim(73, 135)
    ax0.set_ylim(18, 54)
    ax0.add_patch(plt.Rectangle((73, 18), 62, 36, fill=False, lw=1.5))
    ax0.add_patch(plt.Rectangle((73, 35), 25, 15, fill=True, alpha=0.25, color="C1"))
    ax0.text(100, 36, "中国", ha="center", fontsize=14)
    ax0.set_title("(a) 中国")
    ax0.set_xlabel("经度"); ax0.set_ylabel("纬度")

    # Xinjiang / Ili
    ax1.set_xlim(73, 96)
    ax1.set_ylim(34, 50)
    ax1.add_patch(plt.Rectangle((80, 42), 5, 3, fill=True, alpha=0.35, color="C2"))
    ax1.text(82.5, 43.5, "伊犁", ha="center")
    ax1.set_title("(b) 新疆 / 伊犁河谷")

    # Gongliu ROI + samples
    try:
        gdf = gpd.read_file(ROOT / "data/roi/Gongliu.geojson")
        gdf.boundary.plot(ax=ax2, color="k", lw=1)
        ax2.set_title("(c) 巩留县研究区")
    except Exception:
        ax2.set_title("(c) 巩留县")
        ax2.set_xlim(81.5, 83.5); ax2.set_ylim(42.9, 43.7)
    try:
        df = load_feature_years([2021])
        ax2.scatter(df.loc[df.label == 0, "lon"], df.loc[df.label == 0, "lat"], s=4, c="#4C78A8", alpha=0.5, label="非苹果")
        ax2.scatter(df.loc[df.label == 1, "lon"], df.loc[df.label == 1, "lat"], s=6, c="#F58518", alpha=0.7, label="苹果园")
        ax2.legend(markerscale=2, fontsize=8)
    except Exception:
        pass
    ax2.set_xlabel("经度"); ax2.set_ylabel("纬度")

    # independent val if exists
    iv = ROOT / "data/independent_val/independent_val_points.csv"
    ax3.set_title("(d) 独立验证样本")
    if iv.exists():
        v = pd.read_csv(iv)
        if "label" in v.columns or "label_apple" in v.columns:
            lab = v["label"] if "label" in v.columns else v["label_apple"]
            ax3.scatter(v.loc[lab == 0, "lon"], v.loc[lab == 0, "lat"], s=8, c="#4C78A8", label="非苹果")
            ax3.scatter(v.loc[lab == 1, "lon"], v.loc[lab == 1, "lat"], s=10, c="#F58518", label="苹果园")
            ax3.legend(fontsize=8)
            ax3.set_xlabel("经度"); ax3.set_ylabel("纬度")
    else:
        ax3.text(0.5, 0.5, "独立验证集生成中", ha="center", transform=ax3.transAxes)
    fig.savefig(FIGURES / "fig01_study_area.png", bbox_inches="tight")
    plt.close(fig)


def fig2_framework(plt):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.3, 1.5, 2.0, 2.2, "① 数据层\nAlphaEarth\nSentinel-1/2\nDEM / AOMC"),
        (2.8, 1.5, 2.0, 2.2, "② 弱标签纯化\n形态学核心\n硬负样本缓冲\n空间块划分"),
        (5.3, 1.5, 2.0, 2.2, "③ 特征与模型\nB0–B6消融\n分类器对比\n校准/重要性"),
        (7.8, 1.5, 2.0, 2.2, "④ 验证与产品\n空间CV\n独立验证\n跨年概率图"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#E8F1FB", edgecolor="#2C5F8A", lw=2, zorder=2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=11, zorder=3)
    for x in (2.3, 4.8, 7.3):
        ax.annotate("", xy=(x + 0.45, 2.6), xytext=(x, 2.6),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#2C5F8A"))
    ax.set_title("技术流程框架：数据 → 弱标签 → 特征/模型 → 验证/产品", fontsize=13, pad=10)
    fig.savefig(FIGURES / "fig02_framework.png", bbox_inches="tight")
    plt.close(fig)


def fig3_phenology(plt):
    df = load_feature_years([2019, 2020, 2021])
    # Use annual S2 phenology stats as honest proxy (no full DOY series)
    feats = ["NDVI_mean", "NDVI_max", "NDVI_amp", "EVI_mean", "NDMI_mean", "LSWI_mean", "greenup_doy", "senescence_doy"]
    # fake multi-class: use label + NDVI/elevation rough bins for cropland/forest/other
    apple = df[df.label == 1]
    non = df[df.label == 0].copy()
    # approximate: high elev+high NDVI -> forest-like; low elev -> cropland-like
    non = non.assign(
        rough=np.where(
            non["elevation"] > non["elevation"].median(),
            np.where(non["NDVI_mean"] > non["NDVI_mean"].median(), "forest_like", "other"),
            "cropland_like",
        )
    )
    groups = {
        "苹果园": apple,
        "耕地样": non[non.rough == "cropland_like"],
        "林地样": non[non.rough == "forest_like"],
        "其他": non[non.rough == "other"],
    }
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    colors = {"苹果园": "#F58518", "耕地样": "#4C78A8", "林地样": "#54A24B", "其他": "#B279A2"}
    for i, f in enumerate(feats):
        ax = axes[i]
        means, sds, names = [], [], []
        for name, g in groups.items():
            if len(g) == 0 or f not in g:
                continue
            means.append(g[f].mean())
            sds.append(g[f].std())
            names.append(name)
        x = np.arange(len(names))
        ax.bar(x, means, yerr=sds, color=[colors[n] for n in names], alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f)
    fig.suptitle("哨兵-2物候/光谱统计特征：苹果园 vs 非苹果（均值±标准差）\n注：非苹果亚类为基于高程-NDVI的粗分，非正式土地覆盖分类", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIGURES / "fig03_phenology.png", bbox_inches="tight")
    plt.close(fig)


def fig4_ablation(plt):
    p = TABLES / "ablation_B0B6.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))
    ax.bar(x, df["F1"], color="#4C78A8", alpha=0.9)
    yerr = np.vstack([df["F1"] - df["F1_ci_lo"], df["F1_ci_hi"] - df["F1"]])
    ax.errorbar(x, df["F1"], yerr=yerr, fmt="none", ecolor="k", capsize=4)
    labels = [
        "B0 S2", "B1 S1+S2", "B2 +DEM", "B3 AE", "B4 AE+S2", "B5 AE+S1+S2", "B6 全融合"
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("B0–B6 统一LightGBM消融（块自助法95%CI）")
    best = df.loc[df["F1"].idxmax(), "id"]
    ax.axhline(df["F1"].max(), color="#F58518", ls="--", lw=1, label=f"最优 {best}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_ablation_B0B6.png", bbox_inches="tight")
    plt.close(fig)


def fig5_label_eff(plt):
    p = TABLES / "label_efficiency.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, g in df.groupby("feature_set"):
        ax.errorbar(g["frac"] * 100, g["F1_mean"], yerr=g["F1_sd"], marker="o", label=name, capsize=3)
    ax.set_xlabel("训练空间块比例 (%)")
    ax.set_ylabel("F1（空间留出测试）")
    ax.set_title("标签效率曲线（空间块感知抽样）")
    ax.legend()
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_label_efficiency.png", bbox_inches="tight")
    plt.close(fig)


def fig6_spatial_cv(plt):
    p = TABLES / "spatial_cv.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["2km", "5km", "10km"]
    data, positions, colors = [], [], []
    pos = 0
    xticks, xticklabels = [], []
    for b in order:
        for model, color in (("B3", "#F58518"), ("B6", "#4C78A8")):
            vals = df[(df.block_size == b) & (df.model == model)]["F1"].values
            data.append(vals)
            positions.append(pos)
            colors.append(color)
            pos += 1
        xticks.append(pos - 1.5)
        xticklabels.append(b)
        pos += 0.8
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("F1")
    ax.set_title("空间块交叉验证 F1（5折）")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#F58518", label="B3 AE"), Patch(facecolor="#4C78A8", label="B6 全融合")])
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_spatial_cv.png", bbox_inches="tight")
    plt.close(fig)


def fig7_temporal_heatmap(plt):
    p = TABLES / "temporal_stability.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["train_year"].astype(str).str.fullmatch(r"\d{4}")]
    df["train_year"] = df["train_year"].astype(int)
    df["eval_year"] = df["eval_year"].astype(int)
    years = sorted(set(df.train_year) | set(df.eval_year))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, bid in zip(axes, ("B0", "B3", "B6")):
        sub = df[df.model == bid]
        mat = np.full((len(years), len(years)), np.nan)
        for _, r in sub.iterrows():
            i = years.index(int(r.train_year))
            j = years.index(int(r.eval_year))
            mat[i, j] = r.F1
        im = ax.imshow(mat, vmin=0.7, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(years))); ax.set_yticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45)
        ax.set_yticklabels(years)
        ax.set_xlabel("评估年"); ax.set_ylabel("训练年")
        ax.set_title(bid)
        for i in range(len(years)):
            for j in range(len(years)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    fig.suptitle("跨年特征时间稳定性（弱标签协议一致；非独立时序制图）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.92])
    fig.savefig(FIGURES / "fig07_temporal_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig8_importance(plt):
    p = TABLES / "group_permutation_importance.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(df["group"], df["delta_F1_mean"], xerr=df["delta_F1_sd"], color="#54A24B", alpha=0.85, capsize=4)
    ax.set_xlabel("排列重要性 ΔF1（B6）")
    ax.set_title("特征组排列重要性")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig08_group_importance.png", bbox_inches="tight")
    plt.close(fig)
    # SHAP optional
    sp = TABLES / "shap_group.csv"
    if sp.exists():
        sdf = pd.read_csv(sp)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(sdf["group"], sdf["mean_abs_shap"], color="#B279A2")
        ax.set_xlabel("组内 |SHAP| 之和")
        ax.set_title("SHAP 特征组贡献（B6，测试子集）")
        fig.tight_layout()
        fig.savefig(FIGURES / "fig08b_shap_group.png", bbox_inches="tight")
        plt.close(fig)


def main():
    ensure_out_dirs()
    plt = setup_matplotlib_zh()
    fig1_locator(plt)
    fig2_framework(plt)
    fig3_phenology(plt)
    fig4_ablation(plt)
    fig5_label_eff(plt)
    fig6_spatial_cv(plt)
    fig7_temporal_heatmap(plt)
    fig8_importance(plt)
    # maps panel may already exist from 10_maps_preview
    print("figures written to", FIGURES)
    print(sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
