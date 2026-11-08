# -*- coding: utf-8 -*-
"""Main holdout figure v3 (expanded Gongliu polygon holdout) + Chinese caption."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(HERE))
from common import setup_matplotlib_zh  # noqa: E402


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


f20 = _load("f20", "20_fig_main_holdout_validation.py")
v3m = _load("v3m", "22_b3_poly_holdout_v3.py")

OUT = ROOT / "outputs" / "revision" / "polygon_samples"
V3 = OUT / "v3"
FIG3 = ROOT / "outputs" / "revision" / "figures_v3"
C_APPLE, C_OTHER, C_NON, C_TRAIN = "#D62728", "#FF9F1C", "#1F77B4", "#9E9E9E"
OUTC = {"TP": "#2CA02C", "TN": "#1F77B4", "FP": "#FF7F0E", "FN": "#D62728"}
CLS_ZH = {1: "苹果园", 2: "其他果园", 0: "非果园"}


def outcome(y, p):
    return "TP" if y and p else "TN" if (not y and not p) else "FP" if p else "FN"


def main():
    plt = setup_matplotlib_zh()
    FIG3.mkdir(parents=True, exist_ok=True)
    met = json.load(open(V3 / "metrics_poly_holdout_v3.json", encoding="utf-8"))
    pt = pd.read_csv(V3 / "poly_predictions_v3.csv")
    hold = pd.read_csv(V3 / "predictions_holdout_v3.csv")
    man = pd.read_csv(V3 / "split_manifest_v3.csv")
    geoms = {}
    for r in v3m.read_kml_polys(v3m.NEW_KML):
        m = re.match(r"^\s*(YL-POLY-\d{4})\s*\|", r["raw_name"])
        geoms[m.group(1) if m else r["raw_name"].strip()] = v3m.make_geom(r["coords"])[0]

    fig = plt.figure(figsize=(14, 13))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.25, 0.95, 1.0],
                  hspace=0.55, wspace=0.3)

    # (a) map
    ax = fig.add_subplot(gs[0, 0])
    cty = gpd.read_file(v3m.COUNTY)
    cty.boundary.plot(ax=ax, color="#BBBBBB", lw=0.6, zorder=1)
    cty[cty.name.str.contains("巩留")].boundary.plot(ax=ax, color="#222222", lw=1.4, zorder=2)
    tr = man[man.split == "train"]
    ax.scatter(tr.cen_lon, tr.cen_lat, s=26, c=C_TRAIN, edgecolors="white", lw=0.4, zorder=3,
               label=f"训练面（非巩留，n={len(tr)}）")
    for c, mk, col in [(0, "s", C_NON), (2, "D", C_OTHER), (1, "^", C_APPLE)]:
        s = pt[pt.class_code == c]
        ax.scatter(s.cen_lon, s.cen_lat, s=34 if c != 1 else 46, marker=mk, c=col, edgecolors="k",
                   lw=0.4, zorder=4 + (c == 1), label=f"Holdout {CLS_ZH[c]}（n={len(s)}）")
    ax.set_xlim(81.6, 83.45); ax.set_ylim(42.95, 43.92); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)"); ax.grid(alpha=0.25)
    ax.text(82.45, 43.02, "巩留县", fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    ax.set_title("(a) 空间划分：巩留县完全 holdout（面质心）", loc="left", fontsize=11)
    ax.text(1.0, -0.30, "审图号：GS（2024）0650号", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#555", zorder=9)

    # (b) CM + 3-class table
    ax = fig.add_subplot(gs[0, 1])
    pm = met["poly_majority"]; cm = pm["confusion_matrix"]
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], float)
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.05)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测: 非苹果", "预测: 苹果"]); ax.set_yticklabels(["真: 非苹果\n(0+2)", "真: 苹果"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if v > mat.max() * 0.55 else "black")
    ci = pm["wilson_95ci_OA"]; bf = pm["bootstrap_95ci"]["F1"]
    ax.set_title(f"(b) 面级混淆矩阵（多数投票，n={pm['n']}）\n"
                 f"F1={pm['F1']:.3f} [{bf['lo']:.3f}, {bf['hi']:.3f}]  OA={pm['OA']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]\n"
                 f"P={pm['precision']:.3f}  R={pm['recall']:.3f}  BA={pm['balanced_accuracy']:.3f}  MCC={pm['MCC']:.3f}",
                 loc="left", fontsize=9)
    t3 = pm["three_class_by_pred"]
    cell = [[CLS_ZH[c], t3[str(c)]["pred_apple"], t3[str(c)]["pred_non"],
             f"{(t3[str(c)]['pred_apple'] if c == 1 else t3[str(c)]['pred_non']) / max(1, t3[str(c)]['pred_apple'] + t3[str(c)]['pred_non']):.0%}"]
            for c in (1, 2, 0)]
    tb = ax.table(cellText=cell, colLabels=["真实类别", "→苹果", "→非苹果", "正确率"],
                  cellLoc="center", bbox=[0.0, -0.56, 1.0, 0.34])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5)
    for (r_, c_), cl in tb.get_celld().items():
        if r_ == 0: cl.set_facecolor("#E8EEF5")
        if r_ == 2: cl.set_facecolor("#FFF1DC")

    # (c) PR
    ax = fig.add_subplot(gs[0, 2])
    y, pr = hold.y.values, hold.prob.values
    P, R, _ = precision_recall_curve(y, pr); ap = average_precision_score(y, pr)
    apci = met["point_level"]["cluster_bootstrap_95ci"]["AP"]
    ax.plot(R, P, color=C_APPLE, lw=2.2, label=f"v3 全部 (n={len(y)}) AP={ap:.3f}\n[面聚类 bootstrap {apci[0]:.3f}–{apci[1]:.3f}]")
    s2 = hold[hold.point_source == "reused_v2"]
    P2, R2, _ = precision_recall_curve(s2.y, s2.prob)
    ax.plot(R2, P2, color="gray", lw=1.2, ls=":", label=f"v2 子集 (n={len(s2)}) AP={average_precision_score(s2.y, s2.prob):.3f}")
    ax.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"基线（正例比例 {y.mean():.2f}）")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_xlabel("召回率 Recall"); ax.set_ylabel("精确率 Precision")
    ptm = met["point_level"]
    ax.set_title(f"(c) 点级 PR 曲线（辅助；点在面内空间相关）\nF1={ptm['F1']:.3f}  OA={ptm['OA']:.3f}", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left")

    # (d) cases
    p = pt.dropna(subset=["pred_major"]).copy()
    p["out"] = [outcome(a, b) for a, b in zip(p.y, p.pred_major)]
    p = p[~p.tie]

    def pick(cls, out, prefer_new=True):
        s = p[(p.class_code == cls) & (p.out == out)]
        if prefer_new and (~s.is_v2).any(): s = s[~s.is_v2]
        return s.sort_values("area_ha", ascending=False).iloc[0] if len(s) else None
    cases = [pick(1, "TP"), pick(0, "TN"), pick(0, "FP"), pick(1, "FN"), pick(2, "FP"), pick(2, "TN")]
    cases = [c for c in cases if c is not None]
    gsd = gs[1, :].subgridspec(2, 3, hspace=0.38, wspace=0.18)
    n_ok = 0
    for i, r in enumerate(cases):
        ax = fig.add_subplot(gsd[i // 3, i % 3])
        g = geoms[r.poly_name]
        minx, miny, maxx, maxy = g.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        half = max(0.0035, 1.3 * max(maxx - minx, maxy - miny))
        img = f20.try_s2_thumbnail(cx, cy, half_deg=half)
        col = OUTC[r.out]
        if img is not None:
            n_ok += 1
            ax.imshow(img, extent=[cx - half, cx + half, cy - half, cy + half], origin="upper")
        else:
            ax.set_facecolor("#F4F1EA")
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, color="yellow", lw=2.4); ax.plot(xs, ys, color=col, lw=1.2)
        s = hold[hold.poly_name == r.poly_name]
        ax.scatter(s.lon, s.lat, c=np.where(s.pred == 1, "#FF3B30", "#00C8FF"), s=12, edgecolors="k", lw=0.4, zorder=5)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2.2)
        ax.set_title(f"{r.poly_name} [{r.out}]  真={CLS_ZH[int(r.class_code)]}\n"
                     f"p̄={r.mean_prob:.2f}  预测苹果点 {int(round(r.frac_pred_apple * r.n_pts))}/{int(r.n_pts)}  {r.area_ha:.2f} ha",
                     fontsize=8, color=col)
    bb = gs[1, :].get_position(fig)
    fig.text(bb.x0, bb.y1 + 0.045, "(d) 代表性 holdout 面案例（Sentinel-2 2021 年 6–9 月中值真彩色；黄框=面轮廓；红点=点预测苹果，青点=非苹果）",
             fontsize=10)
    fig.suptitle(f"巩留县多边形空间 holdout 独立验证 v3（B3 AlphaEarth-only 2021，训练集同 v2）\n"
                 f"面级为主：n={pm['n']}（苹果 {pm['n_pos']} / 非苹果 {pm['n_neg']}，其中其他果园 {pm['per_class']['other_orchard_rejected_rate']['n']}），"
                 f"F1={pm['F1']:.3f}，OA={pm['OA']:.3f}，平衡精度={pm['balanced_accuracy']:.3f}", fontsize=12.5, y=0.995)
    for d in (FIG3, V3):
        fig.savefig(d / "fig_main_holdout_validation_v3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved; thumbnails {n_ok}/{len(cases)}; cases {[ (c.poly_name, c.out, int(c.class_code)) for c in cases]}")


if __name__ == "__main__":
    main()
