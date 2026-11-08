# -*- coding: utf-8 -*-
"""Main independent-validation figure for Gongliu polygon spatial holdout (v2).

Outputs:
  figures_v2/fig_main_holdout_validation.png
  polygon_samples/metrics_poly_holdout_v2_enriched.json
  figures_v2/fig_main_holdout_caption_zh.md
"""
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch
from shapely.geometry import Polygon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "revision"))
from common import save_json, setup_matplotlib_zh  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "polygon_samples"
FIG = ROOT / "outputs" / "revision" / "figures_v2"
MODELS = OUT / "models"
KML = OUT / "Confirmed_apple_field_points_area.kml"
SAMPLES = OUT / "sample_points_v2.csv"
AE_CSV = OUT / "poly_ae_features_2021_v2.csv"
MANIFEST = OUT / "split_manifest_v2.csv"
B3_MODEL = MODELS / "B3_poly_2021_v2.joblib"
GONGLIU = ROOT / "data" / "roi" / "Gongliu.geojson"
YILI = ROOT / "data" / "roi" / "Yili_county_full.geojson"
PROJECT = "hallowed-chain-458005-g9"
NS = {"kml": "http://www.opengis.net/kml/2.2"}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def ring_to_coords(coord_text: str) -> List[Tuple[float, float]]:
    pts = []
    for tok in coord_text.strip().split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def parse_kml_polys(path: Path) -> Dict[str, Polygon]:
    tree = ET.parse(path)
    root = tree.getroot()

    def local(tag: str) -> str:
        return f"{{http://www.opengis.net/kml/2.2}}{tag}"

    out: Dict[str, Polygon] = {}
    for pm in root.iter(local("Placemark")):
        name_el = pm.find(local("name"))
        name = (name_el.text or "").strip() if name_el is not None else ""
        poly_el = pm.find(".//" + local("Polygon"))
        if poly_el is None:
            continue
        outer = poly_el.find(
            ".//" + local("outerBoundaryIs") + "/" + local("LinearRing") + "/" + local("coordinates")
        )
        if outer is None:
            outer = poly_el.find(".//" + local("LinearRing") + "/" + local("coordinates"))
        if outer is None or not (outer.text or "").strip():
            continue
        coords = ring_to_coords(outer.text)
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        try:
            geom = Polygon(coords)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda g: g.area)
            out[name] = geom
        except Exception:
            continue
    return out


def poly_majority_table(hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in hold.groupby("poly_name"):
        if len(sub) < 3:
            continue
        maj = int(sub["pred"].mode().iloc[0])
        true = int(sub["class_code"].iloc[0])
        mean_p = float(sub["prob"].mean())
        rows.append(
            {
                "poly_name": name,
                "y": true,
                "pred": maj,
                "n_pts": int(len(sub)),
                "mean_prob": mean_p,
                "cen_lon": float(sub["lon"].mean()),
                "cen_lat": float(sub["lat"].mean()),
            }
        )
    return pd.DataFrame(rows)


def outcome_label(y: int, pred: int) -> str:
    if y == 1 and pred == 1:
        return "TP"
    if y == 0 and pred == 0:
        return "TN"
    if y == 0 and pred == 1:
        return "FP"
    return "FN"


def try_s2_thumbnail(lon: float, lat: float, half_deg: float = 0.008) -> Optional[np.ndarray]:
    """Best-effort S2 RGB thumbnail via GEE; return HxWx3 uint8 or None."""
    try:
        import ee
        import urllib.request

        ee.Initialize(project=PROJECT)
        region = ee.Geometry.Rectangle(
            [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]
        )
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate("2021-06-01", "2021-09-30")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .median()
            .clip(region)
        )
        rgb = s2.select(["B4", "B3", "B2"]).visualize(min=200, max=3200)
        url = rgb.getThumbURL({"dimensions": 160, "format": "png", "region": region})
        with urllib.request.urlopen(url, timeout=25) as resp:
            from io import BytesIO
            from PIL import Image

            img = Image.open(BytesIO(resp.read())).convert("RGB")
            return np.asarray(img)
    except Exception as e:
        print(f"  thumbnail fail ({lon:.3f},{lat:.3f}): {e}")
        return None


def build_predictions() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    samples = pd.read_csv(SAMPLES)
    ae = pd.read_csv(AE_CSV)
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    bundle = joblib.load(B3_MODEL)
    clf = bundle["model"]
    feat = bundle.get("features", ae_cols)

    merged = samples.merge(ae[["sample_id"] + ae_cols], on="sample_id", how="left")
    finite = merged[ae_cols].apply(lambda c: np.isfinite(pd.to_numeric(c, errors="coerce"))).all(axis=1)
    merged["ae_ok"] = finite
    hold = merged[(merged["split"] == "val_holdout") & merged["ae_ok"]].copy()
    train = merged[(merged["split"] == "train") & merged["ae_ok"]].copy()

    Xh = hold[feat].astype(float).values
    hold["prob"] = clf.predict_proba(Xh)[:, 1]
    hold["pred"] = clf.predict(Xh)
    return train, hold, {"features": feat, "model_path": str(B3_MODEL)}


def enrich_metrics(hold: pd.DataFrame, poly_df: pd.DataFrame) -> dict:
    y = hold["class_code"].astype(int).values
    pred = hold["pred"].astype(int).values
    prob = hold["prob"].astype(float).values
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    point = {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "OA": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "AP": float(average_precision_score(y, prob)),
        "AUC": float(roc_auc_score(y, prob)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }

    py = poly_df["y"].astype(int).values
    pp = poly_df["pred"].astype(int).values
    pcm = confusion_matrix(py, pp, labels=[0, 1])
    ptn, pfp, pfn, ptp = [int(x) for x in pcm.ravel()]
    n_poly = int(len(poly_df))
    n_correct = int((py == pp).sum())
    lo, hi = wilson_ci(n_correct, n_poly)
    poly = {
        "n_polys": n_poly,
        "n_apple": int((py == 1).sum()),
        "n_non": int((py == 0).sum()),
        "OA": float(accuracy_score(py, pp)),
        "precision": float(precision_score(py, pp, zero_division=0)),
        "recall": float(recall_score(py, pp, zero_division=0)),
        "F1": float(f1_score(py, pp, zero_division=0)),
        "confusion_matrix": {"TN": ptn, "FP": pfp, "FN": pfn, "TP": ptp},
        "wilson_95ci_OA": {"lo": lo, "hi": hi, "k_correct": n_correct, "n": n_poly},
        "details": poly_df.to_dict(orient="records"),
    }
    return {
        "version": "v2",
        "primary_unit": "polygon_majority",
        "note": "Polygon-level metrics are primary; point-level are spatially dependent within polygons.",
        "point_level": point,
        "poly_majority": poly,
    }


def draw_figure(train: pd.DataFrame, hold: pd.DataFrame, poly_df: pd.DataFrame, geoms: Dict[str, Polygon], metrics: dict):
    plt = setup_matplotlib_zh()
    import geopandas as gpd
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1.0], width_ratios=[1.15, 0.9, 1.0],
                  hspace=0.28, wspace=0.28)

    # ---- (a) spatial map ----
    ax = fig.add_subplot(gs[0, 0])
    try:
        gdf_g = gpd.read_file(GONGLIU)
        gdf_g.boundary.plot(ax=ax, color="#333333", lw=1.2, zorder=2)
    except Exception as e:
        print(f"Gongliu boundary fail: {e}")
    # train poly centroids
    man = pd.read_csv(MANIFEST)
    man_ok = man[man["split"].isin(["train", "val_holdout"])]
    tr = man_ok[man_ok["split"] == "train"]
    ax.scatter(tr["cen_lon"], tr["cen_lat"], s=28, c="#9E9E9E", marker="o",
               edgecolors="white", linewidths=0.4, label="训练面 (非巩留)", zorder=3, alpha=0.85)

    ha = poly_df[poly_df["y"] == 1]
    hn = poly_df[poly_df["y"] == 0]
    ax.scatter(ha["cen_lon"], ha["cen_lat"], s=70, c="#D62728", marker="^",
               edgecolors="k", linewidths=0.5, label="Holdout 苹果面", zorder=5)
    ax.scatter(hn["cen_lon"], hn["cen_lat"], s=70, c="#1F77B4", marker="s",
               edgecolors="k", linewidths=0.5, label="Holdout 非果园面", zorder=5)

    # focus inset-ish: show full Ili span but annotate Gongliu
    ax.set_xlim(81.55, 83.05)
    ax.set_ylim(43.05, 43.95)
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_title("(a) 空间划分：巩留完全 holdout", loc="left", fontsize=11)
    ax.text(0.99, 0.01, "审图号：GS（2024）0650号", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555", zorder=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    # Gongliu label
    ax.text(82.55, 43.35, "巩留县", fontsize=10, color="#333", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # ---- (b) polygon CM ----
    ax = fig.add_subplot(gs[0, 1])
    pcm = metrics["poly_majority"]["confusion_matrix"]
    mat = np.array([[pcm["TN"], pcm["FP"]], [pcm["FN"], pcm["TP"]]], dtype=float)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(12, mat.max()))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测: 非果园", "预测: 苹果"])
    ax.set_yticklabels(["真: 非果园", "真: 苹果"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=16,
                color="white" if v > mat.max() * 0.55 else "black", fontweight="bold")
    pm = metrics["poly_majority"]
    ax.set_title(
        f"(b) 面级混淆矩阵 (n={pm['n_polys']})\n"
        f"F1={pm['F1']:.3f}  OA={pm['OA']:.3f}\n"
        f"P={pm['precision']:.3f}  R={pm['recall']:.3f}",
        loc="left", fontsize=10,
    )
    # colorbar small
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # ---- (c) PR curve ----
    ax = fig.add_subplot(gs[0, 2])
    y = hold["class_code"].astype(int).values
    prob = hold["prob"].astype(float).values
    prec, rec, _ = precision_recall_curve(y, prob)
    ap = float(average_precision_score(y, prob))
    ax.plot(rec, prec, color="#D62728", lw=2.2, label=f"PR (AP={ap:.3f})")
    base = y.mean()
    ax.axhline(base, color="gray", ls="--", lw=1, label=f"基线 (正例比例={base:.2f})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    pt = metrics["point_level"]
    ax.set_title(
        f"(c) 点级 PR 曲线 (n={pt['n']})\nF1={pt['F1']:.3f}  OA={pt['OA']:.3f}",
        loc="left", fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ---- (d) case panels ----
    # pick representative: prefer mix of TP/TN/FP/FN
    poly_df = poly_df.copy()
    poly_df["outcome"] = [outcome_label(int(r.y), int(r.pred)) for _, r in poly_df.iterrows()]
    preferred_order = ["TP", "TN", "FP", "FN", "TP", "TN"]
    chosen = []
    used = set()
    for tag in preferred_order:
        cands = poly_df[(poly_df["outcome"] == tag) & (~poly_df["poly_name"].isin(used))]
        if len(cands):
            # pick median area-ish by n_pts
            row = cands.sort_values("n_pts").iloc[len(cands) // 2]
            chosen.append(row)
            used.add(row["poly_name"])
        if len(chosen) >= 6:
            break
    # fill if fewer
    for _, row in poly_df.iterrows():
        if len(chosen) >= 6:
            break
        if row["poly_name"] not in used:
            chosen.append(row)
            used.add(row["poly_name"])
    chosen = chosen[:6]

    # nested grid in bottom row spanning all columns
    gs_d = gs[1, :].subgridspec(2, 3, hspace=0.35, wspace=0.25)
    outcome_colors = {"TP": "#2CA02C", "TN": "#1F77B4", "FP": "#FF7F0E", "FN": "#D62728"}

    thumb_ok = 0
    for i, row in enumerate(chosen):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs_d[r, c])
        name = row["poly_name"]
        lon, lat = float(row["cen_lon"]), float(row["cen_lat"])
        outcome = row["outcome"]
        mean_p = float(row["mean_prob"])

        img = try_s2_thumbnail(lon, lat, half_deg=0.006)
        geom = geoms.get(name)

        if img is not None:
            thumb_ok += 1
            ax.imshow(img, extent=[lon - 0.006, lon + 0.006, lat - 0.006, lat + 0.006], origin="upper")
            if geom is not None:
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, color=outcome_colors[outcome], lw=1.8)
            ax.set_xlim(lon - 0.006, lon + 0.006)
            ax.set_ylim(lat - 0.006, lat + 0.006)
        else:
            # cartographic maplet
            ax.set_facecolor("#F4F1EA")
            pad = 0.004
            if geom is not None:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, color=outcome_colors[outcome], alpha=0.35, zorder=2)
                ax.plot(xs, ys, color=outcome_colors[outcome], lw=1.8, zorder=3)
                minx, miny, maxx, maxy = geom.bounds
                cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                span = max(maxx - minx, maxy - miny, 0.003) * 1.8
                ax.set_xlim(cx - span / 2, cx + span / 2)
                ax.set_ylim(cy - span / 2, cy + span / 2)
            else:
                ax.scatter([lon], [lat], c=outcome_colors[outcome], s=80, zorder=3)
                ax.set_xlim(lon - pad, lon + pad)
                ax.set_ylim(lat - pad, lat + pad)
            # sample points
            pts = hold[hold["poly_name"] == name]
            ax.scatter(pts["lon"], pts["lat"], c="k", s=8, alpha=0.7, zorder=4)

        ax.set_xticks([])
        ax.set_yticks([])
        ytrue = "苹果" if int(row["y"]) == 1 else "非果园"
        ax.set_title(
            f"{name}  [{outcome}]\n真={ytrue}  预测p̄={mean_p:.2f}",
            fontsize=8, color=outcome_colors[outcome],
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(outcome_colors[outcome])
            spine.set_linewidth(2)

    # panel (d) label
    fig.text(0.02, 0.48, "(d) 代表性 holdout 面案例（轮廓 + 预测概率均值 + TP/FP/FN/TN）",
             fontsize=11, va="center")

    ci = metrics["poly_majority"]["wilson_95ci_OA"]
    fig.suptitle(
        "巩留多边形空间 holdout 独立验证（B3 AlphaEarth-only，2021）\n"
        f"面级为主：n={pm['n_polys']}（苹果{pm['n_apple']}/非果园{pm['n_non']}），"
        f"F1={pm['F1']:.3f}，OA={pm['OA']:.3f}（Wilson 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]）",
        fontsize=13, y=0.995,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    out_path = FIG / "fig_main_holdout_validation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    # also into polygon_samples
    fig.savefig(OUT / "fig_main_holdout_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} (thumbnails ok={thumb_ok}/{len(chosen)})")
    return thumb_ok, len(chosen)


def write_caption(metrics: dict, path: Path):
    pm = metrics["poly_majority"]
    pt = metrics["point_level"]
    ci = pm["wilson_95ci_OA"]
    text = f"""# 图注：巩留多边形空间 holdout 独立验证（主图）

**图 X. 巩留县多边形空间 holdout 独立验证结果（B3，AlphaEarth-only，2021）。**

- **(a)** 伊犁/巩留空间背景：灰色圆点为训练多边形质心（非巩留）；红色三角为 holdout 苹果面，蓝色方块为 holdout 非果园面。巩留县域样本全部作为空间 holdout，未参与训练。
- **(b)** **多边形级**混淆矩阵（多数投票，面内点数≥3；n={pm['n_polys']}，苹果 {pm['n_apple']} / 非果园 {pm['n_non']}）。面级 Precision={pm['precision']:.3f}，Recall={pm['recall']:.3f}，F1={pm['F1']:.3f}，OA={pm['OA']:.3f}；OA 的 Wilson 95% CI 为 [{ci['lo']:.3f}, {ci['hi']:.3f}]。
- **(c)** Holdout 样点级 Precision–Recall 曲线（n={pt['n']}），Average Precision (AP)={pt['AP']:.3f}；点级 F1={pt['F1']:.3f}，OA={pt['OA']:.3f}（供参考）。
- **(d)** 代表性 holdout 面案例：面轮廓、面内点预测概率均值，以及 TP/FP/FN/TN 标签（优先真实 Sentinel-2 缩略图，否则为制图示意）。

**说明（重要）：以多边形级评价为主。** 面内多样点存在空间依赖（同一果园/地块相关），点级指标会高估有效独立样本量；论文主结论与外部证据应以面级 Precision/Recall/F1/OA 为准，点级 PR/F1 仅作辅助。
"""
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Loading predictions…")
    train, hold, meta = build_predictions()
    print(f"Train pts={len(train)} Hold pts={len(hold)}")
    poly_df = poly_majority_table(hold)
    print(poly_df[["poly_name", "y", "pred", "n_pts", "mean_prob"]].to_string(index=False))

    metrics = enrich_metrics(hold, poly_df)
    metrics["model"] = meta
    save_json(metrics, OUT / "metrics_poly_holdout_v2_enriched.json")
    save_json(metrics, FIG / "metrics_poly_holdout_v2_enriched.json")

    print("Parsing KML geometries…")
    geoms = parse_kml_polys(KML)
    print(f"KML polys parsed: {len(geoms)}")

    print("Drawing figure…")
    thumb_ok, n_cases = draw_figure(train, hold, poly_df, geoms, metrics)

    write_caption(metrics, FIG / "fig_main_holdout_caption_zh.md")
    write_caption(metrics, OUT / "fig_main_holdout_caption_zh.md")

    print("=== SUMMARY ===")
    print(json.dumps({
        "point_F1": metrics["point_level"]["F1"],
        "point_OA": metrics["point_level"]["OA"],
        "point_AP": metrics["point_level"]["AP"],
        "poly_F1": metrics["poly_majority"]["F1"],
        "poly_OA": metrics["poly_majority"]["OA"],
        "poly_P": metrics["poly_majority"]["precision"],
        "poly_R": metrics["poly_majority"]["recall"],
        "poly_CM": metrics["poly_majority"]["confusion_matrix"],
        "wilson_OA": metrics["poly_majority"]["wilson_95ci_OA"],
        "thumbnails": f"{thumb_ok}/{n_cases}",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
