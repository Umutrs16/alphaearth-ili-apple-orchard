# -*- coding: utf-8 -*-
"""Geographic shortcut diagnostics on Gongliu polygon spatial holdout (v2).

Samples SRTM elevation+slope via GEE, then compares:
  AE-only | DEM-only | lat/lon-only | AE+DEM | AE+lat/lon
on the same train/holdout split as B3_poly v2.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
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
from common import make_lgbm, save_json, setup_matplotlib_zh  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "polygon_samples"
FIG = ROOT / "outputs" / "revision" / "figures_v2"
MODELS = OUT / "models"
PROJECT = "hallowed-chain-458005-g9"
SEED = 42
TERRAIN_CACHE = OUT / "poly_terrain_features_2021.csv"
SAMPLES = OUT / "sample_points_v2.csv"
AE_CSV = OUT / "poly_ae_features_2021_v2.csv"
B3_MODEL = MODELS / "B3_poly_2021_v2.joblib"


def pack_metrics(y, pred, prob=None) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    m = {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "OA": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "F1": float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    if prob is not None and len(np.unique(y)) > 1:
        try:
            m["AUC"] = float(roc_auc_score(y, prob))
        except Exception:
            pass
        try:
            m["AP"] = float(average_precision_score(y, prob))
        except Exception:
            pass
    return m


def poly_majority_metrics(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    rows = []
    for name, sub in df.groupby("poly_name"):
        if len(sub) < 3:
            continue
        maj = int(sub[pred_col].mode().iloc[0])
        true = int(sub["class_code"].iloc[0])
        rows.append({"poly_name": name, "y": true, "pred": maj, "n_pts": int(len(sub))})
    if not rows:
        return {"n_polys": 0, "OA": float("nan"), "F1": float("nan"), "details": []}
    rdf = pd.DataFrame(rows)
    cm = confusion_matrix(rdf["y"], rdf["pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n_polys": int(len(rdf)),
        "OA": float(accuracy_score(rdf["y"], rdf["pred"])),
        "precision": float(precision_score(rdf["y"], rdf["pred"], zero_division=0)),
        "recall": float(recall_score(rdf["y"], rdf["pred"], zero_division=0)),
        "F1": float(f1_score(rdf["y"], rdf["pred"], zero_division=0)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "details": rows,
    }


def sample_terrain(df: pd.DataFrame, chunk: int = 50) -> pd.DataFrame:
    """Sample SRTM elevation + slope for all points; cache to TERRAIN_CACHE."""
    import ee

    ee.Initialize(project=PROJECT)
    dem = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select(["elevation", "slope"])

    done: Dict[str, dict] = {}
    if TERRAIN_CACHE.exists():
        cached = pd.read_csv(TERRAIN_CACHE)
        for _, r in cached.iterrows():
            done[str(r["sample_id"])] = r.to_dict()
        print(f"Terrain cache loaded: {len(done)}")

    todo = df[~df["sample_id"].astype(str).isin(done)].copy()
    print(f"Terrain to sample: {len(todo)} / {len(df)}")
    gee_errors: List[str] = []

    for i in range(0, len(todo), chunk):
        sub = todo.iloc[i : i + chunk]
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(r.lon), float(r.lat)]),
                {"sid": str(r.sample_id)},
            )
            for _, r in sub.iterrows()
        ]
        fc = ee.FeatureCollection(feats)
        try:
            info = dem.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()["features"]
        except Exception as e:
            print(f"Terrain chunk fail @ {i}: {e}; retry once")
            time.sleep(3)
            try:
                info = dem.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()["features"]
            except Exception as e2:
                msg = f"chunk@{i}: {e2}"
                print(f"Terrain chunk fail again: {e2}")
                gee_errors.append(msg)
                continue
        by_sid = {f["properties"]["sid"]: f["properties"] for f in info}
        for _, r in sub.iterrows():
            props = by_sid.get(str(r.sample_id), {})
            done[str(r.sample_id)] = {
                "sample_id": str(r.sample_id),
                "lon": float(r.lon),
                "lat": float(r.lat),
                "elevation": props.get("elevation", np.nan),
                "slope": props.get("slope", np.nan),
            }
        pd.DataFrame(list(done.values())).to_csv(TERRAIN_CACHE, index=False)
        print(f"Terrain progress {len(done)}/{len(df)}")
        time.sleep(0.3)

    out = pd.DataFrame(list(done.values()))
    out.to_csv(TERRAIN_CACHE, index=False)
    return out, gee_errors


def eval_feature_set(
    name: str,
    feat_cols: List[str],
    train: pd.DataFrame,
    hold: pd.DataFrame,
    reload_ae: bool = False,
) -> dict:
    Xtr = train[feat_cols].astype(float).values
    ytr = train["class_code"].astype(int).values
    Xh = hold[feat_cols].astype(float).values
    yh = hold["class_code"].astype(int).values

    if reload_ae and B3_MODEL.exists():
        bundle = joblib.load(B3_MODEL)
        clf = bundle["model"]
        print(f"  [{name}] reloaded existing B3_poly_v2")
    else:
        clf = make_lgbm(seed=SEED)
        clf.fit(Xtr, ytr)
        print(f"  [{name}] trained on {len(train)} pts, {len(feat_cols)} feats")

    pred = clf.predict(Xh)
    prob = clf.predict_proba(Xh)[:, 1]
    hold_eval = hold.copy()
    hold_eval["pred"] = pred
    hold_eval["prob"] = prob
    point_m = pack_metrics(yh, pred, prob)
    poly_m = poly_majority_metrics(hold_eval, "pred")
    return {
        "name": name,
        "features": feat_cols,
        "n_features": len(feat_cols),
        "point_level": point_m,
        "poly_majority": {k: v for k, v in poly_m.items() if k != "details"},
        "poly_majority_details": poly_m.get("details", []),
    }


def plot_shortcut_bars(results: List[dict], ae_f1_point: float, ae_f1_poly: float, path: Path):
    plt = setup_matplotlib_zh()
    names = [r["name"] for r in results]
    point_f1 = [r["point_level"]["F1"] for r in results]
    poly_f1 = [r["poly_majority"]["F1"] for r in results]
    point_oa = [r["point_level"]["OA"] for r in results]
    poly_oa = [r["poly_majority"]["OA"] for r in results]
    d_point = [f - ae_f1_point for f in point_f1]
    d_poly = [f - ae_f1_poly for f in poly_f1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    x = np.arange(len(names))
    w = 0.35
    ax = axes[0]
    b1 = ax.bar(x - w / 2, point_f1, w, label="点级 F1", color="#4C78A8")
    b2 = ax.bar(x + w / 2, poly_f1, w, label="面级多数 F1", color="#F58518")
    ax.axhline(ae_f1_point, color="#4C78A8", ls="--", lw=1, alpha=0.6)
    ax.axhline(ae_f1_poly, color="#F58518", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("(a) Holdout F1（点级 / 面级）")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    b1 = ax.bar(x - w / 2, d_point, w, label="ΔF1 点级 (vs AE)", color="#54A24B")
    b2 = ax.bar(x + w / 2, d_poly, w, label="ΔF1 面级 (vs AE)", color="#E45756")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("ΔF1 vs AE-only")
    ax.set_title("(b) 地理捷径风险：相对 AE-only 的 ΔF1")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (0.01 if h >= 0 else -0.02),
                    f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=7)

    fig.suptitle(globals().get("SUPTITLE", "巩留面样本空间 holdout：地理捷径诊断"), fontsize=13, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def write_report(results: List[dict], ae_name: str, gee_errors: List[str], path: Path):
    by = {r["name"]: r for r in results}
    ae = by[ae_name]
    lines = [
        "# 地理捷径诊断报告（Gongliu polygon holdout v2）",
        "",
        "在与 B3_poly v2 **相同** 的面分组训练 / 巩留 holdout 划分上，比较：",
        "AE-only、DEM-only（elevation+slope）、lat/lon-only、AE+DEM、AE+lat/lon。",
        "LightGBM `make_lgbm`（balanced, seed=42）。",
        "",
        "## 1. Holdout 指标表",
        "",
        "| 特征集 | 点 n | 点 F1 | 点 OA | 面 n | 面 F1 | 面 OA | ΔF1点 vs AE | ΔF1面 vs AE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        dp = r["point_level"]["F1"] - ae["point_level"]["F1"]
        dy = r["poly_majority"]["F1"] - ae["poly_majority"]["F1"]
        lines.append(
            f"| {r['name']} | {r['point_level']['n']} | "
            f"{r['point_level']['F1']:.3f} | {r['point_level']['OA']:.3f} | "
            f"{r['poly_majority']['n_polys']} | {r['poly_majority']['F1']:.3f} | "
            f"{r['poly_majority']['OA']:.3f} | {dp:+.3f} | {dy:+.3f} |"
        )
    lines += ["", "## 2. 解释（捷径风险）", ""]

    dem = by.get("DEM-only")
    ll = by.get("lat/lon-only")
    risk_notes = []
    if dem and dem["point_level"]["F1"] >= 0.80:
        risk_notes.append(
            f"- **DEM-only 点级 F1={dem['point_level']['F1']:.3f}** 偏高：提示苹果/非果园在 "
            "巩留 holdout 上可能沿高程/坡度可分，存在地形捷径风险；AE 增益需对照 ΔF1。"
        )
    elif dem:
        risk_notes.append(
            f"- DEM-only 点级 F1={dem['point_level']['F1']:.3f}，面级 F1={dem['poly_majority']['F1']:.3f}："
            "纯地形不足以复现 AE 表现，地形捷径风险有限。"
        )
    if ll and ll["point_level"]["F1"] >= 0.80:
        risk_notes.append(
            f"- **lat/lon-only 点级 F1={ll['point_level']['F1']:.3f}** 偏高：空间位置 alone 即可较好分割，"
            "需警惕县域/片区捷径（holdout 虽地理独立，但类内空间聚集仍可被坐标拟合）。"
        )
    elif ll:
        risk_notes.append(
            f"- lat/lon-only 点级 F1={ll['point_level']['F1']:.3f}，面级 F1={ll['poly_majority']['F1']:.3f}："
            "纯坐标捷径不明显。"
        )
    ae_dem = by.get("AE+DEM")
    ae_ll = by.get("AE+lat/lon")
    if ae_dem:
        d = ae_dem["point_level"]["F1"] - ae["point_level"]["F1"]
        risk_notes.append(
            f"- AE+DEM 相对 AE ΔF1点={d:+.3f}："
            + ("加入 DEM 几乎无增益，主信号在嵌入。" if abs(d) < 0.02 else "DEM 有可观增益/损失，需审慎解读。")
        )
    if ae_ll:
        d = ae_ll["point_level"]["F1"] - ae["point_level"]["F1"]
        risk_notes.append(
            f"- AE+lat/lon 相对 AE ΔF1点={d:+.3f}："
            + ("坐标附加信息有限。" if abs(d) < 0.02 else "坐标附加改变明显，存在空间捷径可能。")
        )
    if not risk_notes:
        risk_notes.append("- 见上表 ΔF1。")
    lines.extend(risk_notes)
    lines += [
        "",
        "## 3. 方法备注",
        "",
        "- 划分：`sample_points_v2.csv` / 巩留 `val_holdout`；面内点 ≥3 才计入面级多数投票。",
        "- AE-only：优先重载 `B3_poly_2021_v2.joblib`；其余特征集重新训练。",
        "- DEM：USGS/SRTMGL1_003 → elevation, slope（30 m）。",
        f"- GEE 错误条数: {len(gee_errors)}" + (f" — {gee_errors[:3]}" if gee_errors else "（无）"),
        "",
        "## 4. 交付",
        "",
        "- `poly_terrain_features_2021.csv`",
        "- `metrics_shortcut_diagnostics.json`",
        "- `fig_shortcut_diagnostics.png`",
        "- `REPORT_shortcut_diagnostics.md`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(SAMPLES)
    ae = pd.read_csv(AE_CSV)
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    print(f"Samples={len(samples)} AE={len(ae)}")

    terrain, gee_errors = sample_terrain(samples)
    print(f"Terrain rows={len(terrain)} gee_errors={len(gee_errors)}")

    merged = (
        samples.merge(ae[["sample_id"] + ae_cols], on="sample_id", how="left")
        .merge(terrain[["sample_id", "elevation", "slope"]], on="sample_id", how="left")
    )
    # ensure lon/lat present (from samples)
    finite_ae = merged[ae_cols].apply(lambda c: np.isfinite(pd.to_numeric(c, errors="coerce"))).all(axis=1)
    finite_dem = (
        np.isfinite(pd.to_numeric(merged["elevation"], errors="coerce"))
        & np.isfinite(pd.to_numeric(merged["slope"], errors="coerce"))
    )
    merged["ae_ok"] = finite_ae
    merged["dem_ok"] = finite_dem
    print(f"AE ok={finite_ae.sum()} DEM ok={finite_dem.sum()}")

    # Use intersection of usable points for fair comparison across all feature sets
    ok = merged["ae_ok"] & merged["dem_ok"]
    train = merged[(merged["split"] == "train") & ok].copy()
    hold = merged[(merged["split"] == "val_holdout") & ok].copy()
    print(f"Train={len(train)} Hold={len(hold)}")

    feature_sets = [
        ("AE-only", ae_cols, True),
        ("DEM-only", ["elevation", "slope"], False),
        ("lat/lon-only", ["lon", "lat"], False),
        ("AE+DEM", ae_cols + ["elevation", "slope"], False),
        ("AE+lat/lon", ae_cols + ["lon", "lat"], False),
    ]

    results = []
    for name, cols, reload_ae in feature_sets:
        # for AE reload, still need same hold set; model was trained on AE-only
        r = eval_feature_set(name, cols, train, hold, reload_ae=reload_ae)
        results.append(r)
        print(
            f"  -> {name}: point F1={r['point_level']['F1']:.3f} OA={r['point_level']['OA']:.3f} | "
            f"poly F1={r['poly_majority']['F1']:.3f} OA={r['poly_majority']['OA']:.3f}"
        )

    ae_res = next(r for r in results if r["name"] == "AE-only")
    # enrich with delta
    for r in results:
        r["delta_F1_point_vs_AE"] = float(r["point_level"]["F1"] - ae_res["point_level"]["F1"])
        r["delta_F1_poly_vs_AE"] = float(r["poly_majority"]["F1"] - ae_res["poly_majority"]["F1"])

    payload = {
        "version": "v2",
        "seed": SEED,
        "n_train": int(len(train)),
        "n_holdout": int(len(hold)),
        "gee_errors": gee_errors,
        "terrain_cache": str(TERRAIN_CACHE),
        "results": results,
        "ae_baseline": {
            "point_F1": ae_res["point_level"]["F1"],
            "poly_F1": ae_res["poly_majority"]["F1"],
        },
    }
    save_json(payload, OUT / "metrics_shortcut_diagnostics.json")
    save_json(payload, FIG / "metrics_shortcut_diagnostics.json")

    plot_shortcut_bars(
        results,
        ae_res["point_level"]["F1"],
        ae_res["poly_majority"]["F1"],
        FIG / "fig_shortcut_diagnostics.png",
    )
    # also copy into polygon_samples
    plot_shortcut_bars(
        results,
        ae_res["point_level"]["F1"],
        ae_res["poly_majority"]["F1"],
        OUT / "fig_shortcut_diagnostics.png",
    )

    write_report(results, "AE-only", gee_errors, OUT / "REPORT_shortcut_diagnostics.md")
    write_report(results, "AE-only", gee_errors, FIG / "REPORT_shortcut_diagnostics.md")
    print("DONE shortcut diagnostics")


if __name__ == "__main__":
    main()
