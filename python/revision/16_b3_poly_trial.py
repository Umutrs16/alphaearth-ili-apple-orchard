# -*- coding: utf-8 -*-
"""B3 AlphaEarth-only trial from digitized orchard polygons (year 2021)."""
from __future__ import annotations

import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, shape, mapping
from shapely.ops import unary_union
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "revision"))
from common import make_lgbm, make_rf, save_json  # noqa: E402
from utils import compute_metrics  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "polygon_samples"
MODELS = OUT / "models"
KML = OUT / "Confirmed_apple_field_points_area.kml"
PROJECT = "hallowed-chain-458005-g9"
AE_CACHE = OUT / "poly_ae_features_2021.csv"
OLD_B3 = ROOT / "outputs" / "revision" / "models" / "B3_full_2019_2021.joblib"
KIT_AE = ROOT / "outputs" / "revision" / "independent_val" / "kit_ae_features_2021.csv"
FILLED_XLSX = ROOT / "data" / "independent_val" / "Gongliu_independent_val_判读表_filled.xlsx"
SEED = 42
NS = {"kml": "http://www.opengis.net/kml/2.2"}


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def parse_desc(desc: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not desc:
        return out
    # split on English ; or Chinese ；
    parts = re.split(r"[;；]", desc)
    for p in parts:
        p = p.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def ring_to_coords(coord_text: str) -> List[Tuple[float, float]]:
    pts = []
    for tok in coord_text.strip().split():
        parts = tok.split(",")
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            pts.append((lon, lat))
    return pts


def geodetic_area_ha(poly: Polygon) -> float:
    """Approximate geodesic area in hectares via local equirectangular."""
    if poly.is_empty:
        return 0.0
    cen = poly.centroid
    lat0 = math.radians(cen.y)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat0)
    xs, ys = [], []
    for x, y in poly.exterior.coords:
        xs.append((x - cen.x) * m_per_deg_lon)
        ys.append((y - cen.y) * m_per_deg_lat)
    # shoelace
    a = 0.0
    n = len(xs)
    for i in range(n - 1):
        a += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    area_m2 = abs(a) / 2.0
    # subtract holes roughly
    for interior in poly.interiors:
        xs, ys = [], []
        for x, y in interior.coords:
            xs.append((x - cen.x) * m_per_deg_lon)
            ys.append((y - cen.y) * m_per_deg_lat)
        a = 0.0
        n = len(xs)
        for i in range(n - 1):
            a += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
        area_m2 -= abs(a) / 2.0
    return max(0.0, area_m2) / 10000.0


def parse_kml_polygons(path: Path) -> List[Dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()

    def local(tag: str) -> str:
        return f"{{http://www.opengis.net/kml/2.2}}{tag}"

    placemarks = []
    for pm in root.iter(local("Placemark")):
        name_el = pm.find(local("name"))
        desc_el = pm.find(local("description"))
        name = (name_el.text or "").strip() if name_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        # Polygon?
        poly_el = pm.find(".//" + local("Polygon"))
        if poly_el is None:
            continue
        # outer ring
        outer = poly_el.find(".//" + local("outerBoundaryIs") + "/" + local("LinearRing") + "/" + local("coordinates"))
        if outer is None:
            outer = poly_el.find(".//" + local("LinearRing") + "/" + local("coordinates"))
        if outer is None or not (outer.text or "").strip():
            continue
        coords = ring_to_coords(outer.text)
        if len(coords) < 3:
            continue
        # close ring
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        holes = []
        for inn in poly_el.findall(".//" + local("innerBoundaryIs")):
            ic = inn.find(".//" + local("coordinates"))
            if ic is not None and (ic.text or "").strip():
                h = ring_to_coords(ic.text)
                if len(h) >= 3:
                    if h[0] != h[-1]:
                        h = h + [h[0]]
                    holes.append(h)
        try:
            geom = Polygon(coords, holes)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda g: g.area)
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        attrs = parse_desc(desc)
        placemarks.append(
            {
                "name": name,
                "desc": desc,
                "attrs": attrs,
                "geom": geom,
            }
        )
    return placemarks


def build_manifest(placemarks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pm in placemarks:
        attrs = pm["attrs"]
        geom = pm["geom"]
        name = pm["name"]
        # class
        cc_raw = attrs.get("class_code", "")
        try:
            class_code = int(float(cc_raw)) if cc_raw != "" else None
        except Exception:
            class_code = None
        county = attrs.get("county", "") or ""
        # clean county if it absorbed split (shouldn't with our parser)
        if "split" in county:
            county = county.split("split")[0].strip(" ;；")
        source = attrs.get("source", "") or ""
        split_attr = attrs.get("split", "") or ""
        area_ha = geodetic_area_ha(geom)
        cen = geom.centroid
        rows.append(
            {
                "name": name,
                "class_code": class_code,
                "source": source,
                "county": county,
                "split_attr": split_attr,
                "area_ha": area_ha,
                "cen_lon": float(cen.x),
                "cen_lat": float(cen.y),
                "desc": pm["desc"],
                "geom": geom,
            }
        )
    df = pd.DataFrame(rows)

    # exclusions
    excluded = []
    exclude_reason = []
    for _, r in df.iterrows():
        if r["class_code"] not in (0, 1):
            excluded.append(True)
            exclude_reason.append("missing_or_invalid_class")
        elif r["area_ha"] > 50:
            excluded.append(True)
            exclude_reason.append("area_ha_gt_50")
        else:
            excluded.append(False)
            exclude_reason.append("")
    df["excluded"] = excluded
    df["exclude_reason"] = exclude_reason

    # Gongliu apple holdout centroids for proximity of negatives
    gongliu_mask = (
        (~df["excluded"])
        & (df["class_code"] == 1)
        & (
            (df["county"].astype(str).str.contains("巩留"))
            | (df["split_attr"].astype(str).str.contains("val_holdout"))
        )
    )
    gongliu_cents = df.loc[gongliu_mask, ["cen_lon", "cen_lat"]].values

    splits = []
    for _, r in df.iterrows():
        if r["excluded"]:
            splits.append("excluded")
            continue
        is_gongliu = "巩留" in str(r["county"])
        is_holdout_attr = "val_holdout" in str(r["split_attr"])
        if is_gongliu or is_holdout_attr:
            splits.append("val_holdout")
            continue
        # negatives missing county: proximity to Gongliu apple holdout
        if r["class_code"] == 0 and (not r["county"] or str(r["county"]).strip() == ""):
            near = False
            if len(gongliu_cents):
                for glo, gla in gongliu_cents:
                    if haversine_m(r["cen_lon"], r["cen_lat"], glo, gla) <= 2000:
                        near = True
                        break
            splits.append("val_holdout" if near else "train")
        else:
            splits.append("train")
    df["split"] = splits
    return df


def sample_points_in_poly(
    geom: Polygon,
    n_target: int,
    min_spacing_m: float,
    rng: np.random.Generator,
    max_tries: int = 8000,
) -> List[Tuple[float, float]]:
    """Rejection sample with min spacing in approx meters."""
    minx, miny, maxx, maxy = geom.bounds
    cen = geom.centroid
    lat0 = math.radians(cen.y)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat0)
    # spacing in degrees (conservative)
    dlon = min_spacing_m / max(m_per_deg_lon, 1e-6)
    dlat = min_spacing_m / m_per_deg_lat

    pts: List[Tuple[float, float]] = []
    tries = 0
    while len(pts) < n_target and tries < max_tries:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if not geom.contains(p):
            continue
        ok = True
        for px, py in pts:
            # approx euclidean in deg scaled
            dx = (x - px) * m_per_deg_lon
            dy = (y - py) * m_per_deg_lat
            if math.hypot(dx, dy) < min_spacing_m:
                ok = False
                break
        if ok:
            pts.append((x, y))

    # if too few, relax spacing
    if len(pts) < max(1, n_target // 2):
        tries = 0
        while len(pts) < n_target and tries < max_tries:
            tries += 1
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            p = Point(x, y)
            if geom.contains(p):
                # only soft check
                soft = min_spacing_m * 0.4
                ok = True
                for px, py in pts:
                    dx = (x - px) * m_per_deg_lon
                    dy = (y - py) * m_per_deg_lat
                    if math.hypot(dx, dy) < soft:
                        ok = False
                        break
                if ok:
                    pts.append((x, y))

    # last resort: centroid + small jitter inside
    if not pts and geom.centroid.within(geom):
        pts.append((float(geom.centroid.x), float(geom.centroid.y)))
    return pts


def choose_n_points(area_ha: float, class_code: int) -> Tuple[int, float]:
    """Return (n_target, min_spacing_m)."""
    if class_code == 1:
        # ~8-15, at least 3 if area allows
        if area_ha < 0.3:
            n = 3
            spacing = 20.0
        elif area_ha < 1.0:
            n = 5
            spacing = 25.0
        elif area_ha < 3.0:
            n = 8
            spacing = 30.0
        elif area_ha < 10.0:
            n = 12
            spacing = 35.0
        else:
            n = 15
            spacing = 40.0
    else:
        if area_ha < 0.5:
            n = 3
            spacing = 25.0
        elif area_ha < 2.0:
            n = 5
            spacing = 30.0
        elif area_ha < 10.0:
            n = 8
            spacing = 35.0
        else:
            n = 10
            spacing = 40.0
    return n, spacing


def build_sample_points(manifest: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    kept = manifest[~manifest["excluded"]].copy()
    rows = []
    sid = 0
    # first pass collect desired
    plans = []
    for _, r in kept.iterrows():
        n, sp = choose_n_points(float(r["area_ha"]), int(r["class_code"]))
        plans.append((r, n, sp))

    total_target = sum(p[1] for p in plans)
    # cap ~1000
    CAP = 1100
    scale = 1.0
    if total_target > CAP:
        scale = CAP / total_target

    for r, n, sp in plans:
        n_adj = max(3 if float(r["area_ha"]) >= 0.2 else 1, int(round(n * scale)))
        if int(r["class_code"]) == 0:
            n_adj = max(2 if float(r["area_ha"]) >= 0.2 else 1, int(round(n * scale)))
        pts = sample_points_in_poly(r["geom"], n_adj, sp, rng)
        for lon, lat in pts:
            sid += 1
            rows.append(
                {
                    "sample_id": f"POLY-{sid:04d}",
                    "poly_name": r["name"],
                    "class_code": int(r["class_code"]),
                    "county": r["county"],
                    "split": r["split"],
                    "lon": lon,
                    "lat": lat,
                    "area_ha": float(r["area_ha"]),
                }
            )
    return pd.DataFrame(rows)


def sample_ae_features(df: pd.DataFrame, year: int = 2021, chunk: int = 40) -> pd.DataFrame:
    import ee

    ee.Initialize(project=PROJECT)
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    done = {}
    if AE_CACHE.exists():
        cached = pd.read_csv(AE_CACHE)
        for _, r in cached.iterrows():
            done[r["sample_id"]] = r.to_dict()
        print(f"AE cache loaded: {len(done)}")

    todo = df[~df["sample_id"].isin(done)].copy()
    print(f"AE to sample: {len(todo)} / {len(df)}")
    for i in range(0, len(todo), chunk):
        sub = todo.iloc[i : i + chunk]
        feats = [
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"sid": str(r.sample_id)})
            for _, r in sub.iterrows()
        ]
        fc = ee.FeatureCollection(feats)
        try:
            info = ae.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"]
        except Exception as e:
            print(f"AE chunk fail @ {i}: {e}; sleep and retry once")
            time.sleep(3)
            try:
                info = ae.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"]
            except Exception as e2:
                print(f"AE chunk fail again: {e2}")
                continue
        by_sid = {f["properties"]["sid"]: f["properties"] for f in info}
        for _, r in sub.iterrows():
            props = by_sid.get(str(r.sample_id))
            if not props:
                continue
            rec = {"sample_id": r.sample_id, "lon": float(r.lon), "lat": float(r.lat), "year": year}
            emb_keys = sorted([k for k in props.keys() if k.startswith("A") and k != "sid"])
            for bi, k in enumerate(emb_keys[:64]):
                rec[f"AE_{bi:02d}"] = props[k]
            for bi in range(64):
                rec.setdefault(f"AE_{bi:02d}", np.nan)
            done[r.sample_id] = rec
        pd.DataFrame(list(done.values())).to_csv(AE_CACHE, index=False)
        print(f"AE sampled progress {len(done)}/{len(df)}")
        time.sleep(0.35)
    out = pd.DataFrame(list(done.values()))
    out.to_csv(AE_CACHE, index=False)
    return out


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
    if prob is not None:
        try:
            m2 = compute_metrics(y, pred, np.asarray(prob))
            for k in ("AUC", "kappa", "Brier"):
                if k in m2:
                    m[k] = float(m2[k])
        except Exception:
            pass
    return m


def poly_majority_metrics(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    # require >=3 points
    g = df.groupby("poly_name")
    rows = []
    for name, sub in g:
        if len(sub) < 3:
            continue
        maj = int(sub[pred_col].mode().iloc[0])
        true = int(sub["class_code"].iloc[0])
        rows.append({"poly_name": name, "y": true, "pred": maj, "n_pts": len(sub)})
    if not rows:
        return {"n_polys": 0, "note": "no poly with >=3 points"}
    rdf = pd.DataFrame(rows)
    return {
        "n_polys": int(len(rdf)),
        "accuracy": float(accuracy_score(rdf["y"], rdf["pred"])),
        "F1": float(f1_score(rdf["y"], rdf["pred"], zero_division=0)),
        "details": rows,
    }


def plot_cm(cm_dict: dict, title: str, path: Path):
    tn, fp, fn, tp = cm_dict["TN"], cm_dict["FP"], cm_dict["FN"], cm_dict["TP"]
    mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=12)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_summary_md(manifest: pd.DataFrame, samples: pd.DataFrame, path: Path):
    kept = manifest[~manifest["excluded"]]
    lines = ["# Train / Holdout Summary", ""]
    lines.append("## Polygons (kept)")
    lines.append("")
    ct = kept.groupby(["split", "class_code", "county"], dropna=False).agg(
        n=("name", "count"), area_ha=("area_ha", "sum")
    ).reset_index()
    lines.append(ct.to_markdown(index=False))
    lines.append("")
    lines.append("## Exclusions")
    ex = manifest[manifest["excluded"]][["name", "class_code", "area_ha", "exclude_reason", "county"]]
    lines.append(ex.to_markdown(index=False))
    lines.append("")
    lines.append("## Sample points")
    sct = samples.groupby(["split", "class_code"]).size().reset_index(name="n")
    lines.append(sct.to_markdown(index=False))
    lines.append("")
    lines.append(f"- Total sample points: {len(samples)}")
    lines.append(f"- Train points: {(samples.split=='train').sum()}")
    lines.append(f"- Holdout points: {(samples.split=='val_holdout').sum()}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    print("Parsing KML…")
    placemarks = parse_kml_polygons(KML)
    print(f"Found {len(placemarks)} polygons")
    manifest = build_manifest(placemarks)

    # save manifest without geom
    man_out = manifest.drop(columns=["geom"]).copy()
    man_out.to_csv(OUT / "split_manifest.csv", index=False)
    # QC excluded list
    man_out[man_out["excluded"]].to_csv(OUT / "excluded_polygons_qc.csv", index=False)

    print("Sampling points…")
    samples = build_sample_points(manifest)
    samples.to_csv(OUT / "sample_points.csv", index=False)
    write_summary_md(manifest, samples, OUT / "train_holdout_summary.md")
    print(
        f"Samples: {len(samples)} | train={(samples.split=='train').sum()} holdout={(samples.split=='val_holdout').sum()}"
    )
    print(
        f"Polys kept train={((~manifest.excluded)&(manifest.split=='train')).sum()} "
        f"holdout={((~manifest.excluded)&(manifest.split=='val_holdout')).sum()} "
        f"excluded={manifest.excluded.sum()}"
    )

    # GEE AE
    gee_ok = True
    gee_err = ""
    try:
        print("Sampling AE via GEE…")
        ae = sample_ae_features(samples, year=2021, chunk=40)
    except Exception as e:
        gee_ok = False
        gee_err = str(e)
        print(f"GEE FAILED: {e}")
        ae = pd.DataFrame()

    if not gee_ok or ae.empty or sum(c.startswith("AE_") for c in ae.columns) < 64:
        # still write partial report
        report = [
            "# B3 面样本试验报告（未完成特征采样）",
            "",
            f"GEE/AE 采样失败或未完成：`{gee_err}`",
            "",
            "已交付：清洗后的多边形清单、split、样点。请检查 Earth Engine 配额/鉴权后重跑。",
            "",
            f"- 多边形总数: {len(manifest)}",
            f"- 排除: {int(manifest.excluded.sum())}",
            f"- 训练多边形: {int(((~manifest.excluded)&(manifest.split=='train')).sum())}",
            f"- Holdout 多边形: {int(((~manifest.excluded)&(manifest.split=='val_holdout')).sum())}",
            f"- 样点数: {len(samples)}",
        ]
        (OUT / "REPORT_B3_poly_trial.md").write_text("\n".join(report), encoding="utf-8")
        print("Stopped after cleaning due to GEE failure.")
        return

    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    merged = samples.merge(ae[ ["sample_id"] + ae_cols ], on="sample_id", how="left")
    finite = merged[ae_cols].apply(lambda c: np.isfinite(c.astype(float))).all(axis=1)
    merged["ae_ok"] = finite
    print(f"AE finite: {finite.sum()} / {len(merged)}")

    train = merged[(merged["split"] == "train") & merged["ae_ok"]].copy()
    hold = merged[(merged["split"] == "val_holdout") & merged["ae_ok"]].copy()
    print(f"Train pts: {len(train)} (pos={(train.class_code==1).sum()} neg={(train.class_code==0).sum()})")
    print(f"Hold pts: {len(hold)} (pos={(hold.class_code==1).sum()} neg={(hold.class_code==0).sum()})")

    Xtr = train[ae_cols].astype(float).values
    ytr = train["class_code"].astype(int).values
    groups = train["poly_name"].astype(str).values

    clf = make_lgbm(seed=SEED)
    clf.fit(Xtr, ytr)
    model_path = MODELS / "B3_poly_2021.joblib"
    joblib.dump({"model": clf, "features": ae_cols, "year": 2021, "feature_set": "B3"}, model_path)
    print(f"Saved model {model_path}")

    # Internal GroupKFold
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups) if n_groups >= 2 else 0
    cv_scores = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr_i, te_i) in enumerate(gkf.split(Xtr, ytr, groups)):
            m = make_lgbm(seed=SEED)
            m.fit(Xtr[tr_i], ytr[tr_i])
            pred = m.predict(Xtr[te_i])
            prob = m.predict_proba(Xtr[te_i])[:, 1] if hasattr(m, "predict_proba") else None
            cv_scores.append(pack_metrics(ytr[te_i], pred, prob))
    cv_summary = {
        "n_splits": n_splits,
        "n_groups": int(n_groups),
        "folds": cv_scores,
        "mean_F1": float(np.mean([s["F1"] for s in cv_scores])) if cv_scores else None,
        "std_F1": float(np.std([s["F1"] for s in cv_scores])) if cv_scores else None,
        "mean_OA": float(np.mean([s["OA"] for s in cv_scores])) if cv_scores else None,
        "std_OA": float(np.std([s["OA"] for s in cv_scores])) if cv_scores else None,
    }
    save_json(cv_summary, OUT / "metrics_internal_cv.json")

    # Holdout eval
    holdout_metrics = {"n_holdout_ae_ok": int(len(hold))}
    if len(hold) and hold["class_code"].nunique() >= 1:
        Xh = hold[ae_cols].astype(float).values
        yh = hold["class_code"].astype(int).values
        pred_h = clf.predict(Xh)
        prob_h = clf.predict_proba(Xh)[:, 1]
        hold["pred"] = pred_h
        hold["prob"] = prob_h
        point_m = pack_metrics(yh, pred_h, prob_h)
        poly_m = poly_majority_metrics(hold, "pred")
        holdout_metrics["poly_trained_B3"] = {
            "point_level": point_m,
            "poly_majority": {k: v for k, v in poly_m.items() if k != "details"},
            "poly_majority_details": poly_m.get("details", []),
        }
        plot_cm(point_m["confusion_matrix"], "Holdout point CM — B3 poly-trained", OUT / "fig_confusion_poly_holdout_B3.png")

        # Optional old AOMC model comparison
        if OLD_B3.exists():
            try:
                old = joblib.load(OLD_B3)
                old_model = old["model"] if isinstance(old, dict) and "model" in old else old
                old_feats = old.get("features", ae_cols) if isinstance(old, dict) else ae_cols
                # align
                cols = [c for c in old_feats if c in ae_cols]
                if len(cols) == 64 or len(cols) == len(old_feats):
                    Xold = hold[cols].astype(float).values
                else:
                    Xold = Xh
                pred_o = old_model.predict(Xold)
                prob_o = old_model.predict_proba(Xold)[:, 1] if hasattr(old_model, "predict_proba") else None
                hold["pred_aomc"] = pred_o
                point_o = pack_metrics(yh, pred_o, prob_o)
                holdout_metrics["AOMC_trained_B3_full_2019_2021_COMPARISON_ONLY"] = {
                    "point_level": point_o,
                    "note": "Same holdout points scored with existing AOMC-trained B3 — NOT the new poly model.",
                }
            except Exception as e:
                holdout_metrics["AOMC_comparison_error"] = str(e)
    else:
        holdout_metrics["note"] = "Insufficient holdout points with AE"
    save_json(holdout_metrics, OUT / "metrics_poly_holdout.json")

    # Visual independent val
    visual_metrics = None
    if KIT_AE.exists() and FILLED_XLSX.exists():
        try:
            filled = pd.read_excel(FILLED_XLSX)
            filled["interp_class_code"] = pd.to_numeric(filled["interp_class_code"], errors="coerce")
            filled["confidence"] = filled["confidence"].astype(str).str.strip()
            conf = filled[
                (filled["interp_class_code"] != 9)
                & (filled["confidence"].isin(["高", "中"]))
            ].copy()
            if "exclude_flag" in conf.columns:
                conf = conf[~conf["exclude_flag"].fillna(False).astype(bool)]
            kit = pd.read_csv(KIT_AE)
            # join keys
            sid_col = "sample_id" if "sample_id" in conf.columns else None
            if sid_col and "sample_id" in kit.columns:
                j = conf.merge(kit, on="sample_id", how="inner", suffixes=("", "_ae"))
            else:
                # lon/lat join
                j = conf.merge(kit, on=["lon", "lat"], how="inner", suffixes=("", "_ae"))
            ae_ok = j[ae_cols].apply(lambda c: np.isfinite(c.astype(float))).all(axis=1)
            j = j[ae_ok].copy()
            j["y"] = (j["interp_class_code"] == 1).astype(int)
            Xp = j[ae_cols].astype(float).values
            pred_v = clf.predict(Xp)
            prob_v = clf.predict_proba(Xp)[:, 1]
            visual_metrics = {
                "n_confident_joined": int(len(j)),
                "point_level_apple_code1": pack_metrics(j["y"].values, pred_v, prob_v),
                "filters": "drop code=9 and confidence==低; keep 高/中",
                "model": "B3_poly_2021 (NEW poly-trained)",
            }
            save_json(visual_metrics, OUT / "metrics_visual_indep_polyB3.json")
            plot_cm(
                visual_metrics["point_level_apple_code1"]["confusion_matrix"],
                "Visual indep CM — B3 poly-trained",
                OUT / "fig_confusion_visual_indep_polyB3.png",
            )
        except Exception as e:
            visual_metrics = {"error": str(e)}
            save_json(visual_metrics, OUT / "metrics_visual_indep_polyB3.json")

    # Report
    ex_list = man_out[man_out["excluded"]][["name", "class_code", "area_ha", "exclude_reason"]]
    pt = holdout_metrics.get("poly_trained_B3", {}).get("point_level", {})
    pm = holdout_metrics.get("poly_trained_B3", {}).get("poly_majority", {})
    aomc = holdout_metrics.get("AOMC_trained_B3_full_2019_2021_COMPARISON_ONLY", {}).get("point_level", {})

    def fmt_m(m):
        if not m:
            return "N/A"
        return (
            f"n={m.get('n')}, OA={m.get('OA'):.3f}, P={m.get('precision'):.3f}, "
            f"R={m.get('recall'):.3f}, F1={m.get('F1'):.3f}, CM={m.get('confusion_matrix')}"
            if m.get("OA") is not None
            else str(m)
        )

    lines = []
    lines.append("# B3 面样本试验报告（AlphaEarth-only，2021）")
    lines.append("")
    lines.append("本试验用 Google Earth 勾绘的果园/非果园面样本，训练 **新的** B3（仅 AlphaEarth 年度嵌入）分类器，并在巩留 holdout 上评估。**不是**用旧 AOMC 点样本模型直接打分作为主结果。")
    lines.append("")
    lines.append("## 1. 数据清洗与划分")
    lines.append("")
    lines.append(f"- KML 面要素数: **{len(manifest)}**（已忽略参考 Point）")
    lines.append(f"- 排除: **{int(manifest.excluded.sum())}** 个（见 `excluded_polygons_qc.csv`）")
    for _, r in ex_list.iterrows():
        lines.append(f"  - `{r['name']}` class={r['class_code']} area={r['area_ha']:.1f}ha reason={r['exclude_reason']}")
    lines.append(f"- 训练多边形: **{int(((~manifest.excluded)&(manifest.split=='train')).sum())}**")
    lines.append(f"- Holdout 多边形（巩留/近巩留）: **{int(((~manifest.excluded)&(manifest.split=='val_holdout')).sum())}**")
    lines.append(f"- 样点总数: **{len(samples)}**（train={(samples.split=='train').sum()}, holdout={(samples.split=='val_holdout').sum()}）")
    lines.append(f"- AE 有效样点: train **{len(train)}**, holdout **{len(hold)}**")
    lines.append("")
    lines.append("划分规则：`split=val_holdout` 或 county=巩留县 → holdout；其余县（特克斯/伊宁/新源等）→ train；缺 county 的负样本按与巩留苹果 holdout 质心距离 ≤2km 归 holdout，否则 train。面积 >50ha 一律排除建模。")
    lines.append("")
    lines.append("## 2. 模型")
    lines.append("")
    lines.append("- 特征: AE_00..AE_63（2021 AlphaEarth annual）")
    lines.append("- 分类器: LightGBM（`common.make_lgbm`，class_weight=balanced，seed=42）")
    lines.append(f"- 模型文件: `models/B3_poly_2021.joblib`")
    lines.append("")
    lines.append("## 3. 内部空间交叉验证（GroupKFold by poly_name，仅训练集）")
    lines.append("")
    if cv_summary.get("mean_F1") is not None:
        lines.append(
            f"- {cv_summary['n_splits']}-fold / {cv_summary['n_groups']} groups: "
            f"F1 = **{cv_summary['mean_F1']:.3f} ± {cv_summary['std_F1']:.3f}**, "
            f"OA = **{cv_summary['mean_OA']:.3f} ± {cv_summary['std_OA']:.3f}**"
        )
    else:
        lines.append("- 组数不足，未跑 GroupKFold")
    lines.append("")
    lines.append("## 4. 巩留 Holdout 评估（新面样本模型）")
    lines.append("")
    lines.append(f"- 点级: {fmt_m(pt)}")
    if pm:
        lines.append(f"- 面级多数投票 (≥3点/面): n_polys={pm.get('n_polys')}, Acc={pm.get('accuracy')}, F1={pm.get('F1')}")
    lines.append("")
    lines.append("### 对比：同一 holdout 点用旧 AOMC 训练的 B3（仅对照，勿混淆）")
    lines.append("")
    lines.append(f"- AOMC-B3: {fmt_m(aomc)}")
    lines.append("")
    lines.append("## 5. 巩留目视独立验证（若可用）")
    lines.append("")
    if visual_metrics and "point_level_apple_code1" in visual_metrics:
        lines.append(f"- 过滤: 去掉 code=9 与 confidence=低，保留高/中")
        lines.append(f"- {fmt_m(visual_metrics['point_level_apple_code1'])}")
    elif visual_metrics:
        lines.append(f"- 未完成: {visual_metrics}")
    else:
        lines.append("- 未运行（缺表或 AE 缓存）")
    lines.append("")
    lines.append("## 6. 是否够用？局限与注意")
    lines.append("")
    lines.append("- **特克斯样本偏多**：训练集地理/县域不平衡，模型可能偏特克斯景观。")
    lines.append("- **巩留 holdout 面很少**：苹果 holdout 面有限，点级/面级指标方差大，**不能当作全文正式验证结论**。")
    lines.append("- 面内多点采样存在空间自相关；内部 CV 已按 `poly_name` 分组，但仍偏乐观于真正的县域外推。")
    lines.append("- 大面积负样本（>50ha）已排除，避免把整片大类背景当成一块“样本单元”主导训练。")
    lines.append("- 本报告为试验性 B3 面样本试跑，用于判断勾绘策略与样本量是否值得扩大；**不宣称已达论文级独立验证**。")
    lines.append("")
    lines.append("## 7. 交付文件")
    lines.append("")
    for fn in [
        "split_manifest.csv",
        "excluded_polygons_qc.csv",
        "train_holdout_summary.md",
        "sample_points.csv",
        "poly_ae_features_2021.csv",
        "metrics_poly_holdout.json",
        "metrics_internal_cv.json",
        "metrics_visual_indep_polyB3.json",
        "fig_confusion_poly_holdout_B3.png",
        "models/B3_poly_2021.joblib",
        "REPORT_B3_poly_trial.md",
    ]:
        lines.append(f"- `{fn}`")
    (OUT / "REPORT_B3_poly_trial.md").write_text("\n".join(lines), encoding="utf-8")
    print("Done. Report written.")


if __name__ == "__main__":
    main()
