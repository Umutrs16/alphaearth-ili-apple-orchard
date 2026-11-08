# -*- coding: utf-8 -*-
"""B3 AlphaEarth-only polygon-sample trial (v2 re-run)."""
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
from shapely.geometry import Point, Polygon
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "revision"))
from common import make_lgbm, save_json  # noqa: E402
from utils import compute_metrics  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "polygon_samples"
MODELS = OUT / "models"
KML = OUT / "Confirmed_apple_field_points_area.kml"
PROJECT = "hallowed-chain-458005-g9"
AE_CACHE = OUT / "poly_ae_features_2021_v2.csv"
AE_CACHE_V1 = OUT / "poly_ae_features_2021.csv"
OLD_B3_AOMC = ROOT / "outputs" / "revision" / "models" / "B3_full_2019_2021.joblib"
KIT_AE = ROOT / "outputs" / "revision" / "independent_val" / "kit_ae_features_2021.csv"
FILLED_XLSX = ROOT / "data" / "interpretation_kit" / "Gongliu_independent_val_判读表_filled.xlsx"
SEED = 42
VERSION = "v2"


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
    for p in re.split(r"[;；]", desc):
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
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def geodetic_area_ha(poly: Polygon) -> float:
    if poly.is_empty:
        return 0.0
    cen = poly.centroid
    lat0 = math.radians(cen.y)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat0)
    def ring_area(coords):
        xs, ys = [], []
        for x, y in coords:
            xs.append((x - cen.x) * m_per_deg_lon)
            ys.append((y - cen.y) * m_per_deg_lat)
        a = 0.0
        for i in range(len(xs) - 1):
            a += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
        return abs(a) / 2.0
    area_m2 = ring_area(list(poly.exterior.coords))
    for interior in poly.interiors:
        area_m2 -= ring_area(list(interior.coords))
    return max(0.0, area_m2) / 10000.0


def parse_kml_polygons(path: Path) -> List[Dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()
    local = lambda tag: f"{{http://www.opengis.net/kml/2.2}}{tag}"
    placemarks = []
    for pm in root.iter(local("Placemark")):
        name_el = pm.find(local("name"))
        desc_el = pm.find(local("description"))
        name = (name_el.text or "").strip() if name_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
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
        placemarks.append({"name": name, "desc": desc, "attrs": parse_desc(desc), "geom": geom})
    return placemarks


def build_manifest(placemarks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pm in placemarks:
        attrs = pm["attrs"]
        geom = pm["geom"]
        cc_raw = attrs.get("class_code", "")
        try:
            class_code = int(float(cc_raw)) if cc_raw != "" else None
        except Exception:
            class_code = None
        county = attrs.get("county", "") or ""
        if "split" in county:
            county = county.split("split")[0].strip(" ;；")
        source = attrs.get("source", "") or ""
        split_attr = attrs.get("split", "") or ""
        area_ha = geodetic_area_ha(geom)
        cen = geom.centroid
        rows.append(
            {
                "name": pm["name"],
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
    excluded, reason = [], []
    for _, r in df.iterrows():
        if r["class_code"] not in (0, 1):
            excluded.append(True)
            reason.append("missing_or_invalid_class")
        elif r["area_ha"] > 50:
            excluded.append(True)
            reason.append("area_ha_gt_50")
        else:
            excluded.append(False)
            reason.append("")
    df["excluded"] = excluded
    df["exclude_reason"] = reason

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
        if r["class_code"] == 0 and (not r["county"] or str(r["county"]).strip() == "" or str(r["county"]) == "nan"):
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


def sample_points_in_poly(geom, n_target, min_spacing_m, rng, max_tries=8000):
    minx, miny, maxx, maxy = geom.bounds
    cen = geom.centroid
    lat0 = math.radians(cen.y)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat0)
    pts = []
    tries = 0
    while len(pts) < n_target and tries < max_tries:
        tries += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not geom.contains(Point(x, y)):
            continue
        ok = True
        for px, py in pts:
            dx = (x - px) * m_per_deg_lon
            dy = (y - py) * m_per_deg_lat
            if math.hypot(dx, dy) < min_spacing_m:
                ok = False
                break
        if ok:
            pts.append((x, y))
    if len(pts) < max(1, n_target // 2):
        tries = 0
        soft = min_spacing_m * 0.4
        while len(pts) < n_target and tries < max_tries:
            tries += 1
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if not geom.contains(Point(x, y)):
                continue
            ok = True
            for px, py in pts:
                dx = (x - px) * m_per_deg_lon
                dy = (y - py) * m_per_deg_lat
                if math.hypot(dx, dy) < soft:
                    ok = False
                    break
            if ok:
                pts.append((x, y))
    if not pts and geom.centroid.within(geom):
        pts.append((float(geom.centroid.x), float(geom.centroid.y)))
    return pts


def choose_n_points(area_ha: float, class_code: int) -> Tuple[int, float]:
    if class_code == 1:
        if area_ha < 0.3:
            return 3, 20.0
        if area_ha < 1.0:
            return 5, 25.0
        if area_ha < 3.0:
            return 8, 30.0
        if area_ha < 10.0:
            return 12, 35.0
        return 15, 40.0
    if area_ha < 0.5:
        return 3, 25.0
    if area_ha < 2.0:
        return 5, 30.0
    if area_ha < 10.0:
        return 8, 35.0
    return 10, 40.0


def build_sample_points(manifest: pd.DataFrame) -> pd.DataFrame:
    kept = manifest[~manifest["excluded"]].copy()
    plans = []
    for _, r in kept.iterrows():
        n, sp = choose_n_points(float(r["area_ha"]), int(r["class_code"]))
        plans.append((r, n, sp))
    total_target = sum(p[1] for p in plans)
    CAP = 1200
    scale = CAP / total_target if total_target > CAP else 1.0
    rows = []
    for r, n, sp in plans:
        # deterministic per-poly RNG so re-runs are stable for same poly/area
        seed_poly = (SEED + abs(hash(str(r["name"]))) % 100000) % (2**31 - 1)
        rng = np.random.default_rng(seed_poly)
        n_adj = max(3 if float(r["area_ha"]) >= 0.2 else 1, int(round(n * scale)))
        if int(r["class_code"]) == 0:
            n_adj = max(2 if float(r["area_ha"]) >= 0.2 else 1, int(round(n * scale)))
        pts = sample_points_in_poly(r["geom"], n_adj, sp, rng)
        for i, (lon, lat) in enumerate(pts, 1):
            rows.append(
                {
                    "sample_id": f"{r['name']}-P{i:02d}",
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


def _lonlat_key(lon, lat, nd=5):
    return (round(float(lon), nd), round(float(lat), nd))


def load_ae_lookup(paths: List[Path]) -> Dict[Tuple[float, float], dict]:
    """Map rounded lon/lat -> AE record (prefer sample_id-preserving)."""
    lookup = {}
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            if "lon" not in r or "lat" not in r:
                continue
            if not all(c in r and pd.notna(r[c]) for c in ae_cols[:1]):
                continue
            # require mostly finite AE
            vals = [r.get(c) for c in ae_cols]
            if sum(pd.notna(v) and np.isfinite(float(v)) for v in vals) < 60:
                continue
            key = _lonlat_key(r["lon"], r["lat"])
            rec = {"lon": float(r["lon"]), "lat": float(r["lat"]), "year": int(r.get("year", 2021))}
            for c in ae_cols:
                rec[c] = float(r[c]) if pd.notna(r[c]) else np.nan
            lookup[key] = rec
        print(f"AE lookup loaded from {p.name}: keys now {len(lookup)}")
    return lookup


def sample_ae_features(df: pd.DataFrame, year: int = 2021, chunk: int = 40) -> pd.DataFrame:
    import ee

    ee.Initialize(project=PROJECT)
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    # seed from v1 + any existing v2
    lookup = load_ae_lookup([AE_CACHE_V1, AE_CACHE])

    done = {}
    if AE_CACHE.exists():
        cached = pd.read_csv(AE_CACHE)
        for _, r in cached.iterrows():
            done[r["sample_id"]] = r.to_dict()
        print(f"AE v2 cache by sample_id: {len(done)}")

    # try fill from lon/lat lookup for new sample_ids
    for _, r in df.iterrows():
        sid = r["sample_id"]
        if sid in done:
            continue
        key = _lonlat_key(r.lon, r.lat)
        if key in lookup:
            rec = dict(lookup[key])
            rec["sample_id"] = sid
            done[sid] = rec

    todo = df[~df["sample_id"].isin(done)].copy()
    print(f"AE to sample via GEE: {len(todo)} / {len(df)} (reused {len(df)-len(todo)})")

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
            print(f"AE chunk fail @ {i}: {e}; retry")
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

    # keep only current sample_ids in cache file (plus write)
    out_rows = []
    for _, r in df.iterrows():
        if r.sample_id in done:
            out_rows.append(done[r.sample_id])
    out = pd.DataFrame(out_rows)
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
    if prob is not None and len(np.unique(y)) > 1:
        try:
            m["AUC"] = float(roc_auc_score(y, np.asarray(prob)))
        except Exception:
            pass
        try:
            m2 = compute_metrics(y, pred, np.asarray(prob))
            for k in ("kappa", "Brier"):
                if k in m2:
                    m[k] = float(m2[k])
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


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def fmt_m(m: Optional[dict]) -> str:
    if not m or m.get("OA") is None:
        return "N/A"
    return (
        f"n={m.get('n')}, OA={m.get('OA'):.3f}, P={m.get('precision'):.3f}, "
        f"R={m.get('recall'):.3f}, F1={m.get('F1'):.3f}, CM={m.get('confusion_matrix')}"
    )


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    print("Parsing KML…")
    placemarks = parse_kml_polygons(KML)
    print(f"Found {len(placemarks)} polygons")
    manifest = build_manifest(placemarks)

    man_out = manifest.drop(columns=["geom"]).copy()
    man_out.to_csv(OUT / "split_manifest_v2.csv", index=False)
    man_out.to_csv(OUT / "split_manifest.csv", index=False)  # latest
    man_out[man_out["excluded"]].to_csv(OUT / "excluded_polygons_qc_v2.csv", index=False)

    print("Sampling points…")
    samples = build_sample_points(manifest)
    samples.to_csv(OUT / "sample_points_v2.csv", index=False)
    samples.to_csv(OUT / "sample_points.csv", index=False)

    kept = manifest[~manifest["excluded"]]
    summary_rows = (
        kept.groupby(["split", "class_code", "county"], dropna=False)
        .agg(n=("name", "count"), area_ha=("area_ha", "sum"))
        .reset_index()
    )
    summary_md = [
        f"# Train / Holdout Summary ({VERSION})",
        "",
        "## Polygons (kept)",
        "",
        df_to_md_table(summary_rows.round({"area_ha": 2})),
        "",
        "## Exclusions",
        "",
        df_to_md_table(man_out[man_out["excluded"]][["name", "class_code", "area_ha", "exclude_reason", "county"]]),
        "",
        "## Sample points",
        "",
        df_to_md_table(samples.groupby(["split", "class_code"]).size().reset_index(name="n")),
        "",
        f"- Total sample points: {len(samples)}",
        f"- Train points: {(samples.split=='train').sum()}",
        f"- Holdout points: {(samples.split=='val_holdout').sum()}",
    ]
    (OUT / "train_holdout_summary_v2.md").write_text("\n".join(summary_md), encoding="utf-8")

    n_train_poly = int(((~manifest.excluded) & (manifest.split == "train")).sum())
    n_hold_poly = int(((~manifest.excluded) & (manifest.split == "val_holdout")).sum())
    print(f"Samples: {len(samples)} | train={(samples.split=='train').sum()} holdout={(samples.split=='val_holdout').sum()}")
    print(f"Polys kept train={n_train_poly} holdout={n_hold_poly} excluded={int(manifest.excluded.sum())}")

    gee_ok, gee_err = True, ""
    try:
        print("Sampling AE via GEE…")
        ae = sample_ae_features(samples, year=2021, chunk=40)
    except Exception as e:
        gee_ok, gee_err = False, str(e)
        print(f"GEE FAILED: {e}")
        ae = pd.DataFrame()

    if not gee_ok or ae.empty:
        report = [
            f"# B3 面样本试验报告 {VERSION}（AE未完成）",
            "",
            f"GEE失败: `{gee_err}`",
            f"- polys={len(manifest)} excluded={int(manifest.excluded.sum())} train={n_train_poly} holdout={n_hold_poly}",
            f"- samples={len(samples)}",
        ]
        (OUT / f"REPORT_B3_poly_trial_{VERSION}.md").write_text("\n".join(report), encoding="utf-8")
        return

    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    merged = samples.merge(ae[["sample_id"] + ae_cols], on="sample_id", how="left")
    finite = merged[ae_cols].apply(lambda c: np.isfinite(pd.to_numeric(c, errors="coerce"))).all(axis=1)
    merged["ae_ok"] = finite
    print(f"AE finite: {int(finite.sum())} / {len(merged)}")

    train = merged[(merged["split"] == "train") & merged["ae_ok"]].copy()
    hold = merged[(merged["split"] == "val_holdout") & merged["ae_ok"]].copy()
    print(f"Train pts: {len(train)} (pos={(train.class_code==1).sum()} neg={(train.class_code==0).sum()})")
    print(f"Hold pts: {len(hold)} (pos={(hold.class_code==1).sum()} neg={(hold.class_code==0).sum()})")

    Xtr = train[ae_cols].astype(float).values
    ytr = train["class_code"].astype(int).values
    groups = train["poly_name"].astype(str).values

    clf = make_lgbm(seed=SEED)
    clf.fit(Xtr, ytr)
    model_path = MODELS / f"B3_poly_2021_{VERSION}.joblib"
    joblib.dump({"model": clf, "features": ae_cols, "year": 2021, "feature_set": "B3", "version": VERSION}, model_path)
    print(f"Saved {model_path}")

    # Internal CV
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups) if n_groups >= 2 else 0
    cv_scores = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for tr_i, te_i in gkf.split(Xtr, ytr, groups):
            m = make_lgbm(seed=SEED)
            m.fit(Xtr[tr_i], ytr[tr_i])
            pred = m.predict(Xtr[te_i])
            prob = m.predict_proba(Xtr[te_i])[:, 1]
            cv_scores.append(pack_metrics(ytr[te_i], pred, prob))
    cv_summary = {
        "version": VERSION,
        "n_splits": n_splits,
        "n_groups": int(n_groups),
        "folds": cv_scores,
        "mean_F1": float(np.mean([s["F1"] for s in cv_scores])) if cv_scores else None,
        "std_F1": float(np.std([s["F1"] for s in cv_scores])) if cv_scores else None,
        "mean_OA": float(np.mean([s["OA"] for s in cv_scores])) if cv_scores else None,
        "std_OA": float(np.std([s["OA"] for s in cv_scores])) if cv_scores else None,
    }
    save_json(cv_summary, OUT / f"metrics_internal_cv_{VERSION}.json")
    save_json(cv_summary, OUT / "metrics_internal_cv.json")

    # Holdout
    holdout_metrics = {"version": VERSION, "n_holdout_ae_ok": int(len(hold))}
    if len(hold):
        Xh = hold[ae_cols].astype(float).values
        yh = hold["class_code"].astype(int).values
        pred_h = clf.predict(Xh)
        prob_h = clf.predict_proba(Xh)[:, 1]
        hold = hold.copy()
        hold["pred"] = pred_h
        hold["prob"] = prob_h
        point_m = pack_metrics(yh, pred_h, prob_h)
        poly_m = poly_majority_metrics(hold, "pred")
        holdout_metrics["poly_trained_B3"] = {
            "point_level": point_m,
            "poly_majority": {k: v for k, v in poly_m.items() if k != "details"},
            "poly_majority_details": poly_m.get("details", []),
        }
        plot_cm(
            point_m["confusion_matrix"],
            f"Holdout point CM — B3 poly {VERSION}",
            OUT / f"fig_confusion_poly_holdout_B3_{VERSION}.png",
        )
        if OLD_B3_AOMC.exists():
            try:
                old = joblib.load(OLD_B3_AOMC)
                old_model = old["model"] if isinstance(old, dict) and "model" in old else old
                pred_o = old_model.predict(Xh)
                prob_o = old_model.predict_proba(Xh)[:, 1] if hasattr(old_model, "predict_proba") else None
                holdout_metrics["AOMC_trained_B3_full_2019_2021_COMPARISON_ONLY"] = {
                    "point_level": pack_metrics(yh, pred_o, prob_o),
                    "note": "Same holdout points; AOMC-trained B3 — NOT the new poly model.",
                }
            except Exception as e:
                holdout_metrics["AOMC_comparison_error"] = str(e)
    save_json(holdout_metrics, OUT / f"metrics_poly_holdout_{VERSION}.json")
    save_json(holdout_metrics, OUT / "metrics_poly_holdout.json")

    # Visual independent
    visual_metrics = None
    if KIT_AE.exists() and FILLED_XLSX.exists():
        try:
            filled = pd.read_excel(FILLED_XLSX)
            filled["interp_class_code"] = pd.to_numeric(filled["interp_class_code"], errors="coerce")
            filled["confidence"] = filled["confidence"].astype(str).str.strip()
            conf = filled[(filled["interp_class_code"] != 9) & (filled["confidence"].isin(["高", "中"]))].copy()
            if "exclude_flag" in conf.columns:
                ef = conf["exclude_flag"]
                if ef.dtype == object:
                    conf = conf[~ef.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])]
                else:
                    conf = conf[~ef.fillna(False).astype(bool)]
            kit = pd.read_csv(KIT_AE)
            j = conf.merge(kit, on="sample_id", how="inner", suffixes=("", "_ae"))
            ae_ok = j[ae_cols].apply(lambda c: np.isfinite(pd.to_numeric(c, errors="coerce"))).all(axis=1)
            j = j[ae_ok].copy()
            j["y"] = (j["interp_class_code"] == 1).astype(int)
            Xp = j[ae_cols].astype(float).values
            pred_v = clf.predict(Xp)
            prob_v = clf.predict_proba(Xp)[:, 1]
            visual_metrics = {
                "version": VERSION,
                "n_confident_joined": int(len(j)),
                "point_level_apple_code1": pack_metrics(j["y"].values, pred_v, prob_v),
                "filters": "drop code=9 and confidence==低; keep 高/中",
                "model": f"B3_poly_2021_{VERSION}",
            }
            save_json(visual_metrics, OUT / f"metrics_visual_indep_polyB3_{VERSION}.json")
            save_json(visual_metrics, OUT / "metrics_visual_indep_polyB3.json")
            plot_cm(
                visual_metrics["point_level_apple_code1"]["confusion_matrix"],
                f"Visual indep CM — B3 poly {VERSION}",
                OUT / f"fig_confusion_visual_indep_polyB3_{VERSION}.png",
            )
        except Exception as e:
            visual_metrics = {"error": str(e)}
            save_json(visual_metrics, OUT / f"metrics_visual_indep_polyB3_{VERSION}.json")

    # v1 comparison
    v1_hold = load_json(OUT / "metrics_poly_holdout.json")
    # After overwrite, load dedicated v1 if we saved earlier — check REPORT and a backup
    v1_hold_bak = load_json(OUT / "metrics_poly_holdout_v1.json")
    if v1_hold_bak:
        v1_hold = v1_hold_bak
    # Also try reading from report if needed; prefer explicit v1 file
    v1_vis = load_json(OUT / "metrics_visual_indep_polyB3_v1.json") or load_json(
        OUT / "metrics_visual_indep_polyB3.json"
    )
    # If visual file was overwritten to v2 already mid-run, use dedicated v1
    # We'll copy v1 metrics before overwrite at start — see below in launcher

    pt = holdout_metrics.get("poly_trained_B3", {}).get("point_level", {})
    pm = holdout_metrics.get("poly_trained_B3", {}).get("poly_majority", {})
    aomc = holdout_metrics.get("AOMC_trained_B3_full_2019_2021_COMPARISON_ONLY", {}).get("point_level", {})
    vis_m = (visual_metrics or {}).get("point_level_apple_code1", {}) if visual_metrics else {}

    v1_pt = (v1_hold or {}).get("poly_trained_B3", {}).get("point_level", {}) if v1_hold else {}
    v1_vis_m = (v1_vis or {}).get("point_level_apple_code1", {}) if v1_vis else {}
    # If v1_vis was overwritten (same as v2), ignore if version==v2
    if v1_vis and v1_vis.get("version") == VERSION:
        v1_vis_m = {}
        v1_vis = load_json(OUT / "metrics_visual_indep_polyB3_v1.json")
        v1_vis_m = (v1_vis or {}).get("point_level_apple_code1", {}) if v1_vis else {}
    if v1_hold and v1_hold.get("version") == VERSION:
        v1_hold = load_json(OUT / "metrics_poly_holdout_v1.json")
        v1_pt = (v1_hold or {}).get("poly_trained_B3", {}).get("point_level", {}) if v1_hold else {}

    ex_list = man_out[man_out["excluded"]][["name", "class_code", "area_ha", "exclude_reason"]]
    train_apple_ha = float(kept[(kept.split == "train") & (kept.class_code == 1)]["area_ha"].sum())
    train_neg_ha = float(kept[(kept.split == "train") & (kept.class_code == 0)]["area_ha"].sum())
    hold_apple_ha = float(kept[(kept.split == "val_holdout") & (kept.class_code == 1)]["area_ha"].sum())
    hold_neg_ha = float(kept[(kept.split == "val_holdout") & (kept.class_code == 0)]["area_ha"].sum())
    n_train_apple = int(((kept.split == "train") & (kept.class_code == 1)).sum())
    n_train_neg = int(((kept.split == "train") & (kept.class_code == 0)).sum())
    n_hold_apple = int(((kept.split == "val_holdout") & (kept.class_code == 1)).sum())
    n_hold_neg = int(((kept.split == "val_holdout") & (kept.class_code == 0)).sum())

    delta_vis_f1 = None
    if vis_m and v1_vis_m and v1_vis_m.get("F1") is not None and vis_m.get("F1") is not None:
        delta_vis_f1 = float(vis_m["F1"]) - float(v1_vis_m["F1"])

    lines = []
    lines.append(f"# B3 面样本试验报告 {VERSION}（AlphaEarth-only，2021）")
    lines.append("")
    lines.append(
        f"本报告为 **完整重跑 {VERSION}**：更新后的 KML（含更多巩留非果园面 + 更多巩留苹果 holdout），"
        "训练新的 LightGBM B3，并与 v1 对照。"
    )
    lines.append("")
    lines.append("## 1. 数据清洗与划分")
    lines.append("")
    lines.append(f"- KML 面要素数: **{len(manifest)}**（忽略参考 Point）")
    lines.append(f"- 排除 area>50ha: **{int(manifest.excluded.sum())}**")
    for _, r in ex_list.iterrows():
        lines.append(
            f"  - `{r['name']}` class={r['class_code']} area={r['area_ha']:.1f}ha reason={r['exclude_reason']}"
        )
    lines.append(
        f"- 训练多边形: **{n_train_poly}**（苹果 {n_train_apple} / {train_apple_ha:.1f}ha；"
        f"非果园 {n_train_neg} / {train_neg_ha:.1f}ha）"
    )
    lines.append(
        f"- Holdout 多边形: **{n_hold_poly}**（苹果 {n_hold_apple} / {hold_apple_ha:.1f}ha；"
        f"非果园 {n_hold_neg} / {hold_neg_ha:.1f}ha）"
    )
    lines.append(
        f"- 样点: **{len(samples)}**（train={(samples.split=='train').sum()}, "
        f"holdout={(samples.split=='val_holdout').sum()}）；AE 有效 train **{len(train)}** / holdout **{len(hold)}**"
    )
    lines.append("")
    lines.append(
        "划分：county 含巩留 或 split=val_holdout → holdout；其余 → train；"
        "缺 county 的负样本若距巩留苹果 holdout 质心 ≤2km → holdout。"
    )
    lines.append("")
    lines.append("## 2. 模型")
    lines.append("")
    lines.append("- 特征 AE_00..AE_63（2021）；LightGBM `make_lgbm` balanced seed=42")
    lines.append(f"- 模型: `models/B3_poly_2021_{VERSION}.joblib`（保留 v1 模型不覆盖）")
    lines.append("")
    lines.append("## 3. 内部 GroupKFold（按 poly_name）")
    lines.append("")
    if cv_summary.get("mean_F1") is not None:
        lines.append(
            f"- {cv_summary['n_splits']}-fold / {cv_summary['n_groups']} groups: "
            f"F1 = **{cv_summary['mean_F1']:.3f} ± {cv_summary['std_F1']:.3f}**, "
            f"OA = **{cv_summary['mean_OA']:.3f} ± {cv_summary['std_OA']:.3f}**"
        )
    lines.append("")
    lines.append(f"## 4. 巩留 Holdout（{VERSION} 新模型）")
    lines.append("")
    lines.append(f"- 点级: {fmt_m(pt)}")
    if pm:
        lines.append(
            f"- 面级多数投票(≥3点): n_polys={pm.get('n_polys')}, Acc={pm.get('accuracy')}, F1={pm.get('F1')}"
        )
    lines.append("")
    lines.append("### 对照：AOMC 旧 B3（同 holdout 点，勿混淆）")
    lines.append(f"- {fmt_m(aomc)}")
    lines.append("")
    lines.append("### v1 → v2 Holdout 点级对照")
    lines.append("")
    if v1_pt:
        lines.append(f"- **v1**: {fmt_m(v1_pt)}")
    else:
        lines.append("- v1 holdout 指标文件未找到（`metrics_poly_holdout_v1.json`）")
    lines.append(f"- **v2**: {fmt_m(pt)}")
    lines.append("")
    lines.append("## 5. 巩留目视独立验证（高/中置信，去 code=9）")
    lines.append("")
    lines.append(f"- **v2**: {fmt_m(vis_m)}")
    if v1_vis_m:
        lines.append(f"- **v1**: {fmt_m(v1_vis_m)}")
        if delta_vis_f1 is not None:
            lines.append(f"- **ΔF1 (v2−v1)**: **{delta_vis_f1:+.3f}**")
    else:
        lines.append("- v1 visual 指标不可用")
    lines.append("")
    lines.append("## 6. 局限")
    lines.append("")
    lines.append("- 训练集仍可能特克斯偏重；holdout 虽已扩充巩留负样本，仍属小样本。")
    lines.append("- 面内多点空间自相关；内部 CV 按面分组，但对县域外推仍可能乐观。")
    lines.append("- 目视独立集正例稀少时 F1 对假阳性敏感；**不宣称论文级正式验证**。")
    lines.append("")
    lines.append("## 7. 交付")
    lines.append("")
    for fn in [
        f"REPORT_B3_poly_trial_{VERSION}.md",
        "split_manifest_v2.csv",
        "excluded_polygons_qc_v2.csv",
        "sample_points_v2.csv",
        "poly_ae_features_2021_v2.csv",
        f"metrics_poly_holdout_{VERSION}.json",
        f"metrics_internal_cv_{VERSION}.json",
        f"metrics_visual_indep_polyB3_{VERSION}.json",
        f"fig_confusion_poly_holdout_B3_{VERSION}.png",
        f"fig_confusion_visual_indep_polyB3_{VERSION}.png",
        f"models/B3_poly_2021_{VERSION}.joblib",
    ]:
        lines.append(f"- `{fn}`")
    report_path = OUT / f"REPORT_B3_poly_trial_{VERSION}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
