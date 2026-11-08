# -*- coding: utf-8 -*-
"""v3: expanded Gongliu polygon holdout — QC, sampling, AE fetch, evaluation of B3_poly_v2 (reused).

Stages (all run by default):
  qc      -> QC_holdout_v3.csv, split_manifest_v3.csv
  sample  -> sample_points_v3.csv (existing 18 faces reuse v2 points if geometry unchanged)
  ae      -> poly_ae_features_2021_v3.csv (GEE, cached)
  eval    -> metrics_poly_holdout_v3.json, predictions_holdout_v3.csv
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "revision"))
from common import save_json  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "polygon_samples"
V3 = OUT / "v3"
NEW_KML = OUT / "holdout_expansion" / "apple_new.kml"
TRAIN_KML = OUT / "Confirmed_apple_field_points_area.kml"
MAN_V2 = OUT / "split_manifest_v2.csv"
SAMPLES_V2 = OUT / "sample_points_v2.csv"
AE_V2 = OUT / "poly_ae_features_2021_v2.csv"
AE_V3 = OUT / "poly_ae_features_2021_v3.csv"
MODEL = OUT / "models" / "B3_poly_2021_v2.joblib"
COUNTY = ROOT / "data" / "roi" / "Yili_county_full.geojson"
PROJECT = "hallowed-chain-458005-g9"
SEED = 42
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
KNS = "{http://www.opengis.net/kml/2.2}"
TO_UTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True).transform


def utm(g):
    return transform(TO_UTM, g)


def parse_desc(desc: str) -> Dict[str, str]:
    out = {}
    for p in re.split(r"[;；]", desc or ""):
        p = p.strip()
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_kml_polys(path: Path) -> List[dict]:
    root = ET.parse(path).getroot()
    rows = []
    for pm in root.iter(KNS + "Placemark"):
        pel = pm.find(".//" + KNS + "Polygon")
        if pel is None:
            continue
        name = (pm.findtext(KNS + "name") or "").strip()
        desc = (pm.findtext(KNS + "description") or "").strip()
        ce = pel.find(".//" + KNS + "outerBoundaryIs/" + KNS + "LinearRing/" + KNS + "coordinates")
        if ce is None:
            ce = pel.find(".//" + KNS + "coordinates")
        coords = []
        for tok in (ce.text or "").split():
            a = tok.split(",")
            if len(a) >= 2:
                coords.append((float(a[0]), float(a[1])))
        rows.append({"raw_name": name, "desc": desc, "coords": coords})
    return rows


def make_geom(coords):
    if len(coords) < 3:
        return None, "lt3_vertices"
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    if len(set(coords)) < 3:
        return None, "lt3_unique_vertices"
    g = Polygon(coords)
    note = ""
    if not g.is_valid:
        g = g.buffer(0)
        note = "fixed_invalid"
    if g.is_empty:
        return None, "empty"
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda x: x.area)
        note = (note + ";multipart_kept_largest").strip(";")
    return g, note


# ------------------------------------------------------------------ QC
def stage_qc():
    V3.mkdir(parents=True, exist_ok=True)
    raw = read_kml_polys(NEW_KML)
    recs = []
    for r in raw:
        m = re.match(r"^\s*(YL-POLY-\d{4})\s*\|\s*c(\d)\s*$", r["raw_name"])
        attrs = parse_desc(r["desc"])
        if m:
            pid, cc, src = m.group(1), int(m.group(2)), "existing_v2_holdout(name_suffix)"
        else:
            pid = r["raw_name"].strip()
            cc = int(float(attrs["class_code"])) if attrs.get("class_code", "") != "" else None
            src = "new_v3(description)"
        g, gnote = make_geom(r["coords"])
        recs.append({"poly_name": pid, "raw_name": r["raw_name"], "class_code": cc, "label_source": src,
                     "county_attr": attrs.get("county", "巩留县" if m else ""),
                     "split_attr": attrs.get("split", "val_holdout" if m else ""),
                     "geom": g, "geom_note": gnote})
    df = pd.DataFrame(recs)
    print(f"Parsed polygons: {len(df)}; duplicates: {df.poly_name.duplicated().sum()}")

    # train set (v2) — reconstruct geoms
    man2 = pd.read_csv(MAN_V2)
    train_names = man2[(man2["split"] == "train") & (~man2["excluded"])]["name"].tolist()
    tk = {r["raw_name"]: make_geom(r["coords"])[0] for r in read_kml_polys(TRAIN_KML)}
    v2_hold = man2[man2["split"] == "val_holdout"]["name"].tolist()

    # Gongliu boundary
    cty = gpd.read_file(COUNTY)
    gl = cty[cty["name"].astype(str).str.contains("巩留")]
    gl_geom = gl.geometry.union_all() if hasattr(gl.geometry, "union_all") else gl.geometry.unary_union

    rows = []
    hold_utm = {}
    for _, r in df.iterrows():
        g = r["geom"]
        d = r.drop(labels=["geom"]).to_dict()
        if g is None:
            d.update(area_ha=np.nan, excluded=True, exclude_reason=f"degenerate:{r.geom_note}")
            rows.append(d)
            continue
        gu = utm(g)
        area = gu.area / 1e4
        d["area_ha"] = area
        d["n_vertices"] = len(g.exterior.coords) - 1
        d["cen_lon"], d["cen_lat"] = g.centroid.x, g.centroid.y
        d["inside_gongliu"] = bool(gl_geom.contains(g.centroid))
        d["frac_inside_gongliu"] = float(utm(g.intersection(gl_geom)).area / gu.area) if gu.area > 0 else 0
        reasons = []
        if r["class_code"] not in (0, 1, 2):
            reasons.append("invalid_class")
        if area > 50:
            reasons.append("area_ha_gt_50")
        if area < 0.01:
            reasons.append("degenerate_area_lt_0.01ha")
        flags = []
        if not d["inside_gongliu"]:
            # flag only: kept in primary holdout (v2 continuity); strict-inside sensitivity drops these
            flags.append("centroid_outside_gongliu_boundary")
        d["qc_flags"] = ";".join(flags)
        # geometry change vs v2 for existing
        if r["poly_name"] in v2_hold and r["poly_name"] in tk and tk[r["poly_name"]] is not None:
            gv2 = utm(tk[r["poly_name"]])
            iou = gu.intersection(gv2).area / gu.union(gv2).area
            d["iou_vs_v2_geom"] = float(iou)
        d["excluded"] = bool(reasons)
        d["exclude_reason"] = ";".join(reasons)
        rows.append(d)
        if not reasons:
            hold_utm[r["poly_name"]] = gu
    qc = pd.DataFrame(rows)

    # overlaps
    names = list(hold_utm)
    ov_hh = {n: [] for n in qc.poly_name}
    ov_ht = {n: [] for n in qc.poly_name}
    cls = dict(zip(qc.poly_name, qc.class_code))
    overlap_pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = hold_utm[a].intersection(hold_utm[b]).area
            if inter > 1.0:  # >1 m2
                ov_hh[a].append(f"{b}({inter/1e4:.3f}ha)")
                ov_hh[b].append(f"{a}({inter/1e4:.3f}ha)")
                overlap_pairs.append({"a": a, "b": b, "type": "holdout-holdout", "class_a": cls[a],
                                      "class_b": cls[b], "overlap_ha": inter / 1e4,
                                      "frac_of_smaller": inter / min(hold_utm[a].area, hold_utm[b].area)})
    train_utm = {n: utm(tk[n]) for n in train_names if tk.get(n) is not None}
    for a in names:
        for b, gb in train_utm.items():
            inter = hold_utm[a].intersection(gb).area
            if inter > 1.0:
                ov_ht[a].append(f"{b}({inter/1e4:.3f}ha)")
                overlap_pairs.append({"a": a, "b": b, "type": "holdout-train", "class_a": cls[a],
                                      "class_b": None, "overlap_ha": inter / 1e4,
                                      "frac_of_smaller": inter / min(hold_utm[a].area, gb.area)})
    # min distance holdout -> train
    for a in names:
        dmin = min(hold_utm[a].distance(gb) for gb in train_utm.values())
        qc.loc[qc.poly_name == a, "min_dist_to_train_m"] = dmin
    # duplicate re-digitization: new face overlapping an existing v2 face by >50% of smaller -> exclude the NEW one
    for p in overlap_pairs:
        if p["type"] == "holdout-holdout" and p["frac_of_smaller"] > 0.5:
            src = dict(zip(qc.poly_name, qc.label_source))
            a, b = p["a"], p["b"]
            drop = b if src[a].startswith("existing") else (a if src[b].startswith("existing") else b)
            keep = a if drop == b else b
            same = cls[a] == cls[b]
            qc.loc[qc.poly_name == drop, "excluded"] = True
            qc.loc[qc.poly_name == drop, "exclude_reason"] = (
                f"duplicate_of_{keep}(overlap {p['frac_of_smaller']:.0%} of smaller; "
                + ("same class" if same else "CLASS CONFLICT") + ")")
    qc["overlap_holdout"] = qc.poly_name.map(lambda n: "; ".join(ov_hh.get(n, [])))
    qc["overlap_train"] = qc.poly_name.map(lambda n: "; ".join(ov_ht.get(n, [])))
    qc.to_csv(V3 / "QC_holdout_v3.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(OUT / "QC_holdout_v3.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlap_pairs).to_csv(V3 / "QC_overlaps_v3.csv", index=False, encoding="utf-8-sig")

    # manifest v3: train (v2, unchanged) + holdout v3
    tr = man2[(man2["split"] == "train") & (~man2["excluded"])].copy()
    tr_rows = pd.DataFrame({"name": tr["name"], "class_code": tr["class_code"], "county": tr["county"],
                            "area_ha_equirect_v2": tr["area_ha"],
                            "area_ha_utm44n": [utm(tk[n]).area / 1e4 for n in tr["name"]],
                            "cen_lon": tr["cen_lon"], "cen_lat": tr["cen_lat"],
                            "excluded": False, "exclude_reason": "", "split": "train",
                            "label_source": "v2_train"})
    ho = qc.copy()
    ho_rows = pd.DataFrame({"name": ho["poly_name"], "class_code": ho["class_code"], "county": "巩留县",
                            "area_ha_equirect_v2": np.nan, "area_ha_utm44n": ho["area_ha"],
                            "cen_lon": ho.get("cen_lon"), "cen_lat": ho.get("cen_lat"),
                            "excluded": ho["excluded"], "exclude_reason": ho["exclude_reason"],
                            "split": np.where(ho["excluded"], "excluded", "val_holdout"),
                            "label_source": ho["label_source"]})
    man3 = pd.concat([tr_rows, ho_rows], ignore_index=True)
    man3.to_csv(V3 / "split_manifest_v3.csv", index=False, encoding="utf-8-sig")
    man3.to_csv(OUT / "split_manifest_v3.csv", index=False, encoding="utf-8-sig")

    kept = qc[~qc.excluded]
    summary = {
        "n_parsed": int(len(qc)),
        "n_existing": int((qc.label_source.str.startswith("existing")).sum()),
        "n_new": int((qc.label_source.str.startswith("new")).sum()),
        "class_counts_all": {str(k): int(v) for k, v in qc.class_code.value_counts().items()},
        "class_counts_new": {str(k): int(v) for k, v in qc[qc.label_source.str.startswith("new")].class_code.value_counts().items()},
        "n_excluded": int(qc.excluded.sum()),
        "excluded": qc[qc.excluded][["poly_name", "class_code", "area_ha", "exclude_reason"]].to_dict("records"),
        "class_counts_kept": {str(k): int(v) for k, v in kept.class_code.value_counts().items()},
        "area_ha_kept_by_class": {str(k): float(v) for k, v in kept.groupby("class_code").area_ha.sum().items()},
        "n_overlap_holdout_holdout_pairs": int(sum(p["type"] == "holdout-holdout" for p in overlap_pairs)),
        "n_overlap_holdout_train_pairs": int(sum(p["type"] == "holdout-train" for p in overlap_pairs)),
        "overlap_pairs": overlap_pairs,
        "flagged_outside_gongliu_kept": qc[qc.qc_flags.fillna("").str.contains("outside")].poly_name.tolist(),
        "n_outside_gongliu_centroid": int((~qc.inside_gongliu.fillna(False).astype(bool)).sum()),
        "min_frac_inside_gongliu": float(qc.frac_inside_gongliu.min()),
        "min_dist_holdout_to_train_m": float(qc.min_dist_to_train_m.min()),
        "existing_iou_vs_v2_min": float(qc.iou_vs_v2_geom.min()) if "iou_vs_v2_geom" in qc else None,
        "n_train_polys": int(len(tr_rows)),
        "train_names_equal_v2": True,
        "geom_notes": qc[qc.geom_note.astype(str) != ""][["poly_name", "geom_note"]].to_dict("records"),
    }
    save_json(summary, V3 / "qc_summary_v3.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("overlap_pairs",)}, ensure_ascii=False, indent=1, default=str))
    return summary


# ------------------------------------------------------------------ sampling (same rules as v2)
def choose_n_points(area_ha: float, class_code: int) -> Tuple[int, float]:
    if class_code == 1:
        if area_ha < 0.3: return 3, 20.0
        if area_ha < 1.0: return 5, 25.0
        if area_ha < 3.0: return 8, 30.0
        if area_ha < 10.0: return 12, 35.0
        return 15, 40.0
    if area_ha < 0.5: return 3, 25.0
    if area_ha < 2.0: return 5, 30.0
    if area_ha < 10.0: return 8, 35.0
    return 10, 40.0


def sample_points_in_poly(geom, n_target, min_spacing_m, rng, max_tries=8000):
    minx, miny, maxx, maxy = geom.bounds
    lat0 = math.radians(geom.centroid.y)
    mlat, mlon = 111320.0, 111320.0 * math.cos(lat0)
    pts = []

    def fill(spacing):
        tries = 0
        while len(pts) < n_target and tries < max_tries:
            tries += 1
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if not geom.contains(Point(x, y)):
                continue
            if all(math.hypot((x - px) * mlon, (y - py) * mlat) >= spacing for px, py in pts):
                pts.append((x, y))
    fill(min_spacing_m)
    if len(pts) < max(1, n_target // 2):
        fill(min_spacing_m * 0.4)
    if not pts and geom.centroid.within(geom):
        pts.append((geom.centroid.x, geom.centroid.y))
    return pts


def stage_sample():
    qc = pd.read_csv(V3 / "QC_holdout_v3.csv")
    kept = qc[~qc.excluded]
    geoms = {}
    for r in read_kml_polys(NEW_KML):
        m = re.match(r"^\s*(YL-POLY-\d{4})\s*\|", r["raw_name"])
        pid = m.group(1) if m else r["raw_name"].strip()
        geoms[pid] = make_geom(r["coords"])[0]
    sp2 = pd.read_csv(SAMPLES_V2)
    rows = []
    reuse_log = []
    for _, r in kept.iterrows():
        name = r["poly_name"]
        iou = r.get("iou_vs_v2_geom", np.nan)
        old = sp2[(sp2.poly_name == name) & (sp2.split == "val_holdout")]
        if len(old) and pd.notna(iou) and iou > 0.99:
            o = old.copy()
            o["class_code"] = int(r["class_code"])
            o["point_source"] = "reused_v2"
            rows.append(o)
            reuse_log.append((name, len(o), "reused"))
            continue
        n, sp = choose_n_points(float(r["area_ha"]), int(r["class_code"]))
        n_adj = max(3 if r["area_ha"] >= 0.2 else 1, n)
        if int(r["class_code"]) == 0:
            n_adj = max(2 if r["area_ha"] >= 0.2 else 1, n)
        rng = np.random.default_rng((SEED + zlib.crc32(name.encode()) % 100000) % (2**31 - 1))
        pts = sample_points_in_poly(geoms[name], n_adj, sp, rng)
        rows.append(pd.DataFrame([{
            "sample_id": f"{name}-P{i:02d}", "poly_name": name, "class_code": int(r["class_code"]),
            "county": "巩留县", "split": "val_holdout", "lon": x, "lat": y, "area_ha": float(r["area_ha"]),
            "point_source": "new_v3"} for i, (x, y) in enumerate(pts, 1)]))
        reuse_log.append((name, len(pts), "new"))
    hold = pd.concat(rows, ignore_index=True)
    tr = sp2[sp2.split == "train"].copy()
    tr["point_source"] = "v2_train"
    allp = pd.concat([tr, hold], ignore_index=True)
    allp.to_csv(OUT / "sample_points_v3.csv", index=False, encoding="utf-8-sig")
    print(f"Holdout points: {len(hold)} across {hold.poly_name.nunique()} polys; train pts {len(tr)}")
    print(hold.groupby("class_code").agg(n_pts=("sample_id", "count"), n_poly=("poly_name", "nunique")))
    return allp


# ------------------------------------------------------------------ AE
def stage_ae(chunk=40):
    import ee
    ee.Initialize(project=PROJECT)
    ae_img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
              .filterDate("2021-01-01", "2022-01-01").mosaic())
    pts = pd.read_csv(OUT / "sample_points_v3.csv")
    done = {}
    for p in (AE_V2, AE_V3):
        if p.exists():
            for _, r in pd.read_csv(p).iterrows():
                done[r["sample_id"]] = r.to_dict()
    todo = pts[~pts.sample_id.isin(done)]
    print(f"AE cached {len(done)}; to fetch {len(todo)}")
    errors = []
    for i in range(0, len(todo), chunk):
        sub = todo.iloc[i:i + chunk]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                                              {"sid": str(r.sample_id)}) for r in sub.itertuples()])
        info = None
        for attempt in range(3):
            try:
                info = ae_img.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"]
                break
            except Exception as e:
                print(f"AE chunk {i} attempt {attempt} fail: {e}")
                time.sleep(3 * (attempt + 1))
        if info is None:
            errors.append(f"chunk@{i}")
            continue
        by = {f["properties"]["sid"]: f["properties"] for f in info}
        for r in sub.itertuples():
            pr = by.get(str(r.sample_id))
            if not pr:
                continue
            rec = {"sample_id": r.sample_id, "lon": r.lon, "lat": r.lat, "year": 2021}
            keys = sorted(k for k in pr if k.startswith("A") and k != "sid")
            for bi, k in enumerate(keys[:64]):
                rec[f"AE_{bi:02d}"] = pr[k]
            done[r.sample_id] = rec
        time.sleep(0.4)
        print(f"AE progress chunk {i//chunk+1}/{math.ceil(len(todo)/chunk)}")
    out = pd.DataFrame([done[s] for s in pts.sample_id if s in done])
    out.to_csv(AE_V3, index=False)
    missing = int((~pts.sample_id.isin(done)).sum())
    print(f"AE v3 rows {len(out)}; missing {missing}; errors {errors}")
    save_json({"ae_rows": len(out), "missing": missing, "errors": errors}, V3 / "ae_fetch_log_v3.json")


# ------------------------------------------------------------------ eval
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0, c - h), min(1, c + h))


def bin_metrics(y, pred):
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "OA": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "MCC": float(matthews_corrcoef(y, pred)),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def poly_bootstrap(py, pp, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(py); f1s, oas, bas, mccs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = py[idx], pp[idx]
        f1s.append(f1_score(a, b, zero_division=0)); oas.append(accuracy_score(a, b))
        if len(np.unique(a)) > 1:
            bas.append(balanced_accuracy_score(a, b)); mccs.append(matthews_corrcoef(a, b))
    q = lambda v: {"lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975)), "n_valid": len(v)}
    return {"F1": q(f1s), "OA": q(oas), "balanced_accuracy": q(bas), "MCC": q(mccs), "n_boot": n_boot}


def poly_table(hold):
    rows = []
    for name, s in hold.groupby("poly_name"):
        rows.append({"poly_name": name, "class_code": int(s.class_code.iloc[0]),
                     "y": int(s.class_code.iloc[0] == 1), "n_pts": len(s),
                     "pred_major": int(s.pred.mode().iloc[0]) if len(s) >= 3 else np.nan,
                     "pred_major_all": int(s.pred.mode().iloc[0]),
                     "tie": bool(len(s) >= 3 and (s.pred == 1).sum() * 2 == len(s)),
                     "frac_pred_apple": float(s.pred.mean()),
                     "mean_prob": float(s.prob.mean()),
                     "pred_meanprob": int(s.prob.mean() >= 0.5),
                     "cen_lon": float(s.lon.mean()), "cen_lat": float(s.lat.mean()),
                     "area_ha": float(s.area_ha.iloc[0]),
                     "is_v2": s.point_source.iloc[0] == "reused_v2"})
    return pd.DataFrame(rows)


def poly_block(pt, col):
    d = pt.dropna(subset=[col])
    py, pp = d.y.values.astype(int), d[col].values.astype(int)
    m = bin_metrics(py, pp)
    k = int((py == pp).sum())
    lo, hi = wilson(k, len(py))
    m["wilson_95ci_OA"] = {"lo": lo, "hi": hi, "k_correct": k, "n": int(len(py))}
    m["bootstrap_95ci"] = poly_bootstrap(py, pp)
    # per-class
    pc = {}
    for c, lab in [(1, "apple_recall"), (2, "other_orchard_rejected_rate"), (0, "non_orchard_rejected_rate")]:
        s = d[d.class_code == c]
        if len(s) == 0:
            pc[lab] = None; continue
        val = (s[col] == 1).mean() if c == 1 else (s[col] == 0).mean()
        kk = int(round(val * len(s)))
        lo, hi = wilson(kk, len(s))
        pc[lab] = {"rate": float(val), "k": kk, "n": int(len(s)), "wilson_lo": lo, "wilson_hi": hi}
    m["per_class"] = pc
    m["three_class_by_pred"] = {str(c): {"pred_apple": int(((d.class_code == c) & (d[col] == 1)).sum()),
                                         "pred_non": int(((d.class_code == c) & (d[col] == 0)).sum())}
                                for c in (1, 2, 0)}
    return m


def stage_eval():
    pts = pd.read_csv(OUT / "sample_points_v3.csv")
    ae = pd.read_csv(AE_V3)
    b = joblib.load(MODEL); clf = b["model"]; feats = b["features"]
    df = pts.merge(ae[["sample_id"] + AE_COLS], on="sample_id", how="left")
    ok = df[AE_COLS].notna().all(axis=1)
    hold = df[(df.split == "val_holdout") & ok].copy()
    dropped = df[(df.split == "val_holdout") & ~ok]
    hold["y"] = (hold.class_code == 1).astype(int)
    hold["prob"] = clf.predict_proba(hold[feats].values.astype(float))[:, 1]
    hold["pred"] = (hold["prob"] >= 0.5).astype(int)
    assert (hold["pred"].values == clf.predict(hold[feats].values.astype(float))).all()
    hold.drop(columns=AE_COLS).to_csv(V3 / "predictions_holdout_v3.csv", index=False, encoding="utf-8-sig")

    pt = poly_table(hold)
    pt.to_csv(V3 / "poly_predictions_v3.csv", index=False, encoding="utf-8-sig")
    res = {"version": "v3", "model": str(MODEL), "model_reused": True,
           "binary_definition": "apple(1) vs non-apple(0 non-orchard + 2 other orchard)",
           "n_holdout_points_dropped_no_AE": int(len(dropped)),
           "polys_with_lt3_points": pt[pt.n_pts < 3].poly_name.tolist(),
           "majority_ties": pt[pt.tie].poly_name.tolist()}
    res["poly_majority"] = poly_block(pt, "pred_major")
    res["poly_meanprob"] = poly_block(pt, "pred_meanprob")

    y, p, pr = hold.y.values, hold.pred.values, hold.prob.values
    pm = bin_metrics(y, p)
    pm["AP"] = float(average_precision_score(y, pr)); pm["AUC"] = float(roc_auc_score(y, pr))
    pm["per_class_point"] = {str(c): {"n": int((hold.class_code == c).sum()),
                                      "pred_apple_rate": float(hold[hold.class_code == c].pred.mean())}
                             for c in (1, 2, 0)}
    # cluster (polygon) bootstrap for point F1 / AP
    rng = np.random.default_rng(SEED); names = pt.poly_name.values; f1s, aps = [], []
    grp = {n: hold.index[hold.poly_name == n].values for n in names}
    for _ in range(1000):
        idx = np.concatenate([grp[n] for n in rng.choice(names, len(names))])
        yy = hold.loc[idx, "y"].values
        f1s.append(f1_score(yy, hold.loc[idx, "pred"].values, zero_division=0))
        if 0 < yy.sum() < len(yy):
            aps.append(average_precision_score(yy, hold.loc[idx, "prob"].values))
    pm["cluster_bootstrap_95ci"] = {"F1": [float(np.quantile(f1s, .025)), float(np.quantile(f1s, .975))],
                                    "AP": [float(np.quantile(aps, .025)), float(np.quantile(aps, .975))]}
    res["point_level"] = pm

    # v2 subset (same 18 polys) for sanity
    v2p = pt[pt.is_v2].dropna(subset=["pred_major"])
    if len(v2p):
        res["v2_subset_recomputed"] = {"n_polys": int(len(v2p)),
                                       "poly_majority": bin_metrics(v2p.y, v2p.pred_major.astype(int)),
                                       "point": bin_metrics(hold[hold.point_source == "reused_v2"].y,
                                                            hold[hold.point_source == "reused_v2"].pred)}
    # new-only
    nw = pt[~pt.is_v2]
    res["new_only"] = poly_block(nw, "pred_major")
    # sensitivities
    res["sens_all_polys_majority_incl_lt3pts"] = poly_block(pt, "pred_major_all")
    strict = pt[~pt.poly_name.isin(["YL-POLY-0059", "YL-POLY-0061"])]
    res["sens_strict_inside_gongliu"] = poly_block(strict, "pred_major")
    res["n_polys_primary"] = int(pt.pred_major.notna().sum())
    res["primary_class_counts"] = {str(k): int(v) for k, v in pt.dropna(subset=["pred_major"]).class_code.value_counts().items()}
    save_json(res, V3 / "metrics_poly_holdout_v3.json")
    save_json(res, OUT / "metrics_poly_holdout_v3.json")
    print(json.dumps({k: res[k] for k in ("poly_majority", "point_level")}, ensure_ascii=False, indent=1)[:4000])
    return res


if __name__ == "__main__":
    stages = sys.argv[1:] or ["qc", "sample", "ae", "eval"]
    for s in stages:
        print(f"===== {s} =====")
        {"qc": stage_qc, "sample": stage_sample, "ae": stage_ae, "eval": stage_eval}[s]()
