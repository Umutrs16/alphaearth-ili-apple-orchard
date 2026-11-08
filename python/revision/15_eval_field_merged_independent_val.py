# -*- coding: utf-8 -*-
"""Merge field-survey GPS into Gongliu/Ili independent validation; recalculate B3.

Scientifically honest protocol:
- Do NOT treat unspecified orchard as apple.
- Deduplicate field points within ~20 m of each other and against visual apple pts.
- Report Gongliu visual baseline unchanged; add Gongliu field apple (3);
  Ili field apple (31) recall + mixed-domain table with visual negatives;
  orchard binary with Gongliu unspecified orchard as orchard-positives.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "revision"))

from common import FIGURES, METRICS, MODELS, TABLES, ensure_out_dirs, save_json  # noqa: E402
from utils import compute_metrics  # noqa: E402

KIT = ROOT / "data" / "interpretation_kit"
FIELD = ROOT / "data" / "field_survey" / "usable_candidates"
OUT = ROOT / "outputs" / "revision" / "independent_val"
FILLED = KIT / "Gongliu_independent_val_判读表_filled.xlsx"
KIT_AE = OUT / "kit_ae_features_2021.csv"
FIELD_AE = OUT / "field_ae_features_2021.csv"
PROJECT = "hallowed-chain-458005-g9"
DEDUP_M = 20.0
YEAR = 2021

CLASS_MAP = {
    1: "apple_orchard",
    2: "other_orchard",
    3: "cropland",
    4: "forest",
    5: "shrub_grass",
    6: "built_up",
    7: "water",
    8: "bare_other",
    9: "uncertain",
}


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [np.asarray(lon1, float), np.asarray(lat1, float),
                                               np.asarray(lon2, float), np.asarray(lat2, float)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def dedup_within(df: pd.DataFrame, lon_col="lon", lat_col="lat", thresh_m=DEDUP_M) -> pd.DataFrame:
    """Greedy keep-first dedup within thresh_m (stable by input order)."""
    if len(df) == 0:
        return df.copy()
    keep = []
    lons = df[lon_col].values.astype(float)
    lats = df[lat_col].values.astype(float)
    for i in range(len(df)):
        if not keep:
            keep.append(i)
            continue
        d = haversine_m(lons[i], lats[i], lons[keep], lats[keep])
        if np.min(d) >= thresh_m:
            keep.append(i)
    out = df.iloc[keep].copy()
    out["dedup_kept"] = True
    return out.reset_index(drop=True)


def drop_near_others(df: pd.DataFrame, ref: pd.DataFrame, lon="lon", lat="lat",
                     thresh_m=DEDUP_M, tag="near_visual_apple") -> tuple[pd.DataFrame, int]:
    if len(df) == 0 or len(ref) == 0:
        return df.copy(), 0
    rlons, rlats = ref[lon].values.astype(float), ref[lat].values.astype(float)
    keep_mask = []
    n_drop = 0
    for _, r in df.iterrows():
        d = haversine_m(float(r[lon]), float(r[lat]), rlons, rlats)
        if np.min(d) < thresh_m:
            keep_mask.append(False)
            n_drop += 1
        else:
            keep_mask.append(True)
    out = df.loc[keep_mask].copy().reset_index(drop=True)
    return out, n_drop


def load_visual() -> pd.DataFrame:
    df = pd.read_excel(FILLED, sheet_name="判读填写")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["interp_class_code"] = pd.to_numeric(df["interp_class_code"], errors="coerce")
    df["is_apple"] = pd.to_numeric(df["is_apple"], errors="coerce")
    df["confidence"] = df["confidence"].astype(str).str.strip().replace({"nan": np.nan})
    ef = df["exclude_flag"]
    df["exclude_flag"] = ef.notna() & (ef.astype(str).str.strip() != "") & (
        ~ef.astype(str).str.strip().str.lower().isin(["0", "false", "no", "nan"])
    )
    df["interp_class"] = df["interp_class_code"].map(CLASS_MAP)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["source"] = "visual_gongliu"
    df["label_apple"] = (df["interp_class_code"] == 1).astype(int)
    df["label_orchard"] = df["interp_class_code"].isin([1, 2]).astype(int)
    df["county"] = df["county"].astype(str)
    return df


def _field_base(path: Path, source: str, label_apple: int, label_orchard: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    for i, r in raw.iterrows():
        lon = float(r["_lon"] if pd.notna(r.get("_lon")) else r["经度"])
        lat = float(r["_lat"] if pd.notna(r.get("_lat")) else r["纬度"])
        county = str(r.get("county", ""))
        text = str(r.get("text", r.get("备注1", "")))
        sid = f"{source}_{i:04d}_{lon:.5f}_{lat:.5f}"
        rows.append(
            {
                "sample_id": sid,
                "lon": lon,
                "lat": lat,
                "county": county,
                "text": text,
                "source": source,
                "label_apple": int(label_apple),
                "label_orchard": int(label_orchard),
                "interp_class_code": 1 if label_apple == 1 else (2 if label_orchard == 1 else np.nan),
                "is_apple": float(label_apple) if label_apple in (0, 1) else np.nan,
                "confidence": "高",  # field GPS treated as high for inclusion
                "exclude_flag": False,
                "stratum": "FIELD",
                "year_ref": "field_survey_multi_year",
            }
        )
    return pd.DataFrame(rows)


def load_field_sets() -> dict[str, pd.DataFrame]:
    apple_ili = _field_base(
        FIELD / "ili_field_apple_usable.csv", "field_apple_ili", label_apple=1, label_orchard=1
    )
    # Gongliu explicit apples are subset of ili usable (county=巩留县)
    apple_gl = apple_ili[apple_ili["county"].astype(str).str.contains("巩留")].copy()
    apple_gl["source"] = "field_apple_gongliu"
    apple_gl["sample_id"] = apple_gl["sample_id"].str.replace("field_apple_ili", "field_apple_gongliu", regex=False)

    orch_gl = _field_base(
        FIELD / "gongliu_field_orchard_unspecified.csv",
        "field_orchard_gongliu",
        label_apple=0,  # NOT apple — unknown species; used only for orchard binary
        label_orchard=1,
    )
    # Prefer exclude from apple binary: set label_apple to NaN marker via separate flag
    orch_gl["label_apple"] = -1  # sentinel: exclude from apple-vs-nonapple
    orch_gl["is_apple"] = np.nan
    orch_gl["interp_class_code"] = 2  # other_orchard (species unspecified)

    # optional ili unspecified (not primary for Gongliu orchard table)
    orch_ili_path = FIELD / "ili_field_orchard_unspecified.csv"
    orch_ili = None
    if orch_ili_path.exists():
        orch_ili = _field_base(orch_ili_path, "field_orchard_ili", label_apple=-1, label_orchard=1)
        orch_ili["label_apple"] = -1
        orch_ili["is_apple"] = np.nan
        orch_ili["interp_class_code"] = 2

    return {
        "field_apple_ili": apple_ili,
        "field_apple_gongliu": apple_gl.reset_index(drop=True),
        "field_orchard_gongliu": orch_gl,
        "field_orchard_ili": orch_ili,
    }


def sample_ae_features(df: pd.DataFrame, cache_path: Path, year: int = YEAR, chunk: int = 25) -> pd.DataFrame:
    """Sample AlphaEarth annual embeddings; band rename AE_00..AE_63; chunked getInfo."""
    import ee

    ee.Initialize(project=PROJECT)
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    done = {}
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        for _, r in cached.iterrows():
            done[str(r["sample_id"])] = r.to_dict()
        print(f"AE cache loaded ({cache_path.name}): {len(done)}")

    todo = df[~df["sample_id"].astype(str).isin(done)].copy()
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
            print(f"AE chunk fail @ {i}: {e}; sleep+retry")
            time.sleep(5)
            try:
                info = ae.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"]
            except Exception as e2:
                print(f"AE chunk fail again: {e2}")
                # try one-by-one
                info = []
                for _, r in sub.iterrows():
                    try:
                        one = ee.FeatureCollection(
                            [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"sid": str(r.sample_id)})]
                        )
                        info.extend(ae.sampleRegions(collection=one, scale=10, geometries=False).getInfo()["features"])
                        time.sleep(0.4)
                    except Exception as e3:
                        print(f"  single fail {r.sample_id}: {e3}")
        by_sid = {f["properties"]["sid"]: f["properties"] for f in info}
        for _, r in sub.iterrows():
            props = by_sid.get(str(r.sample_id))
            if not props:
                continue
            rec = {"sample_id": str(r.sample_id), "lon": float(r.lon), "lat": float(r.lat), "year": year}
            emb_keys = sorted([k for k in props.keys() if k.startswith("A") and k != "sid"])
            for bi, k in enumerate(emb_keys[:64]):
                rec[f"AE_{bi:02d}"] = props[k]
            for bi in range(64):
                rec.setdefault(f"AE_{bi:02d}", np.nan)
            done[str(r.sample_id)] = rec
        pd.DataFrame(list(done.values())).to_csv(cache_path, index=False)
        print(f"AE progress {len(done)}/{len(df)}")
        time.sleep(0.35)
    out = pd.DataFrame(list(done.values()))
    out.to_csv(cache_path, index=False)
    return out


def attach_grid_fallback(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    gp_path = ROOT / "outputs" / "revision" / "maps" / "grid_pred_2021.csv"
    if not gp_path.exists():
        print("No grid_pred_2021.csv; skip grid fallback")
        return df
    gp = pd.read_csv(gp_path)
    tree = cKDTree(np.c_[gp.lon.values, gp.lat.values])
    d, idx = tree.query(np.c_[df.lon.values, df.lat.values], k=1)
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    out = df.copy()
    for c in ae_cols + ["prob", "pred"]:
        if c in gp.columns:
            out[f"grid_{c}"] = gp.iloc[idx][c].values
    out["grid_nn_m"] = [
        float(haversine_m(lo, la, gp.iloc[j].lon, gp.iloc[j].lat))
        for lo, la, j in zip(df.lon.values, df.lat.values, idx)
    ]
    return out


def metrics_pack(y, pred, prob) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    prob = np.asarray(prob).astype(float) if prob is not None else None
    if len(y) == 0:
        return {
            "OA": float("nan"), "Precision": float("nan"), "Recall": float("nan"),
            "F1": float("nan"), "IoU": float("nan"), "Kappa": float("nan"), "AUC": float("nan"),
            "UA_positive": float("nan"), "PA_positive": float("nan"),
            "UA_negative": float("nan"), "PA_negative": float("nan"),
            "TP": 0, "FP": 0, "FN": 0, "TN": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "n": 0, "n_pos": 0, "n_neg": 0,
        }
    m = compute_metrics(y, pred, prob if (prob is not None and len(np.unique(y)) > 1) else None)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    ua_pos = tp / (tp + fp) if (tp + fp) else float("nan")
    pa_pos = tp / (tp + fn) if (tp + fn) else float("nan")
    ua_neg = tn / (tn + fn) if (tn + fn) else float("nan")
    pa_neg = tn / (tn + fp) if (tn + fp) else float("nan")
    m.update(
        {
            "UA_positive": float(ua_pos),
            "PA_positive": float(pa_pos),
            "UA_negative": float(ua_neg),
            "PA_negative": float(pa_neg),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "n": int(len(y)),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
        }
    )
    return m


def recall_only(y_pos_prob, y_pos_pred) -> dict:
    """Detection rate / recall on positives-only set (no matched negatives)."""
    pred = np.asarray(y_pos_pred).astype(int)
    prob = np.asarray(y_pos_prob).astype(float)
    n = len(pred)
    tp = int((pred == 1).sum())
    fn = int((pred == 0).sum())
    return {
        "n_pos": n,
        "n_detected_pred1": tp,
        "n_missed_pred0": fn,
        "detection_rate_recall": float(tp / n) if n else float("nan"),
        "prob_mean": float(np.nanmean(prob)) if n else float("nan"),
        "prob_median": float(np.nanmedian(prob)) if n else float("nan"),
        "prob_min": float(np.nanmin(prob)) if n else float("nan"),
        "prob_max": float(np.nanmax(prob)) if n else float("nan"),
        "note": "Positives-only; domain-mismatched negatives avoided. Not OA/F1.",
    }


def confident_visual(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (~df["exclude_flag"].astype(bool))
        & (df["interp_class_code"] != 9)
        & (df["confidence"].isin(["高", "中"]))
        & df["prob_B3"].notna()
    ].copy()


def flat_row(task: str, m: dict, extra: dict | None = None) -> dict:
    row = {
        "task": task,
        "model": "B3_AE",
        "n": m.get("n"),
        "n_pos": m.get("n_pos"),
        "n_neg": m.get("n_neg"),
        "OA": m.get("OA"),
        "UA": m.get("UA_positive"),
        "PA": m.get("PA_positive"),
        "Precision": m.get("Precision"),
        "Recall": m.get("Recall"),
        "F1": m.get("F1"),
        "Kappa": m.get("Kappa"),
        "IoU": m.get("IoU"),
        "AUC": m.get("AUC"),
        "TP": m.get("TP"),
        "FP": m.get("FP"),
        "FN": m.get("FN"),
        "TN": m.get("TN"),
    }
    if extra:
        row.update(extra)
    return row


def plot_confusion(results_map: dict, out_path: Path, titles: list[tuple[str, str]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, titles):
        m = results_map[key]
        cm = np.array(
            [
                [m["confusion_matrix"]["TN"], m["confusion_matrix"]["FP"]],
                [m["confusion_matrix"]["FN"], m["confusion_matrix"]["TP"]],
            ]
        )
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{title}\nOA={m['OA']:.3f} F1={m['F1']:.3f} n={m['n']} pos={m['n_pos']}")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    ensure_out_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    dedup_log = {}

    # --- visual ---
    visual = load_visual()
    print(f"visual loaded n={len(visual)}")

    # --- field ---
    fields = load_field_sets()
    fa_ili_raw = fields["field_apple_ili"]
    fa_gl_raw = fields["field_apple_gongliu"]
    fo_gl_raw = fields["field_orchard_gongliu"]
    print(
        f"field apple ili={len(fa_ili_raw)} gongliu={len(fa_gl_raw)}; "
        f"orchard gongliu unspecified={len(fo_gl_raw)}"
    )

    # Dedup field within 20 m
    fa_ili = dedup_within(fa_ili_raw)
    fa_gl = dedup_within(fa_gl_raw)
    fo_gl = dedup_within(fo_gl_raw)
    dedup_log["field_apple_ili_before"] = len(fa_ili_raw)
    dedup_log["field_apple_ili_after_self"] = len(fa_ili)
    dedupum_dropped_ili_self = len(fa_ili_raw) - len(fa_ili)
    dedup_log["field_apple_gongliu_before"] = len(fa_gl_raw)
    dedup_log["field_apple_gongliu_after_self"] = len(fa_gl)
    dedup_log["field_orchard_gongliu_before"] = len(fo_gl_raw)
    dedup_log["field_orchard_gongliu_after_self"] = len(fo_gl)

    # Against visual apple points (code==1 confident-ish: all code==1)
    vis_apple = visual[visual["interp_class_code"] == 1][["lon", "lat", "sample_id"]].copy()
    fa_ili, n_drop_ili = drop_near_others(fa_ili, vis_apple)
    fa_gl, n_drop_gl = drop_near_others(fa_gl, vis_apple)
    # orchard unspecified vs visual orchard (1+2) optional soft dedup
    vis_orch = visual[visual["interp_class_code"].isin([1, 2])][["lon", "lat"]].copy()
    fo_gl, n_drop_fo = drop_near_others(fo_gl, vis_orch)
    dedup_log["field_apple_ili_dropped_near_visual_apple"] = n_drop_ili
    dedup_log["field_apple_gongliu_dropped_near_visual_apple"] = n_drop_gl
    dedup_log["field_orchard_gongliu_dropped_near_visual_orchard"] = n_drop_fo
    dedup_log["field_apple_ili_final"] = len(fa_ili)
    dedup_log["field_apple_gongliu_final"] = len(fa_gl)
    dedup_log["field_orchard_gongliu_final"] = len(fo_gl)
    print("dedup:", json.dumps(dedup_log, ensure_ascii=False))

    # County breakdown after dedup
    county_counts = fa_ili["county"].value_counts().to_dict() if len(fa_ili) else {}

    # --- AE sampling for NEW field points ---
    field_all = pd.concat([fa_ili, fo_gl], ignore_index=True)
    # also include gongliu apples if any not in ili list (should be subset)
    extra_gl = fa_gl[~fa_gl["sample_id"].isin(fa_ili["sample_id"])]
    if len(extra_gl):
        field_all = pd.concat([field_all, extra_gl], ignore_index=True)

    ae_source = "gee_point_sample_2021"
    try:
        field_ae = sample_ae_features(field_all, FIELD_AE, year=YEAR, chunk=20)
    except Exception as e:
        print("GEE field AE sampling FAILED:", e)
        save_json(
            {"blocker": "GEE AE sampling failed for field points", "error": str(e), "dedup": dedup_log},
            OUT / "BLOCKER_field_ae.json",
        )
        raise SystemExit(f"BLOCKER: GEE failed: {e}")

    # Kit AE reuse
    if not KIT_AE.exists():
        raise SystemExit("BLOCKER: kit_ae_features_2021.csv missing")
    kit_ae = pd.read_csv(KIT_AE)
    print(f"kit AE {len(kit_ae)}, field AE cache {len(field_ae)}")

    ae_cols = [f"AE_{i:02d}" for i in range(64)]

    def merge_ae(pts: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
        cols = ["sample_id"] + ae_cols
        have = [c for c in cols if c in ae.columns]
        m = pts.merge(ae[have], on="sample_id", how="left")
        return m

    visual_m = merge_ae(visual, kit_ae)
    fa_ili_m = merge_ae(fa_ili, field_ae)
    fa_gl_m = merge_ae(fa_gl, field_ae)
    fo_gl_m = merge_ae(fo_gl, field_ae)

    # Grid fallback for any missing
    for name, dref in [("visual", visual_m), ("fa_ili", fa_ili_m), ("fa_gl", fa_gl_m), ("fo_gl", fo_gl_m)]:
        miss = dref[ae_cols[0]].isna().sum() if ae_cols[0] in dref.columns else len(dref)
        print(f"missing AE {name}: {miss}")

    combined_pts = pd.concat([visual_m, fa_ili_m, fo_gl_m], ignore_index=True, sort=False)
    # avoid double-counting Gongliu apples: they are already in fa_ili_m
    combined_pts = attach_grid_fallback(combined_pts)
    missing_ae = combined_pts[ae_cols[0]].isna() if ae_cols[0] in combined_pts.columns else pd.Series(True, index=combined_pts.index)
    n_missing = int(missing_ae.sum())
    if n_missing:
        print(f"Filling {n_missing} from grid NN")
        for c in ae_cols:
            gc = f"grid_{c}"
            if gc in combined_pts.columns:
                if c not in combined_pts.columns:
                    combined_pts[c] = np.nan
                combined_pts.loc[missing_ae, c] = combined_pts.loc[missing_ae, gc].values
        ae_source = ae_source + "+grid_nn_fallback"
    combined_pts["ae_source"] = np.where(missing_ae, "grid_nn", "gee_point")

    # Predict B3
    bundle = joblib.load(MODELS / "B3_full_2019_2021.joblib")
    clf, cols = bundle["model"], bundle["cols"]
    X = combined_pts[cols].astype(float).copy()
    X = X.fillna(X.median(numeric_only=True))
    prob = clf.predict_proba(X.values)[:, 1]
    pred = (prob >= 0.5).astype(int)
    combined_pts["prob_B3"] = prob
    combined_pts["pred_B3"] = pred

    # Re-split after prediction for metric subsets
    vis_pred = combined_pts[combined_pts["source"] == "visual_gongliu"].copy()
    fa_ili_pred = combined_pts[combined_pts["source"] == "field_apple_ili"].copy()
    fo_gl_pred = combined_pts[combined_pts["source"] == "field_orchard_gongliu"].copy()
    fa_gl_pred = fa_ili_pred[fa_ili_pred["county"].astype(str).str.contains("巩留")].copy()

    conf = confident_visual(vis_pred)
    print(f"confident visual n={len(conf)} apple_code1={(conf['interp_class_code']==1).sum()}")

    results = {
        "model": "B3_full_2019_2021",
        "ae_year": YEAR,
        "ae_source": ae_source,
        "dedup_m": DEDUP_M,
        "dedup_log": dedup_log,
        "field_apple_by_county": county_counts,
        "protocol": {
            "unspecified_orchard_in_apple_binary": "EXCLUDED (label_apple=-1)",
            "unspecified_orchard_in_orchard_binary": "INCLUDED as orchard-positive",
            "field_points_note": "Multi-year survey GPS; NOT imagery-year-locked to 2021.",
            "visual_confident": "exclude exclude_flag OR code=9 OR confidence=低",
        },
    }

    flat_rows = []

    # ========== 1. Gongliu visual baseline (unchanged protocol) ==========
    y_a = (conf["interp_class_code"] == 1).astype(int).values
    pred_v = conf["pred_B3"].astype(int).values
    prob_v = conf["prob_B3"].astype(float).values
    m_vis_apple = metrics_pack(y_a, pred_v, prob_v)
    results["visual_gongliu_apple_vs_nonapple_code1"] = m_vis_apple
    flat_rows.append(flat_row("visual_gongliu_apple_code1", m_vis_apple))

    y_o = conf["interp_class_code"].isin([1, 2]).astype(int).values
    m_vis_orch = metrics_pack(y_o, pred_v, prob_v)
    results["visual_gongliu_orchard_vs_nonorchard"] = m_vis_orch
    flat_rows.append(flat_row("visual_gongliu_orchard", m_vis_orch))

    # is_apple secondary
    conf_ia = conf[conf["is_apple"].notna()].copy()
    m_vis_ia = metrics_pack(
        conf_ia["is_apple"].astype(int).values,
        conf_ia["pred_B3"].astype(int).values,
        conf_ia["prob_B3"].astype(float).values,
    )
    results["visual_gongliu_apple_vs_nonapple_is_apple"] = m_vis_ia
    flat_rows.append(flat_row("visual_gongliu_apple_is_apple", m_vis_ia))

    # ========== 2. Gongliu visual + Gongliu field apple (3) ==========
    # positives: visual code=1 UNION field Gongliu apple
    # negatives: visual confident non-apple (code!=1); field unspecified EXCLUDED
    vis_pos = conf[conf["interp_class_code"] == 1].copy()
    vis_neg = conf[conf["interp_class_code"] != 1].copy()
    gl_pos = fa_gl_pred.copy()
    gl_pos["y"] = 1
    vis_pos["y"] = 1
    vis_neg["y"] = 0
    comb_gl = pd.concat([vis_pos, gl_pos, vis_neg], ignore_index=True, sort=False)
    # drop rows without pred
    comb_gl = comb_gl[comb_gl["prob_B3"].notna()]
    m_gl_comb = metrics_pack(
        comb_gl["y"].astype(int).values,
        comb_gl["pred_B3"].astype(int).values,
        comb_gl["prob_B3"].astype(float).values,
    )
    m_gl_comb["n_field_apple_added"] = int(len(gl_pos))
    m_gl_comb["n_visual_pos"] = int(len(vis_pos))
    m_gl_comb["n_visual_neg"] = int(len(vis_neg))
    results["gongliu_visual_plus_field_apple"] = m_gl_comb
    flat_rows.append(flat_row("gongliu_visual_plus_field_apple", m_gl_comb,
                              {"n_field_apple_added": m_gl_comb["n_field_apple_added"]}))

    # ========== 3. Ili field apple (31) recall alone ==========
    ili_pos = fa_ili_pred[fa_ili_pred["prob_B3"].notna()].copy()
    rec_ili = recall_only(ili_pos["prob_B3"].values, ili_pos["pred_B3"].values)
    rec_ili["by_county"] = (
        ili_pos.groupby("county")
        .apply(lambda g: {
            "n": int(len(g)),
            "detected": int((g["pred_B3"] == 1).sum()),
            "recall": float((g["pred_B3"] == 1).mean()) if len(g) else float("nan"),
            "prob_mean": float(g["prob_B3"].mean()),
        }, include_groups=False)
        .to_dict()
    )
    results["ili_field_apple_detection_rate"] = rec_ili
    flat_rows.append({
        "task": "ili_field_apple_detection_rate",
        "model": "B3_AE",
        "n": rec_ili["n_pos"],
        "n_pos": rec_ili["n_pos"],
        "n_neg": 0,
        "OA": None, "UA": None,
        "PA": rec_ili["detection_rate_recall"],
        "Precision": None,
        "Recall": rec_ili["detection_rate_recall"],
        "F1": None, "Kappa": None, "IoU": None, "AUC": None,
        "TP": rec_ili["n_detected_pred1"], "FP": None,
        "FN": rec_ili["n_missed_pred0"], "TN": None,
        "note": "positives-only detection rate",
    })

    # ========== 4. Mixed-domain: Ili field apple positives + Gongliu visual non-apple negatives ==========
    ili_pos2 = ili_pos.copy()
    ili_pos2["y"] = 1
    vis_neg2 = vis_neg.copy()
    vis_neg2["y"] = 0
    mixed = pd.concat([ili_pos2, vis_neg2], ignore_index=True, sort=False)
    mixed = mixed[mixed["prob_B3"].notna()]
    m_mixed = metrics_pack(
        mixed["y"].astype(int).values,
        mixed["pred_B3"].astype(int).values,
        mixed["prob_B3"].astype(float).values,
    )
    m_mixed["domain_note"] = (
        "Positives=Ili field apple GPS (multi-county); negatives=Gongliu visual confident non-apple. "
        "Mixed spatial domain — interpret cautiously; prefer detection-rate for field apples alone."
    )
    m_mixed["n_field_pos"] = int(len(ili_pos2))
    m_mixed["n_visual_neg"] = int(len(vis_neg2))
    results["ili_field_apple_plus_visual_gongliu_negatives_mixed_domain"] = m_mixed
    flat_rows.append(flat_row("ili_field_plus_visual_neg_mixed", m_mixed))

    # ========== 5. Recommended primary: Gongliu visual UNION all Ili field apples (as Ili supplementary) ==========
    # Actually protocol says:
    # "recommended primary combined for paper: visual confident Gongliu UNION field apple points
    #  that fall in Gongliu OR all Ili field apples as an Ili field supplementary set"
    # So report BOTH:
    #  (a) visual + Gongliu field apple  [already #2]
    #  (b) visual + ALL Ili field apples as supplementary mixed  [similar to #4 but include visual pos too]
    vis_pos3 = vis_pos.copy(); vis_pos3["y"] = 1
    ili_all = ili_pos.copy(); ili_all["y"] = 1
    vis_neg3 = vis_neg.copy(); vis_neg3["y"] = 0
    # Dedup: if Gongliu field apple already near visual pos, already dropped; union visual pos + all ili
    # Avoid double-counting: Gongliu field apples that are in both vis_pos and ili — ili are field, vis are visual
    primary = pd.concat([vis_pos3, ili_all, vis_neg3], ignore_index=True, sort=False)
    primary = primary[primary["prob_B3"].notna()]
    m_primary = metrics_pack(
        primary["y"].astype(int).values,
        primary["pred_B3"].astype(int).values,
        primary["prob_B3"].astype(float).values,
    )
    m_primary["composition"] = {
        "visual_apple_pos": int(len(vis_pos3)),
        "ili_field_apple_pos": int(len(ili_all)),
        "visual_nonapple_neg": int(len(vis_neg3)),
        "note": "Ili field apples span Tekes/Yining/Gongliu/Xinyuan — mixed domain with Gongliu visual negatives.",
    }
    results["recommended_primary_visual_plus_ili_field_apple"] = m_primary
    flat_rows.append(flat_row("recommended_primary_visual_plus_ili_field", m_primary))

    # ========== 6. Orchard: visual (1+2) + Gongliu field unspecified orchard ==========
    vis_orch_pos = conf[conf["interp_class_code"].isin([1, 2])].copy()
    vis_orch_neg = conf[~conf["interp_class_code"].isin([1, 2])].copy()
    fo_pos = fo_gl_pred[fo_gl_pred["prob_B3"].notna()].copy()
    vis_orch_pos["y"] = 1
    fo_pos["y"] = 1
    vis_orch_neg["y"] = 0
    orch_comb = pd.concat([vis_orch_pos, fo_pos, vis_orch_neg], ignore_index=True, sort=False)
    m_orch = metrics_pack(
        orch_comb["y"].astype(int).values,
        orch_comb["pred_B3"].astype(int).values,
        orch_comb["prob_B3"].astype(float).values,
    )
    m_orch["n_field_orchard_unspecified_added"] = int(len(fo_pos))
    m_orch["n_visual_orchard_pos"] = int(len(vis_orch_pos))
    m_orch["n_visual_nonorchard_neg"] = int(len(vis_orch_neg))
    m_orch["note"] = "Field unspecified orchard = orchard-positive only; EXCLUDED from apple binary."
    results["orchard_visual_plus_gongliu_field_unspecified"] = m_orch
    flat_rows.append(flat_row("orchard_visual_plus_gongliu_field", m_orch))

    # Gongliu field apple detection alone
    if len(fa_gl_pred):
        rec_gl = recall_only(fa_gl_pred["prob_B3"].values, fa_gl_pred["pred_B3"].values)
        results["gongliu_field_apple_detection_rate"] = rec_gl
        flat_rows.append({
            "task": "gongliu_field_apple_detection_rate",
            "model": "B3_AE",
            "n": rec_gl["n_pos"], "n_pos": rec_gl["n_pos"], "n_neg": 0,
            "OA": None, "UA": None, "PA": rec_gl["detection_rate_recall"],
            "Precision": None, "Recall": rec_gl["detection_rate_recall"],
            "F1": None, "Kappa": None, "IoU": None, "AUC": None,
            "TP": rec_gl["n_detected_pred1"], "FP": None,
            "FN": rec_gl["n_missed_pred0"], "TN": None,
        })

    results["honesty"] = (
        "AOMC/WorldCover semi-auto independent_val_B3.json = weak-label fit, NOT independent truth. "
        "Gongliu visual = human interpretation independent set. "
        "Field GPS = multi-year survey supplementary positives (not imagery-year-locked). "
        "Unspecified orchard GPS excluded from apple-vs-nonapple; included only in orchard-vs-nonorchard. "
        f"Dedup within {DEDUP_M} m self and vs visual apple/orchard. AE: {ae_source}."
    )

    # --- Save joined ---
    # Tag evaluation membership for transparency
    joined = combined_pts.copy()
    out_csv = OUT / "combined_joined_labels_preds.csv"
    joined.to_csv(out_csv, index=False)
    print("saved", out_csv, "n=", len(joined))

    save_json(results, OUT / "metrics_visual_plus_field_B3.json")
    save_json(results, METRICS / "independent_val_visual_plus_field_B3.json")

    tab = pd.DataFrame(flat_rows)
    tab.to_csv(OUT / "metrics_visual_plus_field_summary.csv", index=False)
    tab.to_csv(TABLES / "independent_val_visual_plus_field_metrics.csv", index=False)
    print(tab.to_string(index=False))

    # Confusion figures
    try:
        plot_confusion(
            results,
            OUT / "fig_confusion_visual_plus_field_B3.png",
            [
                ("visual_gongliu_apple_vs_nonapple_code1", "Gongliu visual apple"),
                ("gongliu_visual_plus_field_apple", "Visual+GL field apple"),
                ("recommended_primary_visual_plus_ili_field_apple", "Visual+Ili field apple"),
            ],
        )
        plot_confusion(
            results,
            OUT / "fig_confusion_orchard_visual_plus_field_B3.png",
            [
                ("visual_gongliu_orchard_vs_nonorchard", "Gongliu visual orchard"),
                ("orchard_visual_plus_gongliu_field_unspecified", "Visual+GL field orchard"),
            ],
        )
        plot_confusion(
            results,
            FIGURES / "fig_independent_val_visual_plus_field_confusion.png",
            [
                ("visual_gongliu_apple_vs_nonapple_code1", "Gongliu visual apple"),
                ("gongliu_visual_plus_field_apple", "Visual+GL field apple"),
                ("ili_field_apple_plus_visual_gongliu_negatives_mixed_domain", "Ili field+vis neg"),
            ],
        )
        print("saved confusion figures")
    except Exception as e:
        print("figure fail:", e)

    # --- Markdown report ---
    def fmt_m(name, m):
        if "detection_rate_recall" in m:
            return (
                f"### {name}\n"
                f"- n_pos={m['n_pos']} detected={m['n_detected_pred1']} missed={m['n_missed_pred0']}\n"
                f"- detection_rate/recall={m['detection_rate_recall']:.4f}\n"
                f"- prob mean/median/min/max="
                f"{m['prob_mean']:.4f}/{m['prob_median']:.4f}/{m['prob_min']:.4f}/{m['prob_max']:.4f}\n"
                f"- note: {m.get('note','')}\n"
            )
        return (
            f"### {name}\n"
            f"- n={m['n']} (pos={m['n_pos']}, neg={m['n_neg']})\n"
            f"- OA={m['OA']:.4f} UA={m['UA_positive']:.4f} PA={m['PA_positive']:.4f} "
            f"F1={m['F1']:.4f} Kappa={m['Kappa']:.4f} AUC={m.get('AUC', float('nan')):.4f}\n"
            f"- TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}\n"
        )

    md = []
    md.append("# Field-merged independent validation — B3 (AlphaEarth-only)\n")
    md.append("## Protocol honesty\n")
    md.append(results["honesty"] + "\n")
    md.append("## Dedup\n")
    md.append("```json\n" + json.dumps(dedup_log, ensure_ascii=False, indent=2) + "\n```\n")
    md.append(f"## Field apple by county (after dedup)\n`{county_counts}`\n")
    md.append("## Metrics\n")
    for key in [
        "visual_gongliu_apple_vs_nonapple_code1",
        "visual_gongliu_orchard_vs_nonorchard",
        "gongliu_visual_plus_field_apple",
        "gongliu_field_apple_detection_rate",
        "ili_field_apple_detection_rate",
        "ili_field_apple_plus_visual_gongliu_negatives_mixed_domain",
        "recommended_primary_visual_plus_ili_field_apple",
        "orchard_visual_plus_gongliu_field_unspecified",
    ]:
        if key in results:
            md.append(fmt_m(key, results[key]))
    md.append("## Files\n")
    md.append(f"- `{out_csv}`\n- `{OUT / 'metrics_visual_plus_field_B3.json'}`\n")
    md.append(f"- `{OUT / 'metrics_visual_plus_field_summary.csv'}`\n")
    md.append(f"- `{FIELD_AE}`\n")
    (OUT / "REPORT_field_merged.md").write_text("\n".join(md), encoding="utf-8")
    print("DONE", OUT / "REPORT_field_merged.md")
    return results


if __name__ == "__main__":
    main()
