# -*- coding: utf-8 -*-
"""Evaluate Gongliu independent validation with filled visual-interpretation labels.

Uses offline B3 (AlphaEarth-only) model; samples AE at kit points via GEE
(with incremental cache). B6 reported only if full feature vectors available.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import FIGURES, METRICS, MODELS, TABLES, ensure_out_dirs, save_json
from utils import compute_metrics

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "data" / "interpretation_kit"
OUT = ROOT / "outputs" / "revision" / "independent_val"
FILLED = KIT / "Gongliu_independent_val_判读表_filled.xlsx"
POINTS = KIT / "Gongliu_independent_val_points.csv"
AE_CACHE = OUT / "kit_ae_features_2021.csv"
PROJECT = "hallowed-chain-458005-g9"

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
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_filled() -> pd.DataFrame:
    df = pd.read_excel(FILLED, sheet_name="判读填写")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["interp_class_code"] = pd.to_numeric(df["interp_class_code"], errors="coerce")
    df["is_apple"] = pd.to_numeric(df["is_apple"], errors="coerce")
    df["confidence"] = df["confidence"].astype(str).str.strip().replace({"nan": np.nan})
    # normalize exclude
    ef = df["exclude_flag"]
    df["exclude_flag"] = ef.notna() & (ef.astype(str).str.strip() != "") & (
        ~ef.astype(str).str.strip().str.lower().isin(["0", "false", "no", "nan"])
    )
    df["interp_class"] = df["interp_class_code"].map(CLASS_MAP)
    return df


def qc_report(df: pd.DataFrame) -> dict:
    incomplete = df[
        df["interp_class_code"].isna() | df["confidence"].isna()
    ]
    # is_apple may be blank for code 9
    code_to_apple = {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}
    df = df.copy()
    df["is_apple_from_code"] = df["interp_class_code"].map(code_to_apple)
    mismatch = df[
        df["is_apple"].notna()
        & df["is_apple_from_code"].notna()
        & (df["is_apple"] != df["is_apple_from_code"])
    ]
    missing_is_apple = df[df["is_apple"].isna() & (df["interp_class_code"] != 9)]
    conf_ok = df["confidence"].isin(["高", "中", "低"])
    report = {
        "n_total": int(len(df)),
        "counts_interp_class_code": {
            str(int(k)) if pd.notna(k) else "nan": int(v)
            for k, v in df["interp_class_code"].value_counts(dropna=False).sort_index().items()
        },
        "counts_is_apple": {
            str(k): int(v) for k, v in df["is_apple"].value_counts(dropna=False).sort_index().items()
        },
        "counts_confidence": {
            str(k): int(v) for k, v in df["confidence"].value_counts(dropna=False).items()
        },
        "n_exclude_flag": int(df["exclude_flag"].sum()),
        "n_uncertain_code9": int((df["interp_class_code"] == 9).sum()),
        "n_incomplete_code_or_conf": int(len(incomplete)),
        "incomplete_sample_ids": incomplete["sample_id"].tolist(),
        "n_missing_is_apple_non9": int(len(missing_is_apple)),
        "missing_is_apple_ids": missing_is_apple["sample_id"].tolist(),
        "n_is_apple_code_mismatch": int(len(mismatch)),
        "mismatch_rows": mismatch[
            ["sample_id", "interp_class_code", "is_apple", "is_apple_from_code", "confidence"]
        ].to_dict(orient="records"),
        "n_confidence_nonstandard": int((~conf_ok & df["confidence"].notna()).sum()),
        "by_stratum_code": (
            df.groupby("stratum")["interp_class_code"]
            .value_counts()
            .unstack(fill_value=0)
            .astype(int)
            .to_dict()
        ),
        "protocol_notes": [
            "Primary apple label for metrics: interp_class_code==1 (canonical).",
            "Secondary apple label: is_apple field (annotator).",
            "Orchard label: interp_class_code in {1,2}.",
            "Exclude: exclude_flag OR interp_class_code==9 OR confidence==低 (for confident set).",
            "Mismatches between is_apple and code are flagged, not auto-corrected.",
        ],
    }
    return report


def sample_ae_features(df: pd.DataFrame, year: int = 2021, chunk: int = 40) -> pd.DataFrame:
    import ee

    ee.Initialize(project=PROJECT)
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    # resume from cache
    done = {}
    if AE_CACHE.exists():
        cached = pd.read_csv(AE_CACHE)
        for _, r in cached.iterrows():
            done[r["sample_id"]] = r.to_dict()
        print(f"AE cache loaded: {len(done)}")

    todo = df[~df["sample_id"].isin(done)].copy()
    print(f"AE to sample: {len(todo)} / {len(df)}")
    rows_new = []
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
            rows_new.append(rec)
            done[r.sample_id] = rec
        # incremental save
        pd.DataFrame(list(done.values())).to_csv(AE_CACHE, index=False)
        print(f"AE sampled progress {len(done)}/{len(df)}")
        time.sleep(0.35)
    out = pd.DataFrame(list(done.values()))
    out.to_csv(AE_CACHE, index=False)
    return out


def attach_grid_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor AE + grid prob from ~1.2 km preview grid if point AE missing."""
    from scipy.spatial import cKDTree

    gp = pd.read_csv(ROOT / "outputs" / "revision" / "maps" / "grid_pred_2021.csv")
    tree = cKDTree(np.c_[gp.lon.values, gp.lat.values])
    d, idx = tree.query(np.c_[df.lon.values, df.lat.values], k=1)
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    out = df.copy()
    for c in ae_cols + ["prob", "pred"]:
        if c in gp.columns:
            out[f"grid_{c}"] = gp.iloc[idx][c].values
    # approx meters
    out["grid_nn_m"] = [
        float(haversine_m(lo, la, gp.iloc[j].lon, gp.iloc[j].lat))
        for lo, la, j in zip(df.lon.values, df.lat.values, idx)
    ]
    out["grid_nn_deg"] = d
    return out


def metrics_pack(y, pred, prob, prefix="") -> dict:
    m = compute_metrics(y, pred, prob)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    # UA/PA explicit
    tn, fp, fn, tp = cm.ravel()
    ua_pos = tp / (tp + fp) if (tp + fp) else float("nan")  # user accuracy / precision
    pa_pos = tp / (tp + fn) if (tp + fn) else float("nan")  # producer accuracy / recall
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
    if prefix:
        return {f"{prefix}_{k}" if k not in ("confusion_matrix",) else k: v for k, v in m.items()}
    return m


def evaluate(df: pd.DataFrame, prob_col: str = "prob_B3", pred_col: str = "pred_B3") -> dict:
    # Confident set: not exclude, not code 9, confidence in 高/中
    base = df[~df["exclude_flag"]].copy()
    conf = base[
        (base["interp_class_code"] != 9)
        & (base["confidence"].isin(["高", "中"]))
        & base[prob_col].notna()
    ].copy()

    results = {"n_joined_with_pred": int(df[prob_col].notna().sum()), "n_confident": int(len(conf))}

    # Apple vs non-apple: canonical code==1
    conf["y_apple_code"] = (conf["interp_class_code"] == 1).astype(int)
    # Apple via is_apple (drop NA)
    conf_ia = conf[conf["is_apple"].notna()].copy()
    conf_ia["y_apple_is"] = conf_ia["is_apple"].astype(int)

    # Orchard = code 1 or 2
    conf["y_orchard"] = conf["interp_class_code"].isin([1, 2]).astype(int)

    pred = conf[pred_col].astype(int).values
    prob = conf[prob_col].astype(float).values

    results["apple_vs_nonapple_code1"] = metrics_pack(
        conf["y_apple_code"].values, pred, prob
    )
    results["apple_vs_nonapple_is_apple"] = metrics_pack(
        conf_ia["y_apple_is"].values,
        conf_ia[pred_col].astype(int).values,
        conf_ia[prob_col].astype(float).values,
    )
    results["orchard_vs_nonorchard"] = metrics_pack(
        conf["y_orchard"].values, pred, prob
    )

    # High-confidence only subset
    hi = conf[conf["confidence"] == "高"]
    if len(hi) >= 30:
        results["apple_vs_nonapple_code1_high_conf"] = metrics_pack(
            (hi["interp_class_code"] == 1).astype(int).values,
            hi[pred_col].astype(int).values,
            hi[prob_col].astype(float).values,
        )
        results["orchard_vs_nonorchard_high_conf"] = metrics_pack(
            hi["interp_class_code"].isin([1, 2]).astype(int).values,
            hi[pred_col].astype(int).values,
            hi[prob_col].astype(float).values,
        )

    # Per-class prediction rates
    ct = pd.crosstab(
        conf["interp_class"].fillna("unk"),
        conf[pred_col].astype(int),
        rownames=["interp_class"],
        colnames=["pred_apple"],
    )
    results["pred_by_interp_class"] = ct.to_dict()
    results["stratum_counts_confident"] = conf["stratum"].value_counts().to_dict()
    return results, conf


def main():
    ensure_out_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    filled = load_filled()
    qc = qc_report(filled)
    save_json(qc, OUT / "qc_summary.json")
    print("=== QC ===")
    print(json.dumps({k: qc[k] for k in qc if k != "mismatch_rows" and k != "by_stratum_code"}, ensure_ascii=False, indent=2))
    print("mismatches:", qc["n_is_apple_code_mismatch"])

    # Sample AE at points
    try:
        ae = sample_ae_features(filled, year=2021, chunk=40)
        ae_source = "gee_point_sample_2021"
    except Exception as e:
        print("GEE AE sampling failed:", e)
        ae = pd.DataFrame()
        ae_source = "none"

    # Merge
    df = filled.merge(ae, on=["sample_id", "lon", "lat"], how="left", suffixes=("", "_ae"))
    if "year" not in df.columns:
        df["year"] = 2021

    # Grid fallback for missing AE
    df = attach_grid_fallback(df)
    ae_cols = [f"AE_{i:02d}" for i in range(64)]
    missing_ae = df[ae_cols[0]].isna() if ae_cols[0] in df.columns else pd.Series(True, index=df.index)
    n_missing = int(missing_ae.sum()) if ae_cols[0] in df.columns else len(df)
    if n_missing:
        print(f"Filling {n_missing} rows from grid NN AE")
        for c in ae_cols:
            if c not in df.columns:
                df[c] = np.nan
            df.loc[missing_ae, c] = df.loc[missing_ae, f"grid_{c}"].values
        ae_source = ae_source + "+grid_nn_fallback" if ae_source != "none" else "grid_nn_only"
    df["ae_source"] = np.where(missing_ae, "grid_nn", "gee_point")

    # Predict B3
    bundle = joblib.load(MODELS / "B3_full_2019_2021.joblib")
    clf, cols = bundle["model"], bundle["cols"]
    X = df[cols].astype(float).copy()
    X = X.fillna(X.median(numeric_only=True))
    prob = clf.predict_proba(X.values)[:, 1]
    pred = (prob >= 0.5).astype(int)
    df["prob_B3"] = prob
    df["pred_B3"] = pred
    # also keep grid pred for reference
    df["prob_grid"] = df["grid_prob"]
    df["pred_grid"] = df["grid_pred"]

    # Try B6 if ablation model cols subset available from somewhere — skip if no S1/S2/DEM
    b6_note = "B6 skipped: kit points lack S1/S2/DEM features offline; B3 AlphaEarth-only used."
    try:
        b6 = joblib.load(MODELS / "ablation_B6.joblib")
        need = set(b6["cols"])
        if need.issubset(set(df.columns)) and all(df[c].notna().any() for c in list(need)[:5]):
            X6 = df[b6["cols"]].astype(float).fillna(0)
            df["prob_B6"] = b6["model"].predict_proba(X6.values)[:, 1]
            df["pred_B6"] = (df["prob_B6"] >= 0.5).astype(int)
            b6_note = "B6 evaluated with available columns."
        else:
            missing_cols = sorted(need - set(df.columns))
            b6_note = f"B6 skipped; missing {len(missing_cols)} cols e.g. {missing_cols[:8]}"
    except Exception as e:
        b6_note = f"B6 skipped: {e}"

    # Save joined table
    out_csv = OUT / "joined_labels_preds.csv"
    df.to_csv(out_csv, index=False)
    print("saved", out_csv, "n=", len(df))

    results, conf = evaluate(df)
    results["model"] = "B3_full_2019_2021"
    results["ae_source"] = ae_source
    results["b6_note"] = b6_note
    results["qc"] = {
        "n_total": qc["n_total"],
        "n_uncertain_code9": qc["n_uncertain_code9"],
        "n_is_apple_code_mismatch": qc["n_is_apple_code_mismatch"],
        "n_exclude_flag": qc["n_exclude_flag"],
    }
    results["honesty"] = (
        "Labels are human visual interpretation of the Gongliu independent kit (600 pts). "
        "Metrics exclude exclude_flag, code=9 uncertain, and confidence=低. "
        "Primary apple definition uses interp_class_code==1; is_apple-field metrics also reported. "
        "Orchard = codes 1+2. AOMC-protocol independent_val_B3.json remains weak-label fit, not visual truth. "
        f"AE features: {ae_source}."
    )
    save_json(results, OUT / "metrics_visual_B3.json")
    save_json(results, METRICS / "independent_val_visual_B3.json")

    # Flat CSV table
    rows = []
    for name in [
        "apple_vs_nonapple_code1",
        "apple_vs_nonapple_is_apple",
        "orchard_vs_nonorchard",
        "apple_vs_nonapple_code1_high_conf",
        "orchard_vs_nonorchard_high_conf",
    ]:
        if name not in results:
            continue
        m = results[name]
        rows.append(
            {
                "task": name,
                "model": "B3_AE",
                "n": m["n"],
                "n_pos": m["n_pos"],
                "n_neg": m["n_neg"],
                "OA": m["OA"],
                "UA": m["UA_positive"],
                "PA": m["PA_positive"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "Kappa": m["Kappa"],
                "IoU": m["IoU"],
                "AUC": m.get("AUC"),
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TN": m["TN"],
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "metrics_visual_summary.csv", index=False)
    tab.to_csv(TABLES / "independent_val_visual_metrics.csv", index=False)
    print(tab.to_string(index=False))

    # Confusion figure
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, key, title in zip(
            axes,
            ["apple_vs_nonapple_code1", "orchard_vs_nonorchard"],
            ["Apple (code=1) vs non-apple", "Orchard (1+2) vs non-orchard"],
        ):
            m = results[key]
            cm = np.array(
                [
                    [m["confusion_matrix"]["TN"], m["confusion_matrix"]["FP"]],
                    [m["confusion_matrix"]["FN"], m["confusion_matrix"]["TP"]],
                ]
            )
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"{title}\nOA={m['OA']:.3f} F1={m['F1']:.3f} n={m['n']}")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(OUT / "fig_confusion_visual_B3.png", dpi=150)
        fig.savefig(FIGURES / "fig_independent_val_visual_confusion.png", dpi=150)
        plt.close()
        print("saved confusion figure")
    except Exception as e:
        print("figure fail:", e)

    # Keep semi-auto AOMC metrics path noted
    save_json(
        {
            "visual_metrics": str(OUT / "metrics_visual_B3.json"),
            "aomc_weaklabel_metrics": str(METRICS / "independent_val_B3.json"),
            "note": "independent_val_B3.json = AOMC/WorldCover semi-auto labels (weak-label fit). Visual metrics supersede for true independent validation.",
        },
        OUT / "README_metrics.json",
    )
    print("DONE", OUT)


if __name__ == "__main__":
    main()
