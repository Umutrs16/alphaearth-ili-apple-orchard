# -*- coding: utf-8 -*-
"""v4 retrain pipeline on the EXPANDED training set (prepared; run only after the user's training_new.kml arrives).

Usage:
  python 39_retrain_expanded.py --kml <path/to/training_new.kml> [--stages qc,sample,features,train,countymap,diag] [--out DIR]
Stages:
  qc        parse polygons named 'YL-TRN-xxxx | cK' (K in 0/1/2; description: subtype=..; img_date=..; confidence=..),
            QC: duplicate IDs, duplicate/overlapping geometry (new-new, new-old-train), inside Gongliu (GS2024, any part),
            distance to Gongliu holdout polygons (<2 km -> excluded), area outside 0.1-50 ha, invalid class, not a polygon.
            -> QC_training_new.csv, qc_summary.json, split_manifest_expanded.csv (old 38 train + new kept + unchanged holdout)
  sample    points inside new polygons with the SAME area rules as v2/v3 (22_b3_poly_holdout_v3.choose_n_points,
            sample_points_in_poly; deterministic per-polygon seed) -> sample_points_expanded.csv (old 389 train points reused)
  features  AE 2021 (10 m, sampleRegions, as 22_*.stage_ae) and S1/S2/DEM 2021 (30 m, gee_sample_features build_s2/s1/dem,
            as 30_*.py); cached; holdout features re-used from poly_ae_features_2021_v3.csv / poly_s1s2dem_features_2021_v4.csv
  train     B0/B2/B3/B6 with common.make_lgbm (same hyperparameters as B3_poly_2021_v2), seeds 42 + 1..9, threshold 0.5;
            binary target y = (class_code == 1) (class 2 and 0 are both negatives, as in B3_poly_2021_v2);
            + HIER: stage 1 orchard (1|2) vs non-orchard (0), stage 2 apple (1) vs other orchard (2) trained on orchard polygons only,
            P(apple) = P(orchard) * P(apple | orchard), threshold 0.5 (AE features; skipped if no class-2 training polygons).
            Evaluated on the UNCHANGED 101-polygon Gongliu holdout (split_manifest_v3; 113 polygons / 521 points, majority vote),
            same metrics as 34_indep_feature_compare.py (+ paired dF1 vs old B3_poly_2021_v2 and vs new B3).
            The holdout is never used for training, tuning, feature selection or threshold choice.
  countymap 30 m wall-to-wall Gongliu map with the new B3 (and HIER if available) via 33_/37_ scripts (CM_MODEL/CM_OUT env).
  diag      elevation / WorldCover breakdown of the new map vs the old map (ref_elev_30m.tif, ref_wc_30m.tif; same grid).
"""
import argparse, json, math, os, re, subprocess, sys, time, zlib, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, joblib
from shapely.ops import unary_union
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; PY = sys.executable
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(REV)); sys.path.insert(0, str(REV))
from common import make_lgbm
from utils import feature_columns
import official_boundary as ob
from hier_model import HierProba
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
PS = ROOT / "outputs/revision/polygon_samples"; V4 = ROOT / "outputs/revision/v4_experiments"
AE = [f"AE_{i:02d}" for i in range(64)]
HC = feature_columns({}, ["s1", "s2", "dem"])
SETS = {"B0": ["s2"], "B2": ["s1", "s2", "dem"], "B3": ["ae"], "B6": ["ae", "s1", "s2", "dem"]}
SEEDS = [42] + list(range(1, 10))
NAME_RE = re.compile(r"^\s*(YL-TRN-\d{4})\s*[|｜]\s*[cC]\s*([0-9])\s*$")
SUBTYPES = {"apple", "other_orchard", "cropland", "forest", "grass", "shrub", "built", "bare", "water", "mountain_grass", "mountain_forest", "mountain_shrub"}

def save(o, p): json.dump(o, open(p, "w"), ensure_ascii=False, indent=1, default=str)

# ------------------------------------------------------------------ QC
def stage_qc(kml, out):
    raw = v3m.read_kml_polys(Path(kml))
    P = gpd.read_file(PS / "v3/polygons_v3_all.geojson").to_crs(32644)
    hold = P[P.split == "val_holdout"]; trn = P[P.split == "train"]
    ho_u = unary_union(hold.geometry.values)
    gl = ob.gongliu().to_crs(32644).geometry.iloc[0]
    recs = []
    for r in raw:
        m = NAME_RE.match(r["raw_name"]); a = v3m.parse_desc(r["desc"])
        g, note = v3m.make_geom(r["coords"])
        d = {"poly_name": m.group(1) if m else r["raw_name"].strip(), "raw_name": r["raw_name"],
             "class_code": int(m.group(2)) if m else (int(float(a["class_code"])) if a.get("class_code", "").strip() not in ("",) else None),
             "subtype": a.get("subtype", "").strip().lower(), "img_date": a.get("img_date", ""), "confidence": a.get("confidence", "").strip().lower(),
             "notes": a.get("notes", ""), "name_ok": bool(m), "geom_note": note, "geom": None}
        reasons, flags = [], []
        if not m: flags.append("name_not_'YL-TRN-xxxx | cK'")
        if d["class_code"] not in (0, 1, 2): reasons.append("invalid_class")
        if d["subtype"] and d["subtype"] not in SUBTYPES: flags.append("unknown_subtype")
        if d["subtype"] == "apple" and d["class_code"] != 1 or (d["class_code"] == 1 and d["subtype"] not in ("", "apple")): flags.append("subtype_class_mismatch")
        if d["confidence"] == "low": reasons.append("confidence_low")
        if g is None: reasons.append(f"degenerate:{note}")
        else:
            gu = v3m.utm(g); d["geom"] = gu; d["area_ha"] = gu.area / 1e4
            d["cen_lon"], d["cen_lat"] = g.centroid.x, g.centroid.y
            if gu.intersects(gl): reasons.append("inside_or_touching_gongliu")
            d["d_holdout_km"] = gu.distance(ho_u) / 1000
            if d["d_holdout_km"] < 2: reasons.append("lt_2km_from_holdout")
            if d["area_ha"] < 0.1: reasons.append("area_lt_0.1ha")
            if d["area_ha"] > 50: reasons.append("area_gt_50ha")
            ov = [f"{t.name}" for t in trn.itertuples() if gu.intersection(t.geometry).area > 1.0]
            d["overlap_old_train"] = ";".join(ov)
            if ov: reasons.append("overlaps_existing_train")
            d["d_old_train_m"] = float(trn.distance(gu).min())
        d["reasons"], d["flags"] = reasons, flags; recs.append(d)
    df = pd.DataFrame(recs)
    # duplicate IDs: keep first occurrence, exclude later ones
    dup = df.poly_name.duplicated(keep="first")
    for i in df.index[dup]: df.at[i, "reasons"] = df.at[i, "reasons"] + ["duplicate_id"]
    # new-new overlaps: >50% of smaller -> exclude later (or both if class conflict); smaller overlaps flagged
    pairs = []
    idx = [i for i in df.index if df.at[i, "geom"] is not None]
    for a_i, a in enumerate(idx):
        for b in idx[a_i + 1:]:
            ga, gb = df.at[a, "geom"], df.at[b, "geom"]
            if not ga.intersects(gb): continue
            inter = ga.intersection(gb).area
            if inter <= 1.0: continue
            frac = inter / min(ga.area, gb.area); conflict = df.at[a, "class_code"] != df.at[b, "class_code"]
            pairs.append({"a": df.at[a, "poly_name"], "b": df.at[b, "poly_name"], "overlap_ha": inter / 1e4, "frac_of_smaller": frac, "class_conflict": bool(conflict)})
            if conflict: 
                for k in (a, b): df.at[k, "reasons"] = df.at[k, "reasons"] + [f"overlap_class_conflict_with_{df.at[b if k == a else a, 'poly_name']}"]
            elif frac > 0.5: df.at[b, "reasons"] = df.at[b, "reasons"] + [f"duplicate_geometry_of_{df.at[a, 'poly_name']}"]
            else:
                for k in (a, b): df.at[k, "flags"] = df.at[k, "flags"] + ["partial_overlap_same_class"]
    df["excluded"] = df["reasons"].map(bool); df["exclude_reason"] = df["reasons"].map(";".join); df["qc_flags"] = df["flags"].map(";".join)
    kept = df[~df.excluded]
    gpd.GeoDataFrame(kept.drop(columns=["reasons", "flags"]), geometry="geom", crs=32644).to_crs(4326).to_file(out / "training_new_kept.geojson", driver="GeoJSON")
    df.drop(columns=["geom", "reasons", "flags"]).to_csv(out / "QC_training_new.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(out / "QC_overlaps_new.csv", index=False, encoding="utf-8-sig")
    man3 = pd.read_csv(PS / "split_manifest_v3.csv")
    new = pd.DataFrame({"name": kept.poly_name, "class_code": kept.class_code, "county": None, "area_ha_utm44n": kept.area_ha,
                        "cen_lon": kept.cen_lon, "cen_lat": kept.cen_lat, "excluded": False, "exclude_reason": "", "split": "train",
                        "label_source": "yl_trn_expansion", "subtype": kept.subtype})
    cty = ob.county()
    pts = gpd.GeoDataFrame(new[["cen_lon", "cen_lat"]], geometry=gpd.points_from_xy(new.cen_lon, new.cen_lat), crs=4326)
    j = gpd.sjoin(pts, cty[["name", "geometry"]].rename(columns={"name": "cty_name"}), how="left", predicate="within")
    new["county"] = j[~j.index.duplicated()]["cty_name"].reindex(new.index).values
    man = pd.concat([man3, new], ignore_index=True); man.to_csv(out / "split_manifest_expanded.csv", index=False, encoding="utf-8-sig")
    summ = {"kml": str(kml), "n_parsed": int(len(df)), "n_kept": int(len(kept)), "n_excluded": int(df.excluded.sum()),
            "exclusions": df[df.excluded][["poly_name", "class_code", "exclude_reason"]].to_dict("records"),
            "kept_by_class": kept.class_code.value_counts().to_dict(), "kept_by_subtype": kept.subtype.value_counts().to_dict(),
            "kept_area_ha_by_class": kept.groupby("class_code").area_ha.sum().round(2).to_dict(),
            "n_flagged": int((df.qc_flags != "").sum()), "n_overlap_pairs": len(pairs),
            "old_train_polygons": int((man3.split == "train").sum()), "holdout_polygons_unchanged": int((man3.split == "val_holdout").sum()),
            "min_d_holdout_km_kept": float(kept.d_holdout_km.min()) if len(kept) else None}
    save(summ, out / "qc_summary.json"); print(json.dumps(summ, ensure_ascii=False, indent=1, default=str)); return summ

# ------------------------------------------------------------------ sampling
def stage_sample(out):
    kept = gpd.read_file(out / "training_new_kept.geojson")
    rows = []
    for r in kept.itertuples():
        n, sp = v3m.choose_n_points(float(r.area_ha), int(r.class_code))
        rng = np.random.default_rng((v3m.SEED + zlib.crc32(r.poly_name.encode()) % 100000) % (2**31 - 1))
        for i, (x, y) in enumerate(v3m.sample_points_in_poly(r.geometry, n, sp, rng), 1):
            rows.append({"sample_id": f"{r.poly_name}-P{i:02d}", "poly_name": r.poly_name, "class_code": int(r.class_code), "county": None,
                         "split": "train", "lon": x, "lat": y, "area_ha": float(r.area_ha), "point_source": "yl_trn_expansion", "subtype": r.subtype})
    new = pd.DataFrame(rows)
    old = pd.read_csv(PS / "sample_points_v3.csv")
    allp = pd.concat([old, new], ignore_index=True)
    assert not set(new.poly_name) & set(old[old.split == "val_holdout"].poly_name)
    allp.to_csv(out / "sample_points_expanded.csv", index=False, encoding="utf-8-sig")
    print("new points", len(new), "from", new.poly_name.nunique(), "polygons;", new.groupby("class_code").size().to_dict())

# ------------------------------------------------------------------ features
def _sample(img, pts, scale, chunk, cache, key="sample_id"):
    import ee
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=[key])
    todo = pts[~pts[key].isin(done[key])]; rows = done.to_dict("records")
    for s in range(0, len(todo), chunk):
        sub = todo.iloc[s:s + chunk]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {key: str(getattr(r, key))}) for r in sub.itertuples()])
        for att in range(5):
            try:
                rows += [f["properties"] for f in img.sampleRegions(collection=fc, scale=scale, geometries=False).getInfo()["features"]]; break
            except Exception as e:
                print("retry", att, str(e)[:150], flush=True); time.sleep(10 * (att + 1))
        pd.DataFrame(rows).to_csv(cache, index=False); print(cache.name, s + len(sub), "/", len(todo), flush=True)
    return pd.read_csv(cache)

def stage_features(out):
    import ee, gee_sample_features as g
    ee.Initialize(project="hallowed-chain-458005-g9")
    pts = pd.read_csv(out / "sample_points_expanded.csv"); new = pts[pts.point_source == "yl_trn_expansion"]
    ae = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2021-01-01", "2022-01-01").mosaic()
    a = _sample(ae, new, 10, 40, out / "_ae_new_raw.csv")
    a = a.rename(columns={f"A{i:02d}": f"AE_{i:02d}" for i in range(64)})[["sample_id"] + AE]
    old = pd.read_csv(PS / "poly_ae_features_2021_v3.csv")[["sample_id"] + AE]
    pd.concat([old, a], ignore_index=True).drop_duplicates("sample_id").to_csv(out / "poly_ae_features_2021_expanded.csv", index=False)
    aoi = ee.Geometry.Rectangle([new.lon.min() - 0.05, new.lat.min() - 0.05, new.lon.max() + 0.05, new.lat.max() + 0.05])
    img = g.build_s2(2021, aoi).addBands(g.build_s1(2021, aoi)).addBands(g.build_dem(aoi))
    h = _sample(img, new, 30, 80, out / "_s1s2dem_new_raw.csv")
    oldh = pd.read_csv(V4 / "poly_s1s2dem_features_2021_v4.csv")
    pd.concat([oldh, h], ignore_index=True).drop_duplicates("sample_id").to_csv(out / "poly_s1s2dem_features_2021_expanded.csv", index=False)
    miss = {"ae_missing": int(len(set(new.sample_id) - set(a.dropna().sample_id))), "s1s2dem_missing_rows": int(len(set(new.sample_id) - set(h.sample_id)))}
    save(miss, out / "features_fetch_log.json"); print(miss)

# ------------------------------------------------------------------ train / eval
def stage_train(out):
    from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, accuracy_score,
                                 balanced_accuracy_score, matthews_corrcoef, roc_auc_score)
    pts = pd.read_csv(out / "sample_points_expanded.csv")
    df = pts.merge(pd.read_csv(out / "poly_ae_features_2021_expanded.csv"), on="sample_id", how="left") \
            .merge(pd.read_csv(out / "poly_s1s2dem_features_2021_expanded.csv"), on="sample_id", how="left")
    df = df[df[AE].notna().all(1)]
    train = df[df.split == "train"].copy(); hold = df[df.split == "val_holdout"].copy()
    man3 = pd.read_csv(PS / "split_manifest_v3.csv")
    assert set(hold.poly_name) == set(man3[man3.split == "val_holdout"].name) and len(hold) == 521, "holdout changed!"
    assert not set(train.poly_name) & set(hold.poly_name)
    med = train[HC].median()
    for d in (train, hold): d[HC] = d[HC].fillna(med)
    hold["y"] = (hold.class_code == 1).astype(int); ytr = (train.class_code == 1).astype(int).values
    models = out / "models"; models.mkdir(exist_ok=True)
    rng = np.random.default_rng(42); BOOT = [rng.integers(0, 101, 101) for _ in range(1000)]

    def poly_metrics(h):
        pt = v3m.poly_table(h); d = pt.dropna(subset=["pred_major"]).sort_values("poly_name")
        y, p, s = d.y.values.astype(int), d.pred_major.values.astype(int), d.mean_prob.values
        m = v3m.bin_metrics(y, p); m["AP_poly_meanprob"] = float(average_precision_score(y, s)); m["AUC_poly_meanprob"] = float(roc_auc_score(y, s))
        m["AP_poly_fracpos"] = float(average_precision_score(y, d.frac_pred_apple.values))
        bs = {k: [] for k in ("F1", "precision", "recall", "OA", "balanced_accuracy", "MCC", "AP_poly_meanprob")}
        for i in BOOT:
            a, b, c = y[i], p[i], s[i]
            bs["F1"].append(f1_score(a, b, zero_division=0)); bs["precision"].append(precision_score(a, b, zero_division=0))
            bs["recall"].append(recall_score(a, b, zero_division=0)); bs["OA"].append(accuracy_score(a, b))
            bs["balanced_accuracy"].append(balanced_accuracy_score(a, b)); bs["MCC"].append(matthews_corrcoef(a, b)); bs["AP_poly_meanprob"].append(average_precision_score(a, c))
        m["bootstrap_95ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items()}
        pc = {}
        for cc, lab in [(1, "apple_recall"), (2, "other_orchard_rejected"), (0, "non_orchard_rejected")]:
            q = d[d.class_code == cc]; k = int((q.pred_major == 1).sum()) if cc == 1 else int((q.pred_major == 0).sum())
            pc[lab] = {"k": k, "n": int(len(q)), "rate": k / len(q) if len(q) else None, "wilson": list(v3m.wilson(k, len(q)))}
        m["per_class"] = pc; return m, (y, p), pt

    def point_metrics(h):
        m = v3m.bin_metrics(h.y, h.pred); m["AP"] = float(average_precision_score(h.y, h.prob)); m["AUC"] = float(roc_auc_score(h.y, h.prob)); return m

    def fit_bin(cols, seed, X=train, y=ytr):
        c = make_lgbm(seed=seed); c.fit(X[cols].values.astype(float), y); return c

    res = {"design": __doc__, "n_train_points": int(len(train)), "n_train_polygons": int(train.poly_name.nunique()),
           "train_by_class_polygons": train.groupby("class_code").poly_name.nunique().to_dict(),
           "train_by_class_points": train.class_code.value_counts().to_dict(), "n_holdout_points": int(len(hold)), "seeds": SEEDS, "models": {}}
    keep = {}
    variants = dict(SETS)
    has2 = (train.class_code == 2).any() and (train.class_code == 1).any()
    if has2: variants["HIER"] = ["ae"]
    for bid, groups in variants.items():
        cols = feature_columns({}, groups); per = []
        for sd in SEEDS:
            h = hold.copy()
            if bid == "HIER":
                c1 = fit_bin(cols, sd, train, (train.class_code > 0).astype(int).values)
                orc = train[train.class_code > 0]; c2 = fit_bin(cols, sd, orc, (orc.class_code == 1).astype(int).values)
                po = c1.predict_proba(h[cols].values.astype(float))[:, 1]; pa = c2.predict_proba(h[cols].values.astype(float))[:, 1]
                h["prob"] = po * pa; h["p_orchard"] = po; clf = {"stage1_orchard": c1, "stage2_apple_given_orchard": c2}
            else:
                clf = fit_bin(cols, sd); h["prob"] = clf.predict_proba(h[cols].values.astype(float))[:, 1]
            h["pred"] = (h.prob >= 0.5).astype(int)
            pm, yp, pt = poly_metrics(h); per.append({"seed": sd, "poly": pm, "point": point_metrics(h)})
            if sd == 42:
                keep[bid] = yp; pt.to_csv(out / f"pred_polys_{bid}_expanded_seed42.csv", index=False)
                h[["sample_id", "poly_name", "class_code", "y", "prob", "pred"]].to_csv(out / f"pred_points_{bid}_expanded_seed42.csv", index=False)
                joblib.dump({"model": clf, "features": cols, "year": 2021, "feature_set": bid, "version": "expanded_v4",
                             "hierarchical": bid == "HIER"}, models / f"{bid}_poly_2021_expanded.joblib")
        agg = {}
        for k in ("F1", "precision", "recall", "MCC", "OA", "balanced_accuracy", "AP_poly_meanprob"):
            v = np.array([r["poly"][k] for r in per]); agg["poly_" + k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
        res["models"][bid] = {"features": groups, "n_features": len(cols), "seed42": per[0], "seed_summary": agg, "per_seed_poly_F1": [r["poly"]["F1"] for r in per]}
        print(bid, "seed42 F1=%.3f AP=%.3f | seeds %.3f±%.3f" % (per[0]["poly"]["F1"], per[0]["poly"]["AP_poly_meanprob"], agg["poly_F1"]["mean"], agg["poly_F1"]["sd"]), flush=True)
    # references: old frozen B3_poly_2021_v2 predictions (seed 42) from 34_*
    old = pd.read_csv(V4 / "pred_polys_B3_seed42.csv").dropna(subset=["pred_major"]).sort_values("poly_name")
    keep["B3_old"] = (old.y.values.astype(int), old.pred_major.values.astype(int))
    y3 = keep["B3"][0]; assert (keep["B3_old"][0] == y3).all()
    for ref in ("B3", "B3_old"):
        yr, pr = keep[ref]
        for bid in variants:
            if bid == ref: continue
            y, p = keep[bid]
            ds = [f1_score(y[i], p[i], zero_division=0) - f1_score(y[i], pr[i], zero_division=0) for i in BOOT]
            res["models"][bid][f"paired_dF1_vs_{ref}_seed42"] = {"delta": float(f1_score(y, p) - f1_score(yr, pr)),
                                                                 "ci95": [float(np.quantile(ds, .025)), float(np.quantile(ds, .975))]}
    old_m = json.load(open(V4 / "metrics_indep_feature_compare_v4.json"))["models"]
    res["reference_old_training_seed_summary_poly_F1"] = {k: old_m[k]["seed_summary"]["poly_F1"] for k in old_m}
    save(res, out / "metrics_expanded_retrain.json")
    rows = []
    for bid, m in res["models"].items():
        p = m["seed42"]["poly"]; ci = p["bootstrap_95ci"]; pc = p["per_class"]
        rows.append({"model": bid, "poly_F1 [CI]": f"{p['F1']:.3f} [{ci['F1'][0]:.3f}, {ci['F1'][1]:.3f}]", "P": round(p["precision"], 3), "R": round(p["recall"], 3),
                     "MCC": round(p["MCC"], 3), "OA": round(p["OA"], 3), "BA": round(p["balanced_accuracy"], 3),
                     "AP_poly [CI]": f"{p['AP_poly_meanprob']:.3f} [{ci['AP_poly_meanprob'][0]:.3f}, {ci['AP_poly_meanprob'][1]:.3f}]",
                     "point F1/AP": f"{m['seed42']['point']['F1']:.3f}/{m['seed42']['point']['AP']:.3f}",
                     "other_orch_rej": f"{pc['other_orchard_rejected']['k']}/{pc['other_orchard_rejected']['n']}",
                     "non_orch_rej": f"{pc['non_orchard_rejected']['k']}/{pc['non_orchard_rejected']['n']}",
                     "dF1 vs old B3 [CI]": (lambda d: f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")(m["paired_dF1_vs_B3_old_seed42"]),
                     "F1 10 seeds": f"{m['seed_summary']['poly_F1']['mean']:.3f}±{m['seed_summary']['poly_F1']['sd']:.3f}"})
    t = pd.DataFrame(rows); t.to_csv(out / "table_expanded_retrain.csv", index=False, encoding="utf-8-sig"); print(t.to_string())

# ------------------------------------------------------------------ county map + diagnostics
def stage_countymap(out, scale=30):
    for bid in ("B3", "HIER"):
        mp = out / "models" / f"{bid}_poly_2021_expanded.joblib"
        if not mp.exists(): continue
        if bid == "HIER":
            print("HIER county map: 33_ applies a single predict_proba; wrapping two-stage model"); mp = _wrap_hier(mp)
        env = dict(os.environ, CM_MODEL=str(mp), CM_OUT=str(out / f"county_map_{bid}"), CM_TAG=f"{bid}expanded")
        t0 = time.time()
        subprocess.run([PY, str(REV / "33_county_map_wall2wall.py"), str(scale), "0.1"], env=env, check=True)
        subprocess.run([PY, str(REV / "37_county_map_mosaic.py"), str(scale)], env=env, check=True)
        print(bid, "county map", round((time.time() - t0) / 60, 1), "min")

def _wrap_hier(mp):
    b = joblib.load(mp); m = b["model"]
    w = dict(b, model=HierProba(m["stage1_orchard"], m["stage2_apple_given_orchard"]))
    p = mp.with_name(mp.stem + "_wrapped.joblib"); joblib.dump(w, p); return p

def stage_diag(out, scale=30):
    import rasterio
    ref = V4 / "county_map"
    with rasterio.open(ref / "ref_elev_30m.tif") as s: elev = s.read(1); T = s.transform
    with rasterio.open(ref / "ref_wc_30m.tif") as s: wc = s.read(1)
    maps = {"B3_old": ref / f"gongliu_B3poly2021v2_class_{scale}m.tif"}
    for bid in ("B3", "HIER"):
        p = out / f"county_map_{bid}" / f"gongliu_{bid}expanded_class_{scale}m.tif"
        if p.exists(): maps[bid + "_expanded"] = p
    lat = T.f + (np.arange(elev.shape[0]) + 0.5) * T.e; R = 6371008.8
    pa = (np.deg2rad(T.a) * R * np.cos(np.deg2rad(lat)))[:, None] * (np.deg2rad(-T.e) * R) * np.ones((1, elev.shape[1]))
    bins = [0, 1000, 1217, 1400, 1600, 2000, 9000]; lab = ["<1000", "1000-1217", "1217-1400", "1400-1600", "1600-2000", ">=2000"]
    WCN = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland", 100: "moss"}
    res = {}
    for k, p in maps.items():
        with rasterio.open(p) as s: c = s.read(1)
        assert c.shape == elev.shape, f"grid mismatch {k}"
        v = c != 255; a = c == 1
        r = {"apple_area_unadjusted_ha": float((pa * a).sum() / 1e4), "apple_frac": float((pa * a).sum() / (pa * v).sum()), "by_elev": {}, "by_wc_ha": {}}
        for lo, hi, l in zip(bins[:-1], bins[1:], lab):
            m = v & (elev >= lo) & (elev < hi)
            r["by_elev"][l] = {"apple_ha": float((pa * (a & m)).sum() / 1e4), "apple_frac": float((pa * (a & m)).sum() / max((pa * m).sum(), 1))}
        for code, nm in WCN.items(): r["by_wc_ha"][nm] = float((pa * (a & (wc == code))).sum() / 1e4)
        res[k] = r
    res["note"] = "Raw pixel-count areas (unadjusted, not bias-corrected). AOMC 2021 Gongliu reference ~2,704 ha."
    save(res, out / "county_map_elev_wc_diagnostics.json"); print(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kml", required=False)
    ap.add_argument("--stages", default="qc,sample,features,train,countymap,diag")
    ap.add_argument("--out", default=str(V4 / "retrain_expanded")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); st = a.stages.split(",")
    if "qc" in st: assert a.kml, "--kml required for qc"; stage_qc(a.kml, out)
    if "sample" in st: stage_sample(out)
    if "features" in st: stage_features(out)
    if "train" in st: stage_train(out)
    if "countymap" in st: stage_countymap(out)
    if "diag" in st: stage_diag(out)
