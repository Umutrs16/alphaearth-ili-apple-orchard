# -*- coding: utf-8 -*-
"""新源 second-county final holdout: QC + ONE-TIME evaluation of the frozen models (PREREG_xinyuan_holdout.md).
Usage:
  python 47c_xinyuan_eval_once.py <xinyuan_holdout.kml> --qc-only      # geometry/label QC + point sampling only; repeatable; no features, no models
  python 47c_xinyuan_eval_once.py <xinyuan_holdout.kml> --evaluate     # ONE TIME: verify prereg + model hashes, fetch features, evaluate, write EVAL_ONCE.lock
  python 47c_xinyuan_eval_once.py --selftest                            # code test on remaining TRAINING polygons (in-sample, cached features, no 新源 data)
"""
import sys, re, json, math, time, zlib, hashlib, argparse, datetime, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, joblib
from shapely.ops import unary_union
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(REV)); sys.path.insert(0, str(REV / "figures_v3_scripts"))
import official_boundary as ob
from utils import feature_columns
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
OUT = ROOT / "outputs/revision/xinyuan_holdout"; MOD = OUT / "frozen_models"; E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"
LOCK = OUT / "EVAL_ONCE.lock"
NAME_RE = re.compile(r"^\s*(XY-HO-\d{4})\s*[|｜]\s*[cC]\s*([0-9])\s*$")
MODELS = ["B0", "B3", "B6", "HIER"]; SEEDS = [42] + list(range(1, 10)); AE = [f"AE_{i:02d}" for i in range(64)]
HC = feature_columns({}, ["s1", "s2", "dem"]); BANDS = [(0, 900, "<900"), (900, 1200, "900-1200"), (1200, 1400, "1200-1400"), (1400, 9000, ">=1400")]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def save(o, p): json.dump(o, open(p, "w"), ensure_ascii=False, indent=1, default=str)

def verify_frozen():
    rec = json.load(open(OUT / "prereg_record.json"))
    assert sha(OUT / "PREREG_xinyuan_holdout.md") == rec["sha256"], "PREREG file changed after freezing!"
    man = json.load(open(OUT / "frozen_models_manifest.json")); assert man["prereg_sha256"] == rec["sha256"]
    for m in man["models"]: assert sha(MOD / m["file"]) == m["sha256"], f"model file changed: {m['file']}"
    return rec, man

# ------------------------------------------------------------------ QC (no model, no features)
def qc(kml):
    raw = v3m.read_kml_polys(Path(kml))
    cty = ob.county(); xy = cty[cty.name == "新源县"].geometry.iloc[0]
    xy_u = gpd.GeoSeries([xy], crs=4326).to_crs(32644).iloc[0]
    rem = gpd.read_file(OUT / "removed_training_polygons.geojson").to_crs(32644); rem_u = unary_union(rem.geometry.values)
    rt = pd.read_csv(OUT / "training_removal_table.csv"); keepn = set(rt[~rt.dropped].name)
    old = gpd.read_file(ROOT / "outputs/revision/polygon_samples/v3/polygons_v3_all.geojson"); old = old[old.split == "train"][["name", "geometry"]]
    new = gpd.read_file(E / "training_new_kept.geojson").rename(columns={"poly_name": "name"})[["name", "geometry"]]
    trn = pd.concat([old, new]); trn = gpd.GeoDataFrame(trn[trn.name.isin(keepn)], crs=4326).to_crs(32644); trn_u = unary_union(trn.geometry.values)
    recs, ignored = [], []
    for r in raw:
        nm = r["raw_name"]
        if any(k in nm for k in ("候选", "已移除", "新源县")): ignored.append(nm); continue
        m = NAME_RE.match(nm); a = v3m.parse_desc(r["desc"]); g, note = v3m.make_geom(r["coords"])
        d = {"poly_name": m.group(1) if m else nm.strip(), "raw_name": nm, "class_code": int(m.group(2)) if m else None,
             "img_date": a.get("img_date", ""), "temporal_stable": a.get("temporal_stable", "").strip().upper()[:1],
             "confidence": a.get("confidence", "").strip().lower(), "subtype": a.get("subtype", ""), "notes": a.get("notes", ""), "geom_note": note, "geom": None}
        R, Fl = [], []
        if not m: R.append("name_not_'XY-HO-#### | cK'")
        if d["class_code"] not in (0, 1, 2): R.append("invalid_class")
        if d["confidence"] == "low": R.append("confidence_low")
        if d["confidence"] not in ("high", "medium", "low"): Fl.append("confidence_missing")
        ym = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})", d["img_date"]); d["img_year"] = int(ym.group(1)) if ym else None
        if not ym: Fl.append("img_date_missing_or_bad")
        elif not 2020 <= d["img_year"] <= 2022: Fl.append("img_year_outside_2020_2022")
        if d["temporal_stable"] not in ("Y", "N"): Fl.append("temporal_stable_missing")
        if g is None: R.append(f"degenerate:{note}")
        else:
            gu = v3m.utm(g); d["geom"] = gu; d["area_ha"] = gu.area / 1e4; d["cen_lon"], d["cen_lat"] = g.centroid.x, g.centroid.y
            if not g.within(xy): R.append("not_fully_inside_xinyuan")
            if d["area_ha"] < 0.1: R.append("area_lt_0.1ha")
            if d["area_ha"] > 50: R.append("area_gt_50ha")
            if gu.intersection(rem_u).area > 1.0: R.append("overlaps_removed_training_polygon")
            d["d_remaining_train_km"] = gu.distance(trn_u) / 1000
            if d["d_remaining_train_km"] < 2: R.append("lt_2km_from_remaining_training")
        d["R"], d["F"] = R, Fl; recs.append(d)
    df = pd.DataFrame(recs)
    if not len(df):
        sys.exit(f"No 'XY-HO-#### | cK' polygons found ({len(ignored)} candidate/reference placemarks ignored). Digitize polygons first.")
    if len(df):
        for i in df.index[df.poly_name.duplicated(keep="first")]: df.at[i, "R"] = df.at[i, "R"] + ["duplicate_id"]
        idx = [i for i in df.index if df.at[i, "geom"] is not None]
        for k, a in enumerate(idx):
            for b in idx[k + 1:]:
                ga, gb = df.at[a, "geom"], df.at[b, "geom"]
                if not ga.intersects(gb): continue
                inter = ga.intersection(gb).area
                if inter <= 1.0: continue
                if df.at[a, "class_code"] != df.at[b, "class_code"]:
                    for q in (a, b): df.at[q, "R"] = df.at[q, "R"] + ["overlap_class_conflict"]
                elif inter / min(ga.area, gb.area) > 0.5: df.at[b, "R"] = df.at[b, "R"] + [f"duplicate_geometry_of_{df.at[a, 'poly_name']}"]
                else:
                    for q in (a, b): df.at[q, "F"] = df.at[q, "F"] + ["partial_overlap_same_class"]
    df["excluded"] = df.R.map(bool); df["exclude_reason"] = df.R.map(";".join); df["qc_flags"] = df.F.map(";".join)
    kept = df[~df.excluded].copy()
    summ = {"kml": str(kml), "kml_sha256": sha(kml), "n_parsed_polygons": int(len(df)), "n_ignored_reference_placemarks": len(ignored),
            "n_kept": int(len(kept)), "n_excluded": int(df.excluded.sum()),
            "kept_by_class": {str(k): int(v) for k, v in kept.class_code.value_counts().sort_index().items()},
            "exclusions": df[df.excluded][["poly_name", "class_code", "exclude_reason"]].to_dict("records"),
            "n_flagged": int((df.qc_flags != "").sum()) if len(df) else 0,
            "n_apple_kept": int((kept.class_code == 1).sum()), "n_negative_kept": int((kept.class_code != 1).sum())}
    summ["underpowered"] = summ["n_apple_kept"] < 25 or summ["n_negative_kept"] < 25
    df.drop(columns=["geom", "R", "F"]).to_csv(OUT / "QC_xinyuan_holdout.csv", index=False, encoding="utf-8-sig")
    return kept, summ

def sample_points(kept):
    rows = []
    for r in kept.itertuples():
        g4 = gpd.GeoSeries([r.geom], crs=32644).to_crs(4326).iloc[0]
        n, sp = v3m.choose_n_points(float(r.area_ha), int(r.class_code))
        rng = np.random.default_rng((v3m.SEED + zlib.crc32(r.poly_name.encode()) % 100000) % (2**31 - 1))
        for i, (x, y) in enumerate(v3m.sample_points_in_poly(g4, n, sp, rng), 1):
            rows.append({"sample_id": f"{r.poly_name}-P{i:02d}", "poly_name": r.poly_name, "class_code": int(r.class_code), "lon": x, "lat": y, "area_ha": float(r.area_ha)})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ features (same GEE recipe as 39_retrain_expanded.stage_features)
def fetch_features(pts, cache_dir):
    import ee, gee_sample_features as g
    spec39 = importlib.util.spec_from_file_location("r39", REV / "39_retrain_expanded.py"); r39 = importlib.util.module_from_spec(spec39); spec39.loader.exec_module(r39)
    ee.Initialize(project="hallowed-chain-458005-g9"); cache_dir.mkdir(exist_ok=True, parents=True)
    ae = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2021-01-01", "2022-01-01").mosaic()
    a = r39._sample(ae, pts, 10, 40, cache_dir / "_ae_raw.csv").rename(columns={f"A{i:02d}": f"AE_{i:02d}" for i in range(64)})[["sample_id"] + AE]
    aoi = ee.Geometry.Rectangle([pts.lon.min() - 0.05, pts.lat.min() - 0.05, pts.lon.max() + 0.05, pts.lat.max() + 0.05])
    img = g.build_s2(2021, aoi).addBands(g.build_s1(2021, aoi)).addBands(g.build_dem(aoi))
    h = r39._sample(img, pts, 30, 80, cache_dir / "_s1s2dem_raw.csv")
    a["sample_id"] = a.sample_id.astype(str); h["sample_id"] = h.sample_id.astype(str)
    return pts.merge(a, on="sample_id", how="left").merge(h, on="sample_id", how="left")

# ------------------------------------------------------------------ evaluation
def load_model(bid, sd): return joblib.load(MOD / f"{bid}_xyfrozen_2021_seed{sd}.joblib")
def predict(b, df):
    X = df.copy()
    for c, v in b["hc_impute_median"].items():
        if c in X: X[c] = X[c].fillna(v)
    X = X[b["features"]].values.astype(float); m = b["model"]
    if b["hierarchical"]: return m["stage1_orchard"].predict_proba(X)[:, 1] * m["stage2_apple_given_orchard"].predict_proba(X)[:, 1]
    return m.predict_proba(X)[:, 1]

def evaluate(df, polymeta, tag):
    df = df[df[AE].notna().all(axis=1)].copy(); df["y"] = (df.class_code == 1).astype(int)
    if "point_source" not in df: df["point_source"] = "xinyuan_holdout"   # poly_table() needs it (bug found in attempt 1, see EVAL_ONCE.lock)
    tabs, per_seed = {}, {b: [] for b in MODELS}
    for bid in MODELS:
        for sd in SEEDS:
            h = df.copy(); h["prob"] = predict(load_model(bid, sd), h); h["pred"] = (h.prob >= 0.5).astype(int)
            pt = v3m.poly_table(h); d = pt.dropna(subset=["pred_major"]).sort_values("poly_name")
            per_seed[bid].append(f1_score(d.y, d.pred_major.astype(int), zero_division=0))
            if sd == 42: tabs[bid] = d; h.to_csv(OUT / f"pred_points_{bid}_{tag}_seed42.csv", index=False); pt.to_csv(OUT / f"pred_polys_{bid}_{tag}_seed42.csv", index=False)
    names = tabs["B3"].poly_name.values
    for b in MODELS: assert (tabs[b].poly_name.values == names).all()
    y = tabs["B3"].y.values.astype(int); n = len(y); rng = np.random.default_rng(42); BOOT = [rng.integers(0, n, n) for _ in range(1000)]
    res = {"n_polygons_evaluated": int(n), "n_apple": int(y.sum()), "n_negative": int(n - y.sum()), "n_points": int(len(df)), "models": {}}
    for b in MODELS:
        p = tabs[b].pred_major.values.astype(int); s = tabs[b].mean_prob.values; cc = tabs[b].class_code.values
        m = v3m.bin_metrics(y, p); m["AP_poly_meanprob"] = float(average_precision_score(y, s)) if 0 < y.sum() < n else None
        bs = {k: [] for k in ("F1", "precision", "recall", "AP")}
        for i in BOOT:
            bs["F1"].append(f1_score(y[i], p[i], zero_division=0)); bs["precision"].append(precision_score(y[i], p[i], zero_division=0))
            bs["recall"].append(recall_score(y[i], p[i], zero_division=0))
            if 0 < y[i].sum() < n: bs["AP"].append(average_precision_score(y[i], s[i]))
        m["bootstrap_95ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items() if v}
        pc = {}
        for c_, lab in [(1, "apple_recall"), (2, "other_orchard_rejected"), (0, "non_orchard_rejected")]:
            q = cc == c_; k = int((p[q] == 1).sum()) if c_ == 1 else int((p[q] == 0).sum())
            pc[lab] = {"k": k, "n": int(q.sum()), "wilson": list(v3m.wilson(k, int(q.sum())))}
        m["per_class"] = pc; m["F1_10seeds"] = {"mean": float(np.mean(per_seed[b])), "sd": float(np.std(per_seed[b], ddof=1))}
        # elevation-band apple recall (polygon mean DEM elevation from the 30 m feature)
        el = df.groupby("poly_name").elevation.mean() if "elevation" in df else None
        if el is not None:
            e = el.reindex(tabs[b].poly_name).values; eb = {}
            for lo, hi, l in BANDS:
                q = (cc == 1) & (e >= lo) & (e < hi); k = int((p[q] == 1).sum()); eb[l] = {"k": k, "n": int(q.sum()), "wilson": list(v3m.wilson(k, int(q.sum())))}
            m["apple_recall_by_elevation"] = eb
        res["models"][b] = m
    comps = {"PRIMARY_B3_minus_B0": ("B3", "B0"), "B6_minus_B3": ("B6", "B3"), "HIER_minus_B3": ("HIER", "B3")}
    res["paired_dF1"] = {}
    for k, (a_, b_) in comps.items():
        pa, pb = tabs[a_].pred_major.values.astype(int), tabs[b_].pred_major.values.astype(int)
        ds = [f1_score(y[i], pa[i], zero_division=0) - f1_score(y[i], pb[i], zero_division=0) for i in BOOT]
        lo, hi = float(np.quantile(ds, .025)), float(np.quantile(ds, .975))
        res["paired_dF1"][k] = {"delta": float(f1_score(y, pa, zero_division=0) - f1_score(y, pb, zero_division=0)), "ci95": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}
    # temporal-stable subset (descriptive)
    if polymeta is not None:
        ok = polymeta[(polymeta.temporal_stable == "Y") & polymeta.img_year.notna() & ((polymeta.img_year - 2021).abs() <= 1)].poly_name
        q = np.isin(names, ok.values); sub = {"n": int(q.sum()), "n_apple": int(y[q].sum())}
        if q.sum() and 0 < y[q].sum() < q.sum():
            for b in MODELS:
                p = tabs[b].pred_major.values.astype(int)[q]
                sub[b] = {"F1": float(f1_score(y[q], p, zero_division=0)), "precision": float(precision_score(y[q], p, zero_division=0)), "recall": float(recall_score(y[q], p, zero_division=0))}
        res["temporal_stable_subset"] = sub
    return res

def table(res):
    rows = []
    for b, m in res["models"].items():
        ci = m["bootstrap_95ci"]
        rows.append({"model": b, "F1 [95% CI]": f"{m['F1']:.3f} [{ci['F1'][0]:.3f}, {ci['F1'][1]:.3f}]",
                     "P [CI]": f"{m['precision']:.3f} [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}]",
                     "R [CI]": f"{m['recall']:.3f} [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}]",
                     "AP [CI]": (f"{m['AP_poly_meanprob']:.3f} [{ci['AP'][0]:.3f}, {ci['AP'][1]:.3f}]" if m["AP_poly_meanprob"] is not None else "n/a"),
                     "F1 10 seeds": f"{m['F1_10seeds']['mean']:.3f}±{m['F1_10seeds']['sd']:.3f}"})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kml", nargs="?"); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--qc-only", action="store_true"); g.add_argument("--evaluate", action="store_true"); g.add_argument("--selftest", action="store_true"); ap.add_argument("--resume-after-crash", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        verify_frozen(); rt = pd.read_csv(OUT / "training_removal_table.csv"); keepn = set(rt[~rt.dropped].name)
        pts = pd.read_csv(E / "sample_points_expanded.csv"); pts = pts[pts.poly_name.isin(keepn)]
        df = pts.merge(pd.read_csv(E / "poly_ae_features_2021_expanded.csv"), on="sample_id").merge(pd.read_csv(E / "poly_s1s2dem_features_2021_expanded.csv"), on="sample_id")
        res = evaluate(df, None, "SELFTEST_insample_training"); save(res, OUT / "selftest_insample_training_NOT_A_RESULT.json")
        print("SELFTEST (in-sample training fit, code check only):"); print(table(res).to_string()); sys.exit()
    assert a.kml, "kml path required"
    if LOCK.exists() and a.evaluate and a.resume_after_crash:
        lk0 = json.loads(LOCK.read_text())
        assert "finished" not in lk0 and not (OUT / "RESULTS_xinyuan_holdout_ONCE.json").exists(), "a completed evaluation exists; refusing"
        lk0.setdefault("crashed_attempts", []).append({"started": lk0.pop("started"), "error": "AttributeError: poly_table() needs column 'point_source' (absent in 新源 sample points); "
                       "crash occurred inside the first model's first seed BEFORE any metric/prediction file was written or displayed", "fixed_by": "add point_source column in evaluate()"})
        LOCK.write_text(json.dumps(lk0, ensure_ascii=False, indent=1)); RESUME = lk0
    elif LOCK.exists(): sys.exit(f"REFUSED: {LOCK} exists - the 新源 holdout has already been evaluated once.\n" + LOCK.read_text())
    kept, qs = qc(a.kml); save(qs, OUT / "qc_summary_xinyuan.json"); print(json.dumps(qs, ensure_ascii=False, indent=1, default=str))
    pts = sample_points(kept); pts.to_csv(OUT / "sample_points_xinyuan.csv", index=False)
    print("points:", len(pts), pts.groupby("class_code").size().to_dict())
    if a.qc_only: print("QC only - no features fetched, no model applied."); sys.exit()
    rec, man = verify_frozen()
    t0 = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lk = RESUME if "RESUME" in globals() else {}
    assert not lk or lk["kml_sha256"] == qs["kml_sha256"], "resume must use the identical KML"
    lk.update({"started": t0, "kml": str(a.kml), "kml_sha256": qs["kml_sha256"], "prereg_sha256": rec["sha256"]}); LOCK.write_text(json.dumps(lk, ensure_ascii=False, indent=1))
    df = fetch_features(pts, OUT / "feature_cache_xinyuan")
    res = evaluate(df, kept[["poly_name", "temporal_stable", "img_year"]], "xinyuan")
    res.update({"qc": qs, "prereg_sha256": rec["sha256"], "underpowered": qs["underpowered"], "threshold": 0.5,
                "n_points_missing_AE": int(df[AE].isna().any(axis=1).sum()), "evaluated_at": t0})
    save(res, OUT / "RESULTS_xinyuan_holdout_ONCE.json"); t = table(res); t.to_csv(OUT / "RESULTS_xinyuan_holdout_ONCE_table.csv", index=False, encoding="utf-8-sig")
    lk = json.loads(LOCK.read_text()); lk.update({"finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                  "results_sha256": sha(OUT / "RESULTS_xinyuan_holdout_ONCE.json")}); LOCK.write_text(json.dumps(lk, ensure_ascii=False, indent=1))
    print(t.to_string()); print(json.dumps(res["paired_dF1"], indent=1))
