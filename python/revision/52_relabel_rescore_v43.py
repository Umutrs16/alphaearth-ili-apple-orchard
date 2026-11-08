# -*- coding: utf-8 -*-
"""v4.3 main-label change on the Gongliu challenge set (user's explicit decision 2026-09-28: wild apple forest = non-orchard).
Main labels v4.3: YL-POLY-0059/0060/0123/0124/0125 c1->c0 (annotated 野苹果林), YL-POLY-0083 c1->c0 (防护林), YL-POLY-0082 c2->c0 (农田).
Tentative notes (0126/0138/0081/0142) are NOT applied here (sensitivity only).
NO model is changed: training data are untouched (the holdout never enters training). The 10-seed repeats (seeds 42,1..9) of v4 are
regenerated deterministically from the identical training data; the script VERIFIES that (a) seed-42 holdout probabilities equal the frozen
joblib models' predictions and (b) with the ORIGINAL labels every per-seed polygon F1 equals the stored v4 value. Then everything is re-scored
with the v4.3 labels. Same metrics / bootstrap (1000, rng 42 over the 101 majority-vote polygons) as 34_/39_."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score)
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(REV))
from common import make_lgbm
from utils import feature_columns
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
PS = ROOT / "outputs/revision/polygon_samples"; V4 = ROOT / "outputs/revision/v4_experiments"; E = V4 / "retrain_expanded"; ES = V4 / "retrain_expanded_sens2020_2022"
OUT = ROOT / "outputs/revision/v43/relabel"; OUT.mkdir(parents=True, exist_ok=True)
RELABEL = {"YL-POLY-0059": 0, "YL-POLY-0060": 0, "YL-POLY-0123": 0, "YL-POLY-0124": 0, "YL-POLY-0125": 0, "YL-POLY-0083": 0, "YL-POLY-0082": 0}
AE = [f"AE_{i:02d}" for i in range(64)]; HC = feature_columns({}, ["s1", "s2", "dem"])
SETS = {"B0": ["s2"], "B2": ["s1", "s2", "dem"], "B3": ["ae"], "B6": ["ae", "s1", "s2", "dem"]}; SEEDS = [42] + list(range(1, 10))
rng = np.random.default_rng(42); BOOT = [rng.integers(0, 101, 101) for _ in range(1000)]

def load_expanded(d):
    pts = pd.read_csv(d / "sample_points_expanded.csv")
    df = pts.merge(pd.read_csv(d / "poly_ae_features_2021_expanded.csv"), on="sample_id", how="left").merge(pd.read_csv(d / "poly_s1s2dem_features_2021_expanded.csv"), on="sample_id", how="left")
    df = df[df[AE].notna().all(1)]; tr = df[df.split == "train"].copy(); ho = df[df.split == "val_holdout"].copy()
    med = tr[HC].median()
    for x in (tr, ho): x[HC] = x[HC].fillna(med)
    return tr, ho, (tr.class_code == 1).astype(int).values
def load_old():
    pts = pd.read_csv(PS / "sample_points_v3.csv"); ae = pd.read_csv(PS / "poly_ae_features_2021_v3.csv"); hc = pd.read_csv(V4 / "poly_s1s2dem_features_2021_v4.csv")
    df = pts.merge(ae[["sample_id"] + AE], on="sample_id", how="left").merge(hc, on="sample_id", how="left")
    tr = df[df.split == "train"].copy(); ho = df[df.split == "val_holdout"].copy(); med = tr[HC].median()
    for x in (tr, ho): x[HC] = x[HC].fillna(med)
    return tr, ho, tr.class_code.values.astype(int)

def fit_predict(tr, ho, ytr, bid, sd):
    if bid == "HIER":
        cols = feature_columns({}, ["ae"])
        c1 = make_lgbm(seed=sd); c1.fit(tr[cols].values.astype(float), (tr.class_code > 0).astype(int).values)
        orc = tr[tr.class_code > 0]; c2 = make_lgbm(seed=sd); c2.fit(orc[cols].values.astype(float), (orc.class_code == 1).astype(int).values)
        X = ho[cols].values.astype(float); return c1.predict_proba(X)[:, 1] * c2.predict_proba(X)[:, 1]
    cols = feature_columns({}, SETS[bid]); c = make_lgbm(seed=sd); c.fit(tr[cols].values.astype(float), ytr); return c.predict_proba(ho[cols].values.astype(float))[:, 1]

def relabeled(ho, lab):
    h = ho.copy()
    if lab == "v43":
        for k, v in RELABEL.items(): assert (h.poly_name == k).any(), k; h.loc[h.poly_name == k, "class_code"] = v
    h["y"] = (h.class_code == 1).astype(int); return h

def poly_metrics(h, boot=True):
    pt = v3m.poly_table(h); d = pt.dropna(subset=["pred_major"]).sort_values("poly_name"); assert len(d) == 101
    y, p, s = d.y.values.astype(int), d.pred_major.values.astype(int), d.mean_prob.values
    m = v3m.bin_metrics(y, p); m["AP_poly_meanprob"] = float(average_precision_score(y, s)); m["AUC_poly_meanprob"] = float(roc_auc_score(y, s))
    m["AP_poly_fracpos"] = float(average_precision_score(y, d.frac_pred_apple.values))
    bs = {k: [] for k in ("F1", "precision", "recall", "OA", "balanced_accuracy", "MCC", "AP_poly_meanprob")}
    for i in (BOOT if boot else []):
        a, b, c = y[i], p[i], s[i]
        bs["F1"].append(f1_score(a, b, zero_division=0)); bs["precision"].append(precision_score(a, b, zero_division=0)); bs["recall"].append(recall_score(a, b, zero_division=0))
        bs["OA"].append(accuracy_score(a, b)); bs["balanced_accuracy"].append(balanced_accuracy_score(a, b)); bs["MCC"].append(matthews_corrcoef(a, b)); bs["AP_poly_meanprob"].append(average_precision_score(a, c))
    m["bootstrap_95ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items()} if boot else None
    k_ = int(((y == p)).sum()); m["wilson_OA"] = list(v3m.wilson(k_, len(y)))
    pc = {}
    for cc, lab in [(1, "apple_recall"), (2, "other_orchard_rejected"), (0, "non_orchard_rejected")]:
        q = d[d.class_code == cc]; k = int((q.pred_major == 1).sum()) if cc == 1 else int((q.pred_major == 0).sum())
        pc[lab] = {"k": k, "n": int(len(q)), "rate": k / len(q) if len(q) else None, "wilson": list(v3m.wilson(k, len(q)))}
    m["per_class"] = pc; return m, (y, p), pt
def point_metrics(h):
    m = v3m.bin_metrics(h.y, h.pred); m["AP"] = float(average_precision_score(h.y, h.prob)); m["AUC"] = float(roc_auc_score(h.y, h.prob)); return m

CONF = {"expanded": (lambda: load_expanded(E), list(SETS) + ["HIER"], E / "metrics_expanded_retrain.json", "per_seed_poly_F1", E / "models", "_poly_2021_expanded.joblib"),
        "sens2020_2022": (lambda: load_expanded(ES), list(SETS) + ["HIER"], ES / "metrics_expanded_retrain.json", "per_seed_poly_F1", ES / "models", "_poly_2021_expanded.joblib"),
        "old38": (load_old, list(SETS), V4 / "metrics_indep_feature_compare_v4.json", "per_seed_poly_F1", None, None)}
verify = {}; results = {"orig": {}, "v43": {}}; keep = {"orig": {}, "v43": {}}
for cname, (loader, bids, ref_json, key, mdir, suf) in CONF.items():
    tr, ho, ytr = loader(); assert len(ho) == 521; ref = json.load(open(ref_json))["models"]
    for bid in bids:
        per = {"orig": [], "v43": []}
        for sd in SEEDS:
            prob = fit_predict(tr, ho, ytr, bid, sd)
            if sd == 42:
                if mdir is not None:
                    jm = joblib.load(mdir / f"{bid}{suf}"); cols = jm["features"]; X = ho[cols].values.astype(float)
                    fz = (jm["model"]["stage1_orchard"].predict_proba(X)[:, 1] * jm["model"]["stage2_apple_given_orchard"].predict_proba(X)[:, 1]) if bid == "HIER" else jm["model"].predict_proba(X)[:, 1]
                    verify[f"{cname}/{bid}/seed42_equals_frozen_joblib"] = bool(np.allclose(fz, prob))
                sv = pd.read_csv((E if cname == "expanded" else ES) / f"pred_points_{bid}_expanded_seed42.csv") if cname != "old38" else pd.read_csv(V4 / f"pred_points_{bid}_seed42.csv")
                verify[f"{cname}/{bid}/seed42_equals_saved_pred_points"] = bool(np.allclose(sv.set_index("sample_id").loc[ho.sample_id, "prob"].values, prob))
            for lab in ("orig", "v43"):
                h = relabeled(ho, lab); h["prob"] = prob; h["pred"] = (prob >= 0.5).astype(int)
                pm, yp, pt = poly_metrics(h, boot=(sd == 42)); per[lab].append({"seed": sd, "poly": pm, "point": point_metrics(h)})
                if sd == 42:
                    keep[lab][(cname, bid)] = yp
                    od = OUT / lab / cname; od.mkdir(parents=True, exist_ok=True)
                    pt.to_csv(od / f"pred_polys_{bid}_seed42.csv", index=False); h[["sample_id", "poly_name", "class_code", "y", "prob", "pred"]].to_csv(od / f"pred_points_{bid}_seed42.csv", index=False)
        stored = ref[bid][key]; got = [r["poly"]["F1"] for r in per["orig"]]
        verify[f"{cname}/{bid}/orig_label_10seed_F1_reproduced"] = bool(np.allclose(stored, got))
        for lab in ("orig", "v43"):
            agg = {}
            for k in ("F1", "precision", "recall", "MCC", "OA", "balanced_accuracy", "AP_poly_meanprob"):
                v = np.array([r["poly"][k] for r in per[lab]]); agg["poly_" + k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
            results[lab].setdefault(cname, {})[bid] = {"seed42": per[lab][0], "seed_summary": agg, "per_seed_poly_F1": [r["poly"]["F1"] for r in per[lab]]}
        print(cname, bid, {k: v for k, v in verify.items() if k.startswith(f"{cname}/{bid}/")}, "v43 F1=%.3f" % results["v43"][cname][bid]["seed42"]["poly"]["F1"], flush=True)
    # paired dF1
for lab in ("orig", "v43"):
    for cname in CONF:
        for bid in results[lab][cname]:
            y, p = keep[lab][(cname, bid)]
            for ref in [(cname, "B3"), ("old38", "B3")]:
                if ref == (cname, bid): continue
                yr, pr = keep[lab][ref]; assert (yr == y).all()
                ds = [f1_score(y[i], p[i], zero_division=0) - f1_score(y[i], pr[i], zero_division=0) for i in BOOT]
                tag = "vs_sameTrain_B3" if ref[0] == cname else "vs_old38_B3"
                results[lab][cname][bid]["paired_dF1_" + tag] = {"delta": float(f1_score(y, p) - f1_score(yr, pr)), "ci95": [float(np.quantile(ds, .025)), float(np.quantile(ds, .975))],
                                                                 "ci_excludes_0": bool(np.quantile(ds, .025) > 0 or np.quantile(ds, .975) < 0)}
json.dump({"design": __doc__, "relabel": RELABEL, "verification": verify, "results": results}, open(OUT / "metrics_relabel_v43.json", "w"), ensure_ascii=False, indent=1, default=float)
assert all(verify.values()), {k: v for k, v in verify.items() if not v}
rows = []
for lab in ("orig", "v43"):
    for cname in CONF:
        for bid, m in results[lab][cname].items():
            p = m["seed42"]["poly"]; c = p["bootstrap_95ci"]; pc = p["per_class"]; d = m.get("paired_dF1_vs_sameTrain_B3")
            rows.append({"labels": lab, "train": cname, "model": bid, "n": p["n"], "n_apple": p["n_pos"], "F1": p["F1"], "F1_lo": c["F1"][0], "F1_hi": c["F1"][1],
                         "P": p["precision"], "P_lo": c["precision"][0], "P_hi": c["precision"][1], "R": p["recall"], "R_lo": c["recall"][0], "R_hi": c["recall"][1],
                         "AP": p["AP_poly_meanprob"], "AP_lo": c["AP_poly_meanprob"][0], "AP_hi": c["AP_poly_meanprob"][1], "OA": p["OA"], "OA_wlo": p["wilson_OA"][0], "OA_whi": p["wilson_OA"][1],
                         "BA": p["balanced_accuracy"], "MCC": p["MCC"], "AUC": p["AUC_poly_meanprob"], **{k: v for k, v in p["confusion_matrix"].items()},
                         "apple_k": pc["apple_recall"]["k"], "apple_n": pc["apple_recall"]["n"], "oth_k": pc["other_orchard_rejected"]["k"], "oth_n": pc["other_orchard_rejected"]["n"],
                         "non_k": pc["non_orchard_rejected"]["k"], "non_n": pc["non_orchard_rejected"]["n"],
                         "dF1_vs_B3": d["delta"] if d else np.nan, "dF1_lo": d["ci95"][0] if d else np.nan, "dF1_hi": d["ci95"][1] if d else np.nan,
                         "dF1_vs_old38B3": m.get("paired_dF1_vs_old38_B3", {}).get("delta", np.nan), "dF1_old_lo": m.get("paired_dF1_vs_old38_B3", {}).get("ci95", [np.nan] * 2)[0],
                         "dF1_old_hi": m.get("paired_dF1_vs_old38_B3", {}).get("ci95", [np.nan] * 2)[1],
                         "F1_10s_mean": m["seed_summary"]["poly_F1"]["mean"], "F1_10s_sd": m["seed_summary"]["poly_F1"]["sd"], "point_F1": m["seed42"]["point"]["F1"], "point_AP": m["seed42"]["point"]["AP"]})
T = pd.DataFrame(rows); T.to_csv(OUT / "table_relabel_v43.csv", index=False)
print(T[["labels", "train", "model", "n_apple", "F1", "F1_lo", "F1_hi", "P", "R", "AP", "dF1_vs_B3", "dF1_lo", "dF1_hi", "F1_10s_mean"]].round(3).to_string())
