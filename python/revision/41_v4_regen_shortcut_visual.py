# -*- coding: utf-8 -*-
"""v4: regenerate Fig.10 (shortcut diagnostics) and Fig.S2 (600-pt visual sensitivity) with the EXPANDED training set / new B3.
AE-only = saved B3_poly_2021_expanded.joblib (primary). Others trained with common.make_lgbm(seed=42), y=(class_code==1), same as 39_retrain_expanded.
Polygon metrics via 22_b3_poly_holdout_v3.poly_table / bin_metrics (identical to table_old_vs_expanded_v4)."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, joblib
REV = Path(__file__).resolve().parent; ROOT = REV.parents[1]
sys.path.insert(0, str(REV)); sys.path.insert(0, str(ROOT / "python"))
from common import make_lgbm
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
spec = importlib.util.spec_from_file_location("sd21", REV / "21_shortcut_diagnostics.py"); sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)
from sklearn.metrics import average_precision_score, roc_auc_score
E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; OUT = ROOT / "outputs/revision/v4_experiments/regen_v4_figs"; OUT.mkdir(exist_ok=True)
AE = [f"AE_{i:02d}" for i in range(64)]
pts = pd.read_csv(E / "sample_points_expanded.csv")
df = pts.merge(pd.read_csv(E / "poly_ae_features_2021_expanded.csv")[["sample_id"] + AE], on="sample_id", how="left") \
        .merge(pd.read_csv(E / "poly_s1s2dem_features_2021_expanded.csv")[["sample_id", "elevation", "slope"]], on="sample_id", how="left")
df = df[df[AE].notna().all(1)]
train = df[df.split == "train"].copy(); hold = df[df.split == "val_holdout"].copy()
print("train", len(train), train.poly_name.nunique(), "hold", len(hold), "terrain NaN train/hold", train[["elevation","slope"]].isna().any(axis=1).sum(), hold[["elevation","slope"]].isna().any(axis=1).sum())
med = train[["elevation", "slope"]].median()
for d in (train, hold): d[["elevation", "slope"]] = d[["elevation", "slope"]].fillna(med)
hold["y"] = (hold.class_code == 1).astype(int); ytr = (train.class_code == 1).astype(int).values
b3 = joblib.load(E / "models/B3_poly_2021_expanded.joblib"); b3m = b3["model"] if isinstance(b3, dict) else b3
sets = [("AE-only", AE), ("DEM-only", ["elevation", "slope"]), ("lat/lon-only", ["lon", "lat"]),
        ("AE+DEM", AE + ["elevation", "slope"]), ("AE+lat/lon", AE + ["lon", "lat"])]
res = []
for name, cols in sets:
    if name == "AE-only": clf = b3m
    else: clf = make_lgbm(seed=42); clf.fit(train[cols].values.astype(float), ytr)
    h = hold.copy(); h["prob"] = clf.predict_proba(h[cols].values.astype(float))[:, 1]; h["pred"] = (h.prob >= 0.5).astype(int)
    pm = v3m.bin_metrics(h.y, h.pred); pm["AP"] = float(average_precision_score(h.y, h.prob))
    pt = v3m.poly_table(h); d = pt.dropna(subset=["pred_major"])
    pl = v3m.bin_metrics(d.y.values.astype(int), d.pred_major.values.astype(int)); pl["n_polys"] = int(len(d))
    pl["AP_poly_meanprob"] = float(average_precision_score(d.y.values.astype(int), d.mean_prob.values))
    res.append({"name": name, "features": cols if len(cols) < 10 else f"{len(cols)} feats", "point_level": pm, "poly_majority": pl})
    print(f"{name}: pt F1={pm['F1']:.3f} OA={pm['OA']:.3f} | poly n={pl['n_polys']} F1={pl['F1']:.3f} OA={pl['OA']:.3f} P={pl['precision']:.3f} R={pl['recall']:.3f}")
# all-apple baseline
npos = int(d.y.sum()); n = len(d); p = npos / n; res_base = {"poly_all_apple_F1": 2 * p / (1 + p), "n_pos": npos, "n": n}
print("all-apple baseline poly F1", res_base)
ae = res[0]
for r in res:
    r["delta_F1_point_vs_AE"] = r["point_level"]["F1"] - ae["point_level"]["F1"]; r["delta_F1_poly_vs_AE"] = r["poly_majority"]["F1"] - ae["poly_majority"]["F1"]
json.dump({"version": "v4_expanded", "train": "38 old + 239 new polygons (277 / %d pts)" % len(train), "seed": 42, "results": res, "baseline": res_base},
          open(OUT / "metrics_shortcut_diagnostics_v4_expanded.json", "w"), ensure_ascii=False, indent=1, default=float)
sd.SUPTITLE = "巩留跨县域挑战集（101 面[≥3 点]）地理捷径诊断：扩充训练集（277 面）"
sd.plot_shortcut_bars(res, ae["point_level"]["F1"], ae["poly_majority"]["F1"], OUT / "fig10_shortcut_diagnostics_v4_expanded.png")

# ---- visual 600-pt sensitivity with new B3
filled = pd.read_excel(ROOT / "data/interpretation_kit/Gongliu_independent_val_判读表_filled.xlsx")
filled["interp_class_code"] = pd.to_numeric(filled["interp_class_code"], errors="coerce")
filled["confidence"] = filled["confidence"].astype(str).str.strip()
conf = filled[(filled.interp_class_code != 9) & (filled.confidence.isin(["高", "中"]))].copy()
if "exclude_flag" in conf.columns:
    ef = conf["exclude_flag"]
    conf = conf[~ef.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])] if ef.dtype == object else conf[~ef.fillna(False).astype(bool)]
kit = pd.read_csv(ROOT / "outputs/revision/independent_val/kit_ae_features_2021.csv")
j = conf.merge(kit, on="sample_id", how="inner", suffixes=("", "_ae"))
j = j[j[AE].apply(lambda c: np.isfinite(pd.to_numeric(c, errors="coerce"))).all(axis=1)].copy()
j["y"] = (j.interp_class_code == 1).astype(int)
j["prob"] = b3m.predict_proba(j[AE].values.astype(float))[:, 1]; j["pred"] = (j.prob >= 0.5).astype(int)
m = v3m.bin_metrics(j.y, j.pred); m["AP"] = float(average_precision_score(j.y, j.prob)); m["AUC"] = float(roc_auc_score(j.y, j.prob))
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(j.y, j.pred, labels=[0, 1]).ravel()
m.update(dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), n=int(len(j)), n_pos=int(j.y.sum())))
byc = j.groupby("interp_class_code").agg(n=("pred", "size"), pred_apple=("pred", "sum")).reset_index(); byc["rate"] = byc.pred_apple / byc.n
byc.to_csv(OUT / "visual_fp_by_class_B3expanded.csv", index=False)
old = pd.read_csv(ROOT / "outputs/revision/polygon_samples/v3/visual_fp_by_class_v2model.csv")
json.dump({"model": "B3_poly_2021_expanded", "n": int(len(j)), "metrics": m, "by_class": byc.to_dict("records"), "old_model_by_class": old.to_dict("records")},
          open(OUT / "metrics_visual_indep_B3expanded.json", "w"), ensure_ascii=False, indent=1, default=float)
print(m); print(byc); print("old", old)
