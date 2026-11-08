# -*- coding: utf-8 -*-
"""v4 task 1 (P0): independent-domain feature comparison on the Gongliu polygon holdout.
Train: the same 38 non-Gongliu polygons / 389 points as B3_poly_2021_v2. Test: same 101 majority-vote polygons
(113 polygons / 521 points). Same LightGBM (common.make_lgbm), threshold 0.5, no tuning.
B0=S2, B2=S1+S2+DEM, B3=AE, B6=AE+S1+S2+DEM (feature definitions = AOMC ablation, utils.feature_columns; 2021).
Missing S1/S2 values are imputed with the TRAINING median only."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef, roc_auc_score)
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(REV))
from common import make_lgbm, setup_matplotlib_zh
from utils import feature_columns
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
PS = ROOT / "outputs/revision/polygon_samples"; OUT = ROOT / "outputs/revision/v4_experiments"
SETS = {"B0": ["s2"], "B2": ["s1", "s2", "dem"], "B3": ["ae"], "B6": ["ae", "s1", "s2", "dem"]}
NAMES = {"B0": "B0 S2", "B2": "B2 S1+S2+DEM", "B3": "B3 AE", "B6": "B6 AE+S1+S2+DEM"}
SEEDS = [42] + list(range(1, 10))
pts = pd.read_csv(PS / "sample_points_v3.csv")
ae = pd.read_csv(PS / "poly_ae_features_2021_v3.csv")
AE = [f"AE_{i:02d}" for i in range(64)]
hc = pd.read_csv(OUT / "poly_s1s2dem_features_2021_v4.csv")
df = pts.merge(ae[["sample_id"] + AE], on="sample_id", how="left").merge(hc, on="sample_id", how="left")
assert df[AE].notna().all().all()
train = df[df.split == "train"].copy(); hold = df[df.split == "val_holdout"].copy()
assert len(train) == 389 and len(hold) == 521
allc = feature_columns({}, ["s1", "s2", "dem"])
nan_report = {"train": {c: int(train[c].isna().sum()) for c in allc if train[c].isna().any()},
              "holdout": {c: int(hold[c].isna().sum()) for c in allc if hold[c].isna().any()}}
med = train[allc].median()
for d in (train, hold): d[allc] = d[allc].fillna(med)
ytr = train.class_code.values.astype(int); hold["y"] = (hold.class_code == 1).astype(int)

def run(bid, seed):
    cols = feature_columns({}, SETS[bid])
    clf = make_lgbm(seed=seed); clf.fit(train[cols].values.astype(float), ytr)
    h = hold.copy(); h["prob"] = clf.predict_proba(h[cols].values.astype(float))[:, 1]; h["pred"] = (h.prob >= 0.5).astype(int)
    pt = v3m.poly_table(h); return h, pt, clf

# sanity: B3 seed 42 retrained == saved model predictions
h3, pt3, clf3 = run("B3", 42)
saved = joblib.load(PS / "models/B3_poly_2021_v2.joblib")["model"]
same = np.allclose(saved.predict_proba(hold[AE].values.astype(float))[:, 1], h3.prob.values)
rng = np.random.default_rng(42); BOOT = [rng.integers(0, 101, 101) for _ in range(1000)]

def poly_metrics(pt):
    d = pt.dropna(subset=["pred_major"]).sort_values("poly_name"); y = d.y.values.astype(int); p = d.pred_major.values.astype(int); s = d.mean_prob.values
    m = v3m.bin_metrics(y, p); m["AP_poly_meanprob"] = float(average_precision_score(y, s)); m["AUC_poly_meanprob"] = float(roc_auc_score(y, s))
    bs = {k: [] for k in ("F1", "precision", "recall", "OA", "balanced_accuracy", "MCC", "AP_poly_meanprob")}
    for i in BOOT:
        a, b, c = y[i], p[i], s[i]
        bs["F1"].append(f1_score(a, b, zero_division=0)); bs["precision"].append(precision_score(a, b, zero_division=0)); bs["recall"].append(recall_score(a, b, zero_division=0))
        bs["OA"].append(accuracy_score(a, b)); bs["balanced_accuracy"].append(balanced_accuracy_score(a, b)); bs["MCC"].append(matthews_corrcoef(a, b)); bs["AP_poly_meanprob"].append(average_precision_score(a, c))
    m["bootstrap_95ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items()}
    pc = {}
    for cc, lab in [(1, "apple_recall"), (2, "other_orchard_rejected"), (0, "non_orchard_rejected")]:
        q = d[d.class_code == cc]; val = (q.pred_major == 1).mean() if cc == 1 else (q.pred_major == 0).mean(); k = int(round(val * len(q)))
        lo, hi = v3m.wilson(k, len(q)); pc[lab] = {"rate": float(val), "k": k, "n": int(len(q)), "wilson": [lo, hi]}
    m["per_class"] = pc
    return m, (y, p)

def point_metrics(h):
    m = v3m.bin_metrics(h.y, h.pred); m["AP"] = float(average_precision_score(h.y, h.prob)); m["AUC"] = float(roc_auc_score(h.y, h.prob)); return m

res = {"design": __doc__, "seeds": SEEDS, "nan_imputation_train_median": nan_report, "B3_seed42_equals_saved_model": bool(same),
       "n_train_points": 389, "n_train_polygons": int(train.poly_name.nunique()), "n_holdout_points": 521, "models": {}}
preds = {}
for bid in SETS:
    per_seed = []
    for sd in SEEDS:
        h, pt, _ = run(bid, sd)
        pm, yp = poly_metrics(pt); pl = point_metrics(h)
        per_seed.append({"seed": sd, "poly": pm, "point": pl})
        if sd == 42:
            preds[bid] = yp; h[["sample_id", "poly_name", "class_code", "y", "prob", "pred"]].to_csv(OUT / f"pred_points_{bid}_seed42.csv", index=False)
            pt.to_csv(OUT / f"pred_polys_{bid}_seed42.csv", index=False)
    agg = {}
    for k in ("F1", "precision", "recall", "MCC", "OA", "balanced_accuracy", "AP_poly_meanprob"):
        v = np.array([r["poly"][k] for r in per_seed]); agg["poly_" + k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
    for k in ("F1", "AP"):
        v = np.array([r["point"][k] for r in per_seed]); agg["point_" + k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    res["models"][bid] = {"features": SETS[bid], "n_features": len(feature_columns({}, SETS[bid])), "seed42": per_seed[0], "seed_summary": agg,
                          "per_seed_poly_F1": [r["poly"]["F1"] for r in per_seed]}
    print(bid, "seed42 polyF1=%.3f P=%.3f R=%.3f AP=%.3f | seeds F1 %.3f±%.3f" % (per_seed[0]["poly"]["F1"], per_seed[0]["poly"]["precision"], per_seed[0]["poly"]["recall"],
          per_seed[0]["poly"]["AP_poly_meanprob"], agg["poly_F1"]["mean"], agg["poly_F1"]["sd"]), flush=True)
# paired bootstrap delta F1 (seed 42), same polygon resamples
y3, p3 = preds["B3"]
for bid in SETS:
    if bid == "B3": continue
    y, p = preds[bid]; assert (y == y3).all()
    ds = [f1_score(y[i], p[i], zero_division=0) - f1_score(y[i], p3[i], zero_division=0) for i in BOOT]
    d0 = f1_score(y, p) - f1_score(y3, p3)
    res["models"][bid]["paired_delta_F1_vs_B3_seed42"] = {"delta": float(d0), "ci95": [float(np.quantile(ds, .025)), float(np.quantile(ds, .975))],
                                                          "p_boot_delta_le_0": float(np.mean(np.array(ds) <= 0)), "n_boot": 1000}
    # seed-averaged delta
    dsd = np.array(res["models"][bid]["per_seed_poly_F1"]) - np.array(res["models"]["B3"]["per_seed_poly_F1"])
    res["models"][bid]["delta_F1_vs_B3_per_seed"] = {"mean": float(dsd.mean()), "sd": float(dsd.std(ddof=1)), "values": dsd.tolist()}
json.dump(res, open(OUT / "metrics_indep_feature_compare_v4.json", "w"), ensure_ascii=False, indent=1)
# table
rows = []
for bid in SETS:
    m = res["models"][bid]; p = m["seed42"]["poly"]; ci = p["bootstrap_95ci"]; pl = m["seed42"]["point"]; ss = m["seed_summary"]
    dd = m.get("paired_delta_F1_vs_B3_seed42")
    rows.append({"model": NAMES[bid], "n_feat": m["n_features"],
                 "poly_F1 [95%CI]": f"{p['F1']:.3f} [{ci['F1'][0]:.3f}, {ci['F1'][1]:.3f}]",
                 "P [CI]": f"{p['precision']:.3f} [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}]",
                 "R [CI]": f"{p['recall']:.3f} [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}]",
                 "MCC [CI]": f"{p['MCC']:.3f} [{ci['MCC'][0]:.3f}, {ci['MCC'][1]:.3f}]",
                 "OA [CI]": f"{p['OA']:.3f} [{ci['OA'][0]:.3f}, {ci['OA'][1]:.3f}]",
                 "BA [CI]": f"{p['balanced_accuracy']:.3f} [{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]",
                 "poly_AP [CI]": f"{p['AP_poly_meanprob']:.3f} [{ci['AP_poly_meanprob'][0]:.3f}, {ci['AP_poly_meanprob'][1]:.3f}]",
                 "point_F1": f"{pl['F1']:.3f}", "point_AP": f"{pl['AP']:.3f}",
                 "TP/FP/FN/TN": "{TP}/{FP}/{FN}/{TN}".format(**p["confusion_matrix"]),
                 "other_orch_rej": f"{p['per_class']['other_orchard_rejected']['k']}/{p['per_class']['other_orchard_rejected']['n']}",
                 "non_orch_rej": f"{p['per_class']['non_orchard_rejected']['k']}/{p['per_class']['non_orchard_rejected']['n']}",
                 "dF1 vs B3 [CI]": "—" if dd is None else f"{dd['delta']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]",
                 "poly_F1 10 seeds": f"{ss['poly_F1']['mean']:.3f}±{ss['poly_F1']['sd']:.3f}",
                 "poly_AP 10 seeds": f"{ss['poly_AP_poly_meanprob']['mean']:.3f}±{ss['poly_AP_poly_meanprob']['sd']:.3f}"})
tab = pd.DataFrame(rows); tab.to_csv(OUT / "table_indep_feature_compare_v4.csv", index=False, encoding="utf-8-sig"); print(tab.to_string())
# figure: forest plot F1 and delta F1
plt = setup_matplotlib_zh(); plt.rcParams["font.size"] = 12
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
order = ["B0", "B2", "B6", "B3"]; col = {"B0": "#7f7f7f", "B2": "#8c564b", "B6": "#1f77b4", "B3": "#d62728"}
ax = axs[0]
for i, bid in enumerate(order):
    p = res["models"][bid]["seed42"]["poly"]; ci = p["bootstrap_95ci"]["F1"]; ss = res["models"][bid]["seed_summary"]["poly_F1"]
    ax.errorbar(p["F1"], i, xerr=[[p["F1"] - ci[0]], [ci[1] - p["F1"]]], fmt="o", color=col[bid], ms=8, capsize=4, lw=1.8)
    ax.plot(res["models"][bid]["per_seed_poly_F1"], [i + 0.18] * len(SEEDS), "|", color=col[bid], ms=9, alpha=0.7)
    ax.text(1.0, i, f"{p['F1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", va="center", ha="right", fontsize=10)
base = 2 * 36 / (36 + 101); ax.axvline(base, color="k", ls=":", lw=1); ax.text(base, -0.75, " 全判苹果基线", fontsize=9, va="bottom")
ax.set_yticks(range(4)); ax.set_yticklabels([NAMES[b] for b in order]); ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.8, 3.6)
ax.set_xlabel("面级 F1（面 bootstrap 95% CI；竖线为10个随机种子）"); ax.set_title("(a) 巩留园块空间留出：面级 F1（n=101）", loc="left", fontsize=12); ax.grid(axis="x", alpha=0.3)
ax = axs[1]; ob = ["B0", "B2", "B6"]
for i, bid in enumerate(ob):
    d = res["models"][bid]["paired_delta_F1_vs_B3_seed42"]
    ax.errorbar(d["delta"], i, xerr=[[d["delta"] - d["ci95"][0]], [d["ci95"][1] - d["delta"]]], fmt="s", color=col[bid], ms=8, capsize=4, lw=1.8)
    ax.text(ax.get_xlim()[1] if False else 0.0, i + 0.28, f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", ha="center", fontsize=10)
ax.axvline(0, color="k", lw=1); ax.set_yticks(range(3)); ax.set_yticklabels([NAMES[b] + " − B3" for b in ob]); ax.set_ylim(-0.6, 2.7)
ax.set_xlabel("ΔF1（配对面 bootstrap 95% CI，seed 42）"); ax.set_title("(b) 相对 AE-only（B3）的面级 ΔF1", loc="left", fontsize=12); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_indep_feature_compare_v4.png", dpi=300); print("fig saved")
