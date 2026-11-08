"""v4: robustness table (附表S1) + point-level cluster bootstrap for the expanded-training B3, same rules as 22_b3_poly_holdout_v3.stage_eval."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
REV = Path(__file__).resolve().parent; ROOT = REV.parents[1]; sys.path.insert(0, str(REV)); sys.path.insert(0, str(ROOT / "python"))
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; OUT = ROOT / "outputs/revision/v4_experiments/regen_v4_figs"
pp = pd.read_csv(E / "pred_points_B3_expanded_seed42.csv")
pts = pd.read_csv(E / "sample_points_expanded.csv")
dem = pd.read_csv(E / "poly_s1s2dem_features_2021_expanded.csv")[["sample_id", "elevation"]]
hold = pp.merge(pts[["sample_id", "lon", "lat", "area_ha", "point_source"]], on="sample_id").merge(dem, on="sample_id")
assert len(hold) == 521
pt = v3m.poly_table(hold)
res = {"model": "B3_poly_2021_expanded seed42"}
res["poly_majority"] = v3m.poly_block(pt, "pred_major")
res["poly_meanprob_all113"] = v3m.poly_block(pt, "pred_meanprob")
res["sens_all_polys_majority_incl_lt3pts"] = v3m.poly_block(pt, "pred_major_all")
res["sens_strict_inside_gongliu"] = v3m.poly_block(pt[~pt.poly_name.isin(["YL-POLY-0059", "YL-POLY-0061"])], "pred_major")
res["new_only"] = v3m.poly_block(pt[~pt.is_v2], "pred_major")
v2p = pt[pt.is_v2].dropna(subset=["pred_major"]); res["v2_18_subset"] = v3m.poly_block(v2p, "pred_major")
y, p, pr = hold.y.values, hold.pred.values, hold.prob.values
pm = v3m.bin_metrics(y, p); pm["AP"] = float(average_precision_score(y, pr)); pm["AUC"] = float(roc_auc_score(y, pr))
pm["per_class_point"] = {str(c): {"n": int((hold.class_code == c).sum()), "pred_apple_rate": float(hold[hold.class_code == c].pred.mean())} for c in (1, 2, 0)}
rng = np.random.default_rng(42); names = pt.poly_name.values; f1s, aps = [], []
grp = {n: hold.index[hold.poly_name == n].values for n in names}
for _ in range(1000):
    idx = np.concatenate([grp[n] for n in rng.choice(names, len(names))]); yy = hold.loc[idx, "y"].values
    f1s.append(f1_score(yy, hold.loc[idx, "pred"].values, zero_division=0))
    if 0 < yy.sum() < len(yy): aps.append(average_precision_score(yy, hold.loc[idx, "prob"].values))
pm["cluster_bootstrap_95ci"] = {"F1": [float(np.quantile(f1s, .025)), float(np.quantile(f1s, .975))], "AP": [float(np.quantile(aps, .025)), float(np.quantile(aps, .975))]}
res["point_level"] = pm
# error anatomy
d = pt.dropna(subset=["pred_major"]).merge(hold.groupby("poly_name").elevation.median().rename("elev_med"), left_on="poly_name", right_index=True)
fn = d[(d.y == 1) & (d.pred_major == 0)]; fp = d[(d.y == 0) & (d.pred_major == 1)]; tp = d[(d.y == 1) & (d.pred_major == 1)]
res["FN_polys"] = fn[["poly_name", "area_ha", "elev_med", "mean_prob", "frac_pred_apple"]].to_dict("records")
res["FP_polys"] = fp[["poly_name", "class_code", "area_ha", "elev_med", "mean_prob"]].to_dict("records")
res["TP_area_median"] = float(tp.area_ha.median()); res["FN_area_median"] = float(fn.area_ha.median())
json.dump(res, open(OUT / "metrics_robustness_B3expanded.json", "w"), ensure_ascii=False, indent=1, default=float)
def row(k):
    m = res[k]; w = m["wilson_95ci_OA"]; b = m["bootstrap_95ci"]["F1"]; c = m["confusion_matrix"]
    print(k, w["n"], f"OA {m['OA']:.3f} [{w['lo']:.3f},{w['hi']:.3f}] P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['F1']:.3f} [{b["lo"]:.3f},{b["hi"]:.3f}] BA {m['balanced_accuracy']:.3f} MCC {m['MCC']:.3f}", c, m["per_class"]["apple_recall"]["k"], m["per_class"].get("other_orchard_rejected_rate"), m["per_class"]["non_orchard_rejected_rate"]["k"])
for k in ["poly_majority", "poly_meanprob_all113", "sens_all_polys_majority_incl_lt3pts", "sens_strict_inside_gongliu", "new_only", "v2_18_subset"]: row(k)
print({k: v for k, v in pm.items() if k != "confusion_matrix"}, pm["confusion_matrix"])
print("FN", fn[["poly_name", "area_ha", "elev_med", "mean_prob"]].round(3).to_string()); print("FP", fp[["poly_name", "class_code", "area_ha", "elev_med", "mean_prob"]].round(3).to_string())
print("TP area med", res["TP_area_median"], "FN area med", res["FN_area_median"])
