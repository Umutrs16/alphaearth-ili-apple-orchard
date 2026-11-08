# -*- coding: utf-8 -*-
"""v4 retrain: Fig. 9 (main holdout validation, polygon-level PR in panel c) for the EXPANDED-training B3 (seed 42).
Re-uses 32b_fig09_polyPR_v4.py (same layout, official GS(2024)0650 boundaries) with inputs swapped to the expanded model;
panel (c) additionally shows the old B3_poly_2021_v2 polygon PR curve (grey) for reference. Metrics computed here, written to JSON."""
import json, sys, types, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; V4 = ROOT / "outputs/revision/v4_experiments"; R = V4 / "retrain_expanded"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "B3"
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(REV))
import importlib.util
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
TMP = Path(tempfile.mkdtemp())
pt = pd.read_csv(R / f"pred_polys_{MODEL}_expanded_seed42.csv")
pts = pd.read_csv(R / f"pred_points_{MODEL}_expanded_seed42.csv").merge(pd.read_csv(ROOT / "outputs/revision/polygon_samples/sample_points_v3.csv")[["sample_id", "lon", "lat", "point_source"]], on="sample_id")
man = pd.read_csv(R / "split_manifest_expanded.csv"); man = man[man.split.isin(["train", "val_holdout"])]
pp = pt.dropna(subset=["pred_major"]).sort_values("poly_name"); y = pp.y.values.astype(int); mv = pp.pred_major.values.astype(int); assert len(pp) == 101
bm = v3m.bin_metrics(y, mv); bs = v3m.poly_bootstrap(y, mv); lo, hi = v3m.wilson(int((y == mv).sum()), len(y))
t3 = {str(c): {"pred_apple": int(((pp.class_code == c) & (mv == 1)).sum()), "pred_non": int(((pp.class_code == c) & (mv == 0)).sum())} for c in (0, 1, 2)}
met = {"poly_majority": dict(bm, bootstrap_95ci=bs, wilson_95ci_OA={"lo": lo, "hi": hi}, three_class_by_pred=t3)}
rng = np.random.default_rng(42); prc = {"model": f"{MODEL}_poly_2021_expanded (seed 42)", "n_polygons": 101, "n_pos": int(y.sum()), "prevalence": float(y.mean())}
for sc in ("mean_prob", "frac_pred_apple"):
    s = pp[sc].values; aps = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): aps.append(average_precision_score(y[i], s[i]))
    pr = (s >= 0.5).astype(int)
    prc[sc] = {"AP": float(average_precision_score(y, s)), "AP_bootstrap_95ci": [float(np.quantile(aps, .025)), float(np.quantile(aps, .975))], "AUC": float(roc_auc_score(y, s)),
               "operating_point_0.5": {"precision": float(precision_score(y, pr)), "recall": float(recall_score(y, pr)), "F1": float(f1_score(y, pr)), "rule": f"{sc} >= 0.5"}}
prc["majority_vote_operating_point"] = {"precision": bm["precision"], "recall": bm["recall"], "F1": bm["F1"]}
prc["point_level"] = {"AP": float(average_precision_score(pts.y, pts.prob)), "AUC": float(roc_auc_score(pts.y, pts.prob))}
json.dump(prc, open(R / f"metrics_poly_pr_{MODEL}_expanded.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(prc, ensure_ascii=False, indent=1))
json.dump(met, open(TMP / "metrics_poly_holdout_v3.json", "w"), ensure_ascii=False)
pt.to_csv(TMP / "poly_predictions_v3.csv", index=False); pts.to_csv(TMP / "predictions_holdout_v3.csv", index=False); man.to_csv(TMP / "split_manifest_v3.csv", index=False)
src = (REV / "32b_fig09_polyPR_v4.py").read_text(encoding="utf-8")
src = src.replace('ROOT / "outputs/revision/v4_experiments/metrics_poly_pr_v4.json"', f'Path("{R}/metrics_poly_pr_{MODEL}_expanded.json")')
src = src.replace('FIG3 / "fig09_main_holdout_validation_v4.png"', f'Path("{R}/fig09_main_holdout_validation_v4_expanded{MODEL}.png")')
src = src.replace('ax.set_xlim(81.6, 83.45); ax.set_ylim(42.95, 43.92)', 'ax.set_xlim(80.3, 84.3); ax.set_ylim(42.7, 44.7)')
src = src.replace('ax.text(82.45, 43.02, "巩留县"', 'ax.text(82.45, 42.85, "巩留县"')
src = src.replace('(a) 空间划分：巩留县整体留出（面质心）', '(a) 空间划分：扩充训练面（非巩留）与巩留整体留出')
old_curve = '''    _o = pd.read_csv(V4O / "pred_polys_B3_seed42.csv").dropna(subset=["pred_major"])
    _Po, _Ro, _ = precision_recall_curve(_o.y.values.astype(int), _o.mean_prob.values)
    ax.step(_Ro, _Po, where="post", color="#9E9E9E", lw=1.3, alpha=0.9, label=f"原 B3（38 面训练）面均概率 AP={average_precision_score(_o.y.values.astype(int), _o.mean_prob.values):.3f}")
'''
src = src.replace('    ax.axhline(yp.mean(), color="k", ls=":"', old_curve + '    ax.axhline(yp.mean(), color="k", ls=":"')
src = src.replace('FIG3 = ROOT / "outputs" / "revision" / "v4_experiments"', 'FIG3 = ROOT / "outputs" / "revision" / "v4_experiments"; V4O = FIG3')
mod = types.ModuleType("f09x"); mod.__file__ = str(REV / "32b_fig09_polyPR_v4.py")
exec(compile(src, str(REV / "32b_fig09_polyPR_v4.py"), "exec"), mod.__dict__)
mod.V3 = TMP
mod.main()
