# -*- coding: utf-8 -*-
"""v4.1 (review E8): Fig. 9 with panel (d) = six equal-size, equal-scale cases labelled TP, TN, FP-other orchard, FP-non-orchard,
FN-high elevation, FN-low elevation (largest polygon of each type; elevation = median SRTM of its points).
Original: v4 retrain: Fig. 9 (main holdout validation, polygon-level PR in panel c) for the EXPANDED-training B3 (seed 42).
Re-uses 32b_fig09_polyPR_v4.py (same layout, official GS(2024)0650 boundaries) with inputs swapped to the expanded model;
panel (c) additionally shows the old B3_poly_2021_v2 polygon PR curve (grey) for reference. Metrics computed here, written to JSON."""
import json, sys, types, tempfile
from pathlib import Path
import numpy as np, pandas as pd
NEWCASES = '''    ELEV = pd.read_csv(ROOT / "outputs/revision/v4_experiments/retrain_expanded/poly_s1s2dem_features_2021_expanded.csv")[["sample_id", "elevation"]].merge(hold[["sample_id", "poly_name"]], on="sample_id").groupby("poly_name").elevation.median()
    p["elev"] = p.poly_name.map(ELEV)
    def pick2(q):
        return q.sort_values("area_ha", ascending=False).iloc[0] if len(q) else None
    spec_ = [("TP", "苹果园正确检出", p[(p.class_code == 1) & (p.out == "TP")]), ("TN", "非果园正确拒识", p[(p.class_code == 0) & (p.out == "TN")]),
             ("FP-其他果园", "其他果园误报为苹果", p[(p.class_code == 2) & (p.out == "FP")]), ("FP-非果园", "非果园误报为苹果", p[(p.class_code == 0) & (p.out == "FP")]),
             ("FN-高海拔", "高海拔苹果园漏报", p[(p.class_code == 1) & (p.out == "FN") & (p.elev >= 1200)]), ("FN-低海拔", "低海拔苹果园漏报", p[(p.class_code == 1) & (p.out == "FN") & (p.elev < 900)])]
    cases = []
    for lab_, desc_, q in spec_:
        r_ = pick2(q)
        if r_ is not None: r_ = r_.copy(); r_["lab"] = lab_; r_["desc"] = desc_; cases.append(r_)
    HALF = min(0.012, max(1.2 * max(geoms[c.poly_name].bounds[2] - geoms[c.poly_name].bounds[0], geoms[c.poly_name].bounds[3] - geoms[c.poly_name].bounds[1]) for c in cases) / 2 + 0.001)
'''
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
print('AP', prc['mean_prob']['AP'])
json.dump(met, open(TMP / "metrics_poly_holdout_v3.json", "w"), ensure_ascii=False)
pt.to_csv(TMP / "poly_predictions_v3.csv", index=False); pts.to_csv(TMP / "predictions_holdout_v3.csv", index=False); man.to_csv(TMP / "split_manifest_v3.csv", index=False)
src = (REV / "32b_fig09_polyPR_v4.py").read_text(encoding="utf-8")
src = src.replace('ROOT / "outputs/revision/v4_experiments/metrics_poly_pr_v4.json"', f'Path("{R}/metrics_poly_pr_{MODEL}_expanded.json")')
src = src.replace('FIG3 / "fig09_main_holdout_validation_v4.png"', 'Path("/workspace/apple_orchard_project/outputs/revision/v41_review/figures/fig09_main_holdout_validation_v41_expandedB3.png")')
src = src.replace('ax.set_xlim(81.6, 83.45); ax.set_ylim(42.95, 43.92)', 'ax.set_xlim(80.3, 84.3); ax.set_ylim(42.7, 44.7)')
src = src.replace('ax.text(82.45, 43.02, "巩留县"', 'ax.text(82.45, 42.85, "巩留县"')
src = src.replace('(a) 空间划分：巩留县整体留出（面质心）', '(a) 空间划分：扩充训练面（非巩留）与巩留整体留出')
old_curve = '''    _o = pd.read_csv(V4O / "pred_polys_B3_seed42.csv").dropna(subset=["pred_major"])
    _Po, _Ro, _ = precision_recall_curve(_o.y.values.astype(int), _o.mean_prob.values)
    ax.step(_Ro, _Po, where="post", color="#9E9E9E", lw=1.3, alpha=0.9, label=f"原 B3（38 面训练）面均概率 AP={average_precision_score(_o.y.values.astype(int), _o.mean_prob.values):.3f}")
'''
src = src.replace('    ax.axhline(yp.mean(), color="k", ls=":"', old_curve + '    ax.axhline(yp.mean(), color="k", ls=":"')
src = src.replace('FIG3 = ROOT / "outputs" / "revision" / "v4_experiments"', 'FIG3 = ROOT / "outputs" / "revision" / "v4_experiments"; V4O = FIG3')
src = src.replace('    cases = [pick(1, "TP"), pick(0, "TN"), pick(0, "FP"), pick(1, "FN"), pick(2, "FP"), pick(2, "TN")]\n', NEWCASES)
src = src.replace('half = max(0.0035, 1.3 * max(maxx - minx, maxy - miny))', 'half = HALF')
src = src.replace('ax.set_title(f"{r.poly_name} [{r.out}]  真={CLS_ZH[int(r.class_code)]}\\n"', 'ax.set_title(f"{r.lab}：{r.desc}\\n{r.poly_name}  真={CLS_ZH[int(r.class_code)]}  高程{r.elev:.0f} m\\n"')
src = src.replace('(d) 代表性留出面案例（Sentinel-2 2021 年 6–9 月中值真彩色；', '(d) 代表性留出面案例（6个窗口同一比例尺；Sentinel-2 2021 年 6–9 月中值真彩色；')
src = src.replace('hspace=0.38, wspace=0.18', 'hspace=0.55, wspace=0.12')
assert 'NEWCASES' not in src and 'HALF' in src and 'r.lab' in src
mod = types.ModuleType("f09x"); mod.__file__ = str(REV / "32b_fig09_polyPR_v4.py")
exec(compile(src, str(REV / "32b_fig09_polyPR_v4.py"), "exec"), mod.__dict__)
mod.V3 = TMP
mod.main()
