# -*- coding: utf-8 -*-
"""v4 task 2: polygon-level PR for the Gongliu 101-polygon primary holdout (B3_poly_2021_v2, cached predictions).
Scores: polygon mean probability (primary) and fraction of points predicted apple (alternative).
Also writes the point-level PR as a supplementary figure."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score
ROOT = Path("/workspace/apple_orchard_project"); V3 = ROOT / "outputs/revision/polygon_samples/v3"; OUT = ROOT / "outputs/revision/v4_experiments"
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "python/revision"))
from common import setup_matplotlib_zh
pt = pd.read_csv(V3 / "poly_predictions_v3.csv"); hold = pd.read_csv(V3 / "predictions_holdout_v3.csv")
met = json.load(open(V3 / "metrics_poly_holdout_v3.json"))
pp = pt.dropna(subset=["pred_major"]).copy(); y = pp.y.values.astype(int)
assert len(pp) == 101
rng = np.random.default_rng(42); res = {"n_polygons": int(len(pp)), "n_pos": int(y.sum()), "prevalence": float(y.mean())}
for sc in ("mean_prob", "frac_pred_apple"):
    s = pp[sc].values; aps = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): aps.append(average_precision_score(y[i], s[i]))
    pr = (s >= 0.5).astype(int)
    res[sc] = {"AP": float(average_precision_score(y, s)), "AP_bootstrap_95ci": [float(np.quantile(aps, .025)), float(np.quantile(aps, .975))],
               "AUC": float(roc_auc_score(y, s)),
               "operating_point_0.5": {"precision": float(precision_score(y, pr)), "recall": float(recall_score(y, pr)), "F1": float(f1_score(y, pr)),
                                       "rule": f"{sc} >= 0.5"}}
mv = pp.pred_major.astype(int).values
res["majority_vote_operating_point"] = {"precision": float(precision_score(y, mv)), "recall": float(recall_score(y, mv)), "F1": float(f1_score(y, mv)),
                                         "check_vs_v3_json": {"precision": met["poly_majority"]["precision"], "recall": met["poly_majority"]["recall"], "F1": met["poly_majority"]["F1"]}}
res["point_level_AP_v3"] = met["point_level"]["AP"]; res["note"] = "Bootstrap: 1000 polygon resamples, seed 42. No threshold tuning."
json.dump(res, open(OUT / "metrics_poly_pr_v4.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
# supplementary point-level PR
plt = setup_matplotlib_zh(); plt.rcParams["font.size"] = 12
fig, ax = plt.subplots(figsize=(6.4, 5.6))
yy, pr_ = hold.y.values, hold.prob.values; P, R, _ = precision_recall_curve(yy, pr_); apci = met["point_level"]["cluster_bootstrap_95ci"]["AP"]
ax.plot(R, P, color="#D62728", lw=2.2, label=f"v3 全部样点 (n={len(yy)}) AP={average_precision_score(yy, pr_):.3f}\n[面聚类 bootstrap {apci[0]:.3f}–{apci[1]:.3f}]")
s2 = hold[hold.point_source == "reused_v2"]; P2, R2, _ = precision_recall_curve(s2.y, s2.prob)
ax.plot(R2, P2, color="gray", lw=1.2, ls=":", label=f"前期18面子集 (n={len(s2)}) AP={average_precision_score(s2.y, s2.prob):.3f}")
ax.axhline(yy.mean(), color="k", ls="--", lw=0.8, label=f"基线（正例比例 {yy.mean():.2f}）")
pm = met["point_level"]; ax.plot(pm["recall"], pm["precision"], "o", ms=9, mfc="white", mec="#D62728", mew=2, label=f"阈值0.5（P={pm['precision']:.3f}, R={pm['recall']:.3f}）")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.set_xlabel("召回率 Recall"); ax.set_ylabel("精确率 Precision")
ax.legend(fontsize=9, loc="lower left"); fig.tight_layout(); fig.savefig(OUT / "figS_point_PR_holdout_v4.png", dpi=300); print("saved")
