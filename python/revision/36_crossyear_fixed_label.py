# -*- coding: utf-8 -*-
"""v4 task 3: fixed-label temporal embedding transfer (AOMC weak labels, internal consistency only).
Rows = label year r (AOMC r point set, own locations); columns = AlphaEarth feature year c.
Protocol per r: spatial-block split of the r point set (GroupShuffleSplit test 0.3 by 0.05-deg block_id, seed 42, as ablation);
LightGBM (common.make_lgbm, seed 42) trained on AE_r at training blocks; evaluated at held-out blocks with AE_c
(labels fixed to AOMC r). Variant 'retrain': train and test on AE_c (same blocks). Block-bootstrap 95% CI (500)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "python/revision"))
from common import make_lgbm, spatial_split, setup_matplotlib_zh
CY = ROOT / "outputs/revision/v4_experiments/crossyear"; AE = [f"AE_{i:02d}" for i in range(64)]
pts = pd.read_csv(CY / "label_points_2019_2021_with_pid.csv")
FY = [y for y in range(2017, 2026) if (CY / f"ae_{y}_at_labelpts.csv").exists()]
feat = {}
for y in FY:
    a = pd.read_csv(CY / f"ae_{y}_at_labelpts.csv").rename(columns={f"A{i:02d}": f"AE_{i:02d}" for i in range(64)})
    feat[y] = a.set_index("pid")[AE]
def bboot(y, p, blocks, n=500, seed=42):
    rng = np.random.default_rng(seed); ub = np.unique(blocks); idx_b = {b: np.where(blocks == b)[0] for b in ub}; v = []
    for _ in range(n):
        i = np.concatenate([idx_b[b] for b in rng.choice(ub, len(ub))]); v.append(f1_score(y[i], p[i], zero_division=0))
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
res = {"design": __doc__, "feature_years_available": FY, "transfer": {}, "retrain": {}, "n": {}}
for r in (2019, 2020, 2021):
    d = pts[pts.year == r].reset_index(drop=True)
    tr, te = spatial_split(d, test_size=0.3, seed=42)
    dtr, dte = d.iloc[tr], d.iloc[te]
    res["n"][str(r)] = {"train": int(len(dtr)), "test": int(len(dte)), "test_pos": int(dte.label.sum()), "test_blocks": int(dte.block_id.nunique())}
    Xtr = feat[r].loc[dtr.pid].values; ok_tr = np.isfinite(Xtr).all(1)
    clf = make_lgbm(seed=42); clf.fit(Xtr[ok_tr], dtr.label.values[ok_tr])
    res["transfer"][str(r)] = {}; res["retrain"][str(r)] = {}
    for c in FY:
        Xte = feat[c].reindex(dte.pid).values; ok = np.isfinite(Xte).all(1); y = dte.label.values[ok]
        prob = clf.predict_proba(Xte[ok])[:, 1]; pred = (prob >= 0.5).astype(int)
        res["transfer"][str(r)][str(c)] = {"F1": float(f1_score(y, pred)), "AUC": float(roc_auc_score(y, prob)), "n": int(ok.sum()),
                                           "F1_ci95": bboot(y, pred, dte.block_id.values[ok])}
        Xtc = feat[c].reindex(dtr.pid).values; okc = np.isfinite(Xtc).all(1)
        m2 = make_lgbm(seed=42); m2.fit(Xtc[okc], dtr.label.values[okc]); p2 = (m2.predict_proba(Xte[ok])[:, 1] >= 0.5).astype(int)
        res["retrain"][str(r)][str(c)] = {"F1": float(f1_score(y, p2)), "n": int(ok.sum())}
    print(r, {c: round(res["transfer"][str(r)][str(c)]["F1"], 3) for c in FY}, flush=True)
json.dump(res, open(CY.parent / "metrics_crossyear_fixed_label_v4.json", "w"), ensure_ascii=False, indent=1)
rows = [{"label_year": r, "feature_year": c, **{k: v for k, v in res["transfer"][str(r)][str(c)].items() if k != "F1_ci95"},
         "F1_lo": res["transfer"][str(r)][str(c)]["F1_ci95"][0], "F1_hi": res["transfer"][str(r)][str(c)]["F1_ci95"][1],
         "F1_retrain": res["retrain"][str(r)][str(c)]["F1"]} for r in (2019, 2020, 2021) for c in FY]
pd.DataFrame(rows).to_csv(CY.parent / "table_crossyear_fixed_label_v4.csv", index=False)
# figure 3x6 (2019-2024), colorbar in its own axis
plt = setup_matplotlib_zh(); plt.rcParams["font.size"] = 12
cols = [c for c in range(2019, 2025) if c in FY]
M = np.array([[res["transfer"][str(r)][str(c)]["F1"] for c in cols] for r in (2019, 2020, 2021)])
fig = plt.figure(figsize=(9.2, 4.2)); ax = fig.add_axes([0.11, 0.17, 0.74, 0.70]); cax = fig.add_axes([0.88, 0.17, 0.022, 0.70])
im = ax.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")
for i in range(3):
    for j, c in enumerate(cols):
        lo, hi = res["transfer"][str(2019 + i)][str(c)]["F1_ci95"]
        ax.text(j, i, f"{M[i, j]:.3f}\n[{lo:.2f}, {hi:.2f}]", ha="center", va="center", fontsize=9.5, color="white" if M[i, j] > 0.82 else "black",
                fontweight="bold" if c == 2019 + i else "normal")
        if c == 2019 + i: ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#d62728", lw=2))
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(3)); ax.set_yticklabels([f"AOMC {r}" for r in (2019, 2020, 2021)])
ax.set_xlabel("AlphaEarth 嵌入年份（评价时使用）"); ax.set_ylabel("标签年份（训练嵌入年=标签年）")
cb = fig.colorbar(im, cax=cax); cb.set_label("F1（空间块留出）")
fig.savefig(CY.parent / "fig07_crossyear_fixed_label_v4.png", dpi=300, bbox_inches="tight"); print("saved")
