# -*- coding: utf-8 -*-
"""v4.1 review item D: elevation-stratified recall of the expanded B3 (and old B3 for contrast) on the Gongliu challenge set,
plus B3 probability vs elevation (holdout points / polygons) with the training-set elevation distribution.
Elevation = SRTM 'elevation' feature sampled at each point (poly_s1s2dem_features_2021_expanded.csv); polygon = median of its points.
Threshold 0.5 fixed; holdout never used for training/tuning. Bins: <900, 900-1200, 1200-1400, >=1400 m."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
REV = Path(__file__).resolve().parent; ROOT = REV.parents[1]; sys.path.insert(0, str(REV)); sys.path.insert(0, str(ROOT / "python"))
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
sys.path.insert(0, str(ROOT / "outputs/revision/v4_writing/figures")); from v4style import plt, CM
V4 = ROOT / "outputs/revision/v4_experiments"; E = V4 / "retrain_expanded"; OUT = ROOT / "outputs/revision/v41_review/altitude"; OUT.mkdir(parents=True, exist_ok=True)
FO = ROOT / "outputs/revision/v41_review/figures"
pts = pd.read_csv(E / "sample_points_expanded.csv"); dem = pd.read_csv(E / "poly_s1s2dem_features_2021_expanded.csv")[["sample_id", "elevation"]]
BINS = [-1, 900, 1200, 1400, 9999]; LAB = ["<900", "900–1200", "1200–1400", "≥1400"]
def load(fn):
    pp = pd.read_csv(fn).merge(pts[["sample_id", "lon", "lat", "area_ha", "point_source"]], on="sample_id").merge(dem, on="sample_id")
    assert len(pp) == 521; return pp
new = load(E / "pred_points_B3_expanded_seed42.csv"); old = load(V4 / "pred_points_B3_seed42.csv")
res = {"note": __doc__}
tabs = []
for tag, h in (("B3_expanded", new), ("B3_old", old)):
    pt = v3m.poly_table(h).merge(h.groupby("poly_name").elevation.median().rename("elev_med"), left_on="poly_name", right_index=True)
    pt["bin"] = pd.cut(pt.elev_med, BINS, labels=LAB, right=False)
    h = h.copy(); h["bin"] = pd.cut(h.elevation, BINS, labels=LAB, right=False)
    for unit, d, col in (("poly_main101", pt.dropna(subset=["pred_major"]), "pred_major"), ("poly_all113_meanprob", pt, "pred_meanprob")):
        for b in LAB:
            a = d[(d.y == 1) & (d.bin == b)]; k, n = int((a[col] == 1).sum()), len(a)
            nn = d[(d.y == 0) & (d.bin == b)]; fp = int((nn[col] == 1).sum())
            tabs.append({"model": tag, "unit": unit, "elev_bin": b, "n_apple": n, "tp": k, "recall": k / n if n else np.nan,
                         "wilson_lo": v3m.wilson(k, n)[0] if n else np.nan, "wilson_hi": v3m.wilson(k, n)[1] if n else np.nan,
                         "n_nonapple": len(nn), "fp": fp})
    for b in LAB:
        a = h[(h.y == 1) & (h.bin == b)]; k, n = int(a.pred.sum()), len(a)
        tabs.append({"model": tag, "unit": "point521", "elev_bin": b, "n_apple": n, "tp": k, "recall": k / n if n else np.nan,
                     "wilson_lo": v3m.wilson(k, n)[0] if n else np.nan, "wilson_hi": v3m.wilson(k, n)[1] if n else np.nan,
                     "n_nonapple": int(((h.y == 0) & (h.bin == b)).sum()), "fp": int(h[(h.y == 0) & (h.bin == b)].pred.sum())})
    if tag == "B3_expanded": ptn = pt
T = pd.DataFrame(tabs); T.to_csv(OUT / "altitude_recall_B3.csv", index=False)
# correlation of prob with elevation among holdout apple polygons (Spearman)
from scipy.stats import spearmanr
ap = ptn[ptn.y == 1]; r, p = spearmanr(ap.elev_med, ap.mean_prob)
res["spearman_apple_polys_meanprob_vs_elev"] = {"rho": float(r), "p": float(p), "n": int(len(ap))}
tr = pts[pts.split == "train"].merge(dem, on="sample_id")
res["train_elev_by_class_points"] = {int(c): {"n": int(len(g)), "min": float(g.elevation.min()), "median": float(g.elevation.median()), "max": float(g.elevation.max()),
      "n_ge1200": int((g.elevation >= 1200).sum()), "n_ge1400": int((g.elevation >= 1400).sum())} for c, g in tr.groupby("class_code")}
res["holdout_apple_polys_elev"] = {"min": float(ap.elev_med.min()), "max": float(ap.elev_med.max())}
json.dump(res, open(OUT / "altitude_summary.json", "w"), ensure_ascii=False, indent=1)
print(T.to_string()); print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])
# ---- figure
C = {1: "#d55e00", 2: "#009e73", 0: "#7f7f7f"}; NM = {1: "苹果园", 2: "其他果园", 0: "非果园"}
fig, axs = plt.subplots(1, 3, figsize=(17.5 * CM, 6.6 * CM), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
ax = axs[0]
for c in (0, 2, 1):
    q = new[new.class_code == c]; ax.scatter(q.elevation, q.prob, s=5, alpha=0.6, color=C[c], label=f"{NM[c]}（{len(q)}点）", lw=0)
ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.set_xlabel("高程（m）"); ax.set_ylabel("新B3预测概率"); ax.set_title("(a) 挑战集样点", loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax = axs[1]
for c in (0, 2, 1):
    q = ptn[ptn.class_code == c]; ax.scatter(q.elev_med, q.mean_prob, s=12, alpha=0.8, color=C[c], label=NM[c], edgecolor="white", lw=0.3)
ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.set_xlabel("面中位高程（m）"); ax.set_ylabel("面平均概率"); ax.set_title("(b) 挑战集多边形（113面）", loc="left")
for x in (900, 1200, 1400):
    for a in axs[:2]: a.axvline(x, color="#bbbbbb", lw=0.5, ls=":")
ax = axs[2]
bins = np.arange(400, 3500, 100)
for c in (0, 2, 1):
    q = tr[tr.class_code == c]; ax.hist(q.elevation, bins=bins, orientation="horizontal", histtype="step", color=C[c], lw=1, label=f"{NM[c]}（{len(q)}点）")
ax.set_ylabel("高程（m）"); ax.set_xlabel("训练样点数"); ax.set_title("(c) 扩充训练集高程分布", loc="left"); ax.legend(frameon=False, fontsize=6.5)
ax.set_ylim(400, 3500)
fig.tight_layout(); fig.savefig(FO / "figS6_prob_vs_elevation_B3expanded_v41.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.03)
