# -*- coding: utf-8 -*-
"""v4.2 Fig.13: pre-registered 新源 second-county holdout (evaluated once). (a) map (GS(2024)0650 boundaries), (b) polygon F1 with bootstrap CIs."""
import sys, json
from pathlib import Path
import pandas as pd, geopandas as gpd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision/figures_v3_scripts")); import official_boundary as ob
X = ROOT / "outputs/revision/xinyuan_holdout"; OUT = ROOT / "outputs/revision/v42/figures"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
c = ob.county(); xy = c[c.name == "新源县"]; nb = c[c.intersects(xy.geometry.iloc[0].buffer(0.4))]
q = pd.read_csv(X / "QC_xinyuan_holdout.csv"); rm = gpd.read_file(X / "removed_training_polygons.geojson")
R = json.load(open(X / "RESULTS_xinyuan_holdout_ONCE.json"))
fig = plt.figure(figsize=(12, 4.3), dpi=200); ax = fig.add_axes([0.03, 0.1, 0.6, 0.82]); bx = fig.add_axes([0.70, 0.17, 0.28, 0.72])
nb.boundary.plot(ax=ax, color="0.65", lw=0.6); xy.boundary.plot(ax=ax, color="k", lw=1.3)
for n_, r in nb.iterrows():
    p = r.geometry.representative_point(); b = xy.total_bounds
    if b[0] - 0.3 < p.x < b[2] + 0.3 and b[1] - 0.25 < p.y < b[3] + 0.25: ax.text(p.x, p.y, r["name"], fontsize=7, color="0.4", ha="center")
rm.centroid.plot(ax=ax, marker="s", color="0.55", markersize=10, label=f"移除的训练面 (n={len(rm)})")
k = q[~q.excluded]; e = q[q.excluded]
for cc, col, lab in ((1, "red", "苹果园"), (2, "orange", "其他果园"), (0, "deepskyblue", "非果园")):
    s = k[k.class_code == cc]; ax.scatter(s.cen_lon, s.cen_lat, s=18, c=col, edgecolor="k", lw=0.4, zorder=3, label=f"留出面：{lab} (n={len(s)})")
ax.scatter(e.cen_lon, e.cen_lat, s=30, marker="x", c="k", zorder=4, label=f"QC剔除 (n={len(e)}，跨县界)")
b = xy.total_bounds; ax.set_xlim(b[0] - 0.12, b[2] + 0.12); ax.set_ylim(b[1] - 0.1, b[3] + 0.1); ax.set_aspect(1 / 0.73)
ax.legend(fontsize=7, loc="lower right"); ax.set_title("(a) 新源县预注册一次性留出集（QC后49个面）", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
names = {"B3": "B3 (AE)", "B0": "B0 (S2)", "B6": "B6 (AE+S1+S2+DEM)", "HIER": "两级模型 (AE)"}
for i, m in enumerate(["B3", "B0", "B6", "HIER"]):
    v = R["models"][m]; lo, hi = v["bootstrap_95ci"]["F1"]
    bx.errorbar(v["F1"], i, xerr=[[v["F1"] - lo], [hi - v["F1"]]], fmt="o", color="C3" if m == "B3" else "C0", capsize=3)
    bx.text(min(hi + 0.02, 0.9), i - 0.18, f"{v['F1']:.3f}", va="center", fontsize=7)
bx.set_yticks(range(4)); bx.set_yticklabels([names[m] for m in ["B3", "B0", "B6", "HIER"]], fontsize=7.5); bx.invert_yaxis(); bx.set_xlim(0, 1)
bx.set_xlabel("面级F1（面bootstrap 95% CI，种子42，阈值0.5）", fontsize=7.5); bx.tick_params(labelsize=7)
d = R["paired_dF1"]["PRIMARY_B3_minus_B0"]
bx.set_title(f"(b) 苹果园11/非苹果38；样本不足\n主比较ΔF1(B3−B0)={d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", fontsize=8, loc="left")
plt.savefig(OUT / "fig13_xinyuan_holdout_v42.png"); print("ok")
