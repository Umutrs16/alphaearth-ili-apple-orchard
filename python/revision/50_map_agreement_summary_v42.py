# -*- coding: utf-8 -*-
"""v4.2: summary of the user's visual check (20 random AOMC-2021 apple patches, 20 random new-B3 p>=0.8 patches, Gongliu, seed 20260926).
visual_class as entered by the user (apple / other_orchard / non_orchard / unsure); Wilson 95% CIs; area-weighted shares (patch_info);
context only: ESA WorldCover 2021 class at the patch point (GEE). Shelterbelt subtype is NOT recorded per patch in the returned form."""
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd, ee
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/workspace/apple_orchard_project"); OUT = ROOT / "outputs/revision/v42/map_agreement"
X = Path("/workspace/uploads/user_return_0928/map_agreement_check_template.xlsx")
v = pd.read_excel(X, sheet_name="visual_check"); info = pd.read_excel(X, sheet_name="patch_info")
d = v.merge(info[["id", "patch_area_ha", "n_pixels", "mean_prob_newB3"]], on="id"); d["grp"] = np.where(d.id.str.startswith("AOMC"), "AOMC", "MOD")
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return max(0, c - h), min(1, c + h)
ee.Initialize(project="hallowed-chain-458005-g9")
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"id": r.id}) for r in d.itertuples()])
w = {f["properties"]["id"]: f["properties"].get("Map") for f in wc.sampleRegions(collection=fc, scale=10).getInfo()["features"]}
WCN = {10: "tree", 20: "shrub", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare", 80: "water", 90: "wetland"}
d["worldcover2021"] = d.id.map(lambda i: WCN.get(w.get(i), str(w.get(i))))
rows = []
for g, s in d.groupby("grp"):
    n = len(s)
    for c in ("apple", "other_orchard", "non_orchard", "unsure"):
        k = int((s.visual_class == c).sum()); lo, hi = wilson(k, n)
        rows.append({"source": g, "visual_class": c, "k": k, "n": n, "pct": 100 * k / n, "wilson_lo_pct": 100 * lo, "wilson_hi_pct": 100 * hi,
                     "area_ha": float(s[s.visual_class == c].patch_area_ha.sum()), "area_pct": float(100 * s[s.visual_class == c].patch_area_ha.sum() / s.patch_area_ha.sum())})
T = pd.DataFrame(rows); T.to_csv(OUT / "table_visual_agreement.csv", index=False, encoding="utf-8-sig")
ct = pd.crosstab([d.grp, d.visual_class], d.worldcover2021); ct.to_csv(OUT / "crosstab_visual_vs_worldcover.csv", encoding="utf-8-sig")
d.to_csv(OUT / "visual_check_merged.csv", index=False, encoding="utf-8-sig")
summ = {"n_per_source": 20, "counts": T.to_dict("records"), "notes_nonempty": d[d.notes.notna()][["id", "visual_class", "notes"]].to_dict("records"),
        "img_year_counts": d.img_date.astype(str).str[:4].value_counts().to_dict(), "worldcover_by_group_class": {f"{a}|{b}": r.to_dict() for (a, b), r in ct.iterrows()},
        "shelterbelt_subtype_recorded": False,
        "note": "Shelterbelt/roadside tree-belt share cannot be quantified: the returned form has visual_class only (non_orchard) and no per-patch subtype notes for AOMC patches."}
json.dump(summ, open(OUT / "summary_visual_agreement.json", "w"), ensure_ascii=False, indent=1, default=str)
print(T.round(1).to_string()); print(ct.to_string()); print(summ["notes_nonempty"], summ["img_year_counts"])
# figure
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(7, 3.2), dpi=200); cols = {"apple": "#d62728", "other_orchard": "#ff7f0e", "non_orchard": "#7f7f7f", "unsure": "#c7c7c7"}
lab = {"AOMC": "AOMC 2021苹果斑块 (n=20)", "MOD": "扩充B3高概率斑块 p≥0.8 (n=20)"}
for i, g in enumerate(["AOMC", "MOD"]):
    left = 0
    for c in cols:
        r = T[(T.source == g) & (T.visual_class == c)].iloc[0]
        if r.k: ax.barh(i, r.pct, left=left, color=cols[c], edgecolor="k", lw=0.4); ax.text(left + r.pct / 2, i, f"{r.k}", ha="center", va="center", fontsize=8, color="w" if c in ("apple", "non_orchard") else "k")
        left += r.pct
ax.set_yticks([0, 1]); ax.set_yticklabels([lab["AOMC"], lab["MOD"]], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("随机抽取斑块的比例 (%)", fontsize=8)
from matplotlib.patches import Patch
ax.legend([Patch(color=c) for c in cols.values()], ["苹果园", "其他果园", "非果园", "无法判断"], ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
ax.set_title("巩留县随机斑块目视判读（Google Earth历史影像）", fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT / "figS8_visual_agreement_v42.png"); print("fig ok")
