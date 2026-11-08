# -*- coding: utf-8 -*-
"""v4 retrain: comparison figure old (38 training polygons) vs expanded (38 + 239 YL-TRN polygons) on the unchanged
Gongliu 101-polygon holdout. All numbers read from metrics_indep_feature_compare_v4.json and metrics_expanded_retrain.json."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "python/revision"))
from common import setup_matplotlib_zh
V4 = ROOT / "outputs/revision/v4_experiments"; R = V4 / "retrain_expanded"
old = json.load(open(V4 / "metrics_indep_feature_compare_v4.json"))["models"]
new = json.load(open(R / "metrics_expanded_retrain.json"))
NM = {"B0": "B0 S2", "B2": "B2 S1+S2+DEM", "B3": "B3 AE", "B6": "B6 AE+S1+S2+DEM", "HIER": "两级模型（AE）"}
plt = setup_matplotlib_zh(); plt.rcParams["font.size"] = 12
fig = plt.figure(figsize=(13.5, 11.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], width_ratios=[1.35, 1.0], hspace=0.42, wspace=0.28)
axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
ax = axs[0]; rows = []
for b in ["B0", "B2", "B6", "B3"]: rows.append(("old", b))
for b in ["B0", "B2", "B6", "B3", "HIER"]: rows.append(("new", b))
yl = []
for i, (tag, b) in enumerate(rows):
    m = (old if tag == "old" else new["models"])[b]; p = m["seed42"]["poly"]; ci = p["bootstrap_95ci"]["F1"]
    col = "#9e9e9e" if tag == "old" else ("#d62728" if b in ("B3", "HIER") else "#1f77b4")
    y = i + (0.6 if tag == "new" else 0)
    ax.errorbar(p["F1"], y, xerr=[[p["F1"] - ci[0]], [ci[1] - p["F1"]]], fmt="o" if tag == "old" else "s", color=col, ms=8, capsize=4, lw=1.8)
    ax.plot(m["per_seed_poly_F1"], [y + 0.22] * len(m["per_seed_poly_F1"]), "|", color=col, ms=9, alpha=0.7)
    ax.text(1.02, y, f"{p['F1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", va="center", ha="left", fontsize=9.5, transform=ax.get_yaxis_transform())
    yl.append((y, ("原训练 " if tag == "old" else "扩充 ") + NM[b]))
ax.axhline(3.8, color="k", lw=0.6, ls="--")
ax.axvline(2 * 36 / (36 + 101), color="k", ls=":", lw=1); ax.text(2 * 36 / 137, 9.25, " 全判苹果基线", fontsize=9)
ax.set_yticks([a for a, _ in yl]); ax.set_yticklabels([b for _, b in yl]); ax.set_xlim(0.0, 0.88); ax.set_ylim(-1.0, 9.4); ax.invert_yaxis()
ax.set_xlabel("面级 F1（面 bootstrap 95% CI，seed 42；竖线=10 个种子）"); ax.grid(axis="x", alpha=0.3)
ax.set_title("(a) 巩留留出面级 F1（n=101）：原 38 面 vs 扩充 277 面训练", loc="left", fontsize=12)
ax = axs[1]; bs = ["B0", "B2", "B6", "B3", "HIER"]
oldF = np.array(old["B3"]["per_seed_poly_F1"])
for i, b in enumerate(bs):
    d = new["models"][b]["paired_dF1_vs_B3_old_seed42"]; col = "#d62728" if b in ("B3", "HIER") else "#1f77b4"
    ax.errorbar(d["delta"], i, xerr=[[d["delta"] - d["ci95"][0]], [d["ci95"][1] - d["delta"]]], fmt="s", color=col, ms=8, capsize=4, lw=1.8)
    ds = np.array(new["models"][b]["per_seed_poly_F1"]) - oldF
    ax.plot(ds.mean(), i + 0.3, "v", color="k", ms=7)
    ax.text(-0.74, i - 0.34, f"seed42 {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]；10种子均值 {ds.mean():+.3f}±{ds.std(ddof=1):.3f}", ha="left", fontsize=8.8)
ax.axvline(0, color="k", lw=1); ax.set_yticks(range(len(bs))); ax.set_yticklabels(["扩充 " + NM[b] for b in bs]); ax.set_ylim(-0.8, len(bs) - 0.4); ax.invert_yaxis()
ax.set_xlim(-0.75, 0.3); ax.set_xlabel("ΔF1 相对原 B3_poly_2021_v2（配对面 bootstrap 95% CI；▼=10 种子均值差）"); ax.grid(axis="x", alpha=0.3)
ax.set_title("(b) 相对原 B3 的面级 ΔF1", loc="left", fontsize=12)
ax = axs[2]; cats = [("apple_recall", "苹果园\n召回\n(n=36)"), ("other_orchard_rejected", "其他果园\n拒判\n(n=10)"), ("non_orchard_rejected", "非果园\n拒判\n(n=55)")]
sets = [("原 B3", old["B3"], "#9e9e9e"), ("扩充 B3", new["models"]["B3"], "#d62728"), ("扩充 两级模型", new["models"]["HIER"], "#8c1c13")]
w = 0.26
for j, (lab, m, col) in enumerate(sets):
    pc = m["seed42"]["poly"]["per_class"]
    for i, (k, _) in enumerate(cats):
        r = pc[k]; rate = r["rate"]; lo, hi = r["wilson"]
        ax.bar(i + (j - 1) * w, rate, w, color=col, label=lab if i == 0 else None)
        ax.errorbar(i + (j - 1) * w, rate, yerr=[[rate - lo], [hi - rate]], color="k", capsize=3, lw=1)
        ax.text(i + (j - 1) * w, 0.02, f"{r['k']}/{r['n']}", ha="center", va="bottom", fontsize=8.5, color="white", rotation=90)
ax.set_xticks(range(3)); ax.set_xticklabels([c for _, c in cats]); ax.set_ylim(0, 1.05); ax.set_ylabel("比例（Wilson 95% CI）"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16), frameon=False); ax.set_title("(c) 分类别表现（面级多数投票，seed 42）", loc="left", fontsize=12)
fig.savefig(R / "fig12_expanded_retrain_compare_v4.png", dpi=300, bbox_inches="tight"); print("saved")
