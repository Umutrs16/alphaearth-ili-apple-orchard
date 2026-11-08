# -*- coding: utf-8 -*-
"""v4 task 1: plot forest figure from metrics_indep_feature_compare_v4.json (no recomputation)."""
import json, sys
from pathlib import Path
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "python/revision"))
from common import setup_matplotlib_zh
OUT = ROOT / "outputs/revision/v4_experiments"
res = json.load(open(OUT / "metrics_indep_feature_compare_v4.json", encoding="utf-8"))
NAMES = {"B0": "B0 S2", "B2": "B2 S1+S2+DEM", "B3": "B3 AE", "B6": "B6 AE+S1+S2+DEM"}; SEEDS = res["seeds"]
# figure: forest plot F1 and delta F1
plt = setup_matplotlib_zh(); plt.rcParams["font.size"] = 12
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
order = ["B0", "B2", "B6", "B3"]; col = {"B0": "#7f7f7f", "B2": "#8c564b", "B6": "#1f77b4", "B3": "#d62728"}
ax = axs[0]
for i, bid in enumerate(order):
    p = res["models"][bid]["seed42"]["poly"]; ci = p["bootstrap_95ci"]["F1"]; ss = res["models"][bid]["seed_summary"]["poly_F1"]
    ax.errorbar(p["F1"], i, xerr=[[p["F1"] - ci[0]], [ci[1] - p["F1"]]], fmt="o", color=col[bid], ms=8, capsize=4, lw=1.8)
    ax.plot(res["models"][bid]["per_seed_poly_F1"], [i + 0.18] * len(SEEDS), "|", color=col[bid], ms=9, alpha=0.7)
    ax.text(1.17, i, f"{p['F1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", va="center", ha="right", fontsize=10)
base = 2 * 36 / (36 + 101); ax.axvline(base, color="k", ls=":", lw=1); ax.text(base, -0.75, " 全判苹果基线", fontsize=9, va="bottom")
ax.set_yticks(range(4)); ax.set_yticklabels([NAMES[b] for b in order]); ax.set_xlim(0.3, 1.18); ax.set_ylim(-0.8, 3.6)
ax.set_xlabel("面级 F1（面 bootstrap 95% CI；竖线为10个随机种子）"); ax.set_title("(a) 巩留园块空间留出：面级 F1（n=101）", loc="left", fontsize=12); ax.grid(axis="x", alpha=0.3)
ax = axs[1]; ob = ["B0", "B2", "B6"]
for i, bid in enumerate(ob):
    d = res["models"][bid]["paired_delta_F1_vs_B3_seed42"]
    ax.errorbar(d["delta"], i, xerr=[[d["delta"] - d["ci95"][0]], [d["ci95"][1] - d["delta"]]], fmt="s", color=col[bid], ms=8, capsize=4, lw=1.8)
    ax.text(-0.285, i + 0.3, f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", ha="left", fontsize=10)
ax.axvline(0, color="k", lw=1); ax.set_yticks(range(3)); ax.set_yticklabels([NAMES[b] + " − B3" for b in ob]); ax.set_ylim(-0.6, 2.7); ax.set_xlim(-0.29, 0.06)
ax.set_xlabel("ΔF1（配对面 bootstrap 95% CI，seed 42）"); ax.set_title("(b) 相对 AE-only（B3）的面级 ΔF1", loc="left", fontsize=12); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_indep_feature_compare_v4.png", dpi=300); print("fig saved")
