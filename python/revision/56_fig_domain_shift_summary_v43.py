# -*- coding: utf-8 -*-
"""v4.3 summary figure built ONLY from existing result files: F1 (95% CI) of B3 (AE) and B0 (S2) (+B6) in three evaluation domains, and the
paired dF1 (B3 - B0). (1) AOMC weak-label protocol (block bootstrap; fig4_delta_f1_paired_bootstrap.json); (2) Gongliu challenge set, v4.3 main
labels (metrics_relabel_v43.json; hollow = original labels); (3) Xinyuan pre-registered one-shot holdout (RESULTS_xinyuan_holdout_ONCE.json)."""
import sys, json; from pathlib import Path
import numpy as np
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "outputs/revision/v4_writing/figures")); from v4style import plt, CM
R = ROOT / "outputs/revision"
A = json.load(open(R / "v4_writing/figures/fig4_delta_f1_paired_bootstrap.json"))["models"]
G = json.load(open(R / "v43/relabel/metrics_relabel_v43.json"))["results"]; X = json.load(open(R / "xinyuan_holdout/RESULTS_xinyuan_holdout_ONCE.json"))
def g(lab, b): p = G[lab]["expanded"][b]["seed42"]["poly"]; return p["F1"], p["bootstrap_95ci"]["F1"]
dom = ["AOMC弱标签协议\n（空间块留出，882点）", "巩留跨县域挑战集\n（101面，v4.3标签）", "新源预注册一次性留出\n（49面，苹果11；样本不足）"]
val = {"B3": [(A["B3"]["F1_point"], [A["B3"]["F1_ci_lo_stored"], A["B3"]["F1_ci_hi_stored"]]), g("v43", "B3"), (X["models"]["B3"]["F1"], X["models"]["B3"]["bootstrap_95ci"]["F1"])],
       "B0": [(A["B0"]["F1_point"], [A["B0"]["F1_ci_lo_stored"], A["B0"]["F1_ci_hi_stored"]]), g("v43", "B0"), (X["models"]["B0"]["F1"], X["models"]["B0"]["bootstrap_95ci"]["F1"])],
       "B6": [(A["B6"]["F1_point"], [A["B6"]["F1_ci_lo_stored"], A["B6"]["F1_ci_hi_stored"]]), g("v43", "B6"), (X["models"]["B6"]["F1"], X["models"]["B6"]["bootstrap_95ci"]["F1"])]}
orig = {b: g("orig", b)[0] for b in ("B3", "B0", "B6")}
dd = [(-A["B0"]["dF1_point"], [-A["B0"]["dF1_ci_hi"], -A["B0"]["dF1_ci_lo"]])]
q = G["v43"]["expanded"]["B0"]["paired_dF1_vs_sameTrain_B3"]; dd.append((-q["delta"], [-q["ci95"][1], -q["ci95"][0]]))
q = X["paired_dF1"]["PRIMARY_B3_minus_B0"]; dd.append((q["delta"], q["ci95"]))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(17 * CM, 7.2 * CM), gridspec_kw={"width_ratios": [1.6, 1]})
col = {"B3": "#d55e00", "B0": "#0072b2", "B6": "#999999"}; off = {"B3": -0.18, "B0": 0.0, "B6": 0.18}; nm = {"B3": "B3（AE）", "B0": "B0（S2）", "B6": "B6（AE+S1+S2+DEM）"}
for b in ("B3", "B0", "B6"):
    for i, (v, c) in enumerate(val[b]):
        ax.errorbar(i + off[b], v, yerr=[[v - c[0]], [c[1] - v]], fmt="o", color=col[b], ms=4, capsize=2, lw=1, label=nm[b] if i == 0 else None)
        ax.text(i + off[b] + 0.04, v, f"{v:.3f}", fontsize=5.8, va="center", color=col[b])
    ax.plot(1 + off[b], orig[b], marker="o", mfc="none", mec=col[b], ms=4, lw=0)
ax.plot([], [], marker="o", mfc="none", mec="k", lw=0, ms=4, label="巩留：原标签（v4.2）")
ax.set_xticks(range(3)); ax.set_xticklabels(dom, fontsize=6.5); ax.set_ylabel("F1（95% CI）"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3, lw=0.5)
ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=2); ax.set_title("(a) 三个评价域中的F1", loc="left", fontsize=8.5)
for i, (v, c) in enumerate(dd):
    bx.errorbar(v, i, xerr=[[v - c[0]], [c[1] - v]], fmt="s", color="k", ms=4, capsize=2, lw=1); bx.text(v, i - 0.28, f"{v:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]", ha="center", fontsize=6)
bx.axvline(0, color="k", lw=0.7); bx.set_yticks(range(3)); bx.set_yticklabels([d.split("\n")[0] for d in dom], fontsize=6.5); bx.invert_yaxis(); bx.set_ylim(2.6, -0.6)
bx.set_xlabel("ΔF1 = F1(B3) − F1(B0)（配对95% CI）", fontsize=7); bx.set_title("(b) AE相对S2的差值", loc="left", fontsize=8.5); bx.grid(axis="x", alpha=0.3, lw=0.5)
fig.tight_layout(); fig.savefig(R / "v43/figures/fig_domain_shift_summary_v43.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.03)
json.dump({"F1": {b: [(float(v), [float(x) for x in c]) for v, c in val[b]] for b in val}, "dF1_B3_minus_B0": [(float(v), [float(x) for x in c]) for v, c in dd], "orig_gongliu": orig},
          open(R / "v43/figures/fig_domain_shift_summary_v43.json", "w"), indent=1); print("ok", dd)
