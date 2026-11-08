# -*- coding: utf-8 -*-
"""Fig.5 (v4.1): strict label-efficiency curves (no positive top-up). >=10% = main evidence; 1%/5% shaded as reference."""
import sys; from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "outputs/revision/v4_writing/figures"))
from v4style import plt, CM
D = ROOT / "outputs/revision/v41_review/label_efficiency_strict"; FO = ROOT / "outputs/revision/v41_review/figures"; FO.mkdir(parents=True, exist_ok=True)
R = pd.read_csv(D / "label_efficiency_strict.csv"); S = pd.read_csv(D / "label_efficiency_strict_draw_summary.csv")
cols = {"S2": "#56b4e9", "S1+S2": "#0072b2", "AE": "#d55e00", "AE+S1+S2": "#e69f00"}; mk = {"S2": "s", "S1+S2": "D", "AE": "o", "AE+S1+S2": "^"}
fig, ax = plt.subplots(figsize=(12 * CM, 7.6 * CM))
xs = {0.01: 0, 0.05: 1, 0.10: 2, 0.25: 3, 0.50: 4, 1.00: 5}
ax.axvspan(-0.5, 1.5, color="#eeeeee", zorder=0); ax.text(0.5, 1.015, "参考（1%、5%：块数少、波动大）", ha="center", fontsize=7, color="#555")
ax.text(3.5, 1.015, "主要证据（≥10%）", ha="center", fontsize=7, color="#333")
off = {"S2": -0.15, "S1+S2": -0.05, "AE": 0.05, "AE+S1+S2": 0.15}
for k, g in R.groupby("feature_set", sort=False):
    x = np.array([xs[f] for f in g.frac]) + off[k]
    ax.errorbar(x, g.F1_mean, yerr=g.F1_sd, marker=mk[k], ms=3.5, lw=1, capsize=2, color=cols[k], label=k)
lab = [f"{int(f*100)}%\n({int(r.n_train_median)})" for f, r in zip(S.frac, S.itertuples())]
ax.set_xticks(range(6)); ax.set_xticklabels(lab)
ax.set_xlabel("训练空间块比例（每次训练点数中位数）"); ax.set_ylabel("F1（固定空间块留出测试集）")
ax.set_ylim(0.2, 1.04); ax.set_xlim(-0.5, 5.5); ax.grid(axis="y", alpha=0.3, lw=0.5)
ax.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1, 0.02))
fig.savefig(FO / "fig05_label_efficiency_strict_v41.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.03); print("ok")
