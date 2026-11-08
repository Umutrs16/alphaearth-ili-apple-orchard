# -*- coding: utf-8 -*-
"""v4 task 3: supplementary heatmaps (AUC of transfer; F1 when retrained on the feature year) from metrics JSON."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "python/revision"))
from common import setup_matplotlib_zh
O = ROOT / "outputs/revision/v4_experiments"; r = json.load(open(O / "metrics_crossyear_fixed_label_v4.json"))
plt = setup_matplotlib_zh(); plt.rcParams["font.size"] = 11.5
cols = list(range(2019, 2025)); rows = (2019, 2020, 2021)
fig, axs = plt.subplots(1, 2, figsize=(13, 3.9)); fig.subplots_adjust(left=0.08, right=0.93, wspace=0.28, bottom=0.18, top=0.86)
for ax, (key, met, ttl, vmin) in zip(axs, [("transfer", "AUC", "(a) 迁移AUC（训练嵌入年=标签年）", 0.9), ("retrain", "F1", "(b) 按评价嵌入年重训的F1（同一空间块划分）", 0.5)]):
    M = np.array([[r[key][str(a)][str(c)][met] for c in cols] for a in rows])
    im = ax.imshow(M, cmap="YlGnBu", vmin=vmin, vmax=1.0, aspect="auto")
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=10, color="white" if M[i, j] > vmin + 0.64 * (1 - vmin) else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(cols); ax.set_yticks(range(3)); ax.set_yticklabels([f"AOMC {a}" for a in rows])
    ax.set_xlabel("AlphaEarth 嵌入年份"); ax.set_title(ttl, loc="left", fontsize=11.5)
    cax = ax.inset_axes([1.02, 0, 0.03, 1]); fig.colorbar(im, cax=cax)
fig.savefig(O / "figS_crossyear_AUC_retrain_v4.png", dpi=300, bbox_inches="tight"); print("saved")
