# -*- coding: utf-8 -*-
"""v4.3 merged Fig.4: (a)(b) weak-label B0-B6 ablation (unchanged v4 figure) + (c) strict label-efficiency curves:
>=10% fractions = 20 repeats (v43 rerun, 42c), 1%/5% = the v4.1 5-repeat run (reference only). Mean ± 1 SD (np.std, ddof=0, as in v4.1)."""
import sys; from pathlib import Path
import pandas as pd, numpy as np
from PIL import Image
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "outputs/revision/v4_writing/figures"))
from v4style import plt, CM
R = ROOT / "outputs/revision"; FO = R / "v43/figures"
A = pd.read_csv(R / "v41_review/label_efficiency_strict/label_efficiency_strict.csv"); B = pd.read_csv(R / "v43/label_efficiency_strict_20rep/label_efficiency_strict.csv")
SA = pd.read_csv(R / "v41_review/label_efficiency_strict/label_efficiency_strict_draw_summary.csv"); SB = pd.read_csv(R / "v43/label_efficiency_strict_20rep/label_efficiency_strict_draw_summary.csv")
L = pd.concat([A[A.frac < 0.1], B]); S = pd.concat([SA[SA.frac < 0.1], SB]).sort_values("frac")
L.to_csv(R / "v43/label_efficiency_strict_20rep/label_efficiency_merged_v43.csv", index=False)
cols = {"S2": "#56b4e9", "S1+S2": "#0072b2", "AE": "#d55e00", "AE+S1+S2": "#e69f00"}; mk = {"S2": "s", "S1+S2": "D", "AE": "o", "AE+S1+S2": "^"}
fig, ax = plt.subplots(figsize=(12 * CM, 7.6 * CM))
xs = {0.01: 0, 0.05: 1, 0.10: 2, 0.25: 3, 0.50: 4, 1.00: 5}
ax.axvspan(-0.5, 1.5, color="#eeeeee", zorder=0); ax.text(0.5, 1.015, "参考（1%、5%：5次重复，块数少）", ha="center", fontsize=7, color="#555")
ax.text(3.5, 1.015, "主要证据（≥10%：20次重复）", ha="center", fontsize=7, color="#333")
off = {"S2": -0.15, "S1+S2": -0.05, "AE": 0.05, "AE+S1+S2": 0.15}
for k in ["S2", "S1+S2", "AE", "AE+S1+S2"]:
    g = L[L.feature_set == k].sort_values("frac"); x = np.array([xs[round(f, 2)] for f in g.frac]) + off[k]
    ax.errorbar(x, g.F1_mean, yerr=g.F1_sd, marker=mk[k], ms=3.5, lw=1, capsize=2, color=cols[k], label=k)
ax.set_xticks(range(6)); ax.set_xticklabels([f"{int(round(f*100))}%\n({int(m)})" for f, m in zip(S.frac, S.n_train_median)])
ax.set_xlabel("训练空间块比例（每次训练点数中位数）"); ax.set_ylabel("F1（固定空间块留出测试集）"); ax.set_title("(c) 标签效率（严格块内抽样）", loc="left", fontweight="bold", pad=14)
ax.set_ylim(0.2, 1.04); ax.set_xlim(-0.5, 5.5); ax.grid(axis="y", alpha=0.3, lw=0.5); ax.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1, 0.02))
p5 = FO / "fig05_label_efficiency_strict_20rep_v43.png"; fig.savefig(p5, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.03)
a = Image.open(R / "v4_writing/figures/fig04_ablation_forest_v4.png").convert("RGB"); b = Image.open(p5).convert("RGB")
W = a.width; s = 0.62 * W / b.width; b = b.resize((int(b.width * s), int(b.height * s)), Image.LANCZOS)
c = Image.new("RGB", (W, a.height + b.height + 20), "white"); c.paste(a, (0, 0)); c.paste(b, ((W - b.width) // 2, a.height + 20))
c.save(FO / "fig04_ablation_labeleff_merged_v43.png", dpi=(300, 300)); print("ok", c.size)
