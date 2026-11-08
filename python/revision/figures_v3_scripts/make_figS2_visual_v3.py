import json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
R = Path("/workspace/apple_orchard_project/outputs/revision")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})
m = json.load(open(R / "polygon_samples/metrics_visual_indep_polyB3_v2.json"))["point_level_apple_code1"]
cm = m["confusion_matrix"]; d = pd.read_csv(R / "polygon_samples/v3/visual_fp_by_class_v2model.csv")
names = {1: "苹果园", 2: "其他果园", 3: "农田", 4: "林地", 5: "灌丛草地", 6: "建设用地", 7: "水体", 8: "裸地/其他"}
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
M = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
a.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.1)
for i in range(2):
    for j in range(2):
        a.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=15, fontweight="bold", color="white" if M[i, j] > 250 else "black")
a.set_xticks([0, 1], ["预测: 非苹果", "预测: 苹果"]); a.set_yticks([0, 1], ["真: 非苹果", "真: 苹果"])
a.set_title(f"(a) 目视敏感性混淆矩阵（B3_poly_v2；n={m['n']}，n_pos={m['n_pos']}）\nOA={m['OA']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['F1']:.3f}", fontsize=9.5)
d = d.sort_values("interp_class_code")
x = np.arange(len(d)); cols = ["#d62728" if c == 1 else ("#ff9f1c" if c == 2 else "#1f77b4") for c in d.interp_class_code]
b.bar(x, d.rate, color=cols)
for xi, (r, k, n) in enumerate(zip(d.rate, d.pred_apple, d.n)):
    b.text(xi, r + 0.01, f"{k}/{n}", ha="center", fontsize=8)
b.set_xticks(x, [names[c] for c in d.interp_class_code], fontsize=8.5)
b.set_ylabel("预测为苹果的比例"); b.set_ylim(0, 0.62)
b.set_title("(b) 按目视判读类别的“预测为苹果”比例（非苹果类即误报率）", fontsize=9.5)
b.axhspan(d[d.interp_class_code >= 3].rate.min(), d[d.interp_class_code >= 3].rate.max(), color="grey", alpha=0.12)
fig.tight_layout(); out = R / "figures_v3/figS2_visual_sensitivity_v3.png"; fig.savefig(out, dpi=300, facecolor="white"); print(out)
print(d.assign(n2=d.interp_class_code.map(names)))
