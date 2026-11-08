import json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
R = Path("/workspace/apple_orchard_project/outputs/revision"); O = R / "v4_experiments/regen_v4_figs"
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})
J = json.load(open(O / "metrics_visual_indep_B3expanded.json")); m = J["metrics"]
old_m = json.load(open(R / "polygon_samples/metrics_visual_indep_polyB3_v2.json"))["point_level_apple_code1"]
new = pd.DataFrame(J["by_class"]).sort_values("interp_class_code"); old = pd.DataFrame(J["old_model_by_class"]).sort_values("interp_class_code")
names = {1: "苹果园", 2: "其他果园", 3: "农田", 4: "林地", 5: "灌丛草地", 6: "建设用地", 7: "水体", 8: "裸地/其他"}
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.75]))
cm = m["confusion_matrix"]; M = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
a.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.1)
for i in range(2):
    for j in range(2):
        a.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=17, fontweight="bold", color="white" if M[i, j] > 250 else "black")
a.set_xticks([0, 1], ["预测: 非苹果", "预测: 苹果"], fontsize=10.5); a.set_yticks([0, 1], ["真: 非苹果", "真: 苹果"], fontsize=10.5)
a.set_title(f"(a) 扩充训练B3混淆矩阵（n={m['n']}，苹果={m['n_pos']}）\nOA={m['OA']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['F1']:.3f}", fontsize=11)
x = np.arange(len(new)); w = 0.38
b.bar(x - w/2, old.rate, w, color="#bdbdbd", label=f"原B3（38面训练；OA={old_m['OA']:.3f}，F1={old_m['F1']:.3f}）")
b.bar(x + w/2, new.rate, w, color=["#d62728" if c == 1 else ("#ff9f1c" if c == 2 else "#1f77b4") for c in new.interp_class_code], label="扩充训练B3（277面训练）")
for xi, (r, k, n) in enumerate(zip(old.rate, old.pred_apple, old.n)): b.text(xi - w/2, r + 0.01, f"{k}/{n}", ha="center", fontsize=8.5, color="#555")
for xi, (r, k, n) in enumerate(zip(new.rate, new.pred_apple, new.n)): b.text(xi + w/2, r + 0.01, f"{k}/{n}", ha="center", fontsize=8.5)
b.set_xticks(x, [names[c] for c in new.interp_class_code], fontsize=10.5)
b.set_ylabel("预测为苹果的比例", fontsize=11); b.set_ylim(0, 0.62); b.legend(fontsize=9.5, loc="upper right", frameon=False)
b.set_title("(b) 按目视判读类别的“预测为苹果”比例（苹果类为召回，其余为误报率）", fontsize=11)
fig.tight_layout(); out = O / "figS2_visual_sensitivity_v4_expanded.png"; fig.savefig(out, dpi=300, facecolor="white"); print(out)
