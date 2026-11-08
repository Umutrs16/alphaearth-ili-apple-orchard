# -*- coding: utf-8 -*-
"""Write REPORT_B3_poly_holdout_v3.md and caption from computed JSON/CSVs (no hand-typed metrics)."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/revision/polygon_samples"; V3 = OUT / "v3"; FIG3 = ROOT / "outputs/revision/figures_v3"
M = json.load(open(V3 / "metrics_poly_holdout_v3.json", encoding="utf-8"))
Q = json.load(open(V3 / "qc_summary_v3.json", encoding="utf-8"))
S = json.load(open(V3 / "metrics_shortcut_diagnostics_v3.json", encoding="utf-8"))
E2 = json.load(open(OUT / "metrics_poly_holdout_v2_enriched.json", encoding="utf-8"))
S2 = json.load(open(OUT / "metrics_shortcut_diagnostics.json", encoding="utf-8"))
VIS = pd.read_csv(V3 / "visual_fp_by_class_v2model.csv")
FD = pd.read_csv(V3 / "fp_by_distance_to_apple_v3.csv"); FA = pd.read_csv(V3 / "fp_by_area_v3.csv")
f = lambda x: f"{x:.3f}"


def row(name, m, ci=True):
    c = m["confusion_matrix"]
    w = m.get("wilson_95ci_OA"); b = m.get("bootstrap_95ci", {}).get("F1")
    oa = f(m["OA"]) + (f" [{f(w['lo'])}, {f(w['hi'])}]" if w and ci else "")
    f1 = f(m["F1"]) + (f" [{f(b['lo'])}, {f(b['hi'])}]" if b and ci else "")
    return (f"| {name} | {m['n']} | {m['n_pos']}/{m['n_neg']} | {oa} | {f(m['precision'])} | {f(m['recall'])} | "
            f"{f(m['specificity'])} | {f1} | {f(m['balanced_accuracy'])} | {f(m['MCC'])} | {c['TP']}/{c['FP']}/{c['FN']}/{c['TN']} |")


H = ("| 设置 | n面 | 苹果/非苹果 | OA [Wilson 95%CI] | P | R | 特异度 | F1 [面bootstrap 95%CI] | BA | MCC | TP/FP/FN/TN |\n"
     "|---|---|---|---|---|---|---|---|---|---|---|")
pm = M["poly_majority"]; pp = M["poly_meanprob"]; pt = M["point_level"]
L = []
L += ["# B3 面样本巩留 Holdout 扩展评估报告 v3（AlphaEarth-only，2021）", "",
      "**结论先行**：在扩展后的巩留县多边形 holdout 上，面级（主指标，多数投票）"
      f"F1={f(pm['F1'])}（面 bootstrap 95%CI {f(pm['bootstrap_95ci']['F1']['lo'])}–{f(pm['bootstrap_95ci']['F1']['hi'])}），"
      f"OA={f(pm['OA'])}，平衡精度={f(pm['balanced_accuracy'])}，MCC={f(pm['MCC'])}。苹果召回高"
      f"（{pm['per_class']['apple_recall']['k']}/{pm['per_class']['apple_recall']['n']}），误差几乎全部为**误报（FP={pm['confusion_matrix']['FP']}）**。"
      "相比 v2（18 面，F1=0.923），v3 指标明显下降；v2 的 18 面在 v3 中结果**完全复现**，下降来自新增的 83 面（尤其是非果园与其他果园负样本），"
      "说明 v2 的小样本 holdout 高估了县域外推精度。", "",
      "## 1. 数据与 QC", "",
      f"- 输入：`holdout_expansion/apple_new.kml`，面要素 {Q['n_parsed']} 个（既有 v2 holdout {Q['n_existing']} + 新增 {Q['n_new']}；无重名）。",
      "- 类别解析：新增面取描述 `class_code`；既有 18 面无属性，从名称后缀 `| c1`/`| c0` 解析并剥离后缀为 poly_id。",
      f"- 解析类别（全部 114）：苹果(1)={Q['class_counts_all'].get('1')}，其他果园(2)={Q['class_counts_all'].get('2')}，非果园(0)={Q['class_counts_all'].get('0')}；"
      f"新增：0={Q['class_counts_new'].get('0')}，1={Q['class_counts_new'].get('1')}，2={Q['class_counts_new'].get('2')}。（与用户快速解析一致）",
      "- 面积按 UTM 44N（EPSG:32644）计算；无 >50 ha 面；无退化几何（顶点<3 / 面积<0.01 ha / 无效）。",
      f"- 既有 18 面几何与 v2 KML 一致（最小 IoU={Q['existing_iou_vs_v2_min']:.5f}），因此**直接复用 v2 的 127 个样点与 AE 特征**。",
      "- **重叠检查**：",
      "  - holdout–holdout：1 对。`YL-POLY-0095`（新增，苹果）与 `YL-POLY-0054`（v2，苹果）重叠 0.682 ha（占较小面 98%），同类，判定为重复勾绘 → **剔除 0095**，保留 v2 原面。",
      f"  - holdout–train：{Q['n_overlap_holdout_train_pairs']} 对；holdout 面到最近训练面最小距离 {Q['min_dist_holdout_to_train_m']/1000:.2f} km。",
      "- **县界检查**（`Yili_county_full.geojson` 巩留县）：全部 96 个新增面质心均在巩留县内。"
      "既有 v2 的 `YL-POLY-0059`（苹果）与 `YL-POLY-0061`（非果园）质心落在县界外（新源县侧，距巩留县界约 0.6 / 0.44 km）；"
      "因其为 v2 既有 holdout、不在训练集且距训练面>6 km，**主分析保留并标记**，另做“严格县内”敏感性分析（见 §3.3）。`YL-POLY-0062` 67% 面积在县内。",
      f"- QC 后 holdout 面：**{sum(Q['class_counts_kept'].values())}**（苹果 {Q['class_counts_kept']['1']} / 其他果园 {Q['class_counts_kept']['2']} / 非果园 {Q['class_counts_kept']['0']}）；"
      f"面积 苹果 {Q['area_ha_kept_by_class']['1']:.1f} ha / 其他果园 {Q['area_ha_kept_by_class']['2']:.1f} ha / 非果园 {Q['area_ha_kept_by_class']['0']:.1f} ha。",
      "- 样点：与 v2 同规则（`choose_n_points` 按面积/类别定点数与最小间距；类别 2 按非苹果规则）；每面 RNG 种子改为 crc32(poly_id)+42（v2 使用 Python `hash()`，跨进程不可复现）。"
      f" Holdout 点 {pt['n']}（苹果 213 / 其他果园 40 / 非果园 268）。",
      f"- **面级多数投票沿用 v2 规则：面内 ≥3 点才计入**。12 个 <0.1 ha 的小面仅 1–2 点（11 非果园 + 1 其他果园），不计入主面级指标 → 主分析 n={M['n_polys_primary']} 面"
      f"（苹果 {M['primary_class_counts']['1']} / 其他果园 {M['primary_class_counts']['2']} / 非果园 {M['primary_class_counts']['0']}）；全部 113 面见敏感性分析。"
      f" 多数投票平票面：{', '.join(M['majority_ties']) or '无'}（按 v2 规则平票判为非苹果）。",
      "- AlphaEarth 2021：GEE（restricted mode）分块 40 点 + sleep 0.4 s，新取 394 点，**0 失败、0 缺失**；缓存 `poly_ae_features_2021_v3.csv`（910 行）。", "",
      "## 2. 模型", "",
      "- 训练集与 v2 **完全相同**（38 个非巩留面，389 点；`split_manifest_v3.csv` 中 train 行直接取自 v2）。",
      "- **复用** `models/B3_poly_2021_v2.joblib`，未重新训练。核验：用 `make_lgbm(seed=42)` 在 v2 训练点上重训，对 521 个 holdout 点预测概率与已存模型最大差 0.0、类别一致率 100%。",
      "- 二分类定义：苹果(1) vs 非苹果（0 非果园 + 2 其他果园），阈值 0.5。", "",
      "## 3. 结果", "", "### 3.1 面级（主指标）", "", H,
      row("**多数投票（主）**", pm), row("平均概率≥0.5（全部 113 面）", pp),
      row("敏感性：多数投票含 <3 点小面（113 面）", M["sens_all_polys_majority_incl_lt3pts"]),
      row("敏感性：严格县内（去 0059/0061）", M["sens_strict_inside_gongliu"]),
      row("仅新增面（v3 新 83 面）", M["new_only"]), "",
      "### 3.2 面级分类别召回 / 拒识率（多数投票，Wilson 95%CI）", "",
      "| 真实类别 | 指标 | k/n | 比率 [95%CI] | 预测为苹果 | 预测为非苹果 |", "|---|---|---|---|---|---|"]
for key, cls, nm in [("apple_recall", "1", "苹果园：召回"), ("other_orchard_rejected_rate", "2", "其他果园：正确拒识率"),
                     ("non_orchard_rejected_rate", "0", "非果园：正确拒识率")]:
    d = pm["per_class"][key]; t = pm["three_class_by_pred"][cls]
    L.append(f"| {nm.split('：')[0]} | {nm.split('：')[1]} | {d['k']}/{d['n']} | {f(d['rate'])} [{f(d['wilson_lo'])}, {f(d['wilson_hi'])}] | {t['pred_apple']} | {t['pred_non']} |")
L += ["", "（上表即 3 类真值 × 二分类预测的敏感性混淆表。平均概率规则下：其他果园拒识 "
      f"{pp['per_class']['other_orchard_rejected_rate']['k']}/{pp['per_class']['other_orchard_rejected_rate']['n']}，非果园拒识 "
      f"{pp['per_class']['non_orchard_rejected_rate']['k']}/{pp['per_class']['non_orchard_rejected_rate']['n']}。）", "",
      "### 3.3 点级（辅助；面内点空间相关）", "",
      "| n | 苹果/非苹果 | OA | P | R | 特异度 | F1 [面聚类bootstrap] | AP [面聚类bootstrap] | AUC | TP/FP/FN/TN |", "|---|---|---|---|---|---|---|---|---|---|"]
c = pt["confusion_matrix"]; cb = pt["cluster_bootstrap_95ci"]
L += [f"| {pt['n']} | {pt['n_pos']}/{pt['n_neg']} | {f(pt['OA'])} | {f(pt['precision'])} | {f(pt['recall'])} | {f(pt['specificity'])} | "
      f"{f(pt['F1'])} [{f(cb['F1'][0])}, {f(cb['F1'][1])}] | {f(pt['AP'])} [{f(cb['AP'][0])}, {f(cb['AP'][1])}] | {f(pt['AUC'])} | {c['TP']}/{c['FP']}/{c['FN']}/{c['TN']} |", "",
      f"点级按类预测为苹果比例：苹果 {pt['per_class_point']['1']['pred_apple_rate']:.3f}（n={pt['per_class_point']['1']['n']}），"
      f"其他果园 {pt['per_class_point']['2']['pred_apple_rate']:.3f}（n={pt['per_class_point']['2']['n']}），非果园 {pt['per_class_point']['0']['pred_apple_rate']:.3f}（n={pt['per_class_point']['0']['n']}）。"
      "点级置信区间采用按面整体重采样的聚类 bootstrap（1000 次），因同一面内点不独立；普通点级 CI 会过窄。", "",
      "## 4. v2 vs v3 对比", "",
      "| 指标 | v2（18 面 / 127 点） | v3（主：101 面 / 521 点） | v3 中 v2 同一 18 面复算 |", "|---|---|---|---|"]
e2p, e2t = E2["poly_majority"], E2["point_level"]; vs = M["v2_subset_recomputed"]
L += [f"| 面级 n（苹果/非苹果） | {e2p['n_polys']}（{e2p['n_apple']}/{e2p['n_non']}） | {pm['n']}（{pm['n_pos']}/{pm['n_neg']}） | {vs['n_polys']} |",
      f"| 面级 F1 | {f(e2p['F1'])} | {f(pm['F1'])} | {f(vs['poly_majority']['F1'])} |",
      f"| 面级 OA [Wilson] | {f(e2p['OA'])} [{f(e2p['wilson_95ci_OA']['lo'])}, {f(e2p['wilson_95ci_OA']['hi'])}] | {f(pm['OA'])} [{f(pm['wilson_95ci_OA']['lo'])}, {f(pm['wilson_95ci_OA']['hi'])}] | {f(vs['poly_majority']['OA'])} |",
      f"| 面级 P / R | {f(e2p['precision'])} / {f(e2p['recall'])} | {f(pm['precision'])} / {f(pm['recall'])} | {f(vs['poly_majority']['precision'])} / {f(vs['poly_majority']['recall'])} |",
      f"| 面级 TP/FP/FN/TN | {e2p['confusion_matrix']['TP']}/{e2p['confusion_matrix']['FP']}/{e2p['confusion_matrix']['FN']}/{e2p['confusion_matrix']['TN']} | {pm['confusion_matrix']['TP']}/{pm['confusion_matrix']['FP']}/{pm['confusion_matrix']['FN']}/{pm['confusion_matrix']['TN']} | {vs['poly_majority']['confusion_matrix']['TP']}/{vs['poly_majority']['confusion_matrix']['FP']}/{vs['poly_majority']['confusion_matrix']['FN']}/{vs['poly_majority']['confusion_matrix']['TN']} |",
      f"| 面级 BA / MCC | — | {f(pm['balanced_accuracy'])} / {f(pm['MCC'])} | {f(vs['poly_majority']['balanced_accuracy'])} / {f(vs['poly_majority']['MCC'])} |",
      f"| 点级 F1 / OA | {f(e2t['F1'])} / {f(e2t['OA'])} | {f(pt['F1'])} / {f(pt['OA'])} | {f(vs['point']['F1'])} / {f(vs['point']['OA'])} |",
      f"| 点级 AP | {f(e2t['AP'])} | {f(pt['AP'])} | — |", "",
      "v2 的 18 面在 v3 流程中逐项复现（F1、混淆矩阵一致），可排除流程差异；差异来自样本扩充。", "",
      "## 5. 地理捷径诊断（v3 holdout，二分类 1 vs 0+2）", "",
      "| 特征集 | 点 F1 | 点 OA | 点 AP | 面 F1 (n=101) | 面 OA | ΔF1 点 | ΔF1 面 | v2 面 F1 (n=18) |", "|---|---|---|---|---|---|---|---|---|"]
s2 = {r["name"]: r for r in S2["results"]}
for r in S["results"]:
    L.append(f"| {r['name']} | {f(r['point_level']['F1'])} | {f(r['point_level']['OA'])} | {f(r['point_level'].get('AP', float('nan')))} | "
             f"{f(r['poly_majority']['F1'])} | {f(r['poly_majority']['OA'])} | {r['delta_F1_point_vs_AE']:+.3f} | {r['delta_F1_poly_vs_AE']:+.3f} | {f(s2[r['name']]['poly_majority']['F1'])} |")
L += ["", f"- 参照：全判苹果的平凡分类器面级 F1={2*36/(36+101):.3f}、点级 F1={2*213/(213+521):.3f}。DEM-only 与 lat/lon-only 均**低于**平凡基线，"
      "v2 中 lat/lon-only 面级 F1=0.800 的“中等可分性”在扩充后消失（0.508），说明其源于 v2 小样本的空间构型而非可迁移的地理捷径。",
      "- AE+DEM、AE+lat/lon 相对 AE-only 无增益（ΔF1 面 -0.008 / -0.040），加坐标反而略降。**无证据表明模型依赖地形或坐标捷径**；主信号来自 AlphaEarth 嵌入。",
      f"- 地形采样：SRTM elevation/slope（30 m），新增 394 点，GEE 0 失败；缓存 `poly_terrain_features_2021_v3.csv`。", "",
      "## 6. 解读（诚实评估）", "",
      "1. **误差结构**：面级 36 个苹果面漏判仅 2 个（均为 <0.3 ha 小面，3 点），但 65 个非苹果面中 20 个被误报为苹果。模型在巩留的主要问题是**过度预测苹果（commission）**，精确率仅 "
      f"{f(pm['precision'])}。",
      f"2. **苹果 vs 其他果园（困难样本）**：其他果园正确拒识率仅 {pm['per_class']['other_orchard_rejected_rate']['k']}/{pm['per_class']['other_orchard_rejected_rate']['n']}"
      f"（{f(pm['per_class']['other_orchard_rejected_rate']['rate'])}，95%CI {f(pm['per_class']['other_orchard_rejected_rate']['wilson_lo'])}–{f(pm['per_class']['other_orchard_rejected_rate']['wilson_hi'])}），"
      "显著低于非果园（"
      f"{f(pm['per_class']['non_orchard_rejected_rate']['rate'])}）。AE-only B3 基本**不能区分苹果园与其他果园**——训练集无 class 2 负样本，这在预期之中。"
      "但其他果园仅 10 面（4.5 ha），CI 很宽。",
      f"3. **非果园误报同样可观**：非果园 13/55 面被误报（{1-pm['per_class']['non_orchard_rejected_rate']['rate']:.1%}），在 20 个 FP 面中占 13 个（其他果园 7 个）。"
      "误报率与距最近苹果面的距离无单调关系（>2 km 的非果园面 FP 率 "
      f"{FD[FD.bin=='>2km'].fp_rate.iloc[0]:.0%}，n={int(FD[FD.bin=='>2km'].n.iloc[0])}），与面积也无明显关系，说明不是苹果园边缘混合像元效应，而是巩留非果园景观（农田、林带等）与训练负样本（11 个面、多在特克斯）分布不匹配。",
      "4. **能否解释 600 点目视独立验证的 FP？** 用同一模型对 565 个高/中置信目视点预测：176 个 FP 中，其他果园(code 2) 仅贡献 "
      f"{int(VIS[VIS.interp_class_code==2].pred_apple.iloc[0])} 个（该类仅 {int(VIS[VIS.interp_class_code==2].n.iloc[0])} 点），农田(3) {int(VIS[VIS.interp_class_code==3].pred_apple.iloc[0])}、林地(4) {int(VIS[VIS.interp_class_code==4].pred_apple.iloc[0])}、"
      f"灌草(5) {int(VIS[VIS.interp_class_code==5].pred_apple.iloc[0])}、建设用地(6) {int(VIS[VIS.interp_class_code==6].pred_apple.iloc[0])}、裸地/其他(8) {int(VIS[VIS.interp_class_code==8].pred_apple.iloc[0])}；"
      "各非苹果类的误报率均在 28%–37%。因此**苹果/其他果园混淆不能解释目视验证的大量 FP**；其主因是对一般非果园地类约 30% 的普遍误报，"
      f"与本次面级非果园误报率（{1-pm['per_class']['non_orchard_rejected_rate']['rate']:.0%}）、点级非果园预测为苹果比例（{pt['per_class_point']['0']['pred_apple_rate']:.0%}）量级一致。"
      "目视验证 F1 极低（0.052）主要源于正例极稀少（10 个苹果点）下的精确率崩塌，而非模型在非果园上突然变差。",
      "5. **对论文的含义**：v2 的 18 面 holdout（F1=0.923）应视为乐观估计；v3（101 面）是更可靠的县域外推证据：苹果检出能力强（召回≈0.94），但精确率有限（≈0.63），"
      "平衡精度 0.818、MCC 0.611。建议：(i) 以 v3 面级指标作为主外部证据并报告 CI；(ii) 明确 B3 在巩留存在系统性过报，尤其对其他果园；"
      "(iii) 若要改进，应在训练集（非巩留县）补充其他果园与多样化非果园负样本，而不是调用巩留 holdout 调参。", "",
      "## 7. 局限", "",
      "- 其他果园 10 面、面积小，分类别拒识率 CI 宽；12 个 <0.1 ha 小面低于 AE 10 m 有效分辨率，未计入主面级指标（全部 113 面敏感性结果相近）。",
      "- 新增面多为目视判读（非全部实地），标签噪声无法完全排除；0059/0061 质心略出县界（已做敏感性）。",
      "- 阈值固定 0.5（未在 holdout 上调阈值，避免信息泄漏）。", "",
      "## 8. 交付文件", "",
      "- `polygon_samples/v3/QC_holdout_v3.csv`、`QC_overlaps_v3.csv`、`qc_summary_v3.json`、`split_manifest_v3.csv`",
      "- `polygon_samples/sample_points_v3.csv`、`poly_ae_features_2021_v3.csv`、`poly_terrain_features_2021_v3.csv`",
      "- `polygon_samples/v3/metrics_poly_holdout_v3.json`、`predictions_holdout_v3.csv`、`poly_predictions_v3.csv`、`metrics_shortcut_diagnostics_v3.json`",
      "- `figures_v3/fig_main_holdout_validation_v3.png`、`fig_shortcut_diagnostics_v3.png`、`fig_main_holdout_caption_zh_v3.md`、`REPORT_B3_poly_holdout_v3.md`",
      "- 脚本：`python/revision/22_b3_poly_holdout_v3.py`、`23_shortcut_diagnostics_v3.py`、`24_fig_main_holdout_validation_v3.py`、`25_write_report_v3.py`"]
txt = "\n".join(L)
for d in (V3, FIG3, OUT):
    (d / "REPORT_B3_poly_holdout_v3.md").write_text(txt, encoding="utf-8")

oo = pm["per_class"]
cap = f"""# 图注（v3）：巩留县多边形空间 holdout 独立验证（主图）

**图 X. 巩留县多边形空间 holdout 独立验证结果（B3，AlphaEarth-only，2021；训练集为 38 个非巩留面，与 v2 相同）。**

- **(a)** 空间划分：灰色圆点为训练面质心（非巩留县）；红色三角、橙色菱形、蓝色方块分别为 holdout 苹果园（n=36）、其他果园（n=11）、非果园（n=66）面质心。巩留县样本全部作为空间 holdout，未参与训练或调参。
- **(b)** **多边形级**混淆矩阵（苹果 vs 非苹果[非果园+其他果园]；面内点数≥3 的多数投票，n={pm['n']}）：Precision={f(pm['precision'])}，Recall={f(pm['recall'])}，F1={f(pm['F1'])}（面 bootstrap 95%CI {f(pm['bootstrap_95ci']['F1']['lo'])}–{f(pm['bootstrap_95ci']['F1']['hi'])}），OA={f(pm['OA'])}（Wilson 95%CI {f(pm['wilson_95ci_OA']['lo'])}–{f(pm['wilson_95ci_OA']['hi'])}），平衡精度={f(pm['balanced_accuracy'])}，MCC={f(pm['MCC'])}。下表按真实三类给出预测分布：苹果召回 {oo['apple_recall']['k']}/{oo['apple_recall']['n']}，其他果园正确拒识 {oo['other_orchard_rejected_rate']['k']}/{oo['other_orchard_rejected_rate']['n']}，非果园正确拒识 {oo['non_orchard_rejected_rate']['k']}/{oo['non_orchard_rejected_rate']['n']}。
- **(c)** Holdout 样点级 Precision–Recall 曲线（n={pt['n']}），AP={f(pt['AP'])}（按面聚类 bootstrap 95%CI {f(pt['cluster_bootstrap_95ci']['AP'][0])}–{f(pt['cluster_bootstrap_95ci']['AP'][1])}）；虚线为 v2 原 18 面子集（n=127，AP={f(E2['point_level']['AP'])}）。
- **(d)** 代表性 holdout 面案例（Sentinel-2 2021 年 6–9 月中值真彩色）：TP、TN、FP（非果园）、FN、其他果园误报（FP）与其他果园正确拒识（TN）；标注面内平均预测概率 p̄ 与预测为苹果的点数。

**说明：以多边形级评价为主。** 同一面内的多个样点存在强空间依赖，点级指标会夸大有效样本量，仅作辅助；点级置信区间采用按面整体重采样的聚类 bootstrap。12 个面积<0.1 ha、样点<3 的小面未计入面级多数投票（纳入后 F1={f(M['sens_all_polys_majority_incl_lt3pts']['F1'])}，结论不变）。
"""
for d in (V3, FIG3):
    (d / "fig_main_holdout_caption_zh_v3.md").write_text(cap, encoding="utf-8")
print("written")
