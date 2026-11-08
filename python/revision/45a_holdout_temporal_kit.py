# -*- coding: utf-8 -*-
"""B (kit): temporal-metadata kit for all 113 challenge-set (holdout) polygons: KML (labelled by ID + class, coloured by class,
main-101 vs <3-point polygons in separate folders) + xlsx template + Chinese README. No values are pre-filled (dates were not recorded)."""
import sys, json
from pathlib import Path
import pandas as pd, geopandas as gpd
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; sys.path.insert(0, str(REV)); from kml_utils import write_kml, style
PS = ROOT / "outputs/revision/polygon_samples"; E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"
OUT = ROOT / "outputs/revision/v41_review/kits/holdout_temporal_check"; OUT.mkdir(parents=True, exist_ok=True)
P = gpd.read_file(PS / "v3/polygons_v3_all.geojson"); H = P[P.split == "val_holdout"].copy()
man = pd.read_csv(PS / "split_manifest_v3.csv"); H = H.merge(man[["name", "county", "label_source", "cen_lon", "cen_lat"]], on="name", how="left")
pp = pd.read_csv(E / "pred_polys_B3_expanded_seed42.csv")[["poly_name", "n_pts", "pred_major"]]
H = H.merge(pp, left_on="name", right_on="poly_name", how="left"); H["in_main101"] = H.pred_major.notna()
assert len(H) == 113 and H.in_main101.sum() == 101
CN = {1: "苹果园", 2: "其他果园", 0: "非果园"}; ST = {1: "c1", 2: "c2", 0: "c0"}
styles = [style("c1", "ff0000ff", "330000ff"), style("c2", "ff00ff00", "3300ff00"), style("c0", "ffffffff", "22ffffff")]
H = H.sort_values("name")
def items(d): return [{"name": f"{r.name} | c{int(r.class_code)} {CN[int(r.class_code)]}", "desc": f"{r.name}; class_code={int(r.class_code)} ({CN[int(r.class_code)]}); area={r.area_ha_utm44n:.2f} ha; source={r.label_source}; main101={bool(r.in_main101)}",
                      "geom": r.geometry, "point": (r.geometry.representative_point().x, r.geometry.representative_point().y), "style": ST[int(r.class_code)]} for r in d.itertuples()]
folders = []
for flag, nm in ((True, "主评价101面（≥3个样点）"), (False, "其余12面（<3个样点，仅敏感性分析）")):
    for c in (1, 2, 0):
        d = H[(H.in_main101 == flag) & (H.class_code == c)]
        if len(d): folders.append((f"{nm}—{CN[c]}（{len(d)}）", items(d)))
write_kml(OUT / "holdout_113_polygons_temporal_check.kml", "巩留县跨县域挑战集113面：影像日期与时间稳定性核查", styles, folders,
          "红=苹果园(c1)，绿=其他果园(c2)，白=非果园(c0)。")
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Font, PatternFill
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "holdout_temporal"
cols = ["polygon_id", "class", "image_date", "field_verified", "temporal_stable_2019_2023", "notes"]; ws.append(cols)
for r in H.itertuples(): ws.append([r.name, int(r.class_code), "", "", "", ""])
n = len(H) + 1
for rng_, f in ((f"D2:D{n}", '"Y,N"'), (f"E2:E{n}", '"Y,N,unsure"')):
    dv = DataValidation(type="list", formula1=f, allow_blank=True); ws.add_data_validation(dv); dv.add(rng_)
for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDEBF7")
for col, w in zip("ABCDEF", (15, 7, 14, 14, 24, 45)): ws.column_dimensions[col].width = w
ws.freeze_panes = "A2"
wi = wb.create_sheet("polygon_info"); wi.append(["polygon_id", "class", "class_name", "in_main101", "n_points", "area_ha", "cen_lon", "cen_lat", "label_source"])
for r in H.itertuples(): wi.append([r.name, int(r.class_code), CN[int(r.class_code)], bool(r.in_main101), int(r.n_pts), round(r.area_ha_utm44n, 2), round(r.cen_lon, 6), round(r.cen_lat, 6), r.label_source])
wr = wb.create_sheet("填写规则")
for line in ["image_date：勾绘/判读该面时实际使用的Google Earth历史影像日期，格式 YYYY-MM-DD（只知道年月可写 YYYY-MM）。不确定请留空并在notes说明，不要估填。",
             "field_verified：该面是否经过实地核查（Y/N）。",
             "temporal_stable_2019_2023：用历史影像滑块查看2019—2023年，该面地类是否始终与class一致（Y=一致；N=有变化，如新建/砍伐/改种；unsure=影像缺失或无法判断）。",
             "notes：变化年份、影像缺失年份、可能错标等。", "class：1=苹果园，2=其他果园，0=非果园（请勿修改；如认为错标，在notes写明）。"]:
    wr.append([line])
wr.column_dimensions["A"].width = 130
wb.save(OUT / "holdout_113_temporal_template.xlsx")
readme = """# 挑战集113面：影像日期与时间稳定性核查说明

**文件**
- `holdout_113_polygons_temporal_check.kml`：113个挑战集面，每个面都标注了“编号 | 类别”。红色为苹果园（c1），绿色为其他果园（c2），白色为非果园（c0）。
  - 文件夹“主评价101面”：论文主结果用的101个面。
  - 文件夹“其余12面”：少于3个样点的面，只用于敏感性分析。
- `holdout_113_temporal_template.xlsx`：填写表，工作表“holdout_temporal”。“polygon_info”为参考信息；“填写规则”写明每列怎么填。

**用Google Earth Pro核查的步骤**
1. 打开 Google Earth Pro，选择“文件 → 打开”，载入 KML。
2. 在左侧“位置”面板中双击某个面（如 `YL-POLY-0011 | c1 苹果园`），视图会飞到该面。
3. 点击工具栏上的**时钟图标**（显示历史图像），地图上方会出现时间滑块。
4. **image_date**：把滑块拖到你当初勾绘这个面时用的那一景影像，记下左下角（或滑块上方）显示的影像日期。
   - 记不清是哪一景时，留空，并在 notes 中写“不确定，可能为XXXX年”。不要估填。
5. **temporal_stable_2019_2023**：依次查看2019、2020、2021、2022、2023年的影像（每年至少一景，尽量选生长季4—10月）。
   - **Y**：各年地类都与 class 一致。苹果园应能看到成行的树冠和规则株行距；非果园应始终不是果园。
   - **N**：任一年出现明显变化，例如新栽幼树、整片砍伐、改成耕地或建筑、由耕地改为果园。请在 notes 中写明变化年份。
   - **unsure**：关键年份没有影像、有云，或分辨率不足以判断。
6. **field_verified**：该面经过实地核查的填 Y，否则填 N。
7. 如果认为某个面标错了类别，**不要修改 class**，在 notes 中写明，例如“疑似葡萄园”“幼龄园”。

**填好后**：把 xlsx 发回即可。分析脚本 `python/revision/45b_temporal_subset_eval.py` 会自动筛选“时间一致子集”，条件是影像年份满足 |年份−2021|≤1，且 temporal_stable_2019_2023=Y。随后在该子集上重新计算 B0、B3、B6 和两步模型的 F1、P、R 及其置信区间，阈值固定为0.5，模型不重训、不调参。
"""
open(OUT / "README_挑战集时间核查.md", "w", encoding="utf-8").write(readme)
print(H.groupby(["in_main101", "class_code"]).size())
