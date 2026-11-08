# -*- coding: utf-8 -*-
"""A (kit): visual-check kit for AOMC vs model disagreement in Gongliu. Random (seed 20260926) 20 AOMC-2021 apple patches
(8-connected components of AOMC apple pixels on the native 30 m grid, >=2 px, inside the official GS(2024)0650 Gongliu boundary)
and 20 model high-probability apple patches (8-connected components of new-B3 10 m pixels with p>=0.8, >=20 px of the 8.98e-5 deg grid ~ 0.14 ha).
Outputs: KML (two coloured folders, AOMC-01.. cyan, MOD-01.. magenta; patch outline + labelled centre point) and an xlsx template."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.features import shapes, rasterize
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; sys.path.insert(0, str(REV)); sys.path.insert(0, str(REV))
import official_boundary as ob; from kml_utils import write_kml, style
E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; OUT = ROOT / "outputs/revision/v41_review/kits/map_agreement_check"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260926; N = 20; rng = np.random.default_rng(SEED)
gl = ob.gongliu(); ST = np.ones((3, 3))
def comps(mask, T, crs, minpx, extra=None):
    lab, n = ndimage.label(mask, structure=ST); sz = np.bincount(lab.ravel()); ok = np.nonzero(sz >= minpx)[0]; ok = ok[ok > 0]
    return lab, ok, sz
# AOMC patches
with rasterio.open(ROOT / "data/aomc/AOMC_30_Xinjiang_2021.tif") as s:
    b = transform_bounds("EPSG:4326", s.crs, *(gl.total_bounds + np.array([-.01, -.01, .01, .01])))
    w = from_bounds(*b, s.transform).round_offsets().round_lengths(); A = s.read(1, window=w); TA = s.window_transform(w); CA = s.crs
inside = rasterize([(g, 1) for g in gl.to_crs(CA).geometry], out_shape=A.shape, transform=TA, fill=0, dtype="uint8") == 1
labA, okA, szA = comps((A == 1) & inside, TA, CA, 2)
selA = rng.choice(okA, N, replace=False)
# model patches
with rasterio.open(E / "county_map_B3/gongliu_B3expanded_prob_10m.tif") as s: P = s.read(1); TP = s.transform; CP = s.crs
hp = (P != 255) & (P >= 200)  # p = value/250 >= 0.8
labM, okM, szM = comps(hp, TP, CP, 20)
selM = rng.choice(okM, N, replace=False)
def patch_geoms(lab, ks, T, crs):
    res = {}
    objs = ndimage.find_objects(lab)
    for k in ks:
        sl = objs[k - 1]; sub = (lab[sl] == k).astype(np.uint8)
        t = T * rasterio.Affine.translation(sl[1].start, sl[0].start)
        g = unary_union([shape(gg) for gg, v in shapes(sub, mask=sub == 1, transform=t) if v == 1])
        res[k] = gpd.GeoSeries([g], crs=crs).to_crs(4326).iloc[0]
    return res
gA = patch_geoms(labA, selA, TA, CA); gM = patch_geoms(labM, selM, TP, CP)
rows, fa, fm = [], [], []
objsM = ndimage.find_objects(labM)
for i, k in enumerate(selA, 1):
    g = gA[k]; p = g.representative_point(); a_ha = float(szA[k] * 900 / 1e4)
    rows.append({"id": f"AOMC-{i:02d}", "source": "AOMC_2021", "lon": round(p.x, 6), "lat": round(p.y, 6), "visual_class": "", "img_date": "", "notes": "", "_area_ha": round(a_ha, 2), "_n_px": int(szA[k]), "_mean_prob_newB3": None})
    fa.append({"name": f"AOMC-{i:02d}", "desc": f"AOMC 2021 apple patch, {szA[k]} px (30 m), ~{a_ha:.2f} ha", "geom": g, "point": (p.x, p.y), "style": "aomc"})
for i, k in enumerate(selM, 1):
    g = gM[k]; p = g.representative_point(); sl = objsM[k - 1]; mp = float(P[sl][labM[sl] == k].mean() / 250); a_ha = float(gpd.GeoSeries([g], crs=4326).to_crs(32644).area.iloc[0] / 1e4)
    rows.append({"id": f"MOD-{i:02d}", "source": "newB3_p>=0.8", "lon": round(p.x, 6), "lat": round(p.y, 6), "visual_class": "", "img_date": "", "notes": "", "_area_ha": round(a_ha, 2), "_n_px": int(szM[k]), "_mean_prob_newB3": round(mp, 3)})
    fm.append({"name": f"MOD-{i:02d}", "desc": f"new B3 (expanded) p>=0.8 patch, {szM[k]} px (10 m grid), {a_ha:.2f} ha, mean p={mp:.2f}", "geom": g, "point": (p.x, p.y), "style": "mod"})
df = pd.DataFrame(rows)
styles = [style("aomc", "ffffff00", "33ffff00"), style("mod", "ffff00ff", "33ff00ff")]  # KML aabbggrr: cyan / magenta
write_kml(OUT / "map_agreement_check_Gongliu.kml", "AOMC vs 新B3 一致性目视核查（巩留县，各20个随机斑块）", styles,
          [("AOMC 2021 苹果斑块（青色，AOMC-01..20）", fa), ("新B3 高概率苹果斑块 p≥0.8（品红，MOD-01..20）", fm)],
          f"随机种子 {SEED}; AOMC斑块=AOMC 2021苹果像元8连通区（30 m, ≥2像元）；模型斑块=新B3 10 m p≥0.8像元8连通区（≥20像元，约0.14 ha）。县界：GS(2024)0650。")
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Font, PatternFill, Alignment
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "visual_check"
cols = ["id", "source", "lon", "lat", "visual_class", "img_date", "notes"]; ws.append(cols)
for r in df[cols].itertuples(index=False): ws.append(list(r))
dv = DataValidation(type="list", formula1='"apple,other_orchard,non_orchard,unsure"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"E2:E{len(df)+1}")
for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDEBF7")
for col, wdt in zip("ABCDEFG", (10, 14, 12, 12, 16, 14, 40)): ws.column_dimensions[col].width = wdt
ws.freeze_panes = "A2"
wi = wb.create_sheet("patch_info"); ic = ["id", "source", "_area_ha", "_n_px", "_mean_prob_newB3"]; wi.append(["id", "source", "patch_area_ha", "n_pixels", "mean_prob_newB3"])
for r in df[ic].itertuples(index=False): wi.append(list(r))
wr = wb.create_sheet("说明")
for line in ["目的：核查AOMC 2021苹果图层与新B3 10 m苹果图之间低重叠（本模型苹果像元仅5.7%落在AOMC内；AOMC苹果仅14.2%被本模型识别，10 m网格）的原因。",
             "操作：在Google Earth Pro中打开 map_agreement_check_Gongliu.kml；逐个双击 AOMC-xx / MOD-xx，打开历史影像（时钟图标），优先选2020—2022年清晰影像判读斑块主体地类。",
             "visual_class 只能填：apple（苹果园）/ other_orchard（其他果园或经济林）/ non_orchard（耕地、林带、草地、建筑等非果园）/ unsure（无法判断）。",
             "img_date：填写所用Google Earth历史影像的日期（YYYY-MM-DD 或 YYYY-MM）。notes：可写“防护林带”“葡萄”“幼龄园”“斑块只有部分是果园”等。",
             "为减少偏差：建议先不看 patch_info 工作表（含面积和模型概率），判读完成后再看。",
             f"抽样：随机种子 {SEED}；AOMC斑块总体 {len(okA)} 个（≥2像元），模型斑块总体 {len(okM)} 个（p≥0.8、≥20像元）。县界来源：GS(2024)0650。"]:
    wr.append([line])
wr.column_dimensions["A"].width = 120
wb.save(OUT / "map_agreement_check_template.xlsx")
json.dump({"seed": SEED, "n_AOMC_patches_pop": int(len(okA)), "n_model_patches_pop": int(len(okM)), "sampled": df.to_dict("records")}, open(OUT / "sampling_log.json", "w"), ensure_ascii=False, indent=1, default=str)
print(len(okA), len(okM)); print(df.to_string())
