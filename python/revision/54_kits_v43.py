# -*- coding: utf-8 -*-
"""v4.3 optional kits (do not block the paper).
(1) map_accuracy_sampling: stratified random sample on the Gongliu 10 m expanded-B3 class map (the map whose pixel-count area, 6707 ha, is
    discussed in the paper). Strata: A = mapped apple (200 pts); B = mapped non-apple within 10 px (chessboard, ~73-100 m) of mapped apple (100 pts);
    C = remaining mapped non-apple (200 pts). Equal-probability pixel sampling inside each stratum (rng 20260928). No reference label is used.
(2) second_interpreter_blind: 25% of the 113 Gongliu challenge-set polygons, stratified by the v4.3 main label (rng 20260929), labels stripped."""
import sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy import ndimage
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; sys.path.insert(0, str(REV)); from kml_utils import write_kml, style
K = ROOT / "outputs/revision/v43/kits"; PS = ROOT / "outputs/revision/polygon_samples"
MAP = ROOT / "outputs/revision/v4_experiments/retrain_expanded/county_map_B3/gongliu_B3expanded_class_10m.tif"
# ---------------------------------------------------------------- (1)
D1 = K / "map_accuracy_sampling"; FI = D1 / "给判读者_for_interpreter"; HID = D1 / "_design_仅供作者_do_not_open_before_filling"
for d in (FI, HID): d.mkdir(parents=True, exist_ok=True)
with rasterio.open(MAP) as r: a = r.read(1); T = r.transform
valid = a != 255; app = a == 1
buf = ndimage.binary_dilation(app, structure=np.ones((3, 3), bool), iterations=10) & valid & ~app
rest = valid & ~app & ~buf
H, W = a.shape; lat = T.f + T.e * (np.arange(H) + 0.5)
R_E = 6371008.8; dy = np.deg2rad(abs(T.e)) * R_E; dx = np.deg2rad(T.a) * R_E * np.cos(np.deg2rad(lat)); parea = dy * dx / 1e4  # ha per pixel, per row
strata = {"A_mapped_apple": (app, 200), "B_nonapple_buffer_10px": (buf, 100), "C_nonapple_rest": (rest, 200)}
rng = np.random.default_rng(20260928); rows = []; design = {"map": str(MAP), "map_area_rule": "pixel-count; EPSG:4326 pixel area by row latitude (sphere R=6371008.8 m)", "strata": {}}
for sname, (m, n) in strata.items():
    cnt_row = m.sum(1); area = float((cnt_row * parea).sum()); npx = int(cnt_row.sum())
    idx = np.flatnonzero(m.ravel()); pick = rng.choice(idx, n, replace=False); rr, cc = np.divmod(pick, W)
    lon = T.c + T.a * (cc + 0.5); la = T.f + T.e * (rr + 0.5)
    design["strata"][sname] = {"n_pixels": npx, "area_ha": area, "n_sample": n, "map_class": 1 if sname.startswith("A") else 0}
    rows += [{"stratum": sname, "map_class": 1 if sname.startswith("A") else 0, "lon": float(x), "lat": float(y), "row": int(i), "col": int(j)} for x, y, i, j in zip(lon, la, rr, cc)]
    del idx
S = pd.DataFrame(rows).sample(frac=1, random_state=20260928).reset_index(drop=True); S.insert(0, "point_id", [f"MA-{i:04d}" for i in range(1, len(S) + 1)])
S.to_csv(HID / "sampling_design_points.csv", index=False); design["total_area_ha"] = sum(v["area_ha"] for v in design["strata"].values())
design["seed"] = 20260928; json.dump(design, open(HID / "sampling_design.json", "w"), ensure_ascii=False, indent=1)
tmpl = S[["point_id", "lon", "lat"]].copy().round({"lon": 7, "lat": 7})
for c in ("ref_class", "ref_subtype", "img_date", "confidence", "notes"): tmpl[c] = ""
with pd.ExcelWriter(FI / "map_accuracy_points_template.xlsx") as xw:
    tmpl.to_excel(xw, sheet_name="points", index=False)
    pd.DataFrame({"ref_class": [1, 0, 2, 9], "含义": ["栽培苹果园（2021年）", "非苹果（农田、林地/防护林、天山野苹果林、草地、建设用地、水体、裸地等）", "其他果园（非苹果果树园地；统计时并入非苹果）", "无法判断（统计时剔除并报告数量）"]}).to_excel(xw, sheet_name="codes", index=False)
write_kml(FI / "map_accuracy_points.kml", "巩留县分布图精度评价样点（500个，未显示制图类别）", [style("p", "ff00ffff", "00ffffff", 1.5)],
          [("样点 MA-0001..0500", [{"name": r.point_id, "desc": "", "geom": None, "point": (r.lon, r.lat), "style": "p"} for r in S.itertuples()])])
print(json.dumps(design, ensure_ascii=False, indent=1))
# ---------------------------------------------------------------- (2)
D2 = K / "second_interpreter_blind"; FI2 = D2 / "给第二判读者_for_interpreter"; KEY = D2 / "_key_仅供作者_do_not_share"
for d in (FI2, KEY): d.mkdir(parents=True, exist_ok=True)
P = gpd.read_file(PS / "v3/polygons_v3_all.geojson"); Hh = P[P.split == "val_holdout"].copy(); assert len(Hh) == 113
RELAB = {"YL-POLY-0059": 0, "YL-POLY-0060": 0, "YL-POLY-0123": 0, "YL-POLY-0124": 0, "YL-POLY-0125": 0, "YL-POLY-0083": 0, "YL-POLY-0082": 0}
Hh["label_original"] = Hh.class_code.astype(int); Hh["label_v43"] = [RELAB.get(n, c) for n, c in zip(Hh.name, Hh.label_original)]
rg = np.random.default_rng(20260929); pick = []
for c, g in Hh.groupby("label_v43"):
    k = int(round(len(g) * 0.25)); pick += list(rg.choice(g.name.values, k, replace=False))
B = Hh[Hh.name.isin(pick)].sample(frac=1, random_state=20260929).reset_index(drop=True); B["blind_id"] = [f"SI-{i:03d}" for i in range(1, len(B) + 1)]
rp = B.geometry.representative_point(); B["lon"], B["lat"] = rp.x.round(7), rp.y.round(7)
B[["blind_id", "name", "label_original", "label_v43", "lon", "lat"]].to_csv(KEY / "blind_key.csv", index=False)
t2 = B[["blind_id", "lon", "lat"]].copy()
for c in ("interp_class", "interp_subtype", "img_date", "confidence", "notes"): t2[c] = ""
with pd.ExcelWriter(FI2 / "second_interpreter_template.xlsx") as xw:
    t2.to_excel(xw, sheet_name="polygons", index=False)
    pd.DataFrame({"interp_class": [1, 2, 0, 9], "含义": ["栽培苹果园", "其他果园（非苹果果树园地）", "非果园（含天山野苹果林、防护林带、农田、草地等）", "无法判断"]}).to_excel(xw, sheet_name="codes", index=False)
write_kml(FI2 / "second_interpreter_polygons.kml", f"第二判读者盲判：{len(B)}个园块面（不含类别）", [style("b", "ff00ffff", "00ffffff", 2.5)],
          [("园块 SI-xxx", [{"name": r.blind_id, "desc": "", "geom": r.geometry, "point": None, "style": "b"} for r in B.itertuples()])])
print("blind kit:", len(B), B.label_v43.value_counts().to_dict(), B.label_original.value_counts().to_dict())
json.dump({"n": int(len(B)), "by_label_v43": {int(k): int(v) for k, v in B.label_v43.value_counts().items()}, "seed": 20260929}, open(KEY / "blind_sampling.json", "w"), indent=1)
