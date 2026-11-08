# -*- coding: utf-8 -*-
"""Prefill field_verified in the 113-polygon holdout temporal kit from the user's own field GPS exports
(野外调查点\\GPS工具箱导出_*.xls/xlsx, all marked points, photos stripped -> data/field_survey/all_field_points_nophoto.csv).
Rules: point inside polygon -> field_verified=Y (suggested); nearest point <=30 m -> 'Y?' (check); 30-100 m -> note only; else unchanged.
Never writes N. Existing user entries in field_verified / notes are preserved (auto text is appended to notes, field_verified only filled if empty).
Field class is parsed from the point's own text (名称/备注/备注1-3); mismatches with the polygon class are flagged."""
import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; sys.path.insert(0, str(REV)); from kml_utils import write_kml, style
KIT = ROOT / "outputs/revision/v41_review/kits/holdout_temporal_check"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else KIT / "holdout_113_temporal_template.xlsx"
OUTX = KIT / "holdout_temporal_check_prefilled.xlsx"; OUTK = KIT / "holdout_field_points_matched.kml"
F = pd.read_csv(ROOT / "data/field_survey/all_field_points_nophoto.csv", dtype=str)
F = F.rename(columns={"_file": "src_file", "_lon": "plon", "_lat": "plat", "_row": "srow"}); F["plon"] = F.plon.astype(float); F["plat"] = F.plat.astype(float)
txtcols = [c for c in ("名称", "备注", "备注1", "备注2", "备注3") if c in F]
ADDR = re.compile(r"(新疆维吾尔自治区|自治州|县|镇|乡|村|市)$|^中国")
def text(r):
    parts = []
    for c in txtcols:
        v = r[c]
        if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
            v = v.strip()
            if c == "名称" and (ADDR.search(v) or re.fullmatch(r"[A-Za-z]?\d+", v)): continue   # address / pure serial names carry no class
            parts.append(v)
    return " ".join(parts)
APPLE = re.compile(r"苹果"); OTHER = re.compile(r"梨|桃|杏|核桃|葡萄|樱桃|枣|西梅|李子|猕猴桃|石榴|沙棘|海棠|山楂")
ORCH = re.compile(r"果园|果树|园地|林果"); NON = re.compile(r"玉米|水稻|稻|小麦|麦|甜菜|向日葵|葵花|南瓜|土豆|马铃薯|油菜|大豆|豆|棉|番茄|辣椒|打瓜|苜蓿|薰衣草|草地|草|杨树|白杨|林带|防护林|树林|房|建筑|村|道路|荒|裸|水|菜")
def fclass(t):
    a, o, orc, n = bool(APPLE.search(t)), bool(OTHER.search(t)), bool(ORCH.search(t)), bool(NON.search(t))
    base = "apple" if a else "other_fruit" if o else "orchard_unspecified" if orc else "non_orchard" if n else "unknown"
    mixed = (a or o or orc) and n
    return base, mixed
F["field_text"] = F.apply(text, axis=1)
cm = F.field_text.map(fclass); F["field_class"] = cm.str[0]; F["field_mixed"] = cm.str[1]
F["date"] = F["时间"].fillna("") if "时间" in F else ""
F["acc"] = F["精度"].fillna("") if "精度" in F else ""
F["fid"] = F.src_file.str.extract(r"(\d{8})")[0].radd("FP") + "-r" + F.srow.astype(str).str.zfill(5)
# de-duplicate identical records exported twice (same coordinates to 1e-6 deg and same time/text)
F["_k"] = F.plon.round(6).astype(str) + F.plat.round(6).astype(str) + F.date.astype(str) + F.field_text
n_raw = len(F); F = F.drop_duplicates("_k").copy()
G = gpd.GeoDataFrame(F, geometry=gpd.points_from_xy(F.plon, F.plat), crs=4326).to_crs(32644)

P = gpd.read_file(ROOT / "outputs/revision/polygon_samples/v3/polygons_v3_all.geojson"); H = P[P.split == "val_holdout"].to_crs(32644).set_index("name")
assert len(H) == 113
CN = {0: "non_orchard", 1: "apple", 2: "other_orchard"}
def mismatch(pc, fc):
    if fc in ("unknown",): return ""
    if pc == 1 and fc in ("other_fruit", "non_orchard"): return "MISMATCH"
    if pc == 2 and fc in ("apple", "non_orchard"): return "MISMATCH"
    if pc == 0 and fc in ("apple", "other_fruit", "orchard_unspecified"): return "MISMATCH"
    if pc in (1, 2) and fc == "orchard_unspecified": return "partial(orchard, species unspecified)"
    return ""
# all points within 100 m of each polygon (not only nearest-polygon)
rows = []
for pid, poly in H.geometry.items():
    d = G.distance(poly); q = G[d <= 100].copy(); q["d"] = d[d <= 100]
    for r in q.itertuples():
        rows.append({"polygon_id": pid, "poly_class": int(H.at[pid, "class_code"]), "fid": r.fid, "dist_m": round(float(r.d), 1), "field_class": r.field_class,
                     "field_mixed": bool(r.field_mixed), "field_text": r.field_text, "gps_time": r.date, "gps_acc": r.acc, "source_file": r.src_file, "lon": r.plon, "lat": r.plat,
                     "mismatch": mismatch(int(H.at[pid, "class_code"]), r.field_class)})
M = pd.DataFrame(rows).sort_values(["polygon_id", "dist_m"])
M.to_csv(KIT / "holdout_field_point_matches.csv", index=False, encoding="utf-8-sig")

# ---------------- workbook: load the (possibly user-edited) source, fill only empty field_verified, append to notes
wb = load_workbook(SRC); ws = wb["holdout_temporal"]
hdr = [c.value for c in ws[1]]; ci = {h: i + 1 for i, h in enumerate(hdr)}
for newc in ("auto_field_match", "auto_class_check"):
    if newc not in ci: ws.cell(1, ws.max_column + 1, newc); ci[newc] = ws.max_column
Y = PatternFill("solid", fgColor="C6EFCE"); Yq = PatternFill("solid", fgColor="FFEB9C"); R = PatternFill("solid", fgColor="FFC7CE")
stat = {"Y": 0, "Y?": 0, "near_30_100m_only": 0, "none": 0, "mismatch_polygons": 0, "kept_user_field_verified": 0}
mm_list = []
for rr in range(2, ws.max_row + 1):
    pid = ws.cell(rr, ci["polygon_id"]).value
    if pid is None: continue
    m = M[M.polygon_id == pid]
    if m.empty: stat["none"] += 1; continue
    dmin = m.dist_m.min(); tag = "Y" if dmin == 0 else "Y?" if dmin <= 30 else None
    lvl = "inside" if dmin == 0 else "<=30m" if dmin <= 30 else "30-100m"
    parts = []
    for r in m.itertuples():
        s = f"{r.fid} [{r.field_class}{',mixed' if r.field_mixed else ''}] '{r.field_text}' time={r.gps_time or 'n/a'} d={r.dist_m:g}m"
        if r.gps_acc: s += f" acc={r.gps_acc}"
        if r.mismatch == "MISMATCH": s = "⚠" + s
        parts.append(s)
    note = f"[自动预填 field GPS, {len(m)}点≤100m, 最近{dmin:g}m] " + "; ".join(parts)
    cur = ws.cell(rr, ci["notes"]).value
    if cur and "[自动预填" in str(cur): cur = str(cur).split("[自动预填")[0].rstrip(" |")
    ws.cell(rr, ci["notes"], (f"{cur} | {note}" if cur else note))
    fv = ws.cell(rr, ci["field_verified"])
    if tag:
        if fv.value in (None, ""): fv.value = tag; fv.fill = Y if tag == "Y" else Yq
        else: stat["kept_user_field_verified"] += 1
        stat[tag] += 1
    else: stat["near_30_100m_only"] += 1
    ws.cell(rr, ci["auto_field_match"], lvl)
    mm = m[(m.mismatch == "MISMATCH") & (m.dist_m <= 30)]; mm_far = m[(m.mismatch == "MISMATCH") & (m.dist_m > 30)]
    chk = ("MISMATCH(≤30m): " + ",".join(f"{r.fid}={r.field_class}" for r in mm.itertuples())) if len(mm) else ""
    if len(mm_far): chk += (" | " if chk else "") + "mismatch 30-100m: " + ",".join(f"{r.fid}={r.field_class}" for r in mm_far.itertuples())
    if not chk and (m.mismatch.str.startswith("partial")).any(): chk = "orchard (species unspecified) only"
    if not chk and (m.field_class == "unknown").all(): chk = "field class unknown"
    if not chk: chk = "consistent"
    ws.cell(rr, ci["auto_class_check"], chk)
    if len(mm): stat["mismatch_polygons"] += 1; ws.cell(rr, ci["auto_class_check"]).fill = R; mm_list.append(pid)
if "填写规则" in wb.sheetnames:
    wr = wb["填写规则"]; wr.append(["field_verified 自动预填：Y=有野外GPS点落在面内（建议值）；Y?=最近点在30 m内（请核对）；30–100 m 仅写入notes；其余留空。程序不会自动填 N，是否实地核查由你决定。"])
    wr.append(["auto_field_match / auto_class_check：自动匹配级别与类别一致性（⚠/MISMATCH = GPS点记录的类别与面的class不一致，请核对）。点的类别是根据GPS记录的名称/备注文字判断的。"])
wb.save(OUTX)

# ---------------- KML of matched points (<=100 m), coloured by match level
lvl = M.groupby("fid").dist_m.min()
S = [style("in", "ff00ff00"), style("n30", "ff00ffff"), style("n100", "ffff9900"), style("mm", "ff0000ff"),
     style("c1", "ff0000ff", "000000ff", 1.5), style("c2", "ff00c800", "", 1.5), style("c0", "ffffffff", "", 1.5)]
pts = M.drop_duplicates("fid").set_index("fid")
fold = []
for name, lo, hi, sty in (("面内 inside → Y", -1, 0, "in"), ("≤30 m → Y?", 0, 30, "n30"), ("30–100 m（仅备注）", 30, 100, "n100")):
    items = []
    for fid in lvl[(lvl > lo) & (lvl <= hi)].index:
        r = pts.loc[fid]; ms = M[(M.fid == fid)]
        bad = (ms.mismatch == "MISMATCH").any()
        items.append({"name": f"{fid} {r.field_class}", "point": (r.lon, r.lat), "style": "mm" if bad else sty,
                      "desc": f"text={r.field_text}\ntime={r.gps_time}\nacc={r.gps_acc}\nfile={r.source_file}\n" +
                              "\n".join(f"{q.polygon_id} (c{q.poly_class}) d={q.dist_m:g} m {q.mismatch}" for q in ms.itertuples())})
    fold.append((name, items))
hp = H.to_crs(4326); used = set(M.polygon_id)
fold.append(("匹配到的挑战集面", [{"name": f"{pid} | c{int(hp.at[pid, 'class_code'])}", "geom": hp.at[pid, "geometry"], "style": f"c{int(hp.at[pid, 'class_code'])}"} for pid in sorted(used)]))
write_kml(OUTK, "挑战集面 × 野外GPS点匹配（≤100 m）", S, fold, description="红色点 = 与面类别不一致；绿色 = 面内；黄色 = ≤30 m；蓝色 = 30–100 m")
summ = {"source_xlsx": str(SRC), "field_points_total_raw": int(n_raw), "field_points_after_exact_dedup": int(len(F)),
        "field_points_by_class": F.field_class.value_counts().to_dict(), "points_within_100m_of_any_holdout_polygon": int(M.fid.nunique()),
        "polygons": stat, "mismatch_polygon_ids_le30m": mm_list,
        "polygons_Y_by_class": M[M.dist_m == 0].drop_duplicates("polygon_id").poly_class.value_counts().to_dict()}
json.dump(summ, open(KIT / "prefill_summary.json", "w"), ensure_ascii=False, indent=1, default=int); print(json.dumps(summ, ensure_ascii=False, indent=1, default=int))
