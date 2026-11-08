# -*- coding: utf-8 -*-
"""v4 kit finalisation (stage b): select mountain (MGR/MFR/MSH/TGR) and apple (APL) candidates, append to YL-TRN-0001..0174,
write colour-coded KML (+ reference KML of existing training polygons), CSV and summary.
Rules (same as 35b): outside Gongliu (official GS2024 boundary), >=2 km from every Gongliu holdout polygon (v3),
>=300 m from existing training polygons, >=800 m from every other selected point (incl. 0001-0174),
negatives: not AOMC-2021 apple (3x3 window). Mountain: ~70% drawn from B3_poly_2021_v2 prob >= 0.5.
Apple: NOT selected by B3 probability (to avoid biasing positives toward what the model already finds);
field-survey points first, then AOMC points flagged apple in all 3 AOMC years (point-sampled on each year's own grid),
stratified over elevation bins and counties."""
import sys, json, re
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely.ops import unary_union
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision/figures_v3_scripts"))
import official_boundary as ob
KIT = ROOT / "outputs/revision/v4_experiments/training_expansion_kit"
E = pd.read_csv(KIT / "_v1_0001-0174" / "YL_TRN_candidates.csv")
D = pd.read_csv(KIT / "_candidates_mountain_apple_scored.csv")
elev_e = D[D.pool == "existing"].set_index("id").elev
E["elev_m"] = E.id.map(elev_e).round(0)
assert E.elev_m.notna().all()
C = D[D.pool != "existing"].dropna(subset=["b3_prob"]).reset_index(drop=True)
P = gpd.read_file(ROOT / "outputs/revision/polygon_samples/v3/polygons_v3_all.geojson").to_crs(32644)
ho = unary_union(P[P.split == "val_holdout"].geometry.values); trp = unary_union(P[P.split == "train"].geometry.values)
gl4326 = ob.gongliu().geometry.iloc[0]; gl = ob.gongliu().to_crs(32644).geometry.iloc[0]
g = gpd.GeoDataFrame(C, geometry=gpd.points_from_xy(C.lon, C.lat), crs=4326).to_crs(32644)
g["d_holdout_km"] = g.distance(ho) / 1000; g["d_train_m"] = g.distance(trp); g["in_gongliu"] = g.within(gl)
# AOMC flags per year on each raster's own grid
for yr in (2019, 2020, 2021):
    with rasterio.open(ROOT / f"data/aomc/AOMC_30_Xinjiang_{yr}.tif") as src:
        xy = gpd.GeoSeries(gpd.points_from_xy(C.lon, C.lat), crs=4326).to_crs(src.crs)
        pt, w3 = [], []
        for p in xy:
            r, c = src.index(p.x, p.y)
            w = src.read(1, window=((r - 1, r + 2), (c - 1, c + 2)), boundless=True, fill_value=0)
            pt.append(int(w[1, 1] == 1)); w3.append(int((w == 1).any()))
        g[f"aomc{yr}"] = pt
        if yr == 2021: g["aomc_apple_3x3"] = w3
g["aomc_years"] = g.aomc2019 + g.aomc2020 + g.aomc2021
elig = g[(~g.in_gongliu) & (g.d_holdout_km >= 2) & (g.d_train_m >= 300)].copy()
Eg = gpd.GeoDataFrame(E, geometry=gpd.points_from_xy(E.lon, E.lat), crs=4326).to_crs(32644)
used = list(Eg.geometry.values)
def far(pt): return min(pt.distance(u) for u in used) >= 800
def rr_order(df, key="county"):
    groups = {k: q for k, q in df.groupby(key)}; order = []
    while any(len(q) for q in groups.values()):
        for k in list(groups):
            if len(groups[k]): order.append(groups[k].index[0]); groups[k] = groups[k].iloc[1:]
    return order
MT = [("MGR", "山地草地/草甸（≥1500 m）", "mountain_grass", lambda x: (x.pool == "mountain") & (x.wc == 30) & (x.elev >= 1500), 18),
      ("MFR", "山地针叶林/林地（≥1500 m）", "mountain_forest", lambda x: (x.pool == "mountain") & (x.wc == 10) & (x.elev >= 1500), 12),
      ("MSH", "山地灌丛（≥1500 m）", "mountain_shrub", lambda x: (x.pool == "mountain") & (x.wc == 20) & (x.elev >= 1500), 8),
      ("TGR", "过渡带草地/灌丛（1217–1500 m）", "transition_grass", lambda x: (x.pool == "mountain") & x.wc.isin([20, 30]) & (x.elev >= 1217) & (x.elev < 1500), 12)]
sel = []
for code, zh, sub, f, n in MT:
    pool = elig[f(elig) & (elig.aomc_apple_3x3 == 0)]
    n_hi = int(round(0.7 * n)); picked = []
    hi = pool[pool.b3_prob >= 0.5].sort_values("b3_prob", ascending=False); lo = pool[pool.b3_prob < 0.5].sample(frac=1, random_state=42)
    for part, k in ((hi, n_hi), (lo, n)):
        for i in rr_order(part):
            if len(picked) >= k: break
            if far(elig.geometry[i]): picked.append(i); used.append(elig.geometry[i])
    # top-up with hi if lo pool exhausted
    for i in rr_order(hi):
        if len(picked) >= n: break
        if i not in picked and far(elig.geometry[i]): picked.append(i); used.append(elig.geometry[i])
    s = elig.loc[picked].copy(); s["stratum"] = code; s["stratum_zh"] = zh; s["subtype_hint"] = sub; sel.append(s)
    print(code, "pool", len(pool), "hi", len(hi), "->", len(s), "hi", int((s.b3_prob >= .5).sum()))
# apple
N_APL = 25
ap = elig[elig.pool.isin(["field_apple", "aomc_apple"])].copy()
ap["ebin"] = pd.cut(ap.elev, [0, 700, 850, 1000, 1150, 4000], labels=["<700", "700-850", "850-1000", "1000-1150", ">=1150"])
picked = []
fld = ap[ap.pool == "field_apple"].sample(frac=1, random_state=1)
for i in rr_order(fld, "ebin"):
    if len(picked) >= 8: break
    if far(ap.geometry[i]): picked.append(i); used.append(ap.geometry[i])
aom = ap[(ap.pool == "aomc_apple") & (ap.aomc_years == 3)].sample(frac=1, random_state=42)
aom["ck"] = aom.ebin.astype(str) + "|" + aom.county
for i in rr_order(aom, "ck"):
    if len(picked) >= N_APL: break
    if far(ap.geometry[i]): picked.append(i); used.append(ap.geometry[i])
s = ap.loc[picked].copy(); s["stratum"] = "APL"; s["stratum_zh"] = "苹果园候选（野外点/AOMC三年一致，需目视确认）"; s["subtype_hint"] = "apple"
sel.append(s.drop(columns=["ebin"]))
print("APL", len(s), s.pool.value_counts().to_dict(), s.ebin.value_counts().to_dict())
N = pd.concat(sel).reset_index(drop=True)
N["id"] = [f"YL-TRN-{175 + i:04d}" for i in range(len(N))]
WCN = {10: "树木", 20: "灌丛", 30: "草地", 40: "农田", 50: "建设用地", 60: "裸地/稀疏植被", 70: "冰雪", 80: "水体", 90: "湿地", 95: "红树林", 100: "苔藓地衣"}
N["wc"] = N.wc_s.astype(int); N["wc_name"] = N.wc.map(WCN); N["elev_m"] = N.elev.round(0)
N["source"] = N.pool.map({"mountain": "WorldCover2021×SRTM", "field_apple": "野外GPS点", "aomc_apple": "AOMC2019–2021均为苹果"})
N["note"] = N.get("field_note", pd.Series([""] * len(N))).fillna("")
SUBH = {"OTH": "other_orchard", "CRP": "cropland", "FOR": "forest(shelterbelt/plantation)", "GRS": "grass/shrub", "HNG": "built/bare"}
E["subtype_hint"] = E.stratum.map(SUBH); E["source"] = "WorldCover2021（<1500 m）"; E["aomc_years"] = np.nan; E["note"] = ""
cols = ["id", "stratum", "stratum_zh", "subtype_hint", "county", "lon", "lat", "elev_m", "wc", "wc_name", "b3_prob", "aomc_apple_3x3", "aomc_years",
        "source", "d_holdout_km", "d_train_m", "note"]
ALL = pd.concat([E[cols], N[cols]], ignore_index=True)
ALL = ALL.round({"b3_prob": 3, "d_holdout_km": 2, "d_train_m": 0, "lon": 6, "lat": 6})
# final checks
Ag = gpd.GeoDataFrame(ALL, geometry=gpd.points_from_xy(ALL.lon, ALL.lat), crs=4326)
assert not Ag.within(gl4326).any()
Au = Ag.to_crs(32644)
dmin = min(Au.geometry.iloc[i].distance(Au.geometry.iloc[j]) for i in range(len(Au)) for j in range(i + 1, len(Au)))
assert ALL.d_holdout_km.min() >= 2 and dmin >= 800 and ALL.id.is_unique
ALL.to_csv(KIT / "YL_TRN_candidates.csv", index=False, encoding="utf-8-sig")
# ---------- KML
ORDER = [("APL", "ff0000ff"), ("OTH", "ff0080ff"), ("CRP", "ff00ffff"), ("FOR", "ff00a000"), ("GRS", "ff80ff80"), ("HNG", "ffa0a0a0"),
         ("MGR", "ffffff00"), ("MFR", "ff806000"), ("MSH", "ffff00ff"), ("TGR", "ffffb080")]
ZH = dict(zip(ALL.stratum, ALL.stratum_zh))
CLS = {"APL": "预期 c1（苹果园，须目视确认）", "OTH": "预期 c2（其他果园）或 c0", "CRP": "预期 c0 cropland", "FOR": "预期 c0 forest",
       "GRS": "预期 c0 grass/shrub", "HNG": "预期 c0 built/bare", "MGR": "预期 c0 mountain_grass", "MFR": "预期 c0 forest（mountain）",
       "MSH": "预期 c0 shrub（mountain）", "TGR": "预期 c0 mountain_grass/shrub（过渡带）"}
k = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2">', "<Document>",
     f"<name>YL-TRN 训练样本扩充候选点（巩留县外，{len(ALL)} 点）</name>", "<open>1</open>",
     "<description>候选点仅为勾绘位置提示，不是标签。请在点附近勾绘均质地块多边形，命名为“YL-TRN-xxxx | c0/c1/c2”。详见 README_勾绘说明_YL-TRN.md。巩留县（红线）内禁止勾绘。</description>"]
for code, col in ORDER:
    for sfx, sc, ls in (("n", 1.0, 0.8), ("h", 1.3, 1.1)):
        k.append(f'<Style id="{code}_{sfx}"><IconStyle><color>{col}</color><scale>{sc}</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>'
                 f'<LabelStyle><color>{col}</color><scale>{ls}</scale></LabelStyle><BalloonStyle><text>$[name]<br/>$[description]</text></BalloonStyle></Style>')
    k.append(f'<StyleMap id="{code}"><Pair><key>normal</key><styleUrl>#{code}_n</styleUrl></Pair><Pair><key>highlight</key><styleUrl>#{code}_h</styleUrl></Pair></StyleMap>')
k.append('<Style id="gl"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle><PolyStyle><color>300000ff</color><fill>1</fill><outline>1</outline></PolyStyle></Style>')
for code, col in ORDER:
    q = ALL[ALL.stratum == code]
    k.append(f"<Folder><name>{code} {escape(ZH[code])}（{len(q)}）</name><open>0</open>")
    for r in q.itertuples():
        desc = (f"ID={r.id}<br/>stratum={r.stratum}（{escape(r.stratum_zh)}）<br/>{CLS[code]}<br/>县={r.county}<br/>高程={r.elev_m:.0f} m（SRTM）<br/>"
                f"B3概率={r.b3_prob:.2f}（B3_poly_2021_v2，AlphaEarth 2021）<br/>WorldCover2021={r.wc_name}<br/>来源={escape(str(r.source))}<br/>"
                f"距巩留留出面={r.d_holdout_km:.1f} km" + (f"<br/>野外备注={escape(str(r.note))}" if str(r.note) not in ("", "nan") else "") +
                f"<br/>返回命名示例：{r.id} | c{'1' if code == 'APL' else ('2' if code == 'OTH' else '0')}")
        k.append(f"<Placemark><name>{r.id}</name><description><![CDATA[{desc}]]></description><styleUrl>#{code}</styleUrl>"
                 f"<Point><coordinates>{r.lon:.6f},{r.lat:.6f},0</coordinates></Point></Placemark>")
    k.append("</Folder>")
# Gongliu no-go boundary
geoms = [gl4326] if gl4326.geom_type == "Polygon" else list(gl4326.geoms)
k.append("<Folder><name>禁区：巩留县界（GS（2024）0650号），县内不得勾绘</name>")
for gg in geoms:
    cs = " ".join(f"{x:.5f},{y:.5f},0" for x, y in gg.simplify(0.0005).exterior.coords)
    k.append(f"<Placemark><name>巩留县（留出区，禁止勾绘）</name><styleUrl>#gl</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{cs}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
k.append("</Folder></Document></kml>")
(KIT / "YL_TRN_training_expansion_candidates.kml").write_text("\n".join(k), encoding="utf-8")
# ---------- reference KML: existing training polygons (do not re-digitize)
T = P[P.split == "train"].to_crs(4326)
namecol = "name" if "name" in T.columns else T.columns[0]
ccol = "class_code" if "class_code" in T.columns else None
r = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>', f"<name>现有训练面（{len(T)}，勿重复勾绘）</name>",
     '<Style id="tr"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><color>4000ffff</color></PolyStyle><LabelStyle><scale>0.7</scale></LabelStyle></Style>']
for t in T.itertuples():
    gg = t.geometry if t.geometry.geom_type == "Polygon" else max(t.geometry.geoms, key=lambda z: z.area)
    cs = " ".join(f"{x:.7f},{y:.7f},0" for x, y in gg.exterior.coords)
    nm = escape(str(getattr(t, namecol))); cc = getattr(t, ccol) if ccol else ""
    r.append(f"<Placemark><name>{nm} | c{cc}</name><styleUrl>#tr</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{cs}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
r.append("</Document></kml>")
(KIT / "EXISTING_training_polygons_reference.kml").write_text("\n".join(r), encoding="utf-8")
summ = {"n_total": int(len(ALL)), "id_range": [ALL.id.iloc[0], ALL.id.iloc[-1]], "new_ids": [N.id.iloc[0], N.id.iloc[-1]],
        "by_stratum": ALL.stratum.value_counts().reindex([c for c, _ in ORDER]).to_dict(),
        "hi_prob_ge0.5_by_stratum": ALL[ALL.b3_prob >= 0.5].stratum.value_counts().to_dict(),
        "by_county": ALL.county.value_counts().to_dict(),
        "elev_by_stratum_median_min_max": {s: [float(q.elev_m.median()), float(q.elev_m.min()), float(q.elev_m.max())] for s, q in ALL.groupby("stratum")},
        "apple_sources": N[N.stratum == "APL"].source.value_counts().to_dict(),
        "min_d_holdout_km": float(ALL.d_holdout_km.min()), "any_in_gongliu": False, "min_d_train_m": float(ALL.d_train_m.min()),
        "min_spacing_m": float(dmin), "n_train_polygons_reference": int(len(T))}
json.dump(summ, open(KIT / "kit_summary.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(summ, ensure_ascii=False, indent=1))
