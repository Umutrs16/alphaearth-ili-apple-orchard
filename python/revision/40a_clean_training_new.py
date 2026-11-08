# -*- coding: utf-8 -*-
"""v4 retrain step 0: trivial auto-fixes of the user's returned training_new.kml and a pre-QC inventory.
Trivial fixes only (logged): whitespace in names; unique IDs for duplicated / malformed IDs (renamed to YL-TRN-1001+,
per the kit README rule for extra polygons); descriptions '影像日期：2021.6.2' (and typo variants 影响日期, ':', ';', '::')
normalised to 'img_date=2021-06-02; raw_desc=...'. Class codes and geometries are NOT changed.
The user numbered polygons sequentially (not by candidate ID), so each polygon is linked to its candidate spatially
(nearest candidate point; distance reported) for the stratum inventory.
Outputs: training_new_clean.kml, training_new_clean_sens_imgyear2020_2022.kml (sensitivity subset), pre_qc_inventory.csv, pre_qc_summary.json"""
import re, sys, json, importlib.util
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd, geopandas as gpd
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; sys.path.insert(0, str(REV / "figures_v3_scripts"))
import official_boundary as ob
spec = importlib.util.spec_from_file_location("v3m", REV / "22_b3_poly_holdout_v3.py"); v3m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3m)
KIT = ROOT / "outputs/revision/v4_experiments/training_expansion_kit"; OUT = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; OUT.mkdir(parents=True, exist_ok=True)
raw = v3m.read_kml_polys(KIT / "training_new.kml")
cand = pd.read_csv(KIT / "YL_TRN_candidates.csv")
cg = gpd.GeoDataFrame(cand, geometry=gpd.points_from_xy(cand.lon, cand.lat), crs=4326).to_crs(32644)
cty = ob.county()
rows, seen, nxt, fixes = [], set(), 1001, []
for k, r in enumerate(raw):
    nm = re.sub(r"\s+", " ", r["raw_name"]).strip()
    m = re.match(r"^(YL-TRN-)(\d+)\s*[|｜]\s*[cC]\s*(\d)$", nm)
    cls = int(m.group(3)) if m else None
    pid = f"YL-TRN-{m.group(2)}" if m and len(m.group(2)) == 4 else None
    if pid is None or pid in seen:
        new = f"YL-TRN-{nxt:04d}"; nxt += 1
        fixes.append({"raw_name": r["raw_name"], "new_id": new, "why": "malformed_id" if pid is None else "duplicate_id"}); pid = new
    seen.add(pid)
    dm = re.search(r"(\d{3,4})\s*[.\-/年]\s*(\d{1,2})\s*[.\-/月]\s*(\d{1,2})", r["desc"])
    if dm and len(dm.group(1)) == 4:
        y, mo, da = map(int, dm.groups()); img = f"{y:04d}-{mo:02d}-{da:02d}"; yr = y
    else:
        img, yr = "unknown", None
    desc_ok = bool(re.match(r"^\s*影像日期[：:]\s*\d{4}\.\d{1,2}\.\d{1,2}\s*$", r["desc"]))
    g, note = v3m.make_geom(r["coords"])
    d = {"k": k, "raw_name": r["raw_name"], "poly_id": pid, "class_code": cls, "raw_desc": r["desc"], "img_date": img, "img_year": yr,
         "desc_exact_format": desc_ok, "geom_note": note, "n_vertices": len(r["coords"])}
    if g is not None:
        gu = v3m.utm(g); dd = cg.distance(gu); j = dd.idxmin()
        d.update(area_ha=gu.area / 1e4, cand_nearest=cand.id[j], cand_stratum=cand.stratum[j], cand_dist_m=float(dd.min()),
                 cand_b3_prob=cand.b3_prob[j], cand_elev_m=cand.elev_m[j], cen_lon=g.centroid.x, cen_lat=g.centroid.y)
        hit = cty[cty.contains(g.centroid)]; d["county"] = hit.name.iloc[0] if len(hit) else None
    rows.append(d)
inv = pd.DataFrame(rows)
inv.to_csv(OUT / "pre_qc_inventory.csv", index=False, encoding="utf-8-sig")

def write(df, path):
    kml = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>training_new (cleaned)</name><Folder><name>training_new</name>']
    for r in df.itertuples():
        raw_r = raw[r.k]
        cs = " ".join(f"{x:.9f},{y:.9f},0" for x, y in raw_r["coords"])
        desc = f"img_date={r.img_date}; raw_desc={escape(str(r.raw_desc)).replace(';', ',')}; cand_nearest={getattr(r, 'cand_nearest', '')}; cand_stratum={getattr(r, 'cand_stratum', '')}"
        nm = f"{r.poly_id} | c{r.class_code}" if r.class_code is not None else escape(r.raw_name)
        kml.append(f"<Placemark><name>{nm}</name><description>{desc}</description><Polygon><outerBoundaryIs><LinearRing><coordinates>{cs}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    kml.append("</Folder></Document></kml>"); Path(path).write_text("\n".join(kml), encoding="utf-8")
write(inv, OUT / "training_new_clean.kml")
sens = inv[inv.img_year.between(2020, 2022)]
write(sens, OUT / "training_new_clean_sens_imgyear2020_2022.kml")
summ = {"n_placemarks": len(inv), "id_fixes": fixes, "n_desc_not_exact_format": int((~inv.desc_exact_format).sum()),
        "desc_variants_fixed": inv[~inv.desc_exact_format].raw_desc.tolist(), "img_date_unknown": inv[inv.img_date == "unknown"][["poly_id", "raw_desc"]].to_dict("records"),
        "img_year_counts": inv.img_year.value_counts(dropna=False).sort_index().to_dict(),
        "n_sens_subset_2020_2022": int(len(sens))}
json.dump(summ, open(OUT / "pre_qc_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
