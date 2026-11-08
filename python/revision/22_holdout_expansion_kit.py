#!/usr/bin/env python3
"""Build Gongliu val_holdout expansion digitizing kit (candidate POINTS only).

Does NOT invent ground-truth polygons. Outputs KML/KMZ candidate points +
existing holdout reference faces for Google Earth digitizing.
"""
from __future__ import annotations

import html
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point, mapping, box
from shapely.ops import unary_union
from sklearn.cluster import KMeans

BASE = Path("/workspace/apple_orchard_project")
OUT = BASE / "outputs/revision/polygon_samples/holdout_expansion"
OUT.mkdir(parents=True, exist_ok=True)

POLY_FULL = BASE / "outputs/revision/polygon_samples/user_apple_polygons_full_v2.geojson"
MANIFEST = BASE / "outputs/revision/polygon_samples/split_manifest_v2.csv"
GONG_ROI = BASE / "data/roi/Gongliu.geojson"
AOMC21 = BASE / "data/aomc/AOMC_30_Xinjiang_2021.tif"
FIELD_HITS = BASE / "data/field_survey/field_survey_hits_gongliu_dedup.csv"
FIELD_USABLE = BASE / "data/field_survey/usable_candidates/gongliu_field_apple_usable.csv"
FIELD_ORCH = BASE / "data/field_survey/usable_candidates/gongliu_field_orchard_unspecified_in_county.csv"
SURVEY = BASE / "outputs/revision/field_campaign/Ili_apple_orchard_survey_candidates.csv"
FILLED = BASE / "data/interpretation_kit/Gongliu_independent_val_判读表_filled.xlsx"
FOCUS_CSV = BASE / "outputs/revision/maps_v2/gongliu_focus_grid_pred_2021_100m.csv"

BUFFER_M = 80.0  # exclude candidates within ~80 m of existing polys
MIN_SEP_M = 120.0  # min separation among candidates of same class
SEED = 20260925
NEXT_POLY_ID = 64  # YL-POLY-0064

CLASS_NAME = {0: "non_orchard", 1: "apple", 2: "other_orchard"}
CLASS_NAME_ZH = {0: "非果园", 1: "苹果园", 2: "其他果园"}


def ll_to_utm(lon, lat):
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    return tf.transform(np.asarray(lon, float), np.asarray(lat, float))


def dist_to_geoms_m(lon, lat, geoms_utm):
    """Min distance (m) from points to a list of UTM shapely geoms."""
    x, y = ll_to_utm(lon, lat)
    pts = [Point(xx, yy) for xx, yy in zip(x, y)]
    if not geoms_utm:
        return np.full(len(pts), np.inf)
    union = unary_union(geoms_utm)
    return np.array([p.distance(union) for p in pts], dtype=float)


def pairwise_min_sep_filter(df, min_sep_m, lon_col="lon", lat_col="lat", priority_col=None):
    """Greedy keep points with >= min_sep_m separation (higher priority first)."""
    if df.empty:
        return df
    d = df.copy()
    if priority_col and priority_col in d.columns:
        d = d.sort_values(priority_col, ascending=False)
    else:
        d = d.reset_index(drop=True)
    xs, ys = ll_to_utm(d[lon_col].values, d[lat_col].values)
    keep = []
    kept_xy = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if kept_xy:
            dmin = min(math.hypot(x - kx, y - ky) for kx, ky in kept_xy)
            if dmin < min_sep_m:
                continue
        keep.append(i)
        kept_xy.append((x, y))
    return d.iloc[keep].reset_index(drop=True)


def kml_header(name, desc=""):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{html.escape(name)}</name>
  <description>{html.escape(desc)}</description>
"""


def kml_footer():
    return "</Document>\n</kml>\n"


def style_poly(sid, line_hex, fill_hex, fill_alpha="80"):
    # KML colors are aabbggrr
    return f"""  <Style id="{sid}">
    <LineStyle><color>ff{line_hex}</color><width>2.2</width></LineStyle>
    <PolyStyle><color>{fill_alpha}{fill_hex}</color><fill>1</fill><outline>1</outline></PolyStyle>
  </Style>
"""


def style_point(sid, icon_href, color="ff00ff00", scale=1.05):
    return f"""  <Style id="{sid}">
    <IconStyle>
      <color>{color}</color>
      <scale>{scale}</scale>
      <Icon><href>{icon_href}</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0.7</scale></LabelStyle>
  </Style>
"""


def ring_coords(geom):
    if geom.geom_type == "Polygon":
        rings = [geom.exterior]
        rings.extend(list(geom.interiors))
    elif geom.geom_type == "MultiPolygon":
        rings = []
        for g in geom.geoms:
            rings.append(g.exterior)
            rings.extend(list(g.interiors))
    else:
        return ""
    parts = []
    for r in rings:
        coords = " ".join(f"{x:.8f},{y:.8f},0" for x, y in r.coords)
        parts.append(f"<LinearRing><coordinates>{coords}</coordinates></LinearRing>")
    return "".join(parts)


def write_existing_holdout_kml(holdout: gpd.GeoDataFrame, path: Path):
    lines = [kml_header("EXISTING Gongliu val_holdout (v2)", "当前 18 个 holdout 面，仅供参考，请勿重复勾绘")]
    lines.append(style_poly("applePoly", "00aa00", "00aa00", "70"))
    lines.append(style_poly("nonPoly", "0000ff", "0000ff", "70"))
    lines.append(style_poly("otherPoly", "00aaff", "00aaff", "70"))
    for cls, folder, style in [
        (1, "苹果园 class=1", "#applePoly"),
        (0, "非果园 class=0", "#nonPoly"),
        (2, "其他果园 class=2", "#otherPoly"),
    ]:
        sub = holdout[holdout.class_code == cls]
        if sub.empty:
            continue
        lines.append(f"  <Folder><name>{folder} (n={len(sub)})</name>\n")
        for _, r in sub.iterrows():
            nm = str(r["name"])
            area_ha = float(r.get("area_ha") or 0)
            desc = (
                f"poly_id={nm}; class_code={int(r.class_code)}; "
                f"class_name={CLASS_NAME.get(int(r.class_code),'?')}; "
                f"county=巩留县; split=val_holdout; area_ha={area_ha:.3f}; "
                f"EXISTING — do not re-digitize"
            )
            rings = ring_coords(r.geometry)
            lines.append(
                f"    <Placemark>\n"
                f"      <name>{html.escape(nm)} | c{int(r.class_code)} | {area_ha:.2f}ha</name>\n"
                f"      <description>{html.escape(desc)}</description>\n"
                f"      <styleUrl>{style}</styleUrl>\n"
                f"      <Polygon><outerBoundaryIs>{rings.split('</LinearRing>')[0]}</LinearRing></outerBoundaryIs>"
            )
            # handle holes simply: only exterior for GE clarity
            # Rebuild properly
            lines[-1] = (
                f"    <Placemark>\n"
                f"      <name>{html.escape(nm)} | c{int(r.class_code)} | {area_ha:.2f}ha</name>\n"
                f"      <description>{html.escape(desc)}</description>\n"
                f"      <styleUrl>{style}</styleUrl>\n"
                f"      <Polygon>\n"
            )
            geom = r.geometry
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            # KML MultiGeometry if needed
            if len(polys) > 1:
                lines[-1] = lines[-1].replace("<Polygon>\n", "<MultiGeometry>\n")
            for pi, poly in enumerate(polys):
                if len(polys) > 1:
                    lines.append("      <Polygon>\n")
                ext = " ".join(f"{x:.8f},{y:.8f},0" for x, y in poly.exterior.coords)
                lines.append(
                    f"        <outerBoundaryIs><LinearRing><coordinates>{ext}</coordinates>"
                    f"</LinearRing></outerBoundaryIs>\n"
                )
                for interior in poly.interiors:
                    inn = " ".join(f"{x:.8f},{y:.8f},0" for x, y in interior.coords)
                    lines.append(
                        f"        <innerBoundaryIs><LinearRing><coordinates>{inn}</coordinates>"
                        f"</LinearRing></innerBoundaryIs>\n"
                    )
                if len(polys) > 1:
                    lines.append("      </Polygon>\n")
            if len(polys) > 1:
                lines.append("      </MultiGeometry>\n")
            else:
                lines.append("      </Polygon>\n")
            lines.append("    </Placemark>\n")
        lines.append("  </Folder>\n")
    lines.append(kml_footer())
    path.write_text("".join(lines), encoding="utf-8")


def write_candidates_kml(df: pd.DataFrame, path: Path, title: str, style_id: str, icon: str, color: str, desc: str):
    lines = [kml_header(title, desc)]
    lines.append(style_point(style_id, icon, color=color))
    # folder by source
    for src, sub in df.groupby("source", sort=False):
        lines.append(f"  <Folder><name>{html.escape(str(src))} (n={len(sub)})</name>\n")
        for _, r in sub.iterrows():
            cc = int(r.suggested_class_code)
            name = f"{r.cand_id} | c{cc} | {CLASS_NAME_ZH[cc]}"
            d = (
                f"SUGGEST digitize Polygon around this point. "
                f"suggested: poly_id=YL-POLY-0xxx; class_code={cc}; "
                f"class_name={CLASS_NAME[cc]}; source={r.source}; year_ref=2021; "
                f"county=巩留县; split=val_holdout; "
                f"notes={r.reason}; dist_m_to_nearest_existing_poly={r.dist_m_to_nearest_existing_poly:.0f}"
            )
            lines.append(
                f"    <Placemark>\n"
                f"      <name>{html.escape(name)}</name>\n"
                f"      <description>{html.escape(d)}</description>\n"
                f"      <styleUrl>#{style_id}</styleUrl>\n"
                f"      <Point><coordinates>{r.lon:.8f},{r.lat:.8f},0</coordinates></Point>\n"
                f"    </Placemark>\n"
            )
        lines.append("  </Folder>\n")
    lines.append(kml_footer())
    path.write_text("".join(lines), encoding="utf-8")


def sample_aomc_points(gong_gdf, existing_utm, n=40, min_sep=400.0, excl_buf=BUFFER_M):
    """Sample dispersed AOMC=1 pixels inside Gongliu agricultural focus envelope."""
    focus = pd.read_csv(FOCUS_CSV)
    # agricultural envelope from focus + small pad
    minx, maxx = focus.lon.min() - 0.05, focus.lon.max() + 0.05
    miny, maxy = focus.lat.min() - 0.05, focus.lat.max() + 0.05
    # also include west field orchard zone
    minx = min(minx, 81.90)
    maxy = max(maxy, 43.60)
    miny = max(miny, 43.05)
    env = box(minx, miny, maxx, maxy)
    roi = gong_gdf.geometry.iloc[0].intersection(env)
    if roi.is_empty:
        roi = gong_gdf.geometry.iloc[0]

    with rasterio.open(AOMC21) as src:
        g2 = gong_gdf.to_crs(src.crs)
        # mask with intersected ROI in WGS then project
        roi_gdf = gpd.GeoDataFrame(geometry=[roi], crs=4326).to_crs(src.crs)
        out, transform = rio_mask(src, [mapping(roi_gdf.geometry.iloc[0])], crop=True, filled=True)
        arr = out[0]
        nodata = src.nodata
        ys, xs = np.where((arr == 1) & ((arr != nodata) if nodata is not None else True))
        if len(xs) == 0:
            return pd.DataFrame(columns=["lon", "lat"])
        # random subsample pixels then thin
        rng = np.random.default_rng(SEED)
        if len(xs) > 8000:
            idx = rng.choice(len(xs), size=8000, replace=False)
            xs, ys = xs[idx], ys[idx]
        # pixel centers in AOMC CRS
        ax = transform.c + (xs + 0.5) * transform.a
        ay = transform.f + (ys + 0.5) * transform.e
        tf = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(ax, ay)
        # to UTM for filtering
        ux, uy = ll_to_utm(lon, lat)
        # exclude near existing
        if existing_utm:
            union = unary_union(existing_utm)
            d = np.array([Point(x, y).distance(union) for x, y in zip(ux, uy)])
            mask_ok = d >= excl_buf
            lon, lat, ux, uy = lon[mask_ok], lat[mask_ok], ux[mask_ok], uy[mask_ok]
        # greedy spatially dispersed
        order = rng.permutation(len(lon))
        keep_lon, keep_lat = [], []
        kept = []
        for i in order:
            x, y = ux[i], uy[i]
            if kept and min(math.hypot(x - kx, y - ky) for kx, ky in kept) < min_sep:
                continue
            keep_lon.append(float(lon[i]))
            keep_lat.append(float(lat[i]))
            kept.append((x, y))
            if len(keep_lon) >= n:
                break
    return pd.DataFrame({"lon": keep_lon, "lat": keep_lat})


def main():
    gong = gpd.read_file(GONG_ROI)
    polys = gpd.read_file(POLY_FULL)
    manifest = pd.read_csv(MANIFEST)
    holdout = polys[polys["split"] == "val_holdout"].copy()
    assert len(holdout) == 18, len(holdout)

    # UTM versions of existing ALL digitized faces (avoid overlap with train too if any in Gongliu — none expected)
    all_polys_utm = polys.to_crs(32645)
    existing_utm = list(all_polys_utm.geometry)
    holdout_utm = list(holdout.to_crs(32645).geometry)

    # ---- quota tracker ----
    hman = manifest[manifest.split == "val_holdout"]
    have = {
        0: int((hman.class_code == 0).sum()),
        1: int((hman.class_code == 1).sum()),
        2: int((hman.class_code == 2).sum()),
    }
    area_have = {
        0: float(hman.loc[hman.class_code == 0, "area_ha"].sum()),
        1: float(hman.loc[hman.class_code == 1, "area_ha"].sum()),
        2: 0.0,
    }
    target_min = {0: 15, 1: 25, 2: 15}
    target_soft = {0: 18, 1: 28, 2: 18}  # toward ~50 total
    need = {c: max(0, target_min[c] - have[c]) for c in (0, 1, 2)}

    quota = pd.DataFrame(
        [
            {
                "class_code": c,
                "class_name": CLASS_NAME[c],
                "class_name_zh": CLASS_NAME_ZH[c],
                "have_n": have[c],
                "have_area_ha": round(area_have[c], 2),
                "target_min_n": target_min[c],
                "target_soft_n": target_soft[c],
                "need_min_n": need[c],
                "need_soft_n": max(0, target_soft[c] - have[c]),
                "next_poly_id_start": f"YL-POLY-{NEXT_POLY_ID:04d}",
                "split": "val_holdout",
                "county": "巩留县",
            }
            for c in (1, 2, 0)
        ]
    )
    quota.loc[len(quota)] = {
        "class_code": -1,
        "class_name": "TOTAL",
        "class_name_zh": "合计",
        "have_n": sum(have.values()),
        "have_area_ha": round(sum(area_have.values()), 2),
        "target_min_n": 40,
        "target_soft_n": 50,
        "need_min_n": max(0, 40 - sum(have.values())),
        "need_soft_n": max(0, 50 - sum(have.values())),
        "next_poly_id_start": f"YL-POLY-{NEXT_POLY_ID:04d}",
        "split": "val_holdout",
        "county": "巩留县",
    }
    quota.to_csv(OUT / "quota_tracker.csv", index=False)

    # ---- EXISTING holdout KML ----
    write_existing_holdout_kml(holdout, OUT / "EXISTING_holdout_reference.kml")

    # ---- Build candidate pools ----
    rows = []  # will become manifest

    def add_pool(df_pts, class_code, source, reason, priority, id_prefix):
        nonlocal rows
        if df_pts is None or len(df_pts) == 0:
            return
        d = df_pts.copy()
        d["lon"] = pd.to_numeric(d["lon"], errors="coerce")
        d["lat"] = pd.to_numeric(d["lat"], errors="coerce")
        d = d.dropna(subset=["lon", "lat"])
        # inside Gongliu
        gdf = gpd.GeoDataFrame(d, geometry=[Point(xy) for xy in zip(d.lon, d.lat)], crs=4326)
        inside = gdf[gdf.within(gong.geometry.iloc[0]) | gdf.intersects(gong.geometry.iloc[0].buffer(0))]
        if inside.empty:
            return
        dist = dist_to_geoms_m(inside.lon.values, inside.lat.values, existing_utm)
        inside = inside.copy()
        inside["dist_m_to_nearest_existing_poly"] = dist
        inside = inside[inside["dist_m_to_nearest_existing_poly"] >= BUFFER_M]
        if inside.empty:
            return
        for i, (_, r) in enumerate(inside.iterrows()):
            rows.append(
                {
                    "cand_id": f"{id_prefix}-{i+1:03d}",
                    "lon": float(r.lon),
                    "lat": float(r.lat),
                    "suggested_class_code": class_code,
                    "source": source,
                    "reason": reason if isinstance(reason, str) else reason(r),
                    "priority": priority,
                    "dist_m_to_nearest_existing_poly": float(r.dist_m_to_nearest_existing_poly),
                }
            )

    # 1) Field GPS apple (Gongliu usable) — highest priority
    fa = pd.read_csv(FIELD_USABLE)
    fa = fa.rename(columns={"_lon": "lon", "_lat": "lat"})
    add_pool(
        fa[["lon", "lat"]],
        1,
        "field_gps",
        "野外GPS确认苹果园；请勾绘该园完整边界（若已在EXISTING中则可跳过）",
        100,
        "CAND-A-FGPS",
    )

    # Also any apple notes from Gongliu hits (should be same 3)
    hits = pd.read_csv(FIELD_HITS)
    hits["_lon"] = pd.to_numeric(hits["_lon"], errors="coerce")
    hits["_lat"] = pd.to_numeric(hits["_lat"], errors="coerce")
    hits = hits.dropna(subset=["_lon", "_lat"])

    def hit_text(r):
        return " ".join(str(r.get(c, "") or "") for c in ["备注1", "备注", "名称", "备注2", "备注3"])

    apple_hits = hits[hits.apply(lambda r: "苹果" in hit_text(r), axis=1)].copy()
    apple_hits = apple_hits.rename(columns={"_lon": "lon", "_lat": "lat"})
    add_pool(
        apple_hits[["lon", "lat"]],
        1,
        "field_gps",
        "野外备注含苹果园",
        95,
        "CAND-A-FHIT",
    )

    # 2) Indep val clear apples (code=1 & is_apple=1)
    filled = pd.read_excel(FILLED)
    filled["confidence"] = filled["confidence"].astype(str).str.strip()
    apples_iv = filled[(filled.interp_class_code == 1) & (filled.is_apple == 1)].copy()
    apples_iv = apples_iv.rename(columns={"lon": "lon", "lat": "lat"})
    add_pool(
        apples_iv[["lon", "lat", "sample_id", "confidence"]],
        1,
        "visual_hr",
        "独立判读表 code=1 苹果园（请在Google Earth高分影像再确认后勾绘边界）",
        80,
        "CAND-A-IV",
    )

    # Ambiguous is_apple=1 & code=2 → other_orchard with caution note (not apple pool)
    amb = filled[(filled.interp_class_code == 2) & (filled.is_apple == 1)].copy()
    add_pool(
        amb.rename(columns={"lon": "lon", "lat": "lat"})[["lon", "lat"]],
        2,
        "visual_hr",
        "判读表 class=2 但 is_apple=1，树种存疑；勾绘前目视确认，不确定则标 class_code=2",
        70,
        "CAND-O-AMB",
    )

    # Clear other orchard code=2 & is_apple!=1
    other_iv = filled[(filled.interp_class_code == 2) & (filled.is_apple != 1)].copy()
    add_pool(
        other_iv[["lon", "lat"]],
        2,
        "visual_hr",
        "独立判读其他果园 (class=2)；勾绘边界，勿标为苹果",
        75,
        "CAND-O-IV",
    )

    # Field unspecified orchard + apricot
    if FIELD_ORCH.exists():
        fo = pd.read_csv(FIELD_ORCH).rename(columns={"_lon": "lon", "_lat": "lat"})
        add_pool(
            fo[["lon", "lat"]],
            2,
            "field_survey",
            "野外果园未分树种；优先目视判读，苹果则改 class_code=1，否则 class_code=2",
            85,
            "CAND-O-FS",
        )
    apricot = hits[hits.apply(lambda r: any(k in hit_text(r) for k in ["杏", "桃", "梨", "核桃"]), axis=1)].copy()
    if len(apricot):
        apricot = apricot.rename(columns={"_lon": "lon", "_lat": "lat"})
        add_pool(
            apricot[["lon", "lat"]],
            2,
            "field_survey",
            "野外非苹果果树（如杏）；标 class_code=2",
            90,
            "CAND-O-APR",
        )

    # Survey AOMC candidates in Gongliu → apple suggestions (visual confirm)
    survey = pd.read_csv(SURVEY)
    surv_gl = survey[survey.county == "巩留县"][["lon", "lat"]].copy()
    add_pool(
        surv_gl,
        1,
        "aomc_survey_candidate",
        "AOMC果园候选（非真值）；Google Earth确认若为苹果园则勾绘 class_code=1，其他果园则2，非果园跳过",
        55,
        "CAND-A-AOMC",
    )

    # Extra dispersed AOMC samples for spatial coverage (apple suggestions)
    aomc_extra = sample_aomc_points(gong, existing_utm, n=35, min_sep=500.0)
    add_pool(
        aomc_extra,
        1,
        "aomc_dispersed",
        "AOMC=1 分散采样点（非真值）；用于填补空间空白，目视确认后勾绘",
        40,
        "CAND-A-DISP",
    )

    # Half of extra AOMC also as other-orchard suggestions in under-covered zones
    # (reuse unused ones after apple thinning — added below after thinning)

    # Non-orchard hard negatives from indep val
    hard_map = {
        3: ("农田/耕地硬负样本，靠近果园尤佳", 60),
        6: ("建设用地硬负样本", 55),
        7: ("水体硬负样本", 50),
        4: ("林地硬负样本（易与果园混淆）", 58),
        5: ("灌丛草地硬负样本", 45),
        8: ("裸地/其他硬负样本", 40),
    }
    for code, (reason, pri) in hard_map.items():
        sub = filled[filled.interp_class_code == code].copy()
        # prefer high confidence
        sub["_pri"] = sub.confidence.map({"高": 2, "中": 1, "低": 0}).fillna(0) + pri / 100.0
        sub = sub.sort_values("_pri", ascending=False)
        add_pool(
            sub[["lon", "lat"]],
            0,
            "visual_hr",
            reason,
            pri,
            f"CAND-N-C{code}",
        )

    # Near-orchard hard-negative offsets from holdout apple centroids (suggested cropland/shelterbelt)
    near_rows = []
    apple_h = holdout[holdout.class_code == 1]
    rng = np.random.default_rng(SEED + 7)
    offsets_m = [
        (180, 0, "holdout苹果园东侧邻近耕地/防护林候选"),
        (-180, 0, "holdout苹果园西侧邻近耕地/防护林候选"),
        (0, 180, "holdout苹果园北侧邻近耕地/防护林候选"),
        (0, -180, "holdout苹果园南侧邻近耕地/防护林候选"),
        (250, 250, "holdout苹果园对角邻近硬负样本候选"),
        (-250, -250, "holdout苹果园对角邻近硬负样本候选"),
    ]
    for _, r in apple_h.iterrows():
        cx, cy = ll_to_utm([r.cen_lon], [r.cen_lat])
        for dx, dy, note in offsets_m:
            # jitter
            jx, jy = dx + float(rng.normal(0, 25)), dy + float(rng.normal(0, 25))
            tf = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
            lon, lat = tf.transform(cx[0] + jx, cy[0] + jy)
            near_rows.append({"lon": lon, "lat": lat, "reason": note})
    near_df = pd.DataFrame(near_rows)
    # filter inside county & away from existing
    if len(near_df):
        gdf = gpd.GeoDataFrame(near_df, geometry=[Point(xy) for xy in zip(near_df.lon, near_df.lat)], crs=4326)
        inside = gdf[gdf.within(gong.geometry.iloc[0])]
        if len(inside):
            dist = dist_to_geoms_m(inside.lon.values, inside.lat.values, existing_utm)
            inside = inside.copy()
            inside["dist_m"] = dist
            inside = inside[inside.dist_m >= BUFFER_M]
            for i, (_, r) in enumerate(inside.iterrows()):
                rows.append(
                    {
                        "cand_id": f"CAND-N-NEAR-{i+1:03d}",
                        "lon": float(r.lon),
                        "lat": float(r.lat),
                        "suggested_class_code": 0,
                        "source": "hard_negative_near_orchard",
                        "reason": r.reason,
                        "priority": 65,
                        "dist_m_to_nearest_existing_poly": float(r.dist_m),
                    }
                )

    cand = pd.DataFrame(rows)
    if cand.empty:
        raise SystemExit("No candidates generated")

    # Dedup near-identical points across sources (keep higher priority)
    cand = cand.sort_values("priority", ascending=False)
    xs, ys = ll_to_utm(cand.lon.values, cand.lat.values)
    keep_idx = []
    kept_xy = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if kept_xy and min(math.hypot(x - kx, y - ky) for kx, ky in kept_xy) < 60.0:
            continue
        keep_idx.append(i)
        kept_xy.append((x, y))
    cand = cand.iloc[keep_idx].reset_index(drop=True)

    # Per-class thinning to target sizes with spatial dispersion
    targets = {1: 55, 2: 35, 0: 38}
    final_parts = []
    for cc, n_tgt in targets.items():
        sub = cand[cand.suggested_class_code == cc].copy()
        sep = 150.0 if cc == 1 else (180.0 if cc == 0 else 120.0)
        if cc == 0:
            # Rebalance non-orchard: keep mix of visual_hr + near-orchard
            vis = sub[sub.source == "visual_hr"]
            near = sub[sub.source == "hard_negative_near_orchard"]
            other_src = sub[~sub.source.isin(["visual_hr", "hard_negative_near_orchard"])]
            n_vis = min(22, max(18, n_tgt // 2 + 2))
            vis_th = pairwise_min_sep_filter(vis, 200.0, priority_col="priority").head(n_vis)
            n_near = n_tgt - len(vis_th)
            near_th = pairwise_min_sep_filter(near, 100.0, priority_col="priority").head(max(0, n_near))
            # fill remainder from other if short
            th = pd.concat([vis_th, near_th, other_src], ignore_index=True)
            th = pairwise_min_sep_filter(th, 100.0, priority_col="priority").head(n_tgt)
        else:
            th = pairwise_min_sep_filter(sub, sep, priority_col="priority")
            if len(th) > n_tgt:
                th = th.head(n_tgt)
        final_parts.append(th)
    final = pd.concat(final_parts, ignore_index=True)

    # Reassign clean cand_ids
    final = final.sort_values(["suggested_class_code", "priority"], ascending=[True, False]).reset_index(drop=True)
    new_ids = []
    counters = {0: 0, 1: 0, 2: 0}
    prefix = {0: "N", 1: "A", 2: "O"}
    for _, r in final.iterrows():
        cc = int(r.suggested_class_code)
        counters[cc] += 1
        new_ids.append(f"CAND-{prefix[cc]}-{counters[cc]:03d}")
    final["cand_id"] = new_ids

    # Spatial cluster tag (for README)
    if len(final) >= 3:
        km = KMeans(n_clusters=min(5, len(final)), random_state=SEED, n_init=10)
        final["spatial_cluster"] = km.fit_predict(final[["lon", "lat"]].values)
    else:
        final["spatial_cluster"] = 0

    # Write manifest
    man_cols = [
        "cand_id",
        "lon",
        "lat",
        "suggested_class_code",
        "source",
        "reason",
        "dist_m_to_nearest_existing_poly",
        "priority",
        "spatial_cluster",
    ]
    final[man_cols].to_csv(OUT / "candidate_points_manifest.csv", index=False)

    # KMLs per class
    apple = final[final.suggested_class_code == 1]
    other = final[final.suggested_class_code == 2]
    non = final[final.suggested_class_code == 0]

    write_candidates_kml(
        apple,
        OUT / "CANDIDATES_apple.kml",
        "CANDIDATES apple (class_code=1)",
        "applePin",
        "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png",
        "ff00aa00",
        "巩留苹果园候选点（非真值面）。在点附近勾绘园块边界，属性见 README。",
    )
    write_candidates_kml(
        other,
        OUT / "CANDIDATES_other_orchard.kml",
        "CANDIDATES other orchard (class_code=2)",
        "otherPin",
        "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png",
        "ff00d0ff",
        "其他果园/未分树种候选点。优先勾绘；确认苹果则改 class_code=1。",
    )
    write_candidates_kml(
        non,
        OUT / "CANDIDATES_non_orchard.kml",
        "CANDIDATES non-orchard hard negatives (class_code=0)",
        "nonPin",
        "http://maps.google.com/mapfiles/kml/paddle/red-circle.png",
        "ff0000ff",
        "硬负样本候选：农田/防护林/水体/建设用地等。在点附近勾绘面。",
    )

    # Bundle KMZ
    kmz_path = OUT / "CANDIDATES_all.kmz"
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Combined doc.kml that NetworkLinks? Simpler: one combined KML as doc.kml
        combined = [kml_header("Gongliu holdout expansion kit", "Existing holdout + candidates")]
        combined.append(style_poly("applePoly", "00aa00", "00aa00", "70"))
        combined.append(style_poly("nonPoly", "0000ff", "0000ff", "70"))
        combined.append(style_point("applePin", "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png", "ff00aa00"))
        combined.append(style_point("otherPin", "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png", "ff00d0ff"))
        combined.append(style_point("nonPin", "http://maps.google.com/mapfiles/kml/paddle/red-circle.png", "ff0000ff"))
        # embed existing holdout folder by reading and stripping header — easier: copy placemarks via rewrite
        # Just add NetworkLink-style folders by inlining from written files is messy.
        # Re-call write pieces into combined:
        combined.append("  <Folder><name>0_EXISTING_holdout_reference</name>\n")
        for _, r in holdout.iterrows():
            style = "#applePoly" if int(r.class_code) == 1 else "#nonPoly"
            geom = r.geometry
            polys_g = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            area_ha = float(r.get("area_ha") or 0)
            nm = str(r["name"])
            combined.append(
                f"    <Placemark><name>{html.escape(nm)} | c{int(r.class_code)}</name>"
                f"<description>EXISTING {nm}</description><styleUrl>{style}</styleUrl>"
            )
            if len(polys_g) > 1:
                combined.append("<MultiGeometry>")
            for poly in polys_g:
                ext = " ".join(f"{x:.8f},{y:.8f},0" for x, y in poly.exterior.coords)
                combined.append(
                    f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{ext}</coordinates>"
                    f"</LinearRing></outerBoundaryIs></Polygon>"
                )
            if len(polys_g) > 1:
                combined.append("</MultiGeometry>")
            combined.append("</Placemark>\n")
        combined.append("  </Folder>\n")

        for cc, folder, style, sub in [
            (1, "1_CANDIDATES_apple", "#applePin", apple),
            (2, "2_CANDIDATES_other_orchard", "#otherPin", other),
            (0, "3_CANDIDATES_non_orchard", "#nonPin", non),
        ]:
            combined.append(f"  <Folder><name>{folder} (n={len(sub)})</name>\n")
            for _, r in sub.iterrows():
                d = (
                    f"suggested class_code={cc}; source={r.source}; "
                    f"split=val_holdout; county=巩留县; {r.reason}"
                )
                combined.append(
                    f"    <Placemark><name>{html.escape(r.cand_id)} | c{cc}</name>"
                    f"<description>{html.escape(d)}</description><styleUrl>{style}</styleUrl>"
                    f"<Point><coordinates>{r.lon:.8f},{r.lat:.8f},0</coordinates></Point>"
                    f"</Placemark>\n"
                )
            combined.append("  </Folder>\n")
        combined.append(kml_footer())
        zf.writestr("doc.kml", "".join(combined))
        # also include individual files
        for fn in [
            "EXISTING_holdout_reference.kml",
            "CANDIDATES_apple.kml",
            "CANDIDATES_other_orchard.kml",
            "CANDIDATES_non_orchard.kml",
        ]:
            zf.write(OUT / fn, arcname=fn)

    # Cluster summary for README
    cluster_summary = []
    for cid, g in final.groupby("spatial_cluster"):
        cluster_summary.append(
            {
                "cluster": int(cid),
                "n": len(g),
                "lon_mean": round(g.lon.mean(), 3),
                "lat_mean": round(g.lat.mean(), 3),
                "classes": dict(g.suggested_class_code.value_counts()),
            }
        )

    # README Chinese
    n_apple, n_other, n_non = len(apple), len(other), len(non)
    src_apple = apple.source.value_counts().to_dict()
    src_other = other.source.value_counts().to_dict()
    src_non = non.source.value_counts().to_dict()

    gaps = []
    if (apple.source == "field_gps").sum() < 5:
        gaps.append(
            "巩留县境内野外确认苹果GPS点极少（仅约3个，且多已落在现有NW holdout面内/缓冲带），"
            "苹果候选主要依赖独立判读 visual_hr 与 AOMC 候选——勾绘时务必目视确认。"
        )
    if n_other < 25:
        gaps.append("其他果园候选偏少，已尽量纳入野外未分树种果园与判读 class=2。")
    gaps.append(
        "AOMC 仅为多年生果园弱标签，不是苹果真值；标为 aomc_* 的点必须在 Google Earth 确认后再勾绘。"
    )

    cluster_lines = "\n".join(
        f"| {c['cluster']} | {c['n']} | {c['lon_mean']}, {c['lat_mean']} | {c['classes']} |"
        for c in cluster_summary
    )

    readme = f"""# 巩留县 Holdout 面样本扩展勾绘包（Google Earth）

**用途**：在现有 v2 `val_holdout`（18 面）基础上，扩展巩留独立验证面样本至约 40–60 面，随后再训练。  
**重要**：本包提供的是**候选点**与**已有 holdout 参考面**，**不是**新的地面真值多边形。请勿把候选点直接当训练标签。

## 配额一览（当前 → 目标）

| class_code | 类别 | 现有面数 | 现有面积(ha) | 目标下限 | 软目标 | 尚需(下限) | 尚需(软) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 苹果园 | {have[1]} | {area_have[1]:.1f} | 25–30 | ~28 | **+{need[1]}～+{max(0,30-have[1])}** | +{max(0,28-have[1])} |
| 2 | 其他果园 | {have[2]} | {area_have[2]:.1f} | ≥15 | ~18 | **+{need[2]}（优先）** | +{max(0,18-have[2])} |
| 0 | 非果园 | {have[0]} | {area_have[0]:.1f} | ≥15 | ~18 | **+{need[0]}** | +{max(0,18-have[0])} |
| 合计 | — | {sum(have.values())} | {sum(area_have.values()):.1f} | 40–60 | ~50 | +{max(0,40-sum(have.values()))}～ | +{max(0,50-sum(have.values()))} |

新面一律：`split=val_holdout`，`county=巩留县`。  
**不要**把现有 train（非巩留）面改入 holdout。训练继续只用非巩留。

新面 `poly_id` 从 **`YL-POLY-0064`** 起编（现有最大为 YL-POLY-0063）。

## Google Earth 操作步骤

1. 打开 Google Earth Pro（推荐）或 Google Earth Web。
2. 文件 → 打开：
   - `EXISTING_holdout_reference.kml`（绿色=苹果，蓝色=非果园）— **仅参考，勿重复勾绘**
   - `CANDIDATES_apple.kml`（绿点）
   - `CANDIDATES_other_orchard.kml`（黄点）— **优先补 class=2**
   - `CANDIDATES_non_orchard.kml`（红点）
   - 或一次性打开 `CANDIDATES_all.kmz`
3. 也可叠加桌面/项目中的 `Confirmed_apple_field_points_area.kml` 作野外点参考。
4. 在候选点附近，用 **多边形** 工具贴园块/地块边界勾绘（一园一面；面积建议 0.3–20 ha，>50 ha 后期 QC 剔除）。
5. 在多边形**说明/描述**中粘贴属性（一行即可），例如：

```
poly_id=YL-POLY-0064; class_code=1; class_name=apple; source=visual_hr; year_ref=2021; county=巩留县; split=val_holdout; notes=独立判读点附近确认苹果园
```

### 属性约定

| 字段 | 取值 |
|------|------|
| poly_id | YL-POLY-0064 起递增 |
| class_code | 1=苹果园，2=其他果园，0=非果园 |
| class_name | apple / other_orchard / non_orchard |
| source | field_gps / field_survey / visual_hr |
| year_ref | 2021（与论文主年份一致；若依据其他年份影像请注明） |
| county | 巩留县 |
| split | val_holdout |
| notes | 自由备注 |

### 勾绘规则（与既有一致）

- 仅确认苹果标 **1**；拿不准树种标 **2**；非果园硬负样本标 **0**。
- 避开已有面（KML 中已有面及缓冲约 50–100 m 内勿再勾）。
- 空间上尽量分散：除现有 NW、SE 两簇外，至少再覆盖 **1 个以上**新簇（西部/中部农田带等）。
- AOMC / 分散采样点**必须**目视确认；不像果园就跳过，不要硬勾。
- 每个苹果簇附近尽量配若干 **0**（耕地、防护林/林带、水体、建设用地）。

## 本包候选点数量（供挑选，非必须全勾）

| 图层 | 建议 class | 候选点数 | 主要来源 |
|------|------------|---------:|----------|
| CANDIDATES_apple.kml | 1 | **{n_apple}** | {src_apple} |
| CANDIDATES_other_orchard.kml | 2 | **{n_other}** | {src_other} |
| CANDIDATES_non_orchard.kml | 0 | **{n_non}** | {src_non} |

建议至少勾完配额：苹果 +{need[1]}～+{max(0,30-have[1])}，其他果园 +{need[2]}，非果园 +{need[0]}；合计约到 50 面左右即可停。

## 候选点空间簇（KMeans 参考）

| cluster | n | 中心 (lon, lat) | class 构成 |
|--------:|--:|-----------------|------------|
{cluster_lines}

现有 holdout 两簇：NW ≈ 82.41–82.44 / 43.48–43.51；SE ≈ 82.83–82.87 / 43.25–43.26。

## 文件清单

| 文件 | 说明 |
|------|------|
| README_holdout_expansion_zh.md | 本说明 |
| EXISTING_holdout_reference.kml | 当前 18 个 holdout 面 |
| CANDIDATES_apple.kml | 苹果候选**点** |
| CANDIDATES_other_orchard.kml | 其他果园候选**点** |
| CANDIDATES_non_orchard.kml | 非果园硬负样本候选**点** |
| CANDIDATES_all.kmz | 以上打包 |
| quota_tracker.csv | 配额跟踪表 |
| candidate_points_manifest.csv | 全部候选点清单（含距最近已有面距离） |

## 数据缺口 / 注意

{chr(10).join('- ' + g for g in gaps)}

## 勾绘完成后

将新多边形导出为 KML/KMZ 或 GeoJSON，放到  
`outputs/revision/polygon_samples/` 或桌面「苹果论文/面样本勾绘/」下，告知即可合并进 v3 清单并重训。
"""
    (OUT / "README_holdout_expansion_zh.md").write_text(readme, encoding="utf-8")

    # Summary print
    print("=== HOLD OUT EXPANSION KIT DONE ===")
    print(quota.to_string(index=False))
    print(f"candidates apple={n_apple} other={n_other} non={n_non} total={len(final)}")
    print("sources apple", src_apple)
    print("sources other", src_other)
    print("sources non", src_non)
    print("clusters", cluster_summary)
    print("out", OUT)
    print("files", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
