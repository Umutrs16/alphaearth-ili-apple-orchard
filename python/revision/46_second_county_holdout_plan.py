# -*- coding: utf-8 -*-
"""F: second-county final-holdout planning (no model built). Training polygons / points per county and class (old 38 + new 239),
and the training set that would remain if one county were removed."""
from pathlib import Path
import pandas as pd, json
ROOT = Path("/workspace/apple_orchard_project"); E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; OUT = ROOT / "outputs/revision/v41_review/second_county_plan"; OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(E / "split_manifest_expanded.csv"); t = m[m.split == "train"].copy()
import geopandas as gpd, sys
sys.path.insert(0, str(ROOT / "python/revision/figures_v3_scripts")); import official_boundary as ob
cty = ob.county(); miss = t.county.isna()
g = gpd.GeoDataFrame(t[miss], geometry=gpd.points_from_xy(t[miss].cen_lon, t[miss].cen_lat), crs=4326)
j = gpd.sjoin(g, cty[["name", "geometry"]].rename(columns={"name": "cty_name"}), how="left", predicate="within")
t.loc[miss, "county"] = j[~j.index.duplicated()]["cty_name"].values
print("filled county for", int(miss.sum()), "old polygons:", t.loc[miss, "county"].value_counts().to_dict())
t["set"] = t.label_source.map({"v2_train": "old38", "yl_trn_expansion": "new239"})
pts = pd.read_csv(E / "sample_points_expanded.csv"); pts = pts[pts.split == "train"].merge(t[["name", "county"]].rename(columns={"name": "poly_name", "county": "cty"}), on="poly_name")
tab = t.pivot_table(index="county", columns="class_code", values="name", aggfunc="count", fill_value=0).rename(columns={0: "c0_nonorchard", 1: "c1_apple", 2: "c2_other_orchard"})
tab["old38"] = t[t.set == "old38"].groupby("county").size(); tab["old38_apple"] = t[(t.set == "old38") & (t.class_code == 1)].groupby("county").size()
tab["new239"] = t[t.set == "new239"].groupby("county").size(); tab = tab.fillna(0).astype(int)
tab["n_polys"] = tab[["c0_nonorchard", "c1_apple", "c2_other_orchard"]].sum(axis=1)
tab["n_points"] = pts.groupby("cty").size(); tab["apple_points"] = pts[pts.class_code == 1].groupby("cty").size()
tab["apple_area_ha"] = t[t.class_code == 1].groupby("county").area_ha_utm44n.sum().round(1)
tab = tab.fillna(0)
N, NA, NP, NAP = len(t), int((t.class_code == 1).sum()), len(pts), int((pts.class_code == 1).sum())
tab["remaining_polys_if_removed"] = N - tab.n_polys; tab["remaining_apple_polys"] = NA - tab.c1_apple
tab["remaining_points"] = NP - tab.n_points; tab["remaining_apple_points"] = NAP - tab.apple_points
tab["pct_train_polys_lost"] = (100 * tab.n_polys / N).round(1); tab["pct_apple_polys_lost"] = (100 * tab.c1_apple / NA).round(1)
tab = tab.sort_values("c1_apple", ascending=False); tab.to_csv(OUT / "training_by_county.csv", encoding="utf-8-sig")
json.dump({"total_train_polys": N, "total_apple_polys": NA, "total_train_points": NP, "total_apple_points": NAP}, open(OUT / "totals.json", "w"))
print(N, NA, NP, NAP); print(tab.to_string())
