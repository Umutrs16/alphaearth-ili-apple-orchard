"""Build v3 polygon layer (train + QC'd holdout) -> polygon_samples/v3/polygons_v3_all.geojson"""
from pathlib import Path
import geopandas as gpd, pandas as pd, pyogrio
ROOT = Path("/workspace/apple_orchard_project")
PS = ROOT / "outputs/revision/polygon_samples"

def build():
    man = pd.read_csv(PS / "v3/split_manifest_v3.csv")
    kml = PS / "holdout_expansion/apple_new.kml"
    fr = [gpd.read_file(kml, layer=l) for l in pyogrio.list_layers(kml)[:, 0]]
    h = gpd.GeoDataFrame(pd.concat(fr, ignore_index=True), crs=4326)
    h = h[h.geom_type.str.contains("Polygon")].copy()
    h["name"] = h["Name"].str.split("|").str[0].str.strip()
    tr = gpd.read_file(PS / "user_apple_polygons_full_v2.geojson")[["name", "geometry"]]
    g = pd.concat([h[["name", "geometry"]], tr[~tr.name.isin(h.name)]], ignore_index=True)
    g = gpd.GeoDataFrame(g.drop_duplicates("name"), crs=4326)
    g["geometry"] = g.geometry.force_2d()
    g = g.merge(man[["name", "class_code", "split", "area_ha_utm44n"]], on="name", how="inner")
    g = g[(g.split.isin(["train", "val_holdout"])) & (g.name != "YL-POLY-0095")]
    return g

if __name__ == "__main__":
    g = build()
    print(g.groupby(["split", "class_code"]).size())
    g.to_file(PS / "v3/polygons_v3_all.geojson", driver="GeoJSON")
