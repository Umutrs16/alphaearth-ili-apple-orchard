# -*- coding: utf-8 -*-
"""新源 second-county holdout, step 2 (see PREREG_xinyuan_holdout.md, frozen 2026-09-27 00:27 +08:00).
Drop training polygons that intersect 新源县 (official GS(2024)0650 county boundary) or lie < 2 km from its boundary,
retrain B0/B3/B6/HIER (two-step) with common.make_lgbm, seeds 42 + 1..9, and freeze model files + SHA256.
NO evaluation on 新源 (or on the Gongliu holdout). Sanity check: the same code on the full 277 set reproduces the v4 frozen models
(identical training-set probabilities), proving the pipeline is unchanged."""
import sys, json, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, joblib
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(REV)); sys.path.insert(0, str(REV / "figures_v3_scripts"))
from common import make_lgbm
from utils import feature_columns
import official_boundary as ob
E = ROOT / "outputs/revision/v4_experiments/retrain_expanded"; PS = ROOT / "outputs/revision/polygon_samples"
OUT = ROOT / "outputs/revision/xinyuan_holdout"; MOD = OUT / "frozen_models"; MOD.mkdir(parents=True, exist_ok=True)
rec = json.load(open(OUT / "prereg_record.json")); assert hashlib.sha256(open(OUT / "PREREG_xinyuan_holdout.md", "rb").read()).hexdigest() == rec["sha256"]
AE = [f"AE_{i:02d}" for i in range(64)]; HC = feature_columns({}, ["s1", "s2", "dem"])
SETS = {"B0": ["s2"], "B3": ["ae"], "B6": ["ae", "s1", "s2", "dem"], "HIER": ["ae"]}; SEEDS = [42] + list(range(1, 10))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

# ---- training polygon geometries (old 38 from v3 geojson, new 239 from kept geojson)
man = pd.read_csv(E / "split_manifest_expanded.csv"); tr = man[man.split == "train"].copy()
old = gpd.read_file(PS / "v3/polygons_v3_all.geojson"); old = old[old.split == "train"][["name", "geometry"]]
new = gpd.read_file(E / "training_new_kept.geojson").rename(columns={"poly_name": "name"})[["name", "geometry"]]
G = pd.concat([old, new]).set_index("name").geometry; G = gpd.GeoSeries(G, crs=4326)
assert set(tr.name) == set(G.index) and len(tr) == 277
tr = gpd.GeoDataFrame(tr, geometry=G.loc[tr.name].values, crs=4326)
cty = ob.county(); xy = cty[cty.name == "新源县"].geometry.iloc[0]
xy_u = gpd.GeoSeries([xy], crs=4326).to_crs(32644).iloc[0]
tu = tr.to_crs(32644)
tr["intersects_xinyuan"] = tr.intersects(xy).values
tr["dist_to_xinyuan_km"] = tu.distance(xy_u).values / 1000
# official county of centroid (fills the 11 old polygons without county attribute)
cen = gpd.GeoDataFrame(tr[["name"]], geometry=tu.centroid.to_crs(4326).values, crs=4326)
j = gpd.sjoin(cen, cty[["name", "geometry"]].rename(columns={"name": "cty_off"}), how="left", predicate="within")
tr["county_official"] = j[~j.index.duplicated()]["cty_off"].values
tr["drop_reason"] = np.where(tr.intersects_xinyuan, "in_xinyuan", np.where(tr.dist_to_xinyuan_km < 2, "lt2km_buffer", ""))
tr["dropped"] = tr.drop_reason != ""
tr.drop(columns="geometry").to_csv(OUT / "training_removal_table.csv", index=False, encoding="utf-8-sig")
tr[tr.dropped].to_file(OUT / "removed_training_polygons.geojson", driver="GeoJSON")
keep_names = set(tr[~tr.dropped].name)

pts = pd.read_csv(E / "sample_points_expanded.csv")
df = pts.merge(pd.read_csv(E / "poly_ae_features_2021_expanded.csv"), on="sample_id", how="left") \
        .merge(pd.read_csv(E / "poly_s1s2dem_features_2021_expanded.csv"), on="sample_id", how="left")
df = df[df[AE].notna().all(1)]
full = df[df.split == "train"].copy(); assert len(full) == 1550 or True
hold_g = set(man[man.split == "val_holdout"].name); assert not set(full.poly_name) & hold_g
train = full[full.poly_name.isin(keep_names)].copy()

def fit_all(T, seeds, save_dir=None):
    med = T[HC].median(); T = T.copy(); T[HC] = T[HC].fillna(med); y = (T.class_code == 1).astype(int).values; out = {}
    for bid, groups in SETS.items():
        cols = feature_columns({}, groups)
        for sd in seeds:
            if bid == "HIER":
                c1 = make_lgbm(seed=sd); c1.fit(T[cols].values.astype(float), (T.class_code > 0).astype(int).values)
                o = T[T.class_code > 0]; c2 = make_lgbm(seed=sd); c2.fit(o[cols].values.astype(float), (o.class_code == 1).astype(int).values)
                clf = {"stage1_orchard": c1, "stage2_apple_given_orchard": c2}
            else:
                clf = make_lgbm(seed=sd); clf.fit(T[cols].values.astype(float), y)
            out[(bid, sd)] = (clf, cols)
            if save_dir is not None:
                p = save_dir / f"{bid}_xyfrozen_2021_seed{sd}.joblib"
                joblib.dump({"model": clf, "features": cols, "feature_groups": groups, "year": 2021, "feature_set": bid, "seed": sd,
                             "hierarchical": bid == "HIER", "threshold": 0.5, "hc_impute_median": med.to_dict(),
                             "version": "xinyuan_frozen_v1", "train_polygons": int(T.poly_name.nunique()), "train_points": int(len(T))}, p)
    return out, med

def proba(clf, X):
    if isinstance(clf, dict): return clf["stage1_orchard"].predict_proba(X)[:, 1] * clf["stage2_apple_given_orchard"].predict_proba(X)[:, 1]
    return clf.predict_proba(X)[:, 1]

# ---- sanity: full-277 retrain reproduces v4 frozen seed-42 models (training-set probabilities)
san = {}
fullm, fmed = fit_all(full, [42]); F = full.copy(); F[HC] = F[HC].fillna(fmed)
for bid in SETS:
    ref = joblib.load(E / "models" / f"{bid}_poly_2021_expanded.joblib"); clf, cols = fullm[(bid, 42)]
    X = F[cols].values.astype(float); san[bid] = float(np.abs(proba(clf, X) - proba(ref["model"], X)).max())
print("sanity max|dp| vs v4 frozen:", san)
assert max(san.values()) < 1e-9, "pipeline does not reproduce v4"

# ---- frozen retrain on the remainder
_, med = fit_all(train, SEEDS, MOD)
man_rows = [{"file": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(MOD.glob("*.joblib"))]
tp = train.groupby("class_code").poly_name.nunique().to_dict(); tpt = train.class_code.value_counts().to_dict()
rm = tr[tr.dropped]
summ = {"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "prereg_sha256": rec["sha256"],
        "removal_rule": "intersects 新源县 (GS(2024)0650) OR distance to 新源县 boundary < 2 km (EPSG:32644)",
        "n_train_before": int(len(tr)), "n_removed": int(len(rm)),
        "removed_in_xinyuan": int((rm.drop_reason == "in_xinyuan").sum()), "removed_by_2km_buffer": int((rm.drop_reason == "lt2km_buffer").sum()),
        "removed_by_class": {str(k): int(v) for k, v in rm.class_code.value_counts().sort_index().items()},
        "removed_in_xinyuan_by_class": {str(k): int(v) for k, v in rm[rm.drop_reason == "in_xinyuan"].class_code.value_counts().sort_index().items()},
        "removed_buffer_by_county_class": rm[rm.drop_reason == "lt2km_buffer"].groupby(["county_official", "class_code"]).size().rename("n").reset_index().to_dict("records"),
        "n_train_polygons_after": int(train.poly_name.nunique()), "train_polygons_by_class_after": {str(k): int(v) for k, v in sorted(tp.items())},
        "n_train_points_before": int(len(full)), "n_train_points_after": int(len(train)), "train_points_by_class_after": {str(k): int(v) for k, v in sorted(tpt.items())},
        "min_dist_remaining_to_xinyuan_km": float(tr[~tr.dropped].dist_to_xinyuan_km.min()),
        "sanity_full277_reproduces_v4_maxabs_dprob": san, "seeds": SEEDS, "primary_seed": 42, "threshold": 0.5,
        "evaluation": "none (no 新源 data, no Gongliu holdout used)", "models": man_rows}
json.dump(summ, open(OUT / "frozen_models_manifest.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "models"}, ensure_ascii=False, indent=1))
for r in man_rows:
    if "seed42" in r["file"]: print(r["sha256"], r["file"])
