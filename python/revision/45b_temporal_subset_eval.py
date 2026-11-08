# -*- coding: utf-8 -*-
"""B (analysis, run AFTER the user fills holdout_113_temporal_template.xlsx).
Temporal-consistent subset of the Gongliu challenge set: |year(image_date) - 2021| <= 1 AND temporal_stable_2019_2023 == 'Y'
(optionally also field_verified == 'Y' as a second subset). Re-scores the FROZEN seed-42 predictions of the expanded-training
B0 / B3 / B6 / two-step (HIER) models and the old B3 (no retraining, no tuning, threshold 0.5, majority vote over >=3 points =
the main-101 rule). Reports F1 / P / R with polygon-bootstrap 95% CIs (1000, seed 42) and paired dF1 vs new B3.
Usage: python 45b_temporal_subset_eval.py <filled.xlsx> [out_dir]"""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
ROOT = Path("/workspace/apple_orchard_project"); V4 = ROOT / "outputs/revision/v4_experiments"; E = V4 / "retrain_expanded"
xl = Path(sys.argv[1]); OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs/revision/v41_review/temporal_subset"; OUT.mkdir(parents=True, exist_ok=True)
T = pd.read_excel(xl, sheet_name="holdout_temporal", dtype=str).fillna("")
assert len(T) == 113, "expected 113 rows"
def year(s):
    m = re.search(r"(19|20)\d{2}", str(s)); return int(m.group(0)) if m else np.nan
T["year"] = T.image_date.map(year)
for c in ("field_verified", "temporal_stable_2019_2023"): T[c] = T[c].str.strip().str.upper().replace({"UNSURE": "unsure"})
MODELS = {"B3_new": E / "pred_polys_B3_expanded_seed42.csv", "B0_new": E / "pred_polys_B0_expanded_seed42.csv", "B6_new": E / "pred_polys_B6_expanded_seed42.csv",
          "two_step_new": E / "pred_polys_HIER_expanded_seed42.csv", "B3_old": V4 / "pred_polys_B3_seed42.csv"}
subsets = {"all_main101": lambda d: np.ones(len(d), bool),
           "temporal_consistent": lambda d: (np.abs(d.year - 2021) <= 1) & (d.temporal_stable_2019_2023 == "Y"),
           "temporal_consistent_and_field_verified": lambda d: (np.abs(d.year - 2021) <= 1) & (d.temporal_stable_2019_2023 == "Y") & (d.field_verified == "Y"),
           "stable_Y_any_year": lambda d: d.temporal_stable_2019_2023 == "Y"}
preds = {k: pd.read_csv(p).dropna(subset=["pred_major"]).set_index("poly_name") for k, p in MODELS.items()}
base = preds["B3_new"][["class_code", "y"]].join(T.set_index("polygon_id"), how="left")
res = {"design": __doc__, "fill_status": {"n_rows": 113, "n_image_date": int((T.image_date != "").sum()), "n_year_parsed": int(T.year.notna().sum()),
       "stable_counts": T.temporal_stable_2019_2023.value_counts().to_dict(), "field_verified_counts": T.field_verified.value_counts().to_dict(),
       "year_counts": T.year.value_counts().sort_index().to_dict()}, "subsets": {}}
rng = np.random.default_rng(42)
for sn, f in subsets.items():
    d = base[np.asarray(f(base), bool)]; names = d.index.values; n = len(d)
    r = {"n_polys": int(n), "n_apple": int(d.y.sum()), "by_class": d.class_code.value_counts().to_dict()}
    if n == 0 or d.y.sum() == 0: r["note"] = "empty or no apple polygons"; res["subsets"][sn] = r; continue
    if d.y.sum() < 10: r["warning"] = "fewer than 10 apple polygons: CIs very wide, interpret with caution"
    B = [rng.integers(0, n, n) for _ in range(1000)]; y = d.y.values.astype(int)
    P = {k: v.loc[names, "pred_major"].values.astype(int) for k, v in preds.items()}
    for k, p in P.items():
        bs = np.array([[f1_score(y[i], p[i], zero_division=0), precision_score(y[i], p[i], zero_division=0), recall_score(y[i], p[i], zero_division=0)] for i in B])
        dd = np.array([f1_score(y[i], p[i], zero_division=0) - f1_score(y[i], P["B3_new"][i], zero_division=0) for i in B])
        r[k] = {"F1": float(f1_score(y, p, zero_division=0)), "precision": float(precision_score(y, p, zero_division=0)), "recall": float(recall_score(y, p, zero_division=0)),
                "F1_ci": np.quantile(bs[:, 0], [.025, .975]).tolist(), "P_ci": np.quantile(bs[:, 1], [.025, .975]).tolist(), "R_ci": np.quantile(bs[:, 2], [.025, .975]).tolist(),
                "dF1_vs_B3_new": float(f1_score(y, p, zero_division=0) - f1_score(y, P["B3_new"], zero_division=0)), "dF1_vs_B3_new_ci": np.quantile(dd, [.025, .975]).tolist()}
    res["subsets"][sn] = r
json.dump(res, open(OUT / "metrics_temporal_subset.json", "w"), ensure_ascii=False, indent=1, default=float)
rows = [{"subset": sn, "model": k, "n_polys": v["n_polys"], "n_apple": v["n_apple"], **{m: v[k][m] for m in ("F1", "precision", "recall")},
         "F1_ci": v[k]["F1_ci"], "dF1_vs_B3_new_ci": v[k]["dF1_vs_B3_new_ci"]} for sn, v in res["subsets"].items() for k in MODELS if k in v]
pd.DataFrame(rows).to_csv(OUT / "table_temporal_subset.csv", index=False); print(pd.DataFrame(rows).round(3).to_string()); print(json.dumps(res["fill_status"], ensure_ascii=False))
