# -*- coding: utf-8 -*-
"""v4.2 label-sensitivity on the Gongliu challenge set, driven ONLY by the user's own notes in the returned
holdout_temporal_check_prefilled.xlsx (2026-09-28). Frozen seed-42 predictions, threshold 0.5, majority vote (main-101 rule); no retraining.
S0 original labels
S1 explicit corrections written by the user: YL-POLY-0083 c1->c0 ("这应该是防护林，不是苹果园，要改成0"), YL-POLY-0082 c2->c0 ("农田，需要改成0")
S2 S1 + exclude the 5 polygons the user annotated "野苹果林" (wild apple forest: 0059, 0060, 0123, 0124, 0125; labelled c1)
S3 S2 + tentative notes applied: 0126 c2->c1 ("感觉是苹果园"), 0138 c0->c2 ("可能是其他果园"), 0081 c2->c0 ("感觉是个防护林"), 0142 c2->c0 ("像是防护林")
YL-POLY-0100/0107/0108 (c0 with orchard GPS points) are unchanged in all scenarios: the user left them c0 without a correction note.
Subsets: main-101 and temporal_consistent (|year-2021|<=1 & temporal_stable=Y), same rule as 45b."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
ROOT = Path("/workspace/apple_orchard_project"); V4 = ROOT / "outputs/revision/v4_experiments"; E = V4 / "retrain_expanded"
xl = Path(sys.argv[1]); OUT = ROOT / "outputs/revision/v42/label_sensitivity"; OUT.mkdir(parents=True, exist_ok=True)
T = pd.read_excel(xl, sheet_name="holdout_temporal", dtype=str).fillna("")
T["year"] = T.image_date.map(lambda s: (lambda m: int(m.group(0)) if m else np.nan)(re.search(r"(19|20)\d{2}", s)))
T["stable"] = T.temporal_stable_2019_2023.str.strip().str.upper()
MODELS = {"B3_new": E / "pred_polys_B3_expanded_seed42.csv", "B0_new": E / "pred_polys_B0_expanded_seed42.csv", "B6_new": E / "pred_polys_B6_expanded_seed42.csv",
          "two_step_new": E / "pred_polys_HIER_expanded_seed42.csv", "B3_old": V4 / "pred_polys_B3_seed42.csv"}
preds = {k: pd.read_csv(p).dropna(subset=["pred_major"]).set_index("poly_name") for k, p in MODELS.items()}
base = preds["B3_new"][["class_code"]].join(T.set_index("polygon_id")[["year", "stable"]], how="left")
WILD = ["YL-POLY-0059", "YL-POLY-0060", "YL-POLY-0123", "YL-POLY-0124", "YL-POLY-0125"]
S1 = {"YL-POLY-0083": 0, "YL-POLY-0082": 0}; S3 = {**S1, "YL-POLY-0126": 1, "YL-POLY-0138": 2, "YL-POLY-0081": 0, "YL-POLY-0142": 0}
SC = {"S0_original": ({}, []), "S1_explicit_corrections": (S1, []), "S2_S1_excl_wild_apple_forest": (S1, WILD), "S3_S2_plus_tentative": (S3, WILD)}
for n_ in list(S3) + WILD: assert n_ in base.index, n_
res, rows = {"design": __doc__, "scenarios": {}}, []
for sc, (relab, excl) in SC.items():
    b = base.copy()
    for k, v in relab.items(): b.at[k, "class_code"] = v
    b = b.drop(index=excl); b["y"] = (b.class_code == 1).astype(int)
    for sn, mask in (("main101", np.ones(len(b), bool)), ("temporal_consistent", ((b.year - 2021).abs() <= 1) & (b.stable == "Y"))):
        d = b[np.asarray(mask, bool)]; names = d.index.values; y = d.y.values; n = len(d)
        rng = np.random.default_rng(42); B = [rng.integers(0, n, n) for _ in range(1000)]
        P = {k: v.loc[names, "pred_major"].values.astype(int) for k, v in preds.items()}
        r = {"n_polys": int(n), "n_apple": int(y.sum()), "n_neg": int(n - y.sum())}
        for k, p in P.items():
            bs = np.array([[f1_score(y[i], p[i], zero_division=0), precision_score(y[i], p[i], zero_division=0), recall_score(y[i], p[i], zero_division=0)] for i in B])
            dd = np.array([f1_score(y[i], p[i], zero_division=0) - f1_score(y[i], P["B3_new"][i], zero_division=0) for i in B])
            db0 = np.array([f1_score(y[i], P["B3_new"][i], zero_division=0) - f1_score(y[i], P["B0_new"][i], zero_division=0) for i in B])
            r[k] = {"F1": float(f1_score(y, p)), "P": float(precision_score(y, p, zero_division=0)), "R": float(recall_score(y, p)),
                    "F1_ci": np.quantile(bs[:, 0], [.025, .975]).tolist(), "P_ci": np.quantile(bs[:, 1], [.025, .975]).tolist(), "R_ci": np.quantile(bs[:, 2], [.025, .975]).tolist(),
                    "dF1_vs_B3_new_ci": np.quantile(dd, [.025, .975]).tolist()}
            rows.append({"scenario": sc, "subset": sn, "model": k, "n_polys": n, "n_apple": int(y.sum()), "F1": r[k]["F1"], "F1_lo": r[k]["F1_ci"][0], "F1_hi": r[k]["F1_ci"][1],
                         "P": r[k]["P"], "R": r[k]["R"], "dF1_vs_B3new_lo": r[k]["dF1_vs_B3_new_ci"][0], "dF1_vs_B3new_hi": r[k]["dF1_vs_B3_new_ci"][1]})
        r["B3_minus_B0"] = {"delta": float(f1_score(y, P["B3_new"]) - f1_score(y, P["B0_new"])), "ci": np.quantile(db0, [.025, .975]).tolist()}
        res["scenarios"].setdefault(sc, {})[sn] = r
json.dump(res, open(OUT / "metrics_label_sensitivity.json", "w"), ensure_ascii=False, indent=1, default=float)
t = pd.DataFrame(rows); t.to_csv(OUT / "table_label_sensitivity.csv", index=False)
print(t[t.model.isin(["B3_new", "B0_new", "B6_new", "two_step_new"])].round(3).to_string())
print({sc: {sn: v["B3_minus_B0"] for sn, v in d.items()} for sc, d in res["scenarios"].items()})
