# -*- coding: utf-8 -*-
"""v4 retrain: combined table old (38 polygons) vs expanded (277) vs expanded sensitivity (img year 2020-2022 only) on the Gongliu holdout."""
import json
from pathlib import Path
import numpy as np, pandas as pd
V4 = Path("/workspace/apple_orchard_project/outputs/revision/v4_experiments")
old = json.load(open(V4 / "metrics_indep_feature_compare_v4.json"))["models"]
new = json.load(open(V4 / "retrain_expanded/metrics_expanded_retrain.json"))["models"]
sen = json.load(open(V4 / "retrain_expanded_sens2020_2022/metrics_expanded_retrain.json"))["models"]
oldB3 = np.array(old["B3"]["per_seed_poly_F1"]); rows = []
f = lambda d: f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
for tag, M in (("old_38poly", old), ("expanded_277poly", new), ("expanded_sens_img2020_2022", sen)):
    for b, m in M.items():
        p = m["seed42"]["poly"]; ci = p["bootstrap_95ci"]; pc = p["per_class"]; pf = np.array(m["per_seed_poly_F1"]) - oldB3
        rows.append({"training": tag, "model": b, "poly_F1_seed42": round(p["F1"], 3), "F1_95CI": f"[{ci['F1'][0]:.3f}, {ci['F1'][1]:.3f}]",
                     "P": round(p["precision"], 3), "R": round(p["recall"], 3), "MCC": round(p["MCC"], 3), "OA": round(p["OA"], 3), "BA": round(p["balanced_accuracy"], 3),
                     "AP_poly": round(p["AP_poly_meanprob"], 3), "AP_poly_95CI": f"[{ci['AP_poly_meanprob'][0]:.3f}, {ci['AP_poly_meanprob'][1]:.3f}]",
                     "AP_point": round(m["seed42"]["point"]["AP"], 3),
                     "TP/FP/FN/TN": "/".join(str(p["confusion_matrix"][k]) for k in ("TP", "FP", "FN", "TN")),
                     "other_orch_rej": f"{pc['other_orchard_rejected']['k']}/{pc['other_orchard_rejected']['n']}", "non_orch_rej": f"{pc['non_orchard_rejected']['k']}/{pc['non_orchard_rejected']['n']}",
                     "dF1_vs_oldB3_seed42 [paired CI]": ("—" if (tag == "old_38poly" and b == "B3") else f(m["paired_dF1_vs_B3_old_seed42"]) if "paired_dF1_vs_B3_old_seed42" in m else f(m["paired_delta_F1_vs_B3_seed42"])),
                     "dF1_vs_sameset_B3_seed42 [paired CI]": "—" if b == "B3" else f(m.get("paired_dF1_vs_B3_seed42") or m.get("paired_delta_F1_vs_B3_seed42")),
                     "F1_10seeds": f"{np.mean(m['per_seed_poly_F1']):.3f}±{np.std(m['per_seed_poly_F1'], ddof=1):.3f}",
                     "dF1_vs_oldB3_10seeds": "—" if (tag == "old_38poly" and b == "B3") else f"{pf.mean():+.3f}±{pf.std(ddof=1):.3f}"})
t = pd.DataFrame(rows); t.to_csv(V4 / "retrain_expanded/table_old_vs_expanded_v4.csv", index=False, encoding="utf-8-sig"); print(t.to_string())
