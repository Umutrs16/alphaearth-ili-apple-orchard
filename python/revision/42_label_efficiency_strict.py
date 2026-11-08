# -*- coding: utf-8 -*-
"""v4.1 review item C: STRICT label-efficiency rerun (AOMC weak-label protocol).
Same data, same fixed GroupShuffleSplit spatial-block test set, same fractions / feature sets / seed schedule (seed + s*17)
as 03_label_efficiency.py, but WITHOUT the 20-random-positive top-up. A block subsample is valid only if it contains
both classes (>=1 apple and >=1 non-apple point) and >=10 points; otherwise it is marked invalid and blocks are redrawn
with seed' = seed + s*17 + 10007*k (k = 1, 2, ...), max 500 redraws. Redraws / invalid draws are logged."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/workspace/apple_orchard_project"); sys.path.insert(0, str(ROOT / "python/revision")); sys.path.insert(0, str(ROOT / "python"))
from common import FEATURE_SETS, get_estimator, load_config, load_feature_years, matrix_X
from utils import compute_metrics
from sklearn.model_selection import GroupShuffleSplit
OUT = ROOT / "outputs/revision/v41_review/label_efficiency_strict"; OUT.mkdir(parents=True, exist_ok=True)
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]; SETS = {"S2": "B0", "S1+S2": "B1", "AE": "B3", "AE+S1+S2": "B5"}; NRUN = 5
cfg = load_config(); seed = cfg["training"]["random_seed"]
df = load_feature_years([2019, 2020, 2021])
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(df, df["label"], df["block_id"]))
pool, test = df.iloc[tr].copy(), df.iloc[te].copy(); blocks = pool["block_id"].unique()

def draw(frac, s):
    if frac >= 1: return pool, 0, []
    n_keep = max(1, int(round(len(blocks) * frac))); inval = []
    for k in range(501):
        sd = seed + s * 17 + 10007 * k
        keep = np.random.default_rng(sd).choice(blocks, size=n_keep, replace=False)
        sub = pool[pool.block_id.isin(keep)]
        npos, nneg = int((sub.label == 1).sum()), int((sub.label == 0).sum())
        if npos >= 1 and nneg >= 1 and len(sub) >= 10: return sub, k, inval
        inval.append({"seed": sd, "n": len(sub), "n_pos": npos, "n_neg": nneg})
    raise RuntimeError("no valid draw")

draws = {}; log = []
for frac in FRACTIONS:
    for s in range(NRUN):
        sub, k, inval = draw(frac, s); draws[(frac, s)] = sub
        log.append({"frac": frac, "run": s, "n_redraws": k, "invalid_draws": inval, "n_train": len(sub), "n_pos": int((sub.label == 1).sum()),
                    "n_neg": int((sub.label == 0).sum()), "n_blocks": int(sub.block_id.nunique()), "lt5_pos": bool((sub.label == 1).sum() < 5)})
rows = []; runs = []
for label, bid in SETS.items():
    groups = FEATURE_SETS[bid]; Xte, _ = matrix_X(test, groups, cfg); yte = test.label.values.astype(int)
    for frac in FRACTIONS:
        f1s = []
        for s in range(NRUN):
            sub = draws[(frac, s)]; Xtr, _ = matrix_X(sub, groups, cfg)
            clf = get_estimator("lgbm", seed + s); clf.fit(Xtr, sub.label.values.astype(int))
            m = compute_metrics(yte, clf.predict(Xte)); f1s.append(m["F1"])
            runs.append({"feature_set": label, "bid": bid, "frac": frac, "run": s, "F1": m["F1"], "precision": m.get("precision"), "recall": m.get("recall")})
        rows.append({"feature_set": label, "bid": bid, "frac": frac, "F1_mean": float(np.mean(f1s)), "F1_sd": float(np.std(f1s)),
                     "F1_min": float(np.min(f1s)), "F1_max": float(np.max(f1s)), "n_runs": len(f1s)})
        print(f"{label} {frac:.0%} F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}", flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "label_efficiency_strict.csv", index=False); pd.DataFrame(runs).to_csv(OUT / "label_efficiency_strict_runs.csv", index=False)
L = pd.DataFrame(log); L.drop(columns="invalid_draws").to_csv(OUT / "label_efficiency_strict_draws.csv", index=False)
summ = L.groupby("frac").agg(n_redraws_total=("n_redraws", "sum"), runs_redrawn=("n_redraws", lambda v: int((v > 0).sum())),
        n_train_median=("n_train", "median"), n_train_min=("n_train", "min"), n_train_max=("n_train", "max"),
        n_pos_min=("n_pos", "min"), n_pos_max=("n_pos", "max"), runs_lt5_pos=("lt5_pos", "sum")).reset_index()
summ.to_csv(OUT / "label_efficiency_strict_draw_summary.csv", index=False)
json.dump({"design": __doc__, "pool_rows": len(pool), "pool_blocks": int(len(blocks)), "test_rows": len(test), "test_blocks": int(test.block_id.nunique()),
           "test_pos": int((test.label == 1).sum()), "draw_log": log}, open(OUT / "label_efficiency_strict_log.json", "w"), indent=1, default=int)
print(summ.to_string())
