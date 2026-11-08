# -*- coding: utf-8 -*-
"""v4.3: re-run the label-dependent downstream analyses with the v4.3 main labels by executing string-patched copies of the v4/v4.1 scripts
(inputs = relabeled seed-42 predictions written by 52_relabel_rescore_v43.py; outputs go to outputs/revision/v43/*; originals untouched).
Stages: shortcut, robustness, altitude, temporal, tentative, fig09, fig12."""
import sys, json, types, re
from pathlib import Path
ROOT = Path("/workspace/apple_orchard_project"); REV = ROOT / "python/revision"; O = ROOT / "outputs/revision/v43"; RL = O / "relabel/v43"
RELAB = '{"YL-POLY-0059": 0, "YL-POLY-0060": 0, "YL-POLY-0123": 0, "YL-POLY-0124": 0, "YL-POLY-0125": 0, "YL-POLY-0083": 0, "YL-POLY-0082": 0}'
def run(name, src, argv=None):
    print(f"==== {name}", flush=True); g = {"__name__": "__main__", "__file__": str(REV / name)}
    old = sys.argv; sys.argv = [name] + (argv or [])
    try: exec(compile(src, str(REV / name), "exec"), g)
    finally: sys.argv = old
    return g
def patch(src, pairs):
    for a, b in pairs:
        assert src.count(a) >= 1, (a[:80]); src = src.replace(a, b)
    return src
stages = sys.argv[1].split(",") if len(sys.argv) > 1 else ["shortcut", "robustness", "altitude", "temporal", "tentative", "fig09", "fig12"]
for d in ("shortcut", "robustness", "altitude", "temporal_subset", "figures", "tentative"): (O / d).mkdir(parents=True, exist_ok=True)
if "shortcut" in stages:
    s = (REV / "41_v4_regen_shortcut_visual.py").read_text(encoding="utf-8").split("# ---- visual")[0]
    s = patch(s, [('OUT = ROOT / "outputs/revision/v4_experiments/regen_v4_figs"', f'OUT = Path("{O}/shortcut")'),
                  ('hold["y"] = (hold.class_code == 1).astype(int)', f'RELAB = {RELAB}\nfor _k, _v in RELAB.items(): assert (hold.poly_name == _k).any(); hold.loc[hold.poly_name == _k, "class_code"] = _v\nhold["y"] = (hold.class_code == 1).astype(int)'),
                  ('"metrics_shortcut_diagnostics_v4_expanded.json"', '"metrics_shortcut_diagnostics_v43.json"'), ('"fig10_shortcut_diagnostics_v4_expanded.png"', '"fig10_shortcut_diagnostics_v43.png"')])
    run("41_v4_regen_shortcut_visual.py", s)
if "robustness" in stages:
    s = (REV / "41c_robustness_B3expanded.py").read_text(encoding="utf-8")
    s = patch(s, [('pp = pd.read_csv(E / "pred_points_B3_expanded_seed42.csv")', f'pp = pd.read_csv(Path("{RL}/expanded/pred_points_B3_seed42.csv"))'),
                  ('OUT = ROOT / "outputs/revision/v4_experiments/regen_v4_figs"', f'OUT = Path("{O}/robustness")'), ('"metrics_robustness_B3expanded.json"', '"metrics_robustness_B3expanded_v43.json"')])
    run("41c_robustness_B3expanded.py", s)
if "altitude" in stages:
    s = (REV / "43_altitude_recall.py").read_text(encoding="utf-8")
    s = patch(s, [('OUT = ROOT / "outputs/revision/v41_review/altitude"', f'OUT = Path("{O}/altitude")'), ('FO = ROOT / "outputs/revision/v41_review/figures"', f'FO = Path("{O}/figures")'),
                  ('load(E / "pred_points_B3_expanded_seed42.csv")', f'load(Path("{RL}/expanded/pred_points_B3_seed42.csv"))'), ('load(V4 / "pred_points_B3_seed42.csv")', f'load(Path("{RL}/old38/pred_points_B3_seed42.csv"))'),
                  ('"figS6_prob_vs_elevation_B3expanded_v41.png"', '"figS_prob_vs_elevation_v43.png"')])
    run("43_altitude_recall.py", s)
if "temporal" in stages:
    s = (REV / "45b_temporal_subset_eval.py").read_text(encoding="utf-8")
    s = patch(s, [('MODELS = {"B3_new": E / "pred_polys_B3_expanded_seed42.csv", "B0_new": E / "pred_polys_B0_expanded_seed42.csv", "B6_new": E / "pred_polys_B6_expanded_seed42.csv",\n          "two_step_new": E / "pred_polys_HIER_expanded_seed42.csv", "B3_old": V4 / "pred_polys_B3_seed42.csv"}',
                   f'RLX = Path("{RL}"); MODELS = {{"B3_new": RLX / "expanded/pred_polys_B3_seed42.csv", "B0_new": RLX / "expanded/pred_polys_B0_seed42.csv", "B6_new": RLX / "expanded/pred_polys_B6_seed42.csv", "two_step_new": RLX / "expanded/pred_polys_HIER_seed42.csv", "B3_old": RLX / "old38/pred_polys_B3_seed42.csv"}}')])
    run("45b_temporal_subset_eval.py", s, ["/workspace/uploads/user_return_0928/holdout_temporal_check_prefilled.xlsx", str(O / "temporal_subset")])
if "tentative" in stages:
    s = (REV / "49_label_sensitivity_v42.py").read_text(encoding="utf-8")
    s = patch(s, [('OUT = ROOT / "outputs/revision/v42/label_sensitivity"', f'OUT = Path("{O}/tentative")'),
                  ('MODELS = {"B3_new": E / "pred_polys_B3_expanded_seed42.csv", "B0_new": E / "pred_polys_B0_expanded_seed42.csv", "B6_new": E / "pred_polys_B6_expanded_seed42.csv",\n          "two_step_new": E / "pred_polys_HIER_expanded_seed42.csv", "B3_old": V4 / "pred_polys_B3_seed42.csv"}',
                   f'RLX = Path("{RL}"); MODELS = {{"B3_new": RLX / "expanded/pred_polys_B3_seed42.csv", "B0_new": RLX / "expanded/pred_polys_B0_seed42.csv", "B6_new": RLX / "expanded/pred_polys_B6_seed42.csv", "two_step_new": RLX / "expanded/pred_polys_HIER_seed42.csv", "B3_old": RLX / "old38/pred_polys_B3_seed42.csv"}}'),
                  ('SC = {"S0_original": ({}, []), "S1_explicit_corrections": (S1, []), "S2_S1_excl_wild_apple_forest": (S1, WILD), "S3_S2_plus_tentative": (S3, WILD)}',
                   'T4 = {"YL-POLY-0126": 1, "YL-POLY-0138": 2, "YL-POLY-0081": 0, "YL-POLY-0142": 0}\nSC = {"V43_main": ({}, []), "V43_plus_tentative": (T4, [])}'),
                  ('for n_ in list(S3) + WILD: assert n_ in base.index, n_', 'for n_ in list(S3) + WILD: assert n_ in base.index, n_\nassert all(base.loc[w, "class_code"] == 0 for w in WILD + ["YL-POLY-0083", "YL-POLY-0082"]), "inputs must carry v4.3 labels"')])
    run("49_label_sensitivity_v42.py", s, ["/workspace/uploads/user_return_0928/holdout_temporal_check_prefilled.xlsx"])
if "fig09" in stages:
    s = (REV / "40c2_fig09_v41_cases.py").read_text(encoding="utf-8")
    s = patch(s, [('("FN-高海拔", "高海拔苹果园漏报", p[(p.class_code == 1) & (p.out == "FN") & (p.elev >= 1200)])',
                   '("TN-野苹果林", "野苹果林（改标为非果园）正确拒识", p[p.poly_name.isin(["YL-POLY-0059", "YL-POLY-0060", "YL-POLY-0123", "YL-POLY-0124", "YL-POLY-0125"]) & (p.out == "TN")])'),
                  ('("FN-低海拔", "低海拔苹果园漏报", p[(p.class_code == 1) & (p.out == "FN") & (p.elev < 900)])', '("FN", "苹果园漏报", p[(p.class_code == 1) & (p.out == "FN")])'),
                  ('pt = pd.read_csv(R / f"pred_polys_{MODEL}_expanded_seed42.csv")', f'pt = pd.read_csv(Path("{RL}/expanded") / f"pred_polys_{{MODEL}}_seed42.csv")'),
                  ('pts = pd.read_csv(R / f"pred_points_{MODEL}_expanded_seed42.csv")', f'pts = pd.read_csv(Path("{RL}/expanded") / f"pred_points_{{MODEL}}_seed42.csv")'),
                  ('man = man[man.split.isin(["train", "val_holdout"])]', f'man = man[man.split.isin(["train", "val_holdout"])].copy()\nfor _k, _v in {RELAB}.items(): man.loc[man.name == _k, "class_code"] = _v'),
                  ("f'Path(\"{R}/metrics_poly_pr_{MODEL}_expanded.json\")'", f"f'Path(\"{O}/figures/metrics_poly_pr_{{MODEL}}_v43.json\")'"),
                  ('Path("/workspace/apple_orchard_project/outputs/revision/v41_review/figures/fig09_main_holdout_validation_v41_expandedB3.png")', f'Path("{O}/figures/fig09_challenge_v43.png")'),
                  ('"v4_experiments"; V4O = FIG3\'', f'"v4_experiments"; V4O = Path("{RL}/old38")\'')])
    s = s.replace("json.dump(met, open(TMP", f"json.dump(prc, open(Path('{O}/figures/metrics_poly_pr_B3_v43.json'), 'w'), ensure_ascii=False, indent=1)\njson.dump(met, open(TMP", 1)
    run("40c2_fig09_v41_cases.py", s, ["B3"])
if "fig12" in stages:
    M = json.load(open(O / "relabel/metrics_relabel_v43.json"))["results"]["v43"]
    for b, m in M["expanded"].items(): m["paired_dF1_vs_B3_old_seed42"] = m["paired_dF1_vs_old38_B3"]
    json.dump({"models": M["old38"]}, open(O / "relabel/_compat_old38_v43.json", "w")); json.dump({"models": M["expanded"]}, open(O / "relabel/_compat_expanded_v43.json", "w"))
    na = M["expanded"]["B3"]["seed42"]["poly"]["n_pos"]; pc = M["expanded"]["B3"]["seed42"]["poly"]["per_class"]
    s = (REV / "40b2_plot_expanded_compare_fig12.py").read_text(encoding="utf-8")
    s = patch(s, [('old = json.load(open(V4 / "metrics_indep_feature_compare_v4.json"))["models"]', f'old = json.load(open(Path("{O}/relabel/_compat_old38_v43.json")))["models"]'),
                  ('new = json.load(open(R / "metrics_expanded_retrain.json"))', f'new = json.load(open(Path("{O}/relabel/_compat_expanded_v43.json")))'),
                  ('ax.axvline(2 * 36 / (36 + 101), color="k", ls=":", lw=1); ax.text(2 * 36 / 137, 9.25', f'ax.axvline(2 * {na} / ({na} + 101), color="k", ls=":", lw=1); ax.text(2 * {na} / ({na} + 101), 9.25'),
                  ('"苹果园\\n召回\\n(n=36)"', f'"苹果园\\n召回\\n(n={pc["apple_recall"]["n"]})"'), ('"其他果园\\n拒判\\n(n=10)"', f'"其他果园\\n拒判\\n(n={pc["other_orchard_rejected"]["n"]})"'),
                  ('"非果园\\n拒判\\n(n=55)"', f'"非果园\\n拒判\\n(n={pc["non_orchard_rejected"]["n"]})"'), ('(a) 巩留留出面级 F1（n=101）', '(a) 巩留挑战集面级 F1（n=101，v4.3标签）'), ('ax.set_xlim(0.0, 0.88)', 'ax.set_xlim(0.0, 1.0)'),
                  ('fig.savefig(R / "fig12_expanded_retrain_compare_v4.png"', f'fig.savefig(Path("{O}/figures/fig12_training_expansion_v43.png")')])
    run("40b2_plot_expanded_compare_fig12.py", s)
