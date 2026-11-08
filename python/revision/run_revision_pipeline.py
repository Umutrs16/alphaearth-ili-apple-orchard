# -*- coding: utf-8 -*-
"""Run revision experiment pipeline in order."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/workspace/venv/bin/python"

STEPS = [
    ("01_ablation_b0b6.py", True),
    ("02_classifier_compare.py", True),
    ("03_label_efficiency.py", True),
    ("04_spatial_cv.py", True),
    ("05_calibration.py", True),
    ("06_group_importance.py", True),
    ("07_scale_consistency.py", True),
    ("08_temporal_stability.py", True),
    ("09_independent_val.py", False),  # GEE
    ("11_sample_2024.py", False),  # GEE optional
    ("10_maps_preview.py", False),  # GEE
    ("12_make_figures.py", True),
    ("13_write_paper.py", True),
]


def run_one(script: str) -> int:
    print("\n" + "=" * 60)
    print("RUN", script)
    print("=" * 60)
    t0 = time.time()
    r = subprocess.run([PY, str(HERE / script)], cwd=str(HERE))
    print(f"DONE {script} rc={r.returncode} in {time.time() - t0:.1f}s")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true", help="Skip GEE-dependent steps")
    ap.add_argument("--gee-only", action="store_true", help="Only GEE-dependent steps")
    ap.add_argument("--from", dest="from_script", default=None)
    args = ap.parse_args()

    started = args.from_script is None
    for script, is_local in STEPS:
        if not started:
            if script == args.from_script:
                started = True
            else:
                continue
        if args.local_only and not is_local:
            print("SKIP (local-only)", script)
            continue
        if args.gee_only and is_local:
            print("SKIP (gee-only)", script)
            continue
        rc = run_one(script)
        if rc != 0 and is_local:
            print("Local step failed; abort", script)
            sys.exit(rc)
        if rc != 0:
            print("Non-local step failed; continue", script)


if __name__ == "__main__":
    main()
