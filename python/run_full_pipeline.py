# -*- coding: utf-8 -*-
"""一键运行完整流水线：纯化 → 训练 → 迁移 → 面积不确定性 → 制图。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_dirs, load_config, setup_logging

log = setup_logging("pipeline")


def main():
    ap = argparse.ArgumentParser(description="AlphaEarth Yili Apple Orchard full pipeline")
    ap.add_argument("--mode", default="demo", choices=["demo", "gee"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-download", action="store_true", help="跳过 AOMC 下载")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    t0 = time.time()
    mode = args.mode
    log.info("==== 启动流水线 mode=%s ====", mode)

    if not args.skip_download:
        try:
            from aomc_download import main as dl_main
            dl_main(args.config)
        except Exception as e:
            log.warning("AOMC 下载阶段异常（继续）: %s", e)

    from label_purify import main as purify_main
    purify_main(mode, args.config)

    from train_models import main as train_main
    train_main(mode, args.config)

    from transfer_eval import main as transfer_main
    transfer_main(mode, args.config)

    from scale_uncertainty import main as scale_main
    scale_main(mode, args.config)

    from make_figures import main as fig_main
    figs = fig_main(mode, args.config)

    dt = time.time() - t0
    log.info("==== 完成 %.1fs | figures=%d ====", dt, len(figs))
    print(f"SUCCESS mode={mode} figures={len(figs)} elapsed_s={dt:.1f}")


if __name__ == "__main__":
    main()
