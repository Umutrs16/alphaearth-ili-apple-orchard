# -*- coding: utf-8 -*-
"""从 Figshare 下载 AOMC 新疆分幅（Xinjiang tiles）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, ensure_dirs, load_config, setup_logging

log = setup_logging("aomc_download")


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log.info("已存在，跳过: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    log.info("下载 %s -> %s", url, dest)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
                for block in r.iter_content(chunk_size=chunk):
                    if block:
                        f.write(block)
                        bar.update(len(block))
        return True
    except Exception as e:
        log.error("下载失败 %s: %s", url, e)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def main(cfg_path: str | None = None) -> int:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    out_dir = ROOT / cfg["data"]["aomc_dir"]
    urls = cfg["data"]["aomc_figshare"]
    ok = True
    for year in ("2019", "2020", "2021"):
        url = urls.get(year)
        if not url:
            continue
        dest = out_dir / f"AOMC_30_Xinjiang_{year}.tif"
        if not download_file(url, dest):
            ok = False
    if not ok:
        log.warning(
            "部分 AOMC 下载失败。请手动访问 Figshare 文章 %s，下载新疆分幅至 %s",
            urls.get("article"),
            out_dir,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    raise SystemExit(main(args.config))
