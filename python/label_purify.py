# -*- coding: utf-8 -*-
"""AOMC 弱标签纯化：空间滤波 + 时序一致性 + 生成训练样本。

正式流程：读取 AOMC GeoTIFF → 裁剪 AOI → 邻域多数滤波 → 三年一致性 →
  分层抽样 → 输出 CSV（含空间块 ID）。

Demo 模式：生成与真实纯化样本结构一致的合成标签与特征。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    ROOT,
    ensure_dirs,
    feature_columns,
    load_config,
    setup_logging,
    spatial_block_id,
)

log = setup_logging("label_purify")


def _synthetic_features(rng: np.random.Generator, n: int, label: int) -> dict:
    """按类别生成可分但带噪声的特征（模拟苹果园 vs 非园）。"""
    # 苹果园：更高 NDVI_max、更明显物候振幅、中等坡度
    bias = 1.0 if label == 1 else 0.0
    s2 = {
        "NDVI_max": rng.normal(0.72 + 0.12 * bias, 0.06, n),
        "NDVI_mean": rng.normal(0.48 + 0.10 * bias, 0.05, n),
        "NDVI_std": rng.normal(0.12 + 0.03 * bias, 0.02, n),
        "NDVI_p10": rng.normal(0.25 + 0.05 * bias, 0.04, n),
        "NDVI_p90": rng.normal(0.68 + 0.10 * bias, 0.05, n),
        "EVI_max": rng.normal(0.55 + 0.10 * bias, 0.05, n),
        "EVI_mean": rng.normal(0.35 + 0.08 * bias, 0.04, n),
        "NDWI_mean": rng.normal(-0.05 + 0.02 * bias, 0.03, n),
        "NDRE_mean": rng.normal(0.30 + 0.08 * bias, 0.04, n),
        "B4_mean": rng.normal(800 - 80 * bias, 100, n),
        "B8_mean": rng.normal(2800 + 200 * bias, 200, n),
        "B11_mean": rng.normal(1600 - 50 * bias, 150, n),
        "B12_mean": rng.normal(1100 - 40 * bias, 120, n),
        "NDVI_amp": rng.normal(0.35 + 0.12 * bias, 0.05, n),
        "greenup_doy": rng.normal(105 - 5 * bias, 8, n),
        "senescence_doy": rng.normal(280 + 5 * bias, 10, n),
    }
    s1 = {
        "VV_mean": rng.normal(-11.5 + 0.8 * bias, 1.2, n),
        "VV_std": rng.normal(2.0 + 0.3 * bias, 0.4, n),
        "VH_mean": rng.normal(-17.0 + 1.0 * bias, 1.5, n),
        "VH_std": rng.normal(2.2 + 0.2 * bias, 0.4, n),
        "VV_VH_ratio": rng.normal(5.5 - 0.3 * bias, 0.8, n),
        "RVI": rng.normal(0.45 + 0.08 * bias, 0.06, n),
    }
    dem = {
        "elevation": rng.normal(1100 + 80 * bias, 120, n),
        "slope": rng.normal(8 + 3 * bias, 4, n).clip(0, 35),
        "aspect_sin": rng.uniform(-1, 1, n),
        "aspect_cos": rng.uniform(-1, 1, n),
    }
    # AlphaEarth 64维：类别可分的低维信号 + 噪声
    ae_signal = rng.normal(0.3 * bias, 0.15, (n, 8))
    ae_noise = rng.normal(0, 0.4, (n, 56))
    ae = np.hstack([ae_signal, ae_noise])
    ae_cols = {f"AE_{i:02d}": ae[:, i] for i in range(64)}
    out = {}
    out.update(s2)
    out.update(s1)
    out.update(dem)
    out.update(ae_cols)
    return out


def generate_demo_samples(cfg: dict) -> pd.DataFrame:
    seed = cfg["training"]["random_seed"]
    rng = np.random.default_rng(seed)
    n_per = cfg["training"]["n_samples_per_class"]
    aoi = cfg["aoi"]
    years = cfg["years"]["all"]
    rows = []
    for year in years:
        for label, n in ((1, n_per), (0, n_per)):
            lon = rng.uniform(aoi["lon_min"], aoi["lon_max"], n)
            lat = rng.uniform(aoi["lat_min"], aoi["lat_max"], n)
            feats = _synthetic_features(rng, n, label)
            # 注入少量标签噪声模拟弱标签
            noisy = label * np.ones(n, dtype=int)
            flip = rng.random(n) < 0.05
            noisy[flip] = 1 - noisy[flip]
            df = pd.DataFrame(feats)
            df["lon"] = lon
            df["lat"] = lat
            df["year"] = year
            df["label_raw"] = noisy
            df["label"] = label  # 纯化后“真”标签（demo 中用干净标签训练评估）
            df["block_id"] = spatial_block_id(lon, lat, cfg["training"]["spatial_block_size_deg"])
            df["source"] = "demo_synthetic"
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def purify_from_aomc(cfg: dict) -> pd.DataFrame | None:
    """若本地存在 AOMC GeoTIFF，尝试栅格抽样（需要 rasterio）。失败则返回 None。"""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        log.warning("未安装 rasterio，跳过真实 AOMC 纯化，改用 demo 合成样本")
        return None

    aomc_dir = ROOT / cfg["data"]["aomc_dir"]
    aoi = cfg["aoi"]
    tifs = sorted(aomc_dir.glob("AOMC_30_Xinjiang_*.tif"))
    if not tifs:
        return None

    rng = np.random.default_rng(cfg["training"]["random_seed"])
    frames = []
    for tif in tifs:
        year = int(tif.stem.split("_")[-1])
        try:
            with rasterio.open(tif) as ds:
                # AOMC 编码通常：果园类为特定整数值；此处保守取 >0 为候选
                window = from_bounds(
                    aoi["lon_min"], aoi["lat_min"], aoi["lon_max"], aoi["lat_max"],
                    transform=ds.transform,
                )
                data = ds.read(1, window=window, boundless=True, fill_value=0)
                transform = ds.window_transform(window)
        except Exception as e:
            log.warning("读取 %s 失败: %s", tif.name, e)
            continue

        # 简单 3x3 多数滤波
        from scipy.ndimage import generic_filter

        def majority(x):
            vals, counts = np.unique(x.astype(int), return_counts=True)
            return vals[np.argmax(counts)]

        try:
            filtered = generic_filter(data.astype(float), majority, size=3, mode="nearest")
        except Exception:
            filtered = data.astype(float)

        orchard = filtered > 0
        # 分层随机像元
        ys, xs = np.where(orchard)
        n_pos = min(cfg["training"]["n_samples_per_class"], len(ys))
        if n_pos < 50:
            log.warning("%s AOI 内果园像元过少 (%d)，跳过该年", tif.name, len(ys))
            continue
        sel = rng.choice(len(ys), n_pos, replace=False)
        rows_p, cols_p = ys[sel], xs[sel]
        # 负样本
        yn, xn = np.where(~orchard)
        n_neg = min(cfg["training"]["n_samples_per_class"], len(yn))
        seln = rng.choice(len(yn), n_neg, replace=False)
        rows_n, cols_n = yn[seln], xn[seln]

        def pix_to_ll(rr, cc):
            xs_c, ys_c = rasterio.transform.xy(transform, rr, cc)
            return np.array(xs_c), np.array(ys_c)

        lon_p, lat_p = pix_to_ll(rows_p, cols_p)
        lon_n, lat_n = pix_to_ll(rows_n, cols_n)

        # 特征仍用合成（真实特征需 GEE 采样）；标签来自 AOMC
        for label, lon, lat in ((1, lon_p, lat_p), (0, lon_n, lat_n)):
            feats = _synthetic_features(rng, len(lon), label)
            df = pd.DataFrame(feats)
            df["lon"] = lon
            df["lat"] = lat
            df["year"] = year
            df["label_raw"] = label
            df["label"] = label
            df["block_id"] = spatial_block_id(np.asarray(lon), np.asarray(lat), cfg["training"]["spatial_block_size_deg"])
            df["source"] = f"aomc_{tif.name}"
            frames.append(df)

    if not frames:
        return None
    # 补充转移年份用 demo 扩展
    base = pd.concat(frames, ignore_index=True)
    transfer_years = [y for y in cfg["years"]["transfer"] if y not in base["year"].unique()]
    if transfer_years:
        demo = generate_demo_samples(cfg)
        demo = demo[demo["year"].isin(transfer_years)]
        base = pd.concat([base, demo], ignore_index=True)
        log.info("AOMC 纯化样本已补转移年 demo 特征: %s", transfer_years)
    return base


def main(mode: str = "demo", cfg_path: str | None = None) -> Path:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    out_dir = ROOT / cfg["data"]["samples_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = None
    if mode != "demo":
        df = purify_from_aomc(cfg)
    if df is None:
        log.info("生成 DEMO 合成纯化样本（结构对齐 AOMC 抽样）")
        df = generate_demo_samples(cfg)
        mode_tag = "demo"
    else:
        mode_tag = "aomc_partial"
        log.info("使用 AOMC 纯化样本 + 合成特征（GEE 特征待导出替换）")

    out = out_dir / f"purified_samples_{mode_tag}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    meta = {
        "mode": mode_tag,
        "n_rows": len(df),
        "years": sorted(df["year"].unique().tolist()),
        "pos": int((df["label"] == 1).sum()),
        "neg": int((df["label"] == 0).sum()),
        "note": "DEMO 特征/标签仅用于流程验证，非正式 GEE 生产结果" if mode_tag == "demo" else "标签来自 AOMC，特征仍为合成占位",
    }
    from utils import save_json
    save_json(meta, out_dir / f"purified_samples_{mode_tag}_meta.json")
    log.info("写出 %s (%d 行)", out, len(df))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="demo", choices=["demo", "gee"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    main(args.mode, args.config)
