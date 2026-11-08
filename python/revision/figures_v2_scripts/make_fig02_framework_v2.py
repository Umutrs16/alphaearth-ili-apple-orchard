#!/usr/bin/env python3
"""Figure 2 v2: five-column methods framework (journal quality)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

OUT = Path("/workspace/apple_orchard_project/outputs/revision/figures_v2")
OUT.mkdir(parents=True, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

# Column color groups
COLS = [
    {
        "title": "① 数据",
        "subtitle": "Data",
        "color": "#2563eb",  # blue
        "light": "#dbeafe",
        "items": [
            "AlphaEarth 64-D\n年度嵌入",
            "Sentinel-2\n物候/光谱",
            "Sentinel-1\n雷达特征",
            "SRTM DEM\n地形因子",
            "AOMC 30 m\n弱先验",
        ],
    },
    {
        "title": "② 弱标签构建",
        "subtitle": "Weak labels",
        "color": "#7c3aed",  # violet
        "light": "#ede9fe",
        "items": [
            "AOMC 核心腐蚀\n开运算纯化",
            "边界缓冲\n硬负样本",
            "空间去重\n块ID划分",
            "训练/留出\n协议隔离",
        ],
    },
    {
        "title": "③ 特征与模型",
        "subtitle": "Features / models",
        "color": "#059669",  # emerald
        "light": "#d1fae5",
        "items": [
            "B0–B6\n消融组合",
            "RF / XGB\n/ LightGBM",
            "标签效率\n曲线",
            "特征组\n排列重要性",
        ],
    },
    {
        "title": "④ 稳健验证",
        "subtitle": "Robust validation",
        "color": "#d97706",  # amber
        "light": "#fef3c7",
        "items": [
            "多尺度\n空间块CV",
            "块自助法\n置信区间",
            "跨年特征\n时间稳定性",
            "概率校准\nIsotonic/Platt",
        ],
    },
    {
        "title": "⑤ 独立验证与产出",
        "subtitle": "Independent val. & outputs",
        "color": "#dc2626",  # red
        "light": "#fee2e2",
        "items": [
            "★ 园块空间Holdout\n（主独立验证）",
            "目视随机点\n敏感性分析",
            "概率/分类图\n多年度预览",
            "不确定性\n与适用边界",
        ],
    },
]


def rounded_box(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.02, z=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.5, zorder=z,
    )
    ax.add_patch(box)
    return box


def main():
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(COLS)
    margin_x = 0.025
    gap = 0.018
    col_w = (1 - 2 * margin_x - (n - 1) * gap) / n
    top = 0.92
    header_h = 0.12
    body_top = top - header_h - 0.02
    body_bottom = 0.10
    item_gap = 0.012

    # Title
    ax.text(0.5, 0.975, "图2  技术流程框架：数据 → 弱标签 → 模型 → 稳健验证 → 独立验证与产出",
            ha="center", va="top", fontsize=12, fontweight="bold", transform=ax.transAxes)

    for i, col in enumerate(COLS):
        x0 = margin_x + i * (col_w + gap)
        # column background panel
        rounded_box(ax, x0, body_bottom - 0.02, col_w, top - body_bottom + 0.02,
                    fc=col["light"], ec=col["color"], lw=1.4, radius=0.018, z=1)
        # header bar
        rounded_box(ax, x0 + 0.008, top - header_h, col_w - 0.016, header_h - 0.01,
                    fc=col["color"], ec=col["color"], lw=0, radius=0.012, z=3)
        ax.text(x0 + col_w / 2, top - 0.035, col["title"], ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)
        ax.text(x0 + col_w / 2, top - 0.075, col["subtitle"], ha="center", va="center",
                fontsize=7, color="#f8fafc", style="italic", zorder=4)

        items = col["items"]
        avail_h = body_top - body_bottom - 0.02
        item_h = (avail_h - (len(items) - 1) * item_gap) / len(items)
        for j, text in enumerate(items):
            y = body_top - (j + 1) * item_h - j * item_gap
            # primary emphasis for holdout item
            is_primary = "★" in text or "主独立" in text
            fc = "#fff7ed" if is_primary else "white"
            ec = col["color"] if not is_primary else "#c2410c"
            lw = 1.6 if is_primary else 0.9
            rounded_box(ax, x0 + 0.012, y, col_w - 0.024, item_h,
                        fc=fc, ec=ec, lw=lw, radius=0.01, z=3)
            ax.text(
                x0 + col_w / 2, y + item_h / 2, text.replace("★ ", ""),
                ha="center", va="center", fontsize=7.4 if not is_primary else 7.8,
                fontweight="bold" if is_primary else "normal",
                color="#9a3412" if is_primary else "#1f2937",
                linespacing=1.25, zorder=4,
            )

        # arrow to next column
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x0 + col_w + gap * 0.15, 0.52),
                xytext=(x0 + col_w - gap * 0.05, 0.52),
                arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.8,
                                mutation_scale=14),
                zorder=5,
            )

    # footer note
    ax.text(
        0.5, 0.035,
        "主独立证据：巩留园块空间 Holdout　|　辅：600点目视敏感性　|　AOMC 仅作弱标签，不作独立真值",
        ha="center", va="center", fontsize=7.5, color="#475569",
    )

    out_png = OUT / "fig02_framework_v2.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    fig.savefig(OUT / "fig02_framework_v2.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print("Wrote", out_png, "size", out_png.stat().st_size)


if __name__ == "__main__":
    main()
