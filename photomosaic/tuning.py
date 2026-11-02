"""主图特征分析与参数智能推荐：按内容自动适配，不是统一模板。

特征来源：尺寸、细节能量分布（DetailMap）、显著性主体占比（SaliencyMap）、
色彩丰富度（直方图熵）、亮度分布。recommend_params 把特征翻译成推荐参数，
批量出图时逐张主图自动适配（可手动覆盖）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

from .render import DetailMap


def analyze_master(master: Image.Image) -> Dict[str, float]:
    """分析主图特征，返回指标字典（供 recommend_params 与诊断展示）。"""
    W, H = master.size
    rgb = np.asarray(master.convert("RGB"), dtype=np.float32)

    # 尺寸与宽高比
    aspect = W / max(1, H)

    # 细节能量：网格逐格能量的均值与 90 分位（DetailMap 是梯度+对比度）
    dm = DetailMap(master)
    cw, ch = W / 16, H / 16
    energies = np.array([dm.energy((int(gx * cw), int(gy * ch),
                                    int((gx + 1) * cw), int((gy + 1) * ch)))
                         for gy in range(16) for gx in range(16)])
    detail_mean = float(energies.mean())
    detail_p90 = float(np.percentile(energies, 90))

    # 显著性：主体占比（显著性 > 0.5 的像素占比）+ 显著性峰值
    from .saliency import saliency_map
    sal_map = saliency_map(master, 256)
    subject_ratio = float((sal_map > 0.5).mean())       # 主体像素占比
    saliency_peak = float(sal_map.max())

    # 色彩丰富度：RGB 直方图联合熵（每通道 8 bin → 512 桶）
    q = (rgb // 32).astype(np.int32)
    idx = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=512).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    color_entropy = float(-np.sum(p * np.log2(p)))

    # 亮度分布
    lum = rgb.mean(axis=2)
    bright_ratio = float((lum > 235).mean())            # 近白占比
    dark_ratio = float((lum < 20).mean())               # 近黑占比

    return {
        "width": W, "height": H, "aspect": aspect,
        "detail_mean": detail_mean, "detail_p90": detail_p90,
        "subject_ratio": subject_ratio, "saliency_peak": saliency_peak,
        "color_entropy": color_entropy,
        "bright_ratio": bright_ratio, "dark_ratio": dark_ratio,
    }


def recommend_params(master: Image.Image, style: str = "grid") -> Dict[str, object]:
    """按主图特征给出推荐参数（智能适配，非统一模板）。

    返回可直接传给 make_mosaic 的关键参数 + 推荐理由（GUI 展示用）。
    """
    f = analyze_master(master)
    reasons = []

    # 细节信号选择：主体突出（有明确主旨）→ 显著性；否则梯度能量
    if f["subject_ratio"] > 0.03 and f["saliency_peak"] > 0.7:
        signal = "saliency"
        reasons.append(f"主体占比 {f['subject_ratio']:.0%}：用显著性盯主旨")
    else:
        signal = "detail"
        reasons.append("无明显单一主体：用梯度能量抓纹理")

    # 格子粗细：细节多 → 更多格子；大图 → 更多格子
    base_cells = max(8, round((300.0 * f["aspect"]) ** 0.5))
    if f["detail_p90"] > 40:
        cells_x = int(base_cells * 1.5)
        reasons.append(f"细节丰富（p90={f['detail_p90']:.0f}）：格子加密")
    elif f["detail_p90"] < 15:
        cells_x = max(8, int(base_cells * 0.7))
        reasons.append(f"内容平滑（p90={f['detail_p90']:.0f}）：格子放粗")
    else:
        cells_x = base_cells

    # 细分层级：细节多 → 层级深
    if f["detail_p90"] > 45:
        levels = 5
    elif f["detail_p90"] > 25:
        levels = 4
    else:
        levels = 3
    reasons.append(f"细分 {levels} 级")

    # 细节阈值：细节分布越集中阈值越低（更敏感）
    detail_threshold = 18.0 if f["detail_mean"] > 20 else 25.0

    # 融合强度：色彩单调（熵低）→ 强融合补色；色彩丰富 → 弱融合保真
    if f["color_entropy"] < 4.5:
        blend = 0.7
        reasons.append(f"色彩偏单调（熵 {f['color_entropy']:.1f}）：融合加强到 70%")
    elif f["color_entropy"] > 7.0:
        blend = 0.4
        reasons.append(f"色彩丰富（熵 {f['color_entropy']:.1f}）：融合减到 40%")
    else:
        blend = 0.55

    # top_k：细节多则变化多
    top_k = 12 if f["detail_mean"] > 25 else 8

    # 近白/近黑瓦片跳过：主图本身偏亮/偏暗时对应排除
    skip_white = 245 if f["bright_ratio"] > 0.25 else None
    skip_black = 10 if f["dark_ratio"] > 0.25 else None
    if skip_white:
        reasons.append(f"主图偏亮（{f['bright_ratio']:.0%} 近白）：跳过近白瓦片")
    if skip_black:
        reasons.append(f"主图偏暗（{f['dark_ratio']:.0%} 近黑）：跳过近黑瓦片")

    return {
        "cells_x": cells_x,
        "levels": levels,
        "detail_threshold": detail_threshold,
        "top_k": top_k,
        "blend": blend,
        "signal": signal,
        "skip_near_white": skip_white,
        "skip_near_black": skip_black,
        "style": style,
        "_features": f,                  # 诊断展示用
        "reasons": reasons,
    }
