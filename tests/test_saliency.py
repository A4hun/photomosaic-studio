"""显著性检测：盯住主体（人/物），不被平滑背景和平凡纹理带偏。"""

import numpy as np
import pytest
from PIL import Image

from photomosaic.saliency import SaliencyMap, saliency_map


def test_saliency_map_range_and_shape():
    img = Image.fromarray(np.random.default_rng(0).integers(
        0, 255, (300, 400, 3), dtype=np.uint8))
    sal = saliency_map(img, 256)
    assert sal.shape[0] > 0 and sal.shape[1] > 0
    assert 0.0 <= sal.min() and sal.max() <= 1.5


def test_saliency_highlights_subject_over_background():
    """左侧高对比主体 vs 右侧平滑背景：主体区显著性必须明显更高。"""
    a = np.full((200, 400, 3), 180, np.uint8)     # 平滑亮背景
    # 左侧放一个高对比主体（暗底亮圆）
    a[:, :200] = 30
    yy, xx = np.ogrid[:200, :200]
    circle = ((yy - 100) ** 2 + (xx - 100) ** 2) < 2500
    a[:, :200][circle] = 240
    img = Image.fromarray(a)
    sm = SaliencyMap(img)
    subject = sm.energy((20, 50, 180, 150))       # 左：主体
    background = sm.energy((220, 50, 380, 150))   # 右：平滑背景
    assert subject > background * 2, f"主体 {subject:.1f} 应远高于背景 {background:.1f}"


def test_saliency_drives_subdivision_toward_subject():
    """显著性驱动细分：主体区被细分，平滑背景区保持大块。"""
    from photomosaic.render import adaptive_leaves, uniform_base_boxes
    a = np.full((256, 256, 3), 200, np.uint8)
    yy, xx = np.ogrid[:128, :128]
    # 左上象限放主体（亮圆），其余平滑
    a[:128, :128] = 40
    a[:128, :128][((yy - 64) ** 2 + (xx - 64) ** 2) < 900] = 240
    img = Image.fromarray(a)
    base = uniform_base_boxes(256, 256, 8, 8)
    leaves = adaptive_leaves(img, base, levels=3, threshold=25.0,
                             detail=SaliencyMap(img))
    subject = [l for l in leaves if l[2] <= 128 and l[3] <= 128]
    background = [l for l in leaves if l[0] >= 128 and l[1] >= 128]
    assert any(l[4] > 1 for l in subject), "主体区应细分"
    assert all(l[4] == 1 for l in background), "平滑背景区应保持大块"
