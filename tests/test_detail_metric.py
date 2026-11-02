"""细节度量区分度回归测试：噪点区不切、真实细节区切。

度量升级后的行为契约（噪点 vs 边缘/纹理的可分性是用户反馈的痛点）。
"""

import numpy as np
import pytest
from PIL import Image

from photomosaic.render import DetailMap


def _img(arr):
    if arr.ndim == 2:
        arr = arr[:, :, None].repeat(3, axis=2)
    return Image.fromarray(arr.astype(np.uint8))


@pytest.fixture(scope="module")
def cases():
    rng = np.random.default_rng(0)
    flat = np.full((128, 128, 3), 200, np.uint8)
    edge = np.zeros((128, 128, 3), np.uint8)
    edge[:, :64] = 220
    edge[:, 64:] = 40
    noise = np.clip(128 + rng.integers(-6, 7, (128, 128, 3)), 0, 255).astype(np.uint8)
    # 周期条纹（毛发/织物）
    stripe = ((np.indices((128, 128))[0] % 4 // 2) * 140 + 60)[:, :, None].repeat(3, 2)
    # 斑点（花朵/树叶）
    spots = np.full((128, 128), 128, np.uint8)
    yy, xx = np.ogrid[:128, :128]
    for cx, cy in [(32, 32), (96, 48), (48, 96)]:
        spots[((yy - cy) ** 2 + (xx - cx) ** 2) < 256] = 220
    spots = spots[:, :, None].repeat(3, 2)
    return {
        "flat": _img(flat), "edge": _img(edge), "noise": _img(noise),
        "stripe": _img(stripe), "spots": _img(spots),
    }


def test_noise_below_edge_and_texture(cases):
    """噪点的细节分必须显著低于真实细节（边缘/纹理）。"""
    e_noise = DetailMap(cases["noise"]).energy((0, 0, 128, 128))
    e_flat = DetailMap(cases["flat"]).energy((0, 0, 128, 128))
    e_edge = DetailMap(cases["edge"]).energy((0, 0, 128, 128))
    e_stripe = DetailMap(cases["stripe"]).energy((0, 0, 128, 128))
    # 关键序关系：真实边缘与纹理 > 噪点 > 平区
    assert e_edge > e_noise * 3, f"edge {e_edge:.1f} vs noise {e_noise:.1f}"
    assert e_stripe > e_noise * 3, f"stripe {e_stripe:.1f} vs noise {e_noise:.1f}"
    assert e_flat < e_noise, f"flat {e_flat:.1f} 应低于 noise {e_noise:.1f}"
    # 绝对量级标定：真实细节在阈值 25 以上，噪点远低于它
    assert e_edge > 25 and e_stripe > 25
    assert e_noise < 25


def test_noise_not_split_but_edge_split():
    """自适应细分：噪点区不切，边缘区切。"""
    from photomosaic.render import adaptive_leaves, uniform_base_boxes

    rng = np.random.default_rng(1)
    base_img = np.full((256, 256, 3), 128, np.uint8)
    # 右半：垂直条纹纹理（真实细节）；左半：噪点
    yy = np.indices((256, 128))[0]
    stripe = ((yy % 4) // 2 * 140 + 60).astype(np.uint8)
    base_img[:, 128:] = stripe[:, :, None].repeat(3, 2)
    base_img[:, :128] = np.clip(128 + rng.integers(-6, 7, (256, 128, 3)), 0, 255)
    img = Image.fromarray(base_img.astype(np.uint8))
    leaves = adaptive_leaves(img, uniform_base_boxes(256, 256, 8, 8),
                             levels=3, threshold=25.0)
    # 右半（条纹纹理）应有细分，左半（噪点）应保持一级
    right = [l for l in leaves if l[2] > 128]
    left = [l for l in leaves if l[2] <= 128]
    assert any(l[4] > 1 for l in right), "纹理区应细分"
    assert all(l[4] == 1 for l in left), "噪点区不应细分"
