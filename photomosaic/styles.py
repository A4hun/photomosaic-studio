"""马赛克风格渲染器：布局偏移 + 瓦片形状蒙版 + 背景底，复用匹配核心。

风格与匹配解耦：每种风格只负责"格子画在什么位置、瓦片切成什么形状、
缝隙垫什么底"，格子的特征计算与瓦片分配完全复用现有核心。

- grid   经典方形网格（默认，现状）
- brick  错位砖墙：偶数行整体右移半格，手工拼贴感
- hex    六边形蜂窝：瓦片切六边形，缝隙垫高斯模糊的主图
- dot    圆点点彩：瓦片切圆形，缝隙垫高斯模糊的主图（修拉点彩）
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class Style:
    """风格协议：布局偏移 + 瓦片蒙版 + 背景底。"""

    name = "grid"
    label = "经典网格"

    def offset(self, row: int, col: int, cell_w: int, cell_h: int) -> Tuple[int, int]:
        """格子的额外像素偏移（在默认网格位置上叠加）。"""
        return (0, 0)

    def mask(self, size: Tuple[int, int]) -> Optional[Image.Image]:
        """瓦片形状蒙版（L 模式，255=保留）。None = 矩形（不裁）。"""
        return None

    def has_gaps(self) -> bool:
        """瓦片间是否有缝隙（有缝隙才需要垫背景底）。"""
        return False

    def background(self, master: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """画布底色：默认白；有缝隙的风格垫高斯模糊的主图（缝隙才不突兀）。"""
        return Image.new("RGB", size, (255, 255, 255))


class BrickStyle(Style):
    name = "brick"
    label = "错位砖墙"

    def offset(self, row: int, col: int, cell_w: int, cell_h: int) -> Tuple[int, int]:
        # 偶数行右移半格（砖墙式交错）
        return (cell_w // 2, 0) if row % 2 == 1 else (0, 0)


class _GappedStyle(Style):
    """有缝隙风格的公共背景：高斯模糊的主图垫底。"""

    def has_gaps(self) -> bool:
        return True

    def background(self, master: Image.Image, size: Tuple[int, int]) -> Image.Image:
        W, H = master.size
        base = master.convert("RGB").resize(size, Image.Resampling.BOX)
        return base.filter(ImageFilter.GaussianBlur(radius=max(2, min(size) // 64)))


class HexStyle(_GappedStyle):
    name = "hex"
    label = "六边形蜂窝"

    def mask(self, size: Tuple[int, int]) -> Image.Image:
        w, h = size
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        # 平顶六边形内接于格子
        d.polygon([(w * 0.25, 0), (w * 0.75, 0), (w, h * 0.5),
                   (w * 0.75, h), (w * 0.25, h), (0, h * 0.5)], fill=255)
        return m


class DotStyle(_GappedStyle):
    name = "dot"
    label = "圆点点彩"

    def mask(self, size: Tuple[int, int]) -> Image.Image:
        w, h = size
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        d.ellipse([0, 0, w - 1, h - 1], fill=255)
        return m


_STYLES = {s.name: s for s in (Style(), BrickStyle(), HexStyle(), DotStyle())}
# 标签 → name（GUI 下拉用标签）
STYLES_BY_LABEL = {s.label: s for s in _STYLES.values()}


def get_style(name: str) -> Style:
    """按 name 取风格；也接受 GUI 标签。未知名回退 grid。"""
    if name in _STYLES:
        return _STYLES[name]
    return STYLES_BY_LABEL.get(name, _STYLES["grid"])


def all_styles() -> list:
    """[(name, label), ...] 按展示顺序。"""
    return [(s.name, s.label) for s in _STYLES.values()]
