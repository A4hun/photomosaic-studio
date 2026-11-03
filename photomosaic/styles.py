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

    def background_region(self, master: Image.Image,
                          y0: int, y1: int,
                          out_w: int, full_w: int, full_h: int) -> Image.Image:
        """流式条带的背景：与整图 background 同半径模糊 + 上下文扩展防接缝。

        整图路径对 (OW,OH) 模糊一次；流式若每条带独立裁+模糊，半径随条带高
        变化且丢上下文行，条带接缝必然有缝。这里：模糊半径取整图口径
        （min(全宽,全高)//64），条带向上下各扩 4σ 上下文再裁回，条带间
        背景连续，与整图路径视觉一致。"""
        radius = max(2, min(full_w, full_h) // 64)
        margin = 4 * radius                          # 4σ 上下文，模糊边界不吃接缝
        s = full_h / max(1, master.height)           # 输出/主图 行缩放
        ey0 = max(0, y0 - margin)
        ey1 = min(full_h, y1 + margin)
        my0 = min(master.height - 1, int(ey0 / s))
        my1 = max(my0 + 1, min(master.height, int(round(ey1 / s))))
        crop = master.crop((0, my0, master.width, my1)).convert("RGB")
        region = crop.resize((out_w, ey1 - ey0), Image.Resampling.BOX)
        blurred = region.filter(ImageFilter.GaussianBlur(radius=radius))
        off = y0 - ey0                               # 裁回本条带的精确行
        return blurred.crop((0, off, out_w, off + (y1 - y0)))


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
