# -*- coding: utf-8 -*-
"""QA 离屏流程发现的交互/集成 bug 回归测试：

1. brick 错位砖墙：manual/leaf_boxes 分支与流式分支此前丢弃 style.offset，
   风格卡片（uniform 分支）有错位、成品没有 → 现在两条路径都必须错位。
2. clear_region：此前按"格心在框内"命中，与 refine_region 的"相交"命中
   不对称 → 同一矩形先细分再取消留下残留；现在互为逆操作。
3. _materialize_auto_grid：此前 max_level 硬编码 max(5, …)，物化后可越过
   右侧所选「嵌套层级」继续细分；现在粒度上限跟随所选级数。
"""

import os

import numpy as np
import pytest
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.grid import QuadTree
from photomosaic.library import ingest
from photomosaic.render import make_mosaic


@pytest.fixture()
def env(tmp_path):
    """12 张高色度瓦片 + 96×96 噪点主图 + 已建库。"""
    rng = np.random.default_rng(5)
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(12):
        arr = np.empty((48, 48, 3), dtype=np.uint8)
        arr[:, :] = rng.integers(0, 256, 3, dtype=np.uint8)
        arr[:, :, i % 3] = 220
        arr[:, :, (i + 1) % 3] = 20
        Image.fromarray(arr).save(folder / f"t{i}.png")
    master = tmp_path / "m.png"
    Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(master)
    db = TileDB()
    ingest(db, str(folder))
    yield str(master), db, tmp_path
    db.close()


def _uniform_boxes(w: int, h: int, cells: int):
    cw = w / cells
    ch = h / cells
    return [((int(gx * cw), int(gy * ch), int((gx + 1) * cw), int((gy + 1) * ch)), 1)
            for gy in range(cells) for gx in range(cells)]


def test_brick_offset_applies_with_leaf_boxes(env):
    """manual/leaf_boxes 分支：偶数行右移半格（错位砖墙）。"""
    master, db, tmp = env
    out = tmp / "brick_manual.png"
    stats = make_mosaic(master, db, str(out), cells_x=4, top_k=1, seed=1,
                        blend=0.0, leaf_boxes=_uniform_boxes(96, 96, 4),
                        levels=1, style="brick", color_grade=False)
    assert stats["mode"] == "manual"
    with Image.open(out) as im:
        got = im.convert("RGB")
    assert got.size == (96, 96)
    # 第 2 行（row=1）左端半格无瓦片 → 白底
    assert got.getpixel((1, 24 + 12)) == (255, 255, 255)
    assert got.getpixel((10, 24 + 12)) == (255, 255, 255)
    # 第 1 行（row=0）左端有瓦片 → 非白
    assert got.getpixel((1, 12)) != (255, 255, 255)


def test_brick_offset_applies_in_streaming(env):
    """流式分支：偶数行同样右移半格。"""
    master, db, tmp = env
    out = tmp / "brick_stream.png"
    stats = make_mosaic(master, db, str(out), cells_x=4, top_k=1, seed=1,
                        blend=0.0, leaf_boxes=_uniform_boxes(96, 96, 4),
                        levels=1, style="brick", stream_threshold_px=1,
                        color_grade=False)
    assert stats.get("streamed") is True
    with Image.open(out) as im:
        got = im.convert("RGB")
    assert got.getpixel((1, 24 + 12)) == (255, 255, 255)
    assert got.getpixel((1, 12)) != (255, 255, 255)


def test_clear_region_undoes_refine_same_box():
    """同一矩形：refine 后 clear 必须完全撤销（命中规则对称）。"""
    t = QuadTree(256, 256, 4, 4, max_level=4, min_px=1)
    box = (28, 28, 100, 100)                       # 故意不对齐基础格
    assert t.refine_region(box)
    assert any(l > 1 for _b, l in t.leaf_boxes())
    assert t.clear_region(box)
    assert all(l == 1 for _b, l in t.leaf_boxes()), \
        "clear_region 必须撤销 refine_region 碰到的所有细分"


def test_clear_region_respects_untouched_cells():
    """相交才清除：框外（含贴边）的细分必须保留。"""
    t = QuadTree(256, 256, 4, 4, max_level=4, min_px=1)
    t.refine_region((0, 0, 256, 256), 2)           # 全部到 2 级
    assert t.clear_region((0, 0, 64, 64))          # 只清左上角那个基础块
    leaves = t.leaf_boxes()
    assert all(l == 1 for b, l in leaves if b[2] <= 64 and b[3] <= 64)
    assert any(l == 2 for _b, l in leaves)


def test_materialize_respects_selected_level(tmp_path):
    """物化后粒度上限 = 右侧所选级数（不硬编码 5 级）。"""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from photomosaic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    arr = np.zeros((256, 256, 3), np.uint8)
    arr[:, :128] = 220
    arr[:, 128:] = 40
    mp = tmp_path / "m.png"
    Image.fromarray(arr).save(mp)

    win = MainWindow(load_settings=False)
    try:
        win.master_path = str(mp)
        win._show_master()
        win.level_radios[1].setChecked(True)       # 二级
        win.auto_check.setChecked(True)
        win.threshold_slider.setValue(10)
        win._refresh_grid_overlay()
        assert len(win.grid_view._leaves) > len(win.tree.leaf_boxes())
        win._materialize_auto_grid()
        assert win.tree.max_level == 2, \
            f"物化后粒度上限应跟随所选「二级」，实际 {win.tree.max_level}"
        assert win.tree.split_fully_at(64, 128)
        assert max(l for _b, l in win.tree.leaf_boxes()) <= 2
    finally:
        win.close()
