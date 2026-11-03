# -*- coding: utf-8 -*-
"""QA 流程 A：四步完整用户流程（离屏真跑）。

① 打开主图 → ② 建库（点真实"构建"按钮）→ ③ 调配置（风格/层级/阈值/显著性）
→ ④ 生成（点真实"生成"按钮）→ 断言产出文件真实反映③的配置：

- 产出文件存在、尺寸 = cells_x × 每格像素；
- 一级均匀网格的叶子数 = cells_x × cells_y，不同 cells_x 格数应变；
- hex/dot 风格：格子角落是背景底（高斯模糊主图），不是实心瓦片；
- grid 风格：格子角落是实心瓦片（高色度），不是背景底；
- brick 风格：偶数行右移半格（错位砖墙必须在成品里真的错位）。
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (Checker, get_app, make_master, make_tiles, patch_dialogs,
                    pump, run_guarded, wait_worker)  # noqa: E402

OUT = os.path.join(common.ROOT, "_qa", "out", "flow_a")


def _fresh_dir(path: str) -> str:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def main() -> int:
    common.patch_dialogs()
    app = get_app()
    ck = Checker("flow_a")

    from PIL import Image, ImageFilter
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from photomosaic.db import TileDB
    from photomosaic.gui import MainWindow

    work = _fresh_dir(OUT)
    master = make_master(os.path.join(work, "master.png"), 256, 256)
    tiles = make_tiles(os.path.join(work, "tiles"), n=12)
    db_path = os.path.join(work, "tiles.db")
    outdir = _fresh_dir(os.path.join(work, "out"))

    win = MainWindow(load_settings=False)
    win.resize(1300, 900)
    win.show()
    pump(50)

    # 捕获完整 stats（包一层 _render_done）
    orig_done = win._render_done
    def _wrap(stats):
        win._qa_stats = stats
        orig_done(stats)
    win._render_done = _wrap

    ck.section("① 打开主图")
    win.prefill(master=master)
    pump(50)
    ck.check(win.master_path == master, "主图已加载")
    ck.check(win.tree is not None, "打开主图后网格树已建立")
    ck.check(win.db_edit.text() == os.path.join(os.path.dirname(master), "tiles.db"),
             "主图加载后数据库路径自动带出到主图旁")

    ck.section("② 构建单元图数据库（真实按钮）")
    win._goto(1)
    win.folder_list.addItem(tiles)
    win.db_edit.setText(db_path)
    win.workers_spin.setValue(1)          # 脚本内走顺序路径，避免进程池
    QTest.mouseClick(win.btn_ingest, Qt.MouseButton.LeftButton)
    ck.check(wait_worker(win, 120), "建库 worker 在限时内完成")
    ck.check(os.path.exists(db_path), "数据库文件已创建", db_path)
    with TileDB(db_path) as db:
        ck.check(db.count() == 24, "库内容量 = 12 图 × 2 变体", f"实际 {db.count()}")
    ck.check("数据库共 24 块" in win.lib_info.text(), "②页库信息文本更新",
             win.lib_info.text())

    ck.section("③ 网格与配置")
    win._goto(2)
    pump(100)
    win.cells_x_spin.setValue(16)          # 触发 _rebuild_tree
    pump(50)
    ck.check(len(win.tree.leaf_boxes()) == 256, "cells_x=16 → 16×16=256 个基础块")
    win.level_radios[0].click()            # 一级：经典均匀网格
    pump(50)
    ck.check(win._effective_levels() == 1, "选「一级」后有效层级=1")

    def render(tag: str):
        win._qa_stats = None
        QTest.mouseClick(win.btn_render, Qt.MouseButton.LeftButton)
        ok = wait_worker(win, 180)
        ck.check(ok, f"[{tag}] 生成 worker 在限时内完成")
        crits = [c for c in common.MSGBOX_CALLS if c[0] == "critical"]
        ck.check(not crits, f"[{tag}] 生成无失败弹窗", repr(crits))
        return win._qa_stats

    win._goto(3)
    win.outdir_edit.setText(outdir)
    win.cell_px_spin.setValue(32)          # 输出宽度 = 16×32 = 512
    win.blend_slider.setValue(0)           # 关融合：角落像素断言确定性

    ck.section("④ 生成：hex 蜂窝")
    win._style_buttons["hex"].click()      # 真实风格卡片按钮
    pump(50)
    ck.check(win._style == "hex", "风格卡片选中 hex")
    ck.check(win.grid_view._style_name == "hex", "③页网格预览切换为六边形形状")
    stats = render("hex")
    out_hex = stats["output"] if stats else None
    ck.check(bool(out_hex and os.path.exists(out_hex)), "[hex] 产出文件存在",
             str(out_hex))
    shutil.copy(out_hex, os.path.join(work, "hex.png"))

    W, H = 512, 512
    with Image.open(out_hex) as im:
        got = im.convert("RGB")
    ck.check(got.size == (W, H), "[hex] 输出尺寸 = 16 格 × 32px = 512×512",
             f"实际 {got.size}")
    ck.check(stats.get("mode") == "manual", "[hex] 一级手动网格模式",
             str(stats.get("mode")))
    ck.check(len(stats.get("leaf_rects", [])) == 256, "[hex] 叶子数 256",
             str(len(stats.get("leaf_rects", []))))

    # hex 背景 = 高斯模糊主图；格子角落（六边形外）必须精确等于背景
    with Image.open(master) as im:
        mrgb = im.convert("RGB")
    bg = mrgb.resize((W, H), Image.Resampling.BOX).filter(
        ImageFilter.GaussianBlur(radius=max(2, min((W, H)) // 64)))
    corners_ok = 0
    corners_total = 0
    chroma_max = 0
    for (x0, y0, x1, y1, _lv) in stats["leaf_rects"]:
        if x0 == 0 or y0 == 0 or x1 >= W or y1 >= H:
            continue                      # 只采样内部格，避开画布边缘
        corners_total += 1
        px = got.getpixel((x0 + 1, y0 + 1))
        bgpx = bg.getpixel((x0 + 1, y0 + 1))
        if px == bgpx:
            corners_ok += 1
        chroma_max = max(chroma_max, max(px) - min(px))
    ck.check(corners_total > 0 and corners_ok == corners_total,
             "[hex] 每个格子角落精确等于背景底（六边形缝隙）",
             f"{corners_ok}/{corners_total}")
    ck.check(chroma_max < 40, "[hex] 角落是模糊底色（低色度）",
             f"最大色差 {chroma_max}")
    # 格中心应是瓦片（高色度），与底色不同
    cx, cy = stats["leaf_rects"][8 * 16 + 8][:2]
    cc = stats["leaf_rects"][8 * 16 + 8]
    mid = got.getpixel(((cc[0] + cc[2]) // 2, (cc[1] + cc[3]) // 2))
    ck.check(max(mid) - min(mid) > 60, "[hex] 格中心是实心瓦片（高色度）",
             f"中心像素 {mid}")

    ck.section("④ 生成：grid 经典网格（对照组）")
    win._style_buttons["grid"].click()
    pump(50)
    stats_g = render("grid")
    out_grid = stats_g["output"]
    shutil.copy(out_grid, os.path.join(work, "grid.png"))
    with Image.open(out_grid) as im:
        got_g = im.convert("RGB")
    bg_hits = 0
    chroma_ok = 0
    n = 0
    for (x0, y0, x1, y1, _lv) in stats_g["leaf_rects"]:
        if x0 == 0 or y0 == 0 or x1 >= W or y1 >= H:
            continue
        n += 1
        px = got_g.getpixel((x0 + 1, y0 + 1))
        if px == bg.getpixel((x0 + 1, y0 + 1)):
            bg_hits += 1
        if max(px) - min(px) > 60:
            chroma_ok += 1
    ck.check(bg_hits == 0, "[grid] 角落没有残留背景底（瓦片铺满）",
             f"{bg_hits}/{n}")
    ck.check(chroma_ok == n, "[grid] 每个角落都是实心瓦片（高色度）",
             f"{chroma_ok}/{n}")

    ck.section("④ 生成：brick 错位砖墙（偶数行应右移半格）")
    win._style_buttons["brick"].click()
    pump(50)
    stats_b = render("brick")
    out_brick = stats_b["output"]
    shutil.copy(out_brick, os.path.join(work, "brick.png"))
    with Image.open(out_brick) as im:
        got_b = im.convert("RGB")
    # 错位生效时：第 2 行（row=1，0 起）左端半格无瓦片 → 白底 (255,255,255)
    row1_y = 32 + 16                     # 第 2 行中部
    px_shifted = got_b.getpixel((1, row1_y))
    row0_y = 16                          # 第 1 行（不错位）左端应有瓦片
    px_first = got_b.getpixel((1, row0_y))
    ck.check(px_shifted == (255, 255, 255),
             "[brick] 偶数行右移半格：第 2 行左端露出白底",
             f"实际 {px_shifted}")
    ck.check(max(px_first) - min(px_first) > 60,
             "[brick] 奇数行不错位：第 1 行左端是瓦片",
             f"实际 {px_first}")

    ck.section("④ cells_x 变化 → 格数与尺寸必须跟着变")
    win._style_buttons["grid"].click()
    win.cells_x_spin.setValue(8)
    pump(50)
    ck.check(len(win.tree.leaf_boxes()) == 64, "cells_x=8 → 8×8=64 个基础块")
    stats8 = render("cells8")
    ck.check(tuple(stats8["size"]) == (256, 256), "cells_x=8 → 输出 8×32=256px",
             str(stats8.get("size")))
    ck.check(len(stats8.get("leaf_rects", [])) == 64, "cells_x=8 → 64 叶",
             str(len(stats8.get("leaf_rects", []))))
    win.cells_x_spin.setValue(16)
    pump(50)
    stats16 = render("cells16")
    ck.check(tuple(stats16["size"]) == (512, 512), "cells_x=16 → 输出 512px")
    ck.check(len(stats16.get("leaf_rects", [])) == 256, "cells_x=16 → 256 叶")
    with open(os.path.join(work, "grid.png"), "rb") as fh:
        first_bytes = fh.read()
    with open(stats16["output"], "rb") as fh:
        second_bytes = fh.read()
    ck.check(first_bytes != second_bytes, "不同 cells_x 产出内容不同")

    win.close()
    return ck.summary()


if __name__ == "__main__":
    sys.exit(run_guarded(main))
