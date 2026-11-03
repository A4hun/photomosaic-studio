# -*- coding: utf-8 -*-
"""QA 流程 B：③页 GridView 网格编辑手势全测（QTest 真实鼠标事件）。

覆盖（每个手势后断言树的叶子数/层级真的变了）：
  左键单击            细分一级
  右键单击            合并一级
  Ctrl+左键拖拽       自由套索圈选，区域内细分到最细
  Ctrl+右键拖拽       套索取消区域内细分
  Ctrl+Shift+左键拖拽 矩形框选细分到最细（旧行为）
  Ctrl+左键单击       该格一次细分到最细
  Ctrl+右键单击       取消该基础块全部细分
关键集成断言：
  手势改完后点④生成，产出必须真的用了编辑后的网格（手动网格不被自动覆盖）；
  自动细分开着时手动编辑会物化（before_edit → _materialize_auto_grid），
  物化后右键取消必须生效；
  物化后树的粒度上限必须仍受右侧「嵌套层级」所选级数约束。
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (Checker, get_app, make_master, make_tiles, patch_dialogs,
                    pump, run_guarded, wait_worker)  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

OUT = os.path.join(common.ROOT, "_qa", "out", "flow_b")


def _fresh_dir(path: str) -> str:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


class Driver:
    def __init__(self, win):
        self.win = win
        self.gv = win.grid_view

    def cell_center(self, gx: int, gy: int):
        """基础格 (gx,gy) 中心对应的控件坐标。"""
        boxes = self.win.tree.leaf_boxes()
        # 基础格均匀分布：直接用第 gy*base+gx 个 1 级根的 box
        base = max(b for b, l in boxes if l == 1)  # fallback
        target = None
        n = 0
        for b, l in sorted(boxes):
            if l == 1:
                n += 1
        side = int(round(n ** 0.5))
        for b, l in sorted(boxes, key=lambda t: (t[0][1], t[0][0])):
            if l == 1 and (b[1] // max(1, b[3] - b[1]) == gy) \
                    and (b[0] // max(1, b[2] - b[0]) == gx):
                target = b
                break
        if target is None:
            target = base
        r = self.gv.img_rect_to_widget(target)
        return r.center(), target

    def click(self, pos, btn, mods=Qt.KeyboardModifier.NoModifier):
        QTest.mousePress(self.gv, btn, mods, pos)
        pump(10)
        QTest.mouseRelease(self.gv, btn, mods, pos)
        pump(30)

    def drag(self, points, btn, mods):
        QTest.mousePress(self.gv, btn, mods, points[0])
        for p in points[1:]:
            QTest.mouseMove(self.gv, p)
            pump(5)
        QTest.mouseRelease(self.gv, btn, mods, points[-1])
        pump(30)


def main() -> int:
    common.patch_dialogs()
    app = get_app()
    ck = Checker("flow_b")

    from photomosaic.gui import MainWindow

    work = _fresh_dir(OUT)
    master = make_master(os.path.join(work, "master.png"), 256, 256)
    tiles = make_tiles(os.path.join(work, "tiles"), n=12)
    db_path = os.path.join(work, "tiles.db")
    outdir = _fresh_dir(os.path.join(work, "out"))

    win = MainWindow(load_settings=False)
    win.resize(1300, 900)
    win.show()
    orig_done = win._render_done
    def _wrap(stats):
        win._qa_stats = stats
        orig_done(stats)
    win._render_done = _wrap

    # 建库（走真实按钮，④要用）
    win.prefill(master=master)
    win._goto(1)
    win.folder_list.addItem(tiles)
    win.db_edit.setText(db_path)
    win.workers_spin.setValue(1)
    QTest.mouseClick(win.btn_ingest, Qt.MouseButton.LeftButton)
    assert wait_worker(win, 120), "建库超时"

    win._goto(2)
    pump(150)
    win.cells_x_spin.setValue(8)           # 8×8=64 基础块，32px/块
    win.auto_check.setChecked(False)       # 先测纯手动网格
    pump(80)
    d = Driver(win)
    gv = win.grid_view

    base_n = len(win.tree.leaf_boxes())
    ck.section(f"基线：8×8 均匀网格 {base_n} 叶")
    ck.check(base_n == 64, "基线 64 叶", str(base_n))

    # ---- 1. 左键单击：细分一级 -----------------------------------------
    pos, box = d.cell_center(3, 3)
    d.click(pos, Qt.MouseButton.LeftButton)
    n1 = len(win.tree.leaf_boxes())
    lv2 = sum(1 for _b, l in win.tree.leaf_boxes() if l == 2)
    ck.check(n1 == base_n + 3 and lv2 == 4, "左键单击 → 细分一级（+3 叶，4 个 2 级）",
             f"叶 {n1}, 2级 {lv2}")
    ck.check(len(gv._leaves) == n1, "手势后 overlay 与树一致")

    # ---- 2. 右键单击：合并一级 -----------------------------------------
    d.click(pos, Qt.MouseButton.RightButton)
    n2 = len(win.tree.leaf_boxes())
    ck.check(n2 == base_n and all(l == 1 for _b, l in win.tree.leaf_boxes()),
             "右键单击 → 合并回原状", f"叶 {n2}")

    # ---- 3. Ctrl+左键单击：该格一次细分到最细 ---------------------------
    d.click(pos, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    n3 = len(win.tree.leaf_boxes())
    maxlv = max(l for _b, l in win.tree.leaf_boxes())
    ck.check(maxlv == win.tree.max_level, "Ctrl+左键单击 → 细分到最细级",
             f"最细 {maxlv}/{win.tree.max_level}")
    ck.check(n3 == base_n - 1 + 4 ** (win.tree.max_level - 1),
             "最细叶数 = 63 + 4^(级-1)", f"叶 {n3}")

    # ---- 4. Ctrl+右键单击：取消整块 ------------------------------------
    d.click(pos, Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier)
    n4 = len(win.tree.leaf_boxes())
    ck.check(n4 == base_n and all(l == 1 for _b, l in win.tree.leaf_boxes()),
             "Ctrl+右键单击 → 取消整块", f"叶 {n4}")

    # ---- 5. Ctrl+左键拖拽（自由套索）：圈内细分到最细 -------------------
    c, _box = d.cell_center(5, 5)
    diamond = [QPoint(c.x(), c.y() - 40), QPoint(c.x() + 40, c.y()),
               QPoint(c.x(), c.y() + 40), QPoint(c.x() - 40, c.y())]
    d.drag(diamond, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    n5 = len(win.tree.leaf_boxes())
    maxlv5 = max(l for _b, l in win.tree.leaf_boxes())
    ck.check(maxlv5 == win.tree.max_level, "Ctrl+左键套索 → 圈内到最细级",
             f"最细 {maxlv5}")
    ck.check(n5 > base_n, "套索后叶数增加", f"叶 {n5}")
    lv1_left = sum(1 for _b, l in win.tree.leaf_boxes() if l == 1)
    ck.check(lv1_left > 0, "圈外仍有未细分的基础块", f"1级剩 {lv1_left}")

    # ---- 6. Ctrl+右键拖拽（套索）：取消圈内细分 -------------------------
    fine = [b for b, l in win.tree.leaf_boxes() if l > 1]
    bx0 = min(b[0] for b in fine); by0 = min(b[1] for b in fine)
    bx1 = max(b[2] for b in fine); by1 = max(b[3] for b in fine)
    r = gv.img_rect_to_widget((bx0, by0, bx1, by1))
    m = 8
    rect_path = [QPoint(r.left() - m, r.top() - m),
                 QPoint(r.right() + m, r.top() - m),
                 QPoint(r.right() + m, r.bottom() + m),
                 QPoint(r.left() - m, r.bottom() + m)]
    d.drag(rect_path, Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier)
    n6 = len(win.tree.leaf_boxes())
    ck.check(n6 == base_n and all(l == 1 for _b, l in win.tree.leaf_boxes()),
             "Ctrl+右键套索 → 圈内细分全部取消", f"叶 {n6}")

    # ---- 7. Ctrl+Shift+左键拖拽（矩形框选）+ Ctrl+Shift+右键取消 --------
    c2, _ = d.cell_center(1, 6)
    r2 = gv.img_rect_to_widget(_)
    rect2 = [QPoint(r2.left() - 4, r2.top() - 4),
             QPoint(r2.right() + 4, r2.bottom() + 4)]
    d.drag(rect2, Qt.MouseButton.LeftButton,
           Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    n7 = len(win.tree.leaf_boxes())
    ck.check(max(l for _b, l in win.tree.leaf_boxes()) == win.tree.max_level,
             "Ctrl+Shift+左键框选 → 细分到最细", f"叶 {n7}")
    d.drag(rect2, Qt.MouseButton.RightButton,
           Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    n8 = len(win.tree.leaf_boxes())
    ck.check(n8 == base_n and all(l == 1 for _b, l in win.tree.leaf_boxes()),
             "Ctrl+Shift+右键框选 → 取消", f"叶 {n8}")

    # ---- 8. 手动编辑 → ④生成：产出必须用编辑后的网格 --------------------
    ck.section("手动网格 → ④生成（不被自动覆盖）")
    d.click(pos, Qt.MouseButton.LeftButton)                    # 细分 (3,3)
    pos2, _b2 = d.cell_center(6, 1)
    d.click(pos2, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    pump(50)
    edited = sorted((tuple(b), l) for b, l in win.tree.leaf_boxes())
    ck.check(len(edited) > base_n, "编辑后树有细分", f"叶 {len(edited)}")
    ck.check(not win.auto_check.isChecked(), "自动细分未被重新打开")

    win._goto(3)
    win.outdir_edit.setText(outdir)
    win.cell_px_spin.setValue(0)           # 与主图同大 → 叶坐标 1:1
    win.blend_slider.setValue(0)
    win._qa_stats = None
    QTest.mouseClick(win.btn_render, Qt.MouseButton.LeftButton)
    ck.check(wait_worker(win, 180), "生成 worker 完成")
    stats = win._qa_stats
    ck.check(stats is not None, "拿到渲染 stats")
    got_leaves = sorted((tuple(map(int, r[:4])), int(r[4]))
                        for r in stats["leaf_rects"])
    ck.check(got_leaves == edited,
             "产出叶子集合 == 编辑后的树（1:1，手动网格未被自动覆盖）",
             f"产出 {len(got_leaves)} vs 树 {len(edited)}")
    ck.check(stats.get("mode") == "manual", "渲染模式 manual",
             str(stats.get("mode")))

    # ---- 9. 物化回归：自动细分开着 → 手动编辑物化 → 右键取消必须生效 ----
    ck.section("物化（auto→手动）回归")
    win._goto(2)
    pump(80)
    win._reset_grid()
    win.auto_check.setChecked(True)
    win.threshold_slider.setValue(5)       # 低阈值保证幽灵格很多
    pump(350)                              # 等去抖刷新
    overlay_n = len(gv._leaves)
    tree_n = len(win.tree.leaf_boxes())
    ck.check(overlay_n > tree_n and all(l == 1 for _b, l in win.tree.leaf_boxes()),
             "自动细分预览是幽灵格（树里全 1 级）",
             f"overlay {overlay_n} vs 树 {tree_n}")
    # 找一个 overlay 里的深格中心去右键（真实手势触发 before_edit → 物化 → 合并）
    deep = [b for b, l in gv._leaves if l > 1]
    ck.check(bool(deep), "overlay 里有深格可点")
    db0 = deep[len(deep) // 2]
    rc = gv.img_rect_to_widget(db0)
    d.click(rc.center(), Qt.MouseButton.RightButton)
    ck.check(not win.auto_check.isChecked(), "首次手动编辑后 auto 已关闭（物化）")
    mat_n = len(win.tree.leaf_boxes())
    ck.check(mat_n < overlay_n,
             "物化后右键合并真实生效（叶数减少）",
             f"overlay {overlay_n} → 树 {mat_n}")
    ck.check(len(gv._leaves) == mat_n, "overlay 与物化后的树一致")

    # ---- 10. 物化后粒度上限必须仍受所选级数约束 -------------------------
    ck.section("物化后粒度上限回归")
    win._reset_grid()
    win.level_radios[1].click()            # 二级
    win.auto_check.setChecked(True)
    pump(350)
    pos3, _b3 = d.cell_center(2, 2)
    d.click(pos3, Qt.MouseButton.LeftButton)   # 任意编辑 → 触发物化
    pump(50)
    d.click(pos3, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    maxlv10 = max(l for _b, l in win.tree.leaf_boxes())
    ck.check(maxlv10 <= 2,
             "物化后 Ctrl+左键细分不得超过所选「二级」",
             f"实际最细 {maxlv10} 级")

    win.close()
    return ck.summary()


if __name__ == "__main__":
    sys.exit(run_guarded(main))
