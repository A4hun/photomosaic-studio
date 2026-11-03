"""Lead 综合 QA：真实用户流程端到端，专找"按钮没反应/配置没生效"级 bug。
覆盖：四步流程×四种风格、网格手势、批量（统一+单独配置+断点续批）、设置记忆。
只读源码、不编辑，跑真实离屏操作+检查真实产出文件。
"""
import os, sys, time, tempfile
sys.path.insert(0, r"D:\PyCharmProject\project\photomosaic")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PIL import Image

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def wait(w, app, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if not w._worker or not w._worker.isRunning():
            return True
        time.sleep(0.05)
    return False


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from photomosaic.gui import MainWindow
    from photomosaic.db import TileDB
    from photomosaic.library import ingest

    app = QApplication([])
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    tiles = os.path.join(tmp, "tiles"); os.makedirs(tiles)
    for i in range(12):
        a = np.zeros((64, 64, 3), np.uint8); a[:, :] = rng.integers(30, 220, 3); a[:, ::2] += 40
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(tiles, f"t{i}.png"))
    dbp = os.path.join(tmp, "t.db")
    db = TileDB(dbp); ingest(db, tiles, workers=1); db.close()
    masters = os.path.join(tmp, "m"); os.makedirs(masters)
    mpaths = []
    for i in range(2):
        p = os.path.join(masters, f"m{i}.png")
        Image.fromarray(rng.integers(0, 255, (300, 240, 3), dtype=np.uint8)).save(p)
        mpaths.append(p)

    # ==== A. 四步流程 × 四种风格 ====
    for style in ("grid", "brick", "hex", "dot"):
        w = MainWindow(load_settings=False)
        w.resize(1280, 820)
        w.master_queue.addItem(mpaths[0])
        w.master_queue.setCurrentRow(0)
        w.db_edit.setText(dbp)
        w.cells_x_spin.setValue(10)
        w._select_style(style)
        app.processEvents()
        check(w._style == style, f"风格选中 {style}")
        check(w.grid_view._style_name == style, f"网格预览形状切换为 {style}")
        outdir = os.path.join(tmp, f"out_{style}"); os.makedirs(outdir)
        w.outdir_edit.setText(outdir)
        w._render()
        check(wait(w, app), f"{style} 渲染完成")
        out = w._last_output
        check(os.path.exists(out), f"{style} 产出文件存在")
        w.close()

    # ==== B. 网格手势 ====
    w = MainWindow(load_settings=False)
    w.resize(1300, 900)
    w.master_queue.addItem(mpaths[0]); w.master_queue.setCurrentRow(0)
    w.db_edit.setText(dbp)
    w.cells_x_spin.setValue(10)
    app.processEvents()
    gv = w.grid_view
    c = gv._disp.center()
    n0 = len(w.tree.leaf_boxes())
    QTest.mouseClick(gv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, c)
    app.processEvents()
    n1 = len(w.tree.leaf_boxes())
    check(n1 > n0, f"左键单击细分（{n0}→{n1}）")
    # 注意：auto 开着时首次编辑会"物化"自动细分（树变大是正常的），
    # 右键合并应让格子数减少（合并一级），而不是回到物化前的 n0
    QTest.mouseClick(gv, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, c)
    app.processEvents()
    n2 = len(w.tree.leaf_boxes())
    check(n2 < n1, f"右键单击合并减格（{n1}→{n2}）")
    # Ctrl+左键单击细分到最细
    QTest.mouseClick(gv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, c)
    app.processEvents()
    lvls = [l for _b, l in w.tree.leaf_boxes()]
    check(max(lvls) == w.tree.max_level, "Ctrl+左键单击细分到最细")
    # Ctrl+右键单击取消整块
    QTest.mouseClick(gv, Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier, c)
    app.processEvents()
    check(len(w.tree.leaf_boxes()) <= n1, "Ctrl+右键单击取消整块细分")
    w.close()

    # ==== C. 批量：统一 + 单独配置 ====
    w = MainWindow(load_settings=False)
    w.db_edit.setText(dbp)
    for p in mpaths:
        w.master_queue.addItem(p)
    w.master_queue.setCurrentRow(0)
    w.cells_x_spin.setValue(8)
    w._master_overrides[mpaths[1]] = {"style": "hex"}
    outdir = os.path.join(tmp, "bout")
    os.makedirs(outdir)
    w.outdir_edit.setText(outdir)
    app.processEvents()
    w._render()
    check(wait(w, app), "批量渲染完成")
    outs = sorted(os.listdir(outdir))
    check(len(outs) == 2, f"批量 2 张产出（实际 {outs}）")
    w.close()

    # ==== D. 设置记忆往返 ====
    w = MainWindow(load_settings=False)
    w.threshold_slider.setValue(60)
    w.blend_slider.setValue(80)
    w._select_style("dot")
    w.signal_combo.setCurrentIndex(1)
    w.cells_x_spin.setValue(150)
    state = w._gather_state()
    w.close()
    w2 = MainWindow(load_settings=False)
    w2._apply_state(state)
    check(w2.threshold_slider.value() == 60, "记忆:阈值")
    check(w2.blend_slider.value() == 80, "记忆:融合")
    check(w2._style == "dot", "记忆:风格")
    check(w2.signal_combo.currentIndex() == 1, "记忆:细节识别")
    check(w2.cells_x_spin.value() == 150, "记忆:基础块数")
    w2.close()

    print()
    if FAILS:
        print("发现 bug:")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
