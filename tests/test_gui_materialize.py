"""回归：手动编辑时自动细分被物化成真实网格（右键/套索取消生效）。"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")


def test_materialize_auto_grid(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from photomosaic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    # 有边缘细节的主图（左亮右暗分界）
    arr = np.zeros((256, 256, 3), np.uint8)
    arr[:, :128] = 220
    arr[:, 128:] = 40
    mp = tmp_path / "m.png"
    Image.fromarray(arr).save(mp)

    win = MainWindow(load_settings=False)
    win.resize(1300, 900)
    win.master_path = str(mp)
    win._show_master()
    win._goto(2)
    win.show()
    app.processEvents()

    # 自动细分开着：overlay 有幽灵格，tree 里全是 1 级
    win.auto_check.setChecked(True)
    win.threshold_slider.setValue(10)
    win._refresh_grid_overlay()
    overlay_n = len(win.grid_view._leaves)
    assert all(l[1] == 1 for l in win.tree.leaf_boxes()), "tree 里应全是 1 级"
    assert overlay_n > len(win.tree.leaf_boxes()), "overlay 应有更多预览格"

    # 手动编辑前：物化
    win._materialize_auto_grid()
    assert not win.auto_check.isChecked(), "auto 应被关闭"
    assert len(win.tree.leaf_boxes()) == overlay_n, "tree 应等于物化前的 overlay"

    # 物化后：右键取消真实生效
    deep = [l for l in win.tree.leaf_boxes() if l[1] > 1]
    assert deep, "应有细分格"
    x0, y0, x1, y1 = deep[0][0]
    n_before = len(win.tree.leaf_boxes())
    assert win.tree.merge_up_at((x0 + x1) // 2, (y0 + y1) // 2)
    assert len(win.tree.leaf_boxes()) < n_before
    win.close()
