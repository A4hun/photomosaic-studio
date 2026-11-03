"""会话记忆回归：新增结果页设置项的往返 + 坏存档容错。"""

import pytest

pytest.importorskip("PySide6")


def _app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_result_page_settings_roundtrip():
    from photomosaic.gui import MainWindow

    _app()
    w1 = MainWindow(load_settings=False)
    w1.zoom_combo.setCurrentText("100%")
    w1.format_combo.setCurrentText("JPEG")
    w1.out_quality.setValue(90)
    w1._goto(2)                      # 停在第③页
    state = w1._gather_state()
    w1.close()

    w2 = MainWindow(load_settings=False)
    w2._apply_state(state)
    assert w2.zoom_combo.currentText() == "100%"
    assert w2.format_combo.currentText() == "JPEG"
    assert w2.out_quality.value() == 90
    assert w2.pages.currentIndex() == 2
    w2.close()


def test_broken_master_does_not_lose_other_settings(tmp_path):
    """主图指向已删除的文件时，其余设置必须照常恢复（旧版会整体吞掉）。"""
    import numpy as np
    from PIL import Image

    from photomosaic.gui import MainWindow

    _app()
    master = tmp_path / "m.png"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(master)

    w1 = MainWindow(load_settings=False)
    w1.master_path = str(master)
    w1._show_master()
    w1.threshold_slider.setValue(61)
    w1.blend_slider.setValue(77)
    w1.cells_x_spin.setValue(133)
    state = w1._gather_state()
    w1.close()
    master.unlink()                  # 主图被删

    w2 = MainWindow(load_settings=False)
    w2._apply_state(state)           # 不能抛异常
    assert w2.threshold_slider.value() == 61
    assert w2.blend_slider.value() == 77
    assert w2.cells_x_spin.value() == 133
    w2.close()


def test_garbage_state_tolerated():
    """完全损坏的存档（错类型/脏数据）：不崩，全部回默认。"""
    from photomosaic.gui import MainWindow

    _app()
    w = MainWindow(load_settings=False)
    w._apply_state({"threshold": "not-a-number",
                    "zoom": "不存在的档位",
                    "page": 99,
                    "geometry": "garbage",
                    "splitter3": "!!!not-base64!!!",
                    "tree_leaves": "nope",
                    "master": 12345})
    assert w.threshold_slider.value() == 75    # 默认（滑块反转：75=阈值25）
    assert w.pages.currentIndex() == 0
    w.close()
