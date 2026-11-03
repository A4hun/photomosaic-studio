"""③页控件一致性回归：风格/首级保守/细节识别的改动必须体现在网格预览里。"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")


def _win(tmp_path):
    from PySide6.QtWidgets import QApplication
    from photomosaic.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    m = tmp_path / "m.png"
    Image.fromarray(np.random.default_rng(0).integers(
        0, 255, (600, 400, 3), dtype=np.uint8)).save(m)
    w = MainWindow(load_settings=False)
    w.master_path = str(m)
    w._show_master()
    return w


def test_style_selection_paints_shape_in_preview(tmp_path):
    """选蜂窝/点彩后，网格视图的绘制形状必须跟着变。"""
    w = _win(tmp_path)
    assert w.grid_view._style_name == "grid"
    w._select_style("hex")
    assert w.grid_view._style_name == "hex"
    w._select_style("dot")
    assert w.grid_view._style_name == "dot"
    w.close()


def test_first_level_check_affects_overlay(tmp_path):
    """首级保守勾选后，预览的细分必须减少（一级大块不轻易切碎）。"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    w = _win(tmp_path)
    w.auto_check.setChecked(True)
    w.threshold_slider.setValue(10)            # 低阈值确保有细分动力
    w.first_level_check.setChecked(False)
    w._refresh_grid_overlay()
    app.processEvents()
    n_open = len(w.grid_view._leaves)
    w.first_level_check.setChecked(True)       # 触发刷新（已接 toggled）
    app.processEvents()
    n_conservative = len(w.grid_view._leaves)
    assert n_conservative <= n_open, \
        f"首级保守应让细分不增反减：{n_open} -> {n_conservative}"
    w.close()


def test_signal_switch_rebuilds_detail_map(tmp_path):
    """细节识别切换后，信号图类型必须跟着换。"""
    w = _win(tmp_path)
    w.signal_combo.setCurrentIndex(1)          # 显著性主体
    w._on_signal_change()
    from photomosaic.saliency import SaliencyMap
    assert isinstance(w.detail_map, SaliencyMap)
    w.signal_combo.setCurrentIndex(0)          # 梯度能量
    w._on_signal_change()
    from photomosaic.render import DetailMap
    assert isinstance(w.detail_map, DetailMap)
    w.close()
