"""功能2 测试：窗口几何持久化、第③页 QSplitter 分栏、窗口可缩到 900×700。

离屏（QT_QPA_PLATFORM=offscreen），不弹真实窗口：
* geometry [x,y,w,h] / maximized 经 _gather_state/_apply_state（以及真实
  settings 文件启动→关闭→再启动）往返；
* 窗口完全移出屏幕 → 回默认 1280×820；部分越界 → 夹取回可用区域；
* 第③页分栏 setSizes → saveState(base64) → restoreState 往返；损坏 blob
  安全忽略；
* resize(900,700) 后 centralWidget 不超出窗口（布局最小尺寸达标）。
"""

import json
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, QSize, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from photomosaic.gui import MainWindow  # noqa: E402


def test_geometry_roundtrip_and_file(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._apply_state({"geometry": [40, 30, 1100, 760], "maximized": False})
        assert win.geometry() == QRect(40, 30, 1100, 760)
        d = win._gather_state()
        assert d["geometry"] == [40, 30, 1100, 760]
        assert d["maximized"] is False
        p = tmp_path / "state.json"
        win._save_state(str(p))
        win2 = MainWindow(load_settings=False)
        try:
            win2._apply_state(json.loads(p.read_text(encoding="utf-8")))
            assert win2.geometry() == QRect(40, 30, 1100, 760)
        finally:
            win2.close()
    finally:
        win.close()


def test_settings_file_startup_roundtrip(tmp_path, monkeypatch):
    """真实启动路径：关闭时写入 SETTINGS_PATH，下次构造即恢复。"""
    p = tmp_path / "settings.json"
    monkeypatch.setattr("photomosaic.gui.SETTINGS_PATH", str(p))
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=True)      # 文件尚不存在 → 不加载
    try:
        win._apply_state({"geometry": [40, 30, 1100, 760]})
        win.close()                           # closeEvent → 保存
        assert p.exists()
    finally:
        win.close()
    win2 = MainWindow(load_settings=True)     # 启动恢复
    try:
        assert win2.geometry() == QRect(40, 30, 1100, 760)
    finally:
        win2.close()


def test_offscreen_geometry_falls_back_to_default():
    """窗口完全移出屏幕 → 回默认 1280×820 并放回可用区域内。"""
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._apply_state({"geometry": [9999, 9999, 1100, 760]})
        assert win.size() == QSize(*MainWindow._DEFAULT_SIZE)
        avail = win._available_geometry()
        assert avail.intersects(win.geometry())   # 至少一部分在屏内
    finally:
        win.close()


def test_partially_offscreen_geometry_is_clamped():
    """部分越界：位置夹取回可用区域，尺寸保留。"""
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        avail = win._available_geometry()         # offscreen 约 800×800
        # 放到右下角外一点：窗口大部分仍在屏内 → 只夹位置
        x = avail.x() + avail.width() - 60
        y = avail.y() + avail.height() - 60
        win._apply_state({"geometry": [x, y, 260, 240]})
        g = win.geometry()
        assert (g.width(), g.height()) == (260, 240)
        assert g.x() <= avail.x() + avail.width() - 260
        assert g.y() <= avail.y() + avail.height() - 240
        assert g.x() >= avail.x() and g.y() >= avail.y()
    finally:
        win.close()


def test_maximized_roundtrip():
    """maximized 用窗口状态位恢复；geometry 仍存“还原后”的正常几何。"""
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._apply_state({"geometry": [10, 10, 1000, 700], "maximized": True})
        assert bool(win.windowState() & Qt.WindowState.WindowMaximized)
        d = win._gather_state()
        assert d["maximized"] is True
        assert d["geometry"] == [10, 10, 1000, 700]
    finally:
        win.close()


def test_splitter3_state_roundtrip():
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win.splitter3.setSizes([600, 200])
        sizes1 = win.splitter3.sizes()
        assert sum(sizes1) > 0
        d = win._gather_state()
        assert isinstance(d["splitter3"], str) and d["splitter3"]
        win2 = MainWindow(load_settings=False)
        try:
            win2._apply_state(d)
            assert win2.splitter3.sizes() == sizes1
        finally:
            win2.close()
    finally:
        win.close()


def test_splitter3_corrupt_blob_is_ignored():
    """损坏的 base64 状态：restoreState 失败 → 忽略，不抛异常。"""
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        before = win.splitter3.sizes()
        win._apply_state({"splitter3": "###不是 base64###"})
        assert win.splitter3.sizes() == before
        # 恢复后分栏仍可正常工作
        win.splitter3.setSizes([500, 300])
        assert sum(win.splitter3.sizes()) > 0
    finally:
        win.close()


def test_splitter3_drag_then_save():
    """模拟拖动分栏（程序化 setSizes）后状态进入 _gather_state 可恢复。"""
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._goto(2)                             # 激活第③页，分栏才有真实宽度
        win.show()
        app.processEvents()
        app.processEvents()                      # 让初始比例先就位
        win.splitter3.setSizes([600, 200])
        saved = win.splitter3.sizes()
        assert saved[0] > saved[1]
        d = win._gather_state()
        win.splitter3.setSizes([200, 600])       # 再乱动一下
        win._apply_state(d)                      # 恢复"拖动后"的分布
        assert win.splitter3.sizes() == saved
    finally:
        win.close()


def test_splitter3_restored_on_first_show():
    """启动路径：_apply_state 存下分栏状态，首次显示后按真实宽度恢复。"""
    app = QApplication.instance() or QApplication([])
    win1 = MainWindow(load_settings=False)
    try:
        win1._goto(2)
        win1.show()
        app.processEvents()
        app.processEvents()
        win1.splitter3.setSizes([700, 230])
        target = win1.splitter3.sizes()
        d = win1._gather_state()
    finally:
        win1.close()
    win2 = MainWindow(load_settings=False)
    try:
        win2._goto(2)
        win2._apply_state(d)          # 启动时：隐藏状态下先记下/粗恢复
        win2.show()
        app.processEvents()
        app.processEvents()           # 首次显示 → singleShot(0) 精确恢复
        assert win2.splitter3.sizes() == target
    finally:
        win2.close()


def test_window_resizes_to_900x700_without_clipping():
    """窗口可缩到 900×700：布局最小尺寸达标，centralWidget 不超窗口。"""
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        assert win.minimumSizeHint().width() <= 900
        assert win.minimumSizeHint().height() <= 700
        win.show()
        app.processEvents()
        win.resize(900, 700)
        app.processEvents()
        assert (win.width(), win.height()) == (900, 700)
        cw = win.centralWidget()
        assert cw.width() <= win.width() and cw.height() <= win.height()
    finally:
        win.close()


def test_splitter3_present_with_3_to_1_stretch():
    """第③页是 QSplitter：显示后左边占多数（初始比例 3:1，右列最小宽度
    允许把比值顶住，但不会反过来）。"""
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        assert win.splitter3.count() == 2
        assert not win.splitter3.childrenCollapsible()
        win._goto(2)                 # 激活第③页，分栏才有真实宽度
        win.show()
        app.processEvents()
        app.processEvents()          # 让 singleShot(0) 的初始比例跑完
        s = win.splitter3.sizes()
        assert sum(s) > 0 and s[0] > s[1]
        # 请求的初始比例是 3:1；右列内容有最小宽度，实际比值允许被顶住
        assert s[0] / sum(s) <= 0.75 + 1e-6
    finally:
        win.close()
