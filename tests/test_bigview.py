"""超大图分段查看器（bigview.HugeImageViewDialog）离屏冒烟测试。

用 BigTiffStripWriter 造 8000×6000、8 条带（rps=750）的假流式成品 +
4096 宽预览，程序化驱动对话框内部方法（不依赖真实滚轮事件）：
* 适应窗口档用预览、不读条带；
* 放大档触发 read_rows 且返回区域与写入数据逐字节一致（注入记录调用的
  reader 包装）；
* 同一视口不重复读；滚轮缩放的锚点数学正确；Esc 即退。
"""

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from photomosaic.bigtiff import BigTiffStripReader, BigTiffStripWriter  # noqa: E402
from photomosaic.bigview import HugeImageViewDialog  # noqa: E402

W, H, RPS = 8000, 6000, 750            # 8 条带


def _expected_region(x0, y0, x1, y1):
    """已知图案的期望像素（与写入端同一公式，按需生成不占整图内存）。"""
    yy, xx = np.mgrid[y0:y1, x0:x1]
    a = np.empty((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    a[..., 0] = (xx + yy) % 256
    a[..., 1] = (yy * 3) % 256
    a[..., 2] = (xx ^ yy) % 256
    return a


class RecordingReader(BigTiffStripReader):
    """记录 read_rows 调用的包装（测试注入用）。"""

    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        self.calls = []

    def read_rows(self, y0, y1):
        self.calls.append((int(y0), int(y1)))
        return super().read_rows(y0, y1)


@pytest.fixture(scope="module")
def streamed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bigview")
    tif = tmp / "huge.tif"
    full = np.empty((H, W, 3), dtype=np.uint8)
    with BigTiffStripWriter(str(tif), W, H, RPS) as wr:
        assert wr.n_strips == 8
        for i in range(wr.n_strips):
            y0 = i * RPS
            rows = wr.rows_of(i)
            yy, xx = np.mgrid[y0:y0 + rows, 0:W]
            strip = np.empty((rows, W, 3), dtype=np.uint8)
            strip[..., 0] = (xx + yy) % 256
            strip[..., 1] = (yy * 3) % 256
            strip[..., 2] = (xx ^ yy) % 256
            wr.write_strip(i, strip)
            full[y0:y0 + rows] = strip
    preview = tmp / "huge.preview.png"
    Image.fromarray(full).resize((4096, 3072), Image.Resampling.BOX).save(
        str(preview))
    del full
    return str(tif), str(preview)


def _make_dialog(streamed, app, **kw):
    tif, preview = streamed
    reader = RecordingReader(tif, cache_strips=2)
    dlg = HugeImageViewDialog(tif, preview, (W, H), reader=reader, **kw)
    dlg.resize(900, 640)
    dlg.show()
    for _ in range(8):
        app.processEvents()
    return dlg, reader


def _wait_calls(reader, dlg, app, n, timeout=10.0):
    """等到第 n 次读取完成且结果已送回 GUI 线程（worker 清空、状态复位）。"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        if len(reader.calls) >= n and dlg._worker is None and not dlg._reading:
            return True
        time.sleep(0.01)
    return False


def test_dialog_title_shows_name_and_size(streamed):
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        assert "huge.tif" in dlg.windowTitle()
        assert "8000" in dlg.windowTitle() and "6000" in dlg.windowTitle()
        assert dlg.isModal()
    finally:
        dlg.reject()
        reader.close()


def test_fit_uses_preview_without_reading(streamed):
    """适应窗口档：缩放低于预览分辨率 → 只画预览，一次条带都不读。"""
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        assert dlg.canvas.uses_preview()
        assert dlg.canvas.zoom() <= dlg.canvas.preview_scale()
        dlg._start_read()                 # 即便直接催读：预览够用也不读
        assert reader.calls == []
        assert dlg.status_label.text().endswith("%")
    finally:
        dlg.reject()
        reader.close()


def test_zoom_100_reads_correct_region(streamed):
    """放大到 100%：read_rows 覆盖视口行区间，贴片逐字节等于写入数据。"""
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        dlg.canvas.set_zoom(1.0)
        assert not dlg.canvas.uses_preview()
        rect = dlg.canvas.source_rect()
        assert 0 <= rect.x() and rect.x() + rect.width() <= W
        assert 0 <= rect.y() and rect.y() + rect.height() <= H
        assert dlg.status_label.text().startswith("100%")

        dlg._start_read()
        assert _wait_calls(reader, dlg, app, 1), "后台读取未在超时内完成"
        assert reader.calls == [(rect.y(), rect.y() + rect.height())]
        # read_rows 整宽返回 → 对话框按视口 x 裁剪：贴片即期望区域
        assert dlg.canvas._tile_rect == rect
        assert np.array_equal(
            dlg.canvas._tile_arr, _expected_region(
                rect.x(), rect.y(), rect.x() + rect.width(),
                rect.y() + rect.height()))
        assert dlg.canvas._tile_pix.width() == rect.width()
        assert "读取条带" not in dlg.status_label.text()
    finally:
        dlg.reject()
        reader.close()


def test_same_viewport_not_read_twice(streamed):
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        dlg.canvas.set_zoom(1.0)
        dlg._start_read()
        assert _wait_calls(reader, dlg, app, 1)
        n = len(reader.calls)
        dlg._start_read()                 # 同一视口再催 → 不重复读
        for _ in range(20):
            app.processEvents()
        assert len(reader.calls) == n
    finally:
        dlg.reject()
        reader.close()


def test_pan_then_read_new_region(streamed):
    """平移后视口变了：去抖后读取新行区间（行号随平移改变）。"""
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        dlg.canvas.set_zoom(1.0)
        dlg._start_read()
        assert _wait_calls(reader, dlg, app, 1)
        first = reader.calls[-1]
        # 程序化平移：把 pan 移到图像底部（模拟向下拖拽到底）
        dlg.canvas._pan_y = dlg.canvas.height() - H * 1.0
        dlg.canvas._clamp_pan()
        dlg.canvas.viewport_changed.emit()
        dlg._start_read()
        assert _wait_calls(reader, dlg, app, 2), "平移后的新视口未被读取"
        second = reader.calls[-1]
        assert second != first
        assert second[1] <= H
        rect = dlg.canvas.source_rect()
        assert np.array_equal(
            dlg.canvas._tile_arr, _expected_region(
                rect.x(), rect.y(), rect.x() + rect.width(),
                rect.y() + rect.height()))
    finally:
        dlg.reject()
        reader.close()


def test_zoom_anchor_keeps_source_point(streamed):
    """滚轮缩放锚点数学：锚点下的源像素在缩放前后不动。"""
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        anchor = QPointF(300.0, 200.0)
        z0 = dlg.canvas.zoom()
        px, py = dlg.canvas.pan()
        src_x = (anchor.x() - px) / z0
        src_y = (anchor.y() - py) / z0
        dlg.canvas.set_zoom(z0 * 2.0, anchor)
        z1, (px1, py1) = dlg.canvas.zoom(), dlg.canvas.pan()
        assert abs(z1 - z0 * 2.0) < 1e-6
        assert abs((anchor.x() - px1) / z1 - src_x) < 1e-6
        assert abs((anchor.y() - py1) / z1 - src_y) < 1e-6
    finally:
        dlg.reject()
        reader.close()


def test_double_click_toggle_fit_and_100(streamed):
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        dlg.canvas.mouseDoubleClickEvent(None)    # 适应 → 100%
        assert abs(dlg.canvas.zoom() - 1.0) < 5e-3
        dlg.canvas.mouseDoubleClickEvent(None)    # 100% → 适应
        assert abs(dlg.canvas.zoom() - dlg.canvas.fit_zoom()) < 1e-6
    finally:
        dlg.reject()
        reader.close()


def test_escape_rejects(streamed):
    """Esc（QDialog 默认）即退：隐藏 + Rejected。"""
    app = QApplication.instance() or QApplication([])
    dlg, reader = _make_dialog(streamed, app)
    try:
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                       Qt.KeyboardModifier.NoModifier)
        dlg.keyPressEvent(ev)
        assert dlg.result() == QDialog.DialogCode.Rejected
        assert dlg.isHidden()
    finally:
        reader.close()
