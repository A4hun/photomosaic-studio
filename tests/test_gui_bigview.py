"""功能1 GUI 接线测试：第④页“放大查看细节 / 用系统程序打开”按钮。

离屏（QT_QPA_PLATFORM=offscreen），不弹真实窗口：
* 按钮启用状态：仅流式 BigTIFF 成品可放大查看；系统程序打开只要成品在；
* 点击放大查看会打开 HugeImageViewDialog（文件、预览、尺寸都对上）；
* 系统程序打开调用 QDesktopServices.openUrl（本地文件 URL）。
"""

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from photomosaic.db import TileDB  # noqa: E402
from photomosaic.gui import MainWindow  # noqa: E402
from photomosaic.library import ingest  # noqa: E402
from photomosaic.render import make_mosaic  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """小图库 + 主图；真实 make_mosaic（threshold=1）拿流式 stats。"""
    rng = np.random.default_rng(5)
    tmp = tmp_path_factory.mktemp("bigwire")
    folder = tmp / "tiles"
    folder.mkdir()
    for i in range(6):
        Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(
            folder / f"t{i}.png")
    master = tmp / "m.png"
    Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(master)
    db = TileDB()
    ingest(db, str(folder))
    yield str(master), db, tmp
    db.close()


def _streamed(master, db, tmp, name="wire.tif"):
    out = tmp / name
    stats = make_mosaic(master, db, str(out), cells_x=6, top_k=3, seed=1,
                        stream_threshold_px=1)
    assert stats["streamed"] is True and stats["preview"]
    return stats, str(out)


def test_bigview_button_enabled_states(env):
    """放大查看仅流式启用；系统程序打开对任何成品启用。"""
    master, db, tmp = env
    stats, _ = _streamed(master, db, tmp)
    plain_stats = make_mosaic(master, db, str(tmp / "plain.png"), cells_x=6,
                              top_k=3, seed=1, stream_threshold_px=10 ** 12)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        assert not win.btn_bigview.isEnabled()
        assert not win.btn_open_ext.isEnabled()

        win._render_done(stats)
        assert win.btn_bigview.isEnabled()
        assert win.btn_open_ext.isEnabled()
        assert win._preview_path == stats["preview"]
        assert win._last_output == stats["output"]

        win._render_done(plain_stats)          # 非流式：放大查看禁用
        assert not win.btn_bigview.isEnabled()
        assert win.btn_open_ext.isEnabled()    # 普通成品仍可交给系统程序
        assert win._preview_path is None
    finally:
        win.close()


def test_click_bigview_opens_dialog_with_tif_and_preview(env, monkeypatch):
    master, db, tmp = env
    stats, _ = _streamed(master, db, tmp)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    opened = []
    monkeypatch.setattr("photomosaic.gui.HugeImageViewDialog.exec",
                        lambda self: opened.append(self)
                        or QDialog.DialogCode.Rejected)
    try:
        win._render_done(stats)
        win.btn_bigview.click()
        assert len(opened) == 1
        dlg = opened[0]
        try:
            # 对话框接到的是真成品、真预览、真实尺寸
            assert os.path.samefile(dlg.reader.path, stats["output"])
            assert dlg._own_reader is True
            assert (dlg.reader.width, dlg.reader.height) == tuple(stats["size"])
            assert dlg.canvas.preview is not None
            assert not dlg.canvas.preview.isNull()
            assert os.path.basename(stats["output"]) in dlg.windowTitle()
            # 真实流式成品也能按需读条带：放大超过预览分辨率后贴片就位
            # （此例输出 384px 预览即全分辨率 → 100% 仍是预览档，2× 才读条带）
            dlg.resize(600, 400)
            dlg.show()
            app.processEvents()
            dlg.canvas.set_zoom(2.0)
            assert not dlg.canvas.uses_preview()
            rect = dlg.canvas.source_rect()
            dlg._start_read()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 10.0:
                app.processEvents()
                if dlg.canvas._tile_arr is not None and not dlg._reading \
                        and dlg._worker is None:
                    break
                time.sleep(0.01)
            assert dlg.canvas._tile_arr is not None
            assert dlg.reader.io_reads >= 1
            assert dlg.canvas._tile_rect.y() == rect.y()
        finally:
            dlg._shutdown()      # 关掉对话框自建的 reader（Windows 文件句柄）
    finally:
        win.close()


def test_bigview_click_with_missing_file_is_safe(env, monkeypatch, tmp_path):
    """成品被删后点击：不弹对话框，状态栏说明。"""
    master, db, tmp = env
    stats, out = _streamed(master, db, tmp, name="gone.tif")
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    opened = []
    monkeypatch.setattr("photomosaic.gui.HugeImageViewDialog.exec",
                        lambda self: opened.append(self)
                        or QDialog.DialogCode.Rejected)
    try:
        win._render_done(stats)
        os.remove(out)
        win.btn_bigview.click()      # 成品已删：点击安全降级，不弹对话框
        assert opened == []
        win._open_bigview()          # 直接调也安全
        assert opened == []
        assert "不在磁盘" in win.status_text.text()
    finally:
        win.close()


def test_open_output_externally_uses_desktop_services(env, monkeypatch):
    master, db, tmp = env
    stats, out = _streamed(master, db, tmp)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    urls = []
    monkeypatch.setattr("photomosaic.gui.QDesktopServices.openUrl",
                        lambda url: urls.append(url))
    try:
        win._render_done(stats)
        win.btn_open_ext.click()
        assert len(urls) == 1
        # QUrl.toLocalFile 给正斜杠；归一化后与成品路径同一文件
        assert os.path.normcase(os.path.normpath(urls[0].toLocalFile())) \
            == os.path.normcase(os.path.normpath(out))
        assert "系统程序" in win.status_text.text()
    finally:
        win.close()
