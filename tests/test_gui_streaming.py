"""GUI 对流式 BigTIFF 输出的适配（A3）回归测试。

离屏运行（QT_QPA_PLATFORM=offscreen），不弹真实窗口：
* _render_done 后 _render_size/_render_streamed 记录正确；
* 流式 + TIFF 导出走文件复制（不整图解码）；
* 流式输出上的点击替换 / 非 TIFF 导出 / 海报分页诚实降级；
* 非流式替换保存按扩展名分支：JPEG 用设定质量，且临时文件原子替换；
* 对比模式把主图缩放到预览同宽，切换不跳变。
"""

import io
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from photomosaic.db import TileDB  # noqa: E402
from photomosaic.gui import MainWindow  # noqa: E402
from photomosaic.library import ingest  # noqa: E402
from photomosaic.render import make_mosaic  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """小图库 + 主图；离屏跑真实 make_mosaic（threshold=1）拿流式 stats。"""
    rng = np.random.default_rng(7)
    tmp = tmp_path_factory.mktemp("guistream")
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


def _make_streamed(master, db, tmp, name="gui_stream.tif"):
    """真实流式渲染（cells_x=6，不传 levels → uniform 流式分支）。"""
    out = tmp / name
    return make_mosaic(master, db, str(out), cells_x=6, top_k=3, seed=1,
                       stream_threshold_px=1), out


def test_render_done_tracks_streamed_state(env):
    master, db, tmp = env
    stats, _ = _make_streamed(master, db, tmp)
    assert stats["streamed"] is True
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._render_done(stats)
        assert win._render_streamed is True
        assert tuple(win._render_size) == tuple(stats["size"])
        assert win._last_output == stats["output"]
        assert os.path.exists(win._last_output)
        assert win._result_pix.width() <= 4096   # 预览像素图，不是大图本身
        # 非流式渲染完成后标志复位
        png = tmp / "plain.png"
        s2 = make_mosaic(master, db, str(png), cells_x=6, top_k=3, seed=1,
                         stream_threshold_px=10 ** 12)
        assert s2.get("streamed") is None
        win._render_done(s2)
        assert win._render_streamed is False
        assert tuple(win._render_size) == tuple(s2["size"])
    finally:
        win.close()


def test_export_streamed_tiff_copies(env, monkeypatch, tmp_path):
    """流式 + TIFF：直接 shutil.copy 原样导出，不整图载入。"""
    master, db, tmp = env
    stats, _ = _make_streamed(master, db, tmp)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._render_done(stats)
        win.format_combo.setCurrentText("TIFF")
        target = str(tmp_path / "exported.tif")
        monkeypatch.setattr("photomosaic.gui.QFileDialog.getSaveFileName",
                            lambda *a, **k: (target, ""))
        real_copy = shutil.copy
        calls = []
        monkeypatch.setattr("photomosaic.gui.shutil.copy",
                            lambda src, dst: calls.append((src, dst))
                            or real_copy(src, dst))
        win._export()
        assert calls and calls[0][0] == stats["output"]
        assert Path(target).read_bytes() == Path(stats["output"]).read_bytes()
        msg = win.status_text.text()
        assert "已导出" in msg
        assert f"{stats['size'][0]}×{stats['size'][1]}" in msg
    finally:
        win.close()


def test_replace_cell_streamed_degrades(env):
    """流式输出上点击替换：状态栏提示 + 不抛异常（不整图载入）。"""
    master, db, tmp = env
    stats, _ = _make_streamed(master, db, tmp)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._render_done(stats)
        leaf = stats["leaf_rects"][0]
        win._replace_cell(leaf)
        msg = win.status_text.text()
        assert "流式" in msg and "暂不支持" in msg
    finally:
        win.close()


def test_streamed_export_other_format_and_poster_degrade(env, monkeypatch):
    """流式 + JPEG：直接降级不弹对话框；海报分页同样诚实降级。"""
    master, db, tmp = env
    stats, _ = _make_streamed(master, db, tmp)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win._render_done(stats)
        opened = []
        real_open = Image.open
        monkeypatch.setattr("photomosaic.gui.Image.open",
                            lambda p, *a, **k: opened.append(str(p))
                            or real_open(p, *a, **k))
        dialogs = []
        monkeypatch.setattr("photomosaic.gui.QFileDialog.getSaveFileName",
                            lambda *a, **k: dialogs.append(a) or ("", ""))
        win.format_combo.setCurrentText("JPEG")
        win._export()
        assert not dialogs                       # 未弹保存对话框，直接降级
        assert "仅支持 TIFF" in win.status_text.text()
        win._poster(2, 2)
        assert "海报" in win.status_text.text()
        assert not opened                        # 全程没有整图 Image.open
    finally:
        win.close()


def test_replace_cell_save_jpeg_quality_and_atomic(tmp_path):
    """非流式替换保存：JPEG 用界面质量（非 Pillow 默认 75）、临时文件原子替换。"""
    rng = np.random.default_rng(11)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        out = tmp_path / "r.jpg"
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            out, "JPEG", quality=90)
        win._last_output = str(out)
        win.out_quality.setValue(100)
        canvas = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        ref100 = io.BytesIO()
        canvas.save(ref100, "JPEG", quality=100, subsampling=0)
        ref75 = io.BytesIO()
        canvas.save(ref75, "JPEG")
        win._save_output_atomic(canvas)
        data = out.read_bytes()
        assert data == ref100.getvalue()          # 用的是设定质量，不是默认 75
        assert data != ref75.getvalue()
        assert len(list(tmp_path.iterdir())) == 1  # 临时文件已原子替换，无残留
        with Image.open(out) as im:
            assert im.format == "JPEG"
        # TIFF 分支：显式 "TIFF" 格式保存
        out_tif = tmp_path / "r.tif"
        win._last_output = str(out_tif)
        win._save_output_atomic(canvas)
        assert out_tif.exists()
        with Image.open(out_tif) as im:
            assert im.format == "TIFF"
        assert len(list(tmp_path.iterdir())) == 2  # 依旧无临时文件残留
    finally:
        win.close()


def test_toggle_compare_scales_master_to_preview_width(tmp_path):
    """对比模式：主图 QPixmap 缩放到预览同宽，切换时尺寸不跳变。"""
    master = tmp_path / "cm.png"
    Image.new("RGB", (200, 160), (30, 60, 90)).save(master)
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    try:
        win.prefill(master=str(master))
        big = win._master_pixmap()
        assert big is not None and big.width() > 0
        win._result_pix = big.scaledToWidth(97)   # 模拟流式降采样预览（宽度不同）
        win._toggle_compare(True)
        assert win.result_view._pix.width() == 97  # 主图已缩到预览同宽
        win._toggle_compare(False)
        assert win.result_view._pix.width() == 97  # 切回结果不跳变
    finally:
        win.close()
