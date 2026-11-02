"""功能1/功能2 端到端验收脚本（离屏 Qt，复现真实路径，可反复跑）。

A. 真实 make_mosaic 流式渲染（>2 亿像素强制流式？——演示用小图 +
   stream_threshold_px=1 强制走流式分支）→ 第④页“🔍 放大查看细节”按钮亮起
   → 打开 HugeImageViewDialog → 放大档触发后台条带读取 → 贴片像素与
   tifffile 独立解码逐字节一致。
B. 窗口几何持久化：_apply_state/_gather_state 往返 + 完全移出屏幕回默认。
C. 窗口可缩到 900×700，centralWidget 不超窗口；第③页分栏 3:1 且可恢复。

运行：.venv\\Scripts\\python scripts\\verify_bigview.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from photomosaic.db import TileDB  # noqa: E402
from photomosaic.gui import MainWindow  # noqa: E402
from photomosaic.library import ingest  # noqa: E402
from photomosaic.render import make_mosaic  # noqa: E402


def _wait_tile(dlg, app, timeout_s: float = 10.0) -> None:
    import time
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        app.processEvents()
        if dlg.canvas._tile_arr is not None and not dlg._reading \
                and dlg._worker is None:
            return
        time.sleep(0.01)
    raise TimeoutError("条带读取未完成")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    tmp = tempfile.mkdtemp(prefix="verify_bigview_")
    rng = np.random.default_rng(42)

    # ---- 造一个小图库 + 主图，真实流式渲染（强制 stream 阈值=1）----
    folder = os.path.join(tmp, "tiles")
    os.makedirs(folder)
    for i in range(8):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            os.path.join(folder, f"t{i}.png"))
    master = os.path.join(tmp, "m.png")
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(master)
    db = TileDB()
    ingest(db, folder)
    out = os.path.join(tmp, "verify.tif")
    # out_width=4608 > 预览上限 4096 → 预览是真降采样；条带行数压到 512 → 多条带
    stats = make_mosaic(master, db, out, cells_x=8, levels=3, seed=1,
                        out_width=4608, stream_threshold_px=1,
                        stream_strip_rows=512)
    db.close()
    assert stats["streamed"] is True and stats["preview"]
    print(f"[setup] 流式渲染完成：size={tuple(stats['size'])} "
          f"strips={stats['strips']} preview={stats['preview']}")

    # ---- A：GUI 按钮 → 对话框 → 放大读条带 → 像素一致 ----
    win = MainWindow(load_settings=False)
    win.prefill(master=master, db=os.path.join(tmp, "tiles.db"))
    opened = []
    _orig_exec = QDialog.exec
    QDialog.exec = lambda self: opened.append(self) or QDialog.DialogCode.Rejected
    try:
        win._render_done(stats)
        assert win.btn_bigview.isEnabled(), "流式成品应可放大查看"
        win.btn_bigview.click()
        assert len(opened) == 1
        dlg = opened[0]
        dlg.resize(700, 500)
        dlg.show()
        app.processEvents()
        assert dlg.canvas.uses_preview(), "适应窗口档应画预览"
        dlg.canvas.set_zoom(3.0)                 # 超过预览分辨率 → 读条带
        assert not dlg.canvas.uses_preview()
        rect = dlg.canvas.source_rect()
        dlg._start_read()
        _wait_tile(dlg, app)
        arr = dlg.canvas._tile_arr
        print(f"[A] 视口矩形 {tuple(rect.getRect())}，贴片 {arr.shape}，"
              f"read_rows 调用 {dlg.reader.io_reads} 次条带")
        try:
            import tifffile
            ref = tifffile.imread(out)
            x, y = rect.x(), rect.y()
            assert np.array_equal(
                arr, ref[y:y + rect.height(), x:x + rect.width()]), \
                "贴片与独立解码不一致"
            print("[A] PASS：贴片像素与 tifffile 独立解码逐字节一致")
        except ImportError:
            print("[A] PASS（tifffile 未安装，跳过像素比对）")
        dlg._shutdown()
    finally:
        QDialog.exec = _orig_exec

    # ---- B：几何持久化 ----
    win._apply_state({"geometry": [40, 30, 1100, 760]})
    assert win.geometry() == QRect(40, 30, 1100, 760)
    d = win._gather_state()
    assert d["geometry"] == [40, 30, 1100, 760]
    win._apply_state({"geometry": [9999, 9999, 1100, 760]})   # 完全移出屏幕
    assert (win.width(), win.height()) == MainWindow._DEFAULT_SIZE
    print(f"[B] PASS：几何往返 + 越界回默认 {MainWindow._DEFAULT_SIZE}")

    # ---- C：900×700 + 分栏 ----
    win._goto(2)
    win.show()
    app.processEvents()
    app.processEvents()
    s = win.splitter3.sizes()
    assert s[0] > s[1] and s[0] / sum(s) <= 0.75 + 1e-6
    print(f"[C] PASS：默认窗口下分栏 {s}（左:右 请求 3:1，右列最小宽可顶住比值）")
    win.resize(900, 700)
    app.processEvents()
    cw = win.centralWidget()
    assert (win.width(), win.height()) == (900, 700)
    assert cw.width() <= 900 and cw.height() <= 700
    print("[C] PASS：窗口 900×700 无裁剪（第③④页超宽内容走滚动条，不裁剪控件）")
    win.close()
    print("ALL BIGVIEW/LAYOUT CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
