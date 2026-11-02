"""查看器关闭竞态：慢盘在途读的线程必须悬挂到跑完，不被销毁。"""

import time

import pytest

pytest.importorskip("PySide6")


def test_slow_reader_thread_orphaned_not_destroyed(tmp_path):
    import numpy as np
    from PIL import Image

    from photomosaic.bigview import HugeImageViewDialog, _ORPHAN_THREADS
    from photomosaic.bigtiff import BigTiffStripWriter

    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    # 造一个小 BigTIFF
    tif = tmp_path / "t.tif"
    W, H = 256, 256
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    with BigTiffStripWriter(str(tif), W, H, 64) as w:
        for si in range(w.n_strips):
            w.write_strip(si, arr[si * 64:(si + 1) * 64])

    # 慢读 reader：每条带读 2 秒
    from photomosaic.bigtiff import BigTiffStripReader

    class SlowReader(BigTiffStripReader):
        def read_rows(self, y0, y1):
            time.sleep(2.0)
            return super().read_rows(y0, y1)

    preview = Image.new("RGB", (64, 64), (128, 128, 128))
    dlg = HugeImageViewDialog(str(tif), preview, (W, H),
                              reader=SlowReader(str(tif)))
    dlg.show()
    # 触发一次放大读取（zoom 超过预览比率）
    dlg.canvas.full_size = (W, H)
    dlg.canvas.preview = None
    dlg.canvas._zoom = 2.0
    dlg.canvas.viewport_changed.emit()
    # 等去抖+线程启动
    t0 = time.time()
    while time.time() - t0 < 2 and dlg._worker is None:
        app.processEvents()
        time.sleep(0.05)
    assert dlg._worker is not None and dlg._worker.isRunning(), \
        "应有一个在途读线程"

    # 关闭对话框：线程还在跑 → 应被悬挂而不是销毁
    dlg.close()
    assert any(t.isRunning() for t in _ORPHAN_THREADS), \
        "在途读线程应进入悬挂注册表"

    # 等它跑完：自动从注册表移除（此后析构才安全）
    t0 = time.time()
    while any(t.isRunning() for t in _ORPHAN_THREADS) and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.1)
    assert not any(t.isRunning() for t in _ORPHAN_THREADS)
