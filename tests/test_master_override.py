"""主图单独配置：覆盖参数的收集与换算。"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")


def test_batch_overrides_built_and_cellpx_converted(tmp_path):
    from PySide6.QtWidgets import QApplication
    from photomosaic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    m = tmp_path / "m.png"
    Image.fromarray(np.zeros((200, 160, 3), np.uint8)).save(m)
    w = MainWindow(load_settings=False)
    w.cells_x_spin.setValue(10)
    w.master_queue.addItem(str(m))
    w.master_queue.setCurrentRow(0)
    w._master_overrides[str(m)] = {"style": "hex", "cell_px": 100, "levels": 3}
    ov = w._batch_overrides([str(m)])
    assert ov is not None
    p = ov[str(m)]
    assert p["style"] == "hex" and p["levels"] == 3
    assert p["out_width"] == 10 * 100            # cell_px 换算成 out_width
    assert "cell_px" not in p
    w.close()


def test_batch_overrides_empty_when_none(tmp_path):
    from PySide6.QtWidgets import QApplication
    from photomosaic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow(load_settings=False)
    w.master_queue.addItem(str(tmp_path / "m.png"))
    assert w._batch_overrides([str(tmp_path / "m.png")]) is None
    w.close()
