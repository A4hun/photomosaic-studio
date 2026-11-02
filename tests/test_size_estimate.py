"""预估文件大小：按主图内容实测压缩率，PNG/JPEG 与真实输出偏差应在可接受范围。"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")


def _win_with_master(tmp_path, arr):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from photomosaic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    mp = tmp_path / "m.png"
    Image.fromarray(arr).save(mp)
    w = MainWindow(load_settings=False)
    w.master_path = str(mp)
    w._show_master()
    return w


def test_estimate_uses_measured_ratio(tmp_path):
    # 照片类噪声内容：压缩率低（~0.4+）；纯色内容：压缩率极低
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 255, (800, 1000, 3), dtype=np.uint8)
    w = _win_with_master(tmp_path, photo)
    w.format_combo.setCurrentText("PNG")
    w.cells_x_spin.setValue(15)
    w.cell_px_spin.setValue(810)
    # 实测主图样本的 PNG 压缩率
    ratio = w._master_sample_ratio("PNG")
    assert ratio > 0.3, f"噪声图 PNG 比率应较高，got {ratio}"
    ow = 15 * 810
    oh = round(800 * ow / 1000)
    px = ow * oh                               # 输出总像素（宽×高）
    est = px * ratio / 1e6                     # 期望的 MB 数
    assert "约" in w.out_info.text()
    # 文本里的 MB 数应与按比率计算的一致（±5%）
    import re
    m = re.search(r"约 (\d+(?:\.\d+)?) (MB|GB)", w.out_info.text())
    assert m, w.out_info.text()
    shown = float(m.group(1)) * (1000 if m.group(2) == "GB" else 1)
    assert abs(shown - est) / est < 0.05
    w.close()


def test_estimate_follows_format_and_quality(tmp_path):
    photo = np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
    w = _win_with_master(tmp_path, photo)
    w.cells_x_spin.setValue(15)
    w.cell_px_spin.setValue(810)
    w.format_combo.setCurrentText("PNG")
    png_txt = w.out_info.text()
    w.format_combo.setCurrentText("JPEG")
    w.out_quality.setValue(90)
    jpg_txt = w.out_info.text()
    assert png_txt != jpg_txt
    import re
    png_mb = float(re.search(r"(\d+(?:\.\d+)?) (MB|GB)", png_txt).group(1))
    jpg_mb = float(re.search(r"(\d+(?:\.\d+)?) (MB|GB)", jpg_txt).group(1))
    assert jpg_mb < png_mb            # JPEG 必须比 PNG 小
    w.out_quality.setValue(30)
    low = w.out_info.text()
    low_mb = float(re.search(r"(\d+(?:\.\d+)?) (MB|GB)", low).group(1))
    assert low_mb < jpg_mb            # 质量调低 → 预估更小
    w.close()


def test_exact_sizes_for_bmp_tiff(tmp_path):
    photo = np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
    w = _win_with_master(tmp_path, photo)
    w.cells_x_spin.setValue(15)
    w.cell_px_spin.setValue(810)
    px = 15 * 810 * (810 * 800 // 1000)
    for fmt in ("BMP", "TIFF"):
        w.format_combo.setCurrentText(fmt)
        txt = w.out_info.text()
        assert "≈" in txt and "约" not in txt, f"{fmt} 应为精确值：{txt}"
    w.close()
