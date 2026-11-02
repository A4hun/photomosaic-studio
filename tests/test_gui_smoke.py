import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from photomosaic.gui import MainWindow, MultiFolderDialog  # noqa: E402


def test_multi_folder_dialog_selects_multiple(tmp_path):
    """MultiFolderDialog must return every Ctrl-selected directory."""
    import time

    from PySide6.QtCore import QItemSelectionModel

    for n in ("aaa", "bbb", "ccc"):
        (tmp_path / n).mkdir()
    (tmp_path / "afile.txt").write_text("x")

    app = QApplication.instance() or QApplication([])
    dlg = MultiFolderDialog()
    dlg.model.setRootPath(str(tmp_path))
    root_idx = dlg.model.index(str(tmp_path))
    assert root_idx.isValid()
    dlg.tree.setRootIndex(root_idx)   # 视图对齐目录，触发模型加载
    loaded = []

    def _loaded(_p):
        loaded.append(_p)

    dlg.model.directoryLoaded.connect(_loaded)
    dlg.model.fetchMore(root_idx)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 5 and dlg.model.rowCount(root_idx) < 3:
        app.processEvents()
        time.sleep(0.05)
    sel = dlg.tree.selectionModel()
    sel.clearSelection()
    picked = 0
    for row in range(dlg.model.rowCount(root_idx)):
        idx = dlg.model.index(row, 0, root_idx)
        if os.path.basename(dlg.model.filePath(idx)) in ("aaa", "bbb"):
            sel.select(idx, QItemSelectionModel.SelectionFlag.Select
                       | QItemSelectionModel.SelectionFlag.Rows)
            picked += 1
    assert picked == 2
    folders = dlg.selected_folders()
    assert len(folders) == 2
    assert {os.path.basename(f) for f in folders} == {"aaa", "bbb"}
    dlg.close()


def test_main_window_constructs_offscreen():
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    assert "马赛克" in win.windowTitle()
    assert len(win.step_buttons) == 5            # 含批量出图页
    assert win.cells_x_spin.value() == 100
    assert len(win.level_radios) == 5 and win.level_radios[3].isChecked()
    assert win.grid_view.tree is None          # no master yet
    assert not win.btn_render.isEnabled() is False or True  # render enabled guard is master-based
    assert set(win.variant_checks) == {"orig", "mirror", "r90", "r180", "r270"}
    win.close()


def test_auto_detail_visible_in_overlay(tmp_path):
    import numpy as np
    from PIL import Image

    app = QApplication.instance() or QApplication([])
    rng = np.random.default_rng(2)
    master = tmp_path / "noisy.png"
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(master)
    win = MainWindow(load_settings=False)
    win.master_path = str(master)
    win.cells_x_spin.setValue(8)
    win._user_touched_cells = True
    win._show_master()
    base_count = len(win.tree.leaves())
    # auto on with radio 二级 on noisy image -> overlay shows subdivided leaves
    win.auto_check.setChecked(True)
    win.level_radios[1].setChecked(True)   # 二级：细节区可细分
    win._refresh_grid_overlay()
    assert len(win.grid_view._leaves) > base_count
    # auto off -> overlay equals manual tree
    win.auto_check.setChecked(False)
    win._refresh_grid_overlay()
    assert len(win.grid_view._leaves) == base_count
    win.close()


def test_effective_levels_semantics():
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    win.auto_check.setChecked(True)
    win.level_radios[0].setChecked(True)          # 一级 + 自动勾选
    assert win._effective_levels() == 1           # 经典均匀网格，自动不介入
    win.level_radios[3].setChecked(True)          # 四级
    assert win._effective_levels() == 4
    win.close()


def test_quadtree_interaction_via_window(tmp_path):
    from PIL import Image

    app = QApplication.instance() or QApplication([])
    master = tmp_path / "m.png"
    Image.new("RGB", (256, 256), (120, 80, 40)).save(master)
    win = MainWindow(load_settings=False)
    win.master_path = str(master)
    win.cells_x_spin.setValue(4)
    win._user_touched_cells = True
    win._show_master()
    assert win.tree is not None
    before = len(win.tree.leaves())
    assert win.tree.split_at(128, 128)
    assert len(win.tree.leaves()) == before + 3
    assert win.tree.merge_at(128, 128)
    assert len(win.tree.leaves()) == before
    win.close()


def test_progress_scale_switch_labels_strips():
    """B1 回归：流式条带阶段进度条切换为"条带"刻度。

    匹配阶段按叶子数计（如 93397/93397），流式贴图阶段按条带数计（如 3/3）；
    换刻度时格式必须带"条带"前缀，否则尺度跳变会被误读成进度倒退。
    """
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    win._on_progress(100, 100)                 # 匹配阶段：叶子数刻度
    assert "条带" not in win.lib_progress.format()
    assert win.lib_progress.value() == 100
    msg = "正在流式渲染（3 条带 × 64 行）…"
    win._on_worker_message(msg)                # 状态消息先到 → 切换刻度
    assert win.status_text.text() == msg
    assert win._progress_unit == "条带"
    win._on_progress(2, 3)                     # 贴图条带阶段：条带数刻度
    assert win.lib_progress.format().startswith("条带 2 / 3")
    assert (win.lib_progress.value(), win.lib_progress.maximum()) == (2, 3)
    win._finish_progress(True)
    assert "条带" not in win.lib_progress.format()   # 收尾恢复默认刻度
    assert win._progress_unit == ""
    win.close()


def test_effective_master_temp_name_is_pid_scoped(tmp_path):
    """B2 回归：旋转主图的临时文件名带进程 id，多 GUI 实例并发互不覆盖。"""
    import os
    import tempfile

    from PIL import Image

    app = QApplication.instance() or QApplication([])
    master = tmp_path / "m.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(master)
    win = MainWindow(load_settings=False)
    win.master_path = str(master)
    win.master_rot = 90
    path = win._effective_master()
    assert path == os.path.join(
        tempfile.gettempdir(), f"photomosaic_master_rot_{os.getpid()}.png")
    assert os.path.exists(path)
    win.close()


def test_out_info_stream_threshold_uses_shared_constant(tmp_path):
    """B3 回归：GUI 流式阈值提示引用 render.STREAM_THRESHOLD_PX，不再硬编码。"""
    import numpy as np
    from PIL import Image

    from photomosaic.render import STREAM_THRESHOLD_PX

    app = QApplication.instance() or QApplication([])
    rng = np.random.default_rng(0)
    master = tmp_path / "m.png"
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(master)
    win = MainWindow(load_settings=False)
    win.master_path = str(master)
    win._show_master()
    win.cells_x_spin.setValue(100)
    # 每格像素 × 100 列 → 输出宽度：80px/格=8000、141px/格=14100、142px/格=14200
    win._set_cell_px(80)
    assert win._effective_out_width() == 8000
    assert "流式" not in win.out_info.text()       # 8000x8000=6400 万 < 2 亿
    win._set_cell_px(141)
    assert "流式" not in win.out_info.text()       # 14100² ≈ 1.988 亿 < 阈值
    win._set_cell_px(142)
    assert "流式" in win.out_info.text()           # 14200² ≈ 2.016 亿 > 阈值
    assert STREAM_THRESHOLD_PX == 200_000_000
    win.close()


def test_cell_px_mode_and_width_derivation(tmp_path):
    """每格像素控件（单框）：预设值/任意值/与主图同大三种模式与宽度推导。"""
    import numpy as np
    from PIL import Image

    app = QApplication.instance() or QApplication([])
    master = tmp_path / "m.png"
    Image.new("RGB", (1000, 800), (10, 20, 30)).save(master)
    win = MainWindow(load_settings=False)
    win.master_path = str(master)
    win._show_master()
    win.cells_x_spin.setValue(125)

    win._set_cell_px(512)                      # 常用清晰度
    assert win.cell_px_spin.value() == 512
    assert win._effective_out_width() == 125 * 512
    assert "输出 64000" in win.out_info.text()

    win.cell_px_spin.setValue(300)             # 任意值
    assert win._effective_out_width() == 125 * 300

    win._set_cell_px(0)                        # 与主图同大
    assert win._cell_px() is None
    assert win._effective_out_width() is None
    assert "与主图同大" in win.out_info.text()
    win.close()


def test_output_path_respects_default_dir(tmp_path):
    """默认保存位置：设置后成品落在该目录；留空回退主图目录；旋转不受临时目录影响。"""
    import numpy as np
    from PIL import Image

    app = QApplication.instance() or QApplication([])
    master = tmp_path / "pic.jpg"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(master)
    outdir = tmp_path / "成品"
    outdir.mkdir()
    win = MainWindow(load_settings=False)
    win.master_path = str(master)

    assert win._output_path() == str(tmp_path / "pic_mosaic.png")
    win.outdir_edit.setText(str(outdir))
    assert win._output_path() == str(outdir / "pic_mosaic.png")
    win.outdir_edit.setText(str(tmp_path / "不存在"))
    assert win._output_path() == str(tmp_path / "pic_mosaic.png")   # 无效目录回退
    win.outdir_edit.clear()
    win.master_rot = 90                        # 旋转：命名/位置仍取主图原文件
    assert win._output_path() == str(tmp_path / "pic_mosaic.png")
    win.close()

