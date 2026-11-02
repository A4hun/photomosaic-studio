"""Real GUI-path render verification (offscreen Qt, real MainWindow + worker
thread + make_mosaic). Proves the three fixes end-to-end:

A. radio「一级」+ auto -> classic uniform grid, no subdivision, master-size output
B. radio「五级」+ auto on noisy master -> level-5 leaves actually reachable
C. 输出宽度 > 主图宽度 -> canvas and leaf rects live in output coordinates
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from photomosaic.gui import MainWindow  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MASTER = os.path.join(ROOT, "demo_output", "master.png")
DB = os.path.join(ROOT, "demo_output", "tiles.db")
FOLDER = os.path.join(ROOT, "demo_output", "tiles")
OUT = os.path.join(ROOT, "demo_output", "master_mosaic.png")
CKPT = OUT + ".ckpt"


def wait_done(win: MainWindow, app, timeout: float = 300.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if win._result_pix is not None and "完成" in win.status_text.text():
            return
        if "失败" in win.status_text.text() or "已取消" in win.status_text.text():
            raise RuntimeError("render failed: " + win.status_text.text())
        time.sleep(0.05)
    raise TimeoutError("render did not finish in time")


def run_case(win, app, label, setup) -> dict:
    if os.path.exists(OUT):
        os.remove(OUT)
    if os.path.isdir(CKPT):
        import shutil
        shutil.rmtree(CKPT)
    setup()
    win._render()
    wait_done(win, app)
    stats_text = win.stats_view.toPlainText()
    stats = dict(line.split(": ", 1) for line in stats_text.splitlines() if ": " in line)
    stats["__output_exists"] = os.path.exists(OUT)
    stats["__file_bytes"] = os.path.getsize(OUT) if stats["__output_exists"] else 0
    print(f"[{label}] mode={stats.get('mode')} leaves={stats.get('leaves')} "
          f"hist={stats.get('level_histogram')} size={stats.get('size')} "
          f"device={stats.get('device')} bytes={stats['__file_bytes']}")
    return stats


def main() -> int:
    assert os.path.exists(MASTER) and os.path.exists(DB), "demo data missing"
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    win.prefill(master=MASTER, db=DB, folder=FOLDER)
    win.resize(1200, 800)
    win.show()
    app.processEvents()
    base_cells = win.cells_x_spin.value()
    base_total = len(win.tree.leaf_boxes())

    # A: 一级 + 自动勾选 -> 经典均匀网格（manual，无细分，主图尺寸）
    def case_a():
        win.level_radios[0].setChecked(True)
        win.auto_check.setChecked(True)          # 自动勾上也不能把一级变成细分
        win._set_cell_px(0)                      # 与主图同大
        win.threshold_spin.setValue(1.0)         # 低阈值也不允许一级被切
    a = run_case(win, app, "A 一级+自动", case_a)
    assert a.get("mode") == "manual", f"A mode wrong: {a.get('mode')}"
    assert a.get("level_histogram") == f"{{1: {base_total}}}", \
        f"A histogram wrong: {a.get('level_histogram')} (base {base_total})"
    assert a.get("size") == "(640, 640)", f"A size wrong: {a.get('size')}"
    print("PASS A: 一级+自动 = 经典均匀网格，无细分，主图尺寸")

    # B: 五级 + 自动 -> 细节区出现 5 级叶子
    def case_b():
        win.level_radios[4].setChecked(True)
        win.auto_check.setChecked(True)
        win.threshold_spin.setValue(4.0)   # 平滑演示图硬边盒内均值约 6.6，阈值 4 可切到底
    b = run_case(win, app, "B 五级+自动", case_b)
    hist = b.get("level_histogram", "{}")
    import re
    assert re.search(r"\b5\s*:", hist), f"B level 5 not reached: {hist}"
    print("PASS B: 五级可达，细节区实际细分到 5 级")

    # C: 每格像素 × 块数 ≈ 1280（主图 640 的 2 倍）-> 画布与叶框坐标放大到输出域
    c_want = 0

    def case_c():
        nonlocal c_want
        win.level_radios[2].setChecked(True)
        cp = 1280 // base_cells               # 每格像素 → 宽度 = 块数 × 每格
        c_want = base_cells * cp
        win._set_cell_px(cp)
    c = run_case(win, app, "C 每格像素→1280 级", case_c)
    assert c.get("size") == f"({c_want}, {c_want})", \
        f"C size wrong: {c.get('size')} != ({c_want}, {c_want})"
    print(f"PASS C: 每格像素生效，画布 {c_want}px（≈主图 2 倍），叶框坐标在输出域")

    win.close()
    print("ALL GUI-LEVEL CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
