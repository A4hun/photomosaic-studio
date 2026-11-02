"""离屏截取 MainWindow 四页截图 → docs/img/page1.png…page4.png。

用法：.venv\Scripts\python scripts\capture_screenshots.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from photomosaic.gui import MainWindow  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
os.makedirs(OUT, exist_ok=True)


def grab(win, name):
    pix = win.grab()
    pix.save(os.path.join(OUT, name))
    print("saved", name)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    win.resize(1280, 820)
    win.show()
    app.processEvents()
    for i, page in enumerate(("page1.png", "page2.png", "page3.png", "page4.png")):
        win._goto(i)
        app.processEvents()
        grab(win, page)
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
