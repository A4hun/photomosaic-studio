"""Verify Directory-mode + non-native + forced ExtendedSelection supports
multi-select without the ExistingFiles 'Open navigates into dir' trap."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QItemSelectionModel  # noqa: E402
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFileDialog,  # noqa: E402
                               QListView, QTreeView)

app = QApplication.instance() or QApplication([])

tmp = tempfile.mkdtemp(prefix="pm_dir_multi_")
for n in ("aaa", "bbb", "ccc"):
    os.makedirs(os.path.join(tmp, n), exist_ok=True)
open(os.path.join(tmp, "afile.txt"), "w").close()

d = QFileDialog(None, "选择照片文件夹")
d.setFileMode(QFileDialog.FileMode.Directory)          # Directory: accept = "选择",不会进入目录
d.setOption(QFileDialog.Option.ShowDirsOnly, True)
d.setOption(QFileDialog.Option.DontUseNativeDialog, True)

# 关键：目录模式下内部视图默认单选，强制改扩展选择
for v in d.findChildren(QListView):
    v.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
for v in d.findChildren(QTreeView):
    v.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

d.setDirectory(tmp)
app.processEvents()

view = None
for v in d.findChildren(QAbstractItemView):
    if v.model() and v.model().rowCount(v.rootIndex()) > 0:
        view = v
        break
assert view is not None, "no populated internal view"
print("using view:", view.objectName(), "| selectionMode:",
      view.selectionMode())

model = view.model()
root = view.rootIndex()
sel = view.selectionModel()
count = 0
for row in range(model.rowCount(root)):
    idx = model.index(row, 0, root)
    if str(model.data(idx)) in ("aaa", "bbb"):
        sel.select(idx, QItemSelectionModel.SelectionFlag.Select
                   | QItemSelectionModel.SelectionFlag.Rows)
        count += 1
print("selected rows:", count)
files = d.selectedFiles()
print("selectedFiles():", files)
ok = len([f for f in files if os.path.basename(f) in ("aaa", "bbb")]) == 2
print("MULTI-SELECT WORKS (Directory mode)" if ok else "FAILED")
sys.exit(0 if ok else 1)
