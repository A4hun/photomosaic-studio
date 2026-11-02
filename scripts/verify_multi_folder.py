"""Verify the canonical Qt multi-folder-selection idiom actually works.

The industry-standard approach (top SO answer for this exact problem):
    setFileMode(ExistingFiles) + ShowDirsOnly + DontUseNativeDialog
— ExistingFiles enables multi-select, ShowDirsOnly filters to dirs only.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QItemSelectionModel  # noqa: E402
from PySide6.QtWidgets import QAbstractItemView, QApplication, QFileDialog  # noqa: E402

app = QApplication.instance() or QApplication([])

tmp = tempfile.mkdtemp(prefix="pm_multi_")
for n in ("aaa", "bbb", "ccc"):
    os.makedirs(os.path.join(tmp, n), exist_ok=True)
open(os.path.join(tmp, "afile.txt"), "w").close()  # should be hidden by ShowDirsOnly

d = QFileDialog(None, "多选文件夹")
d.setFileMode(QFileDialog.FileMode.ExistingFiles)
d.setOption(QFileDialog.Option.ShowDirsOnly, True)
d.setOption(QFileDialog.Option.DontUseNativeDialog, True)
d.setDirectory(tmp)

views = [v for v in d.findChildren(QAbstractItemView)
         if v.objectName() in ("treeView", "listView")]
assert views, "no internal view found"
app.processEvents()
# pick whichever view actually has rows (list vs detail mode differ)
view = None
for v in views:
    if v.model().rowCount(v.rootIndex()) > 0:
        view = v
        break
if view is None:
    view = views[-1]  # fallback: try last (treeView) anyway
print("using view:", view.objectName())

model = view.model()
root = view.rootIndex()
if model.rowCount(root) == 0:
    # detail-mode treeView may need the dialog's directory watcher tick
    app.processEvents()
    import time
    time.sleep(0.5)
    app.processEvents()
names = []
sel = view.selectionModel()
for row in range(model.rowCount(root)):
    idx = model.index(row, 0, root)
    name = str(model.data(idx))
    names.append(name)
    if name in ("aaa", "bbb"):
        sel.select(idx, QItemSelectionModel.SelectionFlag.Select
                   | QItemSelectionModel.SelectionFlag.Rows)
print("rows visible:", names)
assert "afile.txt" not in names, "ShowDirsOnly failed: file visible"
assert {"aaa", "bbb", "ccc"} <= set(names), "dirs missing"

files = d.selectedFiles()
print("selectedFiles():", files)
ok = (len([f for f in files if os.path.basename(f) in ("aaa", "bbb")]) == 2)
print("MULTI-SELECT WORKS" if ok else "MULTI-SELECT FAILED")
sys.exit(0 if ok else 1)
