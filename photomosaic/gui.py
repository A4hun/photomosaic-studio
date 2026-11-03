"""中文图形界面（PySide6）。

四步向导布局，对齐原版软件的功能结构：
  ① 主图像（预览/旋转）
  ② 单元图数据库（多文件夹、去重、变体、缩略图质量）
  ③ 网格与嵌套层级（基础块数、1-5 级、自动细节、半径、色彩融合、
     网格编辑器：左键分割 / 右键合并、按级全合并）
  ④ 生成与修正（缩放、点击碎片从匹配列表替换、多格式导出、海报分页）
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
import shutil
import sys
import tempfile
import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops
from PySide6.QtCore import (QByteArray, QItemSelectionModel, QPoint, QRect,
                            QSize, Qt, QThread, QTimer, QUrl, Signal)
from PySide6.QtGui import (QColor, QDesktopServices, QGuiApplication, QIcon,
                            QImage, QPainter, QPen, QPixmap, QPolygon,
                            QTransform)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QAbstractItemView
try:  # Qt6: QtGui; Qt5/旧 PySide6: QtWidgets
    from PySide6.QtGui import QFileSystemModel
except ImportError:
    from PySide6.QtWidgets import QFileSystemModel

from .bigview import HugeImageViewDialog
from .db import TileDB
from .grid import QuadTree
from .library import VARIANTS, IngestControl, ingest
from .match import Matcher
from .render import (STREAM_THRESHOLD_PX, DetailMap, RenderCancelled,
                     adaptive_leaves, export_poster, make_mosaic, rect_features)

Image.MAX_IMAGE_PIXELS = None          # 印刷级大图是预期用法
try:
    from PySide6.QtGui import QImageReader
    QImageReader.setAllocationLimit(0)  # Qt 默认拒绝 >256MB 的图片
except Exception:
    pass

ACCENT = "#4f46e5"
ACCENT_DARK = "#4338ca"
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".photomosaic_studio.json")


def _hint(text: str) -> QLabel:
    """Small gray explanatory note shown under a settings row."""
    lbl = QLabel(text)
    lbl.setObjectName("hint")
    lbl.setWordWrap(True)
    # 允许换行扩展高度、不顶住布局宽度（修复窄窗口/长文案被截断）
    lbl.setMinimumWidth(0)
    lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding)
    return lbl


def _shrinkable(lbl: QLabel, wrap: bool = False) -> QLabel:
    """信息类 QLabel 不再用整段文本宽度顶住布局（窄窗口可截断/换行）。"""
    lbl.setMinimumWidth(0)
    if wrap:
        lbl.setWordWrap(True)
    lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return lbl


def _safe_num(d: dict, key: str, default, cast):
    """存档数值安全读取：脏数据/错类型回退默认，绝不抛异常。

    旧版把整段恢复包在一个 try 里——一个坏键会炸掉后面所有设置的恢复。
    """
    try:
        return cast(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _page(title: str, subtitle: str, body: QWidget) -> QWidget:
    """Standard page chrome: big title + one-line subtitle + content."""
    w = QWidget()
    w.setObjectName("page")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(30, 24, 26, 12)
    lay.setSpacing(4)
    t = QLabel(title)
    t.setObjectName("pageTitle")
    s = QLabel(subtitle)
    s.setObjectName("pageSub")
    lay.addWidget(t)
    lay.addWidget(s)
    lay.addSpacing(6)
    lay.addWidget(body, 1)
    return w


def _scrollable(widget: QWidget, h_scroll: bool = False) -> QScrollArea:
    """Wrap a settings page so overflowing explanatory notes stay reachable.

    h_scroll=True 时横向改为按需滚动（给内含 QSplitter 的页面留一条窄窗口
    的活路：分栏最小宽度和超出视口时宁可出横向滚动条也不裁剪控件）。
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded if h_scroll
        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


class MultiFolderDialog(QDialog):
    """多选文件夹对话框（QTreeView + 文件系统模型 + 扩展选择）。

    Qt 原生对话框无法干净地多选目录：ExistingFiles 模式按 Open 会进入目录，
    Directory 模式只回当前目录。此处用 Qt 构件自绘，无额外依赖。
    树根为“我的电脑”（列出所有盘符），打开时自动展开并定位到上次使用的目录；
    选中后可按 Ctrl/Shift 多选，也支持双击展开。
    """

    _last_dir: str = ""        # 进程内记忆上次所在目录

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import QDir
        self.setWindowTitle("选择照片文件夹（Ctrl/Shift 多选）")
        self.resize(720, 540)
        layout = QVBoxLayout(self)

        self.tree = QTreeView()
        self.model = QFileSystemModel()
        self.model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot
                             | QDir.Filter.Drives)
        self.model.setRootPath("")
        self.tree.setModel(self.model)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        for col in range(1, 4):
            self.tree.hideColumn(col)
        layout.addWidget(self.tree, 1)

        hint = QLabel("按住 Ctrl 或 Shift 点击可一次选择多个文件夹；选完点 确定。")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # QFileSystemModel 异步加载：目录加载完成时逐级展开到目标目录
        start = self._last_dir if os.path.isdir(self._last_dir) else os.path.expanduser("~")
        self._target = os.path.normpath(os.path.abspath(start))
        self.model.directoryLoaded.connect(self._on_dir_loaded)
        self._reveal_target()

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.normcase(os.path.normpath(path))

    def _reveal_target(self) -> None:
        """从盘符起逐级展开已加载的祖先目录，最后滚动到目标目录。"""
        target = self._target
        parts = []
        cur = target
        while True:
            parts.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        for path in reversed(parts):               # 盘符 → ... → 目标
            idx = self.model.index(path)
            if not idx.isValid():
                return                              # 尚未加载，等下一个 directoryLoaded
            if path != target:
                self.tree.expand(idx)
        idx = self.model.index(target)
        if idx.isValid():
            self.tree.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
            # 仅移动焦点，不预选，避免确定时误把该目录也加入\n            self.tree.selectionModel().setCurrentIndex(\n                idx, QItemSelectionModel.SelectionFlag.NoUpdate)\n
    def _on_dir_loaded(self, path: str) -> None:
        if not self._target:
            return
        p = self._norm(path)
        t = self._norm(self._target)
        if p == "." or t == p or t.startswith(p.rstrip("\\/") + os.sep) or not p:
            self._reveal_target()

    def selected_folders(self) -> List[str]:
        folders = []
        seen = set()
        for index in self.tree.selectionModel().selectedIndexes():
            if index.column() == 0:
                path = self.model.filePath(index)
                if path and path not in seen:
                    seen.add(path)
                    folders.append(path)
        return folders

    def accept(self) -> None:
        folders = self.selected_folders()
        if folders:
            MultiFolderDialog._last_dir = os.path.dirname(folders[0]) or folders[0]
        self._target = ""                           # 停止自动展开
        super().accept()

def _make_icons() -> Dict[str, str]:
    """Draw the small arrow / tick glyphs used by the stylesheet (no asset files)."""
    d = os.path.join(tempfile.gettempdir(), "photomosaic_icons")
    os.makedirs(d, exist_ok=True)

    def draw(name: str, color: str, pts: List[Tuple[int, int]]) -> str:
        pm = QPixmap(12, 12)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color))
        pen.setWidthF(1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPolyline(QPolygon([QPoint(x, y) for x, y in pts]))
        p.end()
        path = os.path.join(d, name + ".png")
        pm.save(path)
        return path.replace("\\", "/")

    return {
        "UP": draw("up", "#6b7280", [(2, 8), (6, 4), (10, 8)]),
        "DOWN": draw("down", "#6b7280", [(2, 4), (6, 8), (10, 4)]),
        "CHECK": draw("check", "#ffffff", [(2, 6), (5, 9), (10, 3)]),
    }


_QSS = """
* { font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif; font-size: 13px; color: #1f2937; }
QMainWindow, QDialog, #page, QScrollArea, QScrollArea > QWidget > QWidget { background: #f4f5f7; }
QLabel { background: transparent; }
QToolTip { background: #111827; color: #f9fafb; border: none; padding: 6px 9px; }

QMenuBar { background: #ffffff; border-bottom: 1px solid #e5e7eb; padding: 2px 6px; }
QMenuBar::item { padding: 5px 11px; border-radius: 6px; background: transparent; }
QMenuBar::item:selected { background: #eef2ff; }
QMenu { background: #ffffff; border: 1px solid #e5e7eb; padding: 6px; }
QMenu::item { padding: 7px 24px; border-radius: 6px; }
QMenu::item:selected { background: #eef2ff; color: @DARK@; }

#sidebar { background: #111827; }
#sidebar QLabel { color: #f9fafb; font-size: 18px; font-weight: 700; padding: 24px 20px 2px 20px; }
#sidebar QLabel#sideSub { color: #6b7280; font-size: 11px; font-weight: 400; padding: 0 20px 20px 20px; }
#sidebar QPushButton { background: transparent; color: #9ca3af; border: none; border-radius: 0;
    border-left: 3px solid transparent; padding: 14px 20px; text-align: left; font-size: 14px; }
#sidebar QPushButton:hover { background: #1f2937; color: #e5e7eb; }
#sidebar QPushButton[active="true"] { background: #1f2937; color: #ffffff;
    border-left: 3px solid @ACCENT@; font-weight: 600; }

#pageTitle { font-size: 22px; font-weight: 700; color: #111827; }
#pageSub { color: #6b7280; font-size: 13px; }

QGroupBox { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
    margin-top: 0; padding: 40px 16px 16px 16px; }
QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 16px; top: 12px;
    padding: 0; color: #111827; font-size: 14px; font-weight: 700; }

QPushButton { background: #ffffff; border: 1px solid #d1d5db; border-radius: 8px;
    padding: 7px 16px; color: #374151; }
QPushButton:hover { background: #f9fafb; border-color: #9ca3af; }
QPushButton:pressed { background: #f3f4f6; }
QPushButton:disabled { color: #9ca3af; background: #f3f4f6; border-color: #e5e7eb; }
QPushButton#primary { background: @ACCENT@; border: none; color: #ffffff; font-weight: 600; padding: 9px 22px; }
QPushButton#primary:hover { background: @DARK@; }
QPushButton#primary:disabled { background: #c7d2fe; color: #ffffff; }
QPushButton#ghost { background: #ffffff; border: 1px solid @ACCENT@; color: @ACCENT@; font-weight: 600; }
QPushButton#ghost:hover { background: #eef2ff; }
QPushButton:checked { background: #eef2ff; border-color: @ACCENT@; color: @DARK@; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #ffffff; border: 1px solid #d1d5db;
    border-radius: 8px; padding: 6px 10px; min-height: 20px; selection-background-color: @ACCENT@; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid @ACCENT@; }
QSpinBox, QDoubleSpinBox { padding-right: 30px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
    width: 24px; border: none; border-left: 1px solid #e5e7eb; border-top-right-radius: 7px; background: #f9fafb; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
    width: 24px; border: none; border-left: 1px solid #e5e7eb; border-top: 1px solid #e5e7eb;
    border-bottom-right-radius: 7px; background: #f9fafb; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #eef2ff; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@UP@); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@DOWN@); width: 10px; height: 10px; }
QComboBox { padding-right: 30px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 28px; border: none; }
QComboBox::down-arrow { image: url(@DOWN@); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: #ffffff; border: 1px solid #e5e7eb; padding: 4px; outline: 0;
    selection-background-color: #eef2ff; selection-color: #111827; }

QListWidget { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 4px; outline: 0; }
QListWidget::item { padding: 8px 10px; border-radius: 6px; }
QListWidget::item:hover { background: #f9fafb; }
QListWidget::item:selected { background: #eef2ff; color: #111827; }
QTextEdit { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 6px 8px; }
QTreeView { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; outline: 0;
    alternate-background-color: #f9fafb; }
QTreeView::item { padding: 3px 0; }
QTreeView::item:selected { background: #eef2ff; color: #111827; }

QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 14px; height: 14px; border: 2px solid #cbd5e1; border-radius: 5px; background: #ffffff; }
QCheckBox::indicator:hover { border-color: @ACCENT@; }
QCheckBox::indicator:checked { background: @ACCENT@; border-color: @ACCENT@; image: url(@CHECK@); }
QRadioButton::indicator { width: 14px; height: 14px; border: 2px solid #cbd5e1; border-radius: 9px; background: #ffffff; }
QRadioButton::indicator:hover { border-color: @ACCENT@; }
QRadioButton::indicator:checked { width: 8px; height: 8px; border: 5px solid @ACCENT@; border-radius: 9px; background: #ffffff; }

QSlider::groove:horizontal { height: 5px; background: #e5e7eb; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @ACCENT@; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
    background: #ffffff; border: 2px solid @ACCENT@; }

QProgressBar { border: none; border-radius: 7px; background: #e5e7eb; text-align: center; color: #374151;
    min-height: 14px; max-height: 14px; font-size: 11px; }
QProgressBar::chunk { background: @ACCENT@; border-radius: 7px; }

QSplitter::handle { background: #e5e7eb; }
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:hover { background: #c7d2fe; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #d1d5db; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #9ca3af; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #d1d5db; border-radius: 4px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #9ca3af; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QStatusBar { background: #ffffff; border-top: 1px solid #e5e7eb; }
QStatusBar::item { border: none; }
QLabel#hint { color: #6b7280; font-size: 12px; }
QLabel#stat { color: #374151; font-size: 12px; }
QLabel#view { background: #111827; color: #9ca3af; border-radius: 10px; }
"""


def _build_style() -> str:
    qss = _QSS.replace("@ACCENT@", ACCENT).replace("@DARK@", ACCENT_DARK)
    for key, val in _make_icons().items():
        qss = qss.replace("@%s@" % key, val)
    return qss



class _Worker(QThread):
    progress = Signal(int, int)
    message = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[Callable[[int, int], None], Callable[[str], None]], object]):
        super().__init__()
        self._fn = fn

    def run(self) -> None:  # pragma: no cover - thread entry
        try:
            result = self._fn(self.progress.emit, self.message.emit)
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ImageView(QLabel):
    """Base preview label with widget<->image coordinate mapping."""

    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self.setObjectName("view")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 视觉下限 240×240：允许窗口缩到 900×700 而不裁剪控件（布局用
        # stretch 撑大，正常窗口下远大于此）
        self.setMinimumSize(240, 240)
        self._pix: Optional[QPixmap] = None
        self._disp = QRect()

    def set_pixmap(self, pix: QPixmap) -> None:
        self._pix = pix
        self._refresh()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._refresh()

    def _refresh(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        scaled = self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        x = max(0, (self.width() - scaled.width()) // 2)
        y = max(0, (self.height() - scaled.height()) // 2)
        self._disp = QRect(x, y, scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        if self._pix is None or not self._disp.isValid():
            return None
        r = self._disp
        px = min(max(pos.x(), r.x()), r.x() + r.width() - 1)
        py = min(max(pos.y(), r.y()), r.y() + r.height() - 1)
        ix = (px - r.x()) * self._pix.width() / max(1, r.width())
        iy = (py - r.y()) * self._pix.height() / max(1, r.height())
        return int(ix), int(iy)

    def img_rect_to_widget(self, box: Tuple[int, int, int, int]) -> QRect:
        r = self._disp
        sx = r.width() / max(1, self._pix.width())
        sy = r.height() / max(1, self._pix.height())
        return QRect(int(r.x() + box[0] * sx), int(r.y() + box[1] * sy),
                     int((box[2] - box[0]) * sx), int((box[3] - box[1]) * sy))


LEVEL_COLORS = {1: QColor(120, 120, 120, 140), 2: QColor(37, 99, 235, 200),
                3: QColor(217, 119, 6, 220), 4: QColor(220, 38, 38, 220),
                5: QColor(147, 51, 234, 220)}


class GridView(_ImageView):
    """Master preview + effective-grid overlay; left click splits, right merges.

    Ctrl+拖拽为自由套索（任意形状选区），按住 Shift 退回矩形框选。"""

    changed = Signal()
    before_edit = Signal()   # 编辑前钩子：MainWindow 用来物化自动细分为真实网格

    def __init__(self) -> None:
        super().__init__("请先在步骤①打开主图")
        self.tree: Optional[QuadTree] = None
        self._leaves: List[Tuple[Tuple[int, int, int, int], int]] = []
        self._press_pos: Optional[QPoint] = None
        self._press_btn = Qt.MouseButton.LeftButton
        self._press_ctrl = False
        self._press_shift = False
        self._lasso: List[QPoint] = []          # 套索轨迹（控件坐标）
        self._style_name = "grid"               # 当前风格：决定叶子画成什么形状

    def set_style(self, name: str) -> None:
        """网格预览按风格画出真实形状（蜂窝六边形/点彩圆形/砖墙错位/网格矩形）。"""
        self._style_name = name
        self.update()

    def set_tree(self, tree: Optional[QuadTree]) -> None:
        self.tree = tree

    def set_leaves(self, leaves) -> None:
        norm = []
        for item in leaves:
            if len(item) == 5:                      # (x0,y0,x1,y1,level)
                norm.append(((item[0], item[1], item[2], item[3]), int(item[4])))
            else:                                   # (box, level)
                norm.append((tuple(item[0]), int(item[1])))
        self._leaves = norm
        self.update()

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        if self._pix is None:
            return
        p = QPainter(self)
        if self._leaves:
            pen = QPen()
            pen.setWidth(1)
            for box, level in self._leaves:
                pen.setColor(LEVEL_COLORS.get(level, QColor(0, 0, 0)))
                p.setPen(pen)
                r = self.img_rect_to_widget(box)
                # 按风格画真实形状：蜂窝=六边形、点彩=圆形、其余=矩形
                if self._style_name == "hex":
                    x0, y0, w, h = r.x(), r.y(), r.width(), r.height()
                    p.drawPolygon(QPolygon([
                        QPoint(int(x0 + w * 0.25), y0), QPoint(int(x0 + w * 0.75), y0),
                        QPoint(x0 + w, int(y0 + h * 0.5)), QPoint(int(x0 + w * 0.75), y0 + h),
                        QPoint(int(x0 + w * 0.25), y0 + h), QPoint(x0, int(y0 + h * 0.5))]))
                elif self._style_name == "dot":
                    p.drawEllipse(r)
                else:
                    p.drawRect(r)
        # 套索/矩形框选的实时预览
        if self._press_ctrl and len(self._lasso) >= 2:
            sel_pen = QPen(QColor(ACCENT))
            sel_pen.setWidth(2)
            p.setPen(sel_pen)
            p.setBrush(QColor(79, 70, 229, 36))
            if self._press_shift:                     # Shift：矩形框选
                p.drawRect(QRect(self._lasso[0], self._lasso[-1]).normalized())
            else:                                     # 默认：自由套索
                p.drawPolygon(QPolygon(self._lasso))
        p.end()

    # 鼠标操作（粒度上限始终取自右侧“嵌套层级”所选级数）：
    #   左键单击              分割一格（细一级）
    #   右键单击              合并一格（粗一级）
    #   Ctrl+左键拖拽         自由套索圈选任意形状，区域内一次细分到最细粒度
    #   Ctrl+右键拖拽         自由套索圈选，取消区域内的细分
    #   Ctrl+Shift+左/右拖拽  矩形框选（旧行为）
    #   Ctrl+左键单击         该点所在格一次细分到最细粒度
    #   Ctrl+右键单击         取消该点所在基础块的全部细分
    def mousePressEvent(self, ev) -> None:
        if self.tree is None or ev.button() not in (Qt.MouseButton.LeftButton,
                                                    Qt.MouseButton.RightButton):
            return super().mousePressEvent(ev)
        self._press_pos = ev.position().toPoint()
        self._press_btn = ev.button()
        self._press_ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._press_shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self._lasso = [self._press_pos] if self._press_ctrl else []
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._press_pos is not None and self._press_ctrl:
            pt = ev.position().toPoint()
            if not self._lasso or (abs(pt.x() - self._lasso[-1].x()) +
                                   abs(pt.y() - self._lasso[-1].y())) >= 2:
                self._lasso.append(pt)
                self.update()
        super().mouseMoveEvent(ev)

    def _lasso_span(self) -> int:
        """轨迹包围盒的边长（判断拖拽/单击；闭合套索首末点会重合）。"""
        if len(self._lasso) < 2:
            return 0
        xs = [q.x() for q in self._lasso]
        ys = [q.y() for q in self._lasso]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def mouseReleaseEvent(self, ev) -> None:
        if self.tree is None or self._press_pos is None or ev.button() != self._press_btn:
            return super().mouseReleaseEvent(ev)
        start = self._press_pos
        end = ev.position().toPoint()
        ctrl = self._press_ctrl
        shift = self._press_shift
        left = self._press_btn == Qt.MouseButton.LeftButton
        lasso = list(self._lasso)
        if ctrl and (not lasso or lasso[-1] != end):
            lasso.append(end)
        span = self._lasso_span() if ctrl else max(abs(end.x() - start.x()),
                                                   abs(end.y() - start.y()))
        self._press_pos = None
        self._lasso = []
        dragged = span > 4
        changed = False
        self.before_edit.emit()          # 让 MainWindow 先把自动细分物化成真实网格
        if ctrl and dragged:
            if shift or len(lasso) < 3:              # Shift：矩形框选（旧行为）
                p0 = self.to_image(start)
                p1 = self.to_image(end)
                if p0 and p1:
                    box = (min(p0[0], p1[0]), min(p0[1], p1[1]),
                           max(p0[0], p1[0]) + 1, max(p0[1], p1[1]) + 1)
                    if left:
                        changed = self.tree.refine_region(box)      # 到最细粒度
                    else:
                        changed = self.tree.clear_region(box)
            else:                                     # 自由套索：任意形状
                pts = []
                for q in lasso:
                    ip = self.to_image(q)
                    if ip:
                        pts.append(ip)
                if len(pts) >= 3:
                    if left:
                        changed = self.tree.refine_polygon(pts)     # 到最细粒度
                    else:
                        changed = self.tree.clear_polygon(pts)
        else:
            pt = self.to_image(start)
            if pt:
                if left:
                    changed = (self.tree.split_fully_at(*pt) if ctrl
                               else self.tree.split_at(*pt))
                else:
                    changed = (self.tree.collapse_at(*pt) if ctrl
                               else self.tree.merge_up_at(*pt))
        if changed:
            self.changed.emit()
        self.update()
        super().mouseReleaseEvent(ev)


class ResultView(_ImageView):
    cell_clicked = Signal(object)  # (x0,y0,x1,y1) in image coords

    def __init__(self) -> None:
        super().__init__("生成后在此查看 / 点击碎片可替换")
        self.leaves: List[Tuple[int, int, int, int, int]] = []

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self.leaves:
            pt = self.to_image(ev.position().toPoint())
            if pt:
                x, y = pt
                for leaf in self.leaves:
                    if leaf[0] <= x < leaf[2] and leaf[1] <= y < leaf[3]:
                        self.cell_clicked.emit(leaf)
                        break
        super().mousePressEvent(ev)


class ReplaceDialog(QDialog):
    def __init__(self, db: TileDB, candidates: List[Tuple[int, float]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("替换拼贴碎片")
        self.resize(640, 420)
        lay = QHBoxLayout(self)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(96, 96))
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbs: Dict[int, bytes] = {}
        for pos, dist in candidates:
            meta_row = db.conn.execute(
                "SELECT id, path, variant FROM tiles ORDER BY id LIMIT 1 OFFSET ?", (pos,)
            ).fetchone()
            raw = db.get_thumb(meta_row[0])
            self.thumbs[pos] = raw
            item = QListWidgetItem(
                f"#{pos} {os.path.basename(meta_row[1])[:14]}\n{meta_row[2]} d={dist:.0f}")
            item.setIcon(QIcon(QPixmap.fromImage(QImage.fromData(raw))))
            item.setData(Qt.ItemDataRole.UserRole, pos)
            self.list.addItem(item)
        self.list.currentItemChanged.connect(self._preview)
        lay.addWidget(self.list, 2)
        right = QVBoxLayout()
        self.preview = QLabel("选择候选…")
        self.preview.setObjectName("view")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 240)
        right.addWidget(self.preview, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        lay.addLayout(right, 1)
        self.chosen: Optional[int] = None

    def _preview(self, item, _prev) -> None:
        if item is None:
            return
        pos = item.data(Qt.ItemDataRole.UserRole)
        pix = QPixmap()
        pix.loadFromData(self.thumbs[pos])
        self.preview.setPixmap(pix.scaled(self.preview.size(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
        self.chosen = pos

    def result_pos(self) -> Optional[int]:
        return self.chosen


def _ingest_fn(folders: List[str], db_path: str, variants: List[str], thumb: int,
               dedup: bool, recursive: bool, quality: int, workers: int,
               control: Optional[IngestControl] = None,
               skip_near_white: Optional[int] = None,
               skip_near_black: Optional[int] = None):
    def fn(progress, message):
        with TileDB(db_path) as db:
            total = {"files": 0, "added": 0, "duplicates": 0, "unreadable": 0,
                     "rows": 0, "index_hits": 0, "cancelled": 0,
                     "skipped_white": 0, "skipped_black": 0}
            n = len(folders)
            for fi, folder in enumerate(folders, start=1):
                tag = f"[{fi}/{n}] " if n > 1 else ""
                name = os.path.basename(folder.rstrip("/\\")) or folder

                def on_phase(phase, done, tot, tag=tag, name=name):
                    if phase == "scan":
                        progress(0, 0)           # 总数未知 → 滚动条走“忙碌”动画
                        message(f"{tag}正在扫描「{name}」… 已检查 {done} 个文件")
                    elif tot <= 0:
                        progress(1, 1)
                        message(f"{tag}「{name}」没有需要新增的照片")
                    else:
                        progress(done, tot)
                        message(f"{tag}正在处理「{name}」… {done} / {tot} 张")

                stats = ingest(
                    db, folder, ratio=1.0, recursive=recursive, variants=variants,
                    thumb_size=thumb, dedup=dedup, jpeg_quality=quality,
                    workers=workers, control=control,
                    skip_near_white=skip_near_white, skip_near_black=skip_near_black,
                    on_phase=on_phase)
                for k in total:
                    total[k] += stats.get(k, 0)
            total["db_tiles"] = db.count()
        return total
    return fn


def _render_fn(master_file: str, db_path: str, out: str, params: Dict[str, object],
               cancel: Optional[threading.Event] = None,
               pause: Optional[threading.Event] = None):
    def fn(progress, message):
        def p(done: int, total: int) -> None:
            # 暂停时阻塞 worker 线程（取消可中断暂停），实现手动暂停
            while pause is not None and pause.is_set():
                if cancel is not None and cancel.is_set():
                    break
                time.sleep(0.1)
            progress(done, total)
        try:
            with TileDB(db_path) as db:
                return make_mosaic(master_file, db, out, progress=p, status=message,
                                   should_cancel=(cancel.is_set if cancel else None),
                                   **params)
        except RenderCancelled:
            return {"cancelled": True, "output": out}
    return fn


class MainWindow(QMainWindow):
    _DEFAULT_SIZE = (1280, 820)      # 无有效存档时的默认窗口尺寸

    def __init__(self, load_settings: bool = True) -> None:
        super().__init__()
        self._persist = load_settings
        self.setWindowTitle("照片马赛克工作室")
        # 窗口图标：assets/icon.png（发布资产，存在即用）
        icon = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "assets", "icon.png")
        if os.path.exists(icon):
            self.setWindowIcon(QIcon(icon))
        self._welcome_shown = False
        # 主图压缩率缓存（_master_sample_ratio）：(路径,格式,质量) → 字节/像素
        self._ratio_cache_key: Optional[tuple] = None
        self._ratio_cache: float = 0.45
        # 瓦片库压缩率缓存（_tile_sample_ratio）：(库路径,库大小,格式,质量) → 字节/像素
        self._tile_ratio_key: Optional[tuple] = None
        self._tile_ratio: Optional[float] = None
        self.resize(*self._DEFAULT_SIZE)
        self.setStyleSheet(_build_style())
        self.master_path: Optional[str] = None
        self.master_rot = 0
        self.tree: Optional[QuadTree] = None
        self._master_pil: Optional[Image.Image] = None
        self.detail_map: Optional[DetailMap] = None
        self._user_touched_cells = False
        self._setting_cells = False
        self._worker: Optional[_Worker] = None
        self._result_pix: Optional[QPixmap] = None
        self._last_output: Optional[str] = None
        self._render_size: Optional[Tuple[int, int]] = None  # 真实输出尺寸（流式时预览≠输出）
        self._render_streamed: bool = False                   # 最近一次渲染是否流式 BigTIFF
        self._preview_path: Optional[str] = None              # 流式预览图路径（放大查看用）
        self._render_leaves: List[Tuple[int, int, int, int, int]] = []
        self._render_choices: List[int] = []
        self._render_mode = "uniform"
        self._ingest_ctrl: Optional[IngestControl] = None
        self._render_cancel = threading.Event()
        self._render_pause = threading.Event()      # 置位 = 暂停生成（可续跑）
        self._style = "grid"                        # 当前马赛克风格（③页卡片选择）
        self._master_overrides: Dict[str, Dict] = {}  # 主图路径 → 单独配置覆盖
        # 叠加层重算去抖：阈值滑块拖动停稳后才重算（避免每次拖动卡 3 秒）
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(200)
        self._overlay_timer.timeout.connect(self._refresh_grid_overlay)
        # 第③页分栏：保存的 base64 状态待首次显示后应用（构造时布局未定，
        # 立即 restoreState 会被随后的布局冲掉）
        self._splitter3_saved: Optional[str] = None
        self._splitter3_initialized = False
        # 进度条当前计数单位（"" = 格子/文件数；"条带" = 流式条带阶段）。
        # 渲染从"叶子数"刻度切到"条带数"刻度时换前缀标注，避免 93397/93397
        # 突然变成 3/3 的尺度跳变被误读为倒退
        self._progress_unit = ""

        central = QWidget()
        hbox = QHBoxLayout(central)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)

        # ---- sidebar ----------------------------------------------------
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("照片马赛克工作室")
        sub = QLabel("PhotoMosaic Studio · 独立实现")
        sub.setObjectName("sideSub")
        sl.addWidget(title)
        sl.addWidget(sub)
        self.step_buttons: List[QPushButton] = []
        for i, name in enumerate(["① 主图像", "② 单元图数据库", "③ 网格与嵌套层级",
                                  "④ 生成与修正"]):
            b = QPushButton(name)
            b.setProperty("active", i == 0)
            b.clicked.connect(lambda _=False, idx=i: self._goto(idx))
            sl.addWidget(b)
            self.step_buttons.append(b)
        sl.addStretch(1)
        hbox.addWidget(side)

        # ---- pages -------------------------------------------------------
        self.pages = QStackedWidget()
        self.pages.addWidget(self._page_master())
        self.pages.addWidget(self._page_library())
        self.pages.addWidget(self._page_grid())
        self.pages.addWidget(self._page_result())
        hbox.addWidget(self.pages, 1)
        self.setCentralWidget(central)

        # ---- bottom bar ---------------------------------------------------
        bar = QWidget()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 4, 10, 4)
        bl.setSpacing(14)
        self.status_text = QLabel("就绪")
        self.status_text.setSizePolicy(QSizePolicy.Policy.Ignored,
                                       QSizePolicy.Policy.Preferred)
        bl.addWidget(self.status_text, 1)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addWidget(bar, 1)

        self._build_menu()
        if self._persist and os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                    self._apply_state(json.load(fh))
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件(&F)")
        act_master = file_menu.addAction("打开主图(&M)…")
        act_master.triggered.connect(self._open_master)
        act_save = file_menu.addAction("保存项目(&S)…")
        act_save.triggered.connect(self._save_project)
        act_open = file_menu.addAction("打开项目(&O)…")
        act_open.triggered.connect(self._load_project)
        act_quit = file_menu.addAction("退出(&Q)")
        act_quit.triggered.connect(self.close)

    def closeEvent(self, ev) -> None:
        # 先停后台任务：渲染/建库 worker 若在跑，取消并等它退出，
        # 否则窗口销毁时线程被强杀（QThread destroyed while running）。
        try:
            if self._ingest_ctrl is not None:
                self._ingest_ctrl.cancel()
            self._render_cancel.set()
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(3000)
        except Exception:
            pass
        if self._persist:
            try:
                self._save_state(SETTINGS_PATH)
            except Exception:
                pass
        super().closeEvent(ev)

    # ---- first-run welcome -------------------------------------------------
    def _maybe_show_welcome(self) -> None:
        """首次启动（无存档或未看过）弹一次性欢迎指引（非模态）。"""
        if self._welcome_shown:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("欢迎使用照片马赛克工作室")
        dlg.setModal(True)
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            "<h3>四步拼出你的马赛克</h3>"
            "<p>① 打开主图：选一张要做成马赛克的大图。<br>"
            "② 构建单元图库：把你的照片文件夹加进来，建一次库以后随便换主图。<br>"
            "③ 网格与细节：调整格子粗细，圈出想细化的区域（Ctrl+拖拽套索）。<br>"
            "④ 生成与修正：点生成；超大输出会自动用流式渲染不爆内存。</p>"
            "<p>设置会被自动记住；关闭再次打开会从上次的断点继续。</p>")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        from PySide6.QtWidgets import QCheckBox, QDialogButtonBox
        cb = QCheckBox("不再显示")
        cb.setChecked(True)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(cb)
        lay.addWidget(btns)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.open()   # 非模态：真实 GUI 不卡界面，离屏测试不被阻塞
        self._welcome_shown = True   # 只弹一次；勾选框默认勾上，保持语义

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if not self._splitter3_initialized:
            self._splitter3_initialized = True
            # 布局尚未激活：推迟到事件循环下一拍再应用分栏比例
            QTimer.singleShot(0, self._init_splitter3)
        # 欢迎只弹一次（判断是否已看过在方法内做）
        QTimer.singleShot(100, self._maybe_show_welcome)

    def _init_splitter3(self) -> None:
        """首次显示后应用第③页分栏：优先恢复存档状态，否则初始比例 3:1。"""
        if self._splitter3_saved:
            try:
                self.splitter3.restoreState(
                    QByteArray.fromBase64(self._splitter3_saved.encode("ascii")))
                return
            except Exception:
                pass                              # 恢复失败 → 回落默认比例
        total = self.splitter3.width()
        if total <= 0:
            return
        left = max(1, int(round(total * 0.75)))
        self.splitter3.setSizes([left, max(1, total - left)])

    # ---- window geometry -------------------------------------------------
    @staticmethod
    def _available_geometry() -> QRect:
        """主屏可用区域（任务栏等已扣除）；拿不到时给一个安全兜底。"""
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.availableGeometry()
            if g.width() > 0 and g.height() > 0:
                return g
        return QRect(0, 0, 1600, 900)

    def _restore_geometry(self, d: dict) -> None:
        """恢复窗口几何：夹取到主屏可用区域；完全移出屏幕 → 回默认尺寸。

        只在存档带 geometry 键时生效（旧版项目文件没有该键 → 窗口不动）。
        窗口不比屏幕大时整个收进屏内；比屏幕大时只保证左上角可见。
        maximized 用窗口状态位恢复（显示时即最大化），normal 几何仍按存档。
        """
        if "geometry" not in d and not d.get("maximized"):
            return
        geo = d.get("geometry")
        ok = False
        if isinstance(geo, (list, tuple)) and len(geo) == 4:
            try:
                x, y, w, h = (int(v) for v in geo)
            except (TypeError, ValueError):
                x = y = w = h = 0
            if w >= 240 and h >= 200:            # 尺寸过小的存档视为损坏
                avail = self._available_geometry()
                off = (x >= avail.x() + avail.width()
                       or y >= avail.y() + avail.height()
                       or x + w <= avail.x() or y + h <= avail.y())
                if not off:
                    if w <= avail.width():
                        x = min(max(x, avail.x()), avail.x() + avail.width() - w)
                    else:
                        x = max(x, avail.x())
                    if h <= avail.height():
                        y = min(max(y, avail.y()), avail.y() + avail.height() - h)
                    else:
                        y = max(y, avail.y())
                    self.setGeometry(x, y, w, h)
                    ok = True
        if not ok:
            dw, dh = self._DEFAULT_SIZE
            self.resize(dw, dh)
            avail = self._available_geometry()
            self.move(avail.x() + max(0, (avail.width() - dw) // 2),
                      avail.y() + max(0, (avail.height() - dh) // 3))
        if d.get("maximized"):
            self.setWindowState(self.windowState()
                                | Qt.WindowState.WindowMaximized)

    def _restore_splitter3(self, d: dict) -> None:
        """恢复第③页分栏状态（base64）；失败忽略（保持默认比例）。"""
        blob = d.get("splitter3")
        if isinstance(blob, str) and blob:
            self._splitter3_saved = blob
            try:
                self.splitter3.restoreState(
                    QByteArray.fromBase64(blob.encode("ascii")))
            except Exception:
                pass

    # ---- session memory / project files ---------------------------------
    def _radio_level(self) -> int:
        for i, rb in enumerate(self.level_radios, start=1):
            if rb.isChecked():
                return i
        return 1

    def _gather_state(self) -> dict:
        geo = self.normalGeometry()       # 最大化时存"还原后"的正常几何
        if not geo.isValid():
            geo = self.geometry()
        return {
            "welcome_shown": self._welcome_shown,
            "master": self.master_path,
            "master_rot": self.master_rot,
            "masters": self._master_queue_paths(),
            "master_overrides": {k: v for k, v in self._master_overrides.items()},
            "db": self.db_edit.text(),
            "folders": self._folders(),
            "variants": self._variants(),
            "thumb": self.thumb_spin.value(),
            "dedup": self.dedup_check.isChecked(),
            "subfolders": self.subfolder_check.isChecked(),
            "cell_quality": self.cell_quality_spin.value(),
            "workers": self.workers_spin.value(),
            "cells_x": self.cells_x_spin.value(),
            "level": self._radio_level(),
            "auto": self.auto_check.isChecked(),
            "signal": self.signal_combo.currentIndex(),
            "style": self._style,
            "threshold": float(self.threshold_slider.value()),
            "top_k": self.topk_spin.value(),
            "max_reuse": self.reuse_spin.value(),
            "radius": self.radius_spin.value(),
            "blend": self.blend_slider.value(),
            "seed": self.seed_spin.value(),
            "cell_px": self._cell_px() or 0,
            "out_dir": self.outdir_edit.text().strip(),
            "device": self.device_combo.currentText(),
            "first_level": self.first_level_check.isChecked(),
            "skip_white": self.skip_white_check.isChecked(),
            "skip_black": self.skip_black_check.isChecked(),
            # 结果页的查看/导出选项：用户天天调，必须记忆
            "zoom": self.zoom_combo.currentText(),
            "format": self.format_combo.currentText(),
            "out_quality": self.out_quality.value(),
            "page": self.pages.currentIndex(),
            "tree_leaves": [list(b) + [l] for b, l in (
                self.tree.leaf_boxes() if self.tree else [])],
            # 窗口几何（[x, y, w, h]）与最大化状态；第③页分栏 base64 状态
            "geometry": [geo.x(), geo.y(), geo.width(), geo.height()],
            "maximized": bool(self.windowState() & Qt.WindowState.WindowMaximized),
            "splitter3": self.splitter3.saveState().toBase64().data().decode("ascii"),
        }

    def _apply_state(self, d: dict) -> None:
        self._user_touched_cells = True
        self._welcome_shown = bool(d.get("welcome_shown", False))
        if d.get("folders"):
            self.folder_list.clear()
            for f in d["folders"]:
                self.folder_list.addItem(f)
        if d.get("db"):
            self.db_edit.setText(d["db"])
        for name in self.variant_checks:
            self.variant_checks[name].setChecked(name in (d.get("variants") or ["orig"]))
        self.thumb_spin.setValue(_safe_num(d, "thumb", 256, int))
        self.dedup_check.setChecked(bool(d.get("dedup", True)))
        self.subfolder_check.setChecked(bool(d.get("subfolders", True)))
        self.cell_quality_spin.setValue(_safe_num(d, "cell_quality", 100, int))
        self.workers_spin.setValue(_safe_num(d, "workers", 0, int))
        self.cells_x_spin.setValue(_safe_num(d, "cells_x", 100, int))
        lvl = _safe_num(d, "level", 3, int)
        if 1 <= lvl <= len(self.level_radios):
            self.level_radios[lvl - 1].setChecked(True)
        self.auto_check.setChecked(bool(d.get("auto", True)))
        sig = _safe_num(d, "signal", 0, int)
        if 0 <= sig < self.signal_combo.count():
            self.signal_combo.setCurrentIndex(sig)
        self._select_style(str(d.get("style", "grid")))
        self.threshold_slider.setValue(_safe_num(d, "threshold", 25, lambda v: int(round(float(v)))))
        self.topk_spin.setValue(_safe_num(d, "top_k", 8, int))
        self.reuse_spin.setValue(_safe_num(d, "max_reuse", 0, int))
        self.radius_spin.setValue(_safe_num(d, "radius", 0.0, float))
        self.blend_slider.setValue(_safe_num(d, "blend", 55, int))
        self.seed_spin.setValue(_safe_num(d, "seed", -1, int))
        # 输出清晰度：新版存 cell_px；旧版只有 out_width → 按块数换算迁移
        cp = _safe_num(d, "cell_px", 0, int) or 0
        if not cp and d.get("out_width"):
            cp = max(1, round(_safe_num(d, "out_width", 0, int)
                              / max(1, _safe_num(d, "cells_x", 100, int))))
        self._set_cell_px(cp)
        self.outdir_edit.setText(str(d.get("out_dir", "") or ""))
        dev = d.get("device", "auto")
        idx = self.device_combo.findText(dev)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.first_level_check.setChecked(bool(d.get("first_level", False)))
        self.skip_white_check.setChecked(bool(d.get("skip_white", False)))
        self.skip_black_check.setChecked(bool(d.get("skip_black", False)))
        # 结果页查看/导出选项（旧版存档没有 → 保持默认）
        z = self.zoom_combo.findText(str(d.get("zoom", "")))
        if z >= 0:
            self.zoom_combo.setCurrentIndex(z)
        fi = self.format_combo.findText(str(d.get("format", "")))
        if fi >= 0:
            self.format_combo.setCurrentIndex(fi)
        self.out_quality.setValue(_safe_num(d, "out_quality", 100, int))
        # 主图队列（批量来源）恢复：只留仍存在的文件
        masters = d.get("masters") or []
        if masters:
            self.master_queue.clear()
            for p in masters:
                if isinstance(p, str) and os.path.exists(p):
                    self.master_queue.addItem(p)
        # 主图单独配置恢复
        mo = d.get("master_overrides")
        if isinstance(mo, dict):
            self._master_overrides = {str(k): v for k, v in mo.items()
                                      if isinstance(v, dict)}
            for i in range(self.master_queue.count()):
                it = self.master_queue.item(i)
                ov = self._master_overrides.get(it.text())
                if ov:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                    it.setToolTip("单独配置: " + ", ".join(
                        f"{k}={v}" for k, v in ov.items()))
        # 主图/网格恢复依赖磁盘文件：单独容错，失败不影响其余设置
        if d.get("master") and os.path.exists(d["master"]):
            try:
                self.master_path = d["master"]
                self.master_rot = int(d.get("master_rot", 0))
                self._show_master()
                self._restore_tree(d.get("tree_leaves"))
                self.btn_render.setEnabled(True)
            except Exception:
                pass                    # 主图被移动/删除：其余设置照常恢复
        # 窗口几何/最大化与第③页分栏：旧版存档没有这些键 → 保持默认
        self._restore_geometry(d)
        self._restore_splitter3(d)
        # 最后恢复当前页（覆盖启动默认页）
        pg = d.get("page")
        if isinstance(pg, int) and 0 <= pg < self.pages.count():
            self._goto(pg)

    def _restore_tree(self, leaves) -> None:
        pix = self._master_pixmap()
        if pix is None or not leaves:
            return
        base_y = max(1, round(pix.height() / (pix.width() / self.cells_x_spin.value())))
        pairs = [((int(l[0]), int(l[1]), int(l[2]), int(l[3])), int(l[4])) for l in leaves]
        self.tree = QuadTree.from_leaf_boxes(pix.width(), pix.height(),
                                             self.cells_x_spin.value(), base_y, pairs,
                                             max_level=max([5] + [p[1] for p in pairs]),
                                             min_px=1)
        self.tree.set_max_level(self._radio_level())
        self.grid_view.set_tree(self.tree)
        self._refresh_grid_overlay()

    def _save_state(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._gather_state(), fh, ensure_ascii=False, indent=1)

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存项目", "project.pmproj",
                                              "马赛克项目 (*.pmproj)")
        if path:
            self._save_state(path)
            self.status_text.setText(f"项目已保存 → {path}")

    def _load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "",
                                              "马赛克项目 (*.pmproj)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._apply_state(json.load(fh))
            self.status_text.setText(f"项目已加载 → {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "打开项目失败", str(exc))

    # ------------------------------------------------------------------
    def _goto(self, idx: int) -> None:
        self.pages.setCurrentIndex(idx)
        for i, b in enumerate(self.step_buttons):
            b.setProperty("active", i == idx)
            b.style().unpolish(b)
            b.style().polish(b)

    # ---- page 1: master -------------------------------------------------
    def _page_master(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("预览")
        gl = QVBoxLayout(g)
        self.master_view = _ImageView("点击“打开主图…”选择一张照片")
        gl.addWidget(self.master_view, 1)
        row = QHBoxLayout()
        b = QPushButton("打开主图…")
        b.setObjectName("primary")
        b.clicked.connect(self._open_master)
        row.addWidget(b)
        for text, deg in (("左转 90°", -90), ("右转 90°", 90), ("180° 旋转", 180)):
            rb = QPushButton(text)
            rb.clicked.connect(lambda _=False, d=deg: self._rotate(d))
            row.addWidget(rb)
        row.addStretch(1)
        self.master_info = QLabel("")
        self.master_info.setObjectName("hint")
        _shrinkable(self.master_info)
        row.addWidget(self.master_info)
        gl.addLayout(row)
        lay.addWidget(g, 1)

        # 主图队列：批量出图的来源（选中哪张就编辑/预览哪张）
        gq = QGroupBox("主图队列（批量出图时对队列里每一张各出一张成品）")
        gql = QVBoxLayout(gq)
        self.master_queue = QListWidget()
        self.master_queue.setMaximumHeight(120)
        self.master_queue.currentRowChanged.connect(self._on_master_queue_select)
        gql.addWidget(self.master_queue)
        qrow = QHBoxLayout()
        badd = QPushButton("添加主图…")
        badd.clicked.connect(self._add_masters)
        qrow.addWidget(badd)
        bfolder = QPushButton("添加文件夹…")
        bfolder.clicked.connect(self._add_master_folder)
        qrow.addWidget(bfolder)
        bremove = QPushButton("移除选中")
        bremove.clicked.connect(self._remove_selected_master)
        qrow.addWidget(bremove)
        bclear = QPushButton("清空")
        bclear.clicked.connect(self.master_queue.clear)
        qrow.addWidget(bclear)
        bcfg = QPushButton("单独配置…")
        bcfg.setToolTip("给选中的那张主图单独设一套参数（风格/格子/融合等），"
                        "批量生成时它用自己的配置，其余用统一配置。")
        bcfg.clicked.connect(self._configure_master)
        qrow.addWidget(bcfg)
        qrow.addStretch(1)
        gql.addLayout(qrow)
        lay.addWidget(gq)

        lay.setContentsMargins(0, 0, 0, 0)
        return _page("主图像", "选择想要还原的照片，必要时先旋转到正确方向；可多张作批量队列。", w)

    def _open_master(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开主图", "", "图像 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if not path:
            return
        if not self.master_queue.findItems(path, Qt.MatchFlag.MatchExactly):
            self.master_queue.addItem(path)
        self.master_queue.setCurrentRow(
            self.master_queue.count() - 1 if self.master_queue.count() else -1)
        self._set_current_master(path)

    # ---- 主图队列（批量来源） -------------------------------------------
    def _set_current_master(self, path: str) -> None:
        self.master_path = path
        self.master_rot = 0
        self._show_master()
        if not self.db_edit.text():
            self.db_edit.setText(os.path.join(os.path.dirname(path), "tiles.db"))
        self.status_text.setText(f"主图: {os.path.basename(path)}")
        # ③页的主图切换器标签同步
        row = self.master_queue.currentRow()
        n = self.master_queue.count()
        self.grid_master_label.setText(
            f"{os.path.basename(path)}（{row + 1}/{n}）" if n else os.path.basename(path))

    def _step_master(self, delta: int) -> None:
        """③页内切换主图：沿①的队列前进/后退一张。"""
        n = self.master_queue.count()
        if n == 0:
            self.status_text.setText("主图队列为空：先在①添加主图")
            return
        row = self.master_queue.currentRow()
        self.master_queue.setCurrentRow((row + delta) % n)

    def _on_master_queue_select(self, row: int) -> None:
        if 0 <= row < self.master_queue.count():
            self._set_current_master(self.master_queue.item(row).text())

    def _add_masters(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加主图（可多选）", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        for p in paths:
            if not self.master_queue.findItems(p, Qt.MatchFlag.MatchExactly):
                self.master_queue.addItem(p)

    def _add_master_folder(self) -> None:
        from .batch import scan_masters
        folder = QFileDialog.getExistingDirectory(self, "添加主图文件夹")
        if folder:
            for p in scan_masters(folder):
                if not self.master_queue.findItems(p, Qt.MatchFlag.MatchExactly):
                    self.master_queue.addItem(p)

    def _remove_selected_master(self) -> None:
        row = self.master_queue.currentRow()
        if row >= 0:
            self.master_queue.takeItem(row)

    def _master_queue_paths(self) -> List[str]:
        return [self.master_queue.item(i).text()
                for i in range(self.master_queue.count())]

    # ---- 主图单独配置（批量时盖过统一配置） ------------------------------
    def _configure_master(self) -> None:
        item = self.master_queue.currentItem()
        if item is None:
            QMessageBox.information(self, "单独配置", "先在主图队列里选中一张图。")
            return
        path = item.text()
        existing = self._master_overrides.get(path, {})
        dlg = QDialog(self)
        dlg.setWindowTitle(f"单独配置：{os.path.basename(path)}")
        fl = QFormLayout(dlg)
        from .styles import all_styles
        style_combo = QComboBox()
        style_combo.addItem("（跟随统一配置）")
        for _n, label in all_styles():
            style_combo.addItem(label)
        if existing.get("style"):
            from .styles import get_style
            lbl = get_style(existing["style"]).label
            i = style_combo.findText(lbl)
            if i >= 0:
                style_combo.setCurrentIndex(i)
        fl.addRow("风格", style_combo)

        cells = QSpinBox(); cells.setRange(0, 2000)
        cells.setSpecialValueText("（跟随统一）")
        cells.setValue(int(existing.get("cells_x", 0)))
        fl.addRow("基础块数", cells)

        levels = QSpinBox(); levels.setRange(0, 5)
        levels.setSpecialValueText("（跟随统一）")
        levels.setValue(int(existing.get("levels", 0)))
        fl.addRow("细分层级", levels)

        blend = QSpinBox(); blend.setRange(-1, 100)
        blend.setSpecialValueText("（跟随统一）")
        blend.setValue(int(existing.get("blend_pct", -1)))
        fl.addRow("融合强度 %", blend)

        cellpx = QSpinBox(); cellpx.setRange(0, 8192)
        cellpx.setSpecialValueText("（跟随统一）")
        cellpx.setValue(int(existing.get("cell_px", 0)))
        fl.addRow("每格像素", cellpx)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        fl.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # 收集覆盖项（只收集"显式设了"的）
        ov: Dict[str, object] = {}
        if style_combo.currentIndex() > 0:
            from .styles import all_styles as _as
            ov["style"] = _as()[style_combo.currentIndex() - 1][0]
        if cells.value() > 0:
            ov["cells_x"] = cells.value()
        if levels.value() > 0:
            ov["levels"] = levels.value()
        if blend.value() >= 0:
            ov["blend"] = blend.value() / 100.0
        if cellpx.value() > 0:
            ov["cell_px"] = cellpx.value()          # 批量时转 out_width
        if ov:
            self._master_overrides[path] = ov
            item.setText(path)                       # 保持路径为显示文本
            item.setToolTip("单独配置: " + ", ".join(
                f"{k}={v}" for k, v in ov.items()))
            f = item.font(); f.setBold(True); item.setFont(f)
        else:
            self._master_overrides.pop(path, None)
            item.setToolTip("")
            f = item.font(); f.setBold(False); item.setFont(f)
        self.status_text.setText(
            f"{os.path.basename(path)}：{'已设单独配置' if ov else '已恢复统一配置'}")

    def _rotate(self, deg: int) -> None:
        if not self.master_path:
            return
        self.master_rot = (self.master_rot + deg) % 360
        self._show_master()

    def _master_pixmap(self) -> Optional[QPixmap]:
        if not self.master_path:
            return None
        pix = QPixmap(self.master_path)
        if self.master_rot:
            pix = pix.transformed(QTransform().rotate(self.master_rot),
                                  Qt.TransformationMode.SmoothTransformation)
        return pix

    def _show_master(self) -> None:
        pix = self._master_pixmap()
        if pix is None:
            return
        QTimer.singleShot(0, self._update_out_info)
        self.master_view.set_pixmap(pix)
        self.master_info.setText(f"{pix.width()} × {pix.height()} px")
        if not self._user_touched_cells:
            suggested = int(min(400, max(8, round((300.0 * pix.width()
                                                   / pix.height()) ** 0.5))))
            self._setting_cells = True
            self.cells_x_spin.setValue(suggested)
            self._setting_cells = False
        path = self._effective_master()
        if path:
            with Image.open(path) as im:
                self._master_pil = im.convert("RGB")
            self._build_detail_map()
        self._rebuild_tree()
        self._refresh_style_cards()      # 主图就绪：刷新风格卡片的真实小样例

    def _build_detail_map(self) -> None:
        """按"细节识别方式"建细分信号图：梯度能量 or 显著性主体。"""
        if self._master_pil is None:
            return
        if getattr(self, "signal_combo", None) is not None and \
                self.signal_combo.currentIndex() == 1:
            from .saliency import SaliencyMap
            self.detail_map = SaliencyMap(self._master_pil)
        else:
            self.detail_map = DetailMap(self._master_pil)

    def _rebuild_tree(self) -> None:
        pix = self._master_pixmap()
        if pix is None:
            return
        self.tree = QuadTree(pix.width(), pix.height(),
                             self.cells_x_spin.value(), max(1, round(
                                 pix.height() / (pix.width() / self.cells_x_spin.value()))),
                             max_level=self._radio_level(), min_px=1)
        self.grid_view.set_pixmap(pix)
        self.grid_view.set_tree(self.tree)
        self._refresh_grid_overlay()

    def _effective_levels(self) -> int:
        lvl = 1
        for i, rb in enumerate(self.level_radios, start=1):
            if rb.isChecked():
                lvl = i
        if lvl == 1:
            return 1   # 「一级」= 经典均匀网格，自动细分不再介入
        return lvl

    def _refresh_grid_overlay(self) -> None:
        """立即重算叠加层。"""
        self._overlay_timer.stop()
        if self.tree is None:
            return
        if self.auto_check.isChecked() and self.detail_map is not None \
                and self._master_pil is not None:
            leaves = adaptive_leaves(self._master_pil, self.tree.leaf_boxes(),
                                     self._effective_levels(),
                                     float(self.threshold_slider.value()),
                                     detail=self.detail_map,
                                     first_level_factor=2.0
                                     if self.first_level_check.isChecked() else 1.0)
        else:
            leaves = self.tree.leaf_boxes()
        self.grid_view.set_leaves(leaves)
        hist: Dict[int, int] = {}
        for _b, lvl in self.grid_view._leaves:
            hist[lvl] = hist.get(lvl, 0) + 1
        fine = self.tree.min_leaf_px()
        self.grid_info.setText(
            f"有效块数 {len(leaves)} · 层级分布 {dict(sorted(hist.items()))} · "
            f"最细格 ≈{fine}px（由基础块数与级数决定，嫌粗就加大基础块数或级数）")

    def _refresh_grid_overlay_debounced(self) -> None:
        """去抖入口：只给阈值滑块这类连续改动用（停稳 200ms 才重算）。"""
        self._overlay_timer.start()

    def _materialize_auto_grid(self) -> None:
        """把自动细分预览物化成真实网格，让手动编辑能作用在显示的格子上。

        自动细分时 overlay 显示的是 adaptive_leaves 的预览，不在 tree 里；
        直接右键点这些"幽灵格"不会生效（tree 里对应位置还是 1 级基础块）。
        手动编辑时先把预览写回 tree，并关闭自动细分的二次覆盖。"""
        if self.tree is None or not self.auto_check.isChecked() \
                or self.detail_map is None or self._master_pil is None:
            return
        # 当前 overlay 显示的叶子（自动细分结果）
        leaves = self.grid_view._leaves
        if not leaves:
            return
        pix = self._master_pixmap()
        if pix is None:
            return
        base_x = self.cells_x_spin.value()
        base_y = max(1, round(pix.height() / (pix.width() / base_x)))
        pairs = [(tuple(map(int, b)), int(l)) for b, l in leaves]
        self.tree = QuadTree.from_leaf_boxes(
            pix.width(), pix.height(), base_x, base_y, pairs,
            max_level=max([5] + [p[1] for p in pairs]), min_px=1)
        self.grid_view.set_tree(self.tree)
        # 关掉自动细分：手动网格就是最终网格
        self.auto_check.blockSignals(True)
        self.auto_check.setChecked(False)
        self.auto_check.blockSignals(False)
        self.status_text.setText(
            "已进入手动编辑模式：自动细分已固化为真实网格，右键/套索可取消细分")

    # ---- page 2: library --------------------------------------------------
    def _page_library(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(0, 0, 16, 0)
        root.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(18)

        # ---- 照片来源 ----
        g = QGroupBox("照片来源")
        gl = QVBoxLayout(g)
        gl.setSpacing(10)
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.folder_list.setMinimumHeight(200)
        self.folder_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        gl.addWidget(self.folder_list, 1)
        row = QHBoxLayout()
        row.setSpacing(8)
        badd = QPushButton("＋ 添加文件夹…")
        badd.setObjectName("ghost")
        badd.clicked.connect(self._add_folder)
        brm = QPushButton("移除选中")
        brm.clicked.connect(self._remove_folders)
        bclr = QPushButton("清空")
        bclr.clicked.connect(self.folder_list.clear)
        row.addWidget(badd)
        row.addWidget(brm)
        row.addWidget(bclr)
        row.addStretch(1)
        self.folder_count = QLabel("")
        self.folder_count.setObjectName("hint")
        _shrinkable(self.folder_count)
        row.addWidget(self.folder_count)
        gl.addLayout(row)
        self.subfolder_check = QCheckBox("包含子文件夹")
        self.subfolder_check.setChecked(True)
        self.dedup_check = QCheckBox("跳过重复文件（MD5）")
        self.dedup_check.setChecked(True)
        crow = QHBoxLayout()
        crow.setSpacing(22)
        crow.addWidget(self.subfolder_check)
        crow.addWidget(self.dedup_check)
        crow.addStretch(1)
        gl.addLayout(crow)
        top.addWidget(g, 5)
        self.folder_list.model().rowsInserted.connect(self._update_folder_count)
        self.folder_list.model().rowsRemoved.connect(self._update_folder_count)
        self.folder_list.model().modelReset.connect(self._update_folder_count)
        self._update_folder_count()

        # ---- 瓦片设置 ----
        g2 = QGroupBox("瓦片设置")
        g2l = QVBoxLayout(g2)
        g2l.setSpacing(10)
        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setColumnStretch(1, 1)

        form.addWidget(QLabel("单元图变体"), 0, 0)
        # 五个变体复选框分两行摆（3+2）：单行摆放会顶住窄窗口的最小宽度
        self.variant_checks: Dict[str, QCheckBox] = {}
        vrows = [QHBoxLayout(), QHBoxLayout()]
        for vrow in vrows:
            vrow.setSpacing(14)
        for name, cn in (("orig", "原图"), ("mirror", "镜像"), ("r90", "左旋90°"),
                         ("r180", "180°"), ("r270", "右旋90°")):
            cb = QCheckBox(cn)
            cb.setProperty("variant", name)
            cb.setChecked(name in ("orig", "mirror"))
            self.variant_checks[name] = cb
            (vrows[0] if len(self.variant_checks) <= 3 else vrows[1]).addWidget(cb)
        for vrow in vrows:
            vrow.addStretch(1)
        vwrap = QVBoxLayout()
        vwrap.setSpacing(2)
        vwrap.addLayout(vrows[0])
        vwrap.addLayout(vrows[1])
        form.addLayout(vwrap, 0, 1)
        form.addWidget(_hint("每张照片额外生成镜像/旋转版备选瓦片：越多越容易贴出好效果，"
                             "但库成倍变大、建库更慢；一般 原图+镜像 就够。"), 1, 1)

        self.thumb_spin = QSpinBox()
        self.thumb_spin.setRange(32, 1024)
        self.thumb_spin.setValue(256)
        self.thumb_spin.setSuffix(" px")
        self.thumb_spin.setToolTip("瓦片库缓存缩略图的边长。256 对大多数网格够用；"
                                   "大幅打印、格子很大时建议 384–512。")
        form.addWidget(QLabel("缩略图尺寸"), 2, 0)
        form.addWidget(self.thumb_spin, 2, 1)

        self.cell_quality_spin = QSpinBox()
        self.cell_quality_spin.setRange(30, 100)
        self.cell_quality_spin.setValue(100)
        self.cell_quality_spin.setToolTip("缩略图的 JPEG 压缩质量。90 是画质/体积的甜点；"
                                          "库很大想省空间时用 75–80。")
        form.addWidget(QLabel("JPEG 质量"), 3, 0)
        form.addWidget(self.cell_quality_spin, 3, 1)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(0)
        self.workers_spin.setSpecialValueText("全部核心")
        self.workers_spin.setToolTip("建库同时用多少个进程。全部核心最快；机器还要干别的事就调小。")
        form.addWidget(QLabel("并行进程数"), 4, 0)
        form.addWidget(self.workers_spin, 4, 1)
        g2l.addLayout(form)

        self.skip_white_check = QCheckBox("排除近白瓦片（亮度过高）")
        self.skip_white_check.setToolTip("建库时跳过平均亮度 > 245 的照片（如白底文档截图）。")
        self.skip_black_check = QCheckBox("排除近黑瓦片（亮度过低）")
        self.skip_black_check.setToolTip("建库时跳过平均亮度 < 10 的照片。")
        g2l.addWidget(self.skip_white_check)
        g2l.addWidget(self.skip_black_check)
        g2l.addStretch(1)
        top.addWidget(g2, 5)
        root.addLayout(top, 1)

        # ---- 数据库位置 ----
        g3 = QGroupBox("数据库文件")
        g3l = QHBoxLayout(g3)
        self.db_edit = QLineEdit()
        self.db_edit.setPlaceholderText("tiles.db（构建时创建；已存在则增量更新）")
        bdb = QPushButton("浏览…")
        bdb.clicked.connect(self._pick_db)
        g3l.addWidget(self.db_edit, 1)
        g3l.addWidget(bdb)
        root.addWidget(g3)

        # ---- 构建 ----
        g4 = QGroupBox("构建")
        g4l = QVBoxLayout(g4)
        g4l.setSpacing(10)
        brow = QHBoxLayout()
        brow.setSpacing(10)
        self.btn_ingest = QPushButton("构建 / 更新数据库")
        self.btn_ingest.setObjectName("primary")
        self.btn_ingest.setMinimumHeight(38)
        self.btn_ingest.clicked.connect(self._ingest)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setMinimumHeight(38)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_cancel_ingest = QPushButton("取消")
        self.btn_cancel_ingest.setEnabled(False)
        self.btn_cancel_ingest.setMinimumHeight(38)
        self.btn_cancel_ingest.clicked.connect(self._cancel_ingest)
        brow.addWidget(self.btn_ingest, 1)
        brow.addWidget(self.btn_pause)
        brow.addWidget(self.btn_cancel_ingest)
        g4l.addLayout(brow)
        self.lib_progress = QProgressBar()
        self.lib_progress.setRange(0, 100)
        self.lib_progress.setValue(0)
        self.lib_progress.setFormat("%p%")
        g4l.addWidget(self.lib_progress)
        self.lib_info = QLabel("提示：照片越多效果越好，建议 500–1000 张以上；中断后再次构建会从断点继续。")
        self.lib_info.setObjectName("stat")
        self.lib_info.setWordWrap(True)
        g4l.addWidget(self.lib_info)
        root.addWidget(g4)
        root.addStretch(0)
        return _page("单元图数据库", "添加素材照片文件夹，构建用于拼贴的瓦片库（可重复使用、增量更新）。",
                     _scrollable(w))

    def _update_folder_count(self, *_a) -> None:
        try:
            n = self.folder_list.count()
        except RuntimeError:          # 窗口销毁期间的残余信号
            return
        self.folder_count.setText(f"已添加 {n} 个文件夹" if n else "尚未添加文件夹")

    def _remove_folders(self) -> None:
        for it in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(it))

    def _add_folder(self) -> None:
        dialog = MultiFolderDialog(self)
        if dialog.exec():
            for folder in dialog.selected_folders():
                if folder and not self.folder_list.findItems(
                        folder, Qt.MatchFlag.MatchExactly):
                    self.folder_list.addItem(folder)

    def _pick_db(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "单元图数据库", "tiles.db", "SQLite (*.db *.sqlite)")
        if path:
            self.db_edit.setText(path)

    def _folders(self) -> List[str]:
        return [self.folder_list.item(i).text() for i in range(self.folder_list.count())]

    def _variants(self) -> List[str]:
        return [n for n, cb in self.variant_checks.items() if cb.isChecked()] or ["orig"]

    def _ingest(self) -> None:
        folders = self._folders()
        if not folders:
            QMessageBox.warning(self, "单元图数据库", "请先添加至少一个照片文件夹。")
            return
        self._ingest_ctrl = IngestControl()
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("暂停")
        self.btn_cancel_ingest.setEnabled(True)
        self.status_text.setText("正在构建数据库…（中断后再次构建即从断点继续）")
        self._start(_ingest_fn(folders, self._db_path(), self._variants(),
                               self.thumb_spin.value(), self.dedup_check.isChecked(),
                               self.subfolder_check.isChecked(), self.cell_quality_spin.value(),
                               self.workers_spin.value(), self._ingest_ctrl,
                               245 if self.skip_white_check.isChecked() else None,
                               10 if self.skip_black_check.isChecked() else None),
                    self._ingest_done)

    def _toggle_pause(self) -> None:
        if self._ingest_ctrl is None:
            return
        if self._ingest_ctrl.paused.is_set():
            self._ingest_ctrl.resume()
            self.btn_pause.setText("暂停")
            self.status_text.setText("继续扫描…")
        else:
            self._ingest_ctrl.pause()
            self.btn_pause.setText("继续")
            self.status_text.setText("已暂停（完成中的批次后停下）")

    def _cancel_ingest(self) -> None:
        if self._ingest_ctrl is not None:
            self._ingest_ctrl.cancel()
            self.status_text.setText("正在取消…已入库部分已保存，下次构建自动续传")

    def _ingest_done(self, stats: object) -> None:
        self._set_busy(False)
        self._finish_progress(not (isinstance(stats, dict) and stats.get("cancelled")))
        self.btn_pause.setEnabled(False)
        self.btn_cancel_ingest.setEnabled(False)
        s = stats if isinstance(stats, dict) else {}
        if s.get("cancelled"):
            self.lib_info.setText(
                f"已取消：本次入库 {s.get('rows', 0)} 块；数据库累计 {s.get('db_tiles', '?')} 块。"
                "再次点击构建将从断点继续（未变化文件免重哈希）。")
            self.status_text.setText("扫描已取消，进度已保存")
            return
        self.lib_info.setText(
            f"数据库共 {s.get('db_tiles', '?')} 块单元图（本次新增 {s.get('rows', 0)}，"
            f"重复跳过 {s.get('duplicates', 0)}，索引命中免哈希 {s.get('index_hits', 0)}，"
            f"无法读取 {s.get('unreadable', 0)}）")
        self.status_text.setText("数据库构建完成")
        self._refresh_style_cards()      # 库就绪：刷新风格卡片的真实小样例

    def _db_path(self) -> str:
        return self.db_edit.text() or os.path.join(os.getcwd(), "tiles.db")

    # ---- page 3: grid & levels --------------------------------------------
    def _page_grid(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("网格编辑器")
        gl = QVBoxLayout(g)
        # 主图切换器：多张主图时在③页内直接换，不用回①页
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("当前主图"))
        self.grid_master_label = QLabel("（未加载）")
        self.grid_master_label.setObjectName("hint")
        _shrinkable(self.grid_master_label)
        mrow.addWidget(self.grid_master_label, 1)
        bprev = QPushButton("← 上一张")
        bprev.clicked.connect(lambda: self._step_master(-1))
        mrow.addWidget(bprev)
        bnext = QPushButton("下一张 →")
        bnext.clicked.connect(lambda: self._step_master(1))
        mrow.addWidget(bnext)
        gl.addLayout(mrow)
        self.grid_view = GridView()
        self.grid_view.changed.connect(self._update_grid_info)
        self.grid_view.before_edit.connect(self._materialize_auto_grid)
        gl.addWidget(self.grid_view, 1)
        hint = QLabel("网格操作（最细粒度 = 右侧所选级数）：左键 = 细分一级；右键 = 合并一级；"
                      "Ctrl+左键拖拽 = 自由套索圈选任意形状并细分到最细；Ctrl+右键拖拽 = 套索取消细分；"
                      "Ctrl+Shift+拖拽 = 矩形框选；"
                      "Ctrl+左键点击 = 该格细分到最细；Ctrl+右键点击 = 取消该基础块的全部细分。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        gl.addWidget(hint)
        # 重置/合并按钮分两行 2×2 摆：单行四个按钮的文本宽度会顶住窄窗口
        brow = QHBoxLayout()
        bmerge = QHBoxLayout()
        br = QPushButton("重置网格")
        br.clicked.connect(self._reset_grid)
        brow.addWidget(br)
        bm3 = QPushButton("合并全部 3 级元素")
        bm3.clicked.connect(lambda _=False: self._merge_level(3))
        brow.addWidget(bm3)
        bm4 = QPushButton("合并全部 4 级元素")
        bm4.clicked.connect(lambda _=False: self._merge_level(4))
        bmerge.addWidget(bm4)
        bm5 = QPushButton("合并全部 5 级元素")
        bm5.clicked.connect(lambda _=False: self._merge_level(5))
        bmerge.addWidget(bm5)
        bmerge.addStretch(1)
        brow.addStretch(1)
        gl.addLayout(brow)
        gl.addLayout(bmerge)
        self.grid_info = QLabel("")
        self.grid_info.setObjectName("hint")
        self.grid_info.setWordWrap(True)
        _shrinkable(self.grid_info, wrap=True)
        gl.addWidget(self.grid_info)

        right = QVBoxLayout()
        g2 = QGroupBox("嵌套层级")
        g2l = QVBoxLayout(g2)
        self.cells_x_spin = QSpinBox()
        self.cells_x_spin.setRange(4, 2000)
        self.cells_x_spin.setValue(100)
        self.cells_x_spin.setToolTip("拼贴从均匀网格开始：横向基础块数。越大格子越多越细腻；"
                                     "打开主图时默认按宽度/32 自动给值。")
        self.cells_x_spin.valueChanged.connect(self._on_cells_changed)
        f1 = QFormLayout()
        f1.addRow("第 1 级基础块数", self.cells_x_spin)
        g2l.addLayout(f1)
        g2l.addWidget(_hint("横向基础块数；自动按“总数约 300”（原版甜点）换算；改过不再覆盖。"))
        self.level_radios: List[QRadioButton] = []
        descs = ["一级：经典方形网格马赛克", "二级：部分块细分 1/2，提升重点细节",
                 "三级：含 1、1/2、1/4 三种尺寸", "四级：含 1/8 尺寸，细节更丰富",
                 "五级：含 1/16 尺寸，适合大幅打印（较慢）"]
        for i, d in enumerate(descs, start=1):
            rb = QRadioButton(d)
            rb.setChecked(i == 4)
            self.level_radios.append(rb)
            g2l.addWidget(rb)
        for rb in self.level_radios:
            rb.toggled.connect(lambda _c=False: self._on_level_change())
        self.auto_check = QCheckBox("按图像细节自动细分（能量阈值）")
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip("勾选后层级上限=所选单选；选「一级」即为经典均匀网格，不再细分。")
        g2l.addWidget(self.auto_check)
        g2l.addWidget(_hint("勾选后层级上限=所选单选；选「一级」即为经典均匀网格，不再细分。"))
        # 细节识别方式：梯度能量（纹理）vs 显著性主体（智能抓主旨）
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["梯度能量（纹理细节）", "显著性主体（智能抓主旨）"])
        self.signal_combo.setToolTip(
            "梯度能量：按纹理/边缘切分（对毛发、织物等纹理敏感）。\n"
            "显著性主体：按视觉重点切分（人脸/主体切细，天空/背景保大块）——更懂图片主旨。")
        fs = QFormLayout()
        fs.addRow("细节识别", self.signal_combo)
        g2l.addLayout(fs)
        g2l.addWidget(_hint("「显著性主体」更懂图：主体切细、背景保大块；「梯度能量」对纹理更敏感。"))
        self.signal_combo.currentIndexChanged.connect(
            lambda _i=0: self._on_signal_change())
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)          # 0=全细分，100=最保守
        self.threshold_slider.setValue(25)
        self.threshold_slider.setToolTip("细节能量超过该值才细分：往左越多细节（慢），往右越保守（快/整齐）。")
        self.threshold_label = QLabel("25")
        self.threshold_label.setObjectName("hint")
        self.threshold_label.setMinimumWidth(28)
        thr_row = QHBoxLayout()
        thr_row.addWidget(self.threshold_slider, 1)
        thr_row.addWidget(self.threshold_label)
        f2 = QFormLayout()
        f2.addRow("细节阈值", thr_row)
        g2l.addLayout(f2)
        g2l.addWidget(_hint("细节能量超过该值才切小：左=细节多（慢），右=保守（快/整齐）；建议 15–40。"))
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(str(v)))
        self.first_level_check = QCheckBox("首级保守（大块不轻易切碎）")
        self.first_level_check.setToolTip("一级基础块需要 2× 阈值的能量才会细分。")
        g2l.addWidget(self.first_level_check)
        g2l.addWidget(_hint("一级大块不轻易切碎：首级细分门槛 ×2，保持大块完整性。"))
        self.auto_check.toggled.connect(lambda _=False: self._refresh_grid_overlay())
        self.threshold_slider.valueChanged.connect(lambda _v=0: self._refresh_grid_overlay_debounced())
        self.first_level_check.toggled.connect(lambda _=False: self._refresh_grid_overlay())
        right.addWidget(g2)

        # ---- 风格（可视化卡片：每种风格用你的主图实时渲染小样例）------------
        gstyle = QGroupBox("风格（点击卡片切换，缩略图是真实效果预览）")
        gstyle_l = QHBoxLayout(gstyle)
        self._style_buttons: Dict[str, QPushButton] = {}
        from .styles import all_styles
        for name, label in all_styles():
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(name == "grid")
            b.setMinimumHeight(56)
            b.clicked.connect(lambda _c=False, n=name: self._select_style(n))
            self._style_buttons[name] = b
            gstyle_l.addWidget(b)
        gstyle_l.addStretch(1)
        right.addWidget(gstyle)
        self._refresh_style_cards()

        g3 = QGroupBox("匹配参数")
        g3l = QFormLayout(g3)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 100)
        self.topk_spin.setValue(8)
        self.topk_spin.setToolTip("每格选最接近的前 K 名瓦片再随机抽一个。1=总选最优。")
        g3l.addRow("近优随机候选 K", self.topk_spin)
        g3l.addWidget(_hint("从前 K 个最像的瓦片里随机挑 1 个：K=1 总选最优，越大越有变化。"))
        self.reuse_spin = QSpinBox()
        self.reuse_spin.setRange(0, 100000)
        self.reuse_spin.setValue(0)
        self.reuse_spin.setSpecialValueText("不限")
        self.reuse_spin.setToolTip("每张瓦片最多被用几次。不限=不限制。")
        g3l.addRow("单块最大复用", self.reuse_spin)
        g3l.addWidget(_hint("每张瓦片最多被用几次；「不限」适合小图库，设值可多用不同照片。"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 500.0)
        self.radius_spin.setValue(0.0)
        self.radius_spin.setSpecialValueText("关闭")
        self.radius_spin.setToolTip("先按平均色粗筛：超出此 RGB 距离的瓦片不参与打分。推荐 3–5。")
        g3l.addRow("颜色半径预筛", self.radius_spin)
        rnote = QLabel("注意：半径不要设太大；官方推荐 3–5。过小可能找不到足够同色图。")
        rnote.setObjectName("hint")
        rnote.setWordWrap(True)
        g3l.addRow("", rnote)
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(55)
        self.blend_slider.setToolTip("瓦片向所在格均色做柔光（SoftLight）混合；0=保留本色。")
        g3l.addRow("色彩融合强度（柔光）", self.blend_slider)
        g3l.addWidget(_hint("瓦片向所在格的主图均色做柔光混合，远看更像原图；0=保留本色。"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2**31 - 1)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("随机")
        self.seed_spin.setToolTip("固定种子 ⇒ 同库同参数逐字节相同结果；-1=每次随机。")
        g3l.addRow("随机种子", self.seed_spin)
        g3l.addWidget(_hint("同库+同参数+同种子 ⇒ 逐字节相同；「随机」每次不同。"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "gpu"])
        self.device_combo.setToolTip("匹配计算跑在哪：auto=有 NVIDIA 自动用 GPU；cpu 强制 CPU；gpu 强制 GPU。")
        g3l.addRow("计算设备", self.device_combo)
        g3l.addWidget(_hint("auto：检测到 N 卡自动用 GPU，无卡回落 CPU。"))
        right.addWidget(g3)
        right.addStretch(1)
        right_w = QWidget()
        right_w.setLayout(right)
        # 左（网格编辑器）右（设置列）改为可拖动分栏：初始 3:1，状态持久化
        self.splitter3 = QSplitter(Qt.Orientation.Horizontal)
        self.splitter3.setChildrenCollapsible(False)
        self.splitter3.addWidget(g)
        self.splitter3.addWidget(right_w)
        self.splitter3.setStretchFactor(0, 3)
        self.splitter3.setStretchFactor(1, 1)
        lay.addWidget(self.splitter3)
        lay.setContentsMargins(0, 0, 16, 0)
        lay.setSpacing(0)
        return _page("网格与嵌套层级", "设置马赛克的格子密度与细分层级；在左侧预览上直接编辑网格。",
                     _scrollable(w, h_scroll=True))

    def _on_level_change(self) -> None:
        if self.tree is not None:
            self.tree.set_max_level(self._radio_level())   # 手动网格的最细粒度随级数联动
        self._refresh_grid_overlay()

    def _on_cells_changed(self, _value: int) -> None:
        if not self._setting_cells:
            self._user_touched_cells = True
        self._rebuild_tree()

    def _on_signal_change(self) -> None:
        """细节识别方式切换：重建信号图并刷新网格预览。"""
        self._build_detail_map()
        self._refresh_grid_overlay()

    # ---- 风格（可视化卡片） ---------------------------------------------
    def _select_style(self, name: str) -> None:
        self._style = name
        for n, b in self._style_buttons.items():
            b.setChecked(n == name)
        self.grid_view.set_style(name)          # 网格预览画出该风格的真实形状
        self._refresh_grid_overlay()

    def _refresh_style_cards(self) -> None:
        """为每种风格用当前主图渲染真实小样例作为卡片图标（独立线程，不占主渲染 worker）。

        关键：不能用 self._start（它独占 self._worker）——风格卡片是辅助预览，
        绝不能阻塞真正的生成/批量任务。用普通线程跑渲染，结果经 QTimer 回主线程贴图标。"""
        master = self._effective_master()
        db_path = self._db_path()
        if not master or not os.path.exists(master) or not os.path.exists(db_path):
            return

        def work() -> Dict[str, str]:
            from .styles import all_styles
            from .db import TileDB
            paths: Dict[str, str] = {}
            with Image.open(master) as im:
                im.thumbnail((120, 120), Image.Resampling.BOX)
                sample = im.convert("RGB")
            sample_path = os.path.join(
                tempfile.gettempdir(), f"pm_stylesample_{os.getpid()}.png")
            sample.save(sample_path)
            with TileDB(db_path) as db:
                for name, _label in all_styles():
                    out = os.path.join(
                        tempfile.gettempdir(),
                        f"pm_stylecard_{name}_{os.getpid()}.png")
                    try:
                        make_mosaic(sample_path, db, out, cells_x=6,
                                    style=name, blend=self.blend_slider.value() / 100.0,
                                    top_k=2, seed=1)
                        paths[name] = out
                    except Exception:
                        pass
            return paths

        def apply_icons(paths) -> None:
            if not isinstance(paths, dict):
                return
            for name, p in paths.items():
                if name in self._style_buttons and os.path.exists(p):
                    self._style_buttons[name].setIcon(QIcon(p))
                    self._style_buttons[name].setIconSize(QSize(40, 40))

        def run() -> None:
            paths = work()
            # 回主线程贴图标
            QTimer.singleShot(0, lambda: apply_icons(paths))

        threading.Thread(target=run, daemon=True).start()

    def _reset_grid(self) -> None:
        if self.tree:
            self.tree.reset()
            self._refresh_grid_overlay()

    def _merge_level(self, level: int) -> None:
        if self.tree:
            n = self.tree.merge_level(level)
            self._refresh_grid_overlay()
            self.status_text.setText(
                f"已收回 {n} 个块的细分（{level} 级及以上全部合并）"
                if n else f"没有 {level} 级及以上的细分可合并")

    def _update_grid_info(self) -> None:
        self._refresh_grid_overlay()

    # ---- page 4: result ----------------------------------------------------
    def _page_result(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        # 顶部控制拆成两行：窄窗口不挤爆（生成/缩放/对比一行，每格像素一行）
        row = QHBoxLayout()
        self.btn_render = QPushButton("生成照片马赛克")
        self.btn_render.setObjectName("primary")
        self.btn_render.setToolTip(
            "生成马赛克。①的主图队列有多张时，对队列里每一张各出一张成品；"
            "只有一张时只出当前那张。")
        self.btn_render.clicked.connect(self._render)
        row.addWidget(self.btn_render)
        self.btn_cancel_render = QPushButton("取消生成")
        self.btn_cancel_render.setEnabled(False)
        self.btn_cancel_render.clicked.connect(lambda: self._render_cancel.set())
        row.addWidget(self.btn_cancel_render)
        self.btn_pause_render = QPushButton("暂停生成")
        self.btn_pause_render.setEnabled(False)
        self.btn_pause_render.setCheckable(True)
        self.btn_pause_render.setToolTip("暂停后进度保留，点“继续生成”接着跑；与“取消”不同，暂停不写断点。")
        self.btn_pause_render.toggled.connect(self._toggle_pause_render)
        row.addWidget(self.btn_pause_render)
        row.addWidget(QLabel("缩放"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["适应窗口", "50%", "100%", "200%"])
        self.zoom_combo.currentIndexChanged.connect(self._show_result)
        row.addWidget(self.zoom_combo)
        self.btn_compare = QPushButton("对比原图")
        self.btn_compare.setCheckable(True)
        self.btn_compare.toggled.connect(self._toggle_compare)
        row.addWidget(self.btn_compare)
        row.addStretch(1)
        lay.addLayout(row)

        # 输出信息独立一行：不和按钮挤，长文本可读
        info_row = QHBoxLayout()
        self.out_info = QLabel("")
        self.out_info.setObjectName("hint")
        _shrinkable(self.out_info, wrap=True)
        info_row.addWidget(self.out_info, 1)
        lay.addLayout(info_row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("每格像素"))
        self.cell_px_spin = QSpinBox()
        self.cell_px_spin.setRange(0, 8192)
        self.cell_px_spin.setValue(0)                # 0 = 与主图同大
        self.cell_px_spin.setSpecialValueText("与主图同大（快速预览）")
        self.cell_px_spin.setSuffix(" px/格")
        self.cell_px_spin.setMinimumWidth(180)
        self.cell_px_spin.setToolTip(
            "每个碎片贴出来占多少像素，直接决定成品清晰度与总尺寸：\n"
            "输出宽度 = 横向基础块数 × 每格像素。\n"
            "0 = 与主图同大（快速预览）；\n"
            "256 = 库里缩略图的清晰度；\n"
            "512 = 重新读取原照片，细节最实；\n"
            "更大 = 放大（文件更大但不会更清晰）。")
        row2.addWidget(self.cell_px_spin)
        row2.addStretch(1)
        self.cell_px_spin.valueChanged.connect(lambda _v=0: self._update_out_info())
        self.cells_x_spin.valueChanged.connect(lambda _v=0: self._update_out_info())
        lay.addLayout(row2)

        # 流式成品的查看入口：应用内分段放大 + 交给系统程序
        row3 = QHBoxLayout()
        self.btn_bigview = QPushButton("🔍 放大查看细节")
        self.btn_bigview.setObjectName("ghost")
        self.btn_bigview.setEnabled(False)           # 仅流式 BigTIFF 成品可用
        self.btn_bigview.setToolTip(
            "在应用内平移/缩放浏览超大 BigTIFF 成品：缩小看预览，放大超过\n"
            "预览分辨率时按需从文件读条带拼出视口（1:1 像素）。仅流式渲染\n"
            "（超过 2 亿像素）的成品可用；普通 PNG 用“缩放”下拉即可。")
        self.btn_bigview.clicked.connect(self._open_bigview)
        row3.addWidget(self.btn_bigview)
        self.btn_open_ext = QPushButton("用系统程序打开")
        self.btn_open_ext.setEnabled(False)
        self.btn_open_ext.setToolTip(
            "把成品交给系统默认看图程序（Photoshop 等）。注意：超大文件\n"
            "（几十 GB 的流式 BigTIFF）系统程序可能打不开。")
        self.btn_open_ext.clicked.connect(self._open_output_externally)
        row3.addWidget(self.btn_open_ext)
        row3.addStretch(1)
        lay.addLayout(row3)

        g = QGroupBox("结果（点击任意碎片可从匹配列表替换）")
        gl = QVBoxLayout(g)
        self.result_view = ResultView()
        self.result_view.cell_clicked.connect(self._replace_cell)
        gl.addWidget(self.result_view, 1)
        lay.addWidget(g, 1)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("默认保存到"))
        self.outdir_edit = QLineEdit()
        self.outdir_edit.setPlaceholderText("未设置：保存到主图所在文件夹")
        self.outdir_edit.setToolTip(
            "生成结果（<主图名>_mosaic.png）与导出对话框的默认位置。\n"
            "留空则保存到主图所在文件夹。")
        orow.addWidget(self.outdir_edit, 1)
        bdir = QPushButton("浏览…")
        bdir.clicked.connect(self._pick_outdir)
        orow.addWidget(bdir)
        bdirclr = QPushButton("清除")
        bdirclr.clicked.connect(lambda _c=False: self.outdir_edit.clear())
        orow.addWidget(bdirclr)
        lay.addLayout(orow)

        erow = QHBoxLayout()
        erow.addWidget(QLabel("导出格式"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "BMP", "TIFF"])
        self.format_combo.currentIndexChanged.connect(
            lambda _i=0: self._update_out_info())
        erow.addWidget(self.format_combo)
        self.out_quality = QSpinBox()
        self.out_quality.setRange(30, 100)
        self.out_quality.setValue(100)
        self.out_quality.valueChanged.connect(
            lambda _v=0: self._update_out_info())
        self.out_quality.setToolTip("仅 JPEG 有效：100=无损级（文件大），90=肉眼难辨（文件小）。")
        erow.addWidget(QLabel("导出质量"))
        erow.addWidget(self.out_quality)
        bexp = QPushButton("导出…")
        bexp.clicked.connect(self._export)
        erow.addWidget(bexp)
        for text, c, r in (("分页打印 3×2（6 张）", 3, 2), ("分页打印 2×3（6 张）", 2, 3)):
            bp = QPushButton(text)
            bp.setToolTip("把成品切成 6 张分别打印，拼起来就是大幅海报（适合 A4 打印机）。")
            bp.clicked.connect(lambda _=False, cc=c, rr=r: self._poster(cc, rr))
            erow.addWidget(bp)
        erow.addStretch(1)
        lay.addLayout(erow)
        self.stats_view = QTextEdit()
        self.stats_view.setReadOnly(True)
        self.stats_view.setMaximumHeight(110)
        self.stats_view.setVisible(False)      # 空时不占位（结果区才能舒展开）
        lay.addWidget(self.stats_view)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        # 与②③页一致地包进滚动区：窄/矮窗口下控件可滚到，不被裁剪
        return _page("生成与修正", "渲染马赛克；点击任意碎片可从候选列表替换，完成后导出。",
                     _scrollable(w, h_scroll=True))

    # ---- render & correct ---------------------------------------------------
    def _effective_master(self) -> Optional[str]:
        if not self.master_path:
            return None
        if not self.master_rot:
            return self.master_path
        # 临时文件名带进程 id：多个 GUI 实例并发时互不覆盖（同进程内渲染串行）
        tmp = os.path.join(tempfile.gettempdir(),
                           f"photomosaic_master_rot_{os.getpid()}.png")
        with Image.open(self.master_path) as im:
            im.convert("RGB").rotate(-self.master_rot, expand=True).save(tmp)
        return tmp

    # ---- output resolution ------------------------------------------------
    def _master_wh(self) -> Tuple[int, int]:
        if self._master_pil is not None:
            return self._master_pil.size
        pix = self._master_pixmap()
        return (pix.width(), pix.height()) if pix is not None else (0, 0)

    # ---- output resolution（以“每格像素”思考，宽度自动推导） ----------------
    def _set_cell_px(self, cp: int) -> None:
        """恢复设置：0=与主图同大；其他=每格像素数。"""
        self.cell_px_spin.blockSignals(True)
        self.cell_px_spin.setValue(max(0, int(cp)))
        self.cell_px_spin.blockSignals(False)
        self._update_out_info()

    def _cell_px(self) -> Optional[int]:
        """当前每格像素；None = 与主图同大。"""
        v = self.cell_px_spin.value()
        return v if v > 0 else None

    def _effective_out_width(self) -> Optional[int]:
        cp = self._cell_px()
        if cp is None:
            return None
        return max(1, self.cells_x_spin.value()) * cp

    # ---- output resolution（以“每格像素”思考，宽度自动推导） ----------------
    @staticmethod
    def _fmt_bytes(n: float) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.1f} GB"
        if n >= 1e6:
            return f"{n / 1e6:.0f} MB"
        return f"{n / 1e3:.0f} KB"

    def _master_sample_ratio(self, fmt: str) -> float:
        """按主图内容实测压缩率（每像素字节数），缓存于 (主图,格式,质量)。

        把主图缩到 ≤1024px 后按所选格式实际编码一次，得到真实压缩率再外推
        到输出尺寸——比固定系数准得多（照片 PNG 实测 0.3~0.6 字节/像素，
        内容差异极大，固定系数必然不准）。"""
        key = (self.master_path, fmt, self.out_quality.value())
        if self._ratio_cache_key == key:
            return self._ratio_cache
        ratio = 0.45 if fmt == "PNG" else 0.25     # 无主图时的兜底
        try:
            path = self._effective_master()
            if path and os.path.exists(path):
                with Image.open(path) as im:
                    im.thumbnail((1024, 1024), Image.Resampling.BOX)
                    rgb = im.convert("RGB")
                    buf = io.BytesIO()
                    if fmt == "JPEG":
                        rgb.save(buf, "JPEG",
                                 quality=self.out_quality.value(), subsampling=0)
                    else:
                        rgb.save(buf, fmt)
                    ratio = max(0.01, buf.tell() / float(rgb.width * rgb.height))
        except Exception:
            pass
        self._ratio_cache_key = key
        self._ratio_cache = ratio
        return ratio

    def _tile_sample_ratio(self, fmt: str) -> Optional[float]:
        """按瓦片库内容实测压缩率：随机取若干张库内缩略图拼成样本画布，
        按所选格式实际编码测出"字节/像素"。

        成品就是瓦片拼出来的——这是内容上最准的代理（主图只是形状参考）。
        库不存在或为空时返回 None，调用方回退主图实测。"""
        db_path = self._db_path()
        if not db_path or not os.path.exists(db_path):
            return None
        try:
            st = os.path.getsize(db_path)
        except OSError:
            return None
        key = (db_path, st, fmt, self.out_quality.value())
        if self._tile_ratio_key == key:
            return self._tile_ratio
        ratio: Optional[float] = None
        try:
            with TileDB(db_path) as db:
                tiles = db.load_arrays()
                ids = tiles["ids"]
                if len(ids):
                    rng = np.random.default_rng(0)          # 固定采样，同库同估值
                    k = int(min(16, len(ids)))
                    pick = sorted(rng.choice(len(ids), k, replace=False).tolist())
                    raws = db.get_thumbs([int(ids[i]) for i in pick])
                    thumbs = [Image.open(io.BytesIO(raws[t])).convert("RGB")
                              for t in raws]
                    tw = max(1, thumbs[0].size[0])
                    cols = max(1, int(np.ceil(np.sqrt(len(thumbs)))))
                    canvas = Image.new("RGB", (cols * tw, cols * tw), (128, 128, 128))
                    for i, th in enumerate(thumbs):
                        canvas.paste(th, ((i % cols) * tw, (i // cols) * tw))
                    buf = io.BytesIO()
                    if fmt == "JPEG":
                        canvas.save(buf, "JPEG",
                                    quality=self.out_quality.value(), subsampling=0)
                    else:
                        canvas.save(buf, fmt)
                    ratio = max(0.01, buf.tell() / float(canvas.width * canvas.height))
        except Exception:
            ratio = None
        self._tile_ratio_key = key
        self._tile_ratio = ratio
        return ratio

    def _estimate_size_text(self, px: int) -> str:
        """按导出格式给出文件大小预估；TIFF/流式 BigTIFF 与非流式 BMP 为精确值。

        流式路由（与 render._stream_output_path 一致）：TIFF→BigTIFF 精确；
        PNG/JPEG/BMP 超阈值 → PNG 流式（压缩，按内容比率估）。"""
        fmt = self.format_combo.currentText()
        if fmt == "TIFF" or (fmt == "BMP" and px <= STREAM_THRESHOLD_PX):
            return f"≈{self._fmt_bytes(px * 3 + 4096)}"          # 无压缩，精确
        # 超阈值时 JPEG/BMP 实际输出为 PNG 流式 → 按 PNG 比率估
        eff_fmt = fmt if px <= STREAM_THRESHOLD_PX else "PNG"
        # 优先按瓦片库内容实测（成品内容），无库时回退主图实测
        ratio = self._tile_sample_ratio(eff_fmt)
        if ratio is None:
            ratio = self._master_sample_ratio(eff_fmt)
        return f"约 {self._fmt_bytes(px * ratio)}"

    def _update_out_info(self) -> None:
        W, H = self._master_wh()
        if not W:
            self.out_info.setText("")
            return
        cells = max(1, self.cells_x_spin.value())
        cp = self._cell_px()
        if cp is None:
            ow, oh = W, H
            head = f"输出 {ow}×{oh} px（与主图同大）"
        else:
            ow = cells * cp
            oh = int(round(H * ow / W))
            head = f"输出 {ow}×{oh} px"
        px = ow * oh
        parts = [head, f"{px / 1e6:.0f} 百万像素",
                 self._estimate_size_text(px)]
        if px > STREAM_THRESHOLD_PX:
            eff = "BigTIFF（可续传）" if self.format_combo.currentText() == "TIFF" \
                else "PNG"
            parts.append(f"⟳ 流式渲染 {eff}")
        elif px > 150e6:
            parts.append("较大，生成较慢")
        self.out_info.setText("  ·  ".join(parts))

    def _pick_outdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择默认保存文件夹")
        if path:
            self.outdir_edit.setText(path)

    def _output_path(self) -> str:
        """渲染输出文件路径：优先“默认保存到”，否则主图所在文件夹。

        命名始终取自主图原文件（而非旋转后的临时文件），避免旋转渲染时
        把成品写进系统临时目录。"""
        base = self.master_path or ""
        stem = os.path.splitext(os.path.basename(base))[0] or "mosaic"
        outdir = self.outdir_edit.text().strip()
        if outdir and os.path.isdir(outdir):
            return os.path.join(outdir, stem + "_mosaic.png")
        if base:
            return os.path.join(os.path.dirname(os.path.abspath(base)),
                                stem + "_mosaic.png")
        return "mosaic.png"

    def _render(self) -> None:
        # 主图队列有多张时，「生成」= 对队列每张各出一张（批量）；一张则只出当前
        if self.master_queue.count() > 1:
            self._render_batch_queue()
            return
        master_file = self._effective_master()
        if not master_file:
            QMessageBox.warning(self, "主图像", "请先在步骤①打开主图。")
            return
        db_path = self._db_path()
        if not os.path.exists(db_path):
            QMessageBox.warning(self, "单元图数据库", "请先在步骤②构建数据库。")
            return
        if self.tree is None:
            self._rebuild_tree()
        out = self._output_path()
        # 关闭自动细分时，手动网格就是最终网格（不再按能量二次细分）
        levels = self._effective_levels() if self.auto_check.isChecked() else 1
        params: Dict[str, object] = {
            "leaf_boxes": self.tree.leaf_boxes() if self.tree else None,
            "levels": levels,
            "detail_threshold": float(self.threshold_slider.value()),
            "signal": "saliency" if self.signal_combo.currentIndex() == 1 else "detail",
            "style": self._style,
            "top_k": self.topk_spin.value(),
            "max_reuse": self.reuse_spin.value(),
            "radius": self.radius_spin.value() or None,
            "seed": None if self.seed_spin.value() < 0 else self.seed_spin.value(),
            "blend": self.blend_slider.value() / 100.0,
            "device": self.device_combo.currentText(),
            "out_width": self._effective_out_width(),
            "first_level_factor": 2.0 if self.first_level_check.isChecked() else 1.0,
        }
        self._render_cancel.clear()
        self._render_pause.clear()
        self.btn_cancel_render.setEnabled(True)
        self.btn_pause_render.setEnabled(True)
        self.btn_pause_render.setChecked(False)
        self.btn_pause_render.setText("暂停生成")
        self.status_text.setText("正在生成…（可暂停；取消会保存断点，同参数再次生成自动续传）")
        self._last_output = out
        self._start(_render_fn(master_file, db_path, out, params,
                               self._render_cancel, self._render_pause),
                    self._render_done)

    def _render_done(self, stats: object) -> None:
        self._set_busy(False)
        self.btn_cancel_render.setEnabled(False)
        self.btn_pause_render.setEnabled(False)
        self.btn_pause_render.setChecked(False)
        self.btn_pause_render.setText("暂停生成")
        self._render_pause.clear()
        self._finish_progress(not (isinstance(stats, dict) and stats.get("cancelled")))
        if isinstance(stats, dict) and stats.get("cancelled"):
            self.status_text.setText("生成已取消：断点已保存，同参数再次生成将自动续传")
            return
        if not isinstance(stats, dict):
            return
        # 真实输出尺寸 / 是否流式：_replace_cell 的 leaf 坐标换算、导出与
        # 海报的分级降级都依赖这两个事实（流式时 _result_pix 只是 4096px
        # 的降采样预览，不能拿来当输出宽度用）。
        self._render_size = tuple(stats["size"]) if stats.get("size") else None
        self._render_streamed = bool(stats.get("streamed"))
        # 超大流式输出：预览走降采样图，不把几十亿像素读进内存
        preview_path = stats.get("preview")
        self._preview_path = str(preview_path) if (preview_path
                                                   and stats.get("streamed")) else None
        load_path = preview_path if (preview_path and stats.get("streamed")) \
            else stats["output"]
        self._result_pix = QPixmap(load_path)    # preview from disk, no RAM copy
        if preview_path and stats.get("streamed"):
            self._last_output = stats["output"]  # 导出/替换仍指向真正的大图
        # 放大查看：仅流式 BigTIFF（PNG 流式不可随机读；普通 PNG 整图在手）；
        # 系统程序打开：只要成品文件在就可用
        is_bigtiff = bool(self._last_output
                          and self._last_output.lower().endswith((".tif", ".tiff")))
        self.btn_bigview.setEnabled(bool(self._render_streamed
                                          and is_bigtiff
                                          and self._last_output
                                          and os.path.exists(self._last_output)))
        self.btn_open_ext.setEnabled(bool(self._last_output
                                          and os.path.exists(self._last_output)))
        self.btn_compare.setChecked(False)
        self._render_leaves = list(stats.get("leaf_rects", []))
        self._render_choices = list(stats.get("choices", []))
        self._render_mode = stats.get("mode", "uniform")
        self.result_view.leaves = self._render_leaves
        self._show_result()
        self.stats_view.setPlainText("\n".join(f"{k}: {v}" for k, v in stats.items()))
        self.stats_view.setVisible(True)       # 有统计内容才显示
        self._goto(3)
        resumed = stats.get("resumed_pasted", 0)
        extra = f"（续传跳过 {resumed} 块）" if resumed else ""
        self.status_text.setText(
            f"完成（设备 {stats.get('device')}）{extra} → {stats['output']}")

    def _batch_overrides(self, masters: List[str]) -> Optional[Dict[str, Dict]]:
        """把主图单独配置整理成 BatchRunner 的 overrides（cell_px 换算成 out_width）。"""
        master_set = set(masters)
        out: Dict[str, Dict] = {}
        for path, ov in self._master_overrides.items():
            if path not in master_set:
                continue
            p = dict(ov)
            if "cell_px" in p:
                cx = int(p.get("cells_x", self.cells_x_spin.value()))
                p["out_width"] = cx * int(p.pop("cell_px"))
            out[path] = p
        return out or None

    def _render_batch_queue(self) -> None:
        """把当前③④页配置批量应用到①的主图队列（每张各出一张成品）。"""
        masters = self._master_queue_paths()
        if not masters:
            QMessageBox.warning(self, "批量生成", "① 主图像里还没有主图队列，请先添加多张主图。")
            return
        db_path = self._db_path()
        if not os.path.exists(db_path):
            QMessageBox.warning(self, "批量生成", "请先在步骤②构建单元图数据库。")
            return
        outdir = self.outdir_edit.text().strip()
        if not (outdir and os.path.isdir(outdir)):
            outdir = os.path.dirname(masters[0])
        levels = self._effective_levels() if self.auto_check.isChecked() else 1
        base_params: Dict[str, object] = {
            "levels": levels,
            "cells_x": self.cells_x_spin.value(),   # 基础块数要跟③页走
            "detail_threshold": float(self.threshold_slider.value()),
            "top_k": self.topk_spin.value(),
            "max_reuse": self.reuse_spin.value(),
            "radius": self.radius_spin.value() or None,
            "seed": None if self.seed_spin.value() < 0 else self.seed_spin.value(),
            "blend": self.blend_slider.value() / 100.0,
            "first_level_factor": 2.0 if self.first_level_check.isChecked() else 1.0,
            "style": self._style,
            "signal": "saliency" if self.signal_combo.currentIndex() == 1 else "detail",
        }
        out_width = self._effective_out_width()
        if out_width:
            base_params["out_width"] = out_width
        overrides = self._batch_overrides(masters)
        from .batch import BatchRunner

        def fn(progress, message):
            with BatchRunner(db_path,
                             device=self.device_combo.currentText()) as runner:
                stats = runner.run(
                    masters, outdir, styles=[self._style],
                    auto_adapt=self.batch_auto_adapt.isChecked()
                    if hasattr(self, "batch_auto_adapt") else False,
                    base_params=base_params,
                    overrides=overrides or None,
                    should_cancel=self._render_cancel.is_set,
                    progress=lambda d, t, p: progress(d, t),
                    status=lambda m: message(m))
            return stats

        def on_done(stats):
            self._set_busy(False)
            self.btn_cancel_render.setEnabled(False)
            self.btn_pause_render.setEnabled(False)
            self._render_pause.clear()
            if isinstance(stats, dict) and "ok" in stats:
                self.status_text.setText(
                    f"批量完成 {stats['ok']}/{stats['total']}（跳过 {stats['skipped']}，"
                    f"失败 {stats['failed']}）→ {stats.get('out_dir')}")
                if stats.get("errors"):
                    QMessageBox.warning(self, "部分失败",
                                        "\n".join(stats["errors"][:10]))

        self._render_cancel.clear()
        self._render_pause.clear()
        self.btn_cancel_render.setEnabled(True)
        self.btn_pause_render.setEnabled(True)
        self.status_text.setText(f"批量生成中…（{len(masters)} 张主图 × 当前风格）")
        self._start(fn, on_done)

    def _toggle_pause_render(self, checked: bool) -> None:
        """暂停/继续生成：暂停只挂起进度回调（不写断点），继续即接着跑。"""
        if checked:
            self._render_pause.set()
            self.btn_pause_render.setText("继续生成")
            self.status_text.setText("已暂停：进度保留，点“继续生成”接着跑")
        else:
            self._render_pause.clear()
            self.btn_pause_render.setText("暂停生成")
            self.status_text.setText("继续生成…")

    def _show_result(self, _idx: int = 0) -> None:
        if self._result_pix is None or self._result_pix.isNull():
            return
        pix = self._result_pix
        mode = self.zoom_combo.currentText()
        if mode == "适应窗口":
            self.result_view.set_pixmap(pix)
        else:
            pct = int(mode.rstrip("%"))
            self.result_view.set_pixmap(pix.scaledToWidth(
                max(1, int(pix.width() * pct / 100.0)),
                Qt.TransformationMode.SmoothTransformation))

    def _toggle_compare(self, checked: bool) -> None:
        if checked:
            pix = self._master_pixmap()
            if pix is not None:
                # 流式输出显示的是降采样预览：主图也缩到同宽，切换对比时
                # 尺寸一致不跳变
                if self._result_pix is not None and not self._result_pix.isNull() \
                        and pix.width() != self._result_pix.width():
                    pix = pix.scaledToWidth(
                        self._result_pix.width(),
                        Qt.TransformationMode.SmoothTransformation)
                self.result_view.set_pixmap(pix)
        elif self._result_pix is not None:
            self.result_view.set_pixmap(self._result_pix)

    # ---- 超大成品查看 ------------------------------------------------------
    def _open_bigview(self) -> None:
        """流式 BigTIFF 成品的应用内分段查看（放大看 1:1 细节）。"""
        if not (self._render_streamed and self._last_output and self._render_size):
            return
        if not os.path.exists(self._last_output):
            self.status_text.setText("成品文件已不在磁盘上，无法放大查看")
            return
        preview = self._preview_path
        if preview and not os.path.exists(preview):
            preview = None                 # 预览丢了也能看：全程走条带读取
        try:
            dlg = HugeImageViewDialog(self._last_output, preview,
                                      self._render_size, parent=self)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "放大查看", f"无法读取 BigTIFF：\n{exc}")
            return
        dlg.exec()

    def _open_output_externally(self) -> None:
        """把成品交给系统默认程序（Photoshop 等）；超大文件可能打不开。"""
        if not self._last_output or not os.path.exists(self._last_output):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_output))
        self.status_text.setText(f"已请求系统程序打开 → {self._last_output}")

    def _replace_cell(self, leaf) -> None:
        if not self._last_output or not os.path.exists(self._last_output):
            return
        if self._render_streamed:
            # 流式 BigTIFF 无法整图载入内存做单格替换：诚实降级，好过 OOM
            self.status_text.setText("流式超大输出暂不支持点击替换")
            return
        if self._render_mode == "uniform":
            self.status_text.setText("均匀网格模式不支持点击替换；请在步骤③使用网格编辑器（自适应/手动）模式")
            return
        master_file = self._effective_master()
        if not master_file:
            return
        with Image.open(master_file) as im:
            master_rgb = im.convert("RGB")
            # leaf rect is in OUTPUT coordinates; features are computed in MASTER coords.
            # s 必须取真实输出宽度（stats["size"]）：流式时 _result_pix 只是
            # 降采样预览（如 4096px），用它会把特征框算错位置。
            s = 1.0
            if self._render_size is not None and master_rgb.width:
                s = self._render_size[0] / master_rgb.width or 1.0
            elif self._result_pix is not None and not self._result_pix.isNull() \
                    and master_rgb.width:
                s = self._result_pix.width() / master_rgb.width or 1.0
            mbox = (int(leaf[0] / s), int(leaf[1] / s),
                    int(leaf[2] / s), int(leaf[3] / s))
            feat, avg = rect_features(master_rgb, mbox)
        with TileDB(self._db_path()) as db:
            tiles = db.load_arrays()
            matcher = Matcher(tiles, top_k=12, device=self.device_combo.currentText())
            cands = matcher.top_candidates(feat, 12)
            dlg = ReplaceDialog(db, cands, self)
            if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_pos() is None:
                return
            pos = dlg.result_pos()
            raw = db.get_thumb(int(tiles["ids"][pos]))
        size = (leaf[2] - leaf[0], leaf[3] - leaf[1])
        thumb = Image.open(io.BytesIO(raw)).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        if self.blend_slider.value():
            solid = Image.new("RGB", size, (int(avg[0]), int(avg[1]), int(avg[2])))
            mixed = ImageChops.soft_light(thumb, solid)
            thumb = Image.blend(thumb, mixed, self.blend_slider.value() / 100.0)
        with Image.open(self._last_output) as cur:   # load-on-demand, release after
            canvas = cur.convert("RGB")
            canvas.paste(thumb, (leaf[0], leaf[1]))
            self._save_output_atomic(canvas)
        self._result_pix = QPixmap(self._last_output)
        self._show_result()
        self.status_text.setText(f"已替换碎片 {leaf[:4]}")

    def _save_output_atomic(self, canvas: Image.Image) -> None:
        """整图画布落盘：临时文件写完再 os.replace 原子替换。

        按扩展名分支：JPEG 用界面设定的质量（Pillow 默认 75 会悄悄掉质量，
        且替换前半张图损坏比一张临时文件更糟）；TIFF 显式 "TIFF"；PNG 等
        走扩展名默认推断。中途崩溃只会留下临时文件，成品不会被替换掉一半。
        """
        ext = os.path.splitext(self._last_output)[1].lower()
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=ext, dir=os.path.dirname(os.path.abspath(self._last_output)))
        os.close(tmp_fd)
        try:
            if ext in (".jpg", ".jpeg"):
                canvas.save(tmp_path, "JPEG",
                            quality=self.out_quality.value(), subsampling=0)
            elif ext in (".tif", ".tiff"):
                canvas.save(tmp_path, "TIFF")
            else:
                canvas.save(tmp_path)
            os.replace(tmp_path, self._last_output)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _export(self) -> None:
        if not self._last_output or not os.path.exists(self._last_output):
            return
        fmt = self.format_combo.currentText()
        ext = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp", "TIFF": ".tif"}[fmt]
        if self._render_streamed and fmt != "TIFF":
            # 几十上百 GB 的 BigTIFF 重新编码 JPEG/PNG 既慢也会 OOM，不现实
            self.status_text.setText("超大流式输出仅支持 TIFF 原样导出")
            return
        # 导出对话框默认落在“默认保存到”目录（其次成品所在目录）
        outdir = self.outdir_edit.text().strip()
        if not (outdir and os.path.isdir(outdir)) and self._last_output:
            outdir = os.path.dirname(self._last_output)
        start = os.path.join(outdir, f"mosaic{ext}") if outdir else f"mosaic{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "导出马赛克", start,
                                              f"{fmt} (*{ext})")
        if not path:
            return
        if self._render_streamed:
            # 流式 BigTIFF → TIFF：直接复制文件，无损、零内存、快
            shutil.copy(self._last_output, path)
            size = self._render_size or (0, 0)
        else:
            with Image.open(self._last_output) as img:
                size = img.size
                if fmt == "JPEG":
                    img.convert("RGB").save(path, "JPEG", quality=self.out_quality.value(),
                                            subsampling=0)
                else:
                    img.save(path, fmt)
        self.status_text.setText(f"已导出 {size[0]}×{size[1]} px → {path}")

    def _poster(self, cols: int, rows: int) -> None:
        if not self._last_output:
            return
        if self._render_streamed:
            # 海报分页需要整图载入并逐页裁切，对超大流式输出不现实：诚实降级
            self.status_text.setText("流式超大输出暂不支持海报分页")
            return
        out_dir = os.path.join(os.path.dirname(self._last_output), "poster")
        with Image.open(self._last_output) as img:
            paths = export_poster(img.convert("RGB"), cols, rows, out_dir, "poster")
        self.status_text.setText(f"海报分页已导出 {len(paths)} 张 → {out_dir}")

    # ---- workers ---------------------------------------------------------
    def _start(self, fn, on_done: Callable[[object], None]) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "忙", "已有任务在运行。")
            return
        self._worker = _Worker(fn)
        self._progress_unit = ""          # 新任务从默认刻度开始（清掉上次的条带刻度）
        self._worker.progress.connect(self._on_progress)
        self._worker.message.connect(self._on_worker_message)
        self._worker.finished_ok.connect(on_done)
        self._worker.failed.connect(self._on_failed)
        self._set_busy(True)
        self._worker.start()

    def _set_busy(self, busy: bool) -> None:
        for w in (self.btn_render, self.btn_ingest):
            w.setEnabled(not busy)

    def _on_worker_message(self, text: str) -> None:
        self.status_text.setText(text)
        # 匹配阶段进度按"叶子数"计、流式贴图阶段按"条带数"计：收到流式状态
        # 消息即切换刻度（条带从 0 重走属于换阶段，不是倒退）
        if "正在流式渲染" in text:
            self._progress_unit = "条带"

    def _on_progress(self, done: int, total: int) -> None:
        unit = self._progress_unit
        for bar in (self.lib_progress,):
            if total <= 0:
                bar.setRange(0, 0)          # 未知总量：忙碌动画
            else:
                bar.setRange(0, total)
                bar.setValue(min(done, total))
                bar.setFormat(f"{unit} {done} / {total}  (%p%)" if unit
                              else f"{done} / {total}  (%p%)")
        # 渲染期间把进度写到结果区中央（没成品时那里是占位文本）
        if self.btn_cancel_render.isEnabled() and self._result_pix is None:
            pct = int(100 * done / max(1, total)) if total > 0 else 0
            unit_txt = f"{unit} " if unit else ""
            self.result_view.setText(
                f"正在生成… {pct}%\n{unit_txt}{done} / {total}")

    def _finish_progress(self, ok: bool) -> None:
        self._progress_unit = ""            # 结束后回到默认刻度
        for bar in (self.lib_progress,):
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            bar.setValue(100 if ok else 0)

    def _on_failed(self, msg: str) -> None:
        self._set_busy(False)
        self._finish_progress(False)
        self.btn_pause.setEnabled(False)
        self.btn_cancel_ingest.setEnabled(False)
        self.btn_cancel_render.setEnabled(False)
        self.status_text.setText("失败")
        QMessageBox.critical(self, "任务失败", msg)

    # ---- prefill (for demos/tests) -----------------------------------------
    def prefill(self, master: Optional[str] = None, db: Optional[str] = None,
                folder: Optional[str] = None) -> None:
        if folder and not self.folder_list.findItems(folder, Qt.MatchFlag.MatchExactly):
            self.folder_list.addItem(folder)
        if db:
            self.db_edit.setText(db)
        if master and os.path.exists(master):
            self.master_path = master
            self._show_master()
            if not self.db_edit.text():
                self.db_edit.setText(os.path.join(os.path.dirname(master), "tiles.db"))


_ERROR_LOG = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "photomosaic-studio", "error.log")


def _install_excepthook() -> None:
    """全局崩溃兜底：未处理异常先写 error.log，再用 QTimer 异步弹中文提示。

    对话框不能直接在 hook 里模态弹——hook 可能在非 GUI 线程被触发；
    而且模态会卡死离屏测试。写日志是同步的，弹窗延后到事件循环下一拍。"""
    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            os.makedirs(os.path.dirname(_ERROR_LOG), exist_ok=True)
            with open(_ERROR_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
        except Exception:
            pass
        def _popup():
            try:
                if QApplication.instance():
                    QMessageBox.critical(
                        None, "出现未处理的错误",
                        f"{etype.__name__}: {value}\n\n已记录到 {_ERROR_LOG}")
            except Exception:
                pass
        QTimer.singleShot(0, _popup) if QApplication.instance() else None
    sys.excepthook = hook


def _bundled_demo_master() -> Optional[str]:
    """打包进 exe 的首次启动演示主图（add-data 进来；源码跑时用仓库里的）。"""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, "demo_output", "master.png")
    return p if os.path.exists(p) else None


def run_gui(argv: Optional[List[str]] = None, master: Optional[str] = None,
            db: Optional[str] = None, folder: Optional[str] = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv[1:])
    _install_excepthook()
    win = MainWindow()
    win.prefill(master=master, db=db, folder=folder)
    # 首次打开（无存档主图、无命令行主图）→ 默认加载打包进来的演示图
    if win.master_path is None:
        demo = _bundled_demo_master()
        if demo:
            win.prefill(master=demo)
    win.show()
    return app.exec()
