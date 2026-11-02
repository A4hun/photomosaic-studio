"""超大输出分段查看器（在应用内平移/缩放浏览流式 BigTIFF 成品）。

几十 GB 的流式 BigTIFF 无法整图载入内存。本对话框的策略：

* 缩放低于预览分辨率（zoom ≤ 预览宽/全宽）→ 直接把预览图按缩放画，
  内存里已有的数据足够；
* 放大超过预览分辨率 → 计算视口在全分辨率中的矩形，后台 QThread 用
  BigTiffStripReader.read_rows 取行区间（read_rows 是整宽的，读回后按
  视口 x 裁剪），转 QPixmap 贴出；放大渲染用最近邻（FastTransformation，
  看像素就是看像素）。

请求节奏：缩放/拖拽停 150ms 才发起（去抖）；同一视口不重复读；新请求
发起时旧结果按代号（generation）直接丢弃。后台线程绝不碰 QWidget——
结果一律经信号回到 GUI 线程后再贴。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .bigtiff import BigTiffStripReader

_MAX_ZOOM = 64.0        # 最大放大倍数（64× 已远超任何实际需要）
_ZOOM_STEP = 1.25       # 滚轮每档缩放倍率
_EPS = 1e-9

# 已启动但尚未结束的读取线程：对话框可能先关，线程在后台自灭（结果按
# 代号丢弃、信号随接收者销毁自动断开），期间不能被 Python GC 回收。
_ACTIVE_WORKERS = set()


class _RowsWorker(QThread):
    """后台读一段行区间；结果（数组或异常）经 done 信号送回 GUI 线程。"""

    done = Signal(int, QRect, object)      # generation, 源矩形, ndarray|Exception

    def __init__(self, reader: BigTiffStripReader, gen: int, rect: QRect):
        super().__init__()
        self._reader = reader
        self._gen = gen
        self._rect = rect

    def run(self) -> None:  # pragma: no cover - 线程入口
        try:
            arr = self._reader.read_rows(self._rect.y(),
                                         self._rect.y() + self._rect.height())
            self.done.emit(self._gen, self._rect, arr)
        except Exception as exc:  # noqa: BLE001
            self.done.emit(self._gen, self._rect, exc)


class _ZoomCanvas(QWidget):
    """画布：滚轮缩放（锚点=鼠标位置）、左键拖拽平移、双击 适应/100%。

    坐标约定：zoom z 为“全图像素 → 屏幕像素”的倍率；pan 为全图原点在
    控件里的屏幕位置。放大时只画像素（FastTransformation）。
    """

    viewport_changed = Signal()            # 缩放/平移改变（查看器据此去抖取数）
    zoom_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.full_size = (0, 0)            # 全图 (W, H)
        self.preview: Optional[QPixmap] = None
        self._tile_rect: Optional[QRect] = None   # 当前贴片覆盖的全图矩形
        self._tile_pix: Optional[QPixmap] = None
        self._tile_arr: Optional[np.ndarray] = None   # 测试/诊断：贴片原始像素
        self._zoom = 1.0
        self._pan_x, self._pan_y = 0.0, 0.0
        self._drag: Optional[QPointF] = None
        self._fitted = False               # 首次得到有效尺寸时做一次“适应窗口”

    # ---- 数据注入 ---------------------------------------------------------
    def set_full_size(self, w: int, h: int) -> None:
        self.full_size = (int(w), int(h))
        self._fitted = False

    def set_preview(self, pix: Optional[QPixmap]) -> None:
        self.preview = pix if (pix is not None and not pix.isNull()) else None
        self.update()

    def set_tile(self, rect: QRect, pix: QPixmap, arr: np.ndarray) -> None:
        self._tile_rect, self._tile_pix, self._tile_arr = rect, pix, arr
        self.update()

    def clear_tile(self) -> None:
        self._tile_rect = self._tile_pix = self._tile_arr = None
        self.update()

    # ---- 视口几何 ---------------------------------------------------------
    def preview_scale(self) -> float:
        """预览图相对全图的采样密度（0 = 没有预览，永远走条带读取）。"""
        if self.preview is None or not self.full_size[0]:
            return 0.0
        return self.preview.width() / float(self.full_size[0])

    def uses_preview(self) -> bool:
        """当前缩放下预览分辨率是否够用（再放大就要读真条带）。"""
        return self._zoom <= self.preview_scale() * (1.0 + 1e-3)

    def zoom(self) -> float:
        return self._zoom

    def fit_zoom(self) -> float:
        w, h = self.full_size
        if not w or not h:
            return 1.0
        return min(self.width() / float(w), self.height() / float(h))

    def pan(self) -> Tuple[float, float]:
        return (self._pan_x, self._pan_y)

    def set_zoom(self, z: float, anchor: Optional[QPointF] = None) -> None:
        """设缩放倍率；anchor（控件坐标）下的源像素保持在原地不动。"""
        if self._fitted:
            lo = self.fit_zoom()
        else:
            lo = _EPS
        z = min(max(z, lo), _MAX_ZOOM)
        if abs(z - self._zoom) < _EPS * max(1.0, z):
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        src_x = (anchor.x() - self._pan_x) / self._zoom
        src_y = (anchor.y() - self._pan_y) / self._zoom
        self._zoom = z
        self._pan_x = anchor.x() - src_x * z
        self._pan_y = anchor.y() - src_y * z
        self._clamp_pan()
        self.zoom_changed.emit(z)
        self.viewport_changed.emit()
        self.update()

    def fit_window(self) -> None:
        """整图适应窗口（居中）。"""
        w, h = self.full_size
        if not w or not h:
            return
        self._fitted = True
        z = self.fit_zoom()
        self._zoom = z
        self._clamp_pan()
        self.zoom_changed.emit(z)
        self.viewport_changed.emit()
        self.update()

    def _clamp_pan(self) -> None:
        """图像比视口小 → 居中；更大 → 平移范围限制在图像边界内。"""
        w, h = self.full_size
        iw, ih = w * self._zoom, h * self._zoom
        vw, vh = self.width(), self.height()
        if iw <= vw:
            self._pan_x = (vw - iw) / 2.0
        else:
            self._pan_x = min(0.0, max(vw - iw, self._pan_x))
        if ih <= vh:
            self._pan_y = (vh - ih) / 2.0
        else:
            self._pan_y = min(0.0, max(vh - ih, self._pan_y))

    def source_rect(self) -> QRect:
        """当前视口在全图坐标中的矩形（整数、夹取到图像边界内）。"""
        w, h = self.full_size
        z = max(_EPS, self._zoom)
        ox = -self._pan_x / z                  # 视口左上角的全图坐标
        oy = -self._pan_y / z
        vw = self.width() / z
        vh = self.height() / z
        x0 = int(max(0, np.floor(ox)))
        y0 = int(max(0, np.floor(oy)))
        x1 = int(min(w, np.ceil(ox + vw)))
        y1 = int(min(h, np.ceil(oy + vh)))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        return QRect(x0, y0, x1 - x0, y1 - y0)

    # ---- 绘制 -------------------------------------------------------------
    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111827"))
        w, h = self.full_size
        if not w or not h:
            p.end()
            return
        zw, zh = w * self._zoom, h * self._zoom
        target = QRect(int(round(self._pan_x)), int(round(self._pan_y)),
                       int(round(zw)), int(round(zh)))
        if self.uses_preview() and self.preview is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawPixmap(target, self.preview)
        else:
            # 底层：放大画预览（近似内容，等条带真数据到达后覆盖）
            if self.preview is not None:
                p.drawPixmap(target, self.preview)   # 默认最近邻
            # 上层：条带拼出的视口贴片（按全图坐标对位，最近邻放大）
            if self._tile_pix is not None and self._tile_rect is not None \
                    and self._tile_rect.width() > 0:
                sx = self._pan_x + self._tile_rect.x() * self._zoom
                sy = self._pan_y + self._tile_rect.y() * self._zoom
                tw = self._tile_rect.width() * self._zoom
                th = self._tile_rect.height() * self._zoom
                p.drawPixmap(QRect(int(round(sx)), int(round(sy)),
                                   int(round(tw)), int(round(th))),
                             self._tile_pix)
        p.end()

    # ---- 交互 -------------------------------------------------------------
    def wheelEvent(self, ev) -> None:
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            return
        self.set_zoom(self._zoom * (_ZOOM_STEP ** steps), ev.position())

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ev) -> None:
        if self._drag is not None:
            cur = ev.position()
            self._pan_x += cur.x() - self._drag.x()
            self._pan_y += cur.y() - self._drag.y()
            self._drag = cur
            self._clamp_pan()
            self.viewport_changed.emit()
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            self._drag = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, ev) -> None:
        """双击：在“适应窗口”与“100%”之间切换。"""
        if abs(self._zoom - 1.0) < 5e-3:
            self.fit_window()
        else:
            self.set_zoom(1.0)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if not self._fitted and self.width() > 0 and self.full_size[0]:
            self.fit_window()
        else:
            self._clamp_pan()
            self.viewport_changed.emit()
            self.update()


class HugeImageViewDialog(QDialog):
    """流式 BigTIFF 成品的分段查看对话框（模态）。

    参数：tif_path 成品路径；preview 预览图（QPixmap 或路径，可为 None）；
    full_size 声明的全图 (W, H)（与文件不符时以文件为准）；reader 可注入
    记录调用的包装（测试用，注入时对话框不负责关闭它）。
    """

    def __init__(self, tif_path, preview=None, full_size: Tuple[int, int] = (0, 0),
                 parent: Optional[QWidget] = None, reader=None,
                 cache_strips: int = 2, debounce_ms: int = 150):
        super().__init__(parent)
        self._own_reader = reader is None
        self.reader = reader if reader is not None else \
            BigTiffStripReader(str(tif_path), cache_strips=cache_strips)
        W, H = int(self.reader.width), int(self.reader.height)
        self._gen = 0                     # 请求代号：新请求发起后旧结果作废
        self._worker: Optional[_RowsWorker] = None
        self._pending_rect: Optional[QRect] = None
        self._last_rect: Optional[QRect] = None
        self._reading = False
        self._closed = False
        self._err: Optional[str] = None

        self.setWindowTitle(f"放大查看 — {os.path.basename(str(tif_path))}"
                            f" · {W}×{H}px")
        self.setModal(True)
        self.resize(900, 640)

        lay = QVBoxLayout(self)
        self.canvas = _ZoomCanvas()
        self.canvas.set_full_size(W, H)
        pix: Optional[QPixmap] = None
        if isinstance(preview, QPixmap):
            pix = preview
        elif preview:
            pix = QPixmap(str(preview))
        self.canvas.set_preview(pix)
        if full_size and tuple(full_size) != (W, H):
            self._err = (f"声明的尺寸 {tuple(full_size)} 与文件 {W}×{H} 不符，"
                         "已按文件显示")
        lay.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        tip = QLabel("滚轮缩放（锚点=鼠标） · 左键拖拽平移 · 双击 适应/100%")
        tip.setObjectName("hint")
        bottom.addWidget(tip)
        bottom.addStretch(1)
        self.status_label = QLabel("")
        self.status_label.setObjectName("stat")
        bottom.addWidget(self.status_label)
        lay.addLayout(bottom)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(max(0, int(debounce_ms)))
        self._debounce.timeout.connect(self._start_read)
        self.canvas.viewport_changed.connect(self._on_viewport_changed)
        self.canvas.zoom_changed.connect(lambda _z: self._update_status())
        self._update_status()

    # ---- 请求调度 ---------------------------------------------------------
    def _on_viewport_changed(self) -> None:
        """视口变了：预览够用就只重画；需要真数据则去抖后读取。"""
        self._update_status()
        if self._closed or self.canvas.uses_preview():
            return
        if self._debounce.isActive():
            self._debounce.stop()
        self._debounce.start()

    def _start_read(self) -> None:
        """发起（或排队）读取当前视口覆盖的行区间。"""
        if self._closed or self.canvas.uses_preview():
            self._set_reading(False)
            return
        rect = self.canvas.source_rect()
        if rect == self._last_rect:
            self._set_reading(False)       # 同一视口不重复读
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending_rect = rect      # 在途：只记最新需求，完成后接着读
            self._set_reading(True)
            return
        self._launch(rect)

    def _launch(self, rect: QRect) -> None:
        self._gen += 1
        self._last_rect = rect
        worker = _RowsWorker(self.reader, self._gen, rect)
        _ACTIVE_WORKERS.add(worker)
        # 无接收者的清理连接：对话框已销毁时也保证注册表能放手
        worker.finished.connect(lambda w=worker: _ACTIVE_WORKERS.discard(w))
        # 绑定方法的连接：随对话框销毁自动断开（此时线程自灭）
        worker.finished.connect(self._on_worker_finished)
        worker.done.connect(self._on_rows)
        self._worker = worker
        self._set_reading(True)
        worker.start()

    def _on_worker_finished(self) -> None:
        w = self._worker
        if w is not None:
            _ACTIVE_WORKERS.discard(w)
            self._worker = None
        if self._closed:
            return
        # 线程空闲了：若期间视口又变过（积压了最新需求），接着读
        rect, self._pending_rect = self._pending_rect, None
        if rect is not None and rect != self._last_rect \
                and not self.canvas.uses_preview():
            self._launch(rect)
            return
        self._set_reading(False)

    def _on_rows(self, gen: int, rect: QRect, payload: object) -> None:
        if self._closed or gen != self._gen:
            return                          # 过期结果：直接丢弃
        if isinstance(payload, Exception):
            self._err = f"读取条带失败：{payload}"
            self._set_reading(False)
            return
        arr = payload
        try:
            sub = np.ascontiguousarray(arr[:, rect.x():rect.x() + rect.width()])
            h, w = sub.shape[:2]
            if w and h:
                buf = sub.tobytes()
                qimg = QImage(buf, w, h, w * 3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg.copy())   # copy → 自持数据
                self.canvas.set_tile(rect, pix, sub)
        finally:
            self._set_reading(False)

    def _set_reading(self, busy: bool) -> None:
        self._reading = busy
        self._update_status()

    def _update_status(self) -> None:
        pct = int(round(self.canvas.zoom() * 100))
        if self._reading:
            text = f"{pct}% · 读取条带…"
        elif self._err:
            text = f"{pct}% · {self._err}"
        else:
            text = f"{pct}%"
        self.status_label.setText(text)

    # ---- 生命周期 ---------------------------------------------------------
    def _shutdown(self) -> None:
        """Esc/关闭时收尾：作废在途结果、停去抖、等待/放弃后台线程、关 reader。"""
        if self._closed:
            return
        self._closed = True
        self._gen += 1_000_000             # 在途结果全部过期
        self._debounce.stop()
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.done.disconnect(self._on_rows)
            worker.wait(300)               # 条带读取通常毫秒级；超时让后台自灭
        self._worker = None
        if self._own_reader:
            self.reader.close()

    def reject(self) -> None:              # Esc → QDialog.reject
        self._shutdown()
        super().reject()

    def accept(self) -> None:
        self._shutdown()
        super().accept()

    def closeEvent(self, ev) -> None:      # 标题栏关闭
        self._shutdown()
        super().closeEvent(ev)
