# -*- coding: utf-8 -*-
"""QA 脚本公共助手：离屏 Qt、合成素材、worker 等待、手势模拟、断言记账。

所有 QA 脚本一律 QT_QPA_PLATFORM=offscreen 离屏运行，绝不弹真实窗口。
弹窗（QMessageBox）会卡死离屏事件循环——这里统一打桩成"记录后返回"。
"""

from __future__ import annotations

import os
import sys
import time
import traceback

# 必须在任何 PySide6 导入之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MSGBOX_CALLS: list = []          # 所有被打桩拦截的弹窗（标题/文本）


def patch_dialogs() -> None:
    """把阻塞式弹窗打桩成记录器；离屏下弹窗会卡死，必须拦截。"""
    def rec(kind):
        def _fn(*args, **kwargs):
            MSGBOX_CALLS.append((kind, args))
            print(f"  [msgbox:{kind}] {args[-1] if args else ''}")
            return None
        return _fn
    QMessageBox.critical = staticmethod(rec("critical"))
    QMessageBox.warning = staticmethod(rec("warning"))
    QMessageBox.information = staticmethod(rec("information"))
    QMessageBox.question = staticmethod(rec("question"))


def get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def pump(ms: int = 0) -> None:
    """抽干事件队列；ms>0 时等待指定毫秒（覆盖去抖定时器）。"""
    app = get_app()
    app.processEvents()
    if ms > 0:
        QTest.qWait(ms)
    app.processEvents()


def wait_worker(win, timeout_s: float = 120.0) -> bool:
    """等待 win._worker 跑完；返回 False=超时。"""
    app = get_app()
    t0 = time.monotonic()
    while win._worker is not None and win._worker.isRunning():
        app.processEvents()
        QTest.qWait(15)
        if time.monotonic() - t0 > timeout_s:
            return False
    app.processEvents()
    return True


# ---------------------------------------------------------------- 素材合成

def make_tiles(folder: str, n: int = 12, size: int = 48, seed: int = 7) -> str:
    """生成 n 张高色度纯色/渐变瓦片（和模糊主图底色区分度高）。"""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        base = rng.integers(0, 256, 3, dtype=np.int64)
        # 拉满色度：最强通道 200+、最弱 <60，保证和灰底不撞
        hi = int(np.argmax(base))
        lo = int(np.argmin(base))
        base[hi] = 200 + (i * 7) % 55
        base[lo] = (i * 13) % 60
        arr = np.empty((size, size, 3), dtype=np.uint8)
        arr[:, :] = base.astype(np.uint8)
        # 加一点斜向渐变，让瓦片不是死平（匹配特征更丰富）
        grad = np.linspace(-24, 24, size, dtype=np.int64)
        arr = np.clip(arr.astype(np.int64) + grad[None, :, None], 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(folder, f"tile_{i:02d}.png"))
    return folder


def make_master(path: str, w: int = 256, h: int = 256, seed: int = 3) -> str:
    """左半亮平、右半暗平 + 中央高细节噪点圆盘：既有平坦区也有高能量区。"""
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = 210
    arr[:, w // 2:] = 45
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (min(w, h) // 4) ** 2
    noise = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    arr[disc] = noise[disc]
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------- 断言记账

class Checker:
    def __init__(self, name: str):
        self.name = name
        self.failures: list = []
        self.passes: int = 0

    def check(self, cond: bool, label: str, detail: str = "") -> bool:
        if cond:
            self.passes += 1
            print(f"  PASS  {label}")
        else:
            self.failures.append(label)
            print(f"  FAIL  {label}  {detail}")
        return bool(cond)

    def section(self, title: str) -> None:
        print(f"\n== {title}")

    def summary(self) -> int:
        total = self.passes + len(self.failures)
        print(f"\n[{self.name}] {self.passes}/{total} 通过")
        if self.failures:
            print(f"[{self.name}] 失败项:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"[{self.name}] 全部通过")
        return 0


def run_guarded(main_fn) -> int:
    """脚本入口兜底：异常打印 traceback 并以 2 退出。"""
    try:
        return int(main_fn())
    except SystemExit as e:
        return int(e.code or 0)
    except Exception:
        traceback.print_exc()
        return 2
