"""流式 PNG 编码器：超大输出（数百亿像素）不占整图内存。

PNG 结构 = 签名 + IHDR + IDAT（zlib 压缩的逐行原始像素，每行带 1 字节
过滤器类型）+ IEND。行序从上到下固定——正好匹配分条带流式渲染。

性能：zlib 压缩是单线程 CPU 瓶颈。本实现把压缩挪到独立后台线程
（主线程贴条带、后台线程压缩落盘，两者重叠），并默认 level 1——
超大图上 level 6 的压缩率收益不值那 2~3 倍的时间。

与 BigTiffStripWriter 的差异：PNG 的 zlib 流不可中途定位，因此**不支持
断点续传**（取消后重跑只能从头来）；需要续传的超大输出请选 TIFF。
"""

from __future__ import annotations

import queue
import struct
import threading
import zlib

import numpy as np

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_IDAT_FLUSH = 8 * 1024 * 1024          # 攒够 8MB 压缩数据就落一块 IDAT


class PngStreamWriter:
    """RGB 8bit PNG 流式写出；write_strip 必须按条带顺序（从上到下）调用。

    压缩在独立后台线程进行（zlib 流必须按序喂，故压缩串行但与主线程的
    贴图/解码重叠）；有界队列施加背压，内存不随条带数增长。
    """

    def __init__(self, path: str, width: int, height: int,
                 level: int = 1):
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        if self.width < 1 or self.height < 1:
            raise ValueError("width/height must be positive")
        self._fh = open(self.path, "wb")
        self._z = zlib.compressobj(level)
        self._pending = bytearray()
        self._next = 0                   # 期望的下一条带索引（强制顺序写）
        self._rows_done = 0
        self._error: Optional[BaseException] = None
        # 有界队列：背压（主线程贴图快于压缩时阻塞，最多积压 1 条带）
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=1)
        self._fh.write(_PNG_SIG)
        self._write_chunk(b"IHDR", struct.pack(
            ">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        self._worker = threading.Thread(target=self._compress_loop, daemon=True)
        self._worker.start()

    # ---- 后台线程：按序压缩 + 落盘 ------------------------------------
    def _write_chunk(self, ctype: bytes, data: bytes) -> None:
        self._fh.write(struct.pack(">I", len(data)))
        self._fh.write(ctype)
        self._fh.write(data)
        crc = zlib.crc32(ctype)
        crc = zlib.crc32(data, crc)
        self._fh.write(struct.pack(">I", crc & 0xFFFFFFFF))

    def _flush_pending(self) -> None:
        if self._pending:
            self._write_chunk(b"IDAT", bytes(self._pending))
            self._pending.clear()

    def _compress_loop(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:                   # 收尾信号
                    self._pending += self._z.flush()
                    self._flush_pending()
                    self._write_chunk(b"IEND", b"")
                    return
                self._pending += self._z.compress(item)
                if len(self._pending) >= _IDAT_FLUSH:
                    self._flush_pending()
        except BaseException as exc:               # noqa: BLE001
            self._error = exc

    # ---- 主线程：贴条带 ---------------------------------------------
    def write_strip(self, index: int, arr: np.ndarray) -> None:
        """写第 index 条带（rows, W, 3) uint8；必须顺序调用。"""
        if self._error is not None:
            raise RuntimeError(f"PNG 压缩线程已失败：{self._error}") from self._error
        if index != self._next:
            raise ValueError(
                f"PNG 流式写入必须按条带顺序：期望第 {self._next} 条，收到第 {index} 条")
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        rows = a.shape[0]
        if a.shape[1:] != (self.width, 3):
            raise ValueError(f"strip {index}: 宽度应为 {self.width}，实际 {a.shape[1:]}")
        if self._rows_done + rows > self.height:
            raise ValueError("条带行数超出图像高度")
        # 每行前加过滤器字节 0（None filter），再入队给后台线程压缩
        body = a.reshape(rows, self.width * 3)
        filtered = np.concatenate(
            [np.zeros((rows, 1), dtype=np.uint8), body], axis=1)
        self._q.put(filtered.tobytes())
        self._rows_done += rows
        self._next += 1

    def close(self) -> None:
        if self._fh.closed:
            return
        self._q.put(None)                          # 通知后台线程收尾
        self._worker.join()
        try:
            if self._error is not None:
                raise RuntimeError(
                    f"PNG 压缩线程已失败：{self._error}") from self._error
        finally:
            self._fh.close()

    def __enter__(self) -> "PngStreamWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def png_estimate_size(width: int, height: int) -> int:
    """PNG 大小无法预知（zlib 压缩看内容），返回未压缩上界。"""
    return int(width) * int(height) * 3 + 1024
