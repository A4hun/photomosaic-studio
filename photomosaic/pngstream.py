"""流式 PNG 编码器：超大输出（数百亿像素）不占整图内存。

PNG 结构 = 签名 + IHDR + IDAT（zlib 压缩的逐行原始像素，每行带 1 字节
过滤器类型）+ IEND。行序从上到下固定——正好匹配分条带流式渲染。

与 BigTiffStripWriter 的差异：PNG 的 zlib 流不可中途定位，因此**不支持
断点续传**（取消后重跑只能从头来）；需要续传的超大输出请选 TIFF。
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_IDAT_FLUSH = 8 * 1024 * 1024          # 攒够 8MB 压缩数据就落一块 IDAT


class PngStreamWriter:
    """RGB 8bit PNG 流式写出；write_strip 必须按条带顺序（从上到下）调用。"""

    def __init__(self, path: str, width: int, height: int):
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        if self.width < 1 or self.height < 1:
            raise ValueError("width/height must be positive")
        self._fh = open(self.path, "wb")
        self._z = zlib.compressobj(6)
        self._pending = bytearray()
        self._next = 0                   # 期望的下一条带索引（强制顺序写）
        self._rows_done = 0
        self._fh.write(_PNG_SIG)
        self._write_chunk(b"IHDR", struct.pack(
            ">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))

    def _write_chunk(self, ctype: bytes, data: bytes) -> None:
        self._fh.write(struct.pack(">I", len(data)))
        self._fh.write(ctype)
        self._fh.write(data)
        crc = zlib.crc32(ctype)
        crc = zlib.crc32(data, crc)
        self._fh.write(struct.pack(">I", crc & 0xFFFFFFFF))

    def _flush_pending(self, final: bytes = b"") -> None:
        if self._pending or final:
            self._write_chunk(b"IDAT", bytes(self._pending) + final)
            self._pending.clear()

    def write_strip(self, index: int, arr: np.ndarray) -> None:
        """写第 index 条带（rows, W, 3) uint8；必须顺序调用。"""
        if index != self._next:
            raise ValueError(
                f"PNG 流式写入必须按条带顺序：期望第 {self._next} 条，收到第 {index} 条")
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        rows = a.shape[0]
        if a.shape[1:] != (self.width, 3):
            raise ValueError(f"strip {index}: 宽度应为 {self.width}，实际 {a.shape[1:]}")
        if self._rows_done + rows > self.height:
            raise ValueError("条带行数超出图像高度")
        # 每行前加过滤器字节 0（None filter），然后整段喂给 zlib
        body = a.reshape(rows, self.width * 3)
        filtered = np.concatenate(
            [np.zeros((rows, 1), dtype=np.uint8), body], axis=1)
        self._pending += self._z.compress(filtered.tobytes())
        self._rows_done += rows
        self._next += 1
        if len(self._pending) >= _IDAT_FLUSH:
            self._flush_pending()

    def close(self) -> None:
        try:
            if self._fh.closed:
                return
            self._pending += self._z.flush()
            self._flush_pending()
            self._write_chunk(b"IEND", b"")
        finally:
            self._fh.close()

    def __enter__(self) -> "PngStreamWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def png_estimate_size(width: int, height: int) -> int:
    """PNG 大小无法预知（zlib 压缩看内容），返回未压缩上界。"""
    return int(width) * int(height) * 3 + 1024
