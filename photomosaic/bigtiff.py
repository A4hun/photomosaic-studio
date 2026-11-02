"""Minimal BigTIFF (strip) writer + reader for extremely large RGB mosaics.

纯 Python、只写：RGB 8bit、无压缩、按条带 (strip) 顺序存放。文件头、IFD 与
两个条带偏移表在打开时就全部落盘——所有条带（除最后一条）字节数相同，布局
可以预计算——因此任意时刻崩溃/取消后重新打开同一文件，都能按固定偏移续写
未完成的条带（`append=True` 时校验现有文件大小后原位续写）。

规格参考：BigTIFF（magic 43，8 字节偏移）。本写入器只使用无压缩 RGB strip
布局，Tag 集：256/257/258/259/262/273/277/278/279。

读取端 `BigTiffStripReader` 只解析本写入器产出的那 9 个 tag（小端、magic 43、
8 字节偏移），StripOffsets/StripByteCounts 兼容 LONG8 外置数组与 count==1
内联两种写法（写入器两种都会产出）；条带偏移固定 → 可以按行区间随机读，
用于应用内平移/缩放浏览几十 GB 的成品而无需整图载入。
"""

from __future__ import annotations

import os
import struct
import threading
from collections import OrderedDict

import numpy as np

_BYTEORDER = b"II"          # little-endian
_BIGTIFF_VERSION = 43
_OFFSET_SIZE = 8

# TIFF 字段类型
_SHORT = 3                   # 2 字节
_LONG = 4                    # 4 字节
_LONG8 = 16                  # 8 字节（BigTIFF）

_N_TAGS = 9

# 读取端关心的整数字段类型 → struct 格式字符（按 count 展开）
_INT_FMT = {_SHORT: "H", _LONG: "I", _LONG8: "Q"}

# Tag 编号（仅解析本写入器写出的集合）
_TAG_WIDTH = 256
_TAG_HEIGHT = 257
_TAG_BITS = 258
_TAG_COMPRESSION = 259
_TAG_PHOTOMETRIC = 262
_TAG_STRIP_OFFSETS = 273
_TAG_SAMPLES = 277
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_COUNTS = 279


def strip_layout(width: int, height: int, rows_per_strip: int):
    """归一化条带参数，返回 (rows_per_strip, n_strips)。"""
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("width/height must be positive")
    rps = max(1, min(int(rows_per_strip), height))
    n = (height + rps - 1) // rps
    return rps, n


def estimate_size(width: int, height: int, rows_per_strip: int) -> int:
    """预测输出文件字节数（用于渲染前的磁盘空间检查）。"""
    rps, n = strip_layout(width, height, rows_per_strip)
    width, height = int(width), int(height)
    full = rps * width * 3
    last = (height - rps * (n - 1)) * width * 3
    ifd_off = 16 + 16 * n                       # StripOffsets + StripByteCounts 两张 LONG8 表
    data_off = (ifd_off + 8 + 20 * _N_TAGS + 8 + 15) & ~15
    return data_off + (n - 1) * full + last


class BigTiffStripWriter:
    """逐条带写出 RGB 8bit 无压缩 BigTIFF。

    布局（小端、8 字节对齐）::

        0     header 16 B（"II", 43, 8, 0, IFD 偏移）
        16    StripOffsets   表（LONG8 × n_strips）
        +     StripByteCounts 表（LONG8 × n_strips）
        +     IFD（8 + 20×9 + 8）
        +     条带数据（固定步长 = rows_per_strip × width × 3）
    """

    def __init__(self, path: str, width: int, height: int,
                 rows_per_strip: int, append: bool = False):
        self.path = str(path)
        self.width, self.height = int(width), int(height)
        if self.width < 1 or self.height < 1:
            raise ValueError("width/height must be positive")
        self.rps, self.n_strips = strip_layout(self.width, self.height,
                                               rows_per_strip)
        self.full_strip_bytes = self.rps * self.width * 3
        rows_last = self.height - self.rps * (self.n_strips - 1)
        self.counts = np.full(self.n_strips, self.full_strip_bytes, dtype=np.int64)
        self.counts[-1] = rows_last * self.width * 3

        self._soff_off = 16
        self._sbc_off = self._soff_off + 8 * self.n_strips
        self._ifd_off = self._sbc_off + 8 * self.n_strips
        data_off = (self._ifd_off + 8 + 20 * _N_TAGS + 8 + 15) & ~15
        self.offsets = data_off + np.arange(self.n_strips, dtype=np.int64) \
            * self.full_strip_bytes
        self.expected_size = int(self.offsets[-1] + self.counts[-1])
        assert self.expected_size == estimate_size(self.width, self.height,
                                                   self.rps)

        append = append and os.path.exists(self.path)
        if append and os.path.getsize(self.path) != self.expected_size:
            raise ValueError(
                f"{self.path} 的大小与预期布局不符"
                f"（{os.path.getsize(self.path)} != {self.expected_size}），不能续写")
        self._fh = open(self.path, "r+b" if append else "wb")
        if not append:
            self._write_layout()

    # ---- 文件头 / IFD ----------------------------------------------------
    @staticmethod
    def _entry(tag: int, typ: int, count: int, payload: bytes) -> bytes:
        return struct.pack("<HHQ", tag, typ, count) + payload.ljust(8, b"\0")

    def _offset_entry(self, tag: int, values: np.ndarray, table_off: int) -> bytes:
        """LONG8 数组条目：count=1 时值内联在 8 字节 payload 里（BigTIFF 规则：
        值放得下就不外置），否则 payload 是外部表地址。"""
        if len(values) == 1:
            return self._entry(tag, _LONG8, 1, struct.pack("<Q", int(values[0])))
        return self._entry(tag, _LONG8, len(values), struct.pack("<Q", table_off))

    def _write_layout(self) -> None:
        f = self._fh
        f.seek(0)
        f.write(_BYTEORDER)
        f.write(struct.pack("<HHH", _BIGTIFF_VERSION, _OFFSET_SIZE, 0))
        f.write(struct.pack("<Q", self._ifd_off))
        f.seek(self._soff_off)
        f.write(self.offsets.astype("<i8").tobytes())
        f.seek(self._sbc_off)
        f.write(self.counts.astype("<i8").tobytes())
        f.seek(self._ifd_off)
        f.write(struct.pack("<Q", _N_TAGS))
        for e in (
            self._entry(256, _LONG, 1, struct.pack("<I", self.width)),
            self._entry(257, _LONG, 1, struct.pack("<I", self.height)),
            self._entry(258, _SHORT, 3, struct.pack("<HHH", 8, 8, 8)),
            self._entry(259, _SHORT, 1, struct.pack("<H", 1)),        # 无压缩
            self._entry(262, _SHORT, 1, struct.pack("<H", 2)),        # RGB
            self._offset_entry(273, self.offsets, self._soff_off),    # StripOffsets
            self._entry(277, _SHORT, 1, struct.pack("<H", 3)),
            self._entry(278, _LONG, 1, struct.pack("<I", self.rps)),
            self._offset_entry(279, self.counts, self._sbc_off),      # StripByteCounts
        ):
            f.write(e)
        f.write(struct.pack("<Q", 0))                                  # next IFD
        if self.expected_size > 0:                                     # 预分配到完整大小
            f.seek(self.expected_size - 1)
            f.write(b"\0")
        f.flush()

    # ---- 条带 -------------------------------------------------------------
    def rows_of(self, index: int) -> int:
        self._check_index(index)
        return min(self.rps, self.height - index * self.rps)

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self.n_strips:
            raise IndexError(
                f"strip index {index} out of range 0..{self.n_strips - 1}")

    def write_strip(self, index: int, arr: np.ndarray) -> None:
        rows = self.rows_of(index)
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        if a.shape != (rows, self.width, 3):
            raise ValueError(
                f"strip {index}: expect {(rows, self.width, 3)}, got {a.shape}")
        self._fh.seek(int(self.offsets[index]))
        self._fh.write(a.tobytes())

    def flush(self, sync: bool = False) -> None:
        self._fh.flush()
        if sync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "BigTiffStripWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class BigTiffFormatError(ValueError):
    """BigTIFF 读取端发现的结构错误（损坏、截断或非本格式）。"""


class BigTiffStripReader:
    """按行区间随机读取本写入器产出的 BigTIFF（strip 布局）。

    只解析小端、magic 43、8 字节偏移的 BigTIFF，取 9 个标准 tag；值既兼容
    LONG8 外置数组（count>1 时 payload 是表地址），也兼容 count==1 时内联
    在 8 字节 payload 里的写法——写入器两种都会产出。条带偏移在打开时一次
    解析并校验，之后 `read_rows` 用普通文件 seek+read 取行，配合小 LRU 缓存
    让平移浏览时相邻请求不重复读盘。

    返回的数组可能是缓存条带的只读视图（单条带命中时零拷贝），请勿原地修改。
    """

    def __init__(self, path: str, cache_strips: int = 2):
        self.path = str(path)
        # 单条带 ≤384MB：缓存 4 条 ≈1.5GB 上限偏高，默认 2（≈768MB 封顶）
        self._cache_strips = max(0, int(cache_strips))
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()      # 多线程并发 read_rows 时串行化
        self.io_reads = 0                  # 实际读盘的条带次数（诊断/测试用）
        try:
            self._fh = open(self.path, "rb")
            self._parse()
        except Exception:
            try:
                self._fh.close()
            except Exception:
                pass
            raise

    # ---- 打开时解析 ------------------------------------------------------
    def _read_exact(self, n: int, what: str) -> bytes:
        data = self._fh.read(n)
        if len(data) != n:
            where = self._fh.tell()
            raise BigTiffFormatError(
                f"{self.path}: 文件被截断（读 {what} 还需 {n} 字节，"
                f"实际只读到偏移 {where}）")
        return data

    def _parse(self) -> None:
        self._fh.seek(0)
        head = self._read_exact(16, "文件头")
        if head[0:2] != _BYTEORDER:
            raise BigTiffFormatError(
                f"{self.path}: 字节序标记 {head[0:2]!r} 不是 II（小端）——"
                "不是本写入器产出的 BigTIFF")
        version, offset_size, _pad = struct.unpack("<HHH", head[2:8])
        if version != _BIGTIFF_VERSION:
            raise BigTiffFormatError(
                f"{self.path}: magic={version}（应为 {_BIGTIFF_VERSION}），"
                "不是 BigTIFF 或文件已损坏")
        if offset_size != _OFFSET_SIZE:
            raise BigTiffFormatError(
                f"{self.path}: 偏移字节数 {offset_size}（应为 {_OFFSET_SIZE}）")
        ifd_off = struct.unpack("<Q", head[8:16])[0]

        self._fh.seek(ifd_off)
        n_tags = struct.unpack("<Q", self._read_exact(8, "IFD 条目数"))[0]
        if not 0 < n_tags <= 64:
            raise BigTiffFormatError(
                f"{self.path}: IFD 条目数 {n_tags} 异常（应在 1..64）")
        fields: "OrderedDict[int, tuple]" = OrderedDict()
        for _ in range(n_tags):
            raw = self._read_exact(20, "IFD 条目")
            tag, typ, count = struct.unpack("<HHQ", raw[:12])
            payload = raw[12:20]
            if typ not in _INT_FMT:
                continue                      # 非整数字段：跳过（本读取器用不到）
            size = struct.calcsize(_INT_FMT[typ])
            total = size * count
            if total <= 8:
                val_bytes = payload[:total]   # 内联值（含 count==1 的 LONG8）
            else:
                arr_off = struct.unpack("<Q", payload)[0]
                pos = self._fh.tell()
                self._fh.seek(arr_off)
                val_bytes = self._read_exact(total, f"tag {tag} 外置数组")
                self._fh.seek(pos)
            fields[tag] = (typ, count, val_bytes)

        def need(tag: int) -> tuple:
            if tag not in fields:
                raise BigTiffFormatError(
                    f"{self.path}: 缺少必需 tag {tag}——不是本写入器产出的文件")
            return fields[tag]

        def single(tag: int) -> int:
            typ, count, raw = need(tag)
            return struct.unpack(f"<{_INT_FMT[typ]}", raw[:struct.calcsize(_INT_FMT[typ])])[0]

        def values(tag: int) -> list:
            typ, count, raw = need(tag)
            return list(struct.unpack(f"<{count}{_INT_FMT[typ]}", raw))

        # ---- 基本尺寸与像素格式（不匹配 → 明确报错而不是读出垃圾）----
        self.width, self.height = single(_TAG_WIDTH), single(_TAG_HEIGHT)
        if self.width < 1 or self.height < 1:
            raise BigTiffFormatError(
                f"{self.path}: 尺寸 {self.width}×{self.height} 非法")
        if single(_TAG_COMPRESSION) != 1:
            raise BigTiffFormatError(f"{self.path}: 不是无压缩 TIFF（compression≠1）")
        if single(_TAG_PHOTOMETRIC) != 2:
            raise BigTiffFormatError(f"{self.path}: 不是 RGB（photometric≠2）")
        if single(_TAG_SAMPLES) != 3:
            raise BigTiffFormatError(f"{self.path}: 每像素采样数≠3")
        if values(_TAG_BITS) != [8, 8, 8]:
            raise BigTiffFormatError(f"{self.path}: BitsPerSample 不是 8/8/8")

        # ---- 条带布局：rps / 偏移表 / 字节数三方一致性 ----
        self.rps = single(_TAG_ROWS_PER_STRIP)
        if not 1 <= self.rps <= self.height:
            raise BigTiffFormatError(
                f"{self.path}: RowsPerStrip={self.rps} 超出高度 {self.height}")
        self.n_strips = (self.height + self.rps - 1) // self.rps
        self.offsets = np.asarray(values(_TAG_STRIP_OFFSETS), dtype=np.int64)
        counts = np.asarray(values(_TAG_STRIP_COUNTS), dtype=np.int64)
        if len(self.offsets) != self.n_strips or len(counts) != self.n_strips:
            raise BigTiffFormatError(
                f"{self.path}: 条带偏移/计数表长度与 {self.height} 高、"
                f"rps={self.rps} 推出的 {self.n_strips} 条不一致")
        file_size = os.path.getsize(self.path)
        for i in range(self.n_strips):
            rows = min(self.rps, self.height - i * self.rps)
            expect = rows * self.width * 3
            if counts[i] != expect:
                raise BigTiffFormatError(
                    f"{self.path}: 条带 {i} 字节数 {counts[i]} ≠ 行数×宽×3"
                    f"（应为 {expect}）")
            if self.offsets[i] < 0 or self.offsets[i] + counts[i] > file_size:
                raise BigTiffFormatError(
                    f"{self.path}: 条带 {i} 越过文件末尾"
                    f"（[{self.offsets[i]}, {self.offsets[i] + counts[i]})，"
                    f"文件只有 {file_size} 字节）——文件被截断")

    # ---- 条带读取 ------------------------------------------------------
    def rows_of(self, index: int) -> int:
        """条带 index 的行数（最后一条可能不足 rps）。"""
        self._check_index(index)
        return min(self.rps, self.height - index * self.rps)

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self.n_strips:
            raise IndexError(
                f"strip index {index} out of range 0..{self.n_strips - 1}")

    def _strip(self, index: int) -> np.ndarray:
        """取整条带 (rows, W, 3)；LRU 命中不出盘。"""
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        rows = self.rows_of(index)
        nbytes = int(rows * self.width * 3)
        self._fh.seek(int(self.offsets[index]))
        data = self._read_exact(nbytes, f"条带 {index}")
        self.io_reads += 1
        arr = np.frombuffer(data, dtype=np.uint8).reshape(rows, self.width, 3)
        if self._cache_strips > 0:
            self._cache[index] = arr
            while len(self._cache) > self._cache_strips:
                self._cache.popitem(last=False)   # 最久未用先出
        return arr

    def read_rows(self, y0: int, y1: int) -> np.ndarray:
        """读 [y0, y1) 行区间 → (y1-y0, W, 3) uint8（整宽，按需自行裁剪 x）。

        结果可能是缓存条带的只读视图；跨条带时返回新拼接数组。
        """
        y0, y1 = int(y0), int(y1)
        if y0 < 0 or y1 > self.height or y0 >= y1:
            raise ValueError(
                f"行区间 [{y0}, {y1}) 超出图像高度 {self.height}")
        with self._lock:
            i0, i1 = y0 // self.rps, (y1 - 1) // self.rps
            parts = []
            for i in range(i0, i1 + 1):
                strip = self._strip(i)
                top = max(0, y0 - i * self.rps)                 # 首条带内偏移
                bottom = min(strip.shape[0], y1 - i * self.rps)  # 末条带内截尾
                if bottom > top:
                    parts.append(strip[top:bottom])
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    def close(self) -> None:
        acquired = self._lock.acquire(timeout=1.0)  # 等在途读取收尾（最多 1 秒）
        try:
            self._fh.close()
        finally:
            if acquired:
                self._lock.release()
        self._cache.clear()

    def __enter__(self) -> "BigTiffStripReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
