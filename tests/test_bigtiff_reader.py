"""BigTiffStripReader 单元测试（读取端）。

用 BigTiffStripWriter 写已知图案后读回：多条带外置数组与单条带 count==1
内联两种布局都必须逐字节相等；坏文件（错误 magic / 截断 / 尺寸不符）抛
清晰异常；LRU 缓存命中不出盘。
"""

import os
import struct

import numpy as np
import pytest

from photomosaic.bigtiff import (BigTiffFormatError, BigTiffStripReader,
                                 BigTiffStripWriter)


def _pattern(w: int, h: int) -> np.ndarray:
    """已知图案：R=x+行 G=y*3 B=(x^y)，逐字节可验证且各向不同。"""
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.empty((h, w, 3), dtype=np.uint8)
    arr[..., 0] = (xx + yy) % 256
    arr[..., 1] = (yy * 3) % 256
    arr[..., 2] = (xx ^ yy) % 256
    return arr


def _write(path, arr: np.ndarray, rps: int) -> None:
    h, w = arr.shape[:2]
    with BigTiffStripWriter(str(path), w, h, rps) as wr:
        for i in range(wr.n_strips):
            y0 = i * wr.rps
            wr.write_strip(i, arr[y0:y0 + wr.rows_of(i)])


class TestRoundtrip:
    def test_multistrip_arbitrary_ranges(self, tmp_path):
        """多条带（count>1 外置数组）：任意行区间逐字节相等。"""
        W, H, RPS = 97, 250, 64
        exp = _pattern(W, H)
        out = tmp_path / "t.tif"
        _write(out, exp, RPS)

        with BigTiffStripReader(str(out)) as rd:
            assert (rd.width, rd.height, rd.rps, rd.n_strips) == (W, H, RPS, 4)
            for y0, y1 in ((0, 250), (0, 1), (63, 64), (63, 65), (64, 128),
                           (100, 180), (249, 250), (127, 200), (0, 64)):
                got = rd.read_rows(y0, y1)
                assert got.shape == (y1 - y0, W, 3)
                assert np.array_equal(got, exp[y0:y1]), f"rows [{y0},{y1})"

    def test_single_strip_inline_count1(self, tmp_path):
        """单条带 count==1：StripOffsets/ByteCounts 内联（不落外置表）。"""
        W, H = 8, 8
        exp = _pattern(W, H)
        out = tmp_path / "one.tif"
        _write(out, exp, H)          # rps=H → 1 条带（count==1 → 内联）
        with BigTiffStripReader(str(out)) as rd:
            assert rd.n_strips == 1
            got = rd.read_rows(2, 6)
            assert np.array_equal(got, exp[2:6])
            assert np.array_equal(rd.read_rows(0, H), exp)

    def test_cache_hits_do_not_re_read(self, tmp_path):
        """LRU：同一条带重复命中不再读盘；容量外换出后重读。"""
        W, H, RPS = 32, 100, 10
        exp = _pattern(W, H)
        out = tmp_path / "c.tif"
        _write(out, exp, RPS)
        with BigTiffStripReader(str(out), cache_strips=2) as rd:
            rd.read_rows(0, 10)             # 条带 0 → 1 次盘
            rd.read_rows(0, 5)              # 命中缓存
            rd.read_rows(0, 10)             # 命中缓存
            assert rd.io_reads == 1
            rd.read_rows(40, 50)            # 条带 4 → 2 次
            rd.read_rows(50, 60)            # 条带 5 → 3 次（缓存 4、5）
            assert rd.io_reads == 3
            rd.read_rows(0, 10)             # 条带 0 已被换出 → 重读
            assert rd.io_reads == 4
            assert np.array_equal(rd.read_rows(40, 60), exp[40:60])

    def test_cache_strips_zero_disables_cache(self, tmp_path):
        exp = _pattern(32, 40)
        out = tmp_path / "n.tif"
        _write(out, exp, 10)
        with BigTiffStripReader(str(out), cache_strips=0) as rd:
            rd.read_rows(0, 10)
            rd.read_rows(0, 10)
            assert rd.io_reads == 2          # 不缓存：每次都读盘

    def test_full_width_returned_for_x_crop_by_caller(self, tmp_path):
        """read_rows 返回整宽：调用方自行裁剪 x（查看器就这么用）。"""
        W, H, RPS = 40, 30, 10
        exp = _pattern(W, H)
        out = tmp_path / "w.tif"
        _write(out, exp, RPS)
        with BigTiffStripReader(str(out)) as rd:
            got = rd.read_rows(10, 20)
            assert got.shape == (10, W, 3)
            assert np.array_equal(got[:, 5:20], exp[10:20, 5:20])


class TestBadFiles:
    def test_wrong_magic(self, tmp_path):
        p = tmp_path / "bad.tif"
        p.write_bytes(b"MM" + struct.pack("<HHHQ", 43, 8, 0, 16) + b"\0" * 64)
        with pytest.raises(BigTiffFormatError, match="字节序"):
            BigTiffStripReader(str(p))

    def test_not_bigtiff_version(self, tmp_path):
        p = tmp_path / "classic.tif"
        p.write_bytes(b"II" + struct.pack("<HH", 42, 0) + struct.pack("<I", 8)
                      + b"\0" * 32)
        with pytest.raises(BigTiffFormatError, match="magic=42"):
            BigTiffStripReader(str(p))

    def test_truncated_file(self, tmp_path):
        """写完再腰斩：条带越过文件末尾 → 报“截断”。"""
        W, H, RPS = 32, 100, 10
        exp = _pattern(W, H)
        out = tmp_path / "cut.tif"
        _write(out, exp, RPS)
        data = out.read_bytes()
        (tmp_path / "cut2.tif").write_bytes(data[:len(data) // 2])
        with pytest.raises(BigTiffFormatError, match="截断"):
            BigTiffStripReader(str(tmp_path / "cut2.tif"))

    def test_truncated_header(self, tmp_path):
        p = tmp_path / "tiny.tif"
        p.write_bytes(b"II*\0" + b"\0" * 4)
        with pytest.raises(BigTiffFormatError):
            BigTiffStripReader(str(p))

    def test_row_range_out_of_bounds(self, tmp_path):
        exp = _pattern(16, 20)
        out = tmp_path / "r.tif"
        _write(out, exp, 10)
        with BigTiffStripReader(str(out)) as rd:
            with pytest.raises(ValueError, match="超出图像高度"):
                rd.read_rows(15, 21)
            with pytest.raises(ValueError, match="超出图像高度"):
                rd.read_rows(-1, 5)
            with pytest.raises(ValueError, match="超出图像高度"):
                rd.read_rows(5, 5)
