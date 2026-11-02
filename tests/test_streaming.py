"""Streaming (strip-based) giant-output rendering: pixel-exact vs the in-memory
path, strip-boundary leaves, and resume-after-cancel."""

import os

import numpy as np
import pytest
from PIL import Image

tifffile = pytest.importorskip("tifffile")

from photomosaic.bigtiff import BigTiffStripWriter, strip_layout  # noqa: E402
from photomosaic.db import TileDB  # noqa: E402
from photomosaic.library import ingest  # noqa: E402
from photomosaic.render import RenderCancelled, make_mosaic  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = np.random.default_rng(3)
    tmp = tmp_path_factory.mktemp("stream")
    folder = tmp / "tiles"
    folder.mkdir()
    for i in range(8):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            folder / f"t{i}.png")
    master = tmp / "m.png"
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(master)
    db = TileDB()
    ingest(db, str(folder))
    yield str(master), db, tmp
    db.close()


class TestBigTiffWriter:
    def test_known_pattern_roundtrip(self, tmp_path):
        W, H, RPS = 97, 250, 64
        out = tmp_path / "t.tif"
        exp = np.empty((H, W, 3), dtype=np.uint8)
        with BigTiffStripWriter(str(out), W, H, RPS) as w:
            assert w.n_strips == (H + RPS - 1) // RPS
            for i in range(w.n_strips):
                rows = w.rows_of(i)
                y0 = i * RPS
                yy = np.arange(y0, y0 + rows, dtype=np.int32)
                xx = np.arange(W, dtype=np.int32)
                strip = np.empty((rows, W, 3), dtype=np.uint8)
                strip[..., 0] = ((yy[:, None] + xx[None, :]) % 256)
                strip[..., 1] = (40 * (i + 1)) % 256
                strip[..., 2] = (xx * 2) % 256
                w.write_strip(i, strip)
                exp[y0:y0 + rows] = strip
        arr = tifffile.imread(str(out))
        assert arr.shape == (H, W, 3)
        assert np.array_equal(arr, exp)

    def test_single_strip_inline_offsets(self, tmp_path):
        # count==1 的 StripOffsets 必须内联（回归：外置写法会让读者把头部当像素）
        W, H = 8, 8
        out = tmp_path / "one.tif"
        with BigTiffStripWriter(str(out), W, H, H) as w:
            assert w.n_strips == 1
            w.write_strip(0, np.full((H, W, 3), 7, dtype=np.uint8))
        arr = tifffile.imread(str(out))
        assert np.array_equal(arr, np.full((H, W, 3), 7, dtype=np.uint8))

    def test_layout_and_append(self, tmp_path):
        rps, n = strip_layout(100, 130, 64)
        assert (rps, n) == (64, 3)
        out = tmp_path / "a.tif"
        with BigTiffStripWriter(str(out), 100, 130, 64) as w:
            w.write_strip(1, np.zeros((64, 100, 3), dtype=np.uint8))
        with BigTiffStripWriter(str(out), 100, 130, 64, append=True) as w:
            w.write_strip(0, np.full((64, 100, 3), 9, dtype=np.uint8))
            w.write_strip(2, np.full((2, 100, 3), 5, dtype=np.uint8))
        arr = tifffile.imread(str(out))
        assert (arr[:64] == 9).all()
        assert (arr[64:128] == 0).all()
        assert (arr[128:] == 5).all()

    def test_bad_shape_rejected(self, tmp_path):
        with BigTiffStripWriter(str(tmp_path / "b.tif"), 10, 10, 5) as w:
            with pytest.raises(ValueError):
                w.write_strip(0, np.zeros((4, 10, 3), dtype=np.uint8))


def _render(setup, out, **kw):
    master, db, tmp = setup
    common = dict(cells_x=8, levels=3, detail_threshold=15.0, top_k=3, seed=1)
    common.update(kw)
    return make_mosaic(master, db, str(out), **common)


def test_streaming_matches_normal_exactly(setup):
    _, _, tmp = setup
    out_png = tmp / "normal.png"
    out_tif = tmp / "stream.tif"
    s1 = _render(setup, out_png, stream_threshold_px=10 ** 12)
    s2 = _render(setup, out_tif, stream_threshold_px=1)
    assert s1.get("streamed") is None
    assert s2["streamed"] is True
    assert s2["strips"] >= 1
    assert list(s1["choices"]) == list(s2["choices"])
    a = np.asarray(Image.open(out_png).convert("RGB"))
    b = tifffile.imread(str(out_tif))
    assert np.array_equal(a, b)


def test_streaming_multistrip_crossing_leaves(setup):
    """强制小条带：叶子跨条带时按相交部分裁剪，仍与整图逐像素一致。"""
    _, _, tmp = setup
    out_png = tmp / "n2.png"
    out_tif = tmp / "s2.tif"
    _render(setup, out_png, stream_threshold_px=10 ** 12)
    s2 = _render(setup, out_tif, stream_threshold_px=1, stream_strip_rows=7)
    assert s2["strips"] > 1, "must actually exercise multiple strips"
    # 128 高 / 7 行 → 19 条带；自适应叶子高度 ≥4 一定有跨界
    a = np.asarray(Image.open(out_png).convert("RGB"))
    b = tifffile.imread(str(out_tif))
    assert np.array_equal(a, b)


def test_streaming_resume_after_cancel(setup):
    _, _, tmp = setup
    ref = tmp / "ref.tif"
    s_ref = _render(setup, ref, stream_threshold_px=1, stream_strip_rows=7)

    out = tmp / "res.tif"
    state = {"armed": False, "seen": 0}

    def cancel():
        if not state["armed"]:
            return False
        state["seen"] += 1
        return state["seen"] > 3          # 条带循环内第 4 次检查后取消

    class _Arm:
        """matching 阶段不取消；贴图条带开始后才武装。"""

        def __init__(self, fn):
            self._fn = fn

        def __call__(self):
            return self._fn()

    # 用 status 钩子在"正在流式渲染"时武装取消
    def arm_status(msg):
        if "流式渲染" in msg:
            state["armed"] = True

    with pytest.raises(RenderCancelled):
        _render(setup, out, stream_threshold_px=1, stream_strip_rows=7,
                should_cancel=cancel, status=arm_status)
    assert out.exists(), "partial BigTIFF must be kept for resume"

    s2 = _render(setup, out, stream_threshold_px=1, stream_strip_rows=7)
    assert s2["strips"] == s_ref["strips"]
    a = tifffile.imread(str(ref))
    b = tifffile.imread(str(out))
    assert np.array_equal(a, b)


def test_streaming_preview_generated(setup):
    _, _, tmp = setup
    out = tmp / "prev.tif"
    s = _render(setup, out, stream_threshold_px=1, stream_strip_rows=7)
    assert s["preview"]
    with Image.open(s["preview"]) as p:
        assert p.size[0] <= 4096
        assert p.size == (128, 128)      # 小图时预览即原尺寸


def test_streaming_png_keeps_format_and_matches(setup):
    """流式 PNG：保留 .png 扩展名（不再强改 .tif），与整图路径逐像素一致。"""
    _, _, tmp = setup
    ref = tmp / "ref.png"
    out = tmp / "stream.png"
    s1 = _render(setup, ref, stream_threshold_px=10 ** 12)
    s2 = _render(setup, out, stream_threshold_px=1, stream_strip_rows=7)
    assert str(s2["output"]).endswith(".png"), "流式 PNG 不应被改写成 TIFF"
    assert s2["streamed"] is True
    assert s2["strips"] > 1
    a = np.asarray(Image.open(ref).convert("RGB"))
    b = np.asarray(Image.open(out).convert("RGB"))
    assert np.array_equal(a, b)


def test_streaming_jpeg_reroutes_to_png(setup):
    """流式不支持 JPEG（格式上限 65535px + Pillow 只能整图编码）→ 自动改 PNG。"""
    _, _, tmp = setup
    out = tmp / "photo.jpg"
    s = _render(setup, out, stream_threshold_px=1, stream_strip_rows=7)
    assert str(s["output"]).endswith(".png")
    assert os.path.exists(s["output"])
    with Image.open(s["output"]) as img:
        assert img.size == (128, 128)


def test_streaming_png_no_resume_but_clean_rerun(setup):
    """PNG 流式不支持续传（zlib 流不可定位）：取消后重跑从头来，产物完整。"""
    _, _, tmp = setup
    ref = tmp / "ref2.png"
    _render(setup, ref, stream_threshold_px=1, stream_strip_rows=7)

    out = tmp / "res.png"
    state = {"armed": False, "seen": 0}

    def cancel():
        if not state["armed"]:
            return False
        state["seen"] += 1
        return state["seen"] > 2

    def arm(msg):
        if "流式渲染" in msg:
            state["armed"] = True

    with pytest.raises(RenderCancelled):
        _render(setup, out, stream_threshold_px=1, stream_strip_rows=7,
                should_cancel=cancel, status=arm)
    # 重跑（无取消）→ 完整有效 PNG，与一次性渲染逐像素一致
    s2 = _render(setup, out, stream_threshold_px=1, stream_strip_rows=7)
    assert s2["strips"] > 1
    a = np.asarray(Image.open(ref).convert("RGB"))
    b = np.asarray(Image.open(out).convert("RGB"))
    assert np.array_equal(a, b)


def test_streaming_uniform_matches_normal_exactly(setup):
    """A1：uniform（levels=1）超大输出也走流式路径，且与整图逐像素一致。"""
    master, db, tmp = setup
    out_png = tmp / "u_normal.png"
    out_tif = tmp / "u_stream.tif"
    common = dict(cells_x=8, top_k=3, seed=1)      # 不传 levels → uniform 分支
    s1 = make_mosaic(master, db, str(out_png), stream_threshold_px=10 ** 12, **common)
    s2 = make_mosaic(master, db, str(out_tif), stream_threshold_px=1, **common)
    assert s1["mode"] == "uniform" and s1.get("streamed") is None
    assert s2["mode"] == "uniform" and s2["streamed"] is True
    assert s2["strips"] >= 1
    assert s2["size"] == s1["size"] == (512, 512)
    assert list(s1["choices"]) == list(s2["choices"])
    assert s2["preview"] and os.path.exists(s2["preview"])
    a = np.asarray(Image.open(out_png).convert("RGB"))
    b = tifffile.imread(str(out_tif))
    assert np.array_equal(a, b)


def test_streaming_multichunk_decode_matches(setup, monkeypatch):
    """A2：条带内分块解码（极小块预算 + 极低 decoded 上限）仍逐像素一致。

    把块预算压到 1 字节迫使每叶子一块、decoded 上限压到 1 字节迫使每块
    清空缓存——同时破坏"一次解码整条带"与"跨块复用"两条捷径后，结果
    依然必须与整图路径完全相同（解码尺寸始终是 global_need）。
    """
    from photomosaic import streaming

    _, _, tmp = setup
    out_png = tmp / "n3.png"
    out_tif = tmp / "s3.tif"
    _render(setup, out_png, stream_threshold_px=10 ** 12)
    monkeypatch.setattr(streaming, "_CHUNK_BUDGET_BYTES", 1)
    monkeypatch.setattr(streaming, "_CHUNK_BUDGET_JOBS", 1)
    monkeypatch.setattr(streaming, "_DECODE_CAP_BYTES", 1)
    s2 = _render(setup, out_tif, stream_threshold_px=1, stream_strip_rows=7)
    assert s2["strips"] > 1
    a = np.asarray(Image.open(out_png).convert("RGB"))
    b = tifffile.imread(str(out_tif))
    assert np.array_equal(a, b)


def test_streaming_disk_space_check(setup, monkeypatch, tmp_path):
    """A4：流式渲染前磁盘空间不足 → ValueError（消息含 GB）；非流式不检查。"""
    class _Usage:
        def __init__(self, free):
            self.total = self.used = 0
            self.free = free

    monkeypatch.setattr("photomosaic.render.shutil.disk_usage",
                        lambda p: _Usage(1024))
    with pytest.raises(ValueError, match="GB"):
        _render(setup, tmp_path / "disk.tif", stream_threshold_px=1)
    # 非流式路径不做磁盘预检，照常渲染成功
    _render(setup, tmp_path / "disk_ok.png", stream_threshold_px=10 ** 12)
    assert (tmp_path / "disk_ok.png").exists()
