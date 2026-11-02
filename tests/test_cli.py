"""CLI-level tests for the make subcommand's blend / out-width / poster flags."""

import numpy as np
import pytest
from PIL import Image

from photomosaic.cli import build_parser, main
from photomosaic.db import TileDB
from photomosaic.library import ingest


@pytest.fixture()
def tiny_setup(tmp_path):
    rng = np.random.default_rng(3)
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(8):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            folder / f"t{i}.png")
    master_path = tmp_path / "master.png"
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(master_path)
    db_path = tmp_path / "tiles.db"
    with TileDB(str(db_path)) as db:
        ingest(db, str(folder))
    return master_path, db_path, tmp_path


def test_defaults_match_original_spec():
    args = build_parser().parse_args(
        ["make", "m.png", "--db", "x.db", "-o", "out.png"])
    assert args.blend == pytest.approx(0.55)   # 原版 139/255 柔光强度
    assert args.out_width is None
    assert args.poster is None
    assert args.quality == 100


def test_stream_flags_default_none_and_parse():
    """B4 回归：流式参数默认 None（= 用库默认 2 亿像素），显式给值可解析。"""
    args = build_parser().parse_args(
        ["make", "m.png", "--db", "x.db", "-o", "out.png"])
    assert args.stream_threshold is None
    assert args.stream_strip_rows is None
    args = build_parser().parse_args(
        ["run", "m.png", "folder", "-o", "out.png",
         "--stream-threshold", "1000", "--stream-strip-rows", "7"])
    assert args.stream_threshold == 1000
    assert args.stream_strip_rows == 7


def test_make_stream_flags_force_streaming(tiny_setup):
    """B4 回归：--stream-threshold/--stream-strip-rows 透传到 make_mosaic。"""
    master, db_path, tmp = tiny_setup
    out = tmp / "stream.tif"
    rc = main(["make", str(master), "--db", str(db_path), "-o", str(out),
               "--cells-x", "8", "--levels", "2", "--seed", "3",
               "--stream-threshold", "1", "--stream-strip-rows", "32"])
    assert rc == 0
    assert out.exists()                             # 强制流式 → BigTIFF 输出
    assert (tmp / "stream.preview.png").exists()    # 流式路径必产降采样预览


def test_make_blend_zero_vs_default_differ(tiny_setup):
    master, db_path, tmp = tiny_setup
    out_a = tmp / "a.png"
    out_b = tmp / "b.png"
    base = ["make", str(master), "--db", str(db_path),
            "--cells-x", "8", "--seed", "7", "--top-k", "1"]
    assert main(base + ["-o", str(out_a), "--blend", "0"]) == 0
    assert main(base + ["-o", str(out_b)]) == 0  # default blend 0.55
    a = np.asarray(Image.open(out_a).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(out_b).convert("RGB"), dtype=np.int16)
    assert a.shape == b.shape
    assert np.abs(a - b).sum() > 0  # colorization changed pixels


def test_make_out_width_scales_canvas(tiny_setup):
    master, db_path, tmp = tiny_setup
    out = tmp / "wide.png"
    rc = main(["make", str(master), "--db", str(db_path), "-o", str(out),
               "--cells-x", "8", "--levels", "2", "--seed", "1",
               "--out-width", "256"])
    assert rc == 0
    with Image.open(out) as img:
        assert img.size == (256, 256)  # master 128x128 -> x2


def test_make_poster_pages(tiny_setup):
    master, db_path, tmp = tiny_setup
    out = tmp / "mosaic.png"
    rc = main(["make", str(master), "--db", str(db_path), "-o", str(out),
               "--cells-x", "8", "--seed", "2", "--poster", "2x3"])
    assert rc == 0
    pages = sorted((tmp / "mosaic_poster").glob("mosaic_r*c*.png"))
    assert len(pages) == 6  # 2 cols x 3 rows
    with Image.open(out) as whole, Image.open(pages[0]) as page:
        assert page.size[0] == whole.size[0] // 2
