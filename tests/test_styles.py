"""风格渲染：砖墙偏移、蜂窝/点彩蒙版与缝隙背景、与整图流式一致。"""

import numpy as np
import pytest
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic
from photomosaic.styles import get_style, all_styles


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("styles")
    folder = tmp / "tiles"
    folder.mkdir()
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            folder / f"t{i}.png")
    master = tmp / "m.png"
    Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(master)
    db = TileDB()
    ingest(db, str(folder))
    yield str(master), db, tmp
    db.close()


def test_styles_registered():
    names = {n for n, _l in all_styles()}
    assert {"grid", "brick", "hex", "dot"} <= names
    assert get_style("brick").name == "brick"
    assert get_style("不存在").name == "grid"      # 未知回退 grid


def test_brick_offsets_even_rows(setup):
    sty = get_style("brick")
    assert sty.offset(0, 0, 100, 100) == (0, 0)
    assert sty.offset(1, 0, 100, 100) == (50, 0)   # 偶数行右移半格
    assert sty.offset(2, 0, 100, 100) == (0, 0)
    assert not sty.has_gaps()


def test_hex_and_dot_masks(setup):
    for name in ("hex", "dot"):
        sty = get_style(name)
        m = sty.mask((40, 40))
        assert m is not None and m.mode == "L"
        a = np.asarray(m)
        assert a.max() == 255 and a.min() == 0     # 中间实、边缘有镂空
        assert a[0, 0] == 0                        # 角上透明
        assert a[20, 20] == 255                    # 中心实
        assert sty.has_gaps()


def test_style_produces_different_but_valid_output(setup):
    master, db, tmp = setup
    outs = {}
    for style in ("grid", "brick", "hex", "dot"):
        out = tmp / f"{style}.png"
        s = make_mosaic(master, db, str(out), cells_x=8, top_k=2, seed=1,
                        style=style)
        assert s["size"][0] > 0
        arr = np.asarray(Image.open(out).convert("RGB"))
        outs[style] = hash(arr.tobytes())
    # 四种风格产出互不相同
    assert len(set(outs.values())) == 4


def test_gapped_style_fills_background(setup):
    """蜂窝/点彩：角上缝隙处应是模糊主图底色（非纯白）。"""
    master, db, tmp = setup
    out = tmp / "hex.png"
    make_mosaic(master, db, str(out), cells_x=8, top_k=2, seed=1, style="hex")
    img = np.asarray(Image.open(out).convert("RGB"))
    # 角落像素（六边形蒙版镂空处）不应是纯白
    assert not (img[0, 0] == 255).all(), "缝隙处应有背景色而非纯白"
