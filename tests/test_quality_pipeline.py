"""质量管线回归：迭代精炼 + 成品全局校色。"""

import numpy as np
import pytest
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("qp")
    folder = tmp / "tiles"
    folder.mkdir()
    # 瓦片色调偏蓝（系统性偏移——校色的靶子）
    for i in range(12):
        a = np.zeros((64, 64, 3), np.uint8)
        a[..., 0] = rng.integers(30, 200)
        a[..., 1] = rng.integers(30, 200)
        a[..., 2] = np.clip(a[..., 0] + rng.integers(30, 90), 0, 255)
        Image.fromarray(a).save(folder / f"t{i}.png")
    master = tmp / "m.png"
    m = np.zeros((200, 300, 3), np.uint8)
    m[:, :100] = 210
    m[:, 100:] = 40
    Image.fromarray(m).save(master)
    db = TileDB()
    ingest(db, str(folder))
    yield str(master), db, tmp
    db.close()


def test_refine_passes_lower_distance(setup):
    master, db, tmp = setup
    s0 = make_mosaic(master, db, str(tmp / "r0.png"), cells_x=10, levels=1,
                     seed=1, color_grade=False, refine_passes=0)
    s3 = make_mosaic(master, db, str(tmp / "r3.png"), cells_x=10, levels=1,
                     seed=1, color_grade=False, refine_passes=3)
    assert s3["mean_distance"] <= s0["mean_distance"], \
        f"精炼后均距应不升：{s0['mean_distance']:.0f} -> {s3['mean_distance']:.0f}"


def test_color_grade_pulls_toward_master_tone(setup):
    """库整体偏蓝 → 校色后成品的蓝通道均值应更接近主图（降）。"""
    master, db, tmp = setup
    out0, out1 = tmp / "g0.png", tmp / "g1.png"
    make_mosaic(master, db, str(out0), cells_x=10, levels=1, seed=1,
                blend=0.0, color_grade=False)
    make_mosaic(master, db, str(out1), cells_x=10, levels=1, seed=1,
                blend=0.0, color_grade=True)
    m_b = float(np.asarray(Image.open(master).convert("RGB"), dtype=float)[..., 2].mean())
    b0 = float(np.asarray(Image.open(out0).convert("RGB"), dtype=float)[..., 2].mean())
    b1 = float(np.asarray(Image.open(out1).convert("RGB"), dtype=float)[..., 2].mean())
    assert abs(b1 - m_b) < abs(b0 - m_b), \
        f"校色应让蓝通道更贴主图：库{b0:.0f} 校后{b1:.0f} 主图{m_b:.0f}"


def test_grade_consistent_streaming_vs_whole(setup):
    """校色在流式/整图两路用同一 LUT：逐像素一致（不变量不被校色破坏）。"""
    master, db, tmp = setup
    common = dict(cells_x=8, levels=1, seed=3, color_grade=True, refine_passes=2)
    make_mosaic(master, db, str(tmp / "w.png"), **common)
    make_mosaic(master, db, str(tmp / "s.png"), stream_threshold_px=1,
                stream_strip_rows=7, **common)
    a = np.asarray(Image.open(tmp / "w.png").convert("RGB"))
    b = np.asarray(Image.open(tmp / "s.png").convert("RGB"))
    assert np.array_equal(a, b)
