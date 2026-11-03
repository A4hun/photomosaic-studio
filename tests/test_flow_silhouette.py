"""流体照片河 + 负空间剪影：两种新渲染路径的回归。"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("news")
    folder = tmp / "tiles"
    folder.mkdir()
    for i in range(10):
        a = np.zeros((64, 64, 3), np.uint8)
        a[:, :] = rng.integers(30, 220, 3)
        Image.fromarray(a).save(folder / f"t{i}.png")
    # 有主体的主图：亮背景 + 中间暗圆（剪影的靶子，也给流向场结构）
    yy, xx = np.ogrid[:300, :400]
    m = np.full((300, 400, 3), 210, np.uint8)
    m[((yy - 150) ** 2 + (xx - 200) ** 2) < 80 ** 2] = 40
    master = tmp / "m.png"
    Image.fromarray(m).save(master)
    db = TileDB()
    ingest(db, str(folder))
    yield str(master), db, tmp
    db.close()


def test_flow_renders_and_is_deterministic(setup):
    master, db, tmp = setup
    outs = []
    for k in range(2):
        out = tmp / f"flow{k}.png"
        s = make_mosaic(master, db, str(out), cells_x=30, seed=7, style="flow")
        assert s["mode"] == "flow"
        assert s["leaves"] > 50, "流体应放置足够多的瓦片"
        assert os.path.exists(out)
        outs.append(np.asarray(Image.open(out).convert("RGB")))
    assert np.array_equal(outs[0], outs[1]), "同种子流体应逐字节确定"


def test_flow_tiles_follow_structure(setup):
    """竖条主图（强竖向边缘）：瓦片角度应显著偏向竖向（±90° 一带）。"""
    import math
    from photomosaic.flow import flow_placements, orientation_field
    master, db, tmp = setup
    # 竖条纹主图
    v = np.zeros((200, 200, 3), np.uint8)
    v[:, ::16] = 255
    im = Image.fromarray(v)
    theta, fs = orientation_field(im)
    pl = flow_placements(theta, fs, 200, 200, 16, 10, 1)
    assert len(pl) > 10
    # 竖条纹的"沿边方向"是竖向：角度应聚在 90° 附近（或 -90→+90 等价）
    angs = [abs(math.degrees(a)) % 180 for _x, _y, a in pl]
    near90 = sum(1 for a in angs if 60 <= a <= 120)
    assert near90 > len(angs) * 0.5, f"竖向边缘应让过半瓦片竖向（{near90}/{len(angs)}）"


def test_silhouette_keeps_background_white(setup):
    master, db, tmp = setup
    out = tmp / "sil.png"
    s = make_mosaic(master, db, str(out), cells_x=24, levels=1, seed=1,
                    style="silhouette", color_grade=False)
    assert s["mode"] == "adaptive"
    img = np.asarray(Image.open(out).convert("RGB"))
    white = float(np.all(np.abs(img.astype(int) - 255) < 6, axis=2).mean())
    assert white > 0.25, f"剪影应大面积留白（白底 {white:.0%}）"
    assert white < 0.9, "剪影主体区域应有瓦片"
