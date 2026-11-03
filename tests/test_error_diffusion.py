"""误差扩散：开启后产出不同且平均匹配误差下降（远观更贴主图）。"""

import numpy as np
import pytest
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("ed")
    folder = tmp / "tiles"
    folder.mkdir()
    for i in range(12):
        a = np.zeros((64, 64, 3), np.uint8)
        a[:, :] = rng.integers(30, 220, 3)
        Image.fromarray(a).save(folder / f"t{i}.png")
    master = tmp / "m.png"
    # 左亮右暗的主图（误差扩散的舞台）
    m = np.zeros((200, 300, 3), np.uint8)
    m[:, :100] = 210
    m[:, 100:] = 40
    Image.fromarray(m).save(master)
    db = TileDB()
    ingest(db, str(folder))
    yield str(master), db, tmp
    db.close()


def test_error_diffusion_changes_output_and_lowers_distance(setup):
    master, db, tmp = setup
    out_off = tmp / "off.png"
    out_on = tmp / "on.png"
    s_off = make_mosaic(master, db, str(out_off), cells_x=10, levels=1, seed=1,
                        error_diffusion=False)
    s_on = make_mosaic(master, db, str(out_on), cells_x=10, levels=1, seed=1,
                       error_diffusion=True)
    a = np.asarray(Image.open(out_off).convert("RGB"))
    b = np.asarray(Image.open(out_on).convert("RGB"))
    assert not np.array_equal(a, b), "误差扩散应改变产出"
    # 误差扩散让整体远观更贴主图：均距下降
    assert s_on["mean_distance"] < s_off["mean_distance"], \
        f"误差扩散后均距应下降：{s_off['mean_distance']:.0f} -> {s_on['mean_distance']:.0f}"


def test_error_diffusion_deterministic(setup):
    master, db, tmp = setup
    outs = []
    for _ in range(2):
        out = tmp / f"d{len(outs)}.png"
        make_mosaic(master, db, str(out), cells_x=10, levels=1, seed=5,
                    error_diffusion=True)
        outs.append(np.asarray(Image.open(out).convert("RGB")))
    assert np.array_equal(outs[0], outs[1]), "同种子应逐字节确定"
