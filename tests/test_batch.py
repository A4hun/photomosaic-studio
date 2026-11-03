"""智能适配（tuning）与批量矩阵（batch）。"""

import os

import numpy as np
import pytest
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.tuning import analyze_master, recommend_params


def _photo(seed=0, w=400, h=300):
    return Image.fromarray(np.random.default_rng(0).integers(
        0, 255, (h, w, 3), dtype=np.uint8))


def _flat():
    return Image.new("RGB", (400, 300), (200, 200, 200))


def test_analyze_master_features():
    f = analyze_master(_photo())
    for k in ("width", "height", "aspect", "detail_mean", "detail_p90",
              "subject_ratio", "color_entropy", "bright_ratio", "dark_ratio"):
        assert k in f, f"缺特征 {k}"
    assert f["width"] == 400 and f["height"] == 300


def test_recommend_differs_by_content():
    """平滑图与细节图应得到不同推荐（智能适配，非统一模板）。"""
    smooth = recommend_params(_flat())
    busy = recommend_params(_photo())
    # 色彩单调的平滑图 → 融合应更强；细节图 → 格子更多或层级更深
    assert smooth["blend"] >= busy["blend"]
    assert (busy["cells_x"] >= smooth["cells_x"]
            or busy["levels"] >= smooth["levels"])
    assert smooth["reasons"] and busy["reasons"]


def test_recommend_returns_make_mosaic_kwargs():
    rec = recommend_params(_photo())
    for k in ("cells_x", "levels", "detail_threshold", "top_k", "blend", "signal"):
        assert k in rec
    assert rec["signal"] in ("detail", "saliency")


@pytest.fixture(scope="module")
def batch_setup(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("batch")
    tiles = tmp / "tiles"
    tiles.mkdir()
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            tiles / f"t{i}.png")
    db = TileDB(str(tmp / "tiles.db"))
    ingest(db, str(tiles))
    db.close()
    masters = tmp / "masters"
    masters.mkdir()
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (200, 160, 3), dtype=np.uint8)).save(
            masters / f"m{i}.png")
    return str(tmp / "tiles.db"), str(masters), tmp


def test_batch_style_actually_applied_in_auto_adapt(batch_setup, tmp_path):
    """回归：智能适配模式下风格不得被吞掉（同种子 hex 与 grid 必须产出不同）。

    主图用有结构的内容（主体+平滑区），让细分适度、格子大到能看清蜂窝缝。
    """
    from photomosaic.batch import BatchRunner

    db_path, _masters_dir, tmp = batch_setup
    # 结构化主图：左亮右暗分界 + 一个主体圆（细分适度，格子不会碎成灰）
    a = np.zeros((200, 160, 3), np.uint8)
    a[:, :80] = 210
    a[:, 80:] = 40
    yy, xx = np.ogrid[:200, :160]
    a[((yy - 100) ** 2 + (xx - 80) ** 2) < 900] = (230, 90, 60)
    master = tmp_path / "structured.png"
    Image.fromarray(a).save(master)

    out_dir = str(tmp_path / "styled")
    with BatchRunner(db_path) as runner:
        runner.run([str(master)], out_dir, styles=["grid", "hex"], auto_adapt=True,
                   base_params={"levels": 3, "seed": 1})
    g = np.asarray(Image.open(os.path.join(out_dir, "structured_mosaic.png")).convert("RGB"))
    h = np.asarray(Image.open(os.path.join(out_dir, "structured_hex_mosaic.png")).convert("RGB"))
    assert g.shape == h.shape
    assert not np.array_equal(g, h), "hex 风格在智能适配下被吞掉了（与 grid 完全相同）"


def test_batch_matrix_skips_existing(batch_setup):
    from photomosaic.batch import BatchRunner, scan_masters

    db_path, masters_dir, tmp = batch_setup
    masters = scan_masters(masters_dir)
    assert len(masters) == 3
    out_dir = str(tmp / "out")
    with BatchRunner(db_path) as runner:
        stats = runner.run(masters, out_dir, styles=["grid", "hex"],
                           auto_adapt=False,
                           base_params={"levels": 2, "blend": 0.5})
    assert stats["total"] == 6            # 3 主图 × 2 风格
    assert stats["ok"] == 6
    assert stats["failed"] == 0
    # 重跑：输出已存在 → 全部跳过（断点续批）
    with BatchRunner(db_path) as runner:
        stats2 = runner.run(masters, out_dir, styles=["grid", "hex"],
                            auto_adapt=False,
                            base_params={"levels": 2, "blend": 0.5})
    assert stats2["skipped"] == 6 and stats2["ok"] == 0


def test_batch_override_style_reflected_in_filename(batch_setup, tmp_path):
    """回归：单独配置改了风格时，输出文件名后缀必须是实际渲染风格。

    旧实现按批量任务风格命名：m1 覆盖成 dot 却产出 m1_dot 内容、
    m1_hex 名字的文件（名实不符）。"""
    import os

    from photomosaic.batch import BatchRunner, scan_masters

    db_path, masters_dir, tmp = batch_setup
    m = scan_masters(masters_dir)[0]
    out_dir = str(tmp_path / "ovr")
    with BatchRunner(db_path) as runner:
        stats = runner.run([m], out_dir, styles=["hex"], auto_adapt=False,
                           base_params={"levels": 1, "seed": 1},
                           overrides={m: {"style": "dot"}})
    assert stats["ok"] == 1 and stats["failed"] == 0
    out = stats["per_image"][0]["output"]
    assert os.path.basename(out).endswith("_dot_mosaic.png"), out
    assert os.path.exists(out)
    # 断点续批一致性：重跑应按"实际风格名"跳过，而不是按旧名重渲
    with BatchRunner(db_path) as runner:
        stats2 = runner.run([m], out_dir, styles=["hex"], auto_adapt=False,
                            base_params={"levels": 1, "seed": 1},
                            overrides={m: {"style": "dot"}})
    assert stats2["skipped"] == 1 and stats2["ok"] == 0


def test_batch_single_failure_does_not_stop_others(batch_setup, tmp_path):
    from photomosaic.batch import BatchRunner, scan_masters

    db_path, masters_dir, tmp = batch_setup
    masters = scan_masters(masters_dir)
    bad = tmp_path / "broken.png"          # 不存在的文件
    masters = masters + [str(bad)]
    with BatchRunner(db_path) as runner:
        stats = runner.run(masters, str(tmp / "out2"), styles=["grid"],
                           auto_adapt=False, base_params={"levels": 1})
    assert stats["failed"] == 1            # 坏的那张失败
    assert stats["ok"] == 3                # 其余 3 张照常出图
    assert stats["errors"]
