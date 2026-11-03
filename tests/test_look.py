"""Original-look alignment tests: soft-light colorization, near-white/black
tile exclusion, first-level conservatism, GUI defaults."""

import numpy as np
import pytest
from PIL import Image, ImageChops

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic


def _solid(path, color, size=(64, 64)):
    Image.new("RGB", size, color).save(path)
    return path


def _tiny_master(tmp_path, color, size=(64, 64)):
    p = tmp_path / "m.png"
    Image.new("RGB", size, color).save(p)
    return str(p)


def _tiny_db(tmp_path, color=(100, 150, 200)):
    folder = tmp_path / "tiles"
    folder.mkdir(exist_ok=True)
    _solid(folder / "t.png", color)
    return folder


def test_softlight_formula_pixel_exact(tmp_path):
    """blend=1.0 output pixel == ImageChops.soft_light(tile, cell_avg)."""
    master = _tiny_master(tmp_path, color=(50, 200, 80))
    folder = _tiny_db(tmp_path, color=(100, 150, 200))
    out = tmp_path / "out.png"
    with TileDB() as db:
        ingest(db, str(folder), workers=1)
        make_mosaic(master, db, str(out), cells_x=1, cell_px=8, levels=1,
                    blend=1.0, top_k=1, seed=1, color_grade=False)
    got = np.asarray(Image.open(out).convert("RGB"))
    tile = Image.new("RGB", (8, 8), (100, 150, 200))
    solid = Image.new("RGB", (8, 8), (50, 200, 80))
    expect = np.asarray(ImageChops.soft_light(tile, solid))
    # JPEG thumb (q=100) is near-lossless: allow small tolerance
    assert np.abs(got.astype(int) - expect.astype(int)).max() <= 4


def test_softlight_differs_from_naive_blend_and_is_deterministic(tmp_path):
    master = _tiny_master(tmp_path, color=(50, 200, 80))
    folder = _tiny_db(tmp_path, color=(100, 150, 200))
    out0, out1, out2 = (tmp_path / n for n in ("b0.png", "b1.png", "b2.png"))
    with TileDB() as db:
        ingest(db, str(folder), workers=1)
        make_mosaic(master, db, str(out0), cells_x=1, cell_px=8, blend=0.0,
                    top_k=1, seed=1)
        make_mosaic(master, db, str(out1), cells_x=1, cell_px=8, blend=0.55,
                    top_k=1, seed=1)
        make_mosaic(master, db, str(out2), cells_x=1, cell_px=8, blend=0.55,
                    top_k=1, seed=1)
    a0 = np.asarray(Image.open(out0).convert("RGB"))
    a1 = np.asarray(Image.open(out1).convert("RGB"))
    a2 = np.asarray(Image.open(out2).convert("RGB"))
    assert np.abs(a0.astype(int) - a1.astype(int)).max() > 10  # soft-light path applied
    assert np.array_equal(a1, a2)                              # deterministic
    naive = Image.blend(Image.new("RGB", (8, 8), (100, 150, 200)),
                        Image.new("RGB", (8, 8), (50, 200, 80)), 0.55)
    assert not np.array_equal(a1, np.asarray(naive))  # not the naive formula


def test_skip_near_white_black(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    _solid(folder / "white.png", (250, 250, 250))
    _solid(folder / "black.png", (5, 5, 5))
    _solid(folder / "normal.png", (120, 130, 140))
    with TileDB() as db:
        stats = ingest(db, str(folder), workers=1,
                       skip_near_white=245, skip_near_black=10)
        assert db.count() == 1
    assert stats["skipped_white"] == 1
    assert stats["skipped_black"] == 1
    assert stats["added"] == 1


def test_first_level_factor_conservatism():
    from photomosaic.render import DetailMap, adaptive_leaves, uniform_base_boxes

    rng = np.random.default_rng(5)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, ::4] = 30                      # gentle stripes: moderate energy
    img = Image.fromarray(arr)
    dm = DetailMap(img)
    base = uniform_base_boxes(256, 256, 4, 4)
    e0 = dm.energy((0, 0, 64, 64))        # actual base-cell energy
    threshold = e0 * 0.6                  # factor 1 splits, factor 3 doesn't
    n1 = len(adaptive_leaves(img, base, 2, threshold, detail=dm,
                             first_level_factor=1.0))
    n3 = len(adaptive_leaves(img, base, 2, threshold, detail=dm,
                             first_level_factor=3.0))
    assert n1 > n3  # higher first-level factor -> fewer first-level splits


def test_gui_original_look_defaults():
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from photomosaic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_settings=False)
    assert win.blend_slider.value() == 55
    assert win.blend_slider.maximum() == 100
    assert win.level_radios[3].isChecked()          # 四级 default
    assert win.cell_quality_spin.value() == 100
    assert win.out_quality.value() == 100
    assert hasattr(win, "skip_white_check") and hasattr(win, "skip_black_check")
    assert hasattr(win, "first_level_check")
    assert not win.skip_white_check.isChecked()
    win.close()
