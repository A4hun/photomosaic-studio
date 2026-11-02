import numpy as np
import pytest
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import (adaptive_leaves, detail_energy, make_mosaic,
                                 uniform_base_boxes)


def _half_noise_image(size=256, seed=1):
    rng = np.random.default_rng(seed)
    arr = np.full((size, size, 3), 40, dtype=np.uint8)          # flat bottom
    arr[: size // 2] = rng.integers(0, 255, (size // 2, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_detail_energy_separates_flat_from_noisy():
    img = _half_noise_image()
    flat = detail_energy(img, (0, 128, 256, 256))
    noisy = detail_energy(img, (0, 0, 256, 128))
    assert flat < 2.0
    assert noisy > 15.0


def test_adaptive_leaves_subdivide_only_detail():
    img = _half_noise_image()
    leaves = adaptive_leaves(img, uniform_base_boxes(256, 256, 8, 8), 3, 15.0)
    levels = [leaf[4] for leaf in leaves]
    assert max(levels) >= 2                       # noisy half got subdivided
    bottom = [leaf for leaf in leaves if leaf[1] >= 128]
    assert bottom and all(leaf[4] == 1 for leaf in bottom)  # flat half stays coarse

    # exact tiling: leaves cover the image once, without gaps or overlaps
    W, H = img.size
    cover = np.zeros((H, W), dtype=np.int8)
    for x0, y0, x1, y1, _lvl in leaves:
        cover[y0:y1, x0:x1] += 1
    assert np.all(cover == 1)


def test_flat_image_never_splits():
    img = Image.new("RGB", (256, 256), (10, 200, 90))
    leaves = adaptive_leaves(img, uniform_base_boxes(256, 256, 8, 8), 4, 1.0)
    assert len(leaves) == 64
    assert all(leaf[4] == 1 for leaf in leaves)


def _coverage(leaves, w, h):
    cover = np.zeros((h, w), dtype=np.int8)
    for x0, y0, x1, y1, _lvl in leaves:
        cover[y0:y1, x0:x1] += 1
    return cover


def test_forced_region_splits_flat_image():
    img = Image.new("RGB", (256, 256), (10, 200, 90))  # perfectly flat
    leaves = adaptive_leaves(img, uniform_base_boxes(256, 256, 4, 4), 1, 999.0,
                             regions=[(64, 64, 128, 128, 3)])
    inside = [l for l in leaves if l[0] >= 64 and l[2] <= 128 and l[1] >= 64 and l[3] <= 128]
    outside = [l for l in leaves if l not in inside]
    assert inside and all(l[4] == 3 for l in inside)      # forced depth reached
    assert outside and all(l[4] == 1 for l in outside)    # flat area untouched
    assert np.all(_coverage(leaves, 256, 256) == 1)


def test_region_dict_form_and_level_cap():
    img = Image.new("RGB", (128, 128), (200, 30, 30))
    leaves = adaptive_leaves(img, uniform_base_boxes(128, 128, 4, 4), 1, 999.0,
                             regions=[{"box": [0, 0, 64, 64], "level": 2}])
    inside = [l for l in leaves if l[2] <= 64 and l[3] <= 64]
    assert inside and all(l[4] == 2 for l in inside)
    assert np.all(_coverage(leaves, 128, 128) == 1)


def test_make_mosaic_regions_switch_to_adaptive_mode(tmp_path):
    rng = np.random.default_rng(5)
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(6):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            folder / f"t{i}.png")
    master = Image.new("RGB", (128, 128), (40, 40, 120))
    master_path = tmp_path / "m.png"
    master.save(master_path)
    out = tmp_path / "reg.png"
    with TileDB() as db:
        ingest(db, str(folder))
        stats = make_mosaic(str(master_path), db, str(out), cells_x=4,
                            levels=1, regions=[(0, 0, 64, 64, 2)], top_k=2, seed=1)
    assert stats["mode"] == "adaptive"   # regions alone enable adaptive canvas
    assert stats["leaves"] > 16
    assert Image.open(out).size == (128, 128)


def test_detail_map_separates_and_vectorises():
    from photomosaic.render import DetailMap

    img = _half_noise_image()
    dm = DetailMap(img)
    flat = dm.energy((0, 128, 256, 256))
    noisy = dm.energy((0, 0, 256, 128))
    assert flat < 2.0 < 20.0 < noisy          # native-res gradient separates cleanly
    boxes = np.array([[0, 0, 256, 128], [0, 128, 256, 256], [64, 32, 192, 160]])
    vec = dm.energies(boxes)
    for i, b in enumerate(boxes):
        assert vec[i] == pytest.approx(dm.energy(tuple(b)), rel=1e-6)


def test_adaptive_leaves_with_detail_map():
    from photomosaic.render import DetailMap

    img = _half_noise_image()
    dm = DetailMap(img)
    base = uniform_base_boxes(256, 256, 8, 8)
    leaves = adaptive_leaves(img, base, 3, 15.0, detail=dm)
    levels = [leaf[4] for leaf in leaves]
    assert max(levels) >= 2
    bottom = [leaf for leaf in leaves if leaf[1] >= 128]
    assert bottom and all(leaf[4] == 1 for leaf in bottom)


def test_cli_region_parsing():
    from photomosaic.cli import _parse_region

    assert _parse_region("10,20,30,40@3") == (10, 20, 30, 40, 3)
    assert _parse_region("30,40,10,20") == (10, 20, 30, 40, 2)   # normalised + default level
    with pytest.raises(Exception):
        _parse_region("1,2,3")
    with pytest.raises(Exception):
        _parse_region("1,2,3,4@x")


def test_make_mosaic_adaptive_end_to_end(tmp_path):
    rng = np.random.default_rng(5)
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(10):
        arr = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"t{i}.png")
    master = _half_noise_image(256)
    master_path = tmp_path / "master.png"
    master.save(master_path)
    out = tmp_path / "adaptive.png"

    with TileDB() as db:
        ingest(db, str(folder))
        stats = make_mosaic(str(master_path), db, str(out), cells_x=8,
                            levels=3, detail_threshold=15.0, top_k=3, seed=1)
    assert stats["mode"] == "adaptive"
    assert stats["leaves"] > 64
    assert stats["level_histogram"][1] > 0 and sum(stats["level_histogram"].values()) == stats["leaves"]
    img = Image.open(out)
    assert img.size == (256, 256)  # adaptive canvas = master size


def test_manual_leaf_boxes_blend_and_poster(tmp_path):
    from photomosaic.render import export_poster

    rng = np.random.default_rng(9)
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(8):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            folder / f"t{i}.png")
    master = Image.new("RGB", (128, 128), (200, 30, 30))
    master_path = tmp_path / "m.png"
    master.save(master_path)

    boxes = [((0, 0, 64, 64), 1), ((64, 0, 128, 64), 1),
             ((0, 64, 64, 128), 1), ((64, 64, 96, 96), 2),
             ((96, 64, 128, 96), 2), ((64, 96, 128, 128), 2)]
    out = tmp_path / "manual.png"
    with TileDB() as db:
        ingest(db, str(folder))
        stats = make_mosaic(str(master_path), db, str(out), cells_x=2,
                            leaf_boxes=boxes, blend=0.35, top_k=2, seed=3)
    assert stats["mode"] == "manual"
    assert stats["leaves"] == 6
    assert len(stats["leaf_rects"]) == 6 and len(stats["choices"]) == 6
    img = Image.open(out)
    assert img.size == (128, 128)
    # blend pulls pixels toward the cell average colour (solid red master)
    px = np.asarray(img, dtype=np.int16)
    assert px[..., 0].mean() > 120  # red channel dominant after blending

    pages = export_poster(img, 2, 3, str(tmp_path / "poster"), "p")
    assert len(pages) == 6
    assert Image.open(pages[0]).size == (64, 43) or Image.open(pages[0]).size == (64, 42)


def test_level5_reachable_on_small_base():
    """Relative min_px: level 5 must be reachable even with 16px base cells."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    leaves = adaptive_leaves(img, uniform_base_boxes(64, 64, 4, 4),
                             5, 0.0)   # threshold 0 => split everywhere
    levels = [l[4] for l in leaves]
    assert max(levels) == 5


def test_out_width_scales_canvas_and_rects(tmp_path):
    rng = np.random.default_rng(5)
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(6):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            folder / f"t{i}.png")
    master = Image.new("RGB", (128, 96), (40, 40, 120))
    master_path = tmp_path / "m.png"
    master.save(master_path)
    out = tmp_path / "big.png"
    with TileDB() as db:
        ingest(db, str(folder))
        stats = make_mosaic(str(master_path), db, str(out), cells_x=4,
                            levels=2, out_width=512, top_k=2, seed=1)
    assert stats["size"] == (512, 384)   # 128x96 scaled to width 512
    img = Image.open(out)
    assert img.size == (512, 384)
    for (x0, y0, x1, y1, _lvl) in stats["leaf_rects"]:
        assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 384
