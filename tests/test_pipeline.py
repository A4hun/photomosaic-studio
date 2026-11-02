import numpy as np
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic, master_cell_features


def _make_tile_folder(tmp_path, n=12, size=96):
    rng = np.random.default_rng(42)
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(n):
        arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        # add a solid-colour band so tiles are distinguishable in features
        arr[:, :, i % 3] = (i * 17) % 256
        Image.fromarray(arr).save(folder / f"tile_{i:02d}.png")
    return folder


def _make_master(tmp_path, size=(256, 192)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((xx / w) * 255).astype(np.uint8)
    g = ((yy / h) * 255).astype(np.uint8)
    b = np.full_like(r, 120)
    img = Image.fromarray(np.dstack([r, g, b]))
    path = tmp_path / "master.png"
    img.save(path)
    return path


def test_master_cell_features_shape_and_consistency(tmp_path):
    master = Image.open(_make_master(tmp_path))
    feats, avgs = master_cell_features(master, 8, 6)
    assert feats.shape == (6, 8, 27)
    assert avgs.shape == (6, 8, 3)
    # solid image -> all cells identical
    solid = Image.new("RGB", (200, 200), (10, 200, 90))
    f2, a2 = master_cell_features(solid, 5, 5)
    assert np.allclose(f2, f2[0, 0])
    assert np.allclose(a2, [10, 200, 90], atol=1)


def test_ingest_dedup_and_variants(tmp_path):
    folder = _make_tile_folder(tmp_path, n=6)
    # duplicate one file under a different name
    src = folder / "tile_00.png"
    src.with_name("tile_copy.png").write_bytes(src.read_bytes())

    with TileDB() as db:
        stats = ingest(db, str(folder), variants=("orig", "mirror", "r90"))
        assert stats["files"] == 7
        assert stats["duplicates"] == 1
        assert stats["added"] == 6
        assert stats["rows"] == 18
        assert db.count() == 18
        # re-ingest is idempotent thanks to md5 dedup
        stats2 = ingest(db, str(folder), variants=("orig", "mirror", "r90"))
        assert stats2["added"] == 0
        assert db.count() == 18


def test_end_to_end(tmp_path):
    folder = _make_tile_folder(tmp_path, n=12)
    master = _make_master(tmp_path)
    out = tmp_path / "mosaic.png"
    with TileDB() as db:
        ingest(db, str(folder), variants=("orig", "mirror"))
        assert db.count() == 24
        stats = make_mosaic(str(master), db, str(out), cells_x=16, cell_px=32,
                            top_k=4, seed=123)
    assert out.exists()
    assert stats["cells_x"] == 16
    assert stats["cells_y"] == 12  # 192 / (256/16)
    img = Image.open(out)
    assert img.size == (16 * 32, 12 * 32)


def test_ingest_parallel_matches_serial(tmp_path):
    folder = _make_tile_folder(tmp_path, n=10)
    with TileDB() as db1:
        ingest(db1, str(folder), workers=1)
        a = db1.load_arrays()
        n1 = db1.count()
    with TileDB() as db2:
        ingest(db2, str(folder), workers=4)
        b = db2.load_arrays()
        n2 = db2.count()
    assert n1 == n2 == 10
    fa = sorted(f.tobytes() for f in a["feats"])
    fb = sorted(f.tobytes() for f in b["feats"])
    assert fa == fb  # same features regardless of worker count


def test_deterministic_with_seed(tmp_path):
    folder = _make_tile_folder(tmp_path, n=8)
    master = _make_master(tmp_path)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for out in (a, b):
        with TileDB() as db:
            ingest(db, str(folder))
            make_mosaic(str(master), db, str(out), cells_x=8, cell_px=24, seed=99)
    assert a.read_bytes() == b.read_bytes()
