import os

import numpy as np
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.grid import QuadTree
from photomosaic.library import IngestControl, ingest
from photomosaic.render import RenderCancelled, make_mosaic


def _make_folder(tmp_path, n=12, size=64, seed=4):
    rng = np.random.default_rng(seed)
    folder = tmp_path / "photos"
    folder.mkdir(parents=True)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(
            folder / f"p{i:02d}.jpg", "JPEG", quality=88)
    return folder


def test_ingest_cancel_then_resume(tmp_path):
    folder = _make_folder(tmp_path)
    db_path = str(tmp_path / "tiles.db")
    ctrl = IngestControl()

    def stop_after_4(done, _path):
        if done >= 4:
            ctrl.cancel()

    with TileDB(db_path) as db:
        s1 = ingest(db, str(folder), workers=2, control=ctrl, progress=stop_after_4)
        partial = db.count()
    assert s1["cancelled"] > 0
    assert 0 < partial < 12

    with TileDB(db_path) as db:
        s2 = ingest(db, str(folder), workers=2)
        assert db.count() == 12
    assert s2["added"] == 12 - partial

    # third run: everything indexed and present -> no hashing, no adds
    with TileDB(db_path) as db:
        s3 = ingest(db, str(folder), workers=2)
    assert s3["added"] == 0
    assert s3["index_hits"] == 12


def test_render_checkpoint_resume_matches_fresh(tmp_path):
    tiles_dir = _make_folder(tmp_path / "t", n=8, seed=7)
    rng = np.random.default_rng(21)
    master = Image.fromarray(rng.integers(0, 255, (160, 160, 3), dtype=np.uint8))
    master_path = tmp_path / "m.png"
    master.save(master_path)

    db_path = str(tmp_path / "tiles.db")
    with TileDB(db_path) as db:
        ingest(db, str(tiles_dir), workers=1)

    out_a = str(tmp_path / "a.png")
    ckpt_a = str(tmp_path / "a.ckpt")
    calls = {"n": 0}

    def cancel_after_30():
        calls["n"] += 1
        return calls["n"] > 30

    cancelled = False
    with TileDB(db_path) as db:
        try:
            make_mosaic(master_path, db, out_a, cells_x=8, levels=3,
                        detail_threshold=2.0, top_k=2, seed=5,
                        checkpoint=ckpt_a, should_cancel=cancel_after_30)
        except RenderCancelled:
            cancelled = True
    assert cancelled
    assert os.path.isdir(ckpt_a)

    with TileDB(db_path) as db:
        stats = make_mosaic(master_path, db, out_a, cells_x=8, levels=3,
                            detail_threshold=2.0, top_k=2, seed=5,
                            checkpoint=ckpt_a)
    assert stats["resumed_pasted"] > 0
    assert not os.path.exists(ckpt_a)          # cleaned after success

    out_b = str(tmp_path / "b.png")
    with TileDB(db_path) as db:
        make_mosaic(master_path, db, out_b, cells_x=8, levels=3,
                    detail_threshold=2.0, top_k=2, seed=5,
                    checkpoint=str(tmp_path / "b.ckpt"))
    assert open(out_a, "rb").read() == open(out_b, "rb").read()


def test_quadtree_roundtrip_from_leaves():
    t = QuadTree(256, 256, 4, 4)
    t.split_at(32, 32)
    t.split_at(200, 128)
    t.split_at(32, 32)   # deepen one branch
    leaves = t.leaf_boxes()
    t2 = QuadTree.from_leaf_boxes(256, 256, 4, 4, leaves)
    assert sorted(t2.leaf_boxes()) == sorted(leaves)
    # split/merge still work on the restored tree
    assert t2.split_at(128, 128) or t2.merge_at(32, 32)
