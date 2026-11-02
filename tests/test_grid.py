import numpy as np

from photomosaic.grid import QuadTree


def _coverage(tree, w, h):
    cover = np.zeros((h, w), dtype=np.int8)
    for box, _lvl in tree.leaf_boxes():
        cover[box[1]:box[3], box[0]:box[2]] += 1
    return cover


def test_base_grid_and_split_merge():
    t = QuadTree(256, 256, 4, 4)
    assert len(t.leaves()) == 16
    assert all(lvl == 1 for _b, lvl in t.leaf_boxes())
    assert t.split_at(32, 32)
    assert len(t.leaves()) == 19
    assert t.leaf_boxes() and max(l for _b, l in t.leaf_boxes()) == 2
    assert np.all(_coverage(t, 256, 256) == 1)
    assert t.merge_at(32, 32)
    assert len(t.leaves()) == 16
    assert not t.merge_at(32, 32)  # root-level leaves have no parent


def test_split_depth_cap_and_min_size():
    t = QuadTree(64, 64, 1, 1, max_level=5, min_px=16)
    for _ in range(10):
        t.split_at(32, 32)
    levels = {lvl for _b, lvl in t.leaf_boxes()}
    assert max(levels) <= 3          # 64 -> 32 -> 16 stops at 2*min_px
    assert np.all(_coverage(t, 64, 64) == 1)


def test_merge_level_collapses_only_that_level():
    t = QuadTree(256, 256, 4, 4)
    t.split_at(32, 32)      # creates level-2 leaves in first base cell
    t.split_at(200, 200)    # level-2 in last base cell
    merged = t.merge_level(2)
    assert merged == 8
    assert all(lvl == 1 for _b, lvl in t.leaf_boxes())
    assert len(t.leaves()) == 16


def test_reset():
    t = QuadTree(128, 128, 2, 2)
    t.split_at(10, 10)
    t.reset()
    assert len(t.leaves()) == 4


def test_split_fully_and_collapse():
    t = QuadTree(256, 256, 4, 4)
    assert t.split_fully_at(32, 32)
    inside = [l for b, l in t.leaf_boxes() if b[2] <= 64 and b[3] <= 64]
    outside = [l for b, l in t.leaf_boxes() if not (b[2] <= 64 and b[3] <= 64)]
    assert inside and all(l == 3 for l in inside)   # 64px cell maxes at level 3
    assert outside and all(l == 1 for l in outside)
    assert np.all(_coverage(t, 256, 256) == 1)
    assert t.collapse_at(32, 32)                    # cancel whole region
    assert len(t.leaves()) == 16
    assert all(l == 1 for _b, l in t.leaf_boxes())


def test_split_region():
    t = QuadTree(256, 256, 4, 4)
    assert t.split_region((0, 0, 128, 128), 2)
    leaves = t.leaf_boxes()
    in_region = [l for b, l in leaves if b[2] <= 128 and b[3] <= 128]
    out_region = [l for b, l in leaves if b[0] >= 128 or b[1] >= 128]
    assert len(in_region) == 16 and all(l == 2 for l in in_region)
    assert len(out_region) == 12 and all(l == 1 for l in out_region)
    assert np.all(_coverage(t, 256, 256) == 1)

def test_refine_region_reaches_max_level_and_follows_box():
    t = QuadTree(256, 256, 4, 4, max_level=4, min_px=1)
    assert t.refine_region((0, 0, 40, 40))
    levels = [l for b, l in t.leaf_boxes() if b[2] <= 64 and b[3] <= 64]
    assert max(levels) == 4                      # finest allowed level reached
    far = [l for b, l in t.leaf_boxes() if b[0] >= 128 or b[1] >= 128]
    assert all(l == 1 for l in far)              # untouched outside the box
    assert np.all(_coverage(t, 256, 256) == 1)
    # tighter than a whole base cell: part of the 64px cell stays coarse
    assert min(l for b, l in t.leaf_boxes() if b[2] <= 64 and b[3] <= 64) < 4


def test_clear_region_and_set_max_level():
    t = QuadTree(256, 256, 4, 4, max_level=5, min_px=1)
    t.refine_region((0, 0, 256, 256), 3)
    assert t.clear_region((0, 0, 128, 128))
    assert all(l == 1 for b, l in t.leaf_boxes() if b[2] <= 128 and b[3] <= 128)
    assert any(l == 3 for _b, l in t.leaf_boxes())          # elsewhere untouched
    assert t.set_max_level(2)
    assert max(l for _b, l in t.leaf_boxes()) == 2
    t.refine_region((0, 0, 256, 256))
    assert max(l for _b, l in t.leaf_boxes()) == 2           # cap respected
    assert np.all(_coverage(t, 256, 256) == 1)


def test_merge_up_at_undoes_one_level():
    t = QuadTree(256, 256, 4, 4, max_level=5, min_px=1)
    t.split_at(32, 32)
    t.split_at(10, 10)
    assert t.merge_up_at(10, 10)
    assert max(l for b, l in t.leaf_boxes() if b[2] <= 64 and b[3] <= 64) == 2
