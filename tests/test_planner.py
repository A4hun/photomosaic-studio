import numpy as np
import pytest

from photomosaic.match import Matcher
from photomosaic.planner import plan_assignments


def _tiles(feats, avgs=None):
    feats = np.asarray(feats, dtype=np.float32)
    if avgs is None:
        avgs = np.zeros((len(feats), 3), dtype=np.float32)
    return {
        "ids": np.arange(len(feats), dtype=np.int64),
        "feats": feats,
        "avgs": np.asarray(avgs, dtype=np.float32),
        "metas": [{"id": i, "variant": "orig"} for i in range(len(feats))],
    }


def _grid_boxes(cx, cy, s=8):
    gy, gx = np.divmod(np.arange(cx * cy), cx)
    return np.stack([gx * s, gy * s, gx * s + s, gy * s + s], 1).astype(np.int64)


def test_candidates_match_brute_force_across_tile_blocks():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(300, 27)).astype(np.float32) * 10 + 50
    cells = rng.normal(size=(20, 27)).astype(np.float32) * 10 + 50
    m = Matcher(_tiles(feats), device="cpu", seed=0)
    # force several tile blocks by shrinking the block size through a tiny library split
    idx, d = m._cand_block(cells, np.zeros((20, 3), np.float32), 6, 64)
    ref = ((feats[None] - cells[:, None]) ** 2).sum(-1)
    want = np.sort(ref, axis=1)[:, :6]
    assert np.allclose(np.asarray(d), want, rtol=1e-4, atol=1e-2)
    assert np.array_equal(np.asarray(idx)[:, 0], ref.argmin(1))


def test_strict_top_k_1_is_the_classic_best_match():
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(120, 27)).astype(np.float32) * 10 + 50
    cells = rng.normal(size=(30, 27)).astype(np.float32) * 10 + 50
    m = Matcher(_tiles(feats), top_k=1, device="cpu", seed=0)
    picks, dists = plan_assignments(
        m, _grid_boxes(6, 5), cells, np.zeros((30, 3), np.float32),
        np.arange(120, dtype=np.int32), (48, 40), top_k=1, seed=0)
    ref = ((feats[None] - cells[:, None]) ** 2).sum(-1)
    assert np.array_equal(picks, ref.argmin(1))


def test_planner_spreads_photos_and_avoids_same_photo_neighbours():
    # 8 near-identical tiles per colour, each a different photo: a greedy matcher
    # keeps picking the same few, the planner must spread them out.
    rng = np.random.default_rng(2)
    base = rng.normal(size=(1, 27)).astype(np.float32) * 10 + 50
    feats = base + rng.normal(scale=0.3, size=(40, 27)).astype(np.float32)
    cells = np.repeat(base, 100, axis=0)
    m = Matcher(_tiles(feats), top_k=8, device="cpu", seed=3)
    groups = np.arange(40, dtype=np.int32)
    picks, _ = plan_assignments(m, _grid_boxes(10, 10), cells, np.zeros((100, 3), np.float32),
                                groups, (80, 80), top_k=8, seed=3)
    g = groups[picks].reshape(10, 10)
    same = ((g[:, 1:] == g[:, :-1]).mean() + (g[1:] == g[:-1]).mean()) / 2
    assert same < 0.02                       # (a random draw of 8 would give ~12%)
    assert np.bincount(groups[picks]).max() <= 25   # spread over the candidate pool


def test_planner_is_deterministic_for_a_seed():
    rng = np.random.default_rng(4)
    feats = rng.normal(size=(80, 27)).astype(np.float32) * 10 + 50
    cells = rng.normal(size=(36, 27)).astype(np.float32) * 10 + 50

    def run():
        m = Matcher(_tiles(feats), top_k=6, device="cpu", seed=9)
        return plan_assignments(m, _grid_boxes(6, 6), cells, np.zeros((36, 3), np.float32),
                                np.arange(80, dtype=np.int32), (48, 48), top_k=6, seed=9)[0]

    assert np.array_equal(run(), run())


def test_planner_respects_max_reuse_and_radius_fallback():
    target = np.full(27, 50.0, dtype=np.float32)
    other = np.full(27, 52.0, dtype=np.float32)
    m = Matcher(_tiles([target, other]), top_k=2, max_reuse=2, device="cpu", seed=0)
    cells = np.stack([target] * 4)
    picks, _ = plan_assignments(m, _grid_boxes(2, 2), cells, np.zeros((4, 3), np.float32),
                                np.array([0, 1], dtype=np.int32), (16, 16), top_k=2, seed=0,
                                max_reuse=2)
    assert np.bincount(picks, minlength=2).max() <= 2
    # radius excludes every candidate -> exact fallback still returns a tile, no crash
    far = np.array([[0, 0, 250], [0, 250, 0]], dtype=np.float32)
    m2 = Matcher(_tiles([target, other], far), top_k=2, radius=5.0, device="cpu", seed=0)
    p2, _ = plan_assignments(m2, _grid_boxes(1, 1), target[None], np.array([[250, 250, 250]], np.float32),
                             np.array([0, 1], dtype=np.int32), (8, 8), top_k=2, seed=0)
    assert p2.shape == (1,) and 0 <= p2[0] < 2
