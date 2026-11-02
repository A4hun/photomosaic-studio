import numpy as np

from photomosaic.match import Matcher


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


def test_exact_match_wins_with_top_k_1():
    target = np.full(27, 50.0, dtype=np.float32)
    far1 = np.full(27, 10.0, dtype=np.float32)
    far2 = np.full(27, 90.0, dtype=np.float32)
    m = Matcher(_tiles([far1, target, far2]), top_k=1, seed=42)
    pos, dist = m.select(target, np.zeros(3, np.float32))
    assert pos == 1
    assert dist < 1e-3


def test_near_best_random_stays_within_top_k():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(50, 27)).astype(np.float32) * 10 + 50
    cell = feats[7] + rng.normal(scale=0.5, size=27).astype(np.float32)
    m = Matcher(_tiles(feats), top_k=5, seed=1)
    picks = {m.select(cell, np.zeros(3, np.float32))[0] for _ in range(40)}
    # all picks must belong to the true top-5 nearest tiles
    d = ((feats - cell) ** 2).sum(1)
    true_top5 = set(np.argsort(d)[:5].tolist())
    assert picks <= true_top5
    assert len(picks) >= 2  # and it is genuinely randomised


def test_max_reuse_cap():
    target = np.full(27, 50.0, dtype=np.float32)
    other = np.full(27, 55.0, dtype=np.float32)
    m = Matcher(_tiles([target, other]), top_k=1, max_reuse=1, seed=7)
    first, _ = m.select(target, np.zeros(3, np.float32))
    second, _ = m.select(target, np.zeros(3, np.float32))
    third, _ = m.select(target, np.zeros(3, np.float32))
    assert first == 0
    assert second == 1          # best is exhausted, falls to next
    assert third in (0, 1)      # all at cap -> filter relaxed, no crash


def test_batch_top1_equals_argmin():
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(200, 27)).astype(np.float32) * 10 + 50
    cells = rng.normal(size=(50, 27)).astype(np.float32) * 10 + 50
    m = Matcher(_tiles(feats), top_k=1, device="cpu", seed=0)
    picks, dists = m.select_batch(cells, np.zeros((50, 3), np.float32))
    d = ((feats[None, :, :] - cells[:, None, :]) ** 2).sum(-1)
    assert np.array_equal(picks, np.argmin(d, axis=1))
    # float32 accumulation vs float64 reference: compare with relative tolerance
    assert np.allclose(dists, d[np.arange(50), picks], rtol=1e-4, atol=1e-2)


def test_batch_cap_exact():
    target = np.full(27, 50.0, dtype=np.float32)
    other = np.full(27, 52.0, dtype=np.float32)
    m = Matcher(_tiles([target, other]), top_k=1, max_reuse=1, device="cpu", seed=0)
    cells = np.stack([target] * 4)
    picks, _ = m.select_batch(cells, np.zeros((4, 3), np.float32))
    assert list(picks[:2]) == [0, 1]          # cap enforced exactly
    assert set(picks.tolist()) <= {0, 1}


def test_device_cpu_explicit():
    m = Matcher(_tiles([np.zeros(27, np.float32)]), device="cpu")
    assert m.device == "cpu"


def test_radius_prefilter():
    target = np.full(27, 50.0, dtype=np.float32)
    same_colour_avg = np.array([[100.0, 100, 100]], dtype=np.float32)
    far_avg = np.array([[0.0, 0, 250]], dtype=np.float32)
    avgs = np.vstack([same_colour_avg, far_avg])
    m = Matcher(_tiles([target, target.copy()], avgs), top_k=2, radius=10.0, seed=3)
    pos, _ = m.select(target, np.array([100, 100, 100], np.float32))
    assert pos == 0  # tile 1 excluded by average-colour radius despite equal features
